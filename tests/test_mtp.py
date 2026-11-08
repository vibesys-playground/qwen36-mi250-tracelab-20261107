"""CPU checks of engine.mtp on a tiny random config: loading, wiring and its own KV cache.

    python -m pytest tests/test_mtp.py
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3_5MoeTextConfig

from engine.mtp import MTPHead, mtp_config, read_mtp_tensors

HIDDEN, VOCAB = 64, 96


def _text_config() -> Qwen3_5MoeTextConfig:
    config = Qwen3_5MoeTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        max_position_embeddings=4096,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def _random_state(config: Qwen3_5MoeTextConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    head = MTPHead(mtp_config(config))
    return {k: torch.randn_like(v) * 0.2 for k, v in head.state_dict().items()}


@pytest.fixture
def setup() -> tuple[MTPHead, nn.Embedding, nn.Linear, dict[str, torch.Tensor]]:
    config = _text_config()
    state = _random_state(config)
    head = MTPHead.from_state_dict(config, state, "cpu")
    torch.manual_seed(1)
    return head, nn.Embedding(VOCAB, HIDDEN), nn.Linear(HIDDEN, VOCAB, bias=False), state


def test_reads_only_mtp_tensors_from_shards(tmp_path) -> None:
    state = _random_state(_text_config())
    names = sorted(state)
    first = {"mtp." + n: state[n].contiguous() for n in names[:3]}
    second = {"mtp." + n: state[n].contiguous() for n in names[3:]}
    second["model.language_model.norm.weight"] = torch.zeros(HIDDEN)
    save_file(first, tmp_path / "a.safetensors")
    save_file(second, tmp_path / "b.safetensors")
    weight_map = {k: "a.safetensors" for k in first} | {k: "b.safetensors" for k in second}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    read = read_mtp_tensors(str(tmp_path), None)
    assert sorted(read) == names
    assert all(torch.equal(read[n], state[n]) for n in names)
    head = MTPHead.load(str(tmp_path), None, _text_config(), "cpu")
    assert head.layers[0].block_type == "full_attention"


def test_rejects_missing_tensor() -> None:
    config = _text_config()
    state = _random_state(config)
    del state["pre_fc_norm_hidden.weight"]
    with pytest.raises(ValueError, match="pre_fc_norm_hidden"):
        MTPHead.from_state_dict(config, state, "cpu")


def test_wiring_matches_hand_written_reference(setup) -> None:
    """Embedding first in the concat, zero-centred norms, final norm, shared lm_head."""
    head, embed, lm_head, state = setup
    hidden = torch.randn(1, 1, HIDDEN)
    ids = torch.tensor([[5]])

    def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)

    with torch.no_grad():
        got = head(hidden, ids, torch.tensor([[0]]), head.new_cache(), embed, lm_head)
        z = torch.cat(
            [
                rms(embed(ids), state["pre_fc_norm_embedding.weight"]),
                rms(hidden, state["pre_fc_norm_hidden.weight"]),
            ],
            -1,
        ) @ state["fc.weight"].T
        cos_sin = head.rotary_emb(z, torch.zeros(3, 1, 1, dtype=torch.long))
        z = head.layers[0](z, position_embeddings=cos_sin, position_ids=torch.tensor([[0]]))
        want = lm_head(rms(z, state["norm.weight"])).float()
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_incremental_steps_match_one_pass(setup) -> None:
    head, embed, lm_head, _ = setup
    length = 9
    hidden = torch.randn(1, length, HIDDEN)
    ids = torch.randint(0, VOCAB, (1, length))
    positions = torch.arange(length)[None]
    with torch.no_grad():
        whole = head(hidden, ids, positions, head.new_cache(), embed, lm_head, logits_to_keep=length)
        cache = head.new_cache()
        parts = [head(hidden[:, :4], ids[:, :4], positions[:, :4], cache, embed, lm_head, 4)]
        parts.append(head(hidden[:, 4:7], ids[:, 4:7], positions[:, 4:7], cache, embed, lm_head, 3))
        for t in range(7, length):
            parts.append(head(hidden[:, t : t + 1], ids[:, t : t + 1], positions[:, t : t + 1], cache, embed, lm_head))
    torch.testing.assert_close(torch.cat(parts, 1), whole, rtol=1e-4, atol=1e-4)
    assert cache.get_seq_length() == length
