"""The checkpoint's native multi-token-prediction (MTP) head, as a draft model.

Qwen3.6-35B-A3B ships one MTP block (``mtp.*`` tensors, ``mtp_num_hidden_layers:
1``) that Transformers drops on load. It follows the Qwen3-Next / Qwen3.5 MTP
design (SGLang ``qwen3_5_mtp.py``, vLLM ``qwen3_next_mtp.py``):

    e  = pre_fc_norm_embedding(embed_tokens(x[t+1]))      # shared embedding
    h  = pre_fc_norm_hidden(H[t])                          # base model's final, normed hidden
    z  = fc(cat([e, h], -1))                               # 2*hidden -> hidden, no bias
    z  = decoder_layer(z, position t)                      # gated full attention + routed/shared MoE
    y  = norm(z)
    logits(x[t+2]) = lm_head(y)                            # shared lm_head (untied in this checkpoint)

``H[t]`` is the base model's last hidden state after its final norm at position
``t`` (the input of ``lm_head`` that produced ``x[t+1]``). The embedding comes
first in the concatenation. Every RMSNorm is the zero-centred kind, ``x * (1 +
w)``. The block is one ordinary decoder layer of the base architecture with its
layer type forced to full attention, so it keeps its own attention KV cache
over positions ``0..t``: during prefill it sees ``(x[i+1], H[i])`` at position
``i`` for the whole prompt, then one pair per decode step.

``MTPHead.load`` reads only the ``mtp.*`` tensors from the pinned safetensors
shards and places the head on one device. Nothing here changes the base
model's computation; the head only proposes tokens.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)

PREFIX = "mtp."
INDEX = "model.safetensors.index.json"


def mtp_config(text_config: Any) -> Any:
    """The base text config with every layer a full-attention layer, as many as MTP layers."""
    config = copy.deepcopy(text_config)
    layers = int(getattr(text_config, "mtp_num_hidden_layers", 1) or 1)
    config.num_hidden_layers = layers
    config.layer_types = ["full_attention"] * layers
    config._attn_implementation = text_config._attn_implementation
    if hasattr(text_config, "_experts_implementation"):
        config._experts_implementation = text_config._experts_implementation
    return config


class MTPHead(nn.Module):
    """One MTP block. ``embed_tokens`` and ``lm_head`` are the base model's, passed per call."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)

    @classmethod
    def from_state_dict(
        cls, text_config: Any, state: dict[str, torch.Tensor], device: torch.device | str
    ) -> MTPHead:
        """Build the head from ``mtp.``-stripped tensors, every one required, on ``device``."""
        config = mtp_config(text_config)
        with torch.device("meta"):
            head = cls(config)
        expected = set(head.state_dict())
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            message = f"MTP tensors do not match: missing {sorted(missing)}, unexpected {sorted(unexpected)}"
            raise ValueError(message)
        for name, tensor in head.state_dict().items():
            if tuple(state[name].shape) != tuple(tensor.shape):
                message = f"MTP tensor {name}: shape {tuple(state[name].shape)}, expected {tuple(tensor.shape)}"
                raise ValueError(message)
        head.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True, assign=True)
        head.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config).to(device)
        return head.eval()

    @classmethod
    def load(
        cls, model: str, revision: str | None, text_config: Any, device: torch.device | str
    ) -> MTPHead:
        """Read the ``mtp.*`` tensors of ``model`` (a hub id in the HF cache, or a directory)."""
        return cls.from_state_dict(text_config, read_mtp_tensors(model, revision), device)

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.config)

    @property
    def device(self) -> torch.device:
        return self.fc.weight.device

    def forward(
        self,
        hidden: torch.Tensor,
        next_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: DynamicCache,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        logits_to_keep: int = 1,
    ) -> torch.Tensor:
        """Logits for token ``t+2`` from base hidden ``H[t]`` ``[B, L, D]``, ids ``x[t+1]`` ``[B, L]``.

        ``positions`` ``[B, L]`` are the positions ``t`` of the hidden states.
        ``cache`` is this head's own cache and grows by ``L``. Returns float
        logits of the last ``logits_to_keep`` positions, ``[B, logits_to_keep, V]``.
        """
        device = self.device
        embeds = embed_tokens(next_ids.to(embed_tokens.weight.device)).to(device)
        hidden = hidden.to(device=device, dtype=embeds.dtype)
        positions = positions.to(device)
        x = self.fc(
            torch.cat([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], dim=-1)
        )
        mask = create_causal_mask(
            config=self.config,
            inputs_embeds=x,
            attention_mask=None,
            past_key_values=cache,
            position_ids=positions,
        )
        position_embeddings = self.rotary_emb(x, positions[None].expand(3, -1, -1))
        for layer in self.layers:
            x = layer(
                x,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
            )
        y = self.norm(x[:, -logits_to_keep:])
        return lm_head(y.to(lm_head.weight.device)).float()


def _snapshot_files(model: str, revision: str | None) -> tuple[dict[str, str], Any]:
    """The weight map of ``model`` and a function from a shard name to its local path."""
    local = Path(model)
    if local.is_dir():
        index = json.loads((local / INDEX).read_text())
        return index["weight_map"], lambda name: str(local / name)
    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(model, INDEX, revision=revision)
    index = json.loads(Path(index_path).read_text())
    return index["weight_map"], lambda name: hf_hub_download(model, name, revision=revision)


def read_mtp_tensors(model: str, revision: str | None) -> dict[str, torch.Tensor]:
    """Every ``mtp.*`` tensor of the checkpoint, keyed without the prefix, on the CPU."""
    from safetensors import safe_open

    weight_map, path_of = _snapshot_files(model, revision)
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith(PREFIX):
            shards.setdefault(shard, []).append(name)
    if not shards:
        raise ValueError(f"{model}@{revision} has no {PREFIX}* tensors")
    state = {}
    for shard, names in shards.items():
        with safe_open(path_of(shard), framework="pt") as reader:
            for name in names:
                state[name.removeprefix(PREFIX)] = reader.get_tensor(name)
    return state
