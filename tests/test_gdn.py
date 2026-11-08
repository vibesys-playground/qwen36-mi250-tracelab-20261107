"""CPU checks of engine.gdn against the Transformers code it replaces.

Run with the CPU check environment:
    UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/vibesys/qwen36-cpu-check-venv" \
        uv run --project .vibesys/shared/cpu_check python -m pytest tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import gdn  # noqa: E402

REF_CONV = gdn._unwrapped(modeling.causal_conv1d_fn)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 37, 255, 256, 257, 600])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_bucketed_causal_conv_equals_transformers(dtype, length, with_bias, activation):
    torch.manual_seed(length)
    channels, kernel = 48, 4
    x = torch.randn(2, channels, length, dtype=dtype)
    weight = torch.randn(channels, kernel, dtype=dtype)
    bias = torch.randn(channels, dtype=dtype) if with_bias else None
    want = REF_CONV(x, weight, bias, activation=activation)
    got = gdn.causal_conv1d_fn(x, weight, bias, activation=activation)
    assert got.dtype == want.dtype and got.shape == want.shape
    # Right padding never reaches a kept causal output: the same values.
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_causal_conv_reads_a_transposed_view():
    """The GDN forward passes ``in_proj(x).transpose(1, 2)``, a non-contiguous view."""
    torch.manual_seed(0)
    for length in (37, 256):
        x = torch.randn(1, length, 48, dtype=torch.bfloat16).transpose(1, 2)
        weight = torch.randn(48, 4, dtype=torch.bfloat16)
        want = REF_CONV(x, weight, None, activation="silu")
        got = gdn.causal_conv1d_fn(x, weight, None, activation="silu")
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_buckets():
    assert [gdn.bucket(n) for n in (1, 255, 256, 257, 2052)] == [256, 256, 256, 512, 2304]


def test_warm_conv_runs_each_bucket_and_batch_once_per_shape():
    model = _tiny_model(torch.bfloat16)
    try:
        gdn.install(model, "torch")
        # Two GDN layers of one shape on one device: one set of problems.
        assert gdn.warm_conv(model, 2048 + 4) == 9 + gdn.WARM_BATCH
    finally:
        gdn.install(model, "reference")


def _tiny_model(dtype: torch.dtype):
    config = Qwen3_5MoeTextConfig(
        vocab_size=320,
        hidden_size=64,
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
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    torch.manual_seed(0)
    model = modeling.Qwen3_5MoeForCausalLM(config).to(dtype).eval()
    with torch.no_grad():
        for m in gdn.gdn_modules(model):
            m.norm.weight.uniform_(0.5, 1.5)  # not all ones, so the norm weight is exercised
    return model


def _run(model, prompt: list[int], steps: int):
    """Prefill in two chunks, then greedy decode; logits of every step and the final cache."""
    cache = DynamicCache(config=model.config)
    logits = []
    with torch.inference_mode():
        for part in (prompt[:21], prompt[21:]):
            out = model(input_ids=torch.tensor([part]), past_key_values=cache, use_cache=True)
            logits.append(out.logits[0, -1].float())
        for _ in range(steps):
            token = int(logits[-1].argmax())
            out = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            logits.append(out.logits[0, -1].float())
    return torch.stack(logits), cache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_torch_mode_model_matches_reference(dtype):
    model = _tiny_model(dtype)
    prompt = [(37 * i + 5) % 300 + 1 for i in range(50)]
    try:
        assert gdn.install(model, "reference")["mode"] == "reference"
        want, want_cache = _run(model, prompt, 8)
        info = gdn.install(model, "torch")
        assert info == {"mode": "torch", "layers": 2, "fla": None, "fla_error": info["fla_error"]}
        assert modeling.causal_conv1d_fn is gdn.causal_conv1d_fn
        # The decode conv update stays Transformers' torch function.
        assert modeling.causal_conv1d_update is gdn._unwrapped(gdn._ORIGINAL[modeling.__name__]["causal_conv1d_update"])
        got, got_cache = _run(model, prompt, 8)
    finally:
        gdn.install(model, "reference")
    assert (got.argmax(-1) == want.argmax(-1)).all()
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)
    for index in (0, 2):  # the GDN layers
        w_layer, g_layer = want_cache.layers[index], got_cache.layers[index]
        assert g_layer.recurrent_states[0].dtype == torch.float32
        torch.testing.assert_close(g_layer.conv_states[0], w_layer.conv_states[0], rtol=tol, atol=tol)
        torch.testing.assert_close(
            g_layer.recurrent_states[0], w_layer.recurrent_states[0], rtol=tol, atol=tol
        )


def test_reference_mode_restores_transformers_torch_functions():
    model = _tiny_model(torch.float32)
    gdn.install(model, "torch")
    gdn.install(model, "reference")
    saved = gdn._ORIGINAL[modeling.__name__]
    for name in gdn._GLOBALS:
        assert getattr(modeling, name) is gdn._unwrapped(saved[name])
    assert modeling.Qwen3_5MoeRMSNormGated.forward is gdn._ORIGINAL_NORM[modeling.Qwen3_5MoeRMSNormGated]


def test_auto_mode_on_cpu_uses_torch_and_fla_mode_refuses():
    model = _tiny_model(torch.float32)
    try:
        assert gdn.install(model)["mode"] == "torch"
        with pytest.raises(RuntimeError):
            gdn.install(model, "fla")
        with pytest.raises(ValueError):
            gdn.install(model, "fast")
    finally:
        gdn.install(model, "reference")


class _Tokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 1


def test_warmup_leaves_counters_and_prefix_cache_alone():
    from engine.generator import Generator

    model = _tiny_model(torch.bfloat16)
    generator = Generator(
        model,
        _Tokenizer(),
        frozenset([2]),
        max_model_len=4096,
        chunk_tokens=8,
        prefix_cache_tokens=4096,
        prefix_cache_entries=16,
    )
    try:
        gdn.install(model, "torch")
        assert gdn.warmup(generator) >= 0
    finally:
        gdn.install(model, "reference")
    assert (generator.decode_steps, generator.decoded_tokens, generator.largest_batch) == (0, 0, 0)
    assert generator._prefix_cache.lookup(tuple(range(1, 200))) == (0, None)
