"""The fused GDN prefill chain (``engine.fast.gdn_chunk``) against the token-by-token rule.

Runs on a GPU or under Triton's CPU interpreter (``TRITON_INTERPRET=1``); skipped otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fast import gdn_chunk, kernels  # noqa: E402


def _device():
    if torch.cuda.is_available() and kernels.triton is not None:
        return torch.device("cuda")
    if kernels.INTERPRET and kernels.triton is not None:
        return torch.device("cpu")
    return None


if kernels.INTERPRET and kernels.triton is not None:
    # The interpreter holds runtime scalars as 1-element arrays, which newer numpy will not
    # turn into a Python int for ``range``; the kernels loop over a runtime chunk count.
    from triton.runtime import interpreter as _interp

    _orig_patch = _interp._patch_lang_tensor

    def _patch_lang_tensor(tensor):
        _orig_patch(tensor)
        tensor.__index__ = lambda self: int(self.handle.data.reshape(-1)[0])

    _interp._patch_lang_tensor = _patch_lang_tensor

pytestmark = pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")


def _inputs(tokens, heads, hv, seed):
    g = torch.Generator().manual_seed(seed)
    width = 2 * heads * 128 + hv * 128
    qkv = torch.randn(tokens, width, generator=g).bfloat16()
    ab = torch.randn(tokens, 2 * hv, generator=g).bfloat16()
    a_log = torch.randn(hv, generator=g) * 0.5
    dt_bias = torch.randn(hv, generator=g)
    state = torch.randn(hv, 128, 128, generator=g) * 0.1
    conv = torch.randn(width, 4, generator=g).bfloat16()
    weight = (torch.randn(width, 1, 4, generator=g) * 0.5).bfloat16()
    return qkv, conv, weight, ab[:, :hv], ab[:, hv:], a_log, dt_bias, state


def _err(got, ref):
    got, ref = got.float(), ref.float()
    return float((got - ref).abs().max()) / max(1e-6, float(ref.abs().max()))


@pytest.mark.parametrize("tokens,heads,hv", [(2, 1, 2), (37, 2, 4), (64, 1, 2), (130, 2, 4)])
def test_chunk_matches_recurrence(tokens, heads, hv):
    dev = _device()
    qkv, conv, weight, a, b, a_log, dt_bias, state = _inputs(tokens, heads, hv, tokens)
    ref_o, ref_s, ref_conv = gdn_chunk.reference(qkv, conv.clone(), weight, a, b, a_log, dt_bias, state, heads)
    st = state.clone().to(dev)
    cv = conv.clone().to(dev)
    got = gdn_chunk.chunk(qkv.to(dev), cv, weight.to(dev), a.to(dev), b.to(dev), a_log.to(dev), dt_bias.to(dev),
                          st, heads)
    assert got.shape == (tokens, hv, 128)
    assert _err(got.cpu(), ref_o) < 3e-2, _err(got.cpu(), ref_o)
    assert _err(st.cpu(), ref_s) < 3e-2, _err(st.cpu(), ref_s)
    assert torch.equal(cv.cpu(), ref_conv)


@pytest.mark.parametrize("count,size", [(5, 64), (61, 128), (100, 192)])
def test_chunk_padding_rows_and_device_row(count, size):
    """Rows past the device count leave the states as ``count`` real rows do; outputs 0; the
    state row comes from the device and other rows are untouched."""
    dev = _device()
    heads, hv = 1, 2
    qkv, conv, weight, a, b, a_log, dt_bias, state = _inputs(size, heads, hv, count + size)
    ref_o, ref_s, ref_conv = gdn_chunk.reference(qkv, conv.clone(), weight, a, b, a_log, dt_bias, state, heads,
                                                 count=count)
    states = torch.randn(3, hv, 128, 128)
    states[2] = state
    before = states.clone()
    states = states.to(dev)
    cv = conv.clone().to(dev)
    meta = torch.tensor([count, 999], dtype=torch.long, device=dev)
    row = torch.tensor([2], dtype=torch.long, device=dev)
    got = gdn_chunk.chunk(qkv.to(dev), cv, weight.to(dev), a.to(dev), b.to(dev), a_log.to(dev), dt_bias.to(dev),
                          states, heads, row=row, meta=meta)
    got = got.cpu()
    assert _err(got[:count], ref_o[:count]) < 3e-2
    assert torch.equal(got[count:].float(), torch.zeros_like(got[count:].float()))
    s = states.cpu()
    assert _err(s[2], ref_s) < 3e-2
    assert torch.equal(s[0], before[0]) and torch.equal(s[1], before[1])
    assert torch.equal(cv.cpu(), ref_conv)


def test_chunk_strided_inputs():
    """``qkv`` and a/b as column slices of a wider (merged) projection output."""
    dev = _device()
    tokens, heads, hv = 70, 1, 2
    g = torch.Generator().manual_seed(3)
    width = 2 * heads * 128 + hv * 128
    big = torch.randn(tokens, width + 64 + 2 * hv, generator=g).bfloat16()
    qkv = big[:, :width]
    a = big[:, width + 64 : width + 64 + hv]
    b = big[:, width + 64 + hv :]
    a_log = torch.randn(hv, generator=g) * 0.5
    dt_bias = torch.randn(hv, generator=g)
    state = torch.randn(hv, 128, 128, generator=g) * 0.1
    conv = torch.randn(width, 4, generator=g).bfloat16()
    weight = (torch.randn(width, 1, 4, generator=g) * 0.5).bfloat16()
    ref_o, ref_s, _ = gdn_chunk.reference(qkv, conv.clone(), weight, a, b, a_log, dt_bias, state, heads)
    st = state.clone().to(dev)
    bigd = big.to(dev)
    got = gdn_chunk.chunk(bigd[:, :width], conv.clone().to(dev), weight.to(dev), bigd[:, width + 64 : width + 64 + hv],
                          bigd[:, width + 64 + hv :], a_log.to(dev), dt_bias.to(dev), st, heads)
    assert _err(got.cpu(), ref_o) < 3e-2
    assert _err(st.cpu(), ref_s) < 3e-2



def _wy_outputs(variant, tokens, heads, hv, count, seed):
    """The WY stage's outputs (qn, kn, w, u, gc, aqk) in the form ``variant``."""
    from engine.fast.prefill_kernels import gdn_conv_

    dev = _device()
    qkv, conv, weight, a, b, a_log, dt_bias, _ = _inputs(tokens, heads, hv, seed)
    meta = torch.tensor([count, 0], dtype=torch.long, device=dev) if count is not None else None
    mixed = gdn_conv_(qkv.to(dev), conv.clone().to(dev), weight.to(dev), meta=meta)
    outs = (
        torch.full((tokens, heads, 128), 7.0, dtype=torch.bfloat16, device=dev),
        torch.full((tokens, heads, 128), 7.0, dtype=torch.bfloat16, device=dev),
        torch.full((tokens, hv, 128), 7.0, dtype=torch.bfloat16, device=dev),
        torch.full((tokens, hv, 128), 7.0, dtype=torch.bfloat16, device=dev),
        torch.full((tokens, hv), 7.0, dtype=torch.float32, device=dev),
        torch.full((tokens, hv, gdn_chunk.BT), 7.0, dtype=torch.bfloat16, device=dev),
    )
    with kernels.on_device(mixed):
        gdn_chunk.wy(mixed, a.to(dev), b.to(dev), a_log.to(dev), dt_bias.to(dev), meta, heads, *outs, warps=4,
                     variant=variant)
    return [t.cpu() for t in outs]


@pytest.mark.parametrize("variant", [("stack", 2, 4, 64), ("split", 2, 4, 64), ("split1", 1, 4, 128),
                                     ("split", 4, 8, 128)])
@pytest.mark.parametrize("tokens,count", [(61, None), (130, None), (128, 100)])
def test_wy_variants_match_fused(variant, tokens, count):
    """Plan 54: every form of the WY stage writes what the fused kernel writes (the same
    operations per element). Under the interpreter bitwise, except w and u of the split form:
    numpy's matmul rounds differently by operand memory layout, which differs there. On a
    GPU a tolerance (the reductions' order may differ with layout and warps);
    tools/gdn_wy_bench.py reports which outputs are bitwise there."""
    heads, hv = 2, 4
    ref = _wy_outputs(("mono", 0, 0, 0), tokens, heads, hv, count, tokens)
    got = _wy_outputs(variant, tokens, heads, hv, count, tokens)
    for name, x, y in zip(("qn", "kn", "w", "u", "gc", "aqk"), got, ref):
        if kernels.INTERPRET and (name not in ("w", "u") or variant[0] == "stack"):
            assert torch.equal(x, y), (name, _err(x, y))
        else:
            assert _err(x, y) < 1e-2, (name, _err(x, y))


@pytest.mark.parametrize("variant", ["mono", "stack", "split", "split1"])
def test_chunk_variants_match_recurrence(variant, monkeypatch):
    monkeypatch.setenv("ENGINE_GDN_WY", variant)
    test_chunk_padding_rows_and_device_row(61, 128)
    test_chunk_matches_recurrence(130, 2, 4)
