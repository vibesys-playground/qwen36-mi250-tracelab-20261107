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


def _inputs(tokens, heads, hv, seed, stride_pad=0):
    g = torch.Generator().manual_seed(seed)
    width = 2 * heads * 128 + hv * 128
    big = torch.randn(tokens, width + stride_pad + 2 * hv, generator=g)
    mixed = big[:, :width].bfloat16()
    ab = torch.randn(tokens, 2 * hv, generator=g).bfloat16()
    a_log = torch.randn(hv, generator=g) * 0.5
    dt_bias = torch.randn(hv, generator=g)
    state = torch.randn(hv, 128, 128, generator=g) * 0.1
    return mixed, ab[:, :hv], ab[:, hv:], a_log, dt_bias, state


def _err(got, ref):
    got, ref = got.float(), ref.float()
    return float((got - ref).abs().max()) / max(1e-6, float(ref.abs().max()))


@pytest.mark.parametrize("tokens,heads,hv", [(2, 1, 2), (37, 2, 4), (64, 1, 2), (130, 2, 4)])
def test_chunk_matches_recurrence(tokens, heads, hv):
    dev = _device()
    mixed, a, b, a_log, dt_bias, state = _inputs(tokens, heads, hv, tokens)
    ref_o, ref_s = gdn_chunk.reference(mixed, a, b, a_log, dt_bias, state, heads)
    st = state.clone().to(dev)
    got = gdn_chunk.chunk(mixed.to(dev), a.to(dev), b.to(dev), a_log.to(dev), dt_bias.to(dev), st, heads)
    assert got.shape == (tokens, hv, 128)
    assert _err(got.cpu(), ref_o) < 3e-2, _err(got.cpu(), ref_o)
    assert _err(st.cpu(), ref_s) < 3e-2, _err(st.cpu(), ref_s)


@pytest.mark.parametrize("count,size", [(5, 64), (61, 128), (100, 192)])
def test_chunk_padding_rows_and_device_row(count, size):
    """Rows past the device count leave the state as ``count`` real rows do; outputs 0; the
    state row comes from the device and other rows are untouched."""
    dev = _device()
    heads, hv = 1, 2
    mixed, a, b, a_log, dt_bias, state = _inputs(size, heads, hv, count + size)
    ref_o, ref_s = gdn_chunk.reference(mixed, a, b, a_log, dt_bias, state, heads, count=count)
    states = torch.randn(3, hv, 128, 128)
    states[2] = state
    before = states.clone()
    states = states.to(dev)
    meta = torch.tensor([count, 999], dtype=torch.long, device=dev)
    row = torch.tensor([2], dtype=torch.long, device=dev)
    got = gdn_chunk.chunk(mixed.to(dev), a.to(dev), b.to(dev), a_log.to(dev), dt_bias.to(dev), states, heads,
                          row=row, meta=meta)
    got = got.cpu()
    assert _err(got[:count], ref_o[:count]) < 3e-2
    assert torch.equal(got[count:].float(), torch.zeros_like(got[count:].float()))
    s = states.cpu()
    assert _err(s[2], ref_s) < 3e-2
    assert torch.equal(s[0], before[0]) and torch.equal(s[1], before[1])


def test_chunk_strided_inputs():
    """``mixed`` and a/b as column slices of a wider projection output."""
    dev = _device()
    tokens, heads, hv = 70, 1, 2
    g = torch.Generator().manual_seed(3)
    width = 2 * heads * 128 + hv * 128
    big = torch.randn(tokens, width + 64 + 2 * hv, generator=g).bfloat16()
    mixed = big[:, :width]
    a = big[:, width + 64 : width + 64 + hv]
    b = big[:, width + 64 + hv :]
    a_log = torch.randn(hv, generator=g) * 0.5
    dt_bias = torch.randn(hv, generator=g)
    state = torch.randn(hv, 128, 128, generator=g) * 0.1
    ref_o, ref_s = gdn_chunk.reference(mixed, a, b, a_log, dt_bias, state, heads)
    st = state.clone().to(dev)
    bigd = big.to(dev)
    got = gdn_chunk.chunk(bigd[:, :width], bigd[:, width + 64 : width + 64 + hv], bigd[:, width + 64 + hv :],
                          a_log.to(dev), dt_bias.to(dev), st, heads)
    assert _err(got.cpu(), ref_o) < 3e-2
    assert _err(st.cpu(), ref_s) < 3e-2
