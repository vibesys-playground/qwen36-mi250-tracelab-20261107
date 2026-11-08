"""Plan 35: the flash prefill attention kernel (``engine.fast.flash_prefill``) against the
masked-SDPA reference, over the engine's shapes (TP2: 8 query heads, 1 KV head, head dim 256,
64-token pages) and the tile/split variants ``choose`` picks among. GPU, or Triton's CPU
interpreter (``TRITON_INTERPRET=1``) for the small shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fast import flash_prefill as fp  # noqa: E402
from engine.fast import kernels  # noqa: E402
from engine.fast import prefill_kernels as pk  # noqa: E402


def _device():
    if torch.cuda.is_available() and kernels.triton is not None:
        return torch.device("cuda")
    if kernels.INTERPRET and kernels.triton is not None:
        return torch.device("cpu")
    return None


pytestmark = pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="engine-sized shapes: GPU only")


def _inputs(q_heads, kv_heads, dim, page, count, start, dev, garbage=True):
    g = torch.Generator().manual_seed(count * 7 + start)
    total = start + count
    used = -(-total // page)
    pages = used + 5
    k_pool = torch.randn(pages, kv_heads, page, dim, generator=g).bfloat16()
    v_pool = torch.randn(pages, kv_heads, page, dim, generator=g).bfloat16()
    perm = torch.randperm(pages, generator=g)[:used]
    if garbage:  # positions past the sequence hold NaN that must never reach the output
        for j, p in enumerate(perm.tolist()):
            bad = max(0, min(page, total - j * page))
            k_pool[p, :, bad:] = float("nan")
            v_pool[p, :, bad:] = float("nan")
    q = (2.5 * torch.randn(count, q_heads, dim, generator=g)).bfloat16()
    gate = torch.randn(count, q_heads * dim, generator=g).bfloat16()
    table = perm.long()
    return tuple(t.to(dev) for t in (q, k_pool, v_pool, table, gate))


def _check(got, ref, rows_ref):
    """Within 2x the rows kernel's error against the same reference (BF16 output rounding)."""
    assert got.dtype == torch.bfloat16 and got.shape == ref.shape
    assert not got.isnan().any()
    err = float((got.float() - ref.float()).abs().max())
    base = float((rows_ref.float() - ref.float()).abs().max())
    assert err <= max(2 * base, 2e-2 * max(1.0, float(ref.abs().max()))), (err, base)


VARIANTS = [
    {},  # choose()
    {"block_r": 8, "block_n": 64, "split": 1},
    {"block_r": 8, "block_n": 64, "split": 3},
    {"block_r": 8, "block_n": 32, "split": 2},
    {"block_r": 16, "block_n": 128, "split": 1},
    {"block_r": 16, "block_n": 128, "split": 2},
]


@pytest.mark.parametrize("q_heads,kv_heads,dim,page", [(4, 2, 32, 64), (8, 1, 64, 16)])
@pytest.mark.parametrize("count,start", [(1, 0), (5, 0), (37, 0), (40, 100), (300, 70)])
@pytest.mark.parametrize("variant", range(len(VARIANTS)))
def test_flash_matches_masked_sdpa(q_heads, kv_heads, dim, page, count, start, variant):
    dev = _device()
    q, k_pool, v_pool, table, gate = _inputs(q_heads, kv_heads, dim, page, count, start, dev)
    scale = dim ** -0.5
    ref = pk.prefill_attention_reference(q, k_pool.nan_to_num(), v_pool.nan_to_num(), table, start, gate, scale)
    rows = pk.prefill_attention_rows(q, k_pool, v_pool, table, start, gate, scale)
    got = fp.flash_prefill_attention(q, k_pool, v_pool, table, start, gate, scale, **VARIANTS[variant])
    _check(got, ref, rows)


@gpu
@pytest.mark.parametrize("count,start", [(1, 3599), (61, 3540), (561, 3040), (1000, 2600), (1650, 1950), (700, 0)])
def test_flash_engine_shapes(count, start):
    dev = torch.device("cuda")
    q, k_pool, v_pool, table, gate = _inputs(8, 1, 256, 64, count, start, dev)
    scale = 256 ** -0.5
    ref = pk.prefill_attention_reference(q, k_pool.nan_to_num(), v_pool.nan_to_num(), table, start, gate, scale)
    rows = pk.prefill_attention_rows(q, k_pool, v_pool, table, start, gate, scale)
    got = fp.flash_prefill_attention(q, k_pool, v_pool, table, start, gate, scale)
    _check(got, ref, rows)
    # bitwise run to run, and through the engine entry point when the flash path is on
    assert torch.equal(got, fp.flash_prefill_attention(q, k_pool, v_pool, table, start, gate, scale))
    if pk._FLASH:
        assert torch.equal(got, pk.prefill_attention(q, k_pool, v_pool, table, start, gate, scale))
    # into a row slice of a larger output, as the packed prefill does
    out = torch.zeros((count + 3, 8 * 256), dtype=torch.bfloat16, device=dev)
    fp.flash_prefill_attention(q, k_pool, v_pool, table, start, gate, scale, out=out[2 : 2 + count])
    assert torch.equal(out[2 : 2 + count], got)
    assert not out[:2].any() and not out[2 + count :].any()


@gpu
def test_flash_graph_capture():
    dev = torch.device("cuda")
    q, k_pool, v_pool, table, gate = _inputs(8, 1, 256, 64, 61, 3540, dev, garbage=False)
    scale = 256 ** -0.5
    eager = fp.flash_prefill_attention(q, k_pool, v_pool, table, 3540, gate, scale)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fp.flash_prefill_attention(q, k_pool, v_pool, table, 3540, gate, scale)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp.flash_prefill_attention(q, k_pool, v_pool, table, 3540, gate, scale)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


def test_choose_shapes_only():
    for count in (1, 61, 561, 999, 1000, 1650, 4000):
        cfg = fp.choose(count, 3600 - min(count, 3600), 1, 8, 104)
        assert cfg["split"] >= 1 and cfg["block_r"] * 8 >= 16
    assert fp.choose(61, 0, 1, 8, 104) == fp.choose(61, 3000, 1, 8, 104)  # T alone
    assert fp.choose(561, 3040, 1, 8, 104) == fp.choose(561, 3040, 1, 8, 104)


@pytest.mark.parametrize("variant", range(len(VARIANTS)))
@pytest.mark.parametrize("count,start,size", [(1, 0, 16), (5, 3, 16), (37, 0, 64), (40, 100, 64), (300, 70, 320)])
def test_flash_meta_matches_rows(variant, count, start, size):
    """Plan 33's device-side [count, start]: a bucket of ``size`` rows, the first ``count``
    real; the real rows equal a plain call with the same knobs, bit for bit, and padding rows
    are finite."""
    dev = _device()
    q_heads, kv_heads, dim, page = 8, 1, 64, 16
    q, k_pool, v_pool, table, gate = _inputs(q_heads, kv_heads, dim, page, count, start, dev, garbage=False)
    scale = dim ** -0.5
    knobs = dict(VARIANTS[variant]) or fp.choose(size, None, kv_heads, q_heads // kv_heads)
    knobs.pop("waves_per_eu", None)
    plain = fp.flash_prefill_attention(q, k_pool, v_pool, table, start, gate, scale, **knobs)
    qb = torch.zeros((size, q_heads, dim), dtype=q.dtype, device=dev)
    gb = torch.zeros((size, q_heads * dim), dtype=gate.dtype, device=dev)
    qb[:count], gb[:count] = q, gate
    qb[count:] = q[-1]
    meta = torch.tensor([count, start], dtype=torch.long, device=dev)
    got = fp.flash_prefill_attention(qb, k_pool, v_pool, table, 999, gb, scale, meta=meta, **knobs)
    assert torch.equal(got[:count], plain)
    assert torch.isfinite(got.float()).all()
