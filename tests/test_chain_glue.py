"""Plan 51: the chained verify head's glue in Triton and the K-sum in the all-reduce.

- ``kernels.mtp_in`` matches the torch ops it replaces (embedding, two zero-centered RMSNorms,
  cat) to BF16 rounding;
- ``kernels.chain_accept`` gives exactly the torch accept math's count, accepted row, link-1
  draft, positions and output layout;
- ``mfma.gemv(defer=True)``'s parts summed by ``KParts.materialize`` equal ``mfma.gemv`` bit
  for bit, and the all-reduce's K-sum prologue (``tp._ll_allreduce_kernel`` KSPLIT > 1, in its
  loopback mode) gives the same bits as ``_ksum_kernel`` then the all-reduce (GPU only).

Needs Triton's CPU interpreter (``TRITON_INTERPRET=1``) on the CPU, or a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fast import kernels, mfma  # noqa: E402

pytestmark = pytest.mark.skipif(
    kernels.triton is None or not (kernels.INTERPRET or torch.cuda.is_available()),
    reason="needs TRITON_INTERPRET=1 or a GPU",
)
DEV = "cuda" if torch.cuda.is_available() and not kernels.INTERPRET else "cpu"


def _rms(x, w, eps):  # engine.fast.model._rms (not imported: it pulls in Transformers)
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + w.float())).to(x.dtype)


def _close_bf16(a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = a.float().cpu(), b.float().cpu()
    # at most one BF16 ulp apart (8 mantissa bits: relative 2^-7)
    assert torch.all((a - b).abs() <= b.abs() * 2.0 ** -7 + 1e-6), (a - b).abs().max()


@pytest.mark.parametrize("rows,n", [(1, 256), (8, 2048), (5, 2048)])
def test_mtp_in_matches_torch(rows, n) -> None:
    torch.manual_seed(rows * 7 + n)
    vocab = 300
    embed = torch.randn(vocab, n).to(torch.bfloat16).to(DEV)
    hidden = (torch.randn(rows, n) * 3).to(torch.bfloat16).to(DEV)
    w_e = (torch.randn(n) * 0.1).to(torch.bfloat16).to(DEV)
    w_h = (torch.randn(n) * 0.1).to(torch.bfloat16).to(DEV)
    tokens = torch.randint(0, vocab, (rows,), dtype=torch.long, device=DEV)
    eps = 1e-6
    ref = torch.cat((_rms(F.embedding(tokens, embed), w_e, eps), _rms(hidden, w_h, eps)), dim=-1)
    got = kernels.mtp_in(tokens, embed, hidden, w_e, w_h, eps)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    _close_bf16(got, ref)


def _accept_ref(tokens, picks, d, y, positions, groups):
    """``FastModel._chain_head_op``'s torch accept math (plan 51's parent)."""
    rows = y.shape[0]
    b = rows // groups
    tok = tokens[:rows].view(groups, b)
    match = (tok[1:] == picks.view(groups, b)[:-1]).long()
    a = torch.cumprod(match, dim=0).sum(dim=0)
    sel = a * b + torch.arange(b, device=a.device)
    h = y.index_select(0, sel)
    d1 = d.index_select(0, sel)
    base = positions[:b] + a
    posn = [base + j for j in range(1, groups - 1)]
    return a, h, d1, posn


@pytest.mark.parametrize("groups", [2, 3, 4])
@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_chain_accept_matches_torch(groups, b) -> None:
    torch.manual_seed(groups * 11 + b)
    rows, n = groups * b, 256
    for trial in range(6):
        picks = torch.randint(0, 4, (rows,), dtype=torch.long, device=DEV)
        tokens = torch.randint(0, 4, (rows + 3,), dtype=torch.long, device=DEV)
        if trial % 2:  # force long accepted runs
            tokens[b:rows] = picks[: rows - b]
            if trial == 3:
                tokens[rows - 1] = picks[rows - 1 - b] + 1
        d = torch.randint(0, 1000, (rows,), dtype=torch.long, device=DEV)
        y = torch.randn(rows, n).to(torch.bfloat16).to(DEV)
        positions = torch.randint(10, 500, (rows,), dtype=torch.long, device=DEV)
        a, h, d1, posn = _accept_ref(tokens, picks, d, y, positions, groups)
        out, h2, posn2 = kernels.chain_accept(tokens[:rows], picks, d, y, positions[:b], groups)
        assert out.shape == (rows + groups * b,)
        assert torch.equal(out[:rows].cpu(), picks.cpu())
        assert torch.equal(out[rows: rows + b].cpu(), a.cpu())
        assert torch.equal(out[rows + b: rows + 2 * b].cpu(), d1.cpu())
        assert torch.equal(h2.cpu(), h.cpu())
        for j, p in enumerate(posn):
            assert torch.equal(posn2[j].cpu(), p.cpu())


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16])
@pytest.mark.parametrize("ksplit", [2, 4])
def test_gemv_defer_materialize_bitwise(rows, ksplit) -> None:
    torch.manual_seed(rows + 100 * ksplit)
    n, k = 96, 256
    w = (torch.randn(n, k) * k ** -0.5).to(torch.bfloat16).to(DEV)
    x = torch.randn(rows, k).to(torch.bfloat16).to(DEV)
    tiles = (32, 32, 4, ksplit)
    with kernels.on_device(x):
        ref = mfma.gemv(x, w, tiles=tiles).clone()
        parts = mfma.gemv(x, w, tiles=tiles, defer=True)
        assert isinstance(parts, mfma.KParts)
        got = parts.materialize()
    assert torch.equal(got.cpu(), ref.cpu())


@pytest.fixture(scope="module")
def ll():
    from engine.fast import tp

    return tp._LL([torch.device("cuda", 0), torch.device("cuda", 1)])


@pytest.mark.skipif(not torch.cuda.is_available() or kernels.INTERPRET or torch.cuda.device_count() < 2,
                    reason="the LL all-reduce needs two GPUs")
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 12, 16])
def test_allreduce_ksum_prologue_bitwise(rows, ll) -> None:
    dev = torch.device("cuda", 0)
    torch.manual_seed(rows)
    n, k = 2048, 2048
    w = (torch.randn(n, k) * k ** -0.5).to(torch.bfloat16).to(dev)
    tiles = mfma.tiles_for(n, k)
    assert tiles[3] > 1
    for trial in range(4):
        x = (torch.randn(rows, k) * (1 + 10 * trial)).to(torch.bfloat16).to(dev)
        res = torch.randn(rows, n).to(torch.bfloat16).to(dev)
        with torch.cuda.device(dev):
            summed = mfma.gemv(x, w)
            ref = ll._launch(0, summed, None, loop=True).clone()
            ref_res = ll._launch(0, summed, res, loop=True).clone()
            got = ll._launch(0, mfma.gemv(x, w, defer=True), None, loop=True).clone()
            got_res = ll._launch(0, mfma.gemv(x, w, defer=True), res, loop=True).clone()
            torch.cuda.synchronize(dev)
        assert torch.equal(got.view(torch.int16).cpu(), ref.view(torch.int16).cpu())
        assert torch.equal(got_res.view(torch.int16).cpu(), ref_res.view(torch.int16).cpu())
    assert ll.timeouts() == 0
