"""The multi-row fused GEMV (plan 19, ``engine.fast.fused._mr_gemv_kernel``): a row's output
is bitwise the same whatever the row count, matches an fp32 reference, and its conv epilogue
equals the GEMV followed by ``kernels.gdn_conv_``.

Runs on a GPU (the real decode weight shapes) or under Triton's CPU interpreter
(``TRITON_INTERPRET=1``, small shapes); skipped otherwise. Also a script without pytest
(``python tests/test_fused_gemv.py``, as the GPU command jobs run it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    import pytest
except ImportError:  # the serving image has no pytest: run as a script
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fast import fused, kernels  # noqa: E402

GPU = torch.cuda.is_available() and kernels.triton is not None and not kernels.INTERPRET
RUNNABLE = kernels.triton is not None and (GPU or kernels.INTERPRET)
if pytest is not None:
    pytestmark = pytest.mark.skipif(not RUNNABLE, reason="needs a GPU or TRITON_INTERPRET=1")
DEV = torch.device("cuda", 0) if GPU else torch.device("cpu")
# GPU: every tabled decode shape (K split included); CPU: small shapes off the table and a
# small table entry (with a K split) installed by ``_tiles``.
SHAPES = list(fused.GEMV_TILES) if GPU else [(37, 256), (64, 512), (24, 128)]


def _tiles() -> None:
    """The multi-row path on, and on the CPU a small table entry with a K split."""
    fused.GEMV = True
    if not GPU:
        fused.GEMV_TILES[(24, 128)] = (8, 32, 4, 2, 1, {1: 1, 8: 0})


def _decode() -> fused.FusedDecode:
    fd = fused.FusedDecode.__new__(fused.FusedDecode)
    fd.cus = {}
    return fd


def _gemv(fd, x, w, conv=None):
    with kernels.on_device(x):
        return fd._gemv(x.contiguous(), w, conv=conv)


def test_rows_are_bitwise_independent_of_the_row_count() -> None:
    _tiles()
    for shape in SHAPES:
        _rows_case(shape)


def _rows_case(shape: tuple[int, int]) -> None:
    n, k = shape
    g = torch.Generator(device=DEV).manual_seed(n + k)
    w = (torch.randn((n, k), generator=g, device=DEV) * k ** -0.5).to(torch.bfloat16)
    x = torch.randn((12, k), generator=g, device=DEV).to(torch.bfloat16)
    fd = _decode()
    alone = [_gemv(fd, x[r : r + 1], w)[0] for r in range(12)]
    ref = x.float() @ w.float().t()
    for rows in (2, 3, 4, 5, 8, 12):
        y = _gemv(fd, x[:rows], w)
        for r in range(rows):
            assert torch.equal(y[r], alone[r]), (shape, rows, r)
        err = ((y.float() - ref[:rows]).abs() / (ref[:rows].abs() + 1)).max().item()
        assert err < 2e-2, (shape, rows, err)


def test_conv_epilogue_equals_gemv_then_conv() -> None:
    _tiles()
    for rows in (1, 2, 4, 8):
        _conv_case(rows)


def _conv_case(rows: int) -> None:
    n, k = (12352, 2048) if GPU else (40, 256)
    conv_n, kc, slots_total = (8192 if GPU else 24), 4, 10
    g = torch.Generator(device=DEV).manual_seed(rows)
    w = (torch.randn((n, k), generator=g, device=DEV) * k ** -0.5).to(torch.bfloat16)
    x = torch.randn((rows, k), generator=g, device=DEV).to(torch.bfloat16)
    cw = torch.randn((conv_n, 1, kc), generator=g, device=DEV).to(torch.bfloat16)
    state = torch.randn((slots_total, conv_n, kc), generator=g, device=DEV).to(torch.bfloat16)
    slots = torch.tensor([7, 2, 5, 0, 9, 1, 3, 8][:rows], dtype=torch.long, device=DEV)
    fd = _decode()
    s1 = state.clone()
    fusedout = _gemv(fd, x, w, conv=(s1, slots, cw, conv_n))
    s2 = state.clone()
    plain = _gemv(fd, x, w)
    with kernels.on_device(x):
        conv = kernels.gdn_conv_(s2, slots, plain[:, :conv_n], cw)
    assert torch.equal(fusedout[:, conv_n:], plain[:, conv_n:])
    assert torch.equal(fusedout[:, :conv_n], conv)
    assert torch.equal(s1, s2)


if __name__ == "__main__":
    if not RUNNABLE:
        raise SystemExit("needs a GPU or TRITON_INTERPRET=1")
    test_rows_are_bitwise_independent_of_the_row_count()
    print(f"RESULT test_fused_gemv rows_invariant=pass shapes={SHAPES}", flush=True)
    test_conv_epilogue_equals_gemv_then_conv()
    print("RESULT test_fused_gemv conv_epilogue=pass", flush=True)
