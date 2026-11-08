"""The fused MoE decode work order (plan 22, ``engine.fast.fused.MOE_ORDER``): the tile-outer
gate_up and tile-then-row down kernels give bitwise the same output as plan 11's order, at 1,
2, 4 and 8 rows with rows that share experts, and a row's output does not depend on the row
count.

Runs on a GPU (the checkpoint's expert shapes, and the half-width expert shard of a TP2 split)
or under Triton's CPU interpreter (``TRITON_INTERPRET=1``, small shapes); skipped otherwise.
Also a script without pytest (``python tests/test_fused_moe.py``, as the GPU command jobs run
it).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
TOPK = 8
# (experts, hidden, expert intermediate, shared intermediate)
SHAPES = [(256, 2048, 512, 512), (256, 2048, 256, 256)] if GPU else [(32, 128, 32, 32), (32, 128, 16, 16)]
ROWS = (1, 2, 4, 8)


def _decode(shape, seed: int):
    """A FusedDecode with one random MoE layer of ``shape`` (index 0)."""
    E, H, I, IS = shape
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16).to(DEV)

    gate = rnd(E, H, scale=H ** -0.5 * 4)
    sh_gate, sh_up, sh_down = rnd(IS, H, scale=H ** -0.5), rnd(IS, H, scale=H ** -0.5), rnd(H, IS, scale=IS ** -0.5)
    seg = rnd(1, H, scale=H ** -0.5)
    mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=gate),
        shared_expert=SimpleNamespace(gate_proj=SimpleNamespace(weight=sh_gate),
                                      up_proj=SimpleNamespace(weight=sh_up),
                                      down_proj=SimpleNamespace(weight=sh_down)),
        shared_expert_gate=SimpleNamespace(weight=seg),
        experts=SimpleNamespace(gate_up_proj=rnd(E, 2 * I, H, scale=H ** -0.5),
                                down_proj=rnd(E, H, I, scale=I ** -0.5)),
    )
    fd = fused.FusedDecode.__new__(fused.FusedDecode)
    fd.fast = SimpleNamespace(top_k=TOPK)
    fd.max_rows = 8
    fd.cus = {}
    fd.w_moe = {0: torch.cat([gate, sh_gate, sh_up, seg]).contiguous()}
    layer = SimpleNamespace(mlp=mlp)
    fd._layer = lambda i: layer
    return fd, rnd


def _moe(fd, order: int, x, h):
    saved = fused.MOE_ORDER
    fused.MOE_ORDER = order
    try:
        with kernels.on_device(x):
            return fd._moe(0, x, h)
    finally:
        fused.MOE_ORDER = saved


def _inputs(rnd, H):
    """8 rows; rows 4-7 are rows 0-3 slightly perturbed (mostly the same experts), and row 7 is
    row 6 exactly (all experts shared), as the two rows of a verify slot often are."""
    h = rnd(8, H)
    h[4:] = (h[:4].float() + 0.02 * rnd(4, H).float()).to(torch.bfloat16)
    h[7] = h[6]
    x = rnd(8, H)
    return h, x


def _case(shape, seed: int) -> list[str]:
    fd, rnd = _decode(shape, seed)
    h, x = _inputs(rnd, shape[1])
    alone = [_moe(fd, 0, x[r : r + 1].contiguous(), h[r : r + 1].contiguous()) for r in range(8)]
    lines = []
    for R in ROWS:
        hr, xr = h[:R].contiguous(), x[:R].contiguous()
        old = _moe(fd, 0, xr, hr)
        new = _moe(fd, 1, xr, hr)
        assert torch.equal(old, new), (shape, R, (old.float() - new.float()).abs().max().item())
        for r in range(R):
            assert torch.equal(new[r : r + 1], alone[r]), (shape, R, r)
        lines.append(f"shape={shape} R={R} bitwise_new_vs_old=True row_invariant=True")
    return lines


def test_new_order_is_bitwise_the_old_one() -> None:
    for i, shape in enumerate(SHAPES):
        _case(shape, 100 + i)


if __name__ == "__main__":
    if not RUNNABLE:
        print("RESULT test_fused_moe SKIPPED (needs a GPU or TRITON_INTERPRET=1)")
        sys.exit(0)
    for i, shape in enumerate(SHAPES):
        for line in _case(shape, 100 + i):
            print("RESULT test_fused_moe " + line, flush=True)
    print("RESULT test_fused_moe PASSED", flush=True)
