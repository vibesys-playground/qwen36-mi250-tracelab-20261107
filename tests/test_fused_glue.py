"""Plan 29's glue fusions are bitwise the launches they replace.

- ``ENGINE_GLUE_NOZERO``: ``FusedDecode._moe(i, None, h)`` equals ``_moe(i, zeros_like, h)``.
- ``ENGINE_GLUE_TOPK``: the routed gate_up kernel's own top-k (``_topk`` per work item) gives
  the same experts, weights and MoE output as ``_topk_kernel``, on router logits with ties.
- ``ENGINE_GLUE_ARRMS``: ``_ll_ar_rms_kernel`` on two GPUs equals the LL all-reduce followed by
  ``kernels.add_rms`` (both forms the layers use: residual in the add_rms, or in the
  all-reduce), and both ranks get the same bits.

Runs on GPUs (the checkpoint's shapes; the all-reduce part needs two) or under Triton's CPU
interpreter (``TRITON_INTERPRET=1``, small shapes, the all-reduce against a pre-filled
mailbox); skipped otherwise. Also a script without pytest (``python tests/test_fused_glue.py``).
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
sys.path.insert(0, str(ROOT / "tests"))

from engine.fast import fused, kernels  # noqa: E402
from test_fused_moe import DEV, GPU, RUNNABLE, SHAPES, _decode, _inputs  # noqa: E402

if pytest is not None:
    pytestmark = pytest.mark.skipif(not RUNNABLE, reason="needs a GPU or TRITON_INTERPRET=1")
ROWS = (1, 2, 4, 8)
TWO_GPUS = GPU and torch.cuda.device_count() >= 2


def _moe(fd, x, h, order: int, topk: bool):
    saved = fused.MOE_ORDER, fused.GLUE_TOPK
    fused.MOE_ORDER, fused.GLUE_TOPK = order, topk
    try:
        with kernels.on_device(h):
            return fd._moe(0, x, h)
    finally:
        fused.MOE_ORDER, fused.GLUE_TOPK = saved


def _moe_cases() -> list[str]:
    lines = []
    for n, shape in enumerate(SHAPES):
        fd, rnd = _decode(shape, 300 + n)
        h, _ = _inputs(rnd, shape[1])
        for R in ROWS:
            hr = h[:R].contiguous()
            zero = torch.zeros_like(hr)
            for order in (0, 1):
                ref = _moe(fd, zero, hr, order, False)
                nozero = _moe(fd, None, hr, order, False)
                topk = _moe(fd, zero, hr, order, True)
                both = _moe(fd, None, hr, order, True)
                for name, got in (("nozero", nozero), ("topk", topk), ("both", both)):
                    assert torch.equal(ref, got), (name, shape, R, order,
                                                   (ref.float() - got.float()).abs().max().item())
            lines.append(f"shape={shape} R={R} nozero=bitwise topk=bitwise both=bitwise")
    return lines


def _router_rows(n: int, E: int, N3: int, seed: int) -> torch.Tensor:
    """Router logits rows (BF16, row stride N3) with the cases that matter for ties."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(n, N3, generator=g)
    k = n // 4
    r[:k, :E] *= 4.0  # peaked
    r[k : 2 * k, :E] = torch.round(r[k : 2 * k, :E] * 2) / 2  # coarse: many exact ties
    r[2 * k : 2 * k + 8, :E] = 0.5  # all experts tied
    r[2 * k + 8 : 3 * k, :E] *= 0.05  # flat
    r[3 * k :, :E] = r[3 * k :, :E] * 8 + 30  # large logits
    return r.to(torch.bfloat16).to(DEV)


def _topk_mass(rows_per_call: int = 64, calls: int = 32) -> str:
    """The up kernel's experts/weights against ``_topk_kernel`` over many router rows."""
    shape = SHAPES[-1]
    E, H, I, IS = shape
    fd, rnd = _decode(shape, 400)
    w3 = fd.w_moe[0]
    N3 = w3.shape[0]
    gu = fd._layer(0).mlp.experts.gate_up_proj
    top_k, bn = 8, fused.TILES["moe_bn"]
    bad = 0
    total = 0
    for c in range(calls if GPU else 2):
        R = rows_per_call if GPU else 8
        r = _router_rows(R, E, N3, 500 + c)
        h = rnd(R, H)
        idx0 = torch.empty((R, top_k), dtype=torch.int32, device=DEV)
        wt0 = torch.empty((R, top_k), dtype=torch.float32, device=DEV)
        idx1 = torch.full_like(idx0, -1)
        wt1 = torch.full_like(wt0, -1.0)
        act0 = torch.empty((R * top_k, I), dtype=torch.bfloat16, device=DEV)
        act1 = torch.empty_like(act0)
        items = R * top_k * kernels.triton.cdiv(I, bn)
        with kernels.on_device(h):
            fused._topk_kernel[(R,)](r, idx0, wt0, N3, E=E, TOPK=top_k, TOPKP=8, num_warps=1)
            for fuse, idx, wt, act in ((False, idx0, wt0, act0), (True, idx1, wt1, act1)):
                fused._moe_up_kernel[(items,)](
                    h, idx, gu, act, R, gu.stride(0), r, wt, N3,
                    H=H, I=I, TOPK=top_k, BN=bn, BK=min(1024, fused._pow2(H)), ORDER=1, E=E,
                    TOPKP=8, FUSE_TOPK=fuse, num_warps=4,
                )
        same = (idx0 == idx1).all(-1) & (wt0.view(torch.int32) == wt1.view(torch.int32)).all(-1)
        bad += int((~same).sum())
        total += R
        assert torch.equal(act0, act1), ("act", c)
    assert bad == 0, f"{bad}/{total} router rows differ"
    return f"router_rows={total} experts_and_weights=bitwise act=bitwise"


def test_moe_glue_is_bitwise() -> None:
    _moe_cases()


def test_topk_on_tied_router_rows() -> None:
    _topk_mass()


# ------------------------------------------------------------------ all-reduce + RMSNorm


def _hidden(g, R: int, H: int, dev) -> torch.Tensor:
    x = torch.randn(R, H, generator=g)
    x[:, 7] *= 300.0  # massive-activation channels, as real hidden states have
    x[:, H // 2 + 7] *= -80.0
    return x.to(torch.bfloat16).to(dev)


def _ar_rms_two_gpus(iters: int = 12) -> list[str]:
    from engine.fast import tp

    saved = tp.GLUE_ARRMS
    tp.GLUE_ARRMS = True
    try:
        devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
        group = tp.Group(devs)
        ll = group.ll
        assert ll is not None and ll.rms_ready
        H = tp.RMS_WIDTH
        eps = 1e-6
        g = torch.Generator(device="cpu").manual_seed(7)
        w = [(torch.randn(H, generator=g) * 0.3).to(torch.bfloat16).to(d) for d in devs]
        lines = []
        for R in (1, 2, 4, 8, 16):
            for it in range(iters):
                res = _hidden(g, R, H, "cpu")
                parts = [(torch.randn(R, H, generator=g) * (0.5 + it)).to(torch.bfloat16) for _ in devs]
                res_d = [res.to(d) for d in devs]
                part_d = [p.to(d) for p, d in zip(parts, devs)]
                # Reference A: all-reduce, then add_rms(res, s)
                s = [ll._launch(r, part_d[r], None) for r in range(2)]
                ref_a = [kernels.add_rms(res_d[r], s[r], w[r], eps) for r in range(2)]
                # Reference B: all-reduce with the residual, then add_rms(total, None)
                t = [ll._launch(r, part_d[r], res_d[r]) for r in range(2)]
                ref_b = [(t[r], kernels.add_rms(t[r], None, w[r], eps)[1]) for r in range(2)]
                got = [ll._launch_rms(r, part_d[r], res_d[r], w[r], eps) for r in range(2)]
                for d in devs:
                    torch.cuda.synchronize(d)
                for r in range(2):
                    for name, ref in (("A", ref_a[r]), ("B", ref_b[r])):
                        assert torch.equal(ref[0], got[r][0]), ("sum", name, R, it, r)
                        assert torch.equal(ref[1], got[r][1]), (
                            "norm", name, R, it, r,
                            int((ref[1] != got[r][1]).sum()),
                        )
                assert torch.equal(got[0][0].cpu(), got[1][0].cpu())
                assert torch.equal(got[0][1].cpu(), got[1][1].cpu())
            lines.append(f"R={R} iters={iters} sum=bitwise norm=bitwise (vs AR then add_rms, both forms) "
                         f"ranks_equal=True")
        assert ll.timeouts() == 0, ll.timeouts()
        lines.append(f"timeouts={ll.timeouts()}")
        return lines
    finally:
        tp.GLUE_ARRMS = saved


def _ar_rms_interpret() -> list[str]:
    """CPU: rank 0's kernel against a mailbox pre-filled with the peer's words."""
    from engine.fast import tp

    H = 256
    eps = 1e-6
    g = torch.Generator(device="cpu").manual_seed(3)
    lines = []
    for R in (1, 3):
        res = _hidden(g, R, H, "cpu")[:, :H]
        # Partials whose sum is exact in BF16, so the reference's torch rounding of the sum
        # cannot differ from the interpreter's.
        mine = (torch.randint(-64, 64, (R, H), generator=g) / 8).to(torch.bfloat16)
        peer = (torch.randint(-64, 64, (R, H), generator=g) / 8).to(torch.bfloat16)
        w = (torch.randn(H, generator=g) * 0.3).to(torch.bfloat16)
        cap = R * H
        local = torch.zeros(2 * cap, dtype=torch.int64)
        epoch = torch.zeros(R, dtype=torch.int32)
        bits = peer.view(torch.int16).to(torch.int64) & 0xFFFF
        local[cap : 2 * cap] = ((1 << 32) | bits).view(-1)  # epoch 1, parity 1
        sink = torch.zeros(2 * cap, dtype=torch.int64)
        err = torch.zeros(1, dtype=torch.int32)
        total = torch.empty_like(res)
        out = torch.empty_like(res)
        tp._ll_ar_rms_kernel[(R,)](mine, res, w, total, out, local, sink, epoch, err, eps, cap,
                                   N=H, BLOCK=H, MAXIT=10)
        s = (mine.float() + peer.float()).to(torch.bfloat16)
        ref_total, ref_out = kernels.add_rms(res, s, w, eps)
        assert int(err) == 0 and torch.equal(epoch, torch.ones(R, dtype=torch.int32))
        assert torch.equal(total, ref_total)
        assert torch.equal(out, ref_out)
        sent = sink[cap : 2 * cap]
        assert torch.equal(sent >> 32, torch.ones_like(sent))
        assert torch.equal((sent & 0xFFFF).view(R, H), mine.view(torch.int16).to(torch.int64) & 0xFFFF)
        lines.append(f"interpret R={R} sum=bitwise norm=bitwise sent=ok")
    return lines


def test_ar_rms_is_bitwise() -> None:
    if TWO_GPUS:
        _ar_rms_two_gpus()
    elif kernels.INTERPRET:
        _ar_rms_interpret()
    elif pytest is not None:
        pytest.skip("needs two GPUs or TRITON_INTERPRET=1")


if __name__ == "__main__":
    if not RUNNABLE:
        print("RESULT test_fused_glue SKIPPED (needs a GPU or TRITON_INTERPRET=1)")
        sys.exit(0)
    ok = True
    for name, fn in (("moe", _moe_cases), ("topk_mass", lambda: [_topk_mass()]),
                     ("ar_rms", _ar_rms_two_gpus if TWO_GPUS else
                      (_ar_rms_interpret if kernels.INTERPRET else (lambda: ["SKIPPED (one GPU)"])))):
        try:
            for line in fn():
                print(f"RESULT test_fused_glue {name} {line}", flush=True)
            print(f"RESULT test_fused_glue {name} PASSED", flush=True)
        except AssertionError as exc:
            ok = False
            print(f"RESULT test_fused_glue {name} FAILED {exc!r}"[:600], flush=True)
    print(f"RESULT test_fused_glue {'PASSED' if ok else 'FAILED'}", flush=True)
    sys.exit(0 if ok else 1)
