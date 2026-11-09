"""Plan 57 stage 0: one TP2 decode layer's streaming span as ONE grid-synchronized Triton kernel.

    python tools/span_probe.py [--devices 0,1] [--layer 0] [--rows 4,8] [--iters 200]
                               [--stress 2000] [--long 10]

The span of a Gated DeltaNet layer ``L`` of the chained verify step (k = 3, rows = 4 per slot)
as today's captured graph runs it on each rank (``FastModel._fused_layer`` under ENGINE_TP=2,
``tp.AR_KSUM``, ``mfma.TOPK_EPI``), ten kernels:

    out_proj MFMA GEMV (K parts)  -> LL all-reduce (K-sum prologue) -> add_rms (post-attn norm)
    -> router|shared MFMA GEMV (K parts) -> K-sum + router top-k -> MoE up -> MoE down
    -> LL all-reduce + residual -> add_rms (next layer's input norm) -> next in_proj MFMA GEMV

and the same work as one persistent kernel (``_span_kernel``): a resident grid, eight phases
split by seven grid barriers (a monotonic counter, acq_rel arrive, acquire spin with a timeout),
the all-reduce + RMSNorm of a row in one CTA per row (an LL exchange on its own fine-grained
buffers, then exactly ``kernels._add_rms_kernel``'s code). Every phase body is today's kernel
body (same tiles, fp32 accumulation, BF16 rounding points, item order); only the 1-warp router
top-k's two sums are written as an explicit tree (``_topk_emul``) whose order is picked by a
calibration against today's 1-warp kernel (part A) so the 4-warp kernel can match its bits.

Inputs are real: the replica is loaded as the server loads it (``Generator.load``), two slots
are prefilled with replay-like text, and layer L's out_proj input (``core``) and residual of a
chained verify step (B = 1 -> 4 rows, B = 2 -> 8 rows, MTP drafts) are recorded from an eager
step on both ranks.

Parts (all ``RESULT`` lines):
  A. top-k sum-order calibration (1-warp reference vs 4-warp tree variants).
  B. ceiling of today's span: empty-kernel graph node cost; a streaming-read rate; each of the
     ten kernels alone (a graph of ``--long`` repeats, both ranks); the span as one graph and
     as ``--long`` spans in one graph; boundary cost split into node launch and fill/drain.
  C. the fused span kernel: VGPRs / spills / LDS / occupancy, a grid-barrier microbenchmark,
     then per variant (num_stages, pre-barrier tile prefetch) and grid (CTAs per CU) at each row
     count: bitwise compare with today's span on both ranks, span time (1 and ``--long`` per
     graph), per-phase time from ``s_memrealtime`` stamps (25 MHz) and weight-read TB/s
     against each phase's standalone kernel.
  D. stress: ``--stress`` replays of the best variant per row count, 0 barrier / all-reduce
     timeouts and bitwise outputs required.
  Verdict: kill if span(4 rows) > 0.88x today's graph, any streaming phase < 0.95x its
  standalone rate, any bit differs, or any timeout/hang.

A tool for the plan, not the task's metric; it changes nothing in the served path.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pre_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default="0,1")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", default="4,8")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--stress", type=int, default=2000)
    p.add_argument("--long", type=int, default=10)
    p.add_argument("--prompt-len", type=int, default=1800)
    p.add_argument("--grids", default="1,2,3,4", help="CTAs per CU to try (bounded by occupancy)")
    p.add_argument("--variants", default="1:0,1:1,2:0,2:1", help="num_stages:prefetch")
    return p.parse_args()


ARGS = _pre_args()
os.environ["HIP_VISIBLE_DEVICES"] = ARGS.devices
os.environ.setdefault("ENGINE_MTP", "1")
os.environ.setdefault("ENGINE_TP", "2")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

T0 = time.perf_counter()


def out(msg: str) -> None:
    print(msg, flush=True)


def log(msg: str) -> None:
    out(f"[{time.perf_counter() - T0:7.1f}s] {msg}")


MAXIT = 4_000_000     # LL all-reduce spin iterations before giving up (as tp.MAXIT)
MAXSPIN = 2_000_000   # grid-barrier spin iterations before giving up
NBAR = 7              # grid barriers per span launch
NST = NBAR + 2        # stamps per launch: start, 7 barrier exits (CTA 0), end (CTA 0)
RING = 256            # stamp rows (launches) kept
MAXCTA = 1024
RT_HZ = 25e6          # s_memrealtime on MI250 (plan 55, j261)
PFK = 256             # K columns of a prefetched tile

# ============================================================================ kernels


@triton.jit
def _realtime():
    return tl.inline_asm_elementwise("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [],
                                     dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _vm_drain():
    """Every wave waits for its own outstanding vector memory ops (stores acknowledged)."""
    return tl.inline_asm_elementwise("s_waitcnt vmcnt(0)\n\ts_mov_b32 $0, 0", "=s", [],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _arrive(bar_ptr):
    _vm_drain()
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")


@triton.jit
def _wait(bar_ptr, target, err_ptr, MAXSPIN: tl.constexpr):
    cnt = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    spins = 0
    while (cnt < target) & (spins < MAXSPIN):
        cnt = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        spins += 1
    if spins >= MAXSPIN:
        tl.atomic_add(err_ptr + 1, 1)
    tl.debug_barrier()


@triton.jit
def _calib_kernel(bar_ptr):
    _arrive(bar_ptr)


@triton.jit(do_not_specialize=["iters", "per_barrier"])
def _bar_bench_kernel(bar_ptr, err_ptr, buf_ptr, bad_ptr, iters, per_barrier,
                      MAXSPIN: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    per_launch = tl.maximum(per_barrier * iters, 1)
    c0 = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    base = (c0 // per_launch) * per_launch
    nb = (pid + 1) % nprog
    bad = 0
    for it in range(iters):
        tl.store(buf_ptr + (it % 2) * nprog + pid, it * 7 + pid)
        _arrive(bar_ptr)
        _wait(bar_ptr, base + (it + 1) * per_barrier, err_ptr, MAXSPIN)
        v = tl.load(buf_ptr + (it % 2) * nprog + nb, volatile=True)
        bad += (v != it * 7 + nb).to(tl.int32)
    tl.atomic_add(bad_ptr, bad)


@triton.jit
def _empty_kernel(ptr):
    if tl.program_id(0) < 0:
        tl.store(ptr, 0)


@triton.jit
def _stream_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(pid * BLOCK, n, nprog * BLOCK):
        acc += tl.load(x_ptr + i + tl.arange(0, BLOCK)).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


# ---------------------------------------------------------------- explicit sum trees (top-k)


@triton.jit
def _thread4(p, STRIDED: tl.constexpr):
    """A 1-warp [256] reduction's in-thread step: lane t sums its 4 values in order, either
    contiguous (4t .. 4t+3) or strided (t, t+64, t+128, t+192). Each value is picked out
    exactly (a sum of it and zeros)."""
    jj = tl.arange(0, 4)
    if STRIDED:
        q = tl.reshape(p, [4, 64])
        v0 = tl.sum(tl.where(jj[:, None] == 0, q, 0.0), axis=0)
        v1 = tl.sum(tl.where(jj[:, None] == 1, q, 0.0), axis=0)
        v2 = tl.sum(tl.where(jj[:, None] == 2, q, 0.0), axis=0)
        v3 = tl.sum(tl.where(jj[:, None] == 3, q, 0.0), axis=0)
    else:
        q = tl.reshape(p, [64, 4])
        v0 = tl.sum(tl.where(jj[None, :] == 0, q, 0.0), axis=1)
        v1 = tl.sum(tl.where(jj[None, :] == 1, q, 0.0), axis=1)
        v2 = tl.sum(tl.where(jj[None, :] == 2, q, 0.0), axis=1)
        v3 = tl.sum(tl.where(jj[None, :] == 3, q, 0.0), axis=1)
    return ((v0 + v1) + v2) + v3


@triton.jit
def _comb(v, N: tl.constexpr, O: tl.constexpr):
    """v[i] + v[i ^ O] for the [N] vector, the bit of value O dropped from the index."""
    return tl.reshape(tl.sum(tl.reshape(v, [N // (2 * O), 2, O]), axis=1), [N // 2])


@triton.jit
def _tree64(s, TREE: tl.constexpr):
    """A 64-lane cross-lane reduction: 0 the gfx9 DPP order (lanes 8, 4, 2, 1 in a row, then
    rows 16, 32), 1 xor butterfly 32 .. 1, 2 xor butterfly 1 .. 32."""
    if TREE == 0:
        s = _comb(s, 64, 8)
        s = _comb(s, 32, 4)
        s = _comb(s, 16, 2)
        s = _comb(s, 8, 1)
        s = _comb(s, 4, 1)
        s = _comb(s, 2, 1)
    elif TREE == 1:
        s = _comb(s, 64, 32)
        s = _comb(s, 32, 16)
        s = _comb(s, 16, 8)
        s = _comb(s, 8, 4)
        s = _comb(s, 4, 2)
        s = _comb(s, 2, 1)
    else:
        s = _comb(s, 64, 1)
        s = _comb(s, 32, 1)
        s = _comb(s, 16, 1)
        s = _comb(s, 8, 1)
        s = _comb(s, 4, 1)
        s = _comb(s, 2, 1)
    return tl.sum(s, axis=0)


@triton.jit
def _sum8(v, kk, SELMODE: tl.constexpr):
    if SELMODE == 0:
        r = tl.sum(v, axis=0)
    elif SELMODE == 1:
        r = tl.sum(_comb(_comb(_comb(v, 8, 4), 4, 2), 2, 1), axis=0)
    elif SELMODE == 2:
        r = tl.sum(_comb(_comb(_comb(v, 8, 1), 4, 1), 2, 1), axis=0)
    else:
        r = tl.sum(tl.where(kk == 0, v, 0.0), axis=0)
        for j in tl.static_range(1, 8):
            r = r + tl.sum(tl.where(kk == j, v, 0.0), axis=0)
    return r


@triton.jit
def _den(p, STRIDED: tl.constexpr, TREE: tl.constexpr):
    if TREE < 0:
        d = tl.sum(p, axis=0)
    else:
        d = _tree64(_thread4(p, STRIDED), TREE)
    return d


@triton.jit
def _topk_emul(p, ee, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr,
               STRIDED: tl.constexpr, TREE: tl.constexpr, SELMODE: tl.constexpr):
    """``mfma._topk_vals`` with its two sums in an explicit order (TREE < 0, SELMODE 0: the
    kernel's own tl.sum)."""
    p = tl.exp(p - tl.max(p, axis=0))
    p = p / _den(p, STRIDED, TREE)
    kk = tl.arange(0, TOPKP)
    sel_w = tl.zeros([TOPKP], dtype=tl.float32)
    sel_i = tl.zeros([TOPKP], dtype=tl.int32)
    for j in tl.static_range(TOPK):
        mx = tl.max(p, axis=0)
        i = tl.min(tl.where(p == mx, ee, E), axis=0)
        sel_w = tl.where(kk == j, mx, sel_w)
        sel_i = tl.where(kk == j, i, sel_i)
        p = tl.where(ee == i, -1.0, p)
    sel_w = sel_w / _sum8(sel_w, kk, SELMODE)
    return sel_i, sel_w.to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["N"])
def _cal_ref_kernel(part_ptr, den_ptr, sw_ptr, idx_ptr, wt_ptr, N, KSPLIT: tl.constexpr,
                    BM: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr):
    """``mfma._ksum_router_kernel``'s top-k program (1 warp), also storing its two sums."""
    row = tl.program_id(0)
    ee = tl.arange(0, E)
    kk = tl.arange(0, TOPKP)
    s = tl.load(part_ptr + row * N + ee)
    for p in tl.static_range(1, KSPLIT):
        s += tl.load(part_ptr + (p * BM + row) * N + ee)
    v = s.to(tl.bfloat16).to(tl.float32)
    p = tl.exp(v - tl.max(v, axis=0))
    den = tl.sum(p, axis=0)
    p = p / den
    sel_w = tl.zeros([TOPKP], dtype=tl.float32)
    sel_i = tl.zeros([TOPKP], dtype=tl.int32)
    for j in tl.static_range(TOPK):
        mx = tl.max(p, axis=0)
        i = tl.min(tl.where(p == mx, ee, E), axis=0)
        sel_w = tl.where(kk == j, mx, sel_w)
        sel_i = tl.where(kk == j, i, sel_i)
        p = tl.where(ee == i, -1.0, p)
    sw = tl.sum(sel_w, axis=0)
    sel_w = sel_w / sw
    tl.store(den_ptr + row, den)
    tl.store(sw_ptr + row, sw)
    tl.store(idx_ptr + row * TOPK + kk, sel_i, mask=kk < TOPK)
    tl.store(wt_ptr + row * TOPK + kk, sel_w.to(tl.bfloat16).to(tl.float32), mask=kk < TOPK)


@triton.jit(do_not_specialize=["N"])
def _cal_emul_kernel(part_ptr, den_ptr, sw_ptr, idx_ptr, wt_ptr, N, KSPLIT: tl.constexpr,
                     BM: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr,
                     STRIDED: tl.constexpr, TREE: tl.constexpr, SELMODE: tl.constexpr):
    """The same in a 4-warp program with the explicit sum orders."""
    row = tl.program_id(0)
    ee = tl.arange(0, E)
    kk = tl.arange(0, TOPKP)
    s = tl.load(part_ptr + row * N + ee)
    for p in tl.static_range(1, KSPLIT):
        s += tl.load(part_ptr + (p * BM + row) * N + ee)
    v = s.to(tl.bfloat16).to(tl.float32)
    p = tl.exp(v - tl.max(v, axis=0))
    den = _den(p, STRIDED, TREE)
    p = p / den
    sel_w = tl.zeros([TOPKP], dtype=tl.float32)
    sel_i = tl.zeros([TOPKP], dtype=tl.int32)
    for j in tl.static_range(TOPK):
        mx = tl.max(p, axis=0)
        i = tl.min(tl.where(p == mx, ee, E), axis=0)
        sel_w = tl.where(kk == j, mx, sel_w)
        sel_i = tl.where(kk == j, i, sel_i)
        p = tl.where(ee == i, -1.0, p)
    sw = _sum8(sel_w, kk, SELMODE)
    sel_w = sel_w / sw
    tl.store(den_ptr + row, den)
    tl.store(sw_ptr + row, sw)
    tl.store(idx_ptr + row * TOPK + kk, sel_i, mask=kk < TOPK)
    tl.store(wt_ptr + row * TOPK + kk, sel_w.to(tl.bfloat16).to(tl.float32), mask=kk < TOPK)


# ---------------------------------------------------------------- phase bodies (today's kernels)


@triton.jit
def _mm_tile(x_ptr, w_ptr, out_ptr, part_ptr, R, N, stride_out, t,
             K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             KSPLIT: tl.constexpr):
    """``mfma._mfma_gemv_kernel`` for program ``t``."""
    nb = t // KSPLIT
    kp = t % KSPLIT
    KP: tl.constexpr = K // KSPLIT
    rm = tl.arange(0, BM)
    mok = rm < R
    rn = nb * BN + tl.arange(0, BN)
    nok = rn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(kp * KP, kp * KP + KP, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
        b = tl.load(w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None], mask=nok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    m = mok[:, None] & nok[None, :]
    if KSPLIT == 1:
        tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(part_ptr + (kp * BM + rm[:, None]) * N + rn[None, :], acc, mask=m)


@triton.jit
def _pf_mm(w_ptr, N, t, ntiles, K: tl.constexpr, BN: tl.constexpr, KSPLIT: tl.constexpr,
           PFK: tl.constexpr):
    """Load (to L2) the first PFK columns of program t's weight tile; returns their sum."""
    nb = t // KSPLIT
    kp = t % KSPLIT
    KP: tl.constexpr = K // KSPLIT
    rn = nb * BN + tl.arange(0, BN)
    rk = kp * KP + tl.arange(0, PFK)
    ok = (rn < N) & (t < ntiles)
    w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], mask=ok[:, None], other=0.0)
    return tl.sum(tl.sum(w.to(tl.float32), axis=1), axis=0)


@triton.jit
def _arrms(row, AR: tl.constexpr, KSPLIT: tl.constexpr,
           part_ptr, src32_ptr, res_ptr, res32_ptr, hout_ptr, hout32_ptr, sum_ptr, out_ptr,
           w_ptr, eps, recv_local, recv_peer, ep_ptr, err_ptr, CAPW,
           H: tl.constexpr, BM: tl.constexpr, MAXIT: tl.constexpr):
    """Row ``row``'s TP2 all-reduce (``tp._ll_allreduce_kernel``'s exchange and arithmetic:
    AR 0 with the K-sum prologue, AR 1 with the residual), then ``kernels._add_rms_kernel``
    (AR 0: HAS_H with the residual ``res``; AR 1: on the all-reduced sum)."""
    WPR: tl.constexpr = H // 2
    slot = AR * 64 + row
    e = tl.load(ep_ptr + slot, volatile=True) + 1
    buf = (e % 2).to(tl.int64) * CAPW + slot * WPR
    offs = tl.arange(0, WPR)
    if KSPLIT > 1:
        el = offs * 2
        lo_ptr = part_ptr + row * H + el
        s0 = tl.load(lo_ptr)
        s1 = tl.load(lo_ptr + 1)
        for kp in tl.static_range(1, KSPLIT):
            s0 += tl.load(lo_ptr + kp * BM * H)
            s1 += tl.load(lo_ptr + kp * BM * H + 1)
        xi = ((s0.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF)
              | (s1.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16))
    else:
        xi = tl.load(src32_ptr + row * WPR + offs)
    word = (e.to(tl.int64) << 32) | (xi.to(tl.int64) & 0xFFFFFFFF)
    tl.store(recv_peer + buf + offs, word, cache_modifier=".wt")
    src = recv_local + buf + offs
    w = tl.load(src, volatile=True, cache_modifier=".cv")
    pending = tl.sum(((w >> 32).to(tl.int32) != e).to(tl.int32), axis=0)
    it = pending * 0
    while (pending > 0) & (it < MAXIT):
        w = tl.load(src, volatile=True, cache_modifier=".cv")
        pending = tl.sum(((w >> 32).to(tl.int32) != e).to(tl.int32), axis=0)
        it += 1
    if pending > 0:
        tl.atomic_add(err_ptr, 1)
    yi = (w & 0xFFFFFFFF).to(tl.int32)
    s_lo = (((xi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
            + ((yi & 0xFFFF) << 16).to(tl.float32, bitcast=True))
    s_hi = ((xi & -65536).to(tl.float32, bitcast=True)
            + (yi & -65536).to(tl.float32, bitcast=True))
    s_lo = s_lo.to(tl.bfloat16).to(tl.float32)
    s_hi = s_hi.to(tl.bfloat16).to(tl.float32)
    if AR == 1:
        ri = tl.load(res32_ptr + row * WPR + offs, volatile=True)
        s_lo = (((ri & 0xFFFF) << 16).to(tl.float32, bitcast=True) + s_lo).to(tl.bfloat16).to(tl.float32)
        s_hi = ((ri & -65536).to(tl.float32, bitcast=True) + s_hi).to(tl.bfloat16).to(tl.float32)
    lo = s_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    hi = s_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
    tl.store(hout32_ptr + row * WPR + offs, lo | hi)
    tl.store(ep_ptr + slot, e)
    _vm_drain()
    tl.debug_barrier()
    cols = tl.arange(0, H)
    ok = cols < H
    if AR == 0:
        x = tl.load(res_ptr + row * H + cols, mask=ok, other=0.0).to(tl.float32)
        h = tl.load(hout_ptr + row * H + cols, mask=ok, other=0.0, volatile=True).to(tl.float32)
        x = (x + h).to(tl.bfloat16).to(tl.float32)
        tl.store(sum_ptr + row * H + cols, x.to(tl.bfloat16), mask=ok)
    else:
        x = tl.load(hout_ptr + row * H + cols, mask=ok, other=0.0, volatile=True).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    wt = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(var + eps) * (1.0 + wt)
    tl.store(out_ptr + row * H + cols, y.to(tl.bfloat16), mask=ok)


@triton.jit
def _ksum_topk(t, nmain, part_ptr, out_ptr, idx_ptr, wt_ptr, R, N,
               KSPLIT: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr, E: tl.constexpr,
               TOPK: tl.constexpr, TOPKP: tl.constexpr, STRIDED: tl.constexpr,
               TREE: tl.constexpr, SELMODE: tl.constexpr):
    """``mfma._ksum_router_kernel``'s program ``t``."""
    if t < nmain:
        i = t * BLOCK + tl.arange(0, BLOCK)
        r = i // N
        n = i % N
        m = r < R
        s = tl.load(part_ptr + r * N + n, mask=m, other=0.0)
        for p in tl.static_range(1, KSPLIT):
            s += tl.load(part_ptr + (p * BM + r) * N + n, mask=m, other=0.0)
        tl.store(out_ptr + r * N + n, s.to(tl.bfloat16), mask=m)
    else:
        ee = tl.arange(0, E)
        kk = tl.arange(0, TOPKP)
        row = t - nmain
        s = tl.load(part_ptr + row * N + ee)
        for p in tl.static_range(1, KSPLIT):
            s += tl.load(part_ptr + (p * BM + row) * N + ee)
        v = s.to(tl.bfloat16).to(tl.float32)
        sel_i, sel_w = _topk_emul(v, ee, E, TOPK, TOPKP, STRIDED, TREE, SELMODE)
        tl.store(idx_ptr + row * TOPK + kk, sel_i, mask=kk < TOPK)
        tl.store(wt_ptr + row * TOPK + kk, sel_w, mask=kk < TOPK)


@triton.jit
def _up_item(t, P, h_ptr, idx_ptr, w_ptr, act_ptr, stride_we, H: tl.constexpr, I: tl.constexpr,
             TOPK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """``fused._moe_up_kernel`` work item ``t`` (ORDER 1, FUSE_TOPK off)."""
    pair = t % P
    tile = t // P
    row = pair // TOPK
    expert = tl.load(idx_ptr + pair, volatile=True).to(tl.int64)
    rows = tile * BN + tl.arange(0, BN)
    rok = rows < I
    wbase = w_ptr + expert * stride_we
    acc_g = tl.zeros([BN], dtype=tl.float32)
    acc_u = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, H, BK):
        cols = k0 + tl.arange(0, BK)
        cok = cols < H
        m = rok[:, None] & cok[None, :]
        xv = tl.load(h_ptr + row * H + cols, mask=cok, other=0.0).to(tl.float32)
        wg = tl.load(wbase + rows[:, None] * H + cols[None, :], mask=m, other=0.0).to(tl.float32)
        wu = tl.load(wbase + (rows[:, None] + I) * H + cols[None, :], mask=m, other=0.0).to(tl.float32)
        acc_g += tl.sum(wg * xv[None, :], axis=1)
        acc_u += tl.sum(wu * xv[None, :], axis=1)
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    a = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(act_ptr + pair * I + rows, (a * u).to(tl.bfloat16), mask=rok)


@triton.jit
def _down_item(t, R, act_ptr, r_ptr, idx_ptr, wt_ptr, w_ptr, sd_ptr, out_ptr, stride_we, N3,
               H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
               TOPK: tl.constexpr, BN: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr):
    """``fused._moe_down_kernel`` work item ``t`` (ORDER 1, HAS_RES off)."""
    row = t % R
    tile = t // R
    rows = tile * BN + tl.arange(0, BN)
    rok = rows < H
    rbase = r_ptr + row * N3
    acc = tl.zeros([BN], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        pair = row * TOPK + j
        expert = tl.load(idx_ptr + pair, volatile=True).to(tl.int64)
        weight = tl.load(wt_ptr + pair, volatile=True)
        wbase = w_ptr + expert * stride_we
        part = tl.zeros([BN], dtype=tl.float32)
        for k0 in range(0, I, BKI):
            cols = k0 + tl.arange(0, BKI)
            cok = cols < I
            hv = tl.load(act_ptr + pair * I + cols, mask=cok, other=0.0).to(tl.float32)
            w = tl.load(wbase + rows[:, None] * I + cols[None, :],
                        mask=rok[:, None] & cok[None, :], other=0.0).to(tl.float32)
            part += tl.sum(w * hv[None, :], axis=1)
        acc += (part.to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
    routed = acc.to(tl.bfloat16).to(tl.float32)
    shared = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, IS, BKS):
        sc = k0 + tl.arange(0, BKS)
        sok = sc < IS
        g = tl.load(rbase + E + sc, mask=sok, other=0.0).to(tl.float32)
        u = tl.load(rbase + E + IS + sc, mask=sok, other=0.0).to(tl.float32)
        sa = ((g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16).to(tl.float32)
        sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :],
                     mask=rok[:, None] & sok[None, :], other=0.0).to(tl.float32)
        shared += tl.sum(sw * sa[None, :], axis=1)
    shared = shared.to(tl.bfloat16).to(tl.float32)
    sg = tl.load(rbase + E + 2 * IS).to(tl.float32)
    sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
    moe = (routed + (sg * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    res = tl.zeros([BN], dtype=tl.float32)
    tl.store(out_ptr + row * H + rows, (res + moe).to(tl.bfloat16), mask=rok)


@triton.jit
def _pf_down(t, R, idx_ptr, w_ptr, sd_ptr, stride_we, H: tl.constexpr, I: tl.constexpr,
             IS: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr):
    """Load (to L2) down work item t's routed rows of its row's experts and the shared rows."""
    ok = t < R * (H // BN)
    row = t % R
    tile = t // R
    rows = tile * BN + tl.arange(0, BN)
    cols = tl.arange(0, I)
    acc = tl.zeros([BN, I], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        expert = tl.load(idx_ptr + row * TOPK + j, volatile=True).to(tl.int64)
        w = tl.load(w_ptr + expert * stride_we + rows[:, None] * I + cols[None, :],
                    mask=(rows[:, None] < H) & (cols[None, :] < I) & ok, other=0.0)
        acc += w.to(tl.float32)
    sc = tl.arange(0, IS)
    sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :],
                 mask=(rows[:, None] < H) & (sc[None, :] < IS) & ok, other=0.0)
    return tl.sum(tl.sum(acc, axis=1), axis=0) + tl.sum(tl.sum(sw.to(tl.float32), axis=1), axis=0)


@triton.jit(do_not_specialize=["R", "eps", "N3", "NIN", "CAPW", "per_barrier"])
def _span_kernel(
    core_ptr, x_ptr,
    wo_ptr, w3_ptr, gu_ptr, dn_ptr, sd_ptr, win_ptr, lnp_ptr, lnn_ptr,
    part_o, h1_ptr, h1_32, x2_ptr, x2_32, h2_ptr, part_r, r_ptr, idx_ptr, wt_ptr, act_ptr,
    moe_ptr, moe_32, x3_ptr, x3_32, h3_ptr, proj_ptr,
    recv_local, recv_peer, ep_ptr, err_ptr, CAPW,
    bar_ptr, per_barrier, st_ptr, end_ptr,
    R, eps, N3, NIN, stride_ge, stride_de,
    H: tl.constexpr, KO: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, TOPKP: tl.constexpr, BM: tl.constexpr,
    BN_O: tl.constexpr, BK_O: tl.constexpr, KS_O: tl.constexpr,
    BN_R: tl.constexpr, BK_R: tl.constexpr, KS_R: tl.constexpr,
    BN_I: tl.constexpr, BK_I: tl.constexpr,
    MBN: tl.constexpr, BK_UP: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr,
    STRIDED: tl.constexpr, TREE: tl.constexpr, SELMODE: tl.constexpr, PREFETCH: tl.constexpr,
    PFK: tl.constexpr, MAXIT: tl.constexpr, MAXSPIN: tl.constexpr, RING: tl.constexpr,
    NST: tl.constexpr, MAXCTA: tl.constexpr,
):
    """One GDN layer's streaming span on one rank (see the module doc), a persistent grid."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    per_launch = 7 * per_barrier
    c0 = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    launch = c0 // per_launch
    base = launch * per_launch
    slot = (launch % RING).to(tl.int64)
    srow = st_ptr + slot * NST
    if pid == 0:
        tl.store(srow, _realtime())
    pf = tl.sum(tl.zeros([2], dtype=tl.float32), axis=0)

    # P0: out_proj, fp32 K parts
    for t in range(pid, (H // BN_O) * KS_O, nprog):
        _mm_tile(core_ptr, wo_ptr, h1_ptr, part_o, R, H, H, t, K=KO, BM=BM, BN=BN_O, BK=BK_O,
                 KSPLIT=KS_O)
    _arrive(bar_ptr)
    if PREFETCH:
        pf += _pf_mm(w3_ptr, N3, pid, tl.cdiv(N3, BN_R) * KS_R, K=H, BN=BN_R, KSPLIT=KS_R, PFK=PFK)
    _wait(bar_ptr, base + per_barrier, err_ptr, MAXSPIN)
    if pid == 0:
        tl.store(srow + 1, _realtime())

    # P1: all-reduce (K-sum prologue) + residual + post-attention RMSNorm, a CTA per row
    for row in range(pid, R, nprog):
        _arrms(row, 0, KS_O, part_o, moe_32, x_ptr, x2_32, h1_ptr, h1_32, x2_ptr, h2_ptr, lnp_ptr,
               eps, recv_local, recv_peer, ep_ptr, err_ptr, CAPW, H=H, BM=BM, MAXIT=MAXIT)
    _arrive(bar_ptr)
    _wait(bar_ptr, base + 2 * per_barrier, err_ptr, MAXSPIN)
    if pid == 0:
        tl.store(srow + 2, _realtime())

    # P2: router | shared gate | shared up | shared-expert gate, fp32 K parts
    for t in range(pid, tl.cdiv(N3, BN_R) * KS_R, nprog):
        _mm_tile(h2_ptr, w3_ptr, r_ptr, part_r, R, N3, N3, t, K=H, BM=BM, BN=BN_R, BK=BK_R,
                 KSPLIT=KS_R)
    _arrive(bar_ptr)
    _wait(bar_ptr, base + 3 * per_barrier, err_ptr, MAXSPIN)
    if pid == 0:
        tl.store(srow + 3, _realtime())

    # P3: K-sum to BF16 + the router top-k per row
    nmain = tl.cdiv(R * N3, 256)
    for t in range(pid, nmain + R, nprog):
        _ksum_topk(t, nmain, part_r, r_ptr, idx_ptr, wt_ptr, R, N3, KSPLIT=KS_R, BM=BM, BLOCK=256,
                   E=E, TOPK=TOPK, TOPKP=TOPKP, STRIDED=STRIDED, TREE=TREE, SELMODE=SELMODE)
    _arrive(bar_ptr)
    _wait(bar_ptr, base + 4 * per_barrier, err_ptr, MAXSPIN)
    if pid == 0:
        tl.store(srow + 4, _realtime())

    # P4: routed gate_up (tile-outer order)
    P = R * TOPK
    for t in range(pid, P * (I // MBN), nprog):
        _up_item(t, P, h2_ptr, idx_ptr, gu_ptr, act_ptr, stride_ge, H=H, I=I, TOPK=TOPK, BN=MBN,
                 BK=BK_UP)
    _arrive(bar_ptr)
    if PREFETCH:
        pf += _pf_down(pid, R, idx_ptr, dn_ptr, sd_ptr, stride_de, H=H, I=I, IS=IS, TOPK=TOPK, BN=MBN)
    _wait(bar_ptr, base + 5 * per_barrier, err_ptr, MAXSPIN)
    if pid == 0:
        tl.store(srow + 5, _realtime())

    # P5: routed down + combine + shared expert + its gate (no residual)
    for t in range(pid, R * (H // MBN), nprog):
        _down_item(t, R, act_ptr, r_ptr, idx_ptr, wt_ptr, dn_ptr, sd_ptr, moe_ptr, stride_de, N3,
                   H=H, I=I, IS=IS, E=E, TOPK=TOPK, BN=MBN, BKI=BKI, BKS=BKS)
    _arrive(bar_ptr)
    if PREFETCH:
        pf += _pf_mm(win_ptr, NIN, pid, tl.cdiv(NIN, BN_I), K=H, BN=BN_I, KSPLIT=1, PFK=PFK)
    _wait(bar_ptr, base + 6 * per_barrier, err_ptr, MAXSPIN)
    if pid == 0:
        tl.store(srow + 6, _realtime())

    # P6: all-reduce + residual, then the next layer's input RMSNorm, a CTA per row
    for row in range(pid, R, nprog):
        _arrms(row, 1, 1, part_o, moe_32, x2_ptr, x2_32, x3_ptr, x3_32, x3_ptr, h3_ptr, lnn_ptr,
               eps, recv_local, recv_peer, ep_ptr, err_ptr, CAPW, H=H, BM=BM, MAXIT=MAXIT)
    _arrive(bar_ptr)
    _wait(bar_ptr, base + 7 * per_barrier, err_ptr, MAXSPIN)
    if pid == 0:
        tl.store(srow + 7, _realtime())

    # P7: the next layer's in_proj
    for t in range(pid, tl.cdiv(NIN, BN_I), nprog):
        _mm_tile(h3_ptr, win_ptr, proj_ptr, proj_ptr, R, NIN, NIN, t, K=H, BM=BM, BN=BN_I,
                 BK=BK_I, KSPLIT=1)
    _vm_drain()
    tl.debug_barrier()
    if pid < MAXCTA:
        tl.store(end_ptr + slot * MAXCTA + pid, _realtime())
    if pf != pf:
        tl.atomic_add(err_ptr + 2, 1)


# ============================================================================ host


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def same(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int]:
    x, y = bits(a), bits(b)
    if x.shape != y.shape:
        return False, -1
    diff = int((x != y).sum())
    return diff == 0, diff


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def asm_info(k) -> dict:
    info = {}
    for attr in ("n_regs", "n_spills"):
        info[attr] = getattr(k, attr, None)
    md = getattr(k, "metadata", None)
    info["shared"] = getattr(md, "shared", None) if md is not None else None
    try:
        asm = k.asm.get("amdgcn", "") if hasattr(k, "asm") else ""
    except Exception:  # noqa: BLE001
        asm = ""
    keys = {}
    for name, pat in (("vgpr_count", r"\.vgpr_count:\s*(\d+)"), ("agpr_count", r"\.agpr_count:\s*(\d+)"),
                      ("sgpr_count", r"\.sgpr_count:\s*(\d+)"),
                      ("vgpr_spill_count", r"\.vgpr_spill_count:\s*(\d+)"),
                      ("sgpr_spill_count", r"\.sgpr_spill_count:\s*(\d+)"),
                      ("group_segment_fixed_size", r"\.group_segment_fixed_size:\s*(\d+)"),
                      ("private_segment_fixed_size", r"\.private_segment_fixed_size:\s*(\d+)"),
                      ("Occupancy", r";\s*Occupancy:\s*(\d+)"), ("NumVgprs", r";\s*NumVgprs:\s*(\d+)"),
                      ("NumAgprs", r";\s*NumAgprs:\s*(\d+)"), ("TotalNumVgprs", r";\s*TotalNumVgprs:\s*(\d+)"),
                      ("ScratchSize", r";\s*ScratchSize:\s*(\d+)"), ("LDSByteSize", r";\s*LDSByteSize:\s*(\d+)")):
        mm = re.findall(pat, asm)
        if mm:
            keys[name] = max(int(v) for v in mm)
    info["asm"] = keys
    return info


def occupancy(info: dict) -> int:
    """CTAs (4 waves, one per SIMD) per CU on gfx90a from VGPR+AGPR and LDS."""
    a = info["asm"]
    total = a.get("TotalNumVgprs")
    if total is None:
        v = a.get("vgpr_count") or a.get("NumVgprs") or info.get("n_regs") or 128
        ag = a.get("agpr_count") or a.get("NumAgprs") or 0
        total = ((v + 3) // 4) * 4 + ag
    alloc = max(8, ((total + 7) // 8) * 8)
    waves = min(8, 512 // alloc)
    lds = a.get("group_segment_fixed_size") or info.get("shared") or 0
    by_lds = 8 if not lds else 65536 // lds
    occ = a.get("Occupancy")
    res = min(waves, by_lds)
    if occ:
        res = min(res, occ)
    return max(res, 0)


def main() -> None:
    args = ARGS
    from engine.fast import fused as FZ
    from engine.fast import kernels as KN
    from engine.fast import mfma as MF
    from engine.fast import model as MD
    from engine.fast import tp as TP
    from engine.generator import CHECKPOINT, Generator

    L = args.layer
    rows_list = [int(r) for r in args.rows.split(",")]
    verdict = {"hang": False, "bits": True, "notes": []}

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    # ------------------------------------------------------------------ load + real inputs
    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    ranks = getattr(fast, "ranks", None)
    if not ranks:
        raise SystemExit("not a TP2 replica")
    devs = [m.embed.device for m in ranks]
    fds = [m.fused for m in ranks]
    out(f"RESULT load_s={time.perf_counter() - started:.0f} graphs={fast.use_graphs} "
        f"graph_error={fast.graph_error} chain={MF.CHAIN} ar_ksum={TP.AR_KSUM} "
        f"topk_epi={MF.TOPK_EPI} arrms={TP.GLUE_ARRMS} nozero={FZ.GLUE_NOZERO} "
        f"triton={triton.__version__} torch={torch.__version__}")
    assert fds[0]._is_gdn(L) and fds[0]._is_gdn(L + 1), "layer L and L+1 must be GDN layers"
    assert MF.CHAIN and TP.AR_KSUM and MF.TOPK_EPI and not TP.GLUE_ARRMS and FZ.GLUE_NOZERO
    cus = [fd._cus(d) for fd, d in zip(fds, devs)]
    out(f"RESULT devices={devs} cus={cus} layer={L} rows={rows_list}")

    tok = g.tokenizer
    ids: list[int] = []
    i = 0
    while len(ids) < 4 * 9973 + args.prompt_len:
        for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i}"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(prompt) - 1
        pos = 0
        while pos < body:
            end = min((pos // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[pos:end], pos, prompt[end])
            pos = end
        return fast.prefill(slot, prompt[body:], body)

    plen = args.prompt_len
    slots = [0, 1]
    lasts = [int(torch.argmax(prefill(s, ids[9973 * s: 9973 * s + plen]))) for s in slots]

    # Real inputs of layer L for B slots (rows = 4 B): an eager chained verify step with the
    # MTP drafts of a captured one, recording layer L's residual and out_proj input per rank.
    rec: dict[int, list[dict]] = {}
    orig_layer = MD.FastModel._fused_layer
    orig_gemv = FZ.FusedDecode._gemv
    target: dict = {}

    def fl(self, i, x, *a, **kw):
        if i == L and self.tp is not None and "rows" in target:
            rec[target["rows"]][self.rank]["x"] = x.detach().clone()
        return orig_layer(self, i, x, *a, **kw)

    def gv(self, x, w, *a, **kw):
        if "rows" in target and w.data_ptr() == self.fast.layers[L].linear_attn.out_proj.weight.data_ptr():
            rec[target["rows"]][self.fast.rank]["core"] = x.detach().clone()
        return orig_gemv(self, x, w, *a, **kw)

    for R in rows_list:
        b = R // 4
        _, _, _, nxt = fast.verify_chain(lasts[:b], [[0] * 3] * b, slots[:b], [plen] * b)  # captures
        drafts = [list(d)[:3] + [0] * (3 - len(list(d)[:3])) for d in nxt[:b]]
        rec[R] = [{}, {}]
        saved = [m.use_graphs for m in ranks]
        for m in ranks:
            m.use_graphs = False
        MD.FastModel._fused_layer = fl
        FZ.FusedDecode._gemv = gv
        target["rows"] = R
        try:
            fast.verify_chain(lasts[:b], drafts, slots[:b], [plen] * b)
            sync()
        finally:
            target.clear()
            MD.FastModel._fused_layer = orig_layer
            FZ.FusedDecode._gemv = orig_gemv
            for m, u in zip(ranks, saved):
                m.use_graphs = u
        for r in (0, 1):
            assert rec[R][r]["x"].shape == (R, 2048) and rec[R][r]["core"].shape[0] == R, \
                (R, r, {k: tuple(v.shape) for k, v in rec[R][r].items()})
        out(f"RESULT inputs R={R} drafts={drafts} x_equal_ranks={same(rec[R][0]['x'], rec[R][1]['x'])[0]} "
            f"core_shape={tuple(rec[R][0]['core'].shape)}")

    # ------------------------------------------------------------------ weights per rank
    W = []
    for r, m in enumerate(ranks):
        fd = fds[r]
        lay, nxt_l = fd._layer(L), fd._layer(L + 1)
        mlp = lay.mlp
        w = dict(
            wo=lay.linear_attn.out_proj.weight, w3=fd.w_moe[L], gu=mlp.experts.gate_up_proj,
            dn=mlp.experts.down_proj, sd=mlp.shared_expert.down_proj.weight, win=fd.w_in[L + 1],
            lnp=lay.post_attention_layernorm.weight, lnn=nxt_l.input_layernorm.weight,
        )
        for k, v in w.items():
            assert v.is_contiguous(), k
        W.append(w)
    w0 = W[0]
    H = w0["wo"].shape[0]
    KO = w0["wo"].shape[1]
    N3 = w0["w3"].shape[0]
    NIN = w0["win"].shape[0]
    E, TOPKN = fast.ranks[0].fused._layer(L).mlp.experts.num_experts, ranks[0].top_k
    I = w0["gu"].shape[1] // 2
    IS = w0["sd"].shape[1]
    eps = float(ranks[0].eps)
    t_o, t_r, t_i = MF.tiles_for(H, KO), MF.tiles_for(N3, H), MF.tiles_for(NIN, H)
    out(f"RESULT dims H={H} KO={KO} N3={N3} NIN={NIN} E={E} topk={TOPKN} I={I} IS={IS} eps={eps} "
        f"tiles out_proj={t_o} router={t_r} in_proj={t_i} gu={tuple(w0['gu'].shape)} "
        f"dn={tuple(w0['dn'].shape)}")
    assert H == 2048 and E == 256 and TOPKN == 8 and t_i[3] == 1 and t_o[3] > 1 and t_r[3] > 1
    assert all(t[2] == 4 for t in (t_o, t_r, t_i))

    # ------------------------------------------------------------------ A. top-k calibration
    mode = {"STRIDED": 1, "TREE": 0, "SELMODE": 1}
    try:
        dev = devs[0]
        with torch.cuda.device(dev):
            nchunk = 128
            ks, bm = t_r[3], MF.BM
            gen = torch.Generator(device="cpu").manual_seed(57)
            parts = (torch.randn((nchunk, ks * bm * N3), generator=gen) * 0.8).to(dev)
            refs = []
            for c in range(nchunk):
                pc = parts[c]
                den = torch.empty(bm, dtype=torch.float32, device=dev)
                sw = torch.empty_like(den)
                idx = torch.empty((bm, 8), dtype=torch.int32, device=dev)
                wt = torch.empty((bm, 8), dtype=torch.float32, device=dev)
                _cal_ref_kernel[(bm,)](pc, den, sw, idx, wt, N3, KSPLIT=ks, BM=bm, E=E, TOPK=8,
                                       TOPKP=8, num_warps=1)
                ro = torch.empty((bm, N3), dtype=torch.bfloat16, device=dev)
                ridx = torch.empty((bm, 8), dtype=torch.int32, device=dev)
                rwt = torch.empty((bm, 8), dtype=torch.float32, device=dev)
                MF._ksum_router_kernel[(triton.cdiv(bm * N3, 256) + bm,)](
                    pc, ro, ridx, rwt, bm, N3, N3, KSPLIT=ks, BM=bm, BLOCK=256, E=E, TOPK=8,
                    TOPKP=8, num_warps=1)
                refs.append((den, sw, idx, wt, ridx, rwt))
            sync()
            copy_ok = all(same(a[2], a[4])[0] and same(a[3], a[5])[0] for a in refs)
            out(f"RESULT calib ref_copy_vs_engine_topk bitwise={copy_ok} chunks={nchunk} rows={nchunk * bm}")
            def run_mode(st, tr, sm):
                dmis = smis = imis = 0
                for c in range(nchunk):
                    pc = parts[c]
                    den = torch.empty(bm, dtype=torch.float32, device=dev)
                    sw = torch.empty_like(den)
                    idx = torch.empty((bm, 8), dtype=torch.int32, device=dev)
                    wt = torch.empty((bm, 8), dtype=torch.float32, device=dev)
                    _cal_emul_kernel[(bm,)](pc, den, sw, idx, wt, N3, KSPLIT=ks, BM=bm, E=E,
                                            TOPK=8, TOPKP=8, STRIDED=st, TREE=tr, SELMODE=sm,
                                            num_warps=4)
                    r0 = refs[c]
                    dmis += same(den, r0[0])[1]
                    smis += same(sw, r0[1])[1]
                    imis += same(idx, r0[4])[1] + same(wt, r0[5])[1]
                out(f"RESULT calib mode strided={st} tree={tr} selmode={sm} den_mismatch={dmis} "
                    f"selsum_mismatch={smis} topk_vs_engine_mismatch={imis}")
                return dmis, smis, imis

            # stage 1: the [256] softmax sum (sel sum fixed); stage 2: the [8] sum on an exact den
            dens = []
            for st, tr in [(0, -1)] + [(st_, tr_) for st_ in (1, 0) for tr_ in (0, 1, 2)]:
                if run_mode(st, tr, 1)[0] == 0:
                    dens.append((st, tr))
            sels = []
            if dens:
                for sm in (0, 1, 2, 3):
                    if run_mode(dens[0][0], dens[0][1], sm)[1] == 0:
                        sels.append(sm)
            if dens:
                mode = {"STRIDED": dens[0][0], "TREE": dens[0][1], "SELMODE": sels[0] if sels else 1}
            out(f"RESULT calib chosen {mode} exact_den_modes={dens} exact_sel_modes={sels}")
    except Exception as exc:  # noqa: BLE001
        out(f"RESULT calib error: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # ------------------------------------------------------------------ helpers: graphs, timing
    def capture(r: int, fn, reps: int = 1):
        dev = devs[r]
        with torch.cuda.device(dev):
            s = torch.cuda.Stream(dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            torch.cuda.synchronize(dev)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s, capture_error_mode="thread_local"):
                res = None
                for _ in range(reps):
                    res = fn()
            torch.cuda.synchronize(dev)
        return gr, res

    def replay(gs) -> None:
        for r, gr in enumerate(gs):
            with torch.cuda.device(devs[r]):
                gr.replay()

    def time_pair(gs, iters: int, warm: int = 5) -> dict:
        for _ in range(warm):
            replay(gs)
        sync()
        evs = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(iters)] for _ in gs]
        t0 = time.perf_counter()
        for i in range(iters):
            for r, gr in enumerate(gs):
                with torch.cuda.device(devs[r]):
                    evs[r][i][0].record()
                    gr.replay()
                    evs[r][i][1].record()
        sync()
        wall = (time.perf_counter() - t0) * 1e3 / iters
        per = [med([a.elapsed_time(b) for a, b in evs[r]]) for r in range(len(gs))]
        return {"rank_ms": per, "max_ms": max(per), "wall_ms": wall}

    def errs() -> list[int]:
        return [int(fast.group.ll.err[r].item()) for r in (0, 1)]

    # ------------------------------------------------------------------ B. today's span
    ref = {}
    rec_topk: dict = {}
    orig_topk = MF.gemv_topk

    def topk_rec(x, w, idx, wt, experts, top_k):
        o = orig_topk(x, w, idx, wt, experts, top_k)
        rec_topk[x.device.index] = (o, idx, wt)
        return o

    def ref_span(m, core, x):
        fd = m.fused
        lay, nl = fd._layer(L), fd._layer(L + 1)
        kp = fd._gemv(core, lay.linear_attn.out_proj.weight, defer=True)
        h = m.tp.allreduce(m.rank, kp)
        x2, h2 = KN.add_rms(x, h, lay.post_attention_layernorm.weight, m.eps)
        moe = fd._moe(L, None, h2)
        x3 = m.tp.allreduce(m.rank, moe, x2)
        _, h3 = KN.add_rms(x3, None, nl.input_layernorm.weight, m.eps)
        proj = fd._gemv(h3, fd.w_in[L + 1])
        r_, idx, wt = rec_topk[core.device.index]
        return dict(h1=h, x2=x2, h2=h2, r=r_, idx=idx, wt=wt, moe=moe, x3=x3, h3=h3, proj=proj)

    MF.gemv_topk = topk_rec
    ceiling = {}
    try:
        for R in rows_list:
            ins = [(rec[R][r]["core"].contiguous(), rec[R][r]["x"].contiguous()) for r in (0, 1)]
            # eager warm-up of exactly these launches (eager all-reduce path, both threads)
            fast._both(lambda m: ref_span(m, *ins[m.rank]))
            sync()
            gs, outs = [], []
            for r in (0, 1):
                gr, o = capture(r, lambda r=r: ref_span(ranks[r], *ins[r]))
                gs.append(gr)
                outs.append(o)
            replay(gs)
            sync()
            ok = all(same(outs[0][k], outs[1][k])[0] for k in ("x3", "h3", "idx", "wt", "r", "x2", "h2"))
            out(f"RESULT ref R={R} ranks_agree={ok} ar_timeouts={errs()} "
                f"unique_experts={len(set(outs[0]['idx'].flatten().tolist()))}")
            glong = [capture(r, lambda r=r: ref_span(ranks[r], *ins[r]), reps=args.long)[0] for r in (0, 1)]
            t1 = time_pair(gs, args.iters)
            tl_ = time_pair(glong, max(args.iters // args.long, 20))
            per_span_long = tl_["max_ms"] / args.long
            out(f"RESULT ref_graph R={R} one_span_graph rank_ms={['%.4f' % v for v in t1['rank_ms']]} "
                f"wall_ms={t1['wall_ms']:.4f} | {args.long}_spans_graph per_span_ms={per_span_long:.4f} "
                f"rank_ms={['%.4f' % (v / args.long) for v in tl_['rank_ms']]}")
            ref[R] = dict(gs=gs, outs=outs, ins=ins, t1=t1["max_ms"], tlong=per_span_long)
    except Exception as exc:  # noqa: BLE001
        out(f"RESULT ref error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        MF.gemv_topk = orig_topk

    # node cost: graphs of empty kernels
    node = {}
    try:
        with torch.cuda.device(devs[0]):
            dummy = torch.zeros(1, dtype=torch.int32, device=devs[0])
            for grid in (1, 4 * cus[0]):
                for n in (10, 100):
                    _empty_kernel[(grid,)](dummy)
                    gr, _ = capture(0, lambda grid=grid: _empty_kernel[(grid,)](dummy), reps=n)
                    tt = time_pair([gr], 50)
                    node[(grid, n)] = tt["rank_ms"][0] * 1e3 / n
                    out(f"RESULT node empty_graph grid={grid} nodes={n} us_per_node={node[(grid, n)]:.2f} "
                        f"graph_us={tt['rank_ms'][0] * 1e3:.1f}")
    except Exception as exc:  # noqa: BLE001
        out(f"RESULT node error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    node_us = node.get((4 * cus[0], 100), node.get((1, 100), 1.5))

    # streaming rate (one layer's whole expert gate_up, 512 MiB) and a large MFMA GEMV
    r_ss = 1.3
    try:
        with torch.cuda.device(devs[0]):
            x = W[0]["gu"].view(-1)
            n = x.numel() // 2048 * 2048
            o = torch.empty(8 * cus[0], dtype=torch.float32, device=devs[0])
            for _ in range(3):
                _stream_kernel[(8 * cus[0],)](x, o, n, BLOCK=2048, num_warps=4)
            sync()
            a, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(10):
                _stream_kernel[(8 * cus[0],)](x, o, n, BLOCK=2048, num_warps=4)
            b_.record()
            sync()
            ms = a.elapsed_time(b_) / 10
            r_ss = n * 2 / ms / 1e9
            out(f"RESULT stream_read bytes={n * 2} ms={ms:.3f} TBps={r_ss:.3f}")
            lm = ranks[0].lm_head
            for R in rows_list:
                xx = torch.randn((R, lm.shape[1]), device=devs[0]).to(lm.dtype)
                for _ in range(3):
                    MF.gemv(xx, lm)
                sync()
                a.record()
                for _ in range(10):
                    MF.gemv(xx, lm)
                b_.record()
                sync()
                ms = a.elapsed_time(b_) / 10
                out(f"RESULT lm_head_mfma_gemv R={R} bytes={lm.numel() * 2} ms={ms:.3f} "
                    f"TBps={lm.numel() * 2 / ms / 1e9:.3f}")
    except Exception as exc:  # noqa: BLE001
        out(f"RESULT stream error: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # each of today's ten kernels alone: a graph of --long repeats on both ranks
    def unique_experts(idx) -> int:
        return len(set(idx.flatten().tolist()))

    def kernel_bytes(R: int, uexp: int) -> dict:
        return {
            "out_proj": H * KO * 2, "ar1": 0, "rms1": 0, "router": N3 * H * 2, "ksum_topk": 0,
            "up": uexp * 2 * I * H * 2, "down": uexp * H * I * 2 + H * IS * 2, "ar2": 0, "rms2": 0,
            "in_proj": NIN * H * 2,
        }

    standalone: dict = {}
    for R in rows_list:
        if R not in ref:
            continue
        try:
            o = ref[R]["outs"]
            uexp = unique_experts(o[0]["idx"])
            kb = kernel_bytes(R, uexp)
            prep = []
            for r in (0, 1):
                m, fd, w = ranks[r], fds[r], W[r]
                core, x = ref[R]["ins"][r]
                with torch.cuda.device(devs[r]):
                    kp_o = MF.gemv(core, w["wo"], None, defer=True)
                    kp_r = MF.gemv(o[r]["h2"], w["w3"], None, defer=True)
                    act = torch.empty((R * TOPKN, I), dtype=torch.bfloat16, device=devs[r])
                    items = R * TOPKN * triton.cdiv(I, 8)
                    FZ._moe_up_kernel[(items,)](
                        o[r]["h2"], o[r]["idx"], w["gu"], act, R, w["gu"].stride(0), o[r]["r"], o[r]["wt"],
                        N3, H=H, I=I, TOPK=TOPKN, BN=8, BK=min(1024, triton.next_power_of_2(H)),
                        ORDER=1, E=E, TOPKP=8, FUSE_TOPK=False, num_warps=4)
                prep.append(dict(kp_o=kp_o, kp_r=kp_r, act=act))
            sync()

            def launchers(r):
                m, w, oo, pp = ranks[r], W[r], o[r], prep[r]
                core, x = ref[R]["ins"][r]
                cu = cus[r]

                def k_out():
                    return MF.gemv(core, w["wo"], None, defer=True)

                def k_ar1():
                    kp = pp["kp_o"]
                    return m.tp.allreduce(m.rank, MF.KParts(kp.part, kp.out, kp.ksplit))

                def k_rms1():
                    return KN.add_rms(x, oo["h1"], w["lnp"], eps)

                def k_router():
                    return MF.gemv(oo["h2"], w["w3"], None, defer=True)

                def k_ksum():
                    rr = torch.empty((R, N3), dtype=torch.bfloat16, device=devs[r])
                    ii = torch.empty((R, TOPKN), dtype=torch.int32, device=devs[r])
                    ww = torch.empty((R, TOPKN), dtype=torch.float32, device=devs[r])
                    MF._ksum_router_kernel[(triton.cdiv(R * N3, 256) + R,)](
                        pp["kp_r"].part, rr, ii, ww, R, N3, N3, KSPLIT=pp["kp_r"].ksplit, BM=MF.BM,
                        BLOCK=256, E=E, TOPK=TOPKN, TOPKP=8, num_warps=1)
                    return rr, ii, ww

                def k_up():
                    a = torch.empty((R * TOPKN, I), dtype=torch.bfloat16, device=devs[r])
                    FZ._moe_up_kernel[(R * TOPKN * triton.cdiv(I, 8),)](
                        oo["h2"], oo["idx"], w["gu"], a, R, w["gu"].stride(0), oo["r"], oo["wt"], N3,
                        H=H, I=I, TOPK=TOPKN, BN=8, BK=min(1024, triton.next_power_of_2(H)), ORDER=1,
                        E=E, TOPKP=8, FUSE_TOPK=False, num_warps=4)
                    return a

                def k_down():
                    mo = torch.empty((R, H), dtype=torch.bfloat16, device=devs[r])
                    items = R * triton.cdiv(H, 8)
                    FZ._moe_down_kernel[(min(items, FZ.MOE_GRID * cu),)](
                        pp["act"], oo["r"], oo["idx"], oo["wt"], w["dn"], w["sd"], mo, mo, R,
                        w["dn"].stride(0), N3, H=H, I=I, IS=IS, E=E, TOPK=TOPKN, BN=8,
                        BKI=min(1024, triton.next_power_of_2(I)), BKS=min(1024, triton.next_power_of_2(IS)),
                        ORDER=1, HAS_RES=False, num_warps=4)
                    return mo

                def k_ar2():
                    return m.tp.allreduce(m.rank, oo["moe"], oo["x2"])

                def k_rms2():
                    return KN.add_rms(oo["x3"], None, w["lnn"], eps)

                def k_in():
                    return MF.gemv(oo["h3"], w["win"])

                return {"out_proj": k_out, "ar1": k_ar1, "rms1": k_rms1, "router": k_router,
                        "ksum_topk": k_ksum, "up": k_up, "down": k_down, "ar2": k_ar2,
                        "rms2": k_rms2, "in_proj": k_in}

            names = list(launchers(0).keys())
            tot = 0.0
            for name in names:
                gs = []
                res = []
                for r in (0, 1):
                    gr, rr_ = capture(r, launchers(r)[name], reps=args.long)
                    gs.append(gr)
                    res.append(rr_)
                tt = time_pair(gs, max(args.iters // args.long, 20))
                us = tt["max_ms"] * 1e3 / args.long
                chk = ""
                if name == "down":
                    chk = f" bitwise_vs_span={same(res[0], o[0]['moe'])[0] and same(res[1], o[1]['moe'])[0]}"
                elif name == "in_proj":
                    chk = f" bitwise_vs_span={same(res[0], o[0]['proj'])[0]}"
                elif name == "ksum_topk":
                    chk = f" bitwise_vs_span={same(res[0][1], o[0]['idx'])[0] and same(res[0][2], o[0]['wt'])[0]}"
                b = kb[name]
                fl_us = b / (r_ss * 1e12) * 1e6 if b else 0.0
                standalone[(R, name)] = dict(us=us, bytes=b)
                tot += us
                out(f"RESULT kernel R={R} {name:10s} us={us:7.2f} rank_us={['%.2f' % (v * 1e3 / args.long) for v in tt['rank_ms']]} "
                    f"bytes={b} TBps={(b / us / 1e6) if b else 0:.3f} floor_us={fl_us:6.2f} "
                    f"excess_us={us - fl_us:6.2f} fill_drain_us={max(us - fl_us - node_us, 0):6.2f}{chk}")
            nk = len(names)
            floors = sum(kb[n] for n in names) / (r_ss * 1e12) * 1e6
            out(f"RESULT ceiling R={R} span_graph_us={ref[R]['t1'] * 1e3:.1f} "
                f"span_in_long_graph_us={ref[R]['tlong'] * 1e3:.1f} sum_kernels_alone_us={tot:.1f} "
                f"stream_floor_us={floors:.1f} boundary_us={ref[R]['tlong'] * 1e3 - floors:.1f} "
                f"node_launch_us={nk * node_us:.1f} ({nk} x {node_us:.2f}) "
                f"fill_drain_us={ref[R]['tlong'] * 1e3 - floors - nk * node_us:.1f} unique_experts={uexp} "
                f"graph_launch_us={(ref[R]['t1'] - ref[R]['tlong']) * 1e3:.1f}")
            ceiling[R] = dict(floors=floors, tot=tot)
        except Exception as exc:  # noqa: BLE001
            out(f"RESULT kernels R={R} error: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------ C. fused span kernel
    ll = fast.group.ll
    CAPW = 2 * 64 * (H // 2)

    # barrier microbenchmark
    def calib(dev, nprog: int) -> int:
        c = torch.zeros(1, dtype=torch.int32, device=dev)
        _calib_kernel[(nprog,)](c, num_warps=4)
        torch.cuda.synchronize(dev)
        total = int(c.item())
        assert total % nprog == 0, (total, nprog)
        return total

    bar_us = {}
    for gpc in (1, 2, 3, 4):
        try:
            dev = devs[0]
            with torch.cuda.device(dev):
                nprog = gpc * cus[0]
                pb = calib(dev, nprog)
                err = torch.zeros(4, dtype=torch.int32, device=dev)
                bad = torch.zeros(1, dtype=torch.int32, device=dev)
                buf = torch.zeros(2 * nprog, dtype=torch.int32, device=dev)
                bar = torch.zeros(1, dtype=torch.int32, device=dev)

                def go(iters):
                    bar.zero_()
                    _bar_bench_kernel[(nprog,)](bar, err, buf, bad, iters, pb, MAXSPIN=MAXSPIN,
                                                num_warps=4)

                go(10)
                torch.cuda.synchronize(dev)
                if int(err[1].item()):
                    out(f"RESULT barrier grid={nprog} per_cu={gpc} not_resident timeouts={int(err[1].item())}")
                    continue
                a, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ts = {}
                for iters in (0, 2000):
                    go(iters)
                    torch.cuda.synchronize(dev)
                    a.record()
                    for _ in range(5):
                        go(iters)
                    b_.record()
                    torch.cuda.synchronize(dev)
                    ts[iters] = a.elapsed_time(b_) / 5
                bar_us[gpc] = (ts[2000] - ts[0]) * 1e3 / 2000
                out(f"RESULT barrier grid={nprog} per_cu={gpc} calib={pb} us_per_barrier={bar_us[gpc]:.3f} "
                    f"timeouts={int(err[1].item())} neighbour_mismatches={int(bad.item())}")
        except Exception as exc:  # noqa: BLE001
            out(f"RESULT barrier per_cu={gpc} error: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    LLBUF: dict = {}

    class Span:
        """One rank pair's buffers and launch for one (R, variant, grid)."""

        def __init__(self, R: int, stages: int, pf: int, gpc: int):
            self.R, self.stages, self.pf, self.gpc = R, stages, pf, gpc
            self.st = []
            self.bufs = []
            self.loop_bufs = []
            for r in (0, 1):
                d = devs[r]
                with torch.cuda.device(d):
                    bf = torch.bfloat16
                    b = dict(
                        part_o=torch.zeros(t_o[3] * MF.BM * H, dtype=torch.float32, device=d),
                        h1=torch.zeros((R, H), dtype=bf, device=d), x2=torch.zeros((R, H), dtype=bf, device=d),
                        h2=torch.zeros((R, H), dtype=bf, device=d),
                        part_r=torch.zeros(t_r[3] * MF.BM * N3, dtype=torch.float32, device=d),
                        r=torch.zeros((R, N3), dtype=bf, device=d),
                        idx=torch.zeros((R, TOPKN), dtype=torch.int32, device=d),
                        wt=torch.zeros((R, TOPKN), dtype=torch.float32, device=d),
                        act=torch.zeros((R * TOPKN, I), dtype=bf, device=d),
                        moe=torch.zeros((R, H), dtype=bf, device=d), x3=torch.zeros((R, H), dtype=bf, device=d),
                        h3=torch.zeros((R, H), dtype=bf, device=d), proj=torch.zeros((R, NIN), dtype=bf, device=d),
                        err=torch.zeros(4, dtype=torch.int32, device=d),
                        bar=torch.zeros(1, dtype=torch.int32, device=d),
                        st=torch.zeros((RING, NST), dtype=torch.int64, device=d),
                        end=torch.zeros((RING, MAXCTA), dtype=torch.int64, device=d),
                    )
                    shared = LLBUF.get((R, stages, pf, r))
                    if shared is None:
                        shared = dict(recv=ll._fine(d, 2 * CAPW * 8),
                                      ep=torch.zeros(128, dtype=torch.int32, device=d),
                                      lrecv=ll._fine(d, 2 * CAPW * 8),
                                      lep=torch.zeros(128, dtype=torch.int32, device=d))
                        LLBUF[(R, stages, pf, r)] = shared
                    b["recv"], b["ep"] = shared["recv"], shared["ep"]
                    lb = dict(ep=shared["lep"], bar=torch.zeros(1, dtype=torch.int32, device=d),
                              recv=shared["lrecv"])
                self.bufs.append(b)
                self.loop_bufs.append(lb)
            self.nprog = [gpc * cus[r] for r in (0, 1)]
            self.pb = [calib(devs[r], self.nprog[r]) for r in (0, 1)]
            self.kernel = None

        def launch(self, r: int, loop: bool = False, core=None, x=None):
            b, w = self.bufs[r], W[r]
            lb = self.loop_bufs[r]
            if loop:
                local = peer = lb["recv"]
                ep, bar = lb["ep"], lb["bar"]
            else:
                local, peer = b["recv"], self.bufs[1 - r]["recv"]
                ep, bar = b["ep"], b["bar"]
            core = ref[self.R]["ins"][r][0] if core is None else core
            x = ref[self.R]["ins"][r][1] if x is None else x
            with torch.cuda.device(devs[r]):
                k = _span_kernel[(self.nprog[r],)](
                    core, x, w["wo"], w["w3"], w["gu"], w["dn"], w["sd"], w["win"], w["lnp"], w["lnn"],
                    b["part_o"], b["h1"], b["h1"].view(torch.int32), b["x2"], b["x2"].view(torch.int32),
                    b["h2"], b["part_r"], b["r"], b["idx"], b["wt"], b["act"], b["moe"],
                    b["moe"].view(torch.int32), b["x3"], b["x3"].view(torch.int32), b["h3"], b["proj"],
                    local, peer, ep, b["err"], CAPW, bar, self.pb[r], b["st"], b["end"],
                    self.R, eps, N3, NIN, w["gu"].stride(0), w["dn"].stride(0),
                    H=H, KO=KO, I=I, IS=IS, E=E, TOPK=TOPKN, TOPKP=8, BM=MF.BM,
                    BN_O=t_o[0], BK_O=t_o[1], KS_O=t_o[3], BN_R=t_r[0], BK_R=t_r[1], KS_R=t_r[3],
                    BN_I=t_i[0], BK_I=t_i[1], MBN=8, BK_UP=min(1024, triton.next_power_of_2(H)),
                    BKI=min(1024, triton.next_power_of_2(I)), BKS=min(1024, triton.next_power_of_2(IS)),
                    STRIDED=mode["STRIDED"], TREE=mode["TREE"], SELMODE=mode["SELMODE"],
                    PREFETCH=self.pf, PFK=PFK, MAXIT=MAXIT, MAXSPIN=MAXSPIN, RING=RING, NST=NST,
                    MAXCTA=MAXCTA, num_warps=4, num_stages=self.stages,
                )
            if k is not None and self.kernel is None:
                self.kernel = k
            return k

        def pair(self) -> None:
            for r in (0, 1):
                self.launch(r)

        def errs(self) -> list[list[int]]:
            return [self.bufs[r]["err"].tolist() for r in (0, 1)]

        def compare(self) -> tuple[bool, dict]:
            o = ref[self.R]["outs"]
            bad = {}
            for r in (0, 1):
                b = self.bufs[r]
                for k in ("h1", "x2", "h2", "r", "idx", "wt", "moe", "x3", "h3", "proj"):
                    ok_, n = same(b[k], o[r][k])
                    if not ok_:
                        bad[f"r{r}.{k}"] = n
            return not bad, bad

        def stamps(self, r: int, last: int) -> dict | None:
            """Per-phase medians over the last ``last`` launches of rank r (us)."""
            st = self.bufs[r]["st"].cpu()
            end = self.bufs[r]["end"].cpu()[:, : self.nprog[r]]
            launches = int(self.bufs[r]["bar"].item()) // (NBAR * self.pb[r])
            rows_ = [(launches - 1 - j) % RING for j in range(min(last, RING, launches))]
            if not rows_:
                return None
            ph = [[] for _ in range(NBAR + 1)]
            tot = []
            for row in rows_:
                s = st[row].tolist()
                e = int(end[row].max())
                marks = s[:NBAR + 1] + [e]
                for j in range(NBAR + 1):
                    ph[j].append((marks[j + 1] - marks[j]) / RT_HZ * 1e6)
                tot.append((e - s[0]) / RT_HZ * 1e6)
            return {"phase_us": [med(p) for p in ph], "total_us": med(tot), "n": len(rows_)}

    PHASES = ["P0 out_proj", "P1 ar1+rms1", "P2 router", "P3 ksum+topk", "P4 up", "P5 down",
              "P6 ar2+rms2", "P7 in_proj"]
    PH_STANDALONE = [("out_proj",), ("ar1", "rms1"), ("router",), ("ksum_topk",), ("up",), ("down",),
                     ("ar2", "rms2"), ("in_proj",)]
    STREAMING = {0, 2, 4, 5, 7}

    variants = [tuple(int(v) for v in s.split(":")) for s in args.variants.split(",")]
    results = {}
    occ_by_var = {}
    for R in rows_list:
        if R not in ref:
            continue
        uexp = unique_experts(ref[R]["outs"][0]["idx"])
        kb = kernel_bytes(R, uexp)
        for stages, pf in variants:
            occ = None
            for gpc in [int(v) for v in args.grids.split(",")]:
                if occ is not None and gpc > occ:
                    continue
                key = (R, stages, pf, gpc)
                try:
                    sp = Span(R, stages, pf, gpc)
                    # compile + residency: loopback launch on each rank alone (no peer)
                    for r in (0, 1):
                        t_c = time.perf_counter()
                        sp.launch(r, loop=True)
                        torch.cuda.synchronize(devs[r])
                        if (stages, pf) not in occ_by_var and sp.kernel is not None:
                            info = asm_info(sp.kernel)
                            occ_by_var[(stages, pf)] = occupancy(info)
                            out(f"RESULT span_kernel stages={stages} prefetch={pf} compile_s="
                                f"{time.perf_counter() - t_c:.1f} n_regs={info['n_regs']} "
                                f"n_spills={info['n_spills']} shared={info['shared']} asm={info['asm']} "
                                f"occupancy_ctas_per_cu={occ_by_var[(stages, pf)]}")
                    occ = occ_by_var.get((stages, pf))
                    le = sp.errs()
                    sp.bufs[0]["err"].zero_()
                    sp.bufs[1]["err"].zero_()
                    if any(e[1] for e in le):
                        out(f"RESULT span R={R} stages={stages} pf={pf} per_cu={gpc} loopback "
                            f"barrier_timeouts={[e[1] for e in le]} -> not resident, skipped")
                        continue
                    if occ is not None and gpc > occ:
                        out(f"RESULT span R={R} stages={stages} pf={pf} per_cu={gpc} > occupancy {occ}: skipped")
                        continue
                    # paired eager run, then graphs
                    sp.pair()
                    sync()
                    ok, bad = sp.compare()
                    gs = [capture(r, lambda r=r: sp.launch(r))[0] for r in (0, 1)]
                    replay(gs)
                    sync()
                    ok2, bad2 = sp.compare()
                    t1 = time_pair(gs, args.iters)
                    st0 = [sp.stamps(r, args.iters) for r in (0, 1)]
                    glong = [capture(r, lambda r=r: sp.launch(r), reps=args.long)[0] for r in (0, 1)]
                    tl_ = time_pair(glong, max(args.iters // args.long, 20))
                    per_long = tl_["max_ms"] / args.long
                    ok3, bad3 = sp.compare()
                    e = sp.errs()
                    allok = ok and ok2 and ok3
                    if not allok:
                        verdict["bits"] = False
                    if any(x[0] or x[1] for x in e):
                        verdict["hang"] = True
                    r1 = t1["max_ms"] / ref[R]["t1"]
                    rl = per_long / ref[R]["tlong"]
                    out(f"RESULT span R={R} stages={stages} pf={pf} per_cu={gpc} grid={sp.nprog} "
                        f"bitwise={allok} bad={bad or bad2 or bad3} errs(ar,bar,pf)={e} "
                        f"one_graph_ms={t1['max_ms']:.4f} (ref {ref[R]['t1']:.4f}, x{r1:.3f}) "
                        f"long_per_span_ms={per_long:.4f} (ref {ref[R]['tlong']:.4f}, x{rl:.3f}) "
                        f"rank_ms={['%.4f' % v for v in t1['rank_ms']]}")
                    for r in (0, 1):
                        s = st0[r]
                        if not s:
                            continue
                        parts = []
                        worst = None
                        for j, name in enumerate(PHASES):
                            us = s["phase_us"][j]
                            alone = sum(standalone.get((R, n), {}).get("us", float("nan"))
                                        for n in PH_STANDALONE[j])
                            b = sum(kb[n] for n in PH_STANDALONE[j])
                            rate = b / us / 1e6 if b and us > 0 else 0.0
                            arate = b / alone / 1e6 if b and alone > 0 else 0.0
                            rel = rate / arate if arate else float("nan")
                            if j in STREAMING and arate:
                                worst = rel if worst is None else min(worst, rel)
                            parts.append(f"{name}: {us:.2f}us (alone {alone:.2f}) "
                                         + (f"{rate:.3f}TBps/{arate:.3f} x{rel:.2f}" if b else ""))
                        out(f"RESULT phases R={R} stages={stages} pf={pf} per_cu={gpc} rank={r} "
                            f"n={s['n']} total_us={s['total_us']:.1f} worst_stream_rel={worst} | "
                            + " | ".join(parts))
                        results.setdefault(key, {})[f"worst{r}"] = worst
                    results.setdefault(key, {}).update(
                        dict(t1=t1["max_ms"], tlong=per_long, r1=r1, rl=rl, ok=allok, errs=e, sp=sp))
                except Exception as exc:  # noqa: BLE001
                    out(f"RESULT span R={R} stages={stages} pf={pf} per_cu={gpc} error: "
                        f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    verdict["notes"].append(f"error {key}: {type(exc).__name__}")

    # ------------------------------------------------------------------ D. stress
    best = {}
    for R in rows_list:
        cands = [(v["rl"], k) for k, v in results.items() if k[0] == R and v.get("ok")]
        if not cands:
            continue
        _, key = min(cands)
        best[R] = key
        sp = results[key]["sp"]
        try:
            for b in sp.bufs:
                b["err"].zero_()
            gs = [capture(r, lambda r=r: sp.launch(r))[0] for r in (0, 1)]
            t_s = time.perf_counter()
            done = 0
            for i in range(args.stress):
                replay(gs)
                done += 1
                if i % 500 == 499:
                    sync()
            sync()
            ok, bad = sp.compare()
            e = sp.errs()
            if not ok:
                verdict["bits"] = False
            if any(x[0] or x[1] for x in e):
                verdict["hang"] = True
            out(f"RESULT stress R={R} best={key} replays={done} s={time.perf_counter() - t_s:.1f} "
                f"bitwise={ok} bad={bad} errs(ar,bar,pf)={e} engine_ar_timeouts={errs()}")
        except Exception as exc:  # noqa: BLE001
            out(f"RESULT stress R={R} error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            verdict["hang"] = True

    # ------------------------------------------------------------------ verdict
    r4 = best.get(4)
    if r4 is None:
        out("RESULT verdict KILL? no bitwise fused variant at 4 rows (see errors above)")
    else:
        v = results[r4]
        worst = min(x for x in (v.get("worst0"), v.get("worst1"), 1e9) if x is not None)
        kill = []
        if v["rl"] > 0.88:
            kill.append(f"span x{v['rl']:.3f} > 0.88 (long graph)")
        if worst < 0.95:
            kill.append(f"a streaming phase at x{worst:.2f} < 0.95 of standalone")
        if not verdict["bits"]:
            kill.append("bits differ")
        if verdict["hang"]:
            kill.append("timeouts")
        out(f"RESULT verdict best4={r4} span_ratio_long={v['rl']:.3f} span_ratio_one_graph={v['r1']:.3f} "
            f"worst_phase_rel={worst:.2f} bitwise_all={verdict['bits']} timeouts={verdict['hang']} -> "
            f"{'KILL: ' + '; '.join(kill) if kill else 'GO'}")
    for R in rows_list:
        if R in best:
            v = results[best[R]]
            out(f"RESULT best R={R} {best[R]} one_graph_x={v['r1']:.3f} long_x={v['rl']:.3f}")
    out(f"RESULT engine_ar_timeouts={fast.timeouts()} notes={verdict['notes']}")


if __name__ == "__main__":
    main()
