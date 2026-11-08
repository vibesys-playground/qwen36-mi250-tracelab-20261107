"""Plan 23 step 0: can a persistent per-GCD decode kernel hide the step's small dependent ops?

    python tools/mega_probe.py split [--devices 0,1]     # part 1, one 2-GCD replica
    python tools/mega_probe.py proto [--device 0]        # part 2, one GCD, 5 real layers
    TRITON_INTERPRET=1 python tools/mega_probe.py selftest   # CPU: the kernel's math vs fused

Part 1 (``split``): the captured verify B=2 step (4 rows, plans 5/11/15/19 graphs, MTP head)
of the whole model on one 2-GCD replica, under torch.profiler as ``tools/gemv_bw_bench.py``
(plan 19, j100) does. Then one eager step with the launchers wrapped to record each call's
bytes: fused GEMV weights, routed experts actually picked (unique per layer) + shared down,
fp32 recurrent state read+written by the delta-rule step, KV read by paged attention, the
fp32 state copies of the verify halves, BLAS weights (MTP fc/head). Each graph kernel event is
matched to its recorded call (same issue order per device). A call that moves >= 2 MiB is
memory-bound: its excess is max(0, t - bytes / 1.3 TB/s). Everything else is a small op.
Kill (plan 23): small ops + memory-bound excess < 25% of the step's kernel time.

Part 2 (``proto``): 4 real layers at R=4 decode rows on one GCD (the first 5 checkpoint
layers are loaded: GDN 0,1,2, attention 3, GDN 4). Chains ``mixed`` = 0,1,2,3 (3 GDN + 1
attention) and ``gdn4`` = 0,1,2,4. The prototype ``_mega_kernel`` is ONE Triton launch:
grid = CTAs per CU x CUs, all phases of all 4 layers separated by software grid barriers
(atomic counter, epoch base, plan 11's barrier). Per layer:
  GDN:  [input RMSNorm, redundant per CTA] in_proj|z|b|a GEMV + conv epilogue | delta-rule
        step + gated norm (rows x heads items; idle CTAs prefetch out_proj tiles into L2)
        | out_proj GEMV |
  attn: [input RMSNorm] q|k|v GEMV | q/k norm + rotary + KV write | paged attention splits
        | split combine + gate (idle CTAs prefetch o_proj tiles) | o_proj GEMV |
  MoE:  [residual add + RMSNorm, redundant] router|shared GEMV | [top-k redundant per CTA]
        routed gate_up | routed down + shared expert + gate + residual |
Each phase body is the fused kernel's body (engine/fast/fused.py, kernels.py) with the
program id replaced by a work-item loop, so every per-row sum has the fused path's shape when
the warps and tiles are the same. One kernel has one register allocation, the worst phase's:
the delta-rule step ([128, 128] fp32 state per item) and paged attention need far more
registers than a GEMV (offline gfx90a compile: 4 warps -> 512 VGPR+AGPR, 1 wave per SIMD),
so the sweep is over (warps, GEMV outputs per tile, expert rows per tile), ``--configs``,
always one CTA per CU (a 2nd CTA per CU is tried only if the occupancy allows it). The 50-odd
pointers come from a device table (``TAB``) to keep SGPR pressure down.
Against the current fused path (``FusedDecode.layer`` x 4) captured in one HIP graph, per
config: parity (bitwise against the fused path run with the same warps and tiles; max |diff|
against the current fused path, with the fused path's row-tile variant as the yardstick),
GDN/KV state after, determinism, compile time, registers/spills/occupancy, and the time per
4-layer step with L2 prefetch of 0/2/4 tiles per idle CTA; the fused path at the config's
warps/tiles is timed too, to separate tiling from persistence. The best config then runs
2,000 graph replays with no barrier timeout.
Kill (plan 23): prototype > 0.80x the fused graph, compile > 15 min, hang or mismatch.

Prints ``RESULT`` lines. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BW = 1.3e12  # bytes/s: the practical HBM rate of one gfx90a GCD for the kill rule
BIG = 2 * 1024 * 1024  # a call that moves at least this is memory-bound


def out(msg: str) -> None:
    print(msg, flush=True)


def _pre_parse() -> None:
    """HIP_VISIBLE_DEVICES must be set before torch is imported."""
    argv = sys.argv[1:]
    if argv and argv[0] == "split":
        dev = "0,1"
        if "--devices" in argv:
            dev = argv[argv.index("--devices") + 1]
        os.environ["HIP_VISIBLE_DEVICES"] = dev
        os.environ.setdefault("ENGINE_MTP", "1")
    elif argv and argv[0] == "proto":
        dev = "0"
        if "--device" in argv:
            dev = argv[argv.index("--device") + 1]
        os.environ["HIP_VISIBLE_DEVICES"] = dev
        os.environ["ENGINE_MTP"] = "0"


_pre_parse()

import torch  # noqa: E402

from engine.fast import fused as FZ  # noqa: E402
from engine.fast import kernels as KN  # noqa: E402

triton, tl = KN.triton, KN.tl
INTERPRET = KN.INTERPRET
MAX_SPIN = tl.constexpr(300_000)


# ============================================================================================
# Device functions: barrier, and the fused kernels' bodies over work items
# ============================================================================================


@triton.jit
def _epoch_base(bar_ptr, per_launch):
    """Counter value at this launch's start (every earlier launch ran ``per_launch``)."""
    c0 = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    return (c0 // per_launch) * per_launch


@triton.jit
def _fence(fence_ptr, fence, NT: tl.constexpr):
    """fence != 0: every thread does a release atomic, so each wave's stores are done."""
    if fence != 0:
        tl.atomic_add(fence_ptr + tl.arange(0, NT), 0, sem="release", scope="gpu")


@triton.jit
def _gbar(bar_ptr, target, err_ptr, fence_ptr, fence, NT: tl.constexpr):
    """Plan 11's grid barrier (atomic arrive, atomic acquire spin; j70: 1.08 us)."""
    _fence(fence_ptr, fence, NT)
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    cnt = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    spins = 0
    while (cnt < target) & (spins < MAX_SPIN):
        cnt = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        spins += 1
    if spins >= MAX_SPIN:
        tl.atomic_add(err_ptr, 1)
    tl.debug_barrier()


@triton.jit
def _csync(fence_ptr, fence, NT: tl.constexpr):
    """CTA-local: this CTA's stores before its later loads."""
    _fence(fence_ptr, fence, NT)
    tl.debug_barrier()


@triton.jit
def _rms_rows(x_ptr, h_ptr, w_ptr, sum_ptr, out_ptr, R, eps, store_sum,
              N: tl.constexpr, BLOCK: tl.constexpr, HAS_H: tl.constexpr):
    """``kernels._add_rms_kernel`` for rows 0..R-1 (sum stored only when store_sum)."""
    cols = tl.arange(0, BLOCK)
    ok = cols < N
    for row in range(R):
        x = tl.load(x_ptr + row * N + cols, mask=ok, other=0.0, volatile=True).to(tl.float32)
        if HAS_H:
            h = tl.load(h_ptr + row * N + cols, mask=ok, other=0.0, volatile=True).to(tl.float32)
            x = (x + h).to(tl.bfloat16).to(tl.float32)
            tl.store(sum_ptr + row * N + cols, x.to(tl.bfloat16), mask=ok & (store_sum != 0))
        var = tl.sum(x * x, axis=0) / N
        w = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
        y = x * tl.rsqrt(var + eps) * (1.0 + w)
        tl.store(out_ptr + row * N + cols, y.to(tl.bfloat16), mask=ok)


@triton.jit
def _gemv_tiles(x_ptr, w_ptr, out_ptr, R, N, stride_out,
                state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, pid, G,
                K: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                CONV: tl.constexpr, KC: tl.constexpr):
    """``fused._gemv_kernel`` with program id ``pid`` of ``G``."""
    for t in range(pid, tl.cdiv(N, BN) * tl.cdiv(R, BP), G):
        rows = (t // tl.cdiv(R, BP)) * BN + tl.arange(0, BN)
        rok = rows < N
        bs = (t % tl.cdiv(R, BP)) * BP + tl.arange(0, BP)
        bok = bs < R
        acc = tl.zeros([BN, BP], dtype=tl.float32)
        for k0 in range(0, K, BK):
            cols = k0 + tl.arange(0, BK)
            cok = cols < K
            w = tl.load(w_ptr + rows[:, None].to(tl.int64) * K + cols[None, :],
                        mask=rok[:, None] & cok[None, :], other=0.0).to(tl.float32)
            x = tl.load(x_ptr + bs[:, None] * K + cols[None, :],
                        mask=bok[:, None] & cok[None, :], other=0.0, volatile=True).to(tl.float32)
            acc += tl.sum(w[:, None, :] * x[None, :, :], axis=2)
        y = acc.to(tl.bfloat16)
        omask = rok[:, None] & bok[None, :]
        if CONV:
            crow = rok & (rows < CONV_N)
            cm = crow[:, None] & bok[None, :]
            slot = tl.load(slot_ptr + bs, mask=bok, other=0).to(tl.int64)
            sbase = state_ptr + slot[None, :] * stride_slot + rows[:, None] * KC
            cacc = tl.zeros([BN, BP], dtype=tl.float32)
            for j in tl.static_range(KC - 1):
                sj = tl.load(sbase + j + 1, mask=cm, other=0.0)
                wj = tl.load(cw_ptr + rows * KC + j, mask=crow, other=0.0).to(tl.float32)
                cacc += sj.to(tl.float32) * wj[:, None]
                tl.store(sbase + j, sj, mask=cm)
            wl = tl.load(cw_ptr + rows * KC + KC - 1, mask=crow, other=0.0).to(tl.float32)
            cacc += y.to(tl.float32) * wl[:, None]
            tl.store(sbase + KC - 1, y, mask=cm)
            c = cacc.to(tl.bfloat16).to(tl.float32)
            c = (c * tl.sigmoid(c)).to(tl.bfloat16)
            y = tl.where(cm, c, y)
        tl.store(out_ptr + bs[None, :] * stride_out + rows[:, None], y, mask=omask)


@triton.jit
def _gdn_item(state_ptr, slot_ptr, in_ptr, alog_ptr, dtb_ptr, nw_ptr, out_ptr,
              stride_in, V_HEADS, stride_slot, stride_head, eps, scale, Z_OFF, B_OFF, A_OFF,
              row, head,
              RATIO: tl.constexpr, KEY_DIM: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr):
    """``fused._gdn_step_kernel`` for (row, head)."""
    k_head = head // RATIO
    slot = tl.load(slot_ptr + row).to(tl.int64)
    kk = tl.arange(0, DK)
    vv = tl.arange(0, DV)
    base_in = in_ptr + row * stride_in
    q = tl.load(base_in + k_head * DK + kk, volatile=True).to(tl.float32)
    k = tl.load(base_in + KEY_DIM + k_head * DK + kk, volatile=True).to(tl.float32)
    v = tl.load(base_in + 2 * KEY_DIM + head * DV + vv, volatile=True).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * scale
    k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
    beta = tl.sigmoid(tl.load(base_in + B_OFF + head, volatile=True).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    a = tl.load(base_in + A_OFF + head, volatile=True).to(tl.float32) + tl.load(dtb_ptr + head).to(tl.float32)
    softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
    g = -tl.exp(tl.load(alog_ptr + head).to(tl.float32)) * softplus
    sbase = state_ptr + slot * stride_slot + head * stride_head + kk[:, None] * DV + vv[None, :]
    s = tl.load(sbase) * tl.exp(g)
    kv_mem = tl.sum(s * k[:, None], axis=0)
    delta = (v - kv_mem) * beta
    s = s + k[:, None] * delta[None, :]
    o = tl.sum(s * q[:, None], axis=0)
    tl.store(sbase, s)
    o = o.to(tl.bfloat16).to(tl.float32)
    n = (o * tl.rsqrt(tl.sum(o * o, axis=0) / DV + eps)).to(tl.bfloat16).to(tl.float32)
    y = (tl.load(nw_ptr + vv).to(tl.float32) * n).to(tl.bfloat16).to(tl.float32)
    zz = tl.load(base_in + Z_OFF + head * DV + vv, volatile=True).to(tl.float32)
    tl.store(out_ptr + (row * V_HEADS + head) * DV + vv, (y * zz * tl.sigmoid(zz)).to(tl.bfloat16))


@triton.jit
def _prep_item(qkv_ptr, q_ptr, k_pool, v_pool, table_ptr, slot_ptr, pos_ptr, len_ptr,
               cos_ptr, sin_ptr, qn_ptr, kn_ptr, q_eps, k_eps,
               stride_in, stride_table, stride_page, stride_head, stride_tok, row, h,
               QH: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
               PAGE: tl.constexpr):
    """``fused._attn_prep_kernel`` for (row, head)."""
    pos = tl.load(pos_ptr + row)
    d = tl.arange(0, D)
    rot = d < 2 * HALF
    partner = tl.where(d < HALF, d + HALF, d - HALF)
    sign = tl.where(d < HALF, -1.0, 1.0)
    base = qkv_ptr + row * stride_in
    c = tl.load(cos_ptr + pos * HALF + (d % HALF), mask=rot, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + pos * HALF + (d % HALF), mask=rot, other=0.0).to(tl.float32)
    if h == 0:
        tl.store(len_ptr + row, (pos + 1).to(tl.int32))
    if h < QH + KVH:
        if h < QH:
            off = h * 2 * D
            nw_ptr = qn_ptr
            eps = q_eps
        else:
            off = QH * 2 * D + (h - QH) * D
            nw_ptr = kn_ptr
            eps = k_eps
        x = tl.load(base + off + d, volatile=True).to(tl.float32)
        xp = tl.load(base + off + partner, volatile=True).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        y = (x * r * (1.0 + tl.load(nw_ptr + d).to(tl.float32))).to(tl.bfloat16).to(tl.float32)
        yp = (xp * r * (1.0 + tl.load(nw_ptr + partner).to(tl.float32))).to(tl.bfloat16).to(tl.float32)
        t1 = (y * c).to(tl.bfloat16).to(tl.float32)
        t2 = ((sign * yp) * s).to(tl.bfloat16).to(tl.float32)
        y = tl.where(rot, (t1 + t2), y).to(tl.bfloat16)
        if h < QH:
            tl.store(q_ptr + (row * QH + h) * D + d, y)
        else:
            slot = tl.load(slot_ptr + row).to(tl.int64)
            page = tl.load(table_ptr + slot * stride_table + pos // PAGE).to(tl.int64)
            dst = page * stride_page + (h - QH) * stride_head + (pos % PAGE) * stride_tok + d
            tl.store(k_pool + dst, y)
    else:
        v = tl.load(base + QH * 2 * D + KVH * D + (h - QH - KVH) * D + d, volatile=True)
        slot = tl.load(slot_ptr + row).to(tl.int64)
        page = tl.load(table_ptr + slot * stride_table + pos // PAGE).to(tl.int64)
        dst = page * stride_page + (h - QH - KVH) * stride_head + (pos % PAGE) * stride_tok + d
        tl.store(v_pool + dst, v)


@triton.jit
def _paged_item(q_ptr, k_ptr, v_ptr, table_ptr, slot_ptr, len_ptr, acc_ptr, ml_ptr, scale,
                Q_HEADS, stride_table, stride_page, stride_head, stride_tok, row, kv_head, split,
                GROUP: tl.constexpr, GROUP_PAD: tl.constexpr, D: tl.constexpr,
                PAGE: tl.constexpr, SPLITS: tl.constexpr, DOT_F32: tl.constexpr):
    """``kernels._paged_attn_kernel`` for (row, kv head, split)."""
    slot = tl.load(slot_ptr + row).to(tl.int64)
    length = tl.load(len_ptr + row, volatile=True)
    pages = tl.cdiv(length, PAGE)
    per_split = tl.cdiv(pages, SPLITS)
    first = split * per_split
    last = tl.minimum(first + per_split, pages)
    g = tl.arange(0, GROUP_PAD)
    g_ok = g < GROUP
    d = tl.arange(0, D)
    heads = kv_head * GROUP + g
    q = tl.load(q_ptr + (row * Q_HEADS + heads)[:, None] * D + d[None, :], mask=g_ok[:, None],
                other=0.0, volatile=True)
    if DOT_F32:
        q = q.to(tl.float32)
    m = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, D], dtype=tl.float32)
    t = tl.arange(0, PAGE)
    p = first
    while p < last:
        page = tl.load(table_ptr + slot * stride_table + p).to(tl.int64)
        t_ok = (p * PAGE + t) < length
        offs = page * stride_page + kv_head * stride_head + t[:, None] * stride_tok + d[None, :]
        keys = tl.load(k_ptr + offs, mask=t_ok[:, None], other=0.0)
        if DOT_F32:
            keys = keys.to(tl.float32)
        s = tl.dot(q, tl.trans(keys)) * scale
        s = tl.where(t_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        prob = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(prob, axis=1)
        values = tl.load(v_ptr + offs, mask=t_ok[:, None], other=0.0)
        if DOT_F32:
            values = values.to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(prob.to(values.dtype), values)
        m = m_new
        p += 1
    out_rows = (row * Q_HEADS + heads) * SPLITS + split
    tl.store(acc_ptr + out_rows[:, None] * D + d[None, :], acc, mask=g_ok[:, None])
    tl.store(ml_ptr + out_rows * 2, m, mask=g_ok)
    tl.store(ml_ptr + out_rows * 2 + 1, l, mask=g_ok)


@triton.jit
def _combine_item(acc_ptr, ml_ptr, qkv_ptr, out_ptr, stride_in, row, h,
                  QH: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    """``fused._attn_combine_kernel`` for (row, head)."""
    sp = tl.arange(0, SPLITS)
    d = tl.arange(0, D)
    rows = (row * QH + h) * SPLITS + sp
    m = tl.load(ml_ptr + rows * 2, volatile=True)
    l = tl.load(ml_ptr + rows * 2 + 1, volatile=True)
    top = tl.max(m, axis=0)
    w = tl.exp(m - top)
    acc = tl.load(acc_ptr + rows[:, None] * D + d[None, :], volatile=True)
    o = tl.sum(acc * w[:, None], axis=0) / tl.sum(l * w, axis=0)
    o = o.to(tl.bfloat16).to(tl.float32)
    g = tl.load(qkv_ptr + row * stride_in + h * 2 * D + D + d, volatile=True).to(tl.float32)
    g = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * (QH * D) + h * D + d, (o * g).to(tl.bfloat16))


@triton.jit
def _topk_row(logit_ptr, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr):
    """``fused._topk`` with a volatile load."""
    ee = tl.arange(0, E)
    p = tl.load(logit_ptr + ee, volatile=True).to(tl.float32)
    p = tl.exp(p - tl.max(p, axis=0))
    p = p / tl.sum(p, axis=0)
    kk = tl.arange(0, TOPKP)
    sel_w = tl.zeros([TOPKP], dtype=tl.float32)
    sel_i = tl.zeros([TOPKP], dtype=tl.int32)
    for j in tl.static_range(TOPK):
        mx = tl.max(p, axis=0)
        i = tl.min(tl.where(p == mx, ee, E), axis=0)
        sel_w = tl.where(kk == j, mx, sel_w)
        sel_i = tl.where(kk == j, i, sel_i)
        p = tl.where(ee == i, -1.0, p)
    sel_w = sel_w / tl.sum(sel_w, axis=0)
    return sel_i, sel_w.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _moe_up_items(h_ptr, idx_ptr, w_ptr, act_ptr, R, stride_we, pid, G,
                  H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr):
    """``fused._moe_up_kernel`` with program id ``pid`` of ``G``."""
    ntile = tl.cdiv(I, BN)
    for t in range(pid, R * TOPK * ntile, G):
        pair = t // ntile
        tile = t % ntile
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
            xv = tl.load(h_ptr + row * H + cols, mask=cok, other=0.0, volatile=True).to(tl.float32)
            wg = tl.load(wbase + rows[:, None] * H + cols[None, :], mask=m, other=0.0).to(tl.float32)
            wu = tl.load(wbase + (rows[:, None] + I) * H + cols[None, :], mask=m, other=0.0).to(tl.float32)
            acc_g += tl.sum(wg * xv[None, :], axis=1)
            acc_u += tl.sum(wu * xv[None, :], axis=1)
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        a = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        tl.store(act_ptr + pair * I + rows, (a * u).to(tl.bfloat16), mask=rok)


@triton.jit
def _moe_down_items(act_ptr, r_ptr, idx_ptr, wt_ptr, w_ptr, sd_ptr, res_ptr, out_ptr, R,
                    stride_we, N3, pid, G,
                    H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
                    TOPK: tl.constexpr, BN: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr):
    """``fused._moe_down_kernel`` with program id ``pid`` of ``G``."""
    ntile = tl.cdiv(H, BN)
    for t in range(pid, R * ntile, G):
        row = t // ntile
        rows = (t % ntile) * BN + tl.arange(0, BN)
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
                hv = tl.load(act_ptr + pair * I + cols, mask=cok, other=0.0, volatile=True).to(tl.float32)
                w = tl.load(wbase + rows[:, None] * I + cols[None, :],
                            mask=rok[:, None] & cok[None, :], other=0.0).to(tl.float32)
                part += tl.sum(w * hv[None, :], axis=1)
            acc += (part.to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
        routed = acc.to(tl.bfloat16).to(tl.float32)
        shared = tl.zeros([BN], dtype=tl.float32)
        for k0 in range(0, IS, BKS):
            sc = k0 + tl.arange(0, BKS)
            sok = sc < IS
            g = tl.load(rbase + E + sc, mask=sok, other=0.0, volatile=True).to(tl.float32)
            u = tl.load(rbase + E + IS + sc, mask=sok, other=0.0, volatile=True).to(tl.float32)
            sa = ((g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16).to(tl.float32)
            sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :],
                         mask=rok[:, None] & sok[None, :], other=0.0).to(tl.float32)
            shared += tl.sum(sw * sa[None, :], axis=1)
        shared = shared.to(tl.bfloat16).to(tl.float32)
        sg = tl.load(rbase + E + 2 * IS, volatile=True).to(tl.float32)
        sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
        moe = (routed + (sg * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        res = tl.load(res_ptr + row * H + rows, mask=rok, other=0.0, volatile=True).to(tl.float32)
        tl.store(out_ptr + row * H + rows, (res + moe).to(tl.bfloat16), mask=rok)


@triton.jit
def _pf(w_ptr, NROWS, K: tl.constexpr, q, nq, PF, BN: tl.constexpr, BK: tl.constexpr):
    """Idle CTA q of nq: read weight tiles q, q + nq, ... (PF of them) of a [NROWS, K] weight,
    the first ones the next GEMV phase reads, so they sit in L2; returns a checksum (kept)."""
    acc = tl.zeros([BN], dtype=tl.float32)
    for j in range(PF):
        rows = (q + j * nq) * BN + tl.arange(0, BN)
        rok = rows < NROWS
        for k0 in range(0, K, BK):
            cols = k0 + tl.arange(0, BK)
            w = tl.load(w_ptr + rows[:, None].to(tl.int64) * K + cols[None, :],
                        mask=rok[:, None] & (cols[None, :] < K), other=0.0).to(tl.float32)
            acc += tl.sum(w, axis=1)
    return tl.sum(acc, axis=0)


# Pointers come from a table (int64 addresses) loaded where they are used: as 50-odd 64-bit
# kernel arguments they would pin ~100 SGPRs for the whole kernel and spill.
@triton.jit
def _pb(tab, i: tl.constexpr):
    return tl.multiple_of(tl.load(tab + i).to(tl.pointer_type(tl.bfloat16)), 16)


@triton.jit
def _pf32(tab, i: tl.constexpr):
    return tl.multiple_of(tl.load(tab + i).to(tl.pointer_type(tl.float32)), 16)


@triton.jit
def _pi32(tab, i: tl.constexpr):
    return tl.multiple_of(tl.load(tab + i).to(tl.pointer_type(tl.int32)), 16)


@triton.jit
def _pi64(tab, i: tl.constexpr):
    return tl.multiple_of(tl.load(tab + i).to(tl.pointer_type(tl.int64)), 16)


TAB = [('x0_ptr', 'b'), ('xa_ptr', 'b'), ('xb_ptr', 'b'), ('hp_ptr', 'b'), ('proj_ptr', 'b'), ('core_ptr', 'b'), ('aout_ptr', 'b'), ('r_ptr', 'b'), ('act_ptr', 'b'), ('idxp_ptr', 'i32'), ('wtp_ptr', 'f32'), ('q_ptr', 'b'), ('len_ptr', 'i32'), ('acc_ptr', 'f32'), ('ml_ptr', 'f32'), ('slot_ptr', 'i64'), ('st_ptr', 'i64'), ('pos_ptr', 'i64'), ('kind_ptr', 'i32'), ('gw_in', 'b'), ('gcw', 'b'), ('gcs', 'b'), ('grs', 'f32'), ('gw_out', 'b'), ('aw_qkv', 'b'), ('aw_o', 'b'), ('k_pool', 'b'), ('v_pool', 'b'), ('table_ptr', 'i32'), ('w3_ptr', 'b'), ('gu_ptr', 'b'), ('dn_ptr', 'b'), ('sd_ptr', 'b'), ('bar_ptr', 'i32'), ('err_ptr', 'i32'), ('fence_ptr', 'i32'), ('dummy_ptr', 'f32')]
P_X0 = tl.constexpr(0)
P_XA = tl.constexpr(1)
P_XB = tl.constexpr(2)
P_HP = tl.constexpr(3)
P_PROJ = tl.constexpr(4)
P_CORE = tl.constexpr(5)
P_AOUT = tl.constexpr(6)
P_R = tl.constexpr(7)
P_ACT = tl.constexpr(8)
P_IDXP = tl.constexpr(9)
P_WTP = tl.constexpr(10)
P_Q = tl.constexpr(11)
P_LEN = tl.constexpr(12)
P_ACC = tl.constexpr(13)
P_ML = tl.constexpr(14)
P_SLOT = tl.constexpr(15)
P_ST = tl.constexpr(16)
P_POS = tl.constexpr(17)
P_KIND = tl.constexpr(18)
P_GW_IN = tl.constexpr(19)
P_GCW = tl.constexpr(20)
P_GCS = tl.constexpr(21)
P_GRS = tl.constexpr(22)
P_GW_OUT = tl.constexpr(23)
P_AW_QKV = tl.constexpr(24)
P_AW_O = tl.constexpr(25)
P_K_POOL = tl.constexpr(26)
P_V_POOL = tl.constexpr(27)
P_TABLE = tl.constexpr(28)
P_W3 = tl.constexpr(29)
P_GU = tl.constexpr(30)
P_DN = tl.constexpr(31)
P_SD = tl.constexpr(32)
P_BAR = tl.constexpr(33)
P_ERR = tl.constexpr(34)
P_FENCE = tl.constexpr(35)
P_DUMMY = tl.constexpr(36)


@triton.jit
def _mega_kernel(
    tab, R, NL, eps, galog, gdtb, gnw, ln_in_ptr, ln_post_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr,
    cs_l, rs_l, cs_slot, rs_slot, rs_head, g_eps, g_scale, q_eps, k_eps,
    a_scale, stride_table, stride_page, stride_head, stride_tok,
    gu_l, gu_e, dn_l, dn_e, per_launch, pf, fence,
    H: tl.constexpr, NIN: tl.constexpr, CONV_N: tl.constexpr, KC: tl.constexpr,
    VD: tl.constexpr, HEADS: tl.constexpr, RATIO: tl.constexpr, KEY_DIM: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, Z_OFF: tl.constexpr, B_OFF: tl.constexpr,
    A_OFF: tl.constexpr,
    NQKV: tl.constexpr, QH: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr,
    HALF: tl.constexpr, PAGE: tl.constexpr, SPLITS: tl.constexpr, GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    N3: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr,
    I: tl.constexpr, IS: tl.constexpr,
    BP: tl.constexpr, BN: tl.constexpr, BKH: tl.constexpr, BKV: tl.constexpr,
    BKQ: tl.constexpr, MBN: tl.constexpr, MBK: tl.constexpr, BKI: tl.constexpr,
    BKS: tl.constexpr, RMAX: tl.constexpr, NT: tl.constexpr, HBLOCK: tl.constexpr,
    DOT_F32: tl.constexpr, HAS_ATTN: tl.constexpr,
):
    """NL decode layers (kind 0 = GDN, 1 = attention) for R rows in one persistent launch."""
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    base = _epoch_base(_pi32(tab, P_BAR), per_launch)
    fp = _pi32(tab, P_FENCE) + pid * NT
    hp = _pb(tab, P_HP) + pid * (RMAX * H)
    idxp = _pi32(tab, P_IDXP) + pid * (RMAX * TOPK)
    wtp = _pf32(tab, P_WTP) + pid * (RMAX * TOPK)
    dsum = tl.sum(tl.zeros([2], dtype=tl.float32), axis=0)
    hcols = tl.arange(0, HBLOCK)
    if pid == 0:  # the layer input into the residual buffer
        for row in range(R):
            v = tl.load(_pb(tab, P_X0) + row * H + hcols, mask=hcols < H, other=0.0)
            tl.store(_pb(tab, P_XA) + row * H + hcols, v, mask=hcols < H)
    nb = pid * 0 + 1
    _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
    gi = pid * 0
    for l in range(NL):
        kind = tl.load(_pi32(tab, P_KIND) + l)
        lw = tl.cast(l, tl.int64)
        _rms_rows(_pb(tab, P_XA), _pb(tab, P_XA), ln_in_ptr + l * H, _pb(tab, P_XB), hp, R, eps, 0,
                  N=H, BLOCK=HBLOCK, HAS_H=False)
        _csync(fp, fence, NT)
        if kind == 0:
            g64 = tl.cast(gi, tl.int64)
            cs = _pb(tab, P_GCS) + g64 * cs_l
            rs = _pf32(tab, P_GRS) + g64 * rs_l
            wo = _pb(tab, P_GW_OUT) + g64 * (H * VD)
            _gemv_tiles(hp, _pb(tab, P_GW_IN) + g64 * (NIN * H), _pb(tab, P_PROJ), R, NIN, NIN,
                        cs, _pi64(tab, P_ST), _pb(tab, P_GCW) + g64 * (CONV_N * KC), cs_slot, CONV_N, pid, G,
                        K=H, BP=BP, BN=BN, BK=BKH, CONV=True, KC=KC)
            nb += 1
            _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
            for it in range(pid, R * HEADS, G):
                _gdn_item(rs, _pi64(tab, P_ST), _pb(tab, P_PROJ), galog + gi * HEADS, gdtb + gi * HEADS,
                          gnw + gi * DV, _pb(tab, P_CORE), NIN, HEADS, rs_slot, rs_head, g_eps, g_scale,
                          Z_OFF, B_OFF, A_OFF, it // HEADS, it % HEADS,
                          RATIO=RATIO, KEY_DIM=KEY_DIM, DK=DK, DV=DV)
            if pid >= R * HEADS:
                dsum += _pf(wo, H, VD, pid - R * HEADS, G - R * HEADS, pf, BN, BKV)
            nb += 1
            _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
            _gemv_tiles(_pb(tab, P_CORE), wo, _pb(tab, P_AOUT), R, H, H, _pb(tab, P_CORE), _pi64(tab, P_ST), _pb(tab, P_CORE), 0, 0,
                        pid, G, K=VD, BP=BP, BN=BN, BK=BKV, CONV=False, KC=2)
            gi += 1
        elif HAS_ATTN:  # attention code only in a kernel for a chain that has one
                _gemv_tiles(hp, _pb(tab, P_AW_QKV), _pb(tab, P_PROJ), R, NQKV, NQKV, _pb(tab, P_PROJ), _pi64(tab, P_ST), _pb(tab, P_PROJ), 0, 0,
                            pid, G, K=H, BP=BP, BN=BN, BK=BKH, CONV=False, KC=2)
                nb += 1
                _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
                for it in range(pid, R * (QH + 2 * KVH), G):
                    _prep_item(_pb(tab, P_PROJ), _pb(tab, P_Q), _pb(tab, P_K_POOL), _pb(tab, P_V_POOL), _pi32(tab, P_TABLE), _pi64(tab, P_SLOT), _pi64(tab, P_POS), _pi32(tab, P_LEN),
                               cos_ptr, sin_ptr, qn_ptr, kn_ptr, q_eps, k_eps, NQKV, stride_table,
                               stride_page, stride_head, stride_tok,
                               it // (QH + 2 * KVH), it % (QH + 2 * KVH),
                               QH=QH, KVH=KVH, D=D, HALF=HALF, PAGE=PAGE)
                nb += 1
                _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
                for it in range(pid, R * KVH * SPLITS, G):
                    rem = it % (KVH * SPLITS)
                    _paged_item(_pb(tab, P_Q), _pb(tab, P_K_POOL), _pb(tab, P_V_POOL), _pi32(tab, P_TABLE), _pi64(tab, P_SLOT), _pi32(tab, P_LEN), _pf32(tab, P_ACC), _pf32(tab, P_ML),
                                a_scale, QH, stride_table, stride_page, stride_head, stride_tok,
                                it // (KVH * SPLITS), rem // SPLITS, rem % SPLITS,
                                GROUP=GROUP, GROUP_PAD=GROUP_PAD, D=D, PAGE=PAGE, SPLITS=SPLITS,
                                DOT_F32=DOT_F32)
                nb += 1
                _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
                for it in range(pid, R * QH, G):
                    _combine_item(_pf32(tab, P_ACC), _pf32(tab, P_ML), _pb(tab, P_PROJ), _pb(tab, P_CORE), NQKV, it // QH, it % QH,
                                  QH=QH, D=D, SPLITS=SPLITS)
                if pid >= R * QH:
                    dsum += _pf(_pb(tab, P_AW_O), H, QH * D, pid - R * QH, G - R * QH, pf, BN, BKQ)
                nb += 1
                _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
                _gemv_tiles(_pb(tab, P_CORE), _pb(tab, P_AW_O), _pb(tab, P_AOUT), R, H, H, _pb(tab, P_CORE), _pi64(tab, P_ST), _pb(tab, P_CORE), 0, 0,
                            pid, G, K=QH * D, BP=BP, BN=BN, BK=BKQ, CONV=False, KC=2)
        nb += 1
        _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
        # MoE: residual add + norm (CTA 0 keeps the sum), router|shared GEMV.
        _rms_rows(_pb(tab, P_XA), _pb(tab, P_AOUT), ln_post_ptr + l * H, _pb(tab, P_XB), hp, R, eps, pid == 0,
                  N=H, BLOCK=HBLOCK, HAS_H=True)
        _csync(fp, fence, NT)
        _gemv_tiles(hp, _pb(tab, P_W3) + lw * (N3 * H), _pb(tab, P_R), R, N3, N3, _pb(tab, P_R), _pi64(tab, P_ST), _pb(tab, P_R), 0, 0,
                    pid, G, K=H, BP=BP, BN=BN, BK=BKH, CONV=False, KC=2)
        nb += 1
        _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
        kk = tl.arange(0, TOPKP)
        for row in range(R):  # every CTA: its own copy of the routing (no barrier)
            sel_i, sel_w = _topk_row(_pb(tab, P_R) + row * N3, E, TOPK, TOPKP)
            tl.store(idxp + row * TOPK + kk, sel_i, mask=kk < TOPK)
            tl.store(wtp + row * TOPK + kk, sel_w, mask=kk < TOPK)
        _csync(fp, fence, NT)
        _moe_up_items(hp, idxp, _pb(tab, P_GU) + lw * gu_l, _pb(tab, P_ACT), R, gu_e, pid, G,
                      H=H, I=I, TOPK=TOPK, BN=MBN, BK=MBK)
        nb += 1
        _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
        _moe_down_items(_pb(tab, P_ACT), _pb(tab, P_R), idxp, wtp, _pb(tab, P_DN) + lw * dn_l, _pb(tab, P_SD) + lw * (H * IS),
                        _pb(tab, P_XB), _pb(tab, P_XA), R, dn_e, N3, pid, G,
                        H=H, I=I, IS=IS, E=E, TOPK=TOPK, BN=MBN, BKI=BKI, BKS=BKS)
        nb += 1
        _gbar(_pi32(tab, P_BAR), base + nb * G, _pi32(tab, P_ERR), fp, fence, NT)
    if dsum == 1.2345e30:  # never true; keeps the prefetch loads
        tl.store(_pf32(tab, P_DUMMY) + pid, dsum)


# ============================================================================================
# Host side
# ============================================================================================


def _pow2(n: int) -> int:
    return triton.next_power_of_2(n)


class _NW:
    """A launcher wrapper that forces ``num_warps`` (and optionally records calls)."""

    def __init__(self, kernel, num_warps=None, record=None):
        self.kernel, self.num_warps, self.record = kernel, num_warps, record

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def call(*args, **kw):
            if self.num_warps is not None:
                kw["num_warps"] = self.num_warps
            res = launch(*args, **kw)
            if self.record is not None:
                self.record(grid, args, kw)
            return res

        return call


class Patch:
    """Set attributes, restore on exit."""

    def __init__(self, items):
        self.items = items
        self.saved = []

    def __enter__(self):
        for obj, name, value in self.items:
            self.saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


FUSED_KERNELS = [(FZ, "_gemv_kernel"), (FZ, "_gdn_step_kernel"), (FZ, "_attn_prep_kernel"),
                 (FZ, "_attn_combine_kernel"), (FZ, "_topk_kernel"), (FZ, "_moe_up_kernel"),
                 (FZ, "_moe_down_kernel"), (KN, "_paged_attn_kernel"), (KN, "_add_rms_kernel")]


def fused_variant(num_warps: int, bn: int, mbn: int) -> Patch:
    """The fused path with every kernel at ``num_warps`` and GEMV/expert tiles of ``bn``/``mbn``
    outputs: every reduction then has the prototype's shape, so for the same (warps, bn, mbn)
    the prototype's outputs must equal it bitwise."""
    items = [(FZ, "TILES", {**FZ.TILES, "bn": bn, "moe_bn": mbn})]
    items += [(mod, name, _NW(getattr(mod, name), num_warps)) for mod, name in FUSED_KERNELS]
    return Patch(items)


def is_gdn(fast, i: int) -> bool:
    return fast.layer_types[i] == "linear_attention"


class Mega:
    """Packed weights, state copies and scratch of ``_mega_kernel`` for a chain of layers."""

    def __init__(self, fast, chain, rows, slots, states, positions, num_warps=4, bn=None,
                 mbn=None, gmax=None):
        fz = fast.fused
        self.fast, self.chain, self.rows, self.num_warps = fast, list(chain), rows, num_warps
        dev = fast.layers[chain[0]].input_layernorm.weight.device
        self.dev = dev
        self.cus = torch.cuda.get_device_properties(dev).multi_processor_count if dev.type == "cuda" else 2
        self.gmax = gmax or 2 * self.cus
        gl = [i for i in chain if is_gdn(fast, i)]
        al = [i for i in chain if not is_gdn(fast, i)]
        assert len(al) <= 1, "the prototype holds at most one attention layer"
        self.gl, self.al = gl, al
        L = fast.layers
        gdn_ref = next(i for i in range(len(L)) if is_gdn(fast, i))
        att_ref = al[0] if al else next(i for i in range(len(L)) if not is_gdn(fast, i))
        m = L[gdn_ref].linear_attn
        a = L[att_ref].self_attn
        H = fast.embed.shape[1]
        conv_state, rec_state = fast.gdn_state[gdn_ref]
        KC = conv_state.shape[-1]
        conv_dim = 2 * m.key_dim + m.value_dim
        z_off = conv_dim
        b_off = z_off + m.in_proj_z.weight.shape[0]
        a_off = b_off + m.in_proj_b.weight.shape[0]
        nin = fz.w_in[gdn_ref].shape[0]
        vd = m.num_v_heads * m.head_v_dim
        k_pool, v_pool = fast.kv[att_ref]
        kvh, page, D = k_pool.shape[1], k_pool.shape[2], k_pool.shape[3]
        qh = a.o_proj.weight.shape[1] // D
        cos, sin = fz.rope[dev]
        nqkv = fz.w_in[att_ref].shape[0]
        mlp = L[chain[0]].mlp
        gu0 = mlp.experts.gate_up_proj
        E, two_i, _ = gu0.shape
        I = two_i // 2
        IS = mlp.shared_expert.down_proj.weight.shape[1]
        n3 = fz.w_moe[chain[0]].shape[0]
        bp, bn_default, bkh = fz._tiles(rows, H)
        _, _, bkv = fz._tiles(rows, vd)
        _, _, bkq = fz._tiles(rows, qh * D)
        self.const = dict(
            H=H, NIN=nin, CONV_N=conv_dim, KC=KC, VD=vd, HEADS=m.num_v_heads,
            RATIO=m.num_v_heads // m.num_k_heads, KEY_DIM=m.key_dim, DK=m.head_k_dim,
            DV=m.head_v_dim, Z_OFF=z_off, B_OFF=b_off, A_OFF=a_off,
            NQKV=nqkv, QH=qh, KVH=kvh, D=D, HALF=cos.shape[1], PAGE=page, SPLITS=KN.ATTN_SPLITS,
            GROUP=qh // kvh, GROUP_PAD=max(16, _pow2(qh // kvh)),
            N3=n3, E=E, TOPK=fast.top_k, TOPKP=_pow2(fast.top_k), I=I, IS=IS,
            BP=bp, BN=bn_default, BKH=bkh, BKV=bkv, BKQ=bkq, MBN=FZ.TILES["moe_bn"],
            MBK=min(1024, _pow2(H)), BKI=min(1024, _pow2(I)), BKS=min(1024, _pow2(IS)),
            RMAX=rows, NT=num_warps * 64, HBLOCK=_pow2(H), DOT_F32=INTERPRET,
            HAS_ATTN=bool(al),
        )
        self.scal = dict(eps=fast.eps, g_eps=m.norm.variance_epsilon, g_scale=m.head_k_dim ** -0.5,
                         q_eps=a.q_norm.eps, k_eps=a.k_norm.eps, a_scale=a.scaling)
        # Weights: one stacked copy per kind (real checkpoint weights).
        with torch.no_grad():
            st = torch.stack
            lg = [L[i].linear_attn for i in gl] or [m]
            self.gw_in = st([fz.w_in[i] for i in gl] or [fz.w_in[gdn_ref]])
            self.gcw = st([x.conv1d.weight.reshape(conv_dim, KC) for x in lg]).contiguous()
            self.galog = st([x.A_log for x in lg])
            self.gdtb = st([x.dt_bias for x in lg])
            self.gnw = st([x.norm.weight for x in lg])
            self.gw_out = st([x.out_proj.weight for x in lg]).contiguous()
            self.gcs = st([fast.gdn_state[i][0] for i in gl] or [conv_state])
            self.grs = st([fast.gdn_state[i][1] for i in gl] or [rec_state])
            self.aw_qkv, self.aw_o = fz.w_in[att_ref], a.o_proj.weight
            self.qn, self.kn = a.q_norm.weight, a.k_norm.weight
            self.k_pool, self.v_pool = k_pool, v_pool
            self.cos, self.sin = cos, sin
            self.ln_in = st([L[i].input_layernorm.weight for i in chain])
            self.ln_post = st([L[i].post_attention_layernorm.weight for i in chain])
            self.w3 = st([fz.w_moe[i] for i in chain])
            self.gu = st([L[i].mlp.experts.gate_up_proj for i in chain])
            self.dn = st([L[i].mlp.experts.down_proj for i in chain])
            self.sd = st([L[i].mlp.shared_expert.down_proj.weight for i in chain]).contiguous()
        assert self.gu.is_contiguous() and self.dn.is_contiguous()
        self.table = fast.devices[dev].page_table
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        R = rows
        self.x0, self.xa, self.xb, self.aout = z(R, H), z(R, H), z(R, H), z(R, H)
        self.hp = z(self.gmax, R, H)
        self.proj = z(R, max(nin, nqkv))
        self.core = z(R, max(vd, qh * D))
        self.r = z(R, n3)
        self.act = z(R * fast.top_k, I)
        self.idxp = z(self.gmax, R * fast.top_k, dt=i32)
        self.wtp = z(self.gmax, R * fast.top_k, dt=f32)
        self.q = z(R, qh, D)
        self.lens = z(R, dt=i32)
        self.acc = z(R, qh, KN.ATTN_SPLITS, D, dt=f32)
        self.ml = z(R, qh, KN.ATTN_SPLITS, 2, dt=f32)
        self.slots = torch.tensor(slots, dtype=torch.long, device=dev)
        self.states = torch.tensor(states, dtype=torch.long, device=dev)
        self.positions = torch.tensor(positions, dtype=torch.long, device=dev)
        self.kinds = torch.tensor([0 if is_gdn(fast, i) else 1 for i in chain], dtype=i32, device=dev)
        self.bar = z(1, dt=i32)
        self.err = z(1, dt=i32)
        self.fence_buf = z(self.gmax * 16 * 64, dt=i32)
        self.dummy = z(self.gmax, dt=f32)
        self.nbar = 1 + sum(6 if is_gdn(fast, i) else 8 for i in chain)
        self.compiled = None
        self.grid = None
        self.build_tab()
        self.configure(num_warps, bn or bn_default, mbn or FZ.TILES["moe_bn"])

    def configure(self, num_warps: int, bn: int, mbn: int) -> None:
        """Another (warps, GEMV tile, expert tile) of the same kernel: a new compile."""
        self.num_warps = num_warps
        self.const.update(BN=bn, MBN=mbn, NT=num_warps * 64)
        self.compiled = None
        self.reset_counter()

    def ctas_per_cu(self) -> int:
        """Co-resident CTAs per CU from the compiled kernel's occupancy (waves per SIMD, 4 SIMDs)
        and LDS (64 KiB per CU); 1 when unknown."""
        ck = self.compiled
        text = (getattr(ck, "asm", {}) or {}).get("amdgcn", "") if ck is not None else ""
        m = re.search(r"; Occupancy:\s*(\d+)", text)
        if not m:
            return 1
        n = (int(m.group(1)) * 4) // self.num_warps
        lds = getattr(getattr(ck, "metadata", None), "shared", 0) or 0
        if lds:
            n = min(n, 65536 // lds)
        return max(n, 0)

    def load_states(self) -> None:
        for k, i in enumerate(self.gl):
            self.gcs[k].copy_(self.fast.gdn_state[i][0])
            self.grs[k].copy_(self.fast.gdn_state[i][1])

    def reset_counter(self) -> None:
        self.bar.zero_()
        self.err.zero_()

    def launch(self, grid: int, pf: int = 0, fence: int = 0):
        if grid != self.grid:  # the epoch base assumes one grid per counter
            self.reset_counter()
            self.grid = grid
        ck = _mega_kernel[(grid,)](*self.arglist(grid, pf, fence), num_warps=self.num_warps,
                                   **self.const)
        if self.compiled is None and ck is not None:
            self.compiled = ck
        return ck

    def build_tab(self) -> None:
        """Device table of the kernel's pointers (``TAB`` order), each checked for its dtype."""
        attr = {"x0_ptr": "x0", "len_ptr": "lens", "slot_ptr": "slots", "st_ptr": "states",
                "pos_ptr": "positions", "kind_ptr": "kinds", "table_ptr": "table",
                "fence_ptr": "fence_buf", "dummy_ptr": "dummy"}
        dts = {"b": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64}
        ptrs = []
        for name, kind in TAB:
            t = getattr(self, attr.get(name, name[:-4] if name.endswith("_ptr") else name))
            assert t.dtype == dts[kind], (name, t.dtype, kind)
            assert t.data_ptr() % 16 == 0, name
            ptrs.append(t.data_ptr())
        self.tab = torch.tensor(ptrs, dtype=torch.int64, device=self.dev)

    def arglist(self, grid: int, pf: int, fence: int) -> list:
        s = self.scal
        return [
            self.tab, self.rows, len(self.chain), s["eps"], self.galog, self.gdtb, self.gnw,
            self.ln_in, self.ln_post, self.qn, self.kn, self.cos, self.sin,
            self.gcs.stride(0), self.grs.stride(0), self.gcs.stride(1), self.grs.stride(1),
            self.grs.stride(2), s["g_eps"], s["g_scale"], s["q_eps"], s["k_eps"], s["a_scale"],
            self.table.stride(0), self.k_pool.stride(0), self.k_pool.stride(1), self.k_pool.stride(2),
            self.gu.stride(0), self.gu.stride(1), self.dn.stride(0), self.dn.stride(1),
            self.nbar * grid, pf, fence,
        ]

    def gdn_rows(self, k: int):
        st = self.states
        return self.gcs[k][st], self.grs[k][st]

    def kernel_info(self) -> str:
        ck = self.compiled
        if ck is None:
            return "n/a"
        parts = []
        for attr in ("n_regs", "n_spills", "n_max_threads"):
            v = getattr(ck, attr, None)
            if v is not None:
                parts.append(f"{attr}={v}")
        md = getattr(ck, "metadata", None)
        if md is not None:
            for attr in ("shared", "num_warps", "num_stages"):
                v = getattr(md, attr, None)
                if v is not None:
                    parts.append(f"{attr}={v}")
        asm = getattr(ck, "asm", {}) or {}
        text = asm.get("amdgcn", "") if isinstance(asm, dict) else ""
        if text:
            for key in (".vgpr_count", ".agpr_count", ".sgpr_count", ".vgpr_spill_count",
                        ".sgpr_spill_count", ".private_segment_fixed_size",
                        ".group_segment_fixed_size"):
                m = re.search(re.escape(key) + r":\s+(\d+)", text)
                if m:
                    parts.append(f"{key.lstrip('.')}={m.group(1)}")
            m = re.search(r"; Occupancy:\s*(\d+)", text)
            if m:
                parts.append(f"occupancy_waves_per_simd={m.group(1)}")
            m = re.search(r"; ScratchSize:\s*(\d+)", text)
            if m:
                parts.append(f"scratch_bytes={m.group(1)}")
            parts.append(f"isa_lines={text.count(chr(10))}")
        return " ".join(parts)


def fused_chain(fast, chain, x, slots_t, pos_t, meta, states_t):
    for i in chain:
        x = fast.fused.layer(i, x, slots_t, pos_t, meta, states_t)
    return x


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    _sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    _sync()
    return g


def time_ms(fn, n=100, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    _sync()
    return a.elapsed_time(b) / n


def time_med(fn, rounds=5, n=60) -> float:
    return statistics.median(time_ms(fn, n, 3) for _ in range(rounds))


# ============================================================================================
# Part 2: the prototype on real layers
# ============================================================================================


def proto_part(args) -> int:
    from torch.nn import functional as F
    from transformers import AutoConfig

    from engine.generator import CHECKPOINT, Generator

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    ncu = props.multi_processor_count
    out(f"RESULT env torch={torch.__version__} hip={torch.version.hip} triton={triton.__version__} "
        f"gpu={props.name} cus={ncu} visible={os.environ.get('HIP_VISIBLE_DEVICES')}")
    model, revision = CHECKPOINT["model"], CHECKPOINT["revision"]
    config = AutoConfig.from_pretrained(model, revision=revision)
    nl = 5
    for cfg in {id(c): c for c in (config, config.get_text_config())}.values():
        if hasattr(cfg, "layer_types"):
            cfg.layer_types = list(cfg.layer_types)[:nl]
            cfg.num_hidden_layers = nl
        if hasattr(cfg, "mtp_num_hidden_layers"):
            cfg.mtp_num_hidden_layers = 0
    t0 = time.perf_counter()
    gen = Generator.load(model, revision, config=config, max_model_len=16384, chunk_tokens=2048,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = gen.fast
    if fast is None:
        from engine.fast.model import FastModel

        fast = FastModel(gen.model, slots=8, max_len=16384)
    out(f"RESULT proto_load_s={time.perf_counter() - t0:.1f} layer_types={fast.layer_types} "
        f"fused={fast.fused is not None} fused_error={fast.fused_error} row_tile={FZ.ROW_TILE}")
    if fast.fused is None:
        out("RESULT verdict no fused path")
        return 1
    devs = {str(lay.input_layernorm.weight.device) for lay in fast.layers}
    out(f"RESULT proto_devices {sorted(devs)}")
    assert devs == {str(dev)}, "the prototype needs all layers on one GCD"
    fast.use_graphs = False
    torch.set_grad_enabled(False)

    # Replay-like prompts (Request Factory's synthetic coding lines) of ~2-4k tokens.
    tok = gen.tokenizer
    ids: list[int] = []
    n = 0
    while len(ids) < 20000:
        for line in (f"def case_{n}(items):", f"    return sorted(set(items))  # {n}"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        n += 1
    R = args.rows
    plens = [args.prompt_len + 300 * k for k in range(R)]

    def prefill(slot, prompt):
        fast.reset(slot)
        body = len(prompt) - 1
        pos = 0
        while pos < body:
            end = min((pos // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[pos:end], pos)
            pos = end
        return fast.prefill(slot, prompt[body:], body)

    with torch.inference_mode():
        firsts = [int(prefill(s, ids[s * 997 : s * 997 + plens[s]]).argmax()) for s in range(R)]
    slots = list(range(R))
    states = [fast._g(s) for s in slots]
    positions = list(plens)
    for s, p in zip(slots, positions):
        fast._ensure_pages(s, p + 1)
    meta = fast.devices[dev]
    st_t = torch.tensor(slots, device=dev)
    gs_t = torch.tensor(states, device=dev)
    pt_t = torch.tensor(positions, device=dev)
    x_in = F.embedding(torch.tensor(firsts, device=dev), fast.embed).contiguous()
    _sync()

    gdn_layers = [i for i in range(nl) if is_gdn(fast, i)]
    att_layers = [i for i in range(nl) if not is_gdn(fast, i)]
    chains = {"mixed": gdn_layers[:3] + att_layers[:1], "gdn4": gdn_layers[:4]}
    chains["mixed"] = sorted(chains["mixed"])
    verdict = {}

    def save():
        g = {i: (c.clone(), r.clone()) for i, (c, r) in fast.gdn_state.items()}
        kv = {i: (k.clone(), v.clone()) for i, (k, v) in fast.kv.items()}
        return g, kv

    def restore(saved):
        for i, (c, r) in saved[0].items():
            fast.gdn_state[i][0].copy_(c)
            fast.gdn_state[i][1].copy_(r)
        for i, (k, v) in saved[1].items():
            fast.kv[i][0].copy_(k)
            fast.kv[i][1].copy_(v)
        _sync()

    def kv_rows(i):
        k_pool, v_pool = fast.kv[i]
        pages = meta.page_table[st_t, pt_t // k_pool.shape[2]].long()
        off = pt_t % k_pool.shape[2]
        return k_pool[pages, :, off].clone(), v_pool[pages, :, off].clone()

    def snapshot(chain):
        g = {i: (fast.gdn_state[i][0][gs_t].clone(), fast.gdn_state[i][1][gs_t].clone())
             for i in chain if is_gdn(fast, i)}
        kv = {i: kv_rows(i) for i in chain if not is_gdn(fast, i)}
        return g, kv

    def cmp(a, b):
        a, b = a.float(), b.float()
        return bool(torch.equal(a, b)), float((a - b).abs().max()), float((a == b).float().mean())

    saved = save()
    topk_log: list[torch.Tensor] = []

    def rec_topk(grid, a, kw):
        topk_log.append(a[1].clone())

    def state_cmp(sa, sb):
        worst, alleq = 0.0, True
        for part in (0, 1):
            for i in sa[part]:
                for x, y in zip(sa[part][i], sb[part][i]):
                    e = cmp(x, y)
                    alleq &= e[0]
                    worst = max(worst, e[1])
        return alleq, worst

    configs = [tuple(int(v) for v in c.split("x")) for c in args.configs.split(",")]
    for name in args.chains.split(","):
        chain = chains[name]
        out(f"RESULT chain {name} layers={chain} kinds={[fast.layer_types[i] for i in chain]} "
            f"R={R} positions={positions}")
        mega = None
        try:
            # The current fused path (eager reference, routing, yardstick) and its graph time.
            restore(saved)
            topk_log.clear()
            with torch.inference_mode(), Patch([(FZ, "_topk_kernel", _NW(FZ._topk_kernel, None, rec_topk))]):
                ref = fused_chain(fast, chain, x_in, st_t, pt_t, meta, gs_t).clone()
            ref_state = snapshot(chain)
            uniq = [int(t.unique().numel()) for t in topk_log]
            restore(saved)
            tile = FZ.ROW_TILE
            FZ.ROW_TILE = "fixed"
            try:
                with torch.inference_mode():
                    yard = fused_chain(fast, chain, x_in, st_t, pt_t, meta, gs_t).clone()
            finally:
                FZ.ROW_TILE = tile
            yard_state = snapshot(chain)
            restore(saved)
            ey = cmp(yard, ref)
            eys = state_cmp(yard_state, ref_state)
            out(f"RESULT yardstick {name} fused_rowtile_fixed_vs_fused out_max_abs={ey[1]:.4g} "
                f"equal_frac={ey[2]:.4f} state_max_abs={eys[1]:.4g} ref_absmax={float(ref.float().abs().max()):.3g}")

            # Bytes of the 4 layers: weights once, the experts picked, state r/w, KV read.
            c0 = fast.fused
            nbytes = 0
            for k, i in enumerate(chain):
                lay = fast.layers[i]
                nbytes += c0.w_in[i].numel() * 2 + c0.w_moe[i].numel() * 2
                gu = lay.mlp.experts.gate_up_proj
                dn = lay.mlp.experts.down_proj
                nbytes += uniq[k] * (gu[0].numel() + dn[0].numel()) * 2 if k < len(uniq) else 0
                nbytes += lay.mlp.shared_expert.down_proj.weight.numel() * 2
                if is_gdn(fast, i):
                    m = lay.linear_attn
                    nbytes += m.out_proj.weight.numel() * 2
                    nbytes += 2 * R * m.num_v_heads * m.head_k_dim * m.head_v_dim * 4
                else:
                    kp = fast.kv[i][0]
                    nbytes += lay.self_attn.o_proj.weight.numel() * 2
                    nbytes += sum(p + 1 for p in positions) * kp.shape[1] * kp.shape[3] * 2 * 2
            floor_ms = nbytes / BW * 1e3
            out(f"RESULT bytes {name} MB={nbytes / 1e6:.1f} unique_experts_per_layer={uniq} "
                f"floor_at_1.3TBps_ms={floor_ms:.4f}")

            def fused_call():
                with torch.inference_mode():
                    fused_chain(fast, chain, x_in, st_t, pt_t, meta, gs_t)

            restore(saved)
            g_f = graph_of(fused_call)
            t_f = time_med(g_f.replay)
            del g_f
            out(f"RESULT time {name} fused_graph_ms={t_f:.4f} (current fused path, 4 layers, R={R}, "
                f"median of 5 x 60 replays) floor_ms={floor_ms:.4f} fused_over_floor={t_f / floor_ms:.2f}x")
            verdict.setdefault("fused_ms", []).append(t_f)
            best = None
            for ci, (nw, bn, mbn) in enumerate(configs):
                tag = f"{name} warps={nw} bn={bn} moe_bn={mbn}"
                try:
                    # The fused path at the same warps and tiles: bitwise reference and its time.
                    restore(saved)
                    with torch.inference_mode(), fused_variant(nw, bn, mbn):
                        refv = fused_chain(fast, chain, x_in, st_t, pt_t, meta, gs_t).clone()
                        refv_state = snapshot(chain)
                        restore(saved)
                        g_v = graph_of(fused_call)
                    t_v = time_med(g_v.replay)
                    del g_v
                    ev = cmp(refv, ref)
                    if mega is None:
                        mega = Mega(fast, chain, R, slots, states, positions, num_warps=nw, bn=bn, mbn=mbn)
                        mega.x0.copy_(x_in)
                    else:
                        mega.configure(nw, bn, mbn)
                    restore(saved)
                    mega.load_states()
                    _sync()
                    t0 = time.perf_counter()
                    mega.launch(ncu, 0, 0)
                    _sync()
                    compile_s = time.perf_counter() - t0
                    err = int(mega.err.item())
                    per_cu = mega.ctas_per_cu()
                    out(f"RESULT compile {tag} first_launch_s={compile_s:.1f} timeouts={err} "
                        f"ctas_per_cu_possible={per_cu} {mega.kernel_info()}")
                    out(f"RESULT fused_variant {tag} fused_ms={t_v:.4f} (vs current {t_f:.4f}) "
                        f"out_max_abs_vs_current={ev[1]:.4g}")
                    verdict.setdefault("compile_s", []).append(compile_s)

                    def run_once(grid, pf=0, fence=0):
                        restore(saved)
                        mega.load_states()
                        mega.launch(grid, pf, fence)
                        _sync()
                        o = mega.xa.clone()
                        g = {i: mega.gdn_rows(k) for k, i in enumerate(mega.gl)}
                        kv = {i: kv_rows(i) for i in mega.al}
                        return o, (g, kv), int(mega.err.item())

                    grids = []
                    for grid in (ncu, 2 * ncu):
                        if grid > ncu and per_cu < 2:
                            out(f"RESULT resident {tag} grid={grid} skipped (occupancy allows {per_cu}/CU)")
                            continue
                        _, _, perr = run_once(grid)
                        out(f"RESULT resident {tag} grid={grid} ({grid // ncu}/CU) all_resident={perr == 0} "
                            f"timeouts={perr}")
                        if perr == 0:
                            grids.append(grid)
                        elif grid == ncu:
                            verdict["hang"] = True
                    for grid in grids:
                        for fence in ((0, 1) if ci == 0 else (0,)):
                            o, st_m, e1 = run_once(grid, 2, fence)
                            o2, st_m2, e2 = run_once(grid, 0, fence)
                            eb = cmp(o, refv)
                            sb = state_cmp(st_m, refv_state)
                            ec = cmp(o, ref)
                            sc = state_cmp(st_m, ref_state)
                            det = cmp(o, o2)[0] and state_cmp(st_m, st_m2)[0]
                            ok = eb[0] and sb[0] and det and e1 == 0 and e2 == 0
                            out(f"RESULT parity {tag} grid={grid} fence={fence} vs_fused_same_tiles "
                                f"out_equal={eb[0]} out_max_abs={eb[1]:.4g} state_equal={sb[0]} "
                                f"state_max_abs={sb[1]:.4g} | vs_current_fused out_max_abs={ec[1]:.4g} "
                                f"equal_frac={ec[2]:.4f} state_max_abs={sc[1]:.4g} | deterministic={det} "
                                f"timeouts={e1 + e2} -> {'MATCH' if ok else 'MISMATCH'}")
                            verdict.setdefault("match", []).append(ok)
                    for grid in grids:
                        for pf in (0, 2, 4):
                            restore(saved)
                            mega.load_states()
                            mega.reset_counter()
                            mega.grid = grid
                            g_m = graph_of(lambda grid=grid, pf=pf: mega.launch(grid, pf, 0))
                            t_m = time_med(g_m.replay)
                            err = int(mega.err.item())
                            out(f"RESULT time {tag} grid={grid} ({grid // ncu}/CU) prefetch_tiles={pf} "
                                f"mega_ms={t_m:.4f} ratio_vs_current_fused={t_m / t_f:.3f} "
                                f"ratio_vs_fused_same_tiles={t_m / t_v:.3f} timeouts={err} "
                                f"eff_TBps={nbytes / (t_m * 1e-3) / 1e12:.3f}")
                            if err == 0 and (best is None or t_m < best[0]):
                                best = (t_m, (nw, bn, mbn), grid, pf)
                            del g_m
                except Exception:  # noqa: BLE001
                    out(f"RESULT config {tag} error " + traceback.format_exc().replace("\n", " | "))
                    verdict.setdefault("errors", []).append(tag)
                    mega = None
                    torch.cuda.empty_cache()
            if best is not None:
                t_m, cfg, grid, pf = best
                if mega is None:
                    mega = Mega(fast, chain, R, slots, states, positions, num_warps=cfg[0], bn=cfg[1],
                                mbn=cfg[2])
                    mega.x0.copy_(x_in)
                mega.configure(*cfg)
                restore(saved)
                mega.load_states()
                mega.reset_counter()
                mega.grid = grid
                g_m = graph_of(lambda: mega.launch(grid, pf, 0))
                t_plain = time_med(lambda: mega.launch(grid, pf, 0))
                mega.err.zero_()
                t0 = time.perf_counter()
                for _ in range(2000):
                    g_m.replay()
                _sync()
                err = int(mega.err.item())
                del g_m
                out(f"RESULT hangcheck {name} config={cfg} grid={grid} pf={pf} graph_replays=2000 "
                    f"timeouts={err} wall_s={time.perf_counter() - t0:.2f} plain_launch_ms={t_plain:.4f}")
                verdict.setdefault("hang_timeouts", []).append(err)
                verdict.setdefault("ratio", []).append(t_m / t_f)
                out(f"RESULT best {name} mega_ms={t_m:.4f} fused_ms={t_f:.4f} ratio={t_m / t_f:.3f} "
                    f"config(warps,bn,moe_bn)={cfg} grid={grid} prefetch={pf} floor_ms={floor_ms:.4f} "
                    f"mega_over_floor={t_m / floor_ms:.2f}x fused_over_floor={t_f / floor_ms:.2f}x")
        except Exception:  # noqa: BLE001
            out(f"RESULT chain {name} error " + traceback.format_exc().replace("\n", " | "))
            verdict.setdefault("errors", []).append(name)
        mega = None
        torch.cuda.empty_cache()
        restore(saved)

    # Verdict for the prototype half of the kill rules.
    ratios = verdict.get("ratio", [])
    comp = max(verdict.get("compile_s", [0.0]))
    match = all(verdict.get("match", [False])) and bool(verdict.get("match"))
    hang = verdict.get("hang", False) or any(verdict.get("hang_timeouts", []))
    kills = []
    if not ratios or min(ratios) > 0.80:
        kills.append(f"ratio {[round(r, 3) for r in ratios]} > 0.80")
    if comp > 900:
        kills.append(f"compile {comp:.0f}s > 15 min")
    if hang:
        kills.append("hang/timeouts")
    if not match:
        kills.append("mismatch")
    if verdict.get("errors"):
        kills.append(f"errors in {verdict['errors']}")
    out(f"RESULT proto_kill fires={bool(kills)} reasons={kills} ratios={[round(r, 3) for r in ratios]} "
        f"fused_ms={[round(t, 4) for t in verdict.get('fused_ms', [])]} max_compile_s={comp:.1f} match={match}")
    return 0


# ============================================================================================
# Part 1: the captured verify B=2 step, split into memory-bound kernels and small ops
# ============================================================================================


def family_of(name: str) -> str | None:
    if "_gemv_kernel" in name:
        return "gemv"
    if "_moe_up_kernel" in name:
        return "moe_up"
    if "_moe_down_kernel" in name:
        return "moe_down"
    if "_gdn_step_kernel" in name:
        return "gdn_step"
    if "_paged_attn_kernel" in name:
        return "paged_attn"
    if "indexSelectSmallIndex<float" in name:
        return "isel_f32"
    if "index_copy_kernel_impl<at::native::OpaqueType<4>" in name:
        return "icopy_f32"
    if "Cijk" in name or "gemm" in name.lower():
        return "blas"
    return None


def split_part(args) -> int:
    from torch.nn import functional as F
    from torch.profiler import ProfilerActivity, profile

    from engine.generator import CHECKPOINT, Generator

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    out(f"RESULT split_load_s={time.perf_counter() - started:.0f} gpus={torch.cuda.device_count()} "
        f"mtp={fast.mtp is not None} graphs={fast.use_graphs} fused={fast.fused is not None}")
    tok = g.tokenizer
    ids: list[int] = []
    i = 0
    while len(ids) < 4000:
        for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i}"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1

    def prefill(slot, prompt):
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
    lasts = [int(torch.argmax(prefill(s, ids[s * 100 : s * 100 + plen]))) for s in slots]

    def step():
        fast.verify(lasts, [0, 0], slots, [plen, plen])

    for _ in range(5):
        step()
    sync()
    times = []
    for _ in range(args.iters):
        sync()
        t0 = time.perf_counter()
        step()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    out(f"RESULT step verify_B2_ms={statistics.median(times):.2f} (median of {args.iters}, "
        f"graphs={fast.use_graphs}, error={fast.graph_error}, prompt_len={plen})")

    nsteps = 5
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(nsteps):
            step()
        sync()
    events = defaultdict(list)  # profiler device -> [(start, us, name)]
    for e in prof.events():
        dt = str(getattr(e, "device_type", ""))
        if "CUDA" not in dt and "HIP" not in dt:
            continue
        events[e.device_index].append((e.time_range.start, e.time_range.elapsed_us(), e.name))
    for d in events:
        events[d].sort()

    # One eager step with the launchers wrapped to record each call's bytes.
    rec = defaultdict(list)  # (torch device, family) -> [bytes per call, in issue order]
    lay0 = next(lay for lay in fast.layers if hasattr(lay.mlp, "experts"))
    gu = lay0.mlp.experts.gate_up_proj
    dn = lay0.mlp.experts.down_proj
    sd = lay0.mlp.shared_expert.down_proj.weight
    gu_e, dn_e, sd_b = gu[0].numel() * 2, dn[0].numel() * 2, sd.numel() * 2

    orig_gemv = FZ.FusedDecode._gemv
    orig_linear = F.linear
    orig_isel = torch.Tensor.index_select
    orig_icopy = torch.Tensor.index_copy_

    def gemv(self, x, w, *a, **kw):
        rec[(x.device.index, "gemv")].append(w.numel() * w.element_size() + 2 * x.numel() * x.element_size())
        return orig_gemv(self, x, w, *a, **kw)

    def linear(x, w, *a, **kw):
        rec[(x.device.index, "blas")].append(w.numel() * w.element_size() + x.numel() * x.element_size())
        return orig_linear(x, w, *a, **kw)

    def isel(self, dim, index, *a, **kw):
        r = orig_isel(self, dim, index, *a, **kw)
        if self.dtype == torch.float32 and self.is_cuda:
            rec[(self.device.index, "isel_f32")].append(2 * r.numel() * 4)
        return r

    def icopy(self, dim, index, source, *a, **kw):
        if self.dtype == torch.float32 and self.is_cuda:
            rec[(self.device.index, "icopy_f32")].append(2 * source.numel() * 4)
        return orig_icopy(self, dim, index, source, *a, **kw)

    def r_topk(grid, a, kw):
        idx = a[1]
        u = int(idx.unique().numel())
        d = idx.device.index
        rec[(d, "moe_up")].append(u * gu_e)
        rec[(d, "moe_down")].append(u * dn_e + sd_b)

    def r_gdn(grid, a, kw):
        rows, heads = grid[0], grid[1]
        rec[(a[0].device.index, "gdn_step")].append(rows * heads * kw["DK"] * kw["DV"] * 4 * 2)

    def r_paged(grid, a, kw):
        lengths = a[5]
        rows, kvh = grid[0], grid[1]
        tot = int(lengths[:rows].long().sum())
        rec[(a[0].device.index, "paged_attn")].append(tot * kvh * kw["D"] * 2 * a[1].element_size())

    use_graphs = fast.use_graphs
    try:
        with Patch([(FZ.FusedDecode, "_gemv", gemv), (F, "linear", linear),
                    (torch.Tensor, "index_select", isel), (torch.Tensor, "index_copy_", icopy),
                    (FZ, "_topk_kernel", _NW(FZ._topk_kernel, None, r_topk)),
                    (FZ, "_gdn_step_kernel", _NW(FZ._gdn_step_kernel, None, r_gdn)),
                    (KN, "_paged_attn_kernel", _NW(KN._paged_attn_kernel, None, r_paged))]):
            fast.use_graphs = False
            step()
            sync()
            rec.clear()
            step()
            sync()
    finally:
        fast.use_graphs = use_graphs
    tdevs = sorted({d for d, _ in rec})
    pdevs = sorted(d for d in events if any(family_of(n) == "gemv" for _, _, n in events[d]))
    devmap = dict(zip(pdevs, tdevs))
    out(f"RESULT split devmap profiler->torch={devmap} recorded_calls="
        f"{ {f'{d}:{f}': len(v) for (d, f), v in sorted(rec.items())} }")

    total_us = 0.0
    fam = defaultdict(lambda: defaultdict(float))  # family -> stats per step
    small_by_name = defaultdict(float)
    matched = {}
    for pd, evs in events.items():
        td = devmap.get(pd)
        byfam = defaultdict(list)
        for _, us, name in evs:
            total_us += us / nsteps
            f = family_of(name)
            if f is None or td is None or (td, f) not in rec:
                small_by_name[name[:90]] += us / nsteps
                continue
            byfam[f].append(us)
        for f, uss in byfam.items():
            seq = rec[(td, f)]
            n = len(seq)
            per_call = len(uss) == nsteps * n
            matched[f"{pd}:{f}"] = f"{len(uss) // nsteps if nsteps else 0}/{n}{'' if per_call else ' FAMILY-LEVEL'}"
            st = fam[f]
            if per_call:
                for k, us in enumerate(uss):
                    b = seq[k % n]
                    floor = b / BW * 1e6
                    st["calls"] += 1 / nsteps
                    st["us"] += us / nsteps
                    st["bytes"] += b / nsteps
                    if b >= BIG:
                        st["mb_us"] += us / nsteps
                        st["mb_floor"] += floor / nsteps
                        st["excess"] += max(0.0, us - floor) / nsteps
                    else:
                        st["small_us"] += us / nsteps
            else:  # counts differ: family totals, all as memory-bound if the mean call is big
                t = sum(uss) / nsteps
                b = sum(seq)
                st["calls"] += len(uss) / nsteps
                st["us"] += t
                st["bytes"] += b
                if n and b / n >= BIG:
                    st["mb_us"] += t
                    st["mb_floor"] += b / BW * 1e6
                    st["excess"] += max(0.0, t - b / BW * 1e6)
                else:
                    st["small_us"] += t
    out(f"RESULT split matched (calls per step graph/recorded) {matched}")
    out(f"RESULT split total_kernel_us_per_step={total_us:.0f} (both GCDs, graph replay)")
    mb = fl = ex = sm = 0.0
    for f, st in sorted(fam.items(), key=lambda kv: -kv[1]["us"]):
        tbps = st["bytes"] / max(st["us"] * 1e-6, 1e-12) / 1e12
        out(f"RESULT split family={f:10s} calls={st['calls']:6.1f} us={st['us']:8.1f} "
            f"GB={st['bytes'] / 1e9:7.3f} TBps={tbps:5.2f} memory_bound_us={st['mb_us']:8.1f} "
            f"floor_us@1.3TBps={st['mb_floor']:8.1f} excess_us={st['excess']:8.1f} "
            f"small_calls_us={st['small_us']:7.1f}")
        mb += st["mb_us"]
        fl += st["mb_floor"]
        ex += st["excess"]
        sm += st["small_us"]
    other = sum(small_by_name.values())
    for name, us in sorted(small_by_name.items(), key=lambda kv: -kv[1])[:25]:
        out(f"RESULT split small_op us={us:7.1f} name={name}")
    small = sm + other
    share = (small + ex) / max(total_us, 1e-9)
    out(f"RESULT split summary step_kernel_us={total_us:.0f} memory_bound_us={mb:.0f} "
        f"(floor {fl:.0f}, excess {ex:.0f}) small_ops_us={small:.0f} "
        f"(small calls of tracked families {sm:.0f}, other kernels {other:.0f}) "
        f"small+excess={small + ex:.0f} share={share * 100:.1f}%")
    out(f"RESULT split_kill fires={share < 0.25} (small ops + excess {share * 100:.1f}% vs 25%)")
    return 0


# ============================================================================================
# Self test (CPU, TRITON_INTERPRET=1): a tiny random model through the fused path and the
# prototype (one program, so the barriers pass at once)
# ============================================================================================


def selftest(args) -> int:
    from types import SimpleNamespace as NS

    assert INTERPRET, "run with TRITON_INTERPRET=1"
    torch.manual_seed(0)
    bf = torch.bfloat16
    H, NKH, NVH, DK, DV, KC = 64, 2, 4, 16, 16, 4
    E, TOPK, I, IS = 16, 4, 16, 16
    QH, KVH, D, PAGE = 4, 2, 16, 8
    R = 4
    S = 6

    def P(*s, scale=0.2):
        return torch.nn.Parameter((torch.randn(*s) * scale).to(bf), requires_grad=False)

    def mlp():
        return NS(gate=NS(weight=P(E, H)),
                  shared_expert=NS(gate_proj=NS(weight=P(IS, H)), up_proj=NS(weight=P(IS, H)),
                                   down_proj=NS(weight=P(H, IS))),
                  shared_expert_gate=NS(weight=P(1, H)),
                  experts=NS(gate_up_proj=P(E, 2 * I, H).data.contiguous(),
                             down_proj=P(E, H, I).data.contiguous()))

    key_dim, value_dim = NKH * DK, NVH * DV
    conv_dim = 2 * key_dim + value_dim

    def gdn_layer():
        m = NS(in_proj_qkv=NS(weight=P(conv_dim, H)), in_proj_z=NS(weight=P(value_dim, H)),
               in_proj_b=NS(weight=P(NVH, H)), in_proj_a=NS(weight=P(NVH, H)),
               conv1d=NS(weight=P(conv_dim, 1, KC)), key_dim=key_dim, value_dim=value_dim,
               num_v_heads=NVH, num_k_heads=NKH, head_k_dim=DK, head_v_dim=DV,
               A_log=torch.randn(NVH) * 0.1, dt_bias=P(NVH), norm=NS(weight=P(DV) + 1, variance_epsilon=1e-6),
               out_proj=NS(weight=P(H, value_dim)))
        return NS(linear_attn=m, mlp=mlp(), input_layernorm=NS(weight=P(H)),
                  post_attention_layernorm=NS(weight=P(H)))

    def attn_layer():
        a = NS(q_proj=NS(weight=P(2 * QH * D, H)), k_proj=NS(weight=P(KVH * D, H)),
               v_proj=NS(weight=P(KVH * D, H)), o_proj=NS(weight=P(H, QH * D)),
               q_norm=NS(weight=P(D), eps=1e-6), k_norm=NS(weight=P(D), eps=1e-6),
               scaling=D ** -0.5)
        return NS(self_attn=a, mlp=mlp(), input_layernorm=NS(weight=P(H)),
                  post_attention_layernorm=NS(weight=P(H)))

    layers = [gdn_layer(), gdn_layer(), gdn_layer(), attn_layer(), gdn_layer()]
    types = ["linear_attention"] * 3 + ["full_attention", "linear_attention"]
    pages = 128
    max_len = 128
    fast = NS(layers=layers, layer_types=types, top_k=TOPK, eps=1e-6, embed=P(10, H),
              max_len=max_len, inv_freq={torch.device("cpu"): 1.0 / (10000 ** (torch.arange(0, 4) / 4))},
              gdn_state={}, kv={})
    for i, t in enumerate(types):
        if t == "linear_attention":
            fast.gdn_state[i] = (torch.randn(S, conv_dim, KC).to(bf), torch.randn(S, NVH, DK, DV) * 0.1)
        else:
            fast.kv[i] = (torch.randn(pages, KVH, PAGE, D).to(bf), torch.randn(pages, KVH, PAGE, D).to(bf))
    table = torch.zeros(S, max_len // PAGE, dtype=torch.int32)
    perm = torch.randperm(pages)
    for s in range(S):
        table[s] = perm[s * (max_len // PAGE): (s + 1) * (max_len // PAGE)] % pages
    fast.devices = {torch.device("cpu"): NS(page_table=table)}
    fast.mtp = None
    fz = FZ.FusedDecode(fast)
    fast.fused = fz
    slots = [0, 2, 3, 5]
    states = slots
    positions = [37, 5, 90, 64]
    meta = fast.devices[torch.device("cpu")]
    st_t = torch.tensor(slots)
    pt_t = torch.tensor(positions)
    x = (torch.randn(R, H)).to(bf)
    g0 = {i: (c.clone(), r.clone()) for i, (c, r) in fast.gdn_state.items()}
    kv0 = {i: (k.clone(), v.clone()) for i, (k, v) in fast.kv.items()}

    def restore():
        for i, (c, r) in g0.items():
            fast.gdn_state[i][0].copy_(c)
            fast.gdn_state[i][1].copy_(r)
        for i, (k, v) in kv0.items():
            fast.kv[i][0].copy_(k)
            fast.kv[i][1].copy_(v)

    ok_all = True
    for name, chain in (("mixed", [0, 1, 2, 3]), ("gdn4", [0, 1, 2, 4])):
        restore()
        with fused_variant(4, 4, 8):
            ref = fused_chain(fast, chain, x, st_t, pt_t, meta, st_t).clone()
        ref_g = {i: (fast.gdn_state[i][0][st_t].clone(), fast.gdn_state[i][1][st_t].clone())
                 for i in chain if i in fast.gdn_state}
        ref_kv = {i: (fast.kv[i][0].clone(), fast.kv[i][1].clone()) for i in chain if i in fast.kv}
        restore()
        mega = Mega(fast, chain, R, slots, states, positions, num_warps=4, bn=4, mbn=8, gmax=1)
        mega.x0.copy_(x)
        mega.load_states()
        mega.launch(1, 2, 1)
        got = mega.xa.clone()
        d = float((got.float() - ref.float()).abs().max())
        eq = bool(torch.equal(got, ref))
        sd = 0.0
        for k, i in enumerate(mega.gl):
            c, r = mega.gdn_rows(k)
            sd = max(sd, float((c.float() - ref_g[i][0].float()).abs().max()),
                     float((r - ref_g[i][1]).abs().max()))
        for i in mega.al:
            sd = max(sd, float((fast.kv[i][0].float() - ref_kv[i][0].float()).abs().max()),
                     float((fast.kv[i][1].float() - ref_kv[i][1].float()).abs().max()))
        out(f"SELFTEST {name} out_equal={eq} out_max_abs={d:.4g} state_max_abs={sd:.4g} "
            f"timeouts={int(mega.err.item())} ref_absmax={float(ref.float().abs().max()):.3g}")
        ok_all &= d < 0.05 and sd < 0.05 and int(mega.err.item()) == 0
    out(f"SELFTEST {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split")
    sp.add_argument("--devices", default="0,1")
    sp.add_argument("--prompt-len", type=int, default=1800)
    sp.add_argument("--iters", type=int, default=20)
    pp = sub.add_parser("proto")
    pp.add_argument("--device", default="0")
    pp.add_argument("--rows", type=int, default=4)
    pp.add_argument("--prompt-len", type=int, default=2000)
    pp.add_argument("--chains", default="gdn4,mixed")
    pp.add_argument("--configs", default="16x4x8,8x4x8,4x4x8,16x8x16,16x16x16",
                    help="warps x GEMV outputs per tile x expert rows per tile")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "split":
        return split_part(args)
    if args.cmd == "proto":
        return proto_part(args)
    return selftest(args)


if __name__ == "__main__":
    sys.exit(main())
