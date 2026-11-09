"""Plan 75 stage 0: does a tile-outer, all-rows MoE down kernel (shared expert tile loaded once
per work item, the shared activation hoisted out of the item, optional in-item expert dedup)
lift decode down above today's ``fused._moe_down_kernel``? Tools only, no engine change.

    python tools/moe_down_rows_bench.py [--device N] [--rows 2,4,6,8,12,16,24]

Shapes: one TP2 rank's shards (routed experts gate_up [256, 512, 2048], down [256, 2048, 256],
shared expert down [2048, 256], router|shared output N3 = 769), as plan 72's
``tools/layout_bw_bench.py`` (3f4f0b3), whose timing and routing this reuses: one CUDA graph of
>= 64 launches over 6 rotating routing sets (R <= 8: plan 72's disjoint 39-expert pools; R > 8:
8 random distinct experts per row), median of 5 replays.

Variants (each checked bitwise against the engine's up + down on every set, asserted):
- ``engine``: today's kernels as ``fused._moe`` launches them.
- ``sa``: today's down reading the shared activation from a buffer the up kernel writes
  (attribution of the per-item shared-activation recompute).
- ``a0``: one item = (output tile, group of RG rows); the shared down tile is loaded once per item;
  per row today's body (8 routed tiles, combine, shared dot, gate, store).
- ``a1``: one item = one output tile, all rows: the item walks the step's pairs (sorted by
  expert when DEDUP, a tile load skipped when the expert repeats), keeps each pair's fp32 part in
  registers, then combines every row in today's order.
Up-side costs: ``upx`` (the up kernel plus the ``sa`` and pair-sort writes) is timed and its
difference to today's up is added to each variant's down (``eff``).
Prints ``RESULT`` lines; a hint for the plan, never the task's metric.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine.fast import fused, kernels  # noqa: E402

triton, tl = kernels.triton, kernels.tl

H = 2048
E, I2, TOPK = 256, 256, 8
IS = 256
N3 = E + 2 * IS + 1
NSETS = 6
BF = torch.bfloat16
T0 = time.time()


def out(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------- kernels

@triton.jit
def up_x(
    h_ptr, idx_ptr, w_ptr, act_ptr, R, stride_we, r_ptr, sa_ptr, perm_ptr, N3,
    H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, ORDER: tl.constexpr, P2: tl.constexpr,
    WRITE_SA: tl.constexpr, WRITE_PERM: tl.constexpr,
):
    """``fused._moe_up_kernel`` (FUSE_TOPK off) plus: items of pick 0 write the row's shared
    activation bf16(bf16(silu(g)) * u) for their tile (WRITE_SA); items of tile 0 write
    perm[rank(pair)] = pair, pairs sorted by expert, stable (WRITE_PERM)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntile = tl.cdiv(I, BN)
    P = R * TOPK
    for t in range(pid, P * ntile, nprog):
        if ORDER == 1:
            pair = t % P
            tile = t // P
        else:
            pair = t // ntile
            tile = t % ntile
        row = pair // TOPK
        expert = tl.load(idx_ptr + pair).to(tl.int64)
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
        if WRITE_SA:
            if pair % TOPK == 0:
                sok = rows < IS
                rb = r_ptr + row * N3 + E
                sg_ = tl.load(rb + rows, mask=sok, other=0.0).to(tl.float32)
                su_ = tl.load(rb + IS + rows, mask=sok, other=0.0).to(tl.float32)
                sa = ((sg_ * tl.sigmoid(sg_)).to(tl.bfloat16).to(tl.float32) * su_).to(tl.bfloat16)
                tl.store(sa_ptr + row * IS + rows, sa, mask=sok)
        if WRITE_PERM:
            if tile == 0:
                q = tl.arange(0, P2)
                eq = tl.load(idx_ptr + q, mask=q < P, other=E)
                ep = expert.to(tl.int32)
                before = (eq < ep) | ((eq == ep) & (q < pair))
                rank = tl.sum(before.to(tl.int32), axis=0)
                tl.store(perm_ptr + rank, pair)


@triton.jit
def down_sa(
    act_ptr, r_ptr, sa_ptr, idx_ptr, wt_ptr, w_ptr, sd_ptr, res_ptr, out_ptr, R, stride_we, N3,
    H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, BN: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr,
    ORDER: tl.constexpr, HAS_RES: tl.constexpr,
):
    """``fused._moe_down_kernel`` with the shared activation read from ``sa`` [R, IS] bf16."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntile = tl.cdiv(H, BN)
    for t in range(pid, R * ntile, nprog):
        if ORDER == 1:
            row = t % R
            tile = t // R
        else:
            row = t // ntile
            tile = t % ntile
        rows = tile * BN + tl.arange(0, BN)
        rok = rows < H
        rbase = r_ptr + row * N3
        acc = tl.zeros([BN], dtype=tl.float32)
        for j in tl.static_range(TOPK):
            pair = row * TOPK + j
            expert = tl.load(idx_ptr + pair).to(tl.int64)
            weight = tl.load(wt_ptr + pair)
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
            sa = tl.load(sa_ptr + row * IS + sc, mask=sok, other=0.0).to(tl.float32)
            sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :],
                         mask=rok[:, None] & sok[None, :], other=0.0).to(tl.float32)
            shared += tl.sum(sw * sa[None, :], axis=1)
        shared = shared.to(tl.bfloat16).to(tl.float32)
        sg = tl.load(rbase + E + 2 * IS).to(tl.float32)
        sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
        moe = (routed + (sg * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        if HAS_RES:
            res = tl.load(res_ptr + row * H + rows, mask=rok, other=0.0).to(tl.float32)
        else:
            res = tl.zeros([BN], dtype=tl.float32)
        tl.store(out_ptr + row * H + rows, (res + moe).to(tl.bfloat16), mask=rok)


@triton.jit
def down_a0(
    act_ptr, r_ptr, sa_ptr, idx_ptr, wt_ptr, w_ptr, sd_ptr, res_ptr, out_ptr, R, stride_we, N3,
    H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, BN: tl.constexpr, RG: tl.constexpr, TPP: tl.constexpr,
    SA: tl.constexpr, HAS_RES: tl.constexpr,
):
    """Item = (output tile, group of RG rows), group inner; TPP items per program. The shared
    down tile [BN, IS] is loaded once per item; per row today's arithmetic (I == BKI and
    IS == BKS: one K step each, as the engine at the TP2 shard)."""
    pid = tl.program_id(0)
    ntile = tl.cdiv(H, BN)
    ngrp = tl.cdiv(R, RG)
    cols = tl.arange(0, I)
    sc = tl.arange(0, IS)
    for tt in tl.static_range(TPP):
        t = pid * TPP + tt
        tile = t // ngrp
        grp = t % ngrp
        if tile < ntile:
            rows = tile * BN + tl.arange(0, BN)
            rok = rows < H
            sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :], mask=rok[:, None],
                         other=0.0).to(tl.float32)
            r0 = grp * RG
            r1 = tl.minimum(r0 + RG, R)
            for row in range(r0, r1):
                rbase = r_ptr + row * N3
                acc = tl.zeros([BN], dtype=tl.float32)
                for j in tl.static_range(TOPK):
                    pair = row * TOPK + j
                    expert = tl.load(idx_ptr + pair).to(tl.int64)
                    weight = tl.load(wt_ptr + pair)
                    wbase = w_ptr + expert * stride_we
                    part = tl.zeros([BN], dtype=tl.float32)
                    hv = tl.load(act_ptr + pair * I + cols).to(tl.float32)
                    w = tl.load(wbase + rows[:, None] * I + cols[None, :], mask=rok[:, None],
                                other=0.0).to(tl.float32)
                    part += tl.sum(w * hv[None, :], axis=1)
                    acc += (part.to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
                routed = acc.to(tl.bfloat16).to(tl.float32)
                if SA:
                    sa = tl.load(sa_ptr + row * IS + sc).to(tl.float32)
                else:
                    g = tl.load(rbase + E + sc).to(tl.float32)
                    u = tl.load(rbase + E + IS + sc).to(tl.float32)
                    sa = ((g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16).to(tl.float32)
                shared = tl.zeros([BN], dtype=tl.float32)
                shared += tl.sum(sw * sa[None, :], axis=1)
                shared = shared.to(tl.bfloat16).to(tl.float32)
                sg = tl.load(rbase + E + 2 * IS).to(tl.float32)
                sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
                moe = (routed + (sg * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
                if HAS_RES:
                    res = tl.load(res_ptr + row * H + rows, mask=rok, other=0.0).to(tl.float32)
                else:
                    res = tl.zeros([BN], dtype=tl.float32)
                tl.store(out_ptr + row * H + rows, (res + moe).to(tl.bfloat16), mask=rok)


@triton.jit
def down_a1(
    act_ptr, r_ptr, sa_ptr, idx_ptr, wt_ptr, perm_ptr, w_ptr, sd_ptr, res_ptr, out_ptr, R,
    stride_we, N3,
    H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, BN: tl.constexpr, RP: tl.constexpr, CH: tl.constexpr,
    TPP: tl.constexpr, DEDUP: tl.constexpr, HAS_RES: tl.constexpr,
):
    """Item = one output tile, all R rows (RP = pow2(R)); TPP items per program. Pairs are
    walked in ``perm`` order (sorted by expert) with DEDUP, a tile load skipped when the expert
    equals the previous pair's; else in pair order, every tile loaded. Each pair's fp32 part
    (today's [BN, I] dot) is kept in a [RP * TOPK, BN] register tensor; then every row is
    combined in today's j order, the shared dot is per row against the tile loaded once."""
    pid = tl.program_id(0)
    ntile = tl.cdiv(H, BN)
    P = R * TOPK
    cols = tl.arange(0, I)
    sc = tl.arange(0, IS)
    pr = tl.arange(0, RP * TOPK)
    rr = tl.arange(0, RP)
    for tt in tl.static_range(TPP):
        tile = pid * TPP + tt
        if tile < ntile:
            rows = tile * BN + tl.arange(0, BN)
            rok = rows < H
            sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :], mask=rok[:, None],
                         other=0.0).to(tl.float32)
            parts = tl.zeros([RP * TOPK, BN], dtype=tl.float32)
            prev = tl.full([], -1, tl.int64)
            wcur = tl.zeros([BN, I], dtype=tl.bfloat16)
            for q0 in range(0, P, CH):
                for k in tl.static_range(CH):
                    q = q0 + k
                    if DEDUP:
                        p = tl.load(perm_ptr + q)
                    else:
                        p = q
                    e = tl.load(idx_ptr + p).to(tl.int64)
                    hv = tl.load(act_ptr + p * I + cols).to(tl.float32)
                    if DEDUP:
                        new = e != prev
                        wl = tl.load(w_ptr + e * stride_we + rows[:, None] * I + cols[None, :],
                                     mask=rok[:, None] & new, other=0.0)
                        wcur = tl.where(new, wl, wcur)
                        prev = e
                    else:
                        wcur = tl.load(w_ptr + e * stride_we + rows[:, None] * I + cols[None, :],
                                       mask=rok[:, None], other=0.0)
                    part = tl.zeros([BN], dtype=tl.float32)
                    part += tl.sum(wcur.to(tl.float32) * hv[None, :], axis=1)
                    parts = tl.where(pr[:, None] == p, part[None, :], parts)
            # shared expert per row, gathered into [RP, BN]
            shm = tl.zeros([RP, BN], dtype=tl.float32)
            for row in range(0, R):
                sa = tl.load(sa_ptr + row * IS + sc).to(tl.float32)
                shared = tl.zeros([BN], dtype=tl.float32)
                shared += tl.sum(sw * sa[None, :], axis=1)
                shm = tl.where(rr[:, None] == row, shared[None, :], shm)
            # combine: parts [RP*TOPK, BN] -> [RP, BN, TOPK], split j in today's order
            p3 = tl.permute(tl.reshape(parts, [RP, TOPK, BN]), [0, 2, 1])
            wq = tl.load(wt_ptr + rr[:, None] * TOPK + tl.arange(0, TOPK)[None, :],
                         mask=(rr < R)[:, None], other=0.0)
            acc = tl.zeros([RP, BN], dtype=tl.float32)
            for j in tl.static_range(TOPK):
                pj = tl.sum(tl.where(tl.arange(0, TOPK)[None, None, :] == j, p3, 0.0), axis=2)
                wj = tl.sum(tl.where(tl.arange(0, TOPK)[None, :] == j, wq, 0.0), axis=1)
                acc += (pj.to(tl.bfloat16).to(tl.float32) * wj[:, None]).to(tl.bfloat16).to(tl.float32)
            routed = acc.to(tl.bfloat16).to(tl.float32)
            shared = shm.to(tl.bfloat16).to(tl.float32)
            rv = rr < R
            sg = tl.load(r_ptr + rr * N3 + E + 2 * IS, mask=rv, other=0.0).to(tl.float32)
            sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
            moe = (routed + (sg[:, None] * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            m2 = rv[:, None] & rok[None, :]
            optr = rr[:, None] * H + rows[None, :]
            if HAS_RES:
                res = tl.load(res_ptr + optr, mask=m2, other=0.0).to(tl.float32)
            else:
                res = tl.zeros([RP, BN], dtype=tl.float32)
            tl.store(out_ptr + optr, (res + moe).to(tl.bfloat16), mask=m2)


# ------------------------------------------------------------------------------- helpers

def pick_device(arg: int | None) -> int:
    n = torch.cuda.device_count()
    if arg is not None:
        return arg
    free = []
    for d in range(n):
        try:
            f, _ = torch.cuda.mem_get_info(d)
        except Exception:  # noqa: BLE001
            f = 0
        free.append((f, d))
    out(f"RESULT devices n={n} free_GB={[round(f / 1e9, 1) for f, _ in sorted(free, key=lambda x: x[1])]}")
    return max(free)[1]


def time_graph(fns, min_launches: int = 64, reps: int = 5) -> float:
    """us per call of launch closures, captured back to back as one CUDA graph (plan 72)."""
    seq = list(fns) * max(1, math.ceil(min_launches / len(fns)))
    for f in fns:
        f()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fns[0]()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for f in seq:
            f()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(seq))
    del graph
    return statistics.median(times)


def asm_loads(ck) -> str:
    try:
        s = ck.asm["amdgcn"]
    except Exception:  # noqa: BLE001
        return "asm=n/a"
    x4 = s.count("load_dwordx4")
    x2 = s.count("load_dwordx2")
    x1 = s.count("load_dword ") + s.count("load_dword\t")
    sh = s.count("load_ushort") + s.count("load_short")
    vg = ""
    for line in s.splitlines():
        if ".vgpr_count" in line or "NumVgprs:" in line:
            vg = " " + line.strip().replace(" ", "")
            break
    return f"ld_x4={x4} ld_x2={x2} ld_x1={x1} ld_short={sh}{vg}"


def routing_sets(gen, R):
    """R <= 8: plan 72's NSETS disjoint pools of 39 experts (row r: window 5r .. 5r+7);
    R > 8: 8 random distinct experts per row from all E."""
    if R <= 8:
        pool = torch.randperm(E, generator=gen).tolist()
        sets = []
        for s in range(NSETS):
            p = pool[39 * s : 39 * s + 39]
            sets.append([[p[(5 * r + j) % 39] for j in range(TOPK)] for r in range(R)])
        return sets
    return [[torch.randperm(E, generator=gen)[:TOPK].tolist() for _ in range(R)] for _ in range(NSETS)]


# variants: name, kind, kwargs
def variants():
    v = [("sa", "sa", {})]
    for bn, w in ((8, 4), (4, 2)):
        for rg in (0, 4, 2):  # 0: all rows
            v.append((f"a0_rg{rg or 'R'}_bn{bn}w{w}", "a0", dict(RG=rg, BN=bn, W=w, TPP=1, SA=1)))
    v.append(("a0_rgR_bn8w8", "a0", dict(RG=0, BN=8, W=8, TPP=1, SA=1)))
    v.append(("a0_rgR_bn8w4_tpp2", "a0", dict(RG=0, BN=8, W=4, TPP=2, SA=1)))
    v.append(("a0_rgR_bn8w4_nosa", "a0", dict(RG=0, BN=8, W=4, TPP=1, SA=0)))
    for bn, w in ((8, 4), (4, 2)):
        for dd in (1, 0):
            v.append((f"a1_dd{dd}_bn{bn}w{w}", "a1", dict(DEDUP=dd, BN=bn, W=w, TPP=1)))
    v.append(("a1_dd1_bn8w8", "a1", dict(DEDUP=1, BN=8, W=8, TPP=1)))
    v.append(("a1_dd1_bn8w4_tpp2", "a1", dict(DEDUP=1, BN=8, W=4, TPP=2)))
    return v


def run(dev, rows_list) -> None:
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    gen = torch.Generator().manual_seed(75)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=gen) * scale).to(BF).to(dev)

    gu = rnd(E, 2 * I2, H)
    dn = rnd(E, H, I2)
    sd = rnd(H, IS)
    bn0 = fused.TILES["moe_bn"]
    out(f"RESULT setup device={dev} cus={cus} grid={fused.MOE_GRID}/CU tile_rows={fused.MOE_TILE_ROWS} "
        f"moe_bn={bn0}")
    summary = {}
    for R in rows_list:
        sets = routing_sets(gen, R)
        x = rnd(R, H, scale=1.0)
        r = rnd(R, N3, scale=1.0)
        idxs = [torch.tensor(s, dtype=torch.int32, device=dev) for s in sets]
        wts = []
        for _ in sets:
            w_ = torch.rand(R, TOPK, generator=gen)
            wts.append((w_ / w_.sum(1, keepdim=True)).to(torch.bfloat16).float().to(dev))
        distinct = statistics.mean(len({e for row in s for e in row}) for s in sets)
        order = 1 if R >= fused.MOE_TILE_ROWS else 0
        ns = len(sets)
        acts = [torch.empty(R * TOPK, I2, dtype=BF, device=dev) for _ in sets]
        outs = [torch.empty(R, H, dtype=BF, device=dev) for _ in sets]
        sas = [torch.empty(R, IS, dtype=BF, device=dev) for _ in sets]
        perms = [torch.empty(R * TOPK, dtype=torch.int32, device=dev) for _ in sets]
        items_up = R * TOPK * math.ceil(I2 / bn0)
        g_up = items_up if order else min(items_up, fused.MOE_GRID * cus)
        g_dn = min(R * math.ceil(H / bn0), fused.MOE_GRID * cus)
        b_dn = distinct * H * I2 * 2 + H * IS * 2
        b_up = distinct * 2 * I2 * H * 2

        def eng_up(s):
            fused._moe_up_kernel[(g_up,)](
                x, idxs[s], gu, acts[s], R, gu.stride(0), r, wts[s], N3, H=H, I=I2, TOPK=TOPK,
                BN=bn0, BK=1024, ORDER=order, E=E, TOPKP=8, FUSE_TOPK=False, num_warps=4)

        def eng_dn(s, o):
            fused._moe_down_kernel[(g_dn,)](
                acts[s], r, idxs[s], wts[s], dn, sd, o, o, R, dn.stride(0), N3, H=H, I=I2,
                IS=IS, E=E, TOPK=TOPK, BN=bn0, BKI=256, BKS=256, ORDER=order, HAS_RES=False,
                num_warps=4)

        ck = {}

        def upx(s, act):
            ck["upx"] = up_x[(g_up,)](
                x, idxs[s], gu, act, R, gu.stride(0), r, sas[s], perms[s], N3, H=H, I=I2, IS=IS,
                E=E, TOPK=TOPK, BN=bn0, BK=1024, ORDER=order, P2=triton.next_power_of_2(R * TOPK),
                WRITE_SA=True, WRITE_PERM=True, num_warps=4)

        # references: engine act and out per set
        ref_act, ref_out = [], []
        for s in range(ns):
            eng_up(s)
            o = torch.empty(R, H, dtype=BF, device=dev)
            eng_dn(s, o)
            torch.cuda.synchronize()
            ref_act.append(acts[s].clone())
            ref_out.append(o)
        # sa / perm reference (torch) for the up_x check
        try:
            act_x = torch.empty_like(acts[0])
            ok_act = ok_sa = ok_perm = True
            for s in range(ns):
                upx(s, act_x)
                torch.cuda.synchronize()
                ok_act &= torch.equal(act_x, ref_act[s])
                g = r[:, E:E + IS].float()
                u = r[:, E + IS:E + 2 * IS].float()
                sa_ref = ((g * torch.sigmoid(g)).to(BF).float() * u).to(BF)
                ok_sa &= torch.equal(sas[s], sa_ref)
                fl = idxs[s].flatten().long()
                pref = torch.sort(fl, stable=True).indices.to(torch.int32)
                ok_perm &= torch.equal(perms[s], pref)
            out(f"RESULT check R={R} upx bitwise_act={int(ok_act)} sa_eq_torch={int(ok_sa)} "
                f"perm_eq_torch={int(ok_perm)} | upx {asm_loads(ck.get('upx'))}")
        except Exception as e:  # noqa: BLE001
            out(f"RESULT upx R={R} FAIL {type(e).__name__}: {str(e)[:400]}")
            traceback.print_exc()
            continue
        u_eng = time_graph([lambda s=s: eng_up(s) for s in range(ns)])
        d_eng = time_graph([lambda s=s: eng_dn(s, outs[s]) for s in range(ns)])
        a_eng = time_graph([lambda s=s: (eng_up(s), eng_dn(s, outs[s])) for s in range(ns)])
        u_x = time_graph([lambda s=s: upx(s, acts[s]) for s in range(ns)])
        out(f"RESULT moe R={R} distinct={distinct:.1f} engine up_us={u_eng:.1f} "
            f"up_TBps={b_up / u_eng / 1e6:.3f} down_us={d_eng:.1f} down_TBps={b_dn / d_eng / 1e6:.3f} "
            f"all_us={a_eng:.1f} | upx_us={u_x:.1f} (+{u_x - u_eng:.1f})")
        summary[(R, "engine")] = (d_eng, d_eng, a_eng, True)
        up_extra = max(0.0, u_x - u_eng)
        for name, kind, kw in variants():
            try:
                if kind == "sa":
                    def dn_v(s, o):
                        ck["dn"] = down_sa[(g_dn,)](
                            acts[s], r, sas[s], idxs[s], wts[s], dn, sd, o, o, R, dn.stride(0), N3,
                            H=H, I=I2, IS=IS, E=E, TOPK=TOPK, BN=bn0, BKI=256, BKS=256,
                            ORDER=order, HAS_RES=False, num_warps=4)
                elif kind == "a0":
                    rg = kw["RG"] or R
                    bn = kw["BN"]
                    nit = math.ceil(H / bn) * math.ceil(R / rg)
                    grid = math.ceil(nit / kw["TPP"])

                    def dn_v(s, o, rg=rg, bn=bn, grid=grid, kw=kw):
                        ck["dn"] = down_a0[(grid,)](
                            acts[s], r, sas[s], idxs[s], wts[s], dn, sd, o, o, R, dn.stride(0), N3,
                            H=H, I=I2, IS=IS, E=E, TOPK=TOPK, BN=bn, RG=rg, TPP=kw["TPP"],
                            SA=kw["SA"], HAS_RES=False, num_warps=kw["W"])
                else:
                    bn = kw["BN"]
                    grid = math.ceil(math.ceil(H / bn) / kw["TPP"])

                    def dn_v(s, o, bn=bn, grid=grid, kw=kw):
                        ck["dn"] = down_a1[(grid,)](
                            acts[s], r, sas[s], idxs[s], wts[s], perms[s], dn, sd, o, o, R,
                            dn.stride(0), N3, H=H, I=I2, IS=IS, E=E, TOPK=TOPK, BN=bn,
                            RP=triton.next_power_of_2(R), CH=8, TPP=kw["TPP"], DEDUP=kw["DEDUP"],
                            HAS_RES=False, num_warps=kw["W"])
                bit = True
                for s in range(ns):
                    upx(s, acts[s])
                for s in range(ns):
                    o = torch.empty(R, H, dtype=BF, device=dev)
                    dn_v(s, o)
                    torch.cuda.synchronize()
                    bit &= torch.equal(o, ref_out[s])
                dd = time_graph([lambda s=s: dn_v(s, outs[s]) for s in range(ns)])
                aa = time_graph([lambda s=s: (upx(s, acts[s]), dn_v(s, outs[s])) for s in range(ns)])
                # the sa-less a0 needs no up_x; charge only variants that read sa/perm
                extra = 0.0 if (kind == "a0" and not kw.get("SA", 1)) else up_extra
                eff = dd + extra
                summary[(R, name)] = (dd, eff, aa, bit)
                out(f"RESULT moe R={R} {name} bitwise_out={int(bit)} down_us={dd:.1f} "
                    f"down_TBps={b_dn / dd / 1e6:.3f} eff_us={eff:.1f} x_engine={d_eng / eff:.3f} "
                    f"upx+down_us={aa:.1f} all_x_engine={a_eng / aa:.3f}"
                    + (f" | {asm_loads(ck.get('dn'))}" if R == 8 else ""))
            except Exception as e:  # noqa: BLE001
                out(f"RESULT moe R={R} {name} FAIL {type(e).__name__}: {str(e)[:400]}")
                traceback.print_exc()
        torch.cuda.empty_cache()

    # verdict per plan 75's rules (ratios against the engine measured in this job)
    names = sorted({n for (_, n) in summary if n != "engine"})
    out("RESULT table name: " + " ".join(f"R{R}" for R in rows_list) + " (x engine down, eff; * = not bitwise)")
    for n in names:
        cells = []
        for R in rows_list:
            if (R, n) in summary and (R, "engine") in summary:
                dd, eff, aa, bit = summary[(R, n)]
                cells.append(f"{summary[(R, 'engine')][0] / eff:.3f}{'' if bit else '*'}")
            else:
                cells.append("-")
        out(f"RESULT table {n}: " + " ".join(cells))

    def best(R):
        c = [(eff, n) for (rr, n), (dd, eff, aa, bit) in summary.items() if rr == R and n != "engine" and bit]
        return min(c) if c else (float("nan"), "none")

    for R in rows_list:
        if (R, "engine") in summary:
            b = best(R)
            out(f"RESULT best R={R} engine_down_us={summary[(R, 'engine')][0]:.1f} best={b[1]} "
                f"eff_us={b[0]:.1f} x={summary[(R, 'engine')][0] / b[0]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--rows", default="2,4,6,8,12,16,24")
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.cuda.set_device(dev)
    out(f"RESULT env torch={torch.__version__} hip={torch.version.hip} triton={triton.__version__} "
        f"device={dev} name={torch.cuda.get_device_name(dev)} "
        f"cus={torch.cuda.get_device_properties(dev).multi_processor_count}")
    with torch.no_grad():
        run(torch.device("cuda", dev), [int(v) for v in args.rows.split(",")])
    out("RESULT done")


if __name__ == "__main__":
    main()
