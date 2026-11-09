"""Plan 72 stage 0: does a channel-aware weight layout (row stride K + pad, staggered expert
bases, a 256 B chunk XOR swizzle) lift decode weight streaming on one gfx90a GCD? Tools only,
no engine change.

    python tools/layout_bw_bench.py --part a|moe|dense [--device N]

Shapes are one TP2 rank's shards: routed experts gate_up [256, 512, 2048] and down
[256, 2048, 256], the shared expert's down [2048, 256], and the dense GEMVs in_proj 6176x2048,
q|k|v 4608x2048, out_proj 2048x2048, router|shared 769x2048.

Layouts (the same BF16 values, only the addresses change):
- ``base``: today's (row stride = K, expert stride = rows x K, power-of-two everywhere);
- ``pP``: row stride K + P elements (P = 32/64/128/256 = 64/128/256/512 B);
- ``st``: expert stride rows x stride + 2176 elements (4.25 KB), so expert e's base moves by
  e x 4.25 KB against a 2 MB / 1 MB grid;
- ``swz``: 256 B chunk c of row r stored at chunk c ^ (r & m) (m = chunks per row - 1, at most
  15): the kernel reads logical column c at c ^ ((r & m) << 7), so the K order of every sum is
  unchanged.

(a) ``--part a``: a bare read-only probe (sums what it loads, writes one float per work item)
    with today's access shapes: the gate_up tile (8 rows x 1024 columns per step, the gate and
    up halves 512 rows apart, all programs walking K in lockstep, tile-outer item order, R=8 x 8
    pairs over 39 distinct experts) and the in_proj / q|k|v MFMA tiles (64 or 32 rows x 128
    columns per step). Per layout TB/s; plus ``contig`` (each work item reads its own bytes as
    one linear run), ``krot`` (today's layout, each item starts its K walk at a different chunk:
    breaks the lockstep, not bitwise, a diagnostic only) and a plain 1-D stream of the same
    bytes (the ceiling).
(b) ``--part moe``: copies of ``fused._moe_up_kernel`` / ``_moe_down_kernel`` with the row
    stride, expert stride and swizzle as arguments, at R = 2/4/8 rows x 8 experts (39 distinct
    at R = 8), up, down, and up+down; outputs bitwise against the engine's kernels on today's
    layout. ``--part dense``: a copy of ``mfma._mfma_gemv_kernel`` with a row stride and swizzle
    at R = 2/8, bitwise against ``mfma.gemv``.

Every timing is one CUDA graph of >= 64 back-to-back launches over rotating weights (dense:
copies >= 256 MB per rotation; experts: 6 disjoint routing sets, 702 MB of experts per
rotation), median of 5 replays. Bytes are the useful bytes (distinct experts once), the same
for every layout. Prints ``RESULT`` lines; a hint for the plan, never the task's metric.
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

from engine.fast import fused, kernels, mfma  # noqa: E402

triton, tl = kernels.triton, kernels.tl

H = 2048
E, I2, TOPK = 256, 256, 8   # experts, TP2 expert intermediate shard, top-k
IS = 256                    # TP2 shared expert intermediate shard
N3 = E + 2 * IS + 1         # router | shared gate | shared up | shared gate scalar (769)
ST = 2176                   # expert base stagger in elements (4352 B = 4.25 KB)
NSETS = 6                   # disjoint routing sets per rotation
DENSE = {"in_proj": (6176, 2048), "qkv": (4608, 2048), "out_proj": (2048, 2048),
         "router": (769, 2048)}
BF = torch.bfloat16
T0 = time.time()


def out(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------- kernels

@triton.jit
def _swz(cols, rows, SWZM: tl.constexpr):
    """Physical column of logical column ``cols`` in row ``rows`` (2-D [rows, cols])."""
    p = cols[None, :] ^ ((rows[:, None] & SWZM) << 7)
    p = tl.max_contiguous(tl.multiple_of(p, [1, 8]), [1, 128])
    return p


@triton.jit
def probe_kernel(w_ptr, seg_ptr, out_ptr, NSEG, NR: tl.constexpr, NT: tl.constexpr,
                 SW: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 HALF: tl.constexpr, SWZM: tl.constexpr, ORDER: tl.constexpr,
                 KROT: tl.constexpr):
    """Sum of item t's BN-row tile of segment seg (base seg_ptr[seg]) over K in BK chunks; with
    HALF also rows + HALF (the up half). Item order as the MoE up kernel's ORDER."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nk: tl.constexpr = K // BK
    for t in range(pid, NSEG * NT, nprog):
        if ORDER == 1:
            seg = t % NSEG
            tile = t // NSEG
        else:
            seg = t // NT
            tile = t % NT
        base = tl.load(seg_ptr + seg)
        wb = w_ptr + base
        rows = tile * BN + tl.arange(0, BN)
        rok = rows < NR
        acc = tl.zeros([BN, BK], dtype=tl.float32)
        for kk in range(0, nk):
            if KROT:
                k0 = ((kk + t) % nk) * BK
            else:
                k0 = kk * BK
            cols = k0 + tl.arange(0, BK)
            if SWZM == 0:
                pc = cols[None, :]
            else:
                pc = _swz(cols, rows, SWZM)
            acc += tl.load(wb + rows[:, None] * SW + pc, mask=rok[:, None], other=0.0).to(tl.float32)
            if HALF > 0:
                if SWZM == 0:
                    pu = cols[None, :]
                else:
                    pu = _swz(cols, rows + HALF, SWZM)
                acc += tl.load(wb + (rows[:, None] + HALF) * SW + pu, mask=rok[:, None],
                               other=0.0).to(tl.float32)
        tl.store(out_ptr + t, tl.sum(tl.sum(acc, axis=1), axis=0))


@triton.jit
def probe_contig_kernel(w_ptr, seg_ptr, out_ptr, NSEG, NT: tl.constexpr, K: tl.constexpr,
                        BN: tl.constexpr, BK: tl.constexpr, HALF: tl.constexpr,
                        ORDER: tl.constexpr):
    """``probe_kernel``'s items and bytes (today's layout, BN x K rows per tile, full tiles),
    each item reading its tile as one linear run BN x BK elements at a time."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, NSEG * NT, nprog):
        if ORDER == 1:
            seg = t % NSEG
            tile = t // NSEG
        else:
            seg = t // NT
            tile = t % NT
        base = tl.load(seg_ptr + seg)
        wb = w_ptr + base + tile * BN * K
        acc = tl.zeros([BN * BK], dtype=tl.float32)
        for k0 in range(0, BN * K, BN * BK):
            lin = k0 + tl.arange(0, BN * BK)
            acc += tl.load(wb + lin).to(tl.float32)
            if HALF > 0:
                acc += tl.load(wb + HALF * K + lin).to(tl.float32)
        tl.store(out_ptr + t, tl.sum(acc, axis=0))


@triton.jit
def stream_kernel(w_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Plain grid-stride stream of N elements (the ceiling)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in range(pid * BLOCK, N, nprog * BLOCK):
        acc += tl.load(w_ptr + s + tl.arange(0, BLOCK)).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def moe_up_s(h_ptr, idx_ptr, w_ptr, act_ptr, R, stride_we,
             H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr, ORDER: tl.constexpr, SW: tl.constexpr, SWZM: tl.constexpr):
    """``fused._moe_up_kernel`` (FUSE_TOPK=0) with row stride SW and the chunk swizzle."""
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
            if SWZM == 0:
                pg = cols[None, :]
                pu = cols[None, :]
            else:
                pg = _swz(cols, rows, SWZM)
                pu = _swz(cols, rows + I, SWZM)
            wg = tl.load(wbase + rows[:, None] * SW + pg, mask=m, other=0.0).to(tl.float32)
            wu = tl.load(wbase + (rows[:, None] + I) * SW + pu, mask=m, other=0.0).to(tl.float32)
            acc_g += tl.sum(wg * xv[None, :], axis=1)
            acc_u += tl.sum(wu * xv[None, :], axis=1)
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        a = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        tl.store(act_ptr + pair * I + rows, (a * u).to(tl.bfloat16), mask=rok)


@triton.jit
def moe_down_s(act_ptr, r_ptr, idx_ptr, wt_ptr, w_ptr, sd_ptr, res_ptr, out_ptr, R, stride_we, N3,
               H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
               TOPK: tl.constexpr, BN: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr,
               ORDER: tl.constexpr, HAS_RES: tl.constexpr, SW: tl.constexpr,
               SWZM: tl.constexpr):
    """``fused._moe_down_kernel`` with the routed down row stride SW and the chunk swizzle."""
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
                if SWZM == 0:
                    pc = cols[None, :]
                else:
                    pc = _swz(cols, rows, SWZM)
                w = tl.load(wbase + rows[:, None] * SW + pc,
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
        if HAS_RES:
            res = tl.load(res_ptr + row * H + rows, mask=rok, other=0.0).to(tl.float32)
        else:
            res = tl.zeros([BN], dtype=tl.float32)
        tl.store(out_ptr + row * H + rows, (res + moe).to(tl.bfloat16), mask=rok)


@triton.jit(do_not_specialize=["R", "N"])
def mfma_gemv_s(x_ptr, w_ptr, out_ptr, part_ptr, R, N, stride_out,
                K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                KSPLIT: tl.constexpr, SW: tl.constexpr, SWZM: tl.constexpr):
    """``mfma._mfma_gemv_kernel`` (DOT_F32=0) with row stride SW and the chunk swizzle."""
    pid = tl.program_id(0)
    nb = pid // KSPLIT
    kp = pid % KSPLIT
    KP: tl.constexpr = K // KSPLIT
    rm = tl.arange(0, BM)
    mok = rm < R
    rn = nb * BN + tl.arange(0, BN)
    nok = rn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(kp * KP, kp * KP + KP, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
        if SWZM == 0:
            pk = rk[:, None]
        else:
            pk = rk[:, None] ^ ((rn[None, :] & SWZM) << 7)
            pk = tl.max_contiguous(tl.multiple_of(pk, [8, 1]), [128, 1])
        b = tl.load(w_ptr + rn[None, :].to(tl.int64) * SW + pk, mask=nok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    m = mok[:, None] & nok[None, :]
    if KSPLIT == 1:
        tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(part_ptr + (kp * BM + rm[:, None]) * N + rn[None, :], acc, mask=m)


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
    """us per call of launch closures, captured back to back as one CUDA graph."""
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
    """Vector-width summary of a compiled kernel's global loads (x4 = 16 B)."""
    try:
        s = ck.asm["amdgcn"]
    except Exception:  # noqa: BLE001
        return "asm=n/a"
    x4 = s.count("load_dwordx4")
    x2 = s.count("load_dwordx2")
    x1 = s.count("load_dword ") + s.count("load_dword\t")
    sh = s.count("load_ushort") + s.count("load_short")
    return f"ld_x4={x4} ld_x2={x2} ld_x1={x1} ld_short={sh}"


def layout_rows(w: torch.Tensor, pad: int, stagger: int, swz: bool):
    """Physical copy of w [S, N, K] (S segments): row stride K + pad, segment stride
    N (K + pad) + stagger, 256 B chunks swizzled by row. Returns (buffer, view, stride_seg, SW,
    SWZM)."""
    s, n, k = w.shape
    sw = k + pad
    sseg = n * sw + stagger
    buf = torch.zeros(s * sseg + 256, dtype=w.dtype, device=w.device)
    view = buf.as_strided((s, n, k), (sseg, sw, 1))
    nc = k // 128
    m = min(nc, 16) - 1 if swz else 0
    if m == 0:
        view.copy_(w)
    else:
        for sv in range(m + 1):
            perm = torch.arange(nc, device=w.device) ^ sv
            src = w[:, sv :: m + 1].reshape(s, -1, nc, 128)[:, :, perm, :]
            view[:, sv :: m + 1] = src.reshape(s, -1, k)
    return buf, view, sseg, sw, m


def routing_sets(gen):
    """NSETS disjoint pools of 39 experts; row r of a set: window 5r .. 5r+7 of its pool (8 rows
    cover all 39, as plan 65's R=8 count)."""
    pool = torch.randperm(E, generator=gen).tolist()
    sets = []
    for s in range(NSETS):
        p = pool[39 * s : 39 * s + 39]
        sets.append([[p[(5 * r + j) % 39] for j in range(TOPK)] for r in range(8)])
    return sets


# ------------------------------------------------------------------------------- (a) probe

PROBE_LAYOUTS = [  # name, pad, stagger, swz, krot
    ("base", 0, 0, False, False), ("p32", 32, 0, False, False), ("p64", 64, 0, False, False),
    ("p128", 128, 0, False, False), ("p256", 256, 0, False, False), ("st", 0, ST, False, False),
    ("p64st", 64, ST, False, False), ("swz", 0, 0, True, False), ("swzst", 0, ST, True, False),
    ("krot", 0, 0, False, True),
]


def part_a(dev) -> None:
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    gen = torch.Generator().manual_seed(72)
    sets = routing_sets(gen)
    res = {}
    # --- MoE gate_up pattern, R = 8 (64 pairs, 39 distinct)
    R = 8
    nt = I2 // 8
    distinct = 39
    nbytes = distinct * 2 * I2 * H * 2
    o = torch.empty(R * TOPK * nt, dtype=torch.float32, device=dev)
    for name, pad, stg, swz, krot in PROBE_LAYOUTS:
        try:
            sw = H + pad
            sseg = 2 * I2 * sw + stg
            w = torch.empty(E * sseg + 256, dtype=BF, device=dev)
            w.view(torch.int16).random_(-2000, 2000)
            segs = [torch.tensor([e * sseg for row in s for e in row], dtype=torch.int64, device=dev)
                    for s in sets]
            m = 15 if swz else 0
            ck = []

            def f(sg, w=w, sw=sw, m=m, krot=krot):
                ck.append(probe_kernel[(R * TOPK * nt,)](
                    w, sg, o, R * TOPK, NR=I2, NT=nt, SW=sw, K=H, BN=8, BK=1024, HALF=I2,
                    SWZM=m, ORDER=1, KROT=krot, num_warps=4))
            us = time_graph([lambda sg=sg: f(sg) for sg in segs])
            res[("moe_up", name)] = nbytes / us / 1e6
            out(f"RESULT a moe_up R=8 {name} pad_B={2 * pad} stagger_B={2 * stg} swz={int(swz)} "
                f"us={us:.1f} TBps={nbytes / us / 1e6:.3f} {asm_loads(ck[0])}")
            del w
        except Exception as e:  # noqa: BLE001
            out(f"RESULT a moe_up {name} FAIL {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
    # contiguous: same items, each tile read as one linear run (today's layout)
    try:
        sseg = 2 * I2 * H
        w = torch.empty(E * sseg, dtype=BF, device=dev)
        w.view(torch.int16).random_(-2000, 2000)
        segs = [torch.tensor([e * sseg for row in s for e in row], dtype=torch.int64, device=dev)
                for s in sets]
        us = time_graph([lambda sg=sg: probe_contig_kernel[(R * TOPK * nt,)](
            w, sg, o, R * TOPK, NT=nt, K=H, BN=8, BK=256, HALF=I2, ORDER=1, num_warps=4)
            for sg in segs])
        res[("moe_up", "contig")] = nbytes / us / 1e6
        out(f"RESULT a moe_up R=8 contig us={us:.1f} TBps={nbytes / us / 1e6:.3f}")
        # plain stream of the same byte count (39 experts' gate_up, contiguous), rotating
        ne = distinct * sseg
        so = torch.empty(cus * 8, dtype=torch.float32, device=dev)
        for g in (4, 8):
            us = time_graph([lambda s=s: stream_kernel[(cus * g,)](w[s * ne:], so, ne, BLOCK=4096,
                                                                   num_warps=8)
                             for s in range(NSETS)])
            res[("moe_up", f"stream{g}")] = nbytes / us / 1e6
            out(f"RESULT a moe_up stream grid={g}/CU us={us:.1f} TBps={nbytes / us / 1e6:.3f}")
        del w
    except Exception as e:  # noqa: BLE001
        out(f"RESULT a moe_up contig FAIL {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
    # --- dense MFMA tile pattern: in_proj (64 x 128 tiles) and q|k|v (32 x 128)
    for nm, (n, k), bn in (("in_proj", DENSE["in_proj"], 64), ("qkv", DENSE["qkv"], 32)):
        nbytes = n * k * 2
        nt = math.ceil(n / bn)
        o = torch.empty(nt, dtype=torch.float32, device=dev)
        for name, pad, stg, swz, krot in PROBE_LAYOUTS:
            if stg:
                continue
            try:
                sw = k + pad
                copies = max(1, math.ceil(256e6 / (n * sw * 2)))
                w = torch.empty(copies * n * sw + 256, dtype=BF, device=dev)
                w.view(torch.int16).random_(-2000, 2000)
                segs = [torch.tensor([c * n * sw], dtype=torch.int64, device=dev) for c in range(copies)]
                m = 15 if swz else 0
                ck = []

                def f(sg, w=w, sw=sw, m=m, krot=krot, n=n, k=k, bn=bn, nt=nt, o=o):
                    ck.append(probe_kernel[(nt,)](
                        w, sg, o, 1, NR=n, NT=nt, SW=sw, K=k, BN=bn, BK=128, HALF=0, SWZM=m,
                        ORDER=0, KROT=krot, num_warps=4))
                us = time_graph([lambda sg=sg: f(sg) for sg in segs])
                res[(nm, name)] = nbytes / us / 1e6
                out(f"RESULT a {nm} {name} pad_B={2 * pad} swz={int(swz)} copies={copies} us={us:.1f} "
                    f"TBps={nbytes / us / 1e6:.3f} {asm_loads(ck[0])}")
                del w
            except Exception as e:  # noqa: BLE001
                out(f"RESULT a {nm} {name} FAIL {type(e).__name__}: {str(e)[:300]}")
                traceback.print_exc()
        try:
            nf = (n // bn) * bn
            copies = max(1, math.ceil(256e6 / (n * k * 2)))
            w = torch.empty(copies * n * k, dtype=BF, device=dev)
            w.view(torch.int16).random_(-2000, 2000)
            segs = [torch.tensor([c * n * k], dtype=torch.int64, device=dev) for c in range(copies)]
            ntf = nf // bn
            us = time_graph([lambda sg=sg: probe_contig_kernel[(ntf,)](
                w, sg, o, 1, NT=ntf, K=k, BN=bn, BK=32, HALF=0, ORDER=0, num_warps=4) for sg in segs])
            b = nf * k * 2
            res[(nm, "contig")] = b / us / 1e6
            out(f"RESULT a {nm} contig (full tiles {nf} rows) us={us:.1f} TBps={b / us / 1e6:.3f}")
            so = torch.empty(cus * 8, dtype=torch.float32, device=dev)
            us = time_graph([lambda c=c: stream_kernel[(cus * 4,)](w[c * n * k:], so, n * k, BLOCK=4096,
                                                                    num_warps=8)
                             for c in range(copies)])
            res[(nm, "stream")] = nbytes / us / 1e6
            out(f"RESULT a {nm} stream us={us:.1f} TBps={nbytes / us / 1e6:.3f}")
            del w
        except Exception as e:  # noqa: BLE001
            out(f"RESULT a {nm} contig FAIL {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
    # --- summary: spread across pad values (0, 64, 128, 256, 512 B)
    for pat in ("moe_up", "in_proj", "qkv"):
        pads = [res.get((pat, n)) for n in ("base", "p32", "p64", "p128", "p256")]
        pads = [v for v in pads if v]
        if pads:
            spread = (max(pads) - min(pads)) / min(pads)
            best = max((v, n) for (p, n), v in res.items() if p == pat)
            out(f"RESULT a summary {pat} pad_TBps={[round(v, 3) for v in pads]} spread={spread * 100:.1f}% "
                f"best={best[1]} {best[0]:.3f} vs base {res.get((pat, 'base'), 0):.3f}")


# ------------------------------------------------------------------------------- (b) MoE

MOE_LAYOUTS = [  # name, up pad, down pad, stagger, swz
    ("base", 0, 0, 0, False), ("st", 0, 0, ST, False), ("p32", 32, 32, 0, False),
    ("p64", 64, 32, 0, False), ("p128", 128, 32, 0, False), ("p256", 256, 32, 0, False),
    ("p64d64", 64, 64, 0, False), ("p64st", 64, 32, ST, False), ("swz", 0, 0, 0, True),
    ("swzst", 0, 0, ST, True),
]


def part_moe(dev, only: str | None) -> None:
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    gen = torch.Generator().manual_seed(72)
    sets = routing_sets(gen)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=gen) * scale).to(BF).to(dev)

    gu = rnd(E, 2 * I2, H)   # [256, 512, 2048] 512 MB
    dn = rnd(E, H, I2)       # [256, 2048, 256] 256 MB
    sd = rnd(H, IS)
    out(f"RESULT moe setup device={dev} cus={cus} grid={fused.MOE_GRID}/CU tile_rows={fused.MOE_TILE_ROWS}")
    bn = fused.TILES["moe_bn"]
    data = {}
    for R in (2, 4, 8):
        x = rnd(R, H, scale=1.0)
        r = rnd(R, N3, scale=1.0)
        idxs = [torch.tensor(s[:R], dtype=torch.int32, device=dev) for s in sets]
        wts = []
        for _ in sets:
            w_ = torch.rand(R, TOPK, generator=gen)
            wts.append((w_ / w_.sum(1, keepdim=True)).to(torch.bfloat16).float().to(dev))
        distinct = [len({e for row in s[:R] for e in row}) for s in sets]
        order = 1 if R >= fused.MOE_TILE_ROWS else 0
        acts = [torch.empty(R * TOPK, I2, dtype=BF, device=dev) for _ in sets]
        outs = [torch.empty(R, H, dtype=BF, device=dev) for _ in sets]
        data[R] = dict(x=x, r=r, idxs=idxs, wts=wts, distinct=distinct, order=order, acts=acts, outs=outs)

    def grids(R, order):
        iu = R * TOPK * math.ceil(I2 / bn)
        idn = R * math.ceil(H / bn)
        return (iu if order else min(iu, fused.MOE_GRID * cus)), min(idn, fused.MOE_GRID * cus)

    def eng_up(R, s, act):
        d = data[R]
        gu_, _ = grids(R, d["order"])
        fused._moe_up_kernel[(gu_,)](
            d["x"], d["idxs"][s], gu, act, R, gu.stride(0), d["r"], d["wts"][s], N3, H=H, I=I2,
            TOPK=TOPK, BN=bn, BK=1024, ORDER=d["order"], E=E, TOPKP=8, FUSE_TOPK=False, num_warps=4)

    def eng_dn(R, s, act, o):
        d = data[R]
        _, gd = grids(R, d["order"])
        fused._moe_down_kernel[(gd,)](
            act, d["r"], d["idxs"][s], d["wts"][s], dn, sd, o, o, R, dn.stride(0), N3, H=H, I=I2,
            IS=IS, E=E, TOPK=TOPK, BN=bn, BKI=256, BKS=256, ORDER=d["order"], HAS_RES=False,
            num_warps=4)

    # reference outputs (engine kernels, today's layout), set 0
    ref = {}
    for R in (2, 4, 8):
        act = torch.empty(R * TOPK, I2, dtype=BF, device=dev)
        o = torch.empty(R, H, dtype=BF, device=dev)
        eng_up(R, 0, act)
        eng_dn(R, 0, act, o)
        torch.cuda.synchronize()
        ref[R] = (act.clone(), o.clone())

    def report(tag, R, us_up, us_dn, us_all):
        d = data[R]
        dist = statistics.mean(d["distinct"])
        b_up = dist * 2 * I2 * H * 2
        b_dn = dist * H * I2 * 2 + H * IS * 2
        out(f"RESULT moe {tag} R={R} distinct={dist:.1f} up_us={us_up:.1f} up_TBps={b_up / us_up / 1e6:.3f} "
            f"down_us={us_dn:.1f} down_TBps={b_dn / us_dn / 1e6:.3f} all_us={us_all:.1f} "
            f"all_TBps={(b_up + b_dn) / us_all / 1e6:.3f}")
        return us_all

    times = {}
    for R in (2, 4, 8):
        d = data[R]
        try:
            ns = len(sets)
            u = time_graph([lambda s=s: eng_up(R, s, d["acts"][s]) for s in range(ns)])
            dd = time_graph([lambda s=s: eng_dn(R, s, d["acts"][s], d["outs"][s]) for s in range(ns)])
            a = time_graph([lambda s=s: (eng_up(R, s, d["acts"][s]), eng_dn(R, s, d["acts"][s], d["outs"][s]))
                            for s in range(ns)])
            times[("engine", R)] = (u, dd, a)
            report("engine", R, u, dd, a)
        except Exception as e:  # noqa: BLE001
            out(f"RESULT moe engine R={R} FAIL {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()

    for name, pu, pd, stg, swz in MOE_LAYOUTS:
        if only and name not in only.split(","):
            continue
        try:
            bu, vu, seu, swu, mu = layout_rows(gu, pu, stg, swz)
            bd, vd, sed, swd, md = layout_rows(dn, pd, stg, swz)
            mem = (bu.numel() + bd.numel()) / (gu.numel() + dn.numel()) - 1
        except Exception as e:  # noqa: BLE001
            out(f"RESULT moe {name} LAYOUT FAIL {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
            continue
        ck = {}
        for R in (2, 4, 8):
            d = data[R]
            gu_, gd = grids(R, d["order"])

            def up(s, act, R=R, d=d, gu_=gu_):
                ck["up"] = moe_up_s[(gu_,)](
                    d["x"], d["idxs"][s], bu, act, R, seu, H=H, I=I2, TOPK=TOPK, BN=bn, BK=1024,
                    ORDER=d["order"], SW=swu, SWZM=mu, num_warps=4)

            def down(s, act, o, R=R, d=d, gd=gd):
                ck["dn"] = moe_down_s[(gd,)](
                    act, d["r"], d["idxs"][s], d["wts"][s], bd, sd, o, o, R, sed, N3, H=H, I=I2,
                    IS=IS, E=E, TOPK=TOPK, BN=bn, BKI=256, BKS=256, ORDER=d["order"],
                    HAS_RES=False, SW=swd, SWZM=md, num_warps=4)
            try:
                act = torch.empty(R * TOPK, I2, dtype=BF, device=dev)
                o = torch.empty(R, H, dtype=BF, device=dev)
                up(0, act)
                down(0, act, o)
                torch.cuda.synchronize()
                eq_a = torch.equal(act, ref[R][0])
                eq_o = torch.equal(o, ref[R][1])
                ns = len(sets)
                u = time_graph([lambda s=s: up(s, d["acts"][s]) for s in range(ns)])
                dd = time_graph([lambda s=s: down(s, d["acts"][s], d["outs"][s]) for s in range(ns)])
                a = time_graph([lambda s=s: (up(s, d["acts"][s]), down(s, d["acts"][s], d["outs"][s]))
                                for s in range(ns)])
                times[(name, R)] = (u, dd, a)
                report(f"{name} mem=+{mem * 100:.2f}% bitwise_act={int(eq_a)} bitwise_out={int(eq_o)}",
                       R, u, dd, a)
                if R == 8:
                    out(f"RESULT moe {name} asm up {asm_loads(ck.get('up'))} | down {asm_loads(ck.get('dn'))}")
            except Exception as e:  # noqa: BLE001
                out(f"RESULT moe {name} R={R} FAIL {type(e).__name__}: {str(e)[:300]}")
                traceback.print_exc()
        del bu, bd, vu, vd
        torch.cuda.empty_cache()

    for R in (2, 4, 8):
        if ("base", R) not in times:
            continue
        bu_, bd_, ba_ = times[("base", R)]
        rows = []
        for (nm, rr), (u, dd, a) in times.items():
            if rr == R and nm not in ("base", "engine"):
                rows.append((ba_ / a, nm, bu_ / u, bd_ / dd))
        rows.sort(reverse=True)
        eng = times.get(("engine", R))
        out(f"RESULT moe summary R={R} base_vs_engine_all={eng[2] / ba_ if eng else float('nan'):.3f} "
            + " ".join(f"{nm}:all={s:.3f}/up={su:.3f}/dn={sd_:.3f}" for s, nm, su, sd_ in rows))


# ------------------------------------------------------------------------------- (b) dense

DENSE_LAYOUTS = [("base", 0, False), ("p32", 32, False), ("p64", 64, False), ("p128", 128, False),
                 ("p256", 256, False), ("swz", 0, True), ("p64swz", 64, True)]


def part_dense(dev) -> None:
    gen = torch.Generator().manual_seed(72)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=gen) * scale).to(BF).to(dev)

    summary = {}
    for nm, (n, k) in DENSE.items():
        bn, bk, warps, ksplit = mfma.tiles_for(n, k)
        copies = max(1, math.ceil(256e6 / (n * k * 2)))
        ws = [rnd(n, k) for _ in range(copies)]
        nbytes = n * k * 2
        for R in (2, 8):
            x = rnd(R, k, scale=1.0)
            o = torch.empty(R, n, dtype=BF, device=dev)
            try:
                refo = mfma.gemv(x, ws[0])
                torch.cuda.synchronize()
                us = time_graph([lambda w=w: mfma.gemv(x, w, o) for w in ws])
                summary[(nm, R, "engine")] = us
                out(f"RESULT dense {nm} {n}x{k} R={R} engine tiles=({bn},{bk},{warps},{ksplit}) copies={copies} "
                    f"us={us:.1f} TBps={nbytes / us / 1e6:.3f}")
            except Exception as e:  # noqa: BLE001
                out(f"RESULT dense {nm} R={R} engine FAIL {type(e).__name__}: {str(e)[:300]}")
                traceback.print_exc()
                continue
            for lname, pad, swz in DENSE_LAYOUTS:
                try:
                    phys = [layout_rows(w[None], pad, 0, swz) for w in ws]
                    sw, m = phys[0][3], phys[0][4]
                    part = torch.empty((ksplit * mfma.BM * n,), dtype=torch.float32, device=dev)
                    ck = []

                    def g(buf, x=x, o=o, part=part, sw=sw, m=m, n=n, k=k, bn=bn, bk=bk, warps=warps,
                          ksplit=ksplit, R=R):
                        ck.append(mfma_gemv_s[(triton.cdiv(n, bn) * ksplit,)](
                            x, buf, o, part if ksplit > 1 else o, R, n, o.stride(0), K=k, BM=mfma.BM,
                            BN=bn, BK=bk, KSPLIT=ksplit, SW=sw, SWZM=m, num_warps=warps))
                        if ksplit > 1:
                            mfma._ksum_kernel[(triton.cdiv(R * n, 1024),)](
                                part, o, R, n, o.stride(0), KSPLIT=ksplit, BM=mfma.BM, BLOCK=1024,
                                num_warps=4)
                    g(phys[0][0])
                    torch.cuda.synchronize()
                    eq = torch.equal(o, refo)
                    us = time_graph([lambda b=p[0]: g(b) for p in phys])
                    summary[(nm, R, lname)] = us
                    out(f"RESULT dense {nm} R={R} {lname} pad_B={2 * pad} swz={int(swz)} bitwise={int(eq)} "
                        f"mem=+{pad / k * 100:.2f}% us={us:.1f} TBps={nbytes / us / 1e6:.3f} {asm_loads(ck[0])}")
                    del phys
                except Exception as e:  # noqa: BLE001
                    out(f"RESULT dense {nm} R={R} {lname} FAIL {type(e).__name__}: {str(e)[:300]}")
                    traceback.print_exc()
        del ws
        torch.cuda.empty_cache()
    for nm in DENSE:
        for R in (2, 8):
            b = summary.get((nm, R, "base"))
            if not b:
                continue
            row = sorted(((b / v, ln) for (n_, r_, ln), v in summary.items()
                          if n_ == nm and r_ == R and ln not in ("base", "engine")), reverse=True)
            e = summary.get((nm, R, "engine"), float("nan"))
            out(f"RESULT dense summary {nm} R={R} base_vs_engine={e / b:.3f} "
                + " ".join(f"{ln}={s:.3f}" for s, ln in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["a", "moe", "dense"])
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--only", default=None, help="moe: comma-separated layout names")
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.cuda.set_device(dev)
    out(f"RESULT env torch={torch.__version__} hip={torch.version.hip} triton={triton.__version__} "
        f"device={dev} name={torch.cuda.get_device_name(dev)} "
        f"cus={torch.cuda.get_device_properties(dev).multi_processor_count}")
    with torch.no_grad():
        if args.part == "a":
            part_a(torch.device("cuda", dev))
        elif args.part == "moe":
            part_moe(torch.device("cuda", dev), args.only)
        else:
            part_dense(torch.device("cuda", dev))
    out("RESULT done")


if __name__ == "__main__":
    main()
