"""Plan 31 step 0: do MFMA (``tl.dot``) decode kernels with a fixed 16-row tile make extra
verify rows nearly free, and does deep chained MTP speculation then pay under TP2?

    python tools/mfma_probe.py [--devices 0,1] [--sessions 16] [--gen 192] [--depth 7]
    TRITON_INTERPRET=1 python tools/mfma_probe.py --smoke     # CPU check of the new kernels

One 2-GCD replica is loaded as the server loads it (``Generator.load``, ``ENGINE_TP=2``, the
MTP head on), with ``ENGINE_FUSED_MAX_ROWS=16`` so today's fused layers also serve 16 rows
(their "pairs" row tiling is row-count invariant, so this changes no output). Tools only:
no engine file changes.

(b) Acceptance (first, it also records the inputs of (a)). ``--sessions`` replay-like
    prompts (the replay's synthetic coding lines, as plan 14's ``deep_mtp_probe.py`` builds
    them) and as many natural chat prompts are greedy-decoded for ``--gen`` tokens, 8 rows
    at a time, by an eager TP2 decode loop (both ranks in two threads, ``TPModel._both``)
    that keeps each step's final normed hidden. After every step the MTP head drafts a chain
    of ``--depth`` tokens from (hidden, pick), each next link from (the head's own normed
    output, the previous draft) one position on (plan 14's variant "normed", vLLM/SGLang
    multi-step MTP with one layer), through the engine's own TP2 head pieces
    (``_mtp_in``, ``_fused_layer(mtp_index)``, the sharded lm_head and ``_argmax``).
    Per set: per-position conditional acceptance ``p_j`` and ``E_k`` (tokens per step of a
    simulated k-draft decoder that accepts the longest correct prefix plus the target pick).
(a) Kernels, rank 0's TP2 shard, real weights, real routing. During the replay-like decode
    rank 0 records the input of ``--layers`` GDN in_proj GEMVs and MoE layers and the final
    normed hidden (lm_head input). Row sets mirror a verify step of 2 slots x (k+1)
    consecutive tokens: R = 2, 4, 8, 16 (k = 0, 1, 3, 7), ``--samples`` (step, slot pair)
    each. Kernels:
    - today: ``fused._gemv`` (pairs, 4 x 2048), ``FusedDecode.head`` (lm_head tiles), the MoE
      as ``FusedDecode._moe`` launches it under plan 22 (order 1 from 4 rows, 8 programs per
      CU), router/top-k from today's kernels;
    - mfma: ``_mfma_gemv`` (one 16-row tile, rows zero-padded, BN x BK tiles, K in a fixed
      order, ``tl.dot``); the routed experts grouped by expert: ``_mfma_moe_up`` (gate|up of
      every row routed to one expert in one 16-row tile per (expert, N tile), so each
      expert's shard is read once whatever the row count), ``_mfma_moe_down`` (routed down
      into per-pair BF16 parts, same grouping) and ``_mfma_moe_combine`` (today's weighted
      top-k sum in the same order and roundings, the shared expert's down as a 16-row
      ``tl.dot``, its sigmoid gate, the residual).
    Timed as CUDA graphs of back-to-back launches cycling over every (layer, sample) set.
    Effective TB/s over unique bytes (dense: the weight; MoE: distinct experts x (up + down)
    shard bytes + the shared down). Bitwise checks per config, on every R=16 set: a row's
    output equals its output in the R-row prefix runs (R = 1, 2, 4, 8) and after the rows
    are permuted (roll by 5, reverse) inside the tile; and, reported only, the share of
    elements equal to today's kernel and the max abs difference (dense also vs an fp64
    reference). Also: every dense weight of rank 0's step (in_proj, out_proj, q|k|v, o_proj,
    router|shared gate|up) timed as one graph per R, today vs the fixed MFMA config.
(c) Today's engine steps under TP2 graphs: decode of R unrelated slots (R = 2, 4, 8, 16) and
    verify of B slots (2B rows, B = 1, 2, 4, 8), median of ``--iters`` replays with every
    device synchronized and the GDN state restored outside the timed region. Projection,
    2 slots, k drafts, R = 2(k+1): step(R) = decode_today(R) - kernels_today(R) +
    kernels_mfma(R) (kernels = all dense GEMVs + 40 MoE layers + the lm_head shard), plus k
    chained head passes at 2 rows (each ~ one layer + the lm_head, from the same numbers);
    tokens/ms = 2 E_k / step. Also with a served host overhead per step added (the parent's
    TPOT p50 x its tokens/step - its graph-only verify step).

Prints ``RESULT`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENGINE_FUSED_MAX_ROWS", "16")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.fast import fused  # noqa: E402
from engine.fast import kernels as _kernels  # noqa: E402

triton, tl = _kernels.triton, _kernels.tl
BM = 16  # the fixed row tile
ROWS = (2, 4, 8, 16)

if triton is not None:

    @triton.jit(do_not_specialize=["R", "N"])
    def _mfma_gemv(
        x_ptr, w_ptr, out_ptr, R, N, stride_out,
        K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        DOT_F32: tl.constexpr,
    ):
        """out[r, n] = bf16(x[r, :] . W[n, :]) for R <= BM rows in one zero-padded BM-row tile;
        one program per BN outputs, K in BK chunks in a fixed order (``tl.dot``)."""
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        mok = rm < R
        rn = pid * BN + tl.arange(0, BN)
        nok = rn < N
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            kok = rk < K
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None] & kok[None, :], other=0.0)
            b = tl.load(w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None],
                        mask=nok[None, :] & kok[:, None], other=0.0)
            if DOT_F32:
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
            else:
                acc = tl.dot(a, b, acc)
        tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16),
                 mask=mok[:, None] & nok[None, :])

    @triton.jit
    def _group(idx_ptr, lead, P, PP: tl.constexpr, BM: tl.constexpr, TOPK: tl.constexpr):
        """The pairs routed to pair ``lead``'s expert, in pair order, as BM tile rows; whether
        ``lead`` is the first of them (the group's leader)."""
        pp = tl.arange(0, PP)
        pok = pp < P
        e_all = tl.load(idx_ptr + pp, mask=pok, other=-1)
        e = tl.load(idx_ptr + lead)
        same = (e_all == e) & pok
        first = tl.min(tl.where(same, pp, PP), axis=0)
        rank = tl.cumsum(same.to(tl.int32), axis=0) - 1
        mm = tl.arange(0, BM)
        sel = same[None, :] & (rank[None, :] == mm[:, None])
        pair = tl.sum(tl.where(sel, pp[None, :], 0), axis=1)
        ok = tl.sum(sel.to(tl.int32), axis=1) > 0
        return e, first == lead, pair, ok

    @triton.jit(do_not_specialize=["R"])
    def _mfma_moe_up(
        h_ptr, idx_ptr, w_ptr, act_ptr, R, stride_we,
        H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, PP: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT_F32: tl.constexpr,
    ):
        """act[p, n] = bf16(silu(bf16(Wg[e, n] . h_row)) * bf16(Wu[e, n] . h_row)) with every
        pair of expert e in one BM-row tile: program (leader pair, N tile)."""
        lead = tl.program_id(0)
        tile = tl.program_id(1)
        P = R * TOPK
        e, is_lead, pair, ok = _group(idx_ptr, lead, P, PP, BM, TOPK)
        if is_lead:
            row = pair // TOPK
            rn = tile * BN + tl.arange(0, BN)
            wbase = w_ptr + e.to(tl.int64) * stride_we
            acc_g = tl.zeros([BM, BN], dtype=tl.float32)
            acc_u = tl.zeros([BM, BN], dtype=tl.float32)
            for k0 in range(0, H, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(h_ptr + row[:, None] * H + rk[None, :], mask=ok[:, None], other=0.0)
                bg = tl.load(wbase + rn[None, :] * H + rk[:, None])
                bu = tl.load(wbase + (rn[None, :] + I) * H + rk[:, None])
                if DOT_F32:
                    acc_g = tl.dot(a.to(tl.float32), bg.to(tl.float32), acc_g)
                    acc_u = tl.dot(a.to(tl.float32), bu.to(tl.float32), acc_u)
                else:
                    acc_g = tl.dot(a, bg, acc_g)
                    acc_u = tl.dot(a, bu, acc_u)
            g = acc_g.to(tl.bfloat16).to(tl.float32)
            u = acc_u.to(tl.bfloat16).to(tl.float32)
            s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
            tl.store(act_ptr + pair[:, None] * I + rn[None, :], (s * u).to(tl.bfloat16), mask=ok[:, None])

    @triton.jit(do_not_specialize=["R"])
    def _mfma_moe_down(
        act_ptr, idx_ptr, w_ptr, part_ptr, R, stride_we,
        H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, PP: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT_F32: tl.constexpr,
    ):
        """part[p, n] = bf16(Wd[e, n, :] . act[p]) with every pair of expert e in one BM-row
        tile: program (leader pair, H tile)."""
        lead = tl.program_id(0)
        tile = tl.program_id(1)
        P = R * TOPK
        e, is_lead, pair, ok = _group(idx_ptr, lead, P, PP, BM, TOPK)
        if is_lead:
            rn = tile * BN + tl.arange(0, BN)
            wbase = w_ptr + e.to(tl.int64) * stride_we
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k0 in range(0, I, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(act_ptr + pair[:, None] * I + rk[None, :], mask=ok[:, None], other=0.0)
                b = tl.load(wbase + rn[None, :] * I + rk[:, None])
                if DOT_F32:
                    acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
                else:
                    acc = tl.dot(a, b, acc)
            tl.store(part_ptr + pair[:, None] * H + rn[None, :], acc.to(tl.bfloat16), mask=ok[:, None])

    @triton.jit(do_not_specialize=["R"])
    def _mfma_moe_combine(
        part_ptr, r_ptr, wt_ptr, sd_ptr, res_ptr, out_ptr, R, N3,
        H: tl.constexpr, IS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BKS: tl.constexpr, DOT_F32: tl.constexpr,
    ):
        """``fused._moe_down_kernel``'s epilogue math for BM rows of one H tile: the weighted
        top-k sum of the BF16 parts in top-k order, the shared expert's down (a BM-row
        ``tl.dot``), its sigmoid gate and the residual, each rounding as there."""
        tile = tl.program_id(0)
        mm = tl.arange(0, BM)
        mok = mm < R
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for j in tl.static_range(TOPK):
            pair = mm * TOPK + j
            w = tl.load(wt_ptr + pair, mask=mok, other=0.0)
            p = tl.load(part_ptr + pair[:, None] * H + rn[None, :], mask=mok[:, None], other=0.0).to(tl.float32)
            acc += (p * w[:, None]).to(tl.bfloat16).to(tl.float32)
        routed = acc.to(tl.bfloat16).to(tl.float32)
        shared = tl.zeros([BM, BN], dtype=tl.float32)
        rbase = r_ptr + mm[:, None] * N3 + E
        for k0 in range(0, IS, BKS):
            sc = k0 + tl.arange(0, BKS)
            g = tl.load(rbase + sc[None, :], mask=mok[:, None], other=0.0).to(tl.float32)
            u = tl.load(rbase + IS + sc[None, :], mask=mok[:, None], other=0.0).to(tl.float32)
            sa = ((g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16)
            sw = tl.load(sd_ptr + rn[None, :] * IS + sc[:, None])
            if DOT_F32:
                shared = tl.dot(sa.to(tl.float32), sw.to(tl.float32), shared)
            else:
                shared = tl.dot(sa, sw, shared)
        shared = shared.to(tl.bfloat16).to(tl.float32)
        sg = tl.load(r_ptr + mm * N3 + E + 2 * IS, mask=mok, other=0.0).to(tl.float32)
        sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
        moe = (routed + (sg[:, None] * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        res = tl.load(res_ptr + mm[:, None] * H + rn[None, :], mask=mok[:, None], other=0.0).to(tl.float32)
        tl.store(out_ptr + mm[:, None] * H + rn[None, :], (res + moe).to(tl.bfloat16), mask=mok[:, None])


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def p2(n: int) -> int:
    return triton.next_power_of_2(n)


# ============================================================================ launchers


def gemv_today(fd, x, w, head=False):
    return fd.head(x, w) if head else fd._gemv(x, w)


def gemv_mfma(x, w, cfg, outt=None):
    bn, bk, warps = cfg
    R, K = x.shape
    N = w.shape[0]
    assert R <= BM and x.is_contiguous()
    if outt is None:
        outt = torch.empty((R, N), dtype=x.dtype, device=x.device)
    _mfma_gemv[(triton.cdiv(N, bn),)](
        x, w, outt, R, N, outt.stride(0), K=K, BM=BM, BN=bn, BK=min(bk, p2(K)),
        DOT_F32=_kernels.INTERPRET, num_warps=warps,
    )
    return outt


class MoELayer:
    """One MoE layer's (shard) weights: views of the engine's tensors, or random in --smoke."""

    def __init__(self, name, gu, dn, sd, w3):
        self.name, self.gu, self.dn, self.sd, self.w3 = name, gu, dn, sd, w3
        self.E, two_i, self.H = gu.shape
        self.I = two_i // 2
        self.IS = sd.shape[1]
        self.N3 = w3.shape[0]


class MoESet:
    """R rows of one layer: h (MoE input), x (residual), router out r, top-k idx/wt."""

    def __init__(self, layer: MoELayer, h, x, r, topk):
        self.layer, self.h, self.x, self.r, self.topk = layer, h.contiguous(), x.contiguous(), r, topk
        self.R = h.shape[0]
        dev = h.device
        self.idx = torch.empty((self.R, topk), dtype=torch.int32, device=dev)
        self.wt = torch.empty((self.R, topk), dtype=torch.float32, device=dev)
        fused._topk_kernel[(self.R,)](r, self.idx, self.wt, layer.N3, E=layer.E, TOPK=topk,
                                      TOPKP=p2(topk), num_warps=1)
        self.act = torch.empty((self.R * topk, layer.I), dtype=h.dtype, device=dev)
        self.act_m = torch.empty((self.R * topk, layer.I), dtype=h.dtype, device=dev)
        self.part = torch.empty((self.R * topk, layer.H), dtype=h.dtype, device=dev)
        self.out = torch.empty_like(self.x)
        self.nu = len(set(self.idx.flatten().tolist()))

    def rows(self, sel: list[int]) -> "MoESet":
        t = torch.tensor(sel, device=self.h.device)
        return MoESet(self.layer, self.h[t], self.x[t], self.r[t].contiguous(), self.topk)


def moe_today(s: MoESet, cus: int):
    """``FusedDecode._moe``'s routed up + down launches (plan 22 order and grid)."""
    L = s.layer
    bn = fused.TILES["moe_bn"]
    new = fused.MOE_ORDER == 1
    order = 1 if new and s.R >= fused.MOE_TILE_ROWS else 0
    grid = fused.MOE_GRID if new else 2
    items = s.R * s.topk * triton.cdiv(L.I, bn)
    fused._moe_up_kernel[(items if order else min(items, grid * cus),)](
        s.h, s.idx, L.gu, s.act, s.R, L.gu.stride(0),
        H=L.H, I=L.I, TOPK=s.topk, BN=bn, BK=min(1024, p2(L.H)), ORDER=order, num_warps=4,
    )
    items = s.R * triton.cdiv(L.H, bn)
    fused._moe_down_kernel[(min(items, grid * cus),)](
        s.act, s.r, s.idx, s.wt, L.dn, L.sd, s.x, s.out, s.R, L.dn.stride(0), L.N3,
        H=L.H, I=L.I, IS=L.IS, E=L.E, TOPK=s.topk, BN=bn,
        BKI=min(1024, p2(L.I)), BKS=min(1024, p2(L.IS)), ORDER=order, num_warps=4,
    )
    return s.out


def moe_mfma_up(s: MoESet, cfg):
    bn, bk, warps = cfg
    L = s.layer
    _mfma_moe_up[(s.R * s.topk, triton.cdiv(L.I, bn))](
        s.h, s.idx, L.gu, s.act_m, s.R, L.gu.stride(0), H=L.H, I=L.I, TOPK=s.topk,
        PP=p2(BM * s.topk), BM=BM, BN=bn, BK=min(bk, p2(L.H)), DOT_F32=_kernels.INTERPRET,
        num_warps=warps,
    )
    return s.act_m


def moe_mfma_down(s: MoESet, cfg, act=None):
    bn, bk, warps = cfg
    L = s.layer
    act = s.act_m if act is None else act
    _mfma_moe_down[(s.R * s.topk, triton.cdiv(L.H, bn))](
        act, s.idx, L.dn, s.part, s.R, L.dn.stride(0), H=L.H, I=L.I, TOPK=s.topk,
        PP=p2(BM * s.topk), BM=BM, BN=bn, BK=min(bk, p2(L.I)), DOT_F32=_kernels.INTERPRET,
        num_warps=warps,
    )
    return s.part


COMBINE = (16, 64, 4)


def moe_mfma_combine(s: MoESet, cfg=COMBINE):
    bn, bks, warps = cfg
    L = s.layer
    bn = min(bn, p2(L.H))
    _mfma_moe_combine[(triton.cdiv(L.H, bn),)](
        s.part, s.r, s.wt, L.sd, s.x, s.out, s.R, L.N3, H=L.H, IS=L.IS, E=L.E, TOPK=s.topk,
        BM=BM, BN=bn, BKS=min(bks, p2(L.IS)), DOT_F32=_kernels.INTERPRET, num_warps=warps,
    )
    return s.out


def moe_mfma(s: MoESet, ucfg, dcfg):
    moe_mfma_up(s, ucfg)
    moe_mfma_down(s, dcfg)
    return moe_mfma_combine(s)


# ============================================================================ timing


def time_us(fns, min_launches: int = 32, reps: int = 5) -> float:
    """us per call of a list of launch closures, captured back to back as one CUDA graph."""
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


# ============================================================================ checks


PERMS = ("roll5", "reverse")


def perm_of(name: str, n: int) -> list[int]:
    if name == "roll5":
        return [(i + 5) % n for i in range(n)]
    return list(range(n - 1, -1, -1))


def dense_checks(x16, w, run):
    """Row-count (prefix R) and row-position (permutation) invariance of ``run(x) -> out``."""
    ref = run(x16).clone()
    n = x16.shape[0]
    res = {}
    for R in (1, 2, 4, 8):
        if R < n:
            res[f"prefix{R}"] = bool(torch.equal(run(x16[:R].contiguous()), ref[:R]))
    for name in PERMS:
        p = perm_of(name, n)
        t = torch.tensor(p, device=x16.device)
        res[name] = bool(torch.equal(run(x16[t].contiguous()), ref[t]))
    return ref, res


def moe_checks(s16: MoESet, run):
    ref = run(s16).clone()
    n = s16.R
    res = {}
    for R in (1, 2, 4, 8):
        if R < n:
            sub = s16.rows(list(range(R)))
            res[f"prefix{R}"] = bool(torch.equal(run(sub), ref[:R]))
    for name in PERMS:
        p = perm_of(name, n)
        sub = s16.rows(p)
        res[name] = bool(torch.equal(run(sub), ref[torch.tensor(p, device=ref.device)]))
    return ref, res


def diff(a, b) -> dict:
    return {"equal_share": round(float((a == b).float().mean()), 4),
            "max_abs": float((a.float() - b.float()).abs().max())}


# ============================================================================ (a) kernels


DENSE_CFGS = [(16, 128, 4), (16, 256, 4), (32, 128, 4), (32, 256, 4), (64, 128, 4), (64, 128, 8),
              (64, 256, 8)]
UP_CFGS = [(16, 128, 4), (16, 256, 4), (32, 128, 4), (32, 128, 8), (32, 256, 8), (64, 128, 8)]
DN_CFGS = [(16, 64, 4), (16, 128, 4), (32, 64, 4), (32, 128, 4), (64, 64, 4), (64, 128, 8)]


def bench_dense(label, fd, sets_by_r, cfgs, head, timing, cus):
    """sets_by_r: R -> list of (x [R, K], w [N, K]). Returns (best fixed cfg, {R: (today_us,
    mfma_us)}) for the cfg with the lowest R=16 time among the bitwise-invariant ones."""
    nbytes = statistics.mean(w.numel() * w.element_size() for _, w in sets_by_r[max(sets_by_r)])
    checks = {}
    for cfg in cfgs:
        ok, eq_today, err = True, [], []
        try:
            for x16, w in sets_by_r[16]:
                ref, res = dense_checks(x16, w, lambda x, w=w, cfg=cfg: gemv_mfma(x, w, cfg))
                ok = ok and all(res.values())
                eq_today.append(diff(ref, gemv_today(fd, x16, w, head)))
                if not head:
                    exact = (x16.double() @ w.double().T)
                    err.append((float((ref.double() - exact).abs().max()),
                                float((gemv_today(fd, x16, w, head).double() - exact).abs().max())))
            checks[cfg] = ok
            line = dict(kernel=label, cfg=cfg, invariant=ok, checks=res,
                        vs_today_equal_share=round(statistics.mean(d["equal_share"] for d in eq_today), 4),
                        vs_today_max_abs=max(d["max_abs"] for d in eq_today))
            if err:
                line["fp64_max_abs_mfma"] = max(e[0] for e in err)
                line["fp64_max_abs_today"] = max(e[1] for e in err)
            out("bitwise", **line)
        except Exception as exc:  # noqa: BLE001
            checks[cfg] = None
            out("error", kernel=label, cfg=cfg, error=f"{type(exc).__name__}: {str(exc)[:400]}")
    if not timing:
        return None, {}
    table = {}
    for R, sets in sets_by_r.items():
        t_today = time_us([lambda x=x, w=w: gemv_today(fd, x, w, head) for x, w in sets])
        table[(R, "today")] = t_today
        out("t", kernel=label, R=R, cfg="today", us=round(t_today, 1),
            TBps_unique=round(nbytes / t_today / 1e6, 3))
        for cfg in cfgs:
            if checks.get(cfg) is None:
                continue
            outs = {id(x): torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device) for x, w in sets}
            us = time_us([lambda x=x, w=w, cfg=cfg: gemv_mfma(x, w, cfg, outs[id(x)]) for x, w in sets])
            table[(R, cfg)] = us
            out("t", kernel=label, R=R, cfg=cfg, us=round(us, 1), TBps_unique=round(nbytes / us / 1e6, 3),
                invariant=checks[cfg])
    good = [c for c in cfgs if checks.get(c)] or [c for c in cfgs if checks.get(c) is not None]
    best = min(good, key=lambda c: table.get((16, c), float("inf")))
    res = {R: (table[(R, "today")], table[(R, best)]) for R in sets_by_r}
    out("pick", kernel=label, cfg=best, invariant=checks.get(best), mbytes=round(nbytes / 1e6, 2),
        by_R={R: {"today_us": round(a, 1), "mfma_us": round(b, 1), "today_TBps": round(nbytes / a / 1e6, 3),
                  "mfma_TBps": round(nbytes / b / 1e6, 3), "speedup": round(a / b, 3)} for R, (a, b) in res.items()})
    return best, res


def bench_moe(sets_by_r, timing, cus):
    any_set = sets_by_r[max(sets_by_r)][0]
    L0 = any_set.layer
    up_b = 2 * L0.I * L0.H * 2
    dn_b = L0.H * L0.I * 2
    sd_b = L0.H * L0.IS * 2

    def ubytes(sets, kind):
        uq = statistics.mean(s.nu for s in sets)
        return {"up": uq * up_b, "dn": uq * dn_b + sd_b, "both": uq * (up_b + dn_b) + sd_b}[kind]

    for R, sets in sets_by_r.items():
        nus = [s.nu for s in sets]
        out("routing", R=R, sets=len(sets), pairs=R * any_set.topk, distinct_mean=round(statistics.mean(nus), 2),
            distinct_min=min(nus), distinct_max=max(nus), shard_I=L0.I, shard_IS=L0.IS,
            unique_MB_mean=round(ubytes(sets, "both") / 1e6, 2))
    # today's outputs (and act) per set
    for sets in sets_by_r.values():
        for s in sets:
            moe_today(s, cus)
            s.ref_act = s.act.clone()
            s.ref_out = s.out.clone()

    # bitwise: up configs (act) and down configs (full output, with the first up config)
    upok, dnok = {}, {}
    for cfg in UP_CFGS:
        try:
            ok, eq = True, []
            for s in sets_by_r[16]:
                # act rows are (row, k) pairs: compare per row as [R, top_k * I]
                ref, res = moe_checks(s, lambda t, cfg=cfg: moe_mfma_up(t, cfg).view(t.R, -1).clone())
                ok = ok and all(res.values())
                eq.append(diff(ref, s.ref_act.view(s.R, -1)))
            upok[cfg] = ok
            out("bitwise", kernel="moe_up", cfg=cfg, invariant=ok, checks=res,
                vs_today_act_equal_share=round(statistics.mean(d["equal_share"] for d in eq), 4),
                vs_today_act_max_abs=max(d["max_abs"] for d in eq))
        except Exception as exc:  # noqa: BLE001
            upok[cfg] = None
            out("error", kernel="moe_up", cfg=cfg, error=f"{type(exc).__name__}: {str(exc)[:400]}")
    u0 = next((c for c in UP_CFGS if upok.get(c)), None) or next((c for c in UP_CFGS if upok.get(c) is not None), None)
    if u0 is None:
        out("fatal_moe", why="no MFMA up config ran")
        return None
    for cfg in DN_CFGS:
        try:
            ok, eq = True, []
            for s in sets_by_r[16]:
                ref, res = moe_checks(s, lambda t, cfg=cfg: moe_mfma(t, u0, cfg).clone())
                ok = ok and all(res.values())
                eq.append(diff(ref, s.ref_out))
            dnok[cfg] = ok
            out("bitwise", kernel="moe_full(up+down+combine)", up=u0, cfg=cfg, invariant=ok, checks=res,
                vs_today_out_equal_share=round(statistics.mean(d["equal_share"] for d in eq), 4),
                vs_today_out_max_abs=max(d["max_abs"] for d in eq))
        except Exception as exc:  # noqa: BLE001
            dnok[cfg] = None
            out("error", kernel="moe_down", cfg=cfg, error=f"{type(exc).__name__}: {str(exc)[:400]}")
    if not timing:
        return None
    tu, td = {}, {}
    for R, sets in sets_by_r.items():
        us = time_us([lambda s=s: moe_today(s, cus) for s in sets])
        tu[(R, "today")] = us
        out("t", kernel="moe_up+down", R=R, cfg="today", us=round(us, 1),
            TBps_unique=round(ubytes(sets, "both") / us / 1e6, 3))
        for cfg in UP_CFGS:
            if upok.get(cfg) is None:
                continue
            us = time_us([lambda s=s, cfg=cfg: moe_mfma_up(s, cfg) for s in sets])
            tu[(R, cfg)] = us
            out("t", kernel="moe_up", R=R, cfg=cfg, us=round(us, 1),
                TBps_unique=round(ubytes(sets, "up") / us / 1e6, 3), invariant=upok[cfg])
        for cfg in DN_CFGS:
            if dnok.get(cfg) is None:
                continue
            us = time_us([lambda s=s, cfg=cfg: moe_mfma_down(s, cfg, act=s.ref_act) for s in sets])
            td[(R, cfg)] = us
            out("t", kernel="moe_down", R=R, cfg=cfg, us=round(us, 1),
                TBps_unique=round((ubytes(sets, "dn") - sd_b) / us / 1e6, 3), invariant=dnok[cfg])
        us = time_us([lambda s=s: moe_mfma_combine(s) for s in sets])
        out("t", kernel="moe_combine", R=R, cfg=COMBINE, us=round(us, 1))

    def pick(table, oks, cfgs):
        good = [c for c in cfgs if oks.get(c)] or [c for c in cfgs if oks.get(c) is not None]
        return min(good, key=lambda c: table.get((16, c), float("inf")))

    ub, db = pick(tu, upok, UP_CFGS), pick(td, dnok, DN_CFGS)
    res = {}
    for R, sets in sets_by_r.items():
        a = tu[(R, "today")]
        b = time_us([lambda s=s: moe_mfma(s, ub, db) for s in sets])
        res[R] = (a, b)
    inv = bool(upok.get(ub)) and bool(dnok.get(db))
    out("pick", kernel="moe (routed up+down, shared down, combine)", up=ub, down=db, combine=COMBINE,
        invariant=inv,
        by_R={R: {"today_us": round(a, 1), "mfma_us": round(b, 1),
                  "today_TBps": round(ubytes(sets_by_r[R], "both") / a / 1e6, 3),
                  "mfma_TBps": round(ubytes(sets_by_r[R], "both") / b / 1e6, 3),
                  "speedup": round(a / b, 3)} for R, (a, b) in res.items()})
    tb16 = ubytes(sets_by_r[16], "both") / res[16][1] / 1e6
    out("kill_i", mfma_moe_TBps_R16=round(tb16, 3), invariant=inv,
        verdict="GO" if (tb16 >= 0.9 and inv) else "KILL")
    return {"up": ub, "down": db, "res": res, "invariant": inv}


# ============================================================================ smoke


def smoke() -> None:
    """CPU (TRITON_INTERPRET=1): small shapes, random weights; MFMA vs torch and invariance."""
    torch.manual_seed(0)
    dt = torch.bfloat16
    N, K = 40, 64
    w = (torch.randn(N, K) * K ** -0.5).to(dt)
    x = torch.randn(16, K).to(dt)
    cfg = (16, 32, 4)
    y = gemv_mfma(x, w, cfg)
    exact = (x.float() @ w.float().T)
    out("smoke_dense", max_abs_vs_fp32=float((y.float() - exact).abs().max()))
    _, res = dense_checks(x, w, lambda t: gemv_mfma(t, w, cfg))
    out("smoke_dense_invariance", **res)

    E, H, I, IS, topk = 16, 64, 32, 32, 4
    gu = (torch.randn(E, 2 * I, H) * H ** -0.5).to(dt)
    dn = (torch.randn(E, H, I) * I ** -0.5).to(dt)
    sd = (torch.randn(H, IS) * IS ** -0.5).to(dt)
    w3 = (torch.randn(E + 2 * IS + 1, H) * H ** -0.5).to(dt)
    L = MoELayer("L0", gu, dn, sd, w3)
    h = torch.randn(16, H).to(dt)
    xr = torch.randn(16, H).to(dt)
    r = (h.float() @ w3.float().T).to(dt)
    r[1::2, :E] = r[0, :E]  # force shared experts
    s = MoESet(L, h, xr, r.contiguous(), topk)
    moe_today(s, cus=4)
    ucfg, dcfg = (16, 32, 4), (16, 32, 4)
    moe_mfma_up(s, ucfg)
    out("smoke_moe_act", max_abs_vs_today=float((s.act_m.float() - s.act.float()).abs().max()),
        today_absmax=float(s.act.float().abs().max()))
    o = moe_mfma(s, ucfg, dcfg).clone()
    out("smoke_moe_out", max_abs_vs_today=float((o.float() - s.out.float()).abs().max()),
        today_absmax=float((s.out.float() - xr.float()).abs().max()), distinct=s.nu)
    _, res = moe_checks(s, lambda t: moe_mfma(t, ucfg, dcfg).clone())
    out("smoke_moe_invariance", **res)

    # the real run's check code paths (no timing), small configs
    global DENSE_CFGS, UP_CFGS, DN_CFGS
    DENSE_CFGS, UP_CFGS, DN_CFGS = [(16, 32, 4)], [(16, 32, 4)], [(16, 32, 4), (32, 16, 4)]
    fd = fused.FusedDecode.__new__(fused.FusedDecode)
    fd.max_rows, fd.cus = 16, {}
    xs = {R: [(torch.randn(R, K).to(dt), w)] for R in ROWS}
    bench_dense("smoke_dense", fd, xs, DENSE_CFGS, False, False, 4)
    bench_dense("smoke_head", fd, xs, DENSE_CFGS, True, False, 4)
    bench_moe({16: [s], 2: [s.rows([0, 1])]}, False, 4)


# ============================================================================ the run


NATURAL = [
    "Write a Python function that parses an ISO-8601 timestamp string into a timezone-aware "
    "datetime, with input validation, a docstring and three doctest examples.",
    "Implement a thread-safe LRU cache class in Python with get, put and a max size, and "
    "briefly explain the design choices.",
    "Here is a buggy function:\n\n```python\ndef mean(xs):\n    total = 0\n    for i in range(1, len(xs)):\n"
    "        total += xs[i]\n    return total / len(xs)\n```\n\nFind the bugs, fix them, and add "
    "unit tests with pytest.",
    "Explain the difference between TCP and UDP to a new backend engineer, with concrete "
    "examples of when to use each.",
    "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
    "Write a Go HTTP handler that accepts a JSON body with a list of integers and returns their "
    "median, with error handling and a table-driven test.",
    "Explain Rust's ownership and borrowing rules with three small code examples, including one "
    "that does not compile and why.",
    "Write a SQL query that finds, for each customer, their three most recent orders and the "
    "running total of their spend, and explain how the window functions work.",
    "Refactor this JavaScript into modern async/await style and explain the changes:\n\n"
    "function load(url, cb) {\n  fetch(url).then(function (r) { return r.json(); })\n"
    "    .then(function (d) { cb(null, d); })\n    .catch(function (e) { cb(e); });\n}",
    "Describe how a B-tree index speeds up lookups in a relational database, and when an index "
    "can make writes slower.",
    "Write a bash script that finds the ten largest files under a directory, prints their sizes "
    "in human-readable form, and skips hidden directories.",
    "Summarize the causes and consequences of the 2008 financial crisis in about 250 words.",
    "Implement Dijkstra's shortest-path algorithm in C++ with a priority queue, and state its "
    "time complexity.",
    "Write a unit-tested Python class for a bank account with deposits, withdrawals, an "
    "overdraft limit, and a transaction history.",
    "Explain gradient descent and the role of the learning rate to a high-school student, "
    "using an analogy.",
    "Write a Dockerfile for a small Flask app with a non-root user, a health check and a "
    "multi-stage build, then explain each stage.",
]


def real(args) -> None:
    os.environ.setdefault("ENGINE_MTP", "1")
    os.environ.setdefault("ENGINE_TP", "2")
    os.environ.setdefault("ENGINE_SLOTS", "40")
    os.environ.setdefault("ENGINE_KV_FRACTION", "0.3")
    from engine.fast.model import _rms
    from engine.generator import CHECKPOINT, Generator

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    tok = g.tokenizer
    if not hasattr(fast, "ranks"):
        raise SystemExit("needs ENGINE_TP=2 (TPModel)")
    r0, r1 = fast.ranks
    if r0.mtp is None or r0.fused is None:
        raise SystemExit(f"needs the MTP head and the fused layers: fused_error={r0.fused_error}")
    dev0 = r0.embed.device
    props = torch.cuda.get_device_properties(dev0)
    cus = props.multi_processor_count
    out("env", load_s=round(time.perf_counter() - started), device=props.name,
        gcn=getattr(props, "gcnArchName", "?"), cus=cus, torch=torch.__version__, triton=triton.__version__,
        visible=os.environ.get("HIP_VISIBLE_DEVICES"), gpus=torch.cuda.device_count(),
        max_rows=r0.fused.max_rows, graphs=r0.use_graphs, graph_error=fast.graph_error,
        moe_order=fused.MOE_ORDER, moe_grid=fused.MOE_GRID, row_tile=fused.ROW_TILE,
        slots=r0.slots, kv_pages=r0.pool.pages)
    n_layers = len(r0.layers)
    gdn_layers = [i for i in range(n_layers) if r0.layer_types[i] == "linear_attention"]
    pick_gdn = [gdn_layers[round(j * (len(gdn_layers) - 1) / max(1, args.layers - 1))] for j in range(args.layers)]
    pick_moe = [round(j * (n_layers - 1) / max(1, args.layers - 1)) for j in range(args.layers)]
    pick_gdn, pick_moe = list(dict.fromkeys(pick_gdn)), list(dict.fromkeys(pick_moe))
    out("layers", gdn=pick_gdn, moe=pick_moe)
    eos = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                       tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(t, int) and t >= 0}

    def prefill(slot: int, ids: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(ids) - 1
        position = 0
        while position < body:
            end = min((position // args.chunk + 1) * args.chunk, body)
            fast.prefill(slot, ids[position:end], position, ids[end])
            position = end
        return fast.prefill(slot, ids[body:], body)

    S = args.sessions
    pool: list[int] = []
    index = 0
    while len(pool) < S * args.stride + args.first:
        for line in (f"def case_{index}(items):", f"    return sorted(set(items))  # {index} 0f3a9c"):
            pool.extend(tok.encode(line, add_special_tokens=False))
        index += 1
    replay = [pool[s * args.stride : s * args.stride + args.first] for s in range(S)]
    natural = []
    for s in range(S):
        rendered = tok.apply_chat_template([{"role": "user", "content": NATURAL[s % len(NATURAL)]}],
                                           add_generation_prompt=True, tokenize=False, enable_thinking=False)
        natural.append(tok.encode(rendered, add_special_tokens=False))
    sets = {"replay": replay, "natural": natural}

    # ---------------------------------------------------------------- recorders (rank 0)
    rec = {"on": False, "gdn": {i: [] for i in pick_gdn}, "moe": {i: [] for i in pick_moe}, "hid": []}
    fd0 = r0.fused
    gdn0, moe0 = fd0._gdn, fd0._moe

    def gdn_rec(i, h, states):
        if rec["on"] and i in rec["gdn"]:
            rec["gdn"][i].append(h.detach().clone())
        return gdn0(i, h, states)

    def moe_rec(i, x, h):
        if rec["on"] and i in rec["moe"]:
            rec["moe"][i].append((h.detach().clone(), x.detach().clone()))
        return moe0(i, x, h)

    fd0._gdn, fd0._moe = gdn_rec, moe_rec
    depth = args.depth

    def step(model, n: int, record: bool):
        """One eager decode step of ``n`` rows (inputs already packed) and the head's chain."""
        seg = model.segments[0]
        meta = model.devices[seg.device]
        x = meta.tokens[:n]
        if record and model.rank == 0:
            rec["on"] = True
        try:
            for op in seg.ops[:-1]:
                x = op(x, meta)
        finally:
            if model.rank == 0:
                rec["on"] = False
        hidden = x
        picks = model._argmax(seg.ops[-1](x, meta))
        pos0 = meta.positions[:n].clone()
        h, xin, ds = hidden, picks, []
        for j in range(depth):
            meta.positions[:n].copy_(pos0 + j)
            z = model._fused_layer(model.mtp_index, model._mtp_in(h, xin), meta.slots[:n], meta.positions[:n], meta)
            y = _rms(z, model.mtp.norm.weight, model.eps)
            d = model._argmax(F.linear(y, model.lm_head))
            ds.append(d)
            h, xin = y, d
        meta.positions[:n].copy_(pos0)
        return picks, torch.stack(ds, dim=1), hidden

    seqs: dict[str, list[list[int]]] = {}
    drafts: dict[str, list[list[list[int]]]] = {}
    slot_of: dict[tuple[str, int], int] = {}
    t_all = time.perf_counter()
    for si, (name, prompts) in enumerate(sets.items()):
        seqs[name], drafts[name] = [], []
        for b0 in range(0, S, 8):
            batch = list(range(b0, min(S, b0 + 8)))
            slots = [si * S + s for s in batch]
            for s, slot in zip(batch, slots, strict=True):
                slot_of[(name, s)] = slot
            t0 = time.perf_counter()
            gen = [[int(prefill(slot, prompts[s]).argmax())] for s, slot in zip(batch, slots, strict=True)]
            chains = [[] for _ in batch]
            n = len(batch)
            record = name == "replay" and b0 == 0
            for st in range(args.gen):
                positions = [len(prompts[s]) + st for s in batch]
                for slot, p in zip(slots, positions, strict=True):
                    r0._ensure_pages(slot, p + depth + 2)
                host = torch.zeros((4, r0._rows), dtype=torch.long)
                host[1].fill_(r0.dummy)
                host[3].fill_(r0.dummy)
                host[0, :n] = torch.tensor([gl[-1] for gl in gen])
                host[1, :n] = torch.tensor(slots)
                host[2, :n] = torch.tensor(positions)
                host[3, :n] = torch.tensor([r0._g(sl) for sl in slots])
                fast._load_inputs(host)
                res = fast._both(lambda m: step(m, n, record))
                picks, ds, hidden = res[0]
                if record:
                    rec["hid"].append(hidden.detach().clone())
                picks = picks.tolist()
                ds = ds.tolist()
                for r in range(n):
                    gen[r].append(picks[r])
                    chains[r].append(ds[r])
            for r, s in enumerate(batch):
                seqs[name].append(gen[r])
                drafts[name].append(chains[r])
            out("generated", set=name, sessions=batch, steps=args.gen, seconds=round(time.perf_counter() - t0, 1),
                head=[tok.decode(gen[r][:24]) for r in range(min(2, n))])
    fd0._gdn, fd0._moe = gdn0, moe0
    out("accept_loop", seconds=round(time.perf_counter() - t_all, 1))

    def cut(name: str, s: int) -> int:
        gl = seqs[name][s]
        if name == "natural":
            for i, t in enumerate(gl):
                if t in eos:
                    return i + 1
        return len(gl)

    accept: dict[str, dict[int, float]] = {}
    for name in sets:
        hits = [0] * (depth + 1)
        starts = 0
        sim: dict[int, tuple[int, int]] = {}
        for s in range(S):
            gl, ch, end = seqs[name][s], drafts[name][s], cut(name, s)
            lens = []
            for st in range(len(ch)):
                if st + 1 + depth >= end:
                    break
                a = 0
                while a < depth and ch[st][a] == gl[st + 2 + a]:
                    a += 1
                lens.append(a)
            starts += len(lens)
            for a in lens:
                for j in range(a + 1):
                    hits[j] += 1
            for k in range(1, depth + 1):
                tokens = steps = 0
                st = 0
                while st < len(lens):
                    a = min(lens[st], k)
                    tokens += a + 1
                    steps += 1
                    st += a + 1
                tk, sp = sim.get(k, (0, 0))
                sim[k] = (tk + tokens, sp + steps)
        p = [hits[j] / max(1, hits[j - 1]) for j in range(1, depth + 1)]
        accept[name] = {k: sim[k][0] / max(1, sim[k][1]) for k in sim}
        out("accept", set=name, starts=starts, p_j=[round(v, 4) for v in p],
            E_k={k: round(e, 3) for k, e in accept[name].items()},
            kill_iii=("GO" if min(p[:4]) >= 0.97 else "KILL"))

    # ---------------------------------------------------------------- (a) kernels on rank 0
    torch.cuda.set_device(dev0)
    T = len(rec["hid"])

    def rowspec(R: int, k: int) -> list[tuple[int, int]]:
        """(step, batch row) for 2 slots x R/2 consecutive tokens, sample k."""
        per = R // 2
        a, b = (2 * k) % 8, (2 * k + 1) % 8
        t = (k * 11 + 3) % (T - per)
        return [(t + j, a) for j in range(per)] + [(t + j, b) for j in range(per)]

    dense_sets: dict[str, dict[int, list]] = {"in_proj": {}, "lm_head": {}}
    moe_sets: dict[int, list[MoESet]] = {}
    layers = {}
    with torch.inference_mode():
        for i in pick_moe:
            mlp = r0.layers[i].mlp
            layers[i] = MoELayer(f"L{i}", mlp.experts.gate_up_proj, mlp.experts.down_proj,
                                 mlp.shared_expert.down_proj.weight, fd0.w_moe[i])
        for R in ROWS:
            dense_sets["in_proj"][R], dense_sets["lm_head"][R], moe_sets[R] = [], [], []
            for k in range(args.samples):
                spec = rowspec(R, k)
                for i in pick_gdn:
                    x = torch.stack([rec["gdn"][i][t][b] for t, b in spec]).contiguous()
                    dense_sets["in_proj"][R].append((x, fd0.w_in[i]))
                dense_sets["lm_head"][R].append(
                    (torch.stack([rec["hid"][t][b] for t, b in spec]).to(dev0).contiguous(), r0.lm_head))
                for i in pick_moe:
                    h = torch.stack([rec["moe"][i][t][0][b] for t, b in spec]).contiguous()
                    x = torch.stack([rec["moe"][i][t][1][b] for t, b in spec]).contiguous()
                    r = fd0._gemv(h, layers[i].w3)
                    moe_sets[R].append(MoESet(layers[i], h, x, r, r0.top_k))
        sync()
        out("shapes", in_proj=list(fd0.w_in[pick_gdn[0]].shape), lm_head=list(r0.lm_head.shape),
            gate_up=list(layers[pick_moe[0]].gu.shape), down=list(layers[pick_moe[0]].dn.shape),
            shared_down=list(layers[pick_moe[0]].sd.shape))

        kern: dict = {}
        for label in ("in_proj", "lm_head"):
            try:
                kern[label] = bench_dense(label, fd0, dense_sets[label], DENSE_CFGS, label == "lm_head", True, cus)
            except Exception:  # noqa: BLE001
                out("error", kernel=label, error=traceback.format_exc().replace("\n", " | ")[-1500:])
        try:
            kern["moe"] = bench_moe(moe_sets, True, cus)
        except Exception:  # noqa: BLE001
            out("error", kernel="moe", error=traceback.format_exc().replace("\n", " | ")[-1500:])

        # every dense GEMV of rank 0's step (random inputs: timing only), today vs MFMA
        dense_w = []
        for i in range(n_layers):
            layer = r0.layers[i]
            dense_w.append(fd0.w_in[i])
            dense_w.append(layer.linear_attn.out_proj.weight if i in gdn_layers else layer.self_attn.o_proj.weight)
            dense_w.append(fd0.w_moe[i])
        dense_bytes = sum(w.numel() * 2 for w in dense_w)
        step_dense = {}
        dcfg = kern.get("in_proj", (None,))[0] or (32, 256, 4)
        for R in ROWS:
            xs = {}
            for w in dense_w:
                if w.shape[1] not in xs:
                    xs[w.shape[1]] = torch.randn((R, w.shape[1]), device=dev0).to(torch.bfloat16)
            outs = [torch.empty((R, w.shape[0]), dtype=torch.bfloat16, device=dev0) for w in dense_w]
            a = time_us([lambda w=w: fd0._gemv(xs[w.shape[1]], w) for w in dense_w], min_launches=len(dense_w))
            b = time_us([lambda w=w, o=o: gemv_mfma(xs[w.shape[1]], w, dcfg, o) for w, o in zip(dense_w, outs)],
                        min_launches=len(dense_w))
            step_dense[R] = (a, b)
            out("step_dense", R=R, gemvs=len(dense_w), mbytes=round(dense_bytes / 1e6), cfg=dcfg,
                today_ms=round(a * len(dense_w) / 1e3, 3), mfma_ms=round(b * len(dense_w) / 1e3, 3),
                today_TBps=round(dense_bytes / (a * len(dense_w)) / 1e6, 3),
                mfma_TBps=round(dense_bytes / (b * len(dense_w)) / 1e6, 3))
        big = torch.empty(int(256e6), dtype=torch.uint8, device=dev0)
        big2 = torch.empty_like(big)
        us = time_us([lambda: big2.copy_(big)], min_launches=8)
        out("copy_ref", bytes_rw=5.12e8, us=round(us, 1), TBps=round(5.12e8 / us / 1e6, 3))
        del big, big2, rec
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- (c) today's engine steps
    def all_gdn():
        return [pair for m in fast.ranks for pair in m.gdn_state.values()]

    def make_restore(rows: list[int]):
        saved = []
        for conv, recs in all_gdn():
            t = torch.tensor(rows, device=conv.device)
            saved.append((conv, recs, t, conv.index_select(0, t).clone(), recs.index_select(0, t).clone()))

        def go():
            for conv, recs, t, c, r in saved:
                conv.index_copy_(0, t, c)
                recs.index_copy_(0, t, r)

        return go

    lengths = {slot_of[("replay", s)]: len(replay[s]) + args.gen for s in range(S)}
    last = {slot_of[("replay", s)]: seqs["replay"][s][-1] for s in range(S)}

    def time_engine(kind: str, B: int) -> float:
        slots = [slot_of[("replay", s)] for s in range(B)]
        verify = kind == "verify"
        size = r0.bucket(B)
        rows = 2 * size if verify else size
        host = torch.zeros((4, r0._rows), dtype=torch.long)
        host[1].fill_(r0.dummy)
        host[3].fill_(r0.dummy)
        pos = [lengths[sl] for sl in slots]
        for sl, p in zip(slots, pos, strict=True):
            r0._ensure_pages(sl, p + 2)
        toks = [last[sl] for sl in slots]
        if verify:
            host[3, size:rows] = r0.dummy + r0.gdn_rows
            cur = [r0._g(sl) for sl in slots]
            other = [sl + (1 - r0.bank[sl]) * r0.gdn_rows for sl in slots]
            for half, pp, st in ((0, pos, cur), (size, [p + 1 for p in pos], other)):
                host[0, half : half + B] = torch.tensor(toks)
                host[1, half : half + B] = torch.tensor(slots)
                host[2, half : half + B] = torch.tensor(pp)
                host[3, half : half + B] = torch.tensor(st)
            rows_state = cur + other
        else:
            host[0, :B] = torch.tensor(toks)
            host[1, :B] = torch.tensor(slots)
            host[2, :B] = torch.tensor(pos)
            host[3, :B] = torch.tensor([r0._g(sl) for sl in slots])
            rows_state = [r0._g(sl) for sl in slots]
        restore = make_restore(rows_state)
        fast._load_inputs(host)
        fast._run(size, rows, verify)  # captures if needed
        restore()
        sync()
        times = []
        for _ in range(args.iters + 2):
            restore()
            sync()
            t0 = time.perf_counter()
            fast._run(size, rows, verify)
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
        restore()
        return statistics.median(times[2:])

    eng: dict = {}
    for kind, B in [("decode", 2), ("decode", 4), ("decode", 8), ("decode", 16),
                    ("verify", 1), ("verify", 2), ("verify", 4), ("verify", 8)]:
        try:
            ms = time_engine(kind, B)
            rows = 2 * B if kind == "verify" else B
            eng[(kind, rows)] = ms
            out("engine_step", kind=kind, slots=B, rows=rows, ms=round(ms, 3), graph_error=fast.graph_error)
        except Exception:  # noqa: BLE001
            out("error", what=f"engine {kind} {B}", error=traceback.format_exc().replace("\n", " | ")[-1500:])

    # ---------------------------------------------------------------- projection
    try:
        project(args, accept, kern, step_dense, eng, n_layers)
    except Exception:  # noqa: BLE001
        out("error", what="projection", error=traceback.format_exc().replace("\n", " | ")[-1500:])
    out("done", seconds=round(time.perf_counter() - started))


def project(args, accept, kern, step_dense, eng, n_layers) -> None:
    moe = kern["moe"]["res"]  # R -> (today us, mfma us) per layer
    head = kern["lm_head"][1]
    layers_moe = n_layers  # the MTP head's layer is counted in the chained head passes
    kt, km = {}, {}
    for R in ROWS:
        kt[R] = (step_dense[R][0] * 3 * n_layers + moe[R][0] * layers_moe + head[R][0]) / 1e3
        km[R] = (step_dense[R][1] * 3 * n_layers + moe[R][1] * layers_moe + head[R][1]) / 1e3
    step_t, step_m = {}, {}
    for R in ROWS:
        d = eng.get(("decode", R))
        if d is None:
            continue
        step_t[R] = d
        step_m[R] = d - kt[R] + km[R]
        out("proj_step", R=R, decode_today_ms=round(d, 3), kernels_today_ms=round(kt[R], 3),
            kernels_mfma_ms=round(km[R], 3), residual_ms=round(d - kt[R], 3), step_mfma_ms=round(step_m[R], 3))
    # one chained head pass at 2 rows: one layer's share of the 2-row step plus the lm_head
    def head_pass(step, lm):
        return (step[2] - lm[2] / 1e3) / n_layers + lm[2] / 1e3

    lm_t = {R: head[R][0] for R in ROWS}
    lm_m = {R: head[R][1] for R in ROWS}
    hp_t, hp_m = head_pass(step_t, lm_t), head_pass(step_m, lm_m)
    v4 = eng.get(("verify", 4))
    out("proj_head_pass", today_ms=round(hp_t, 3), mfma_ms=round(hp_m, 3),
        check_decode4_plus_head4_today=round(step_t[4] + (step_t[4] - lm_t[4] / 1e3) / n_layers + lm_t[4] / 1e3, 3),
        verify_B2_today_ms=None if v4 is None else round(v4, 3))
    overhead = args.parent_tpot * args.parent_tokens_per_step - (v4 if v4 else 0.0)
    out("proj_overhead", parent_tpot_ms=args.parent_tpot, parent_tokens_per_step=args.parent_tokens_per_step,
        served_step_ms=round(args.parent_tpot * args.parent_tokens_per_step, 3),
        host_overhead_ms=round(overhead, 3))

    def interp(table, R):
        if R in table:
            return table[R]
        lo = max(r for r in table if r < R)
        hi = min(r for r in table if r > R)
        return table[lo] + (table[hi] - table[lo]) * (R - lo) / (hi - lo)

    for name, ek in accept.items():
        for label, step, hp in (("mfma", step_m, hp_m), ("today", step_t, hp_t)):
            for served in (False, True):
                o = overhead if served else 0.0
                rows = {}
                for k in range(1, args.depth + 1):
                    R = 2 * (k + 1)
                    t = interp(step, R) + k * hp + o
                    rows[k] = {"R": R, "step_ms": round(t, 3), "E_k": round(ek[k], 3),
                               "tok_per_ms": round(2 * ek[k] / t, 4), "tpot_ms": round(t / ek[k], 3),
                               "interp": R not in step}
                base = rows[1]["tok_per_ms"]
                best = max(rows, key=lambda k: rows[k]["tok_per_ms"])
                v = (v4 + o) if v4 else None
                today_k1 = None if v is None else 2 * accept[name][1] / v
                out("proj", set=name, kernels=label, served_overhead=served, by_k=rows, best_k=best,
                    best_vs_k1=round(rows[best]["tok_per_ms"] / base, 3),
                    best_vs_today_verify_k1=None if today_k1 is None else round(rows[best]["tok_per_ms"] / today_k1, 3),
                    kill_ii=("GO" if rows[best]["tok_per_ms"] / base >= 1.35 else "KILL"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--first", type=int, default=1800)
    parser.add_argument("--stride", type=int, default=2400)
    parser.add_argument("--gen", type=int, default=192)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--parent-tpot", type=float, default=7.68, help="parent's official TPOT p50 (ms)")
    parser.add_argument("--parent-tokens-per-step", type=float, default=1.93)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    try:
        if args.smoke:
            smoke()
        else:
            real(args)
    except Exception:  # noqa: BLE001
        out("fatal", error=traceback.format_exc().replace("\n", " | "))
        raise


if __name__ == "__main__":
    main()
