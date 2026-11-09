"""Plan 47: retrieval-scored MTP drafts, an inverted-file (IVF) index over one lm_head shard.

A draft only has to be a likely token, so instead of the full-vocabulary GEMV + argmax the
draft can be chosen from a shortlist: the ``P`` clusters of lm_head rows whose centroids score
highest against the head's normed output ``y``, then the exact argmax over those clusters'
rows. The verify lm_head is never touched (served tokens stay the exact greedy argmax).

Built at load **only from the checkpoint's lm_head weights** (one TP rank's vocabulary shard):
no text, no token statistics. Deterministic (seeded init, fixed iteration count, ties broken
by index).

- ``build``: k-means over the shard's rows (``spherical``: on unit-normalized rows, assignment
  by cosine; else Euclidean), then a balanced assignment so no cluster holds more than ``cap``
  rows (overflow rows go to their next-best centroid with room), then the rows sorted by
  cluster (CSR ``offsets``). The search centroid of a cluster is the mean of its raw rows, so a
  centroid's score ``y . c`` is the mean logit of its members.
- ``search_rank``: (evaluation) per query, the rank of a given row's cluster among all
  centroids by score; the IVF draft equals the full-shard argmax ``v`` iff that rank < P.
- ``Draft``: the GPU path (graph-capturable, fixed shapes): centroid GEMV, top-P, one Triton
  kernel for the gathered GEMV + per-chunk argmax over the P clusters' rows, one for the final
  pick, written in ``tp.argmax_pack``'s ``[rows, 4]`` layout so ``Group.argmax``'s all-reduce
  and pick stay as they are.

Stage 0 (tools/ivf_probe.py) uses it; nothing on the served path imports it yet.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import dataclass

import torch

# The served draft (``Served``, used by ``FastModel._chain_head_op``): ENGINE_DRAFT_IVF=1 (default)
# picks each chained MTP draft from the top ``P`` of ``CLUSTERS`` spherical clusters of each lm_head
# shard's rows (plan 47; stage 0, j237: P 64, a gap rule at 17.875). Every rank holds both shards
# and both indices (iteration 3), so it searches the whole vocabulary and applies the global gap
# rule alone: both ranks reach the same draft with no rendezvous. When a row's shortlist max is
# within ``GAP`` of the best unscanned centroid score, both ranks take today's shard pass and
# all-reduce instead (for every row, and for the rest of the step's chain). ENGINE_DRAFT_IVF=0
# restores the full pass for every draft.
ENABLED = os.environ.get("ENGINE_DRAFT_IVF", "1") == "1"
P_SERVED = int(os.environ.get("ENGINE_DRAFT_IVF_P", "64"))
CLUSTERS = int(os.environ.get("ENGINE_DRAFT_IVF_CLUSTERS", "4096"))
CAP = 64
ITERS = 16
GAP = float(os.environ.get("ENGINE_DRAFT_IVF_GAP", "17.875"))
# Measuring only: also run the full draft and count the drafts that differ from it (one extra
# GEMV and all-reduce per draft).
AUDIT = os.environ.get("ENGINE_DRAFT_IVF_AUDIT", "0") == "1"
# Plan 71: the same search and fallback in fewer, cheaper launches (``Served._search_fast``):
# one top-P launch for both vocabulary parts (grid rows x parts), one gather launch for both
# parts, one launch for the shortlist max + gap flag of every row (it also clears the flag on
# a step's first link: no memset), the predicated full-pass GEMV on a persistent grid (with
# the flag at 0 its programs exit after one load each, ~1 per CU instead of one per tile), and
# the draft written straight into the chain's output. Same selection, ties, flag and bits.
FAST = os.environ.get("ENGINE_DRAFT_IVF_FAST", "0") == "1"
# The top-P select of the fast path: "radix" (today's 16-step radix, both parts in one launch)
# or "hist" (two 8-bit histogram passes). Both give the same P indices and next score.
TOPP = os.environ.get("ENGINE_DRAFT_IVF_TOPP", "radix")
# Programs per CU of the predicated full-pass GEMV (fast path); 0: one per tile (today's grid).
PRED_GRID = int(os.environ.get("ENGINE_DRAFT_IVF_PRED_GRID", "2"))


def usable(w: torch.Tensor) -> bool:
    """Whether a shard is big enough for the index (at least 8 rows per cluster on average;
    the tiny test checkpoints are not)."""
    return w.shape[0] >= 8 * CLUSTERS

try:  # the GPU kernels; the index build and the reference need only torch
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = tl = None


@dataclass
class Index:
    """One shard's index. Row ``j`` of the cluster-sorted order is shard row ``perm[j]``."""

    centroids: torch.Tensor  # [C, K] bf16, mean of each cluster's raw rows
    offsets: torch.Tensor  # [C + 1] int32, CSR into the sorted rows
    perm: torch.Tensor  # [N] int64, sorted position -> shard row
    cluster: torch.Tensor  # [N] int64, shard row -> its cluster
    cap: int
    build_s: float
    stats: dict

    @property
    def clusters(self) -> int:
        return self.centroids.shape[0]


def _chunks(n: int, size: int):
    for a in range(0, n, size):
        yield a, min(n, a + size)


def _assign_scores(x: torch.Tensor, c: torch.Tensor, spherical: bool, chunk: int, topk: int):
    """Top-``topk`` centroids per row of ``x`` (scores, ids), in row chunks."""
    vals, ids = [], []
    half = None if spherical else 0.5 * (c * c).sum(1)
    for a, b in _chunks(x.shape[0], chunk):
        s = x[a:b] @ c.T
        if half is not None:
            s = s - half  # argmax of x.c - |c|^2/2 is the nearest centroid
        v, i = s.topk(topk, dim=1)
        vals.append(v)
        ids.append(i)
    return torch.cat(vals), torch.cat(ids)


def _balanced(score: torch.Tensor, cand: torch.Tensor, clusters: int, cap: int,
              x: torch.Tensor, c: torch.Tensor, spherical: bool, chunk: int) -> tuple[torch.Tensor, int, int]:
    """Assign every row to one of its candidate clusters (best first) so no cluster holds more
    than ``cap`` rows; in each round the rows proposing to a cluster are taken best score
    first, ties to the lower row. Rows left after their candidates get new candidates among
    the clusters with room, and so on until every row is placed. Returns (assignment, rows
    moved off their first choice, candidate passes)."""
    n, rounds = cand.shape
    dev = score.device
    out = torch.full((n,), -1, dtype=torch.long, device=dev)
    fill = torch.zeros(clusters, dtype=torch.long, device=dev)
    rows = torch.arange(n, device=dev)
    first = cand[:, 0].clone()
    passes = 0
    todo = rows
    while todo.numel():
        passes += 1
        for r in range(cand.shape[1]):
            live = out[todo] < 0
            t = todo[live]
            if t.numel() == 0:
                break
            cl = cand[live, r]
            sc = score[live, r]
            ok0 = torch.isfinite(sc)
            t, cl, sc = t[ok0], cl[ok0], sc[ok0]
            o1 = torch.argsort(-sc, stable=True)  # best first (ties: lower row, t is sorted)
            o2 = torch.argsort(cl[o1], stable=True)  # group by cluster, keep that order
            order = o1[o2]
            cls = cl[order]
            counts = torch.bincount(cls, minlength=clusters)
            starts = torch.cumsum(counts, 0) - counts
            pos = torch.arange(order.numel(), device=dev) - starts[cls]
            ok = pos < (cap - fill)[cls]
            out[t[order[ok]]] = cls[ok]
            fill += torch.bincount(cls[ok], minlength=clusters)
        todo = rows[out < 0]
        if todo.numel():  # new candidates among the clusters with room
            k = min(rounds, int((fill < cap).sum()))
            half = None if spherical else 0.5 * (c * c).sum(1)
            vals, ids = [], []
            for a, b in _chunks(todo.numel(), chunk):
                s = x[todo[a:b]] @ c.T
                if half is not None:
                    s = s - half
                s = torch.where(fill < cap, s, torch.full_like(s, float("-inf")))
                v, i = s.topk(k, dim=1)
                vals.append(v)
                ids.append(i)
            score, cand = torch.cat(vals), torch.cat(ids)
    moved = int((out != first).sum())
    return out, moved, passes


@torch.no_grad()
def build(w: torch.Tensor, clusters: int, cap: int, iters: int = 16, spherical: bool = False,
          seed: int = 0, chunk: int = 32768, candidates: int = 8) -> Index:
    """The index of lm_head shard ``w`` [N, K] (any float dtype, on its device)."""
    if w.is_cuda:
        torch.cuda.synchronize(w.device)
    t0 = time.perf_counter()
    n = w.shape[0]
    assert clusters * cap >= n, f"{clusters} clusters x cap {cap} cannot hold {n} rows"
    x = w.float()
    if spherical:
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(n, generator=gen)[:clusters].to(w.device)
    c = x[init].clone()
    for _ in range(iters):
        _, a = _assign_scores(x, c, spherical, chunk, 1)
        a = a[:, 0]
        counts = torch.bincount(a, minlength=clusters)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        empty = counts == 0
        c = torch.where(empty[:, None], c, sums / counts.clamp_min(1)[:, None].float())
        if spherical:
            c = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if bool(empty.any()):  # reseed empty clusters with the rows farthest from their centroid
            e = int(empty.sum())
            fit = (x * c[a]).sum(1) if spherical else -((x - c[a]) ** 2).sum(1)
            far = torch.argsort(fit, stable=True)[:e]
            c[torch.nonzero(empty)[:, 0]] = x[far]
    score, cand = _assign_scores(x, c, spherical, chunk, candidates)
    raw_sizes = torch.bincount(cand[:, 0], minlength=clusters)
    assign, moved, passes = _balanced(score, cand, clusters, cap, x, c, spherical, chunk)
    sizes = torch.bincount(assign, minlength=clusters)
    perm = torch.argsort(assign, stable=True)
    offsets = torch.zeros(clusters + 1, dtype=torch.int32, device=w.device)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)
    wf = w.float()
    cent = torch.zeros((clusters, w.shape[1]), dtype=torch.float32, device=w.device)
    cent.index_add_(0, assign, wf)
    cent = cent / sizes.clamp_min(1)[:, None].float()
    if w.is_cuda:
        torch.cuda.synchronize(w.device)
    build_s = time.perf_counter() - t0
    stats = dict(rows=n, clusters=clusters, cap=cap, iters=iters, spherical=spherical,
                 raw_min=int(raw_sizes.min()), raw_med=float(raw_sizes.float().median()),
                 raw_max=int(raw_sizes.max()), raw_empty=int((raw_sizes == 0).sum()),
                 min=int(sizes.min()), max=int(sizes.max()), mean=n / clusters,
                 empty=int((sizes == 0).sum()), moved=moved, moved_frac=moved / n,
                 balance_passes=passes)
    return Index(cent.to(torch.bfloat16), offsets, perm, assign, cap, build_s, stats)


@torch.no_grad()
def cluster_scores(y: torch.Tensor, index: Index) -> torch.Tensor:
    """Centroid scores ``bf16(y . c)`` as fp32, [rows, C] (the GPU path rounds them too)."""
    return (y.float() @ index.centroids.float().T).to(torch.bfloat16).float()


@torch.no_grad()
def search_rank(y: torch.Tensor, index: Index, rows: torch.Tensor) -> torch.Tensor:
    """Per query ``y[i]``, the rank of shard row ``rows[i]``'s cluster among all clusters by
    centroid score (0 = best; ties counted to the lower cluster id, as a top-P keeps them)."""
    s = cluster_scores(y, index)
    cl = index.cluster[rows]
    mine = s.gather(1, cl[:, None])
    ids = torch.arange(s.shape[1], device=s.device)[None, :]
    return ((s > mine) | ((s == mine) & (ids < cl[:, None]))).sum(1)


@torch.no_grad()
def reference_draft(y: torch.Tensor, w: torch.Tensor, index: Index, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(max logit, shard row) of the IVF draft per query: the argmax of ``bf16(y . w_v)`` over
    the rows of the top-``p`` clusters, ties to the lower row. Torch; slow and exact."""
    s = cluster_scores(y, index)
    top = s.topk(p, dim=1).indices
    sel = torch.zeros_like(s, dtype=torch.bool).scatter_(1, top, True)
    logits = (y.float() @ w.float().T).to(torch.bfloat16).float()
    logits = torch.where(sel.gather(1, index.cluster[None, :].expand(y.shape[0], -1)), logits,
                         torch.full_like(logits, float("-inf")))
    best, row = logits.max(1)
    return best, row


# ================================================================================ GPU kernels

if triton is not None:

    @triton.jit
    def _ivf_gather_kernel(y_ptr, w_ptr, ids_ptr, off_ptr, sel_ptr, outv_ptr, outi_ptr, P,
                           K: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr,
                           READ_ONLY: tl.constexpr, PERM: tl.constexpr = False,
                           flag_ptr=None, PRED: tl.constexpr = False):
        """Program (r, p, j): rows ``j*BR..`` of query r's p-th selected cluster; writes the
        chunk's (max bf16 logit, lowest vocab id at that max) to slot ``(r*P + p)*CH + j``.
        READ_ONLY: the same loads with no math (a gather-read ceiling). PRED: every load
        masked off when flag[0] != 0 (the rank already fell back this step)."""
        r = tl.program_id(0)
        p = tl.program_id(1)
        j = tl.program_id(2)
        c = tl.load(sel_ptr + r * P + p)  # always a valid cluster (sel starts zeroed)
        start = tl.load(off_ptr + c)
        end = tl.load(off_ptr + c + 1)
        if PRED:  # an empty cluster: every load below masked off
            end = tl.where(tl.load(flag_ptr) == 0, end, start)
        rows = start + j * BR + tl.arange(0, BR)
        ok = rows < end
        if PERM:  # read the shard's own rows through the permutation (no sorted copy)
            src = tl.load(ids_ptr + rows, mask=ok, other=0)
        else:
            src = rows
        acc = tl.zeros([BR], dtype=tl.float32)
        for k0 in range(0, K, BK):
            cols = k0 + tl.arange(0, BK)
            w = tl.load(w_ptr + src[:, None].to(tl.int64) * K + cols[None, :], mask=ok[:, None],
                        other=0.0).to(tl.float32)
            if READ_ONLY:
                acc += tl.sum(w, axis=1)
            else:
                x = tl.load(y_ptr + r * K + cols).to(tl.float32)
                acc += tl.sum(w * x[None, :], axis=1)
        v = acc.to(tl.bfloat16).to(tl.float32)
        v = tl.where(ok, v, float("-inf"))
        m = tl.max(v, axis=0)
        ids = tl.load(ids_ptr + rows, mask=ok, other=2147483647)
        best = tl.min(tl.where(v == m, ids, 2147483647), axis=0)
        slot = (r * P + p) * CH + j
        tl.store(outv_ptr + slot, m)
        tl.store(outi_ptr + slot, best)

    @triton.jit
    def _ivf_pick_kernel(outv_ptr, outi_ptr, both_ptr, M, offset, RANK: tl.constexpr,
                         BM: tl.constexpr):
        """both[r] = (max, lowest id at it + offset) over query r's M chunk slots, in columns
        2 RANK, 2 RANK + 1 of four, zeros in the peer's (``tp.argmax_pack``'s layout)."""
        r = tl.program_id(0)
        lanes = tl.arange(0, BM)
        best = tl.full([BM], float("-inf"), tl.float32)
        bid = tl.full([BM], 2147483647, tl.int32)
        for m0 in range(0, M, BM):
            i = m0 + lanes
            ok = i < M
            v = tl.load(outv_ptr + r * M + i, mask=ok, other=float("-inf"))
            d = tl.load(outi_ptr + r * M + i, mask=ok, other=2147483647)
            upd = (v > best) | ((v == best) & (d < bid))
            best = tl.where(upd, v, best)
            bid = tl.where(upd, d, bid)
        m = tl.max(best, axis=0)
        idx = tl.min(tl.where(best == m, bid, 2147483647), axis=0)
        base = both_ptr + r * 4
        tl.store(base + 2 * RANK, m)
        tl.store(base + 2 * RANK + 1, (idx + offset).to(tl.float32))
        tl.store(base + 2 * (1 - RANK), 0.0)
        tl.store(base + 2 * (1 - RANK) + 1, 0.0)


class Draft:
    """The GPU IVF draft over one shard: ``pack(y)`` -> ``[rows, 4]`` (``tp.argmax_pack``'s
    layout). Holds a cluster-sorted copy of the shard (same bytes as the shard)."""

    BR = 16
    BK = 256

    def __init__(self, w: torch.Tensor, index: Index, p: int, rank: int = 0, offset: int = 0,
                 max_rows: int = 16, centroid_gemv=None) -> None:
        self.index, self.p, self.rank, self.offset = index, p, rank, offset
        self.k = w.shape[1]
        self.ws = w.index_select(0, index.perm).contiguous()  # cluster-sorted rows
        self.ids = index.perm.to(torch.int32).contiguous()  # sorted row -> shard row
        self.offsets = index.offsets
        self.cent = index.centroids.contiguous()
        self.ch = max(1, math.ceil(index.cap / self.BR))
        m = p * self.ch
        self.outv = torch.empty((max_rows, m), dtype=torch.float32, device=w.device)
        self.outi = torch.empty((max_rows, m), dtype=torch.int32, device=w.device)
        self.cgemv = centroid_gemv or (lambda y, c: y @ c.T)

    def select(self, y: torch.Tensor) -> torch.Tensor:
        s = self.cgemv(y, self.cent)
        return torch.topk(s.float(), self.p, dim=1, sorted=False).indices

    def gather(self, y: torch.Tensor, sel: torch.Tensor, read_only: bool = False) -> None:
        rows = y.shape[0]
        _ivf_gather_kernel[(rows, self.p, self.ch)](
            y, self.ws, self.ids, self.offsets, sel, self.outv, self.outi, self.p,
            K=self.k, CH=self.ch, BR=self.BR, BK=self.BK, READ_ONLY=read_only, num_warps=4)

    def pick(self, rows: int) -> torch.Tensor:
        both = torch.empty((rows, 4), dtype=torch.float32, device=self.ws.device)
        m = self.p * self.ch
        _ivf_pick_kernel[(rows,)](self.outv, self.outi, both, m, self.offset, RANK=self.rank,
                                  BM=min(1024, triton.next_power_of_2(m)), num_warps=4)
        return both

    def pack(self, y: torch.Tensor) -> torch.Tensor:
        y = y.contiguous()
        sel = self.select(y)
        self.gather(y, sel)
        return self.pick(y.shape[0])


# ================================================================================ served path

if triton is not None:

    @triton.jit(do_not_specialize=["R", "N"])
    def _pred_gemv_kernel(flag_ptr, x_ptr, w_ptr, out_ptr, R, N, stride_out, K: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT_F32: tl.constexpr):
        """``mfma._mfma_gemv_kernel`` (no K split) run only when flag[0] != 0."""
        if tl.load(flag_ptr) != 0:
            nb = tl.program_id(0)
            rm = tl.arange(0, BM)
            mok = rm < R
            rn = nb * BN + tl.arange(0, BN)
            nok = rn < N
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k0 in range(0, K, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
                b = tl.load(w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None], mask=nok[None, :], other=0.0)
                if DOT_F32:  # the interpreter's bf16 dot
                    acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
                else:
                    acc = tl.dot(a, b, acc)
            m = mok[:, None] & nok[None, :]
            tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)

    @triton.jit
    def _radix_topp_kernel(s_ptr, out_ptr, nxt_ptr, S, C: tl.constexpr, P: tl.constexpr):
        """out[r, :P] = indices of the P largest bf16 scores of s[r, :C] (row stride S; ties to
        the lower index; in index order), nxt[r] = the best unselected score. A 16-step radix
        select on the scores' 16-bit order keys, then a compaction by prefix sums; one program
        per row (stage 0c, j237: 12.3 us at 2 rows, exact)."""
        r = tl.program_id(0)
        i = tl.arange(0, C)
        b = tl.load(s_ptr + r * S + i)
        bi = b.to(tl.int16, bitcast=True).to(tl.int32)
        key = tl.where(bi >= 0, bi + 32768, ~bi)
        t = tl.zeros([], dtype=tl.int32)
        for j in tl.static_range(16):
            cand = t | (1 << (15 - j))
            cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
            t = tl.where(cnt >= P, cand, t)
        gt = key > t
        ngt = tl.sum(gt.to(tl.int32), axis=0)
        eq = key == t
        ceq = tl.cumsum(eq.to(tl.int32), axis=0)
        sel = gt | (eq & (ceq <= P - ngt))
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(out_ptr + r * P + pos, i.to(tl.int64), mask=sel)
        rest = tl.where(sel, float("-inf"), b.to(tl.float32))
        tl.store(nxt_ptr + r, tl.max(rest, axis=0))

    @triton.jit(do_not_specialize=["M", "PART_ROWS", "OFF1"])
    def _ivf_local_kernel(outv_ptr, outi_ptr, nxt_ptr, loc_ptr, ids_ptr, M, PART_ROWS, OFF1,
                          NP: tl.constexpr, BM: tl.constexpr):
        """loc[r] = (shortlist max over every vocabulary part, its lowest vocabulary id, the best
        unscanned centroid score over the parts); ids[r] = that id. Part i's slots start at
        ``i * PART_ROWS * M``; part 1's ids at vocabulary offset ``OFF1``."""
        r = tl.program_id(0)
        lanes = tl.arange(0, BM)
        top = tl.full([], float("-inf"), tl.float32)
        tid = tl.full([], 0, tl.int32)
        nxt = tl.full([], float("-inf"), tl.float32)
        for part in tl.static_range(NP):
            base = part * PART_ROWS * M + r * M
            best = tl.full([BM], float("-inf"), tl.float32)
            bid = tl.full([BM], 2147483647, tl.int32)
            for m0 in range(0, M, BM):
                i = m0 + lanes
                ok = i < M
                v = tl.load(outv_ptr + base + i, mask=ok, other=float("-inf"))
                d = tl.load(outi_ptr + base + i, mask=ok, other=2147483647)
                upd = (v > best) | ((v == best) & (d < bid))
                best = tl.where(upd, v, best)
                bid = tl.where(upd, d, bid)
            m = tl.max(best, axis=0)
            idx = tl.min(tl.where(best == m, bid, 2147483647), axis=0) + part * OFF1
            if part == 0:
                top = m
                tid = idx
            else:  # a later part wins only when strictly larger (ties to the lower id)
                tid = tl.where(m > top, idx, tid)
                top = tl.maximum(m, top)
            nxt = tl.maximum(nxt, tl.load(nxt_ptr + part * PART_ROWS + r))
        tl.store(loc_ptr + r * 3, top)
        tl.store(loc_ptr + r * 3 + 1, tid.to(tl.float32))
        tl.store(loc_ptr + r * 3 + 2, nxt)
        tl.store(ids_ptr + r, tid.to(tl.int64))

    @triton.jit(do_not_specialize=["R", "gap"])
    def _ivf_flag_kernel(loc_ptr, flag_ptr, stats_ptr, R, gap, RB: tl.constexpr):
        """The step falls back (flag[0] = 1, kept for the rest of the step's chain) when any
        row's shortlist max is within ``gap`` of the best unscanned centroid score. Counts
        drafts and drafts on the full pass."""
        r = tl.arange(0, RB)
        ok = r < R
        f = tl.load(flag_ptr)
        if f == 0:
            t = tl.load(loc_ptr + r * 3, mask=ok, other=0.0)
            n = tl.load(loc_ptr + r * 3 + 2, mask=ok, other=0.0)
            f = tl.max(((t - n < gap) & ok).to(tl.int32), axis=0)
            tl.store(flag_ptr, f)
        tl.atomic_add(stats_ptr, R.to(tl.int64))
        tl.atomic_add(stats_ptr + 1, (R * f).to(tl.int64))

    @triton.jit(do_not_specialize=["V", "offset"])
    def _pred_argmax_pack_kernel(flag_ptr, l_ptr, both_ptr, V, stride, offset, RANK: tl.constexpr,
                                 SOLO: tl.constexpr, BLOCK: tl.constexpr):
        """When flag[0] != 0: ``tp._argmax_pack_kernel`` (the peer's columns -inf / 0 when
        SOLO: no all-reduce follows)."""
        if tl.load(flag_ptr) != 0:
            row = tl.program_id(0)
            base = l_ptr + row.to(tl.int64) * stride
            lanes = tl.arange(0, BLOCK)
            best = tl.full([BLOCK], float("-inf"), tl.float32)
            bidx = tl.zeros([BLOCK], dtype=tl.int32)
            for v0 in range(0, V, BLOCK):
                cols = v0 + lanes
                x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
                upd = x > best
                best = tl.where(upd, x, best)
                bidx = tl.where(upd, cols, bidx)
            m = tl.max(best, axis=0)
            i = tl.min(tl.where(best == m, bidx, V), axis=0)
            i = tl.where(m == float("-inf"), 0, i)
            k = tl.arange(0, 4)
            other = float("-inf") if SOLO else 0.0
            val = tl.where(k == 2 * RANK, m, tl.where(k == 2 * RANK + 1, (i + offset).to(tl.float32),
                                                      tl.where(k == 2 * (1 - RANK), other, 0.0)))
            tl.store(both_ptr + row * 4 + k, val)

    @triton.jit(do_not_specialize=["R"])
    def _ivf_final_kernel(flag_ptr, both_ptr, ids_ptr, R, RB: tl.constexpr):
        """When flag[0] != 0: ids[r] = ``tp.argmax_pick`` of the all-reduced pack."""
        if tl.load(flag_ptr) != 0:
            r = tl.arange(0, RB)
            ok = r < R
            a0 = tl.load(both_ptr + r * 4, mask=ok, other=0.0)
            a1 = tl.load(both_ptr + r * 4 + 1, mask=ok, other=0.0)
            a2 = tl.load(both_ptr + r * 4 + 2, mask=ok, other=0.0)
            a3 = tl.load(both_ptr + r * 4 + 3, mask=ok, other=0.0)
            tl.store(ids_ptr + r, tl.where(a0 >= a2, a1, a3).to(tl.int64), mask=ok)

    # ---------------------------------------------------------------- plan 71: the fast path

    @triton.jit(do_not_specialize=["S", "OUT_PART", "NXT_PART"])
    def _topp2_kernel(s_ptr, out_ptr, nxt_ptr, S, OUT_PART, NXT_PART, C: tl.constexpr,
                      P: tl.constexpr, HIST: tl.constexpr):
        """``_radix_topp_kernel`` for vocabulary part ``program_id(1)`` (its C scores at column
        ``part * C`` of row r; out/nxt at ``part * OUT_PART`` / ``part * NXT_PART``): both
        parts in one launch. The threshold is the P-th largest order key, found by the
        16-step radix (HIST False) or by two 8-bit histogram passes (HIST True); the same
        key, so the same compaction and the same bits."""
        r = tl.program_id(0)
        part = tl.program_id(1)
        i = tl.arange(0, C)
        b = tl.load(s_ptr + r * S + part * C + i)
        bi = b.to(tl.int16, bitcast=True).to(tl.int32)
        key = tl.where(bi >= 0, bi + 32768, ~bi)
        if HIST:
            bins = tl.arange(0, 256)
            hi = key >> 8
            h1 = tl.histogram(hi, 256)
            ge1 = C - tl.cumsum(h1, axis=0) + h1  # keys whose high byte is >= the bin
            bh = tl.max(tl.where(ge1 >= P, bins, -1), axis=0)
            need = P - tl.sum(tl.where(bins > bh, h1, 0), axis=0)
            bins2 = tl.arange(0, 512)
            h2 = tl.histogram(tl.where(hi == bh, key & 255, 256), 512)  # others in bin 256
            h2 = tl.where(bins2 < 256, h2, 0)
            ge2 = tl.sum(h2, axis=0) - tl.cumsum(h2, axis=0) + h2
            bl = tl.max(tl.where((ge2 >= need) & (bins2 < 256), bins2, -1), axis=0)
            t = bh * 256 + bl
        else:
            t = tl.zeros([], dtype=tl.int32)
            for j in tl.static_range(16):
                cand = t | (1 << (15 - j))
                cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
                t = tl.where(cnt >= P, cand, t)
        gt = key > t
        ngt = tl.sum(gt.to(tl.int32), axis=0)
        eq = key == t
        ceq = tl.cumsum(eq.to(tl.int32), axis=0)
        sel = gt | (eq & (ceq <= P - ngt))
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(out_ptr + part * OUT_PART + r * P + pos, i.to(tl.int64), mask=sel)
        rest = tl.where(sel, float("-inf"), b.to(tl.float32))
        tl.store(nxt_ptr + part * NXT_PART + r, tl.max(rest, axis=0))

    @triton.jit(do_not_specialize=["P", "SEL_PART", "OUT_PART", "IDS_PART", "OFF_PART"])
    def _ivf_gather2_kernel(y_ptr, w_ptr, wdelta, ids_ptr, off_ptr, sel_ptr, outv_ptr, outi_ptr, P,
                            SEL_PART, OUT_PART, IDS_PART, OFF_PART, K: tl.constexpr, CH: tl.constexpr,
                            BR: tl.constexpr, BK: tl.constexpr):
        """``_ivf_gather_kernel`` (PERM) for both parts in one launch: program (r, part * P + p,
        j). Part 1's weights start ``wdelta`` elements after part 0's (two tensors), its ids,
        offsets, selection and output slots ``*_PART`` entries after part 0's."""
        r = tl.program_id(0)
        q = tl.program_id(1)
        j = tl.program_id(2)
        part = q // P
        p = q - part * P
        c = tl.load(sel_ptr + part * SEL_PART + r * P + p)
        start = tl.load(off_ptr + part * OFF_PART + c)
        end = tl.load(off_ptr + part * OFF_PART + c + 1)
        rows = start + j * BR + tl.arange(0, BR)
        ok = rows < end
        src = tl.load(ids_ptr + part * IDS_PART + rows, mask=ok, other=0)
        wp = w_ptr + part.to(tl.int64) * wdelta
        acc = tl.zeros([BR], dtype=tl.float32)
        for k0 in range(0, K, BK):
            cols = k0 + tl.arange(0, BK)
            w = tl.load(wp + src[:, None].to(tl.int64) * K + cols[None, :], mask=ok[:, None],
                        other=0.0).to(tl.float32)
            x = tl.load(y_ptr + r * K + cols).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        v = acc.to(tl.bfloat16).to(tl.float32)
        v = tl.where(ok, v, float("-inf"))
        m = tl.max(v, axis=0)
        ids = tl.load(ids_ptr + part * IDS_PART + rows, mask=ok, other=2147483647)
        best = tl.min(tl.where(v == m, ids, 2147483647), axis=0)
        slot = part * OUT_PART + (r * P + p) * CH + j
        tl.store(outv_ptr + slot, m)
        tl.store(outi_ptr + slot, best)

    @triton.jit(do_not_specialize=["R", "M", "PART_ROWS", "OFF1", "gap"])
    def _ivf_local_flag_kernel(outv_ptr, outi_ptr, nxt_ptr, loc_ptr, ids_ptr, flag_ptr, stats_ptr, R,
                               M, PART_ROWS, OFF1, gap, NP: tl.constexpr, RB: tl.constexpr,
                               BM: tl.constexpr, FIRST: tl.constexpr):
        """``_ivf_local_kernel`` for every row (a [RB, BM] tile) then ``_ivf_flag_kernel``, in
        one program. FIRST: the step's first link (the flag starts at 0, as after its memset)."""
        rr = tl.arange(0, RB)
        okr = rr < R
        lanes = tl.arange(0, BM)
        top = tl.full([RB], float("-inf"), tl.float32)
        tid = tl.zeros([RB], dtype=tl.int32)
        nxt = tl.full([RB], float("-inf"), tl.float32)
        for part in tl.static_range(NP):
            base = part * PART_ROWS * M + rr * M
            best = tl.full([RB, BM], float("-inf"), tl.float32)
            bid = tl.full([RB, BM], 2147483647, tl.int32)
            for m0 in range(0, M, BM):
                i = m0 + lanes
                ok = okr[:, None] & (i < M)[None, :]
                v = tl.load(outv_ptr + base[:, None] + i[None, :], mask=ok, other=float("-inf"))
                d = tl.load(outi_ptr + base[:, None] + i[None, :], mask=ok, other=2147483647)
                upd = (v > best) | ((v == best) & (d < bid))
                best = tl.where(upd, v, best)
                bid = tl.where(upd, d, bid)
            m = tl.max(best, axis=1)
            idx = tl.min(tl.where(best == m[:, None], bid, 2147483647), axis=1)
            idx = tl.where(okr, idx, 0) + part * OFF1
            if part == 0:
                top = m
                tid = idx
            else:  # a later part wins only when strictly larger (ties to the lower id)
                tid = tl.where(m > top, idx, tid)
                top = tl.maximum(m, top)
            nxt = tl.maximum(nxt, tl.load(nxt_ptr + part * PART_ROWS + rr, mask=okr, other=0.0))
        tl.store(loc_ptr + rr * 3, top, mask=okr)
        tl.store(loc_ptr + rr * 3 + 1, tid.to(tl.float32), mask=okr)
        tl.store(loc_ptr + rr * 3 + 2, nxt, mask=okr)
        tl.store(ids_ptr + rr, tid.to(tl.int64), mask=okr)
        if FIRST:
            f = tl.max(((top - nxt < gap) & okr).to(tl.int32), axis=0)
            tl.store(flag_ptr, f)
        else:
            f = tl.load(flag_ptr)
            if f == 0:
                f = tl.max(((top - nxt < gap) & okr).to(tl.int32), axis=0)
                tl.store(flag_ptr, f)
        tl.atomic_add(stats_ptr, R.to(tl.int64))
        tl.atomic_add(stats_ptr + 1, (R * f).to(tl.int64))

    @triton.jit(do_not_specialize=["R", "N", "TILES"])
    def _pred_gemv_p_kernel(flag_ptr, x_ptr, w_ptr, out_ptr, R, N, stride_out, TILES, K: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT_F32: tl.constexpr):
        """``_pred_gemv_kernel`` on a persistent grid: program i runs tiles i, i + grid, ...
        (each tile's math unchanged), only when flag[0] != 0."""
        if tl.load(flag_ptr) != 0:
            rm = tl.arange(0, BM)
            mok = rm < R
            for nb in range(tl.program_id(0), TILES, tl.num_programs(0)):
                rn = nb * BN + tl.arange(0, BN)
                nok = rn < N
                acc = tl.zeros([BM, BN], dtype=tl.float32)
                for k0 in range(0, K, BK):
                    rk = k0 + tl.arange(0, BK)
                    a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
                    b = tl.load(w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None], mask=nok[None, :],
                                other=0.0)
                    if DOT_F32:
                        acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
                    else:
                        acc = tl.dot(a, b, acc)
                m = mok[:, None] & nok[None, :]
                tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)


def _to(index: Index, dev: torch.device) -> Index:
    """``index`` copied to ``dev`` (the same bits)."""
    return Index(index.centroids.to(dev), index.offsets.to(dev), index.perm.to(dev), index.cluster.to(dev),
                 index.cap, index.build_s, index.stats)


class Served:
    """One rank's served IVF draft over the **whole** vocabulary (iteration 3): the rank holds
    a copy of the peer's lm_head shard and both shards' indices (built once, by rank 0, and
    copied: both ranks hold the same bits), so it runs the search and the global gap rule on
    its own and both ranks reach the same draft and the same fallback decision without a
    rendezvous. On a fallback (any row's shortlist max within ``gap`` of the best unscanned
    centroid score over both parts; then for the rest of the step's chain) both ranks run
    today's shard GEMV + argmax pack + all-reduce, predicated on that shared flag.

    ``w`` is this rank's shard (vocabulary rows from ``offset``); ``peer`` the other rank's
    shard (any device) or None (one device); ``src`` rank 0's ``Served`` (rank 1 copies its
    indices instead of building them)."""

    BR = 16
    BK = 256
    ARGMAX_BLOCK = 8192

    def __init__(self, w: torch.Tensor, max_rows: int, p: int = P_SERVED, clusters: int = CLUSTERS,
                 gap: float = GAP, offset: int = 0, peer: torch.Tensor | None = None,
                 src: "Served | None" = None) -> None:
        from engine.fast import mfma

        self.w, self.p, self.gap = w, p, gap
        k = w.shape[1]
        self.tiles_w = mfma.tiles_for(w.shape[0], k)
        if self.tiles_w[3] != 1:
            raise ValueError("the predicated GEMV has no K split")
        dev = w.device
        t0 = time.perf_counter()
        with torch.cuda.device(dev) if w.is_cuda else contextlib.nullcontext():
            own = 0 if peer is None or offset == 0 else 1
            weights = [w] if peer is None else ([w, peer.to(dev)] if own == 0 else [peer.to(dev), w])
            if src is not None:  # rank 1: rank 0's indices, the same bits
                self.indices = [_to(ix, dev) for ix in src.indices]
            else:
                self.indices = [build(x, clusters, CAP, iters=ITERS, spherical=True) for x in weights]
            self.weights = weights
            self.np = len(weights)
            self.part_rows = weights[0].shape[0]
            if self.np == 2 and weights[1].shape[0] > self.part_rows:
                raise ValueError("vocabulary part 1 is larger than part 0")
            self.cent = torch.cat([ix.centroids for ix in self.indices]).contiguous()
            self.c = self.indices[0].clusters
            self.ids = [ix.perm.to(torch.int32).contiguous() for ix in self.indices]
            self.offsets = [ix.offsets for ix in self.indices]
            self.ch = max(1, math.ceil(max(ix.cap for ix in self.indices) / self.BR))
            self.m = p * self.ch
            self.max_rows = max_rows
            np_ = self.np
            self.sel = torch.zeros((np_, max_rows, p), dtype=torch.int64, device=dev)
            self.nxt = torch.empty((np_, max_rows), dtype=torch.float32, device=dev)
            self.outv = torch.empty((np_, max_rows, self.m), dtype=torch.float32, device=dev)
            self.outi = torch.empty((np_, max_rows, self.m), dtype=torch.int32, device=dev)
            self.loc = torch.empty((max_rows, 3), dtype=torch.float32, device=dev)
            self.flag = torch.zeros((1,), dtype=torch.int32, device=dev)
            # drafts, drafts on the full pass, audited drafts, audited drafts unlike the full one
            self.stats = torch.zeros((4,), dtype=torch.int64, device=dev)
            self.rb = max(16, triton.next_power_of_2(max_rows))
            # plan 71 (fast path): the parts' ids and offsets back to back, part 1's weights as
            # an element offset from part 0's
            self.fast = FAST
            self.topp_hist = TOPP == "hist"
            self.pred_grid = PRED_GRID
            self.cus = torch.cuda.get_device_properties(dev).multi_processor_count if w.is_cuda else 4
            self.ids_cat = torch.zeros((np_ * self.part_rows,), dtype=torch.int32, device=dev)
            for part, ids in enumerate(self.ids):
                self.ids_cat[part * self.part_rows: part * self.part_rows + ids.numel()].copy_(ids)
            self.off_cat = torch.cat(self.offsets).contiguous()
            self.wdelta = 0
            if np_ == 2:
                es = weights[0].element_size()
                delta = weights[1].data_ptr() - weights[0].data_ptr()
                assert delta % es == 0 and weights[0].is_contiguous() and weights[1].is_contiguous()
                self.wdelta = delta // es
            self._warm()
            self.stats.zero_()
            if w.is_cuda:
                torch.cuda.empty_cache()  # the build's temporaries, before the KV pool is sized
        extra = 0 if peer is None else self.weights[1 - own].numel() * self.weights[1 - own].element_size()
        print(f"engine: IVF drafts on {dev}: {self.np} x {self.c} clusters, P {p}, gap {gap}, "
              f"{'copied' if src is not None else 'built'} in {time.perf_counter() - t0:.1f}s, "
              f"peer shard copy {extra / 2**30:.2f} GiB", flush=True)

    def _search(self, y: torch.Tensor, ids: torch.Tensor) -> None:
        """The IVF draft of every row into ``ids`` and ``loc``, then the fallback flag."""
        from engine.fast import mfma

        rows = y.shape[0]
        s = mfma.gemv(y, self.cent)  # [rows, NP * C]
        for part in range(self.np):
            _radix_topp_kernel[(rows,)](s[:, part * self.c:], self.sel[part], self.nxt[part],
                                        self.np * self.c, C=self.c, P=self.p, num_warps=4)
            _ivf_gather_kernel[(rows, self.p, self.ch)](
                y, self.weights[part], self.ids[part], self.offsets[part], self.sel[part],
                self.outv[part], self.outi[part], self.p, K=self.w.shape[1], CH=self.ch, BR=self.BR,
                BK=self.BK, READ_ONLY=False, PERM=True, num_warps=4)
        _ivf_local_kernel[(rows,)](self.outv, self.outi, self.nxt, self.loc, ids, self.m, self.max_rows,
                                   self.part_rows, NP=self.np, BM=min(1024, triton.next_power_of_2(self.m)),
                                   num_warps=4)
        _ivf_flag_kernel[(1,)](self.loc, self.flag, self.stats, rows, self.gap, RB=self.rb, num_warps=1)

    def _search_fast(self, y: torch.Tensor, ids: torch.Tensor, first: bool = False) -> None:
        """``_search`` in four launches (plan 71): centroid GEMV, both parts' top-P, both parts'
        gather, every row's shortlist max and the flag (cleared first on a step's first link).
        Same ``sel``, ``nxt``, ``loc``, ``ids``, flag and counters."""
        from engine.fast import mfma

        rows = y.shape[0]
        np_, c, p = self.np, self.c, self.p
        s = mfma.gemv(y, self.cent)  # [rows, NP * C]
        mr = self.max_rows
        _topp2_kernel[(rows, np_)](s, self.sel, self.nxt, np_ * c, mr * p, mr, C=c, P=p,
                                   HIST=self.topp_hist, num_warps=4)
        _ivf_gather2_kernel[(rows, np_ * p, self.ch)](
            y, self.weights[0], self.wdelta, self.ids_cat, self.off_cat, self.sel, self.outv, self.outi, p,
            mr * p, mr * self.m, self.part_rows, c + 1, K=self.w.shape[1], CH=self.ch, BR=self.BR, BK=self.BK,
            num_warps=4)
        rb = max(2, triton.next_power_of_2(rows))
        _ivf_local_flag_kernel[(1,)](
            self.outv, self.outi, self.nxt, self.loc, ids, self.flag, self.stats, rows, self.m, mr,
            self.part_rows, self.gap, NP=np_, RB=rb, BM=min(1024, max(16, 4096 // rb), triton.next_power_of_2(self.m)),
            FIRST=first, num_warps=4)

    def _full_pack(self, y: torch.Tensor, rank: int, offset: int, solo: bool) -> torch.Tensor:
        """When the flag is set: this rank's ``[rows, 4]`` argmax pack of its shard."""
        from engine.fast import kernels, mfma

        rows, k = y.shape
        n = self.w.shape[0]
        bn, bk, warps, _ = self.tiles_w
        logits = torch.empty((rows, n), dtype=y.dtype, device=y.device)
        tiles = triton.cdiv(n, bn)
        grid = tiles
        if self.fast and self.pred_grid > 0 and y.is_cuda:
            grid = min(tiles, self.pred_grid * self.cus)
        for r0 in range(0, rows, mfma.BM):
            r1 = min(rows, r0 + mfma.BM)
            if grid < tiles:
                _pred_gemv_p_kernel[(grid,)](
                    self.flag, y[r0:r1], self.w, logits[r0:r1], r1 - r0, n, n, tiles, K=k, BM=mfma.BM, BN=bn,
                    BK=bk, DOT_F32=kernels.INTERPRET, num_warps=warps)
                continue
            _pred_gemv_kernel[(triton.cdiv(n, bn),)](
                self.flag, y[r0:r1], self.w, logits[r0:r1], r1 - r0, n, n, K=k, BM=mfma.BM, BN=bn, BK=bk,
                DOT_F32=kernels.INTERPRET, num_warps=warps)
        pack = torch.empty((rows, 4), dtype=torch.float32, device=y.device)
        _pred_argmax_pack_kernel[(rows,)](self.flag, logits, pack, n, n, offset, RANK=rank, SOLO=solo,
                                          BLOCK=self.ARGMAX_BLOCK, num_warps=8)
        return pack

    def _warm(self) -> None:
        """Compile every kernel variant before any capture (both ranks', alone, both flags)."""
        y = torch.zeros((1, self.w.shape[1]), dtype=self.w.dtype, device=self.w.device)
        ids = torch.empty((1,), dtype=torch.int64, device=self.w.device)
        fast = self.fast
        for f in (0, 1):
            self.flag.fill_(f)
            self._search(y, ids)
            for rank in (0, 1):
                for solo in (False, True):
                    for self.fast in (False, True) if fast else (False,):
                        pack = self._full_pack(y, rank, 0, solo)
                    _ivf_final_kernel[(1,)](self.flag, pack, ids, 1, RB=self.rb, num_warps=1)
        self.fast = fast
        if fast:  # every row-tile size of the one-program shortlist/flag kernel, both FIRST
            rb = 2
            while True:
                rows = min(rb, self.max_rows)
                yy = torch.zeros((rows, self.w.shape[1]), dtype=self.w.dtype, device=self.w.device)
                ii = torch.empty((rows,), dtype=torch.int64, device=self.w.device)
                for first in (False, True):
                    self._search_fast(yy, ii, first)
                if rb >= self.max_rows:
                    break
                rb *= 2
        self.flag.zero_()
        if self.w.is_cuda:
            torch.cuda.synchronize(self.w.device)

    def draft(self, y: torch.Tensor, model, first: bool = False, out: torch.Tensor | None = None
              ) -> torch.Tensor:
        """Each row's draft token (int64 [rows]) from the head's normed output ``y`` (the same
        bits on both ranks). ``first``: the step's first link (clears the fallback flag).
        ``out``: a contiguous int64 [rows] to write the drafts into (returned)."""
        y = y.contiguous()
        rows = y.shape[0]
        tp, rank, offset = model.tp, model.rank, model.vocab_offset
        solo = tp is None or tp.solo
        ids = out if out is not None else torch.empty((rows,), dtype=torch.int64, device=y.device)
        assert ids.shape == (rows,) and ids.dtype == torch.int64 and ids.is_contiguous()
        if self.fast:
            self._search_fast(y, ids, first)
        else:
            if first:
                self.flag.zero_()
            self._search(y, ids)
        pack = self._full_pack(y, rank, offset, solo)
        if not solo:
            pack = tp.allreduce_pred(rank, pack, self.flag)  # both ranks run it, or neither
        _ivf_final_kernel[(1,)](self.flag, pack, ids, rows, RB=self.rb, num_warps=1)
        if AUDIT:
            full = model._argmax(model.fused.gemv(y, model.lm_head))
            self.stats[2:4] += torch.stack((torch.full_like(full[0], full.numel()), (full != ids).sum()))
        return ids
