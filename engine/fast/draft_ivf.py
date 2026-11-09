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

import math
import os
import time
from dataclasses import dataclass

import torch

# The served draft (``Served``, used by ``FastModel._chain_head_op``): ENGINE_DRAFT_IVF=1 (default)
# picks each chained MTP draft from the top ``P`` of ``CLUSTERS`` spherical clusters of the rank's
# lm_head shard rows, and falls back to today's full-vocabulary pass for a row whose shortlist
# top logit is within ``GAP`` of the best unscanned centroid score (plan 47 stage 0, j237:
# P 64, gap 17.875; agreement with the full draft >= 99.36% at every depth on held-out natural
# text and prose). ENGINE_DRAFT_IVF=0 restores the full pass for every draft.
ENABLED = os.environ.get("ENGINE_DRAFT_IVF", "1") == "1"
P_SERVED = int(os.environ.get("ENGINE_DRAFT_IVF_P", "64"))
CLUSTERS = int(os.environ.get("ENGINE_DRAFT_IVF_CLUSTERS", "4096"))
CAP = 64
ITERS = 16
GAP = float(os.environ.get("ENGINE_DRAFT_IVF_GAP", "17.875"))

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
                           READ_ONLY: tl.constexpr, PERM: tl.constexpr = False):
        """Program (r, p, j): rows ``j*BR..`` of query r's p-th selected cluster; writes the
        chunk's (max bf16 logit, lowest vocab id at that max) to slot ``(r*P + p)*CH + j``.
        READ_ONLY: the same loads with no math (a gather-read ceiling)."""
        r = tl.program_id(0)
        p = tl.program_id(1)
        j = tl.program_id(2)
        c = tl.load(sel_ptr + r * P + p)
        start = tl.load(off_ptr + c)
        end = tl.load(off_ptr + c + 1)
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

    @triton.jit
    def _radix_topp_kernel(s_ptr, out_ptr, nxt_ptr, C: tl.constexpr, P: tl.constexpr):
        """out[r, :P] = indices of the P largest bf16 scores of s[r, :C] (ties to the lower
        index; in index order); nxt[r] = the best unselected score. A 16-step radix select on
        the scores' 16-bit order keys, then a compaction by prefix sums; one program per row
        (plan 47 stage 0c, j237: 12.3 us at 2 rows, exact against a stable sort)."""
        r = tl.program_id(0)
        i = tl.arange(0, C)
        b = tl.load(s_ptr + r * C + i)
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

    @triton.jit(do_not_specialize=["offset"])
    def _ivf_pack_kernel(outv_ptr, outi_ptr, nxt_ptr, pack_ptr, M, offset, RANK: tl.constexpr,
                         SOLO: tl.constexpr, BM: tl.constexpr):
        """pack[r] (6 fp32): this rank's (shortlist max, its lowest vocabulary id + offset) in
        columns 2 RANK, 2 RANK + 1 and its best unscanned centroid score in 4 + RANK; the
        peer's three columns 0 (summed by the all-reduce), or -inf / 0 alone (SOLO)."""
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
        base = pack_ptr + r * 6
        other = float("-inf") if SOLO else 0.0
        tl.store(base + 2 * RANK, m)
        tl.store(base + 2 * RANK + 1, (idx + offset).to(tl.float32))
        tl.store(base + 4 + RANK, tl.load(nxt_ptr + r))
        tl.store(base + 2 * (1 - RANK), other)
        tl.store(base + 2 * (1 - RANK) + 1, 0.0)
        tl.store(base + 4 + (1 - RANK), other)

    @triton.jit(do_not_specialize=["R", "gap"])
    def _ivf_decide_kernel(pack_ptr, ids_ptr, fb_ptr, flag_ptr, R, gap, RB: tl.constexpr):
        """Per row: the draft over both ranks (larger shortlist max, ties to rank 0) into
        ids, and whether it falls back (its max minus the best unscanned centroid score on
        either rank is below ``gap``); flag[0] = any row falls back."""
        r = tl.arange(0, RB)
        ok = r < R
        a0 = tl.load(pack_ptr + r * 6, mask=ok, other=0.0)
        a1 = tl.load(pack_ptr + r * 6 + 1, mask=ok, other=0.0)
        a2 = tl.load(pack_ptr + r * 6 + 2, mask=ok, other=0.0)
        a3 = tl.load(pack_ptr + r * 6 + 3, mask=ok, other=0.0)
        n0 = tl.load(pack_ptr + r * 6 + 4, mask=ok, other=0.0)
        n1 = tl.load(pack_ptr + r * 6 + 5, mask=ok, other=0.0)
        win = a0 >= a2
        top = tl.where(win, a0, a2)
        fb = (top - tl.maximum(n0, n1)) < gap
        tl.store(ids_ptr + r, tl.where(win, a1, a3).to(tl.int64), mask=ok)
        tl.store(fb_ptr + r, fb.to(tl.int32), mask=ok)
        tl.store(flag_ptr, tl.max((fb & ok).to(tl.int32), axis=0))

    @triton.jit(do_not_specialize=["R", "N"])
    def _pred_gemv_kernel(flag_ptr, x_ptr, w_ptr, out_ptr, R, N, stride_out, K: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
                acc = tl.dot(a, b, acc)
            m = mok[:, None] & nok[None, :]
            tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)

    @triton.jit(do_not_specialize=["V", "offset"])
    def _pred_argmax_pack_kernel(flag_ptr, l_ptr, both_ptr, V, stride, offset, RANK: tl.constexpr,
                                 SOLO: tl.constexpr, BLOCK: tl.constexpr):
        """``tp._argmax_pack_kernel`` run only when flag[0] != 0; the peer's columns -inf / 0
        when SOLO (no all-reduce follows)."""
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
    def _ivf_final_kernel(flag_ptr, both_ptr, ids_ptr, fb_ptr, R, RB: tl.constexpr):
        """ids[r] = the full-vocabulary pick (``tp.argmax_pick``'s rule) for rows that fall
        back, when the full pass ran."""
        if tl.load(flag_ptr) != 0:
            r = tl.arange(0, RB)
            ok = r < R
            fb = tl.load(fb_ptr + r, mask=ok, other=0) != 0
            a0 = tl.load(both_ptr + r * 4, mask=ok, other=0.0)
            a1 = tl.load(both_ptr + r * 4 + 1, mask=ok, other=0.0)
            a2 = tl.load(both_ptr + r * 4 + 2, mask=ok, other=0.0)
            a3 = tl.load(both_ptr + r * 4 + 3, mask=ok, other=0.0)
            full = tl.where(a0 >= a2, a1, a3).to(tl.int64)
            tl.store(ids_ptr + r, full, mask=ok & fb)


class Served:
    """One rank's served IVF draft: the index of its lm_head shard (built here, at load) and
    the launches of ``draft``. Reads the shard through the cluster permutation (no copy)."""

    BR = 16
    BK = 256
    ARGMAX_BLOCK = 8192

    def __init__(self, w: torch.Tensor, max_rows: int, p: int = P_SERVED, clusters: int = CLUSTERS,
                 gap: float = GAP) -> None:
        self.w, self.p, self.gap = w, p, gap
        with torch.cuda.device(w.device):
            self.index = build(w, clusters, CAP, iters=ITERS, spherical=True)
            idx = self.index
            self.cent = idx.centroids.contiguous()
            self.ids = idx.perm.to(torch.int32).contiguous()
            self.offsets = idx.offsets
            self.ch = max(1, math.ceil(idx.cap / self.BR))
            self.m = p * self.ch
            self.max_rows = max_rows
            dev = w.device
            self.sel = torch.empty((max_rows, p), dtype=torch.int64, device=dev)
            self.nxt = torch.empty((max_rows,), dtype=torch.float32, device=dev)
            self.outv = torch.empty((max_rows, self.m), dtype=torch.float32, device=dev)
            self.outi = torch.empty((max_rows, self.m), dtype=torch.int32, device=dev)
            self.fb = torch.empty((max_rows,), dtype=torch.int32, device=dev)
            self.flag = torch.zeros((1,), dtype=torch.int32, device=dev)
            self.rb = max(16, triton.next_power_of_2(max_rows))
            self._warm()
            if w.is_cuda:
                torch.cuda.empty_cache()  # the build's temporaries, before the KV pool is sized
        print(f"engine: IVF drafts on {w.device}: {clusters} clusters, P {p}, gap {gap}, "
              f"built in {idx.build_s:.1f}s (sizes {idx.stats['min']}..{idx.stats['max']})", flush=True)

    def _warm(self) -> None:
        """Compile every kernel variant before any capture (both ranks' and the alone ones)."""
        from engine.fast import mfma

        y = torch.zeros((1, self.w.shape[1]), dtype=self.w.dtype, device=self.w.device)
        one = torch.ones(1, dtype=torch.int32, device=self.w.device)
        for rank in (0, 1):
            for solo in (False, True):
                self._launch(y, rank, 0, solo, mfma)
                pack = torch.zeros((1, 6), dtype=torch.float32, device=self.w.device)
                self._decide(pack, 1)
                self._full(y, one, rank, 0, solo, mfma)
        torch.cuda.synchronize(self.w.device)

    def _launch(self, y, rank, offset, solo, mfma) -> torch.Tensor:
        rows = y.shape[0]
        s = mfma.gemv(y, self.cent)
        sel = self.sel[:rows]
        _radix_topp_kernel[(rows,)](s, sel, self.nxt, C=self.cent.shape[0], P=self.p, num_warps=4)
        _ivf_gather_kernel[(rows, self.p, self.ch)](
            y, self.w, self.ids, self.offsets, sel, self.outv, self.outi, self.p,
            K=self.w.shape[1], CH=self.ch, BR=self.BR, BK=self.BK, READ_ONLY=False, PERM=True,
            num_warps=4)
        pack = torch.empty((rows, 6), dtype=torch.float32, device=y.device)
        _ivf_pack_kernel[(rows,)](self.outv, self.outi, self.nxt, pack, self.m, offset, RANK=rank,
                                  SOLO=solo, BM=min(1024, triton.next_power_of_2(self.m)), num_warps=4)
        return pack

    def _decide(self, pack, rows) -> torch.Tensor:
        ids = torch.empty((rows,), dtype=torch.int64, device=pack.device)
        _ivf_decide_kernel[(1,)](pack, ids, self.fb, self.flag, rows, self.gap, RB=self.rb, num_warps=1)
        return ids

    def _full(self, y, flag, rank, offset, solo, mfma) -> torch.Tensor:
        """The predicated full pass: logits, this rank's argmax pack (rows that do not fall
        back get it too: the GEMV costs the same for them)."""
        rows, k = y.shape
        n = self.w.shape[0]
        bn, bk, warps, ksplit = mfma.tiles_for(n, k)
        logits = torch.empty((rows, n), dtype=y.dtype, device=y.device)
        for r0 in range(0, rows, mfma.BM):
            r1 = min(rows, r0 + mfma.BM)
            _pred_gemv_kernel[(triton.cdiv(n, bn),)](
                flag, y[r0:r1], self.w, logits[r0:r1], r1 - r0, n, n, K=k, BM=mfma.BM, BN=bn, BK=bk,
                num_warps=warps)
        both = torch.empty((rows, 4), dtype=torch.float32, device=y.device)
        _pred_argmax_pack_kernel[(rows,)](flag, logits, both, n, n, offset, RANK=rank, SOLO=solo,
                                          BLOCK=self.ARGMAX_BLOCK, num_warps=8)
        return both

    def draft(self, y: torch.Tensor, model) -> torch.Tensor:
        """Each row's draft token (int64 [rows]) from the head's normed output ``y``."""
        from engine.fast import mfma

        y = y.contiguous()
        rows = y.shape[0]
        tp, rank, offset = model.tp, model.rank, model.vocab_offset
        solo = tp is None or tp.solo
        pack = self._launch(y, rank, offset, solo, mfma)
        if not solo:
            pack = tp.allreduce(rank, pack)
        ids = self._decide(pack, rows)
        fb = self.fb[:rows]
        if y.is_cuda and torch.cuda.is_current_stream_capturing():
            both = self._full(y, self.flag, rank, offset, solo, mfma)
            if not solo:
                both = tp.allreduce_pred(rank, both, self.flag)
            _ivf_final_kernel[(1,)](self.flag, both, ids, fb, rows, RB=self.rb, num_warps=1)
            return ids
        full = model._argmax(model.fused.gemv(y, model.lm_head))  # eager: always (both ranks)
        return torch.where(fb.bool(), full, ids)
