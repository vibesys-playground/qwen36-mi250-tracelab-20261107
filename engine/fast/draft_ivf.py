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
import time
from dataclasses import dataclass

import torch

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
              x: torch.Tensor, c: torch.Tensor, spherical: bool) -> tuple[torch.Tensor, int]:
    """Assign every row to one of its candidate clusters (best first) so no cluster holds more
    than ``cap`` rows; in each round the rows proposing to a cluster are taken best score
    first, ties to the lower row. Rows left after the candidates go, one at a time in row
    order, to the best-scoring cluster with room. Returns (assignment, rows moved off their
    first choice)."""
    n, rounds = cand.shape
    dev = score.device
    out = torch.full((n,), -1, dtype=torch.long, device=dev)
    fill = torch.zeros(clusters, dtype=torch.long, device=dev)
    rows = torch.arange(n, device=dev)
    for r in range(rounds):
        todo = rows[out < 0]
        if todo.numel() == 0:
            break
        cl = cand[todo, r]
        sc = score[todo, r]
        o1 = torch.argsort(-sc, stable=True)  # best first (ties: lower row, todo is sorted)
        o2 = torch.argsort(cl[o1], stable=True)  # group by cluster, keep that order
        order = o1[o2]
        cls = cl[order]
        counts = torch.bincount(cls, minlength=clusters)
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(order.numel(), device=dev) - starts[cls]
        ok = pos < (cap - fill)[cls]
        take = todo[order[ok]]
        out[take] = cls[ok]
        fill += torch.bincount(cls[ok], minlength=clusters)
    left = rows[out < 0].tolist()
    if left:
        half = None if spherical else 0.5 * (c * c).sum(1)
        for i in left:
            s = x[i] @ c.T
            if half is not None:
                s = s - half
            s = torch.where(fill < cap, s, torch.full_like(s, float("-inf")))
            j = int(torch.argmax(s))
            out[i] = j
            fill[j] += 1
    moved = int((out != cand[:, 0]).sum())
    return out, moved


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
    assign, moved = _balanced(score, cand, clusters, cap, x, c, spherical)
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
                 empty=int((sizes == 0).sum()), moved=moved, moved_frac=moved / n)
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
                           READ_ONLY: tl.constexpr):
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
        acc = tl.zeros([BR], dtype=tl.float32)
        for k0 in range(0, K, BK):
            cols = k0 + tl.arange(0, BK)
            w = tl.load(w_ptr + rows[:, None].to(tl.int64) * K + cols[None, :], mask=ok[:, None],
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
