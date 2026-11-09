"""Plan 47's IVF draft index (``engine.fast.draft_ivf``): the balanced build is deterministic
and respects its cap; with every cluster searched the IVF draft is the exact shard argmax;
``search_rank`` agrees with the torch reference draft; the Triton gather/pick kernels match
the reference (on a GPU, or under ``TRITON_INTERPRET=1``).

Also a script without pytest (``python tests/test_draft_ivf.py``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fast import draft_ivf  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cuda" if torch.cuda.is_available() and not INTERPRET else "cpu"


def _weights(n: int = 50_000, k: int = 256, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((n, k), generator=g) * 0.05).to(torch.bfloat16).to(DEV)


def test_build_deterministic_and_capped() -> None:
    w = _weights()
    for spherical in (False, True):
        a = draft_ivf.build(w, 512, 160, iters=4, spherical=spherical)
        b = draft_ivf.build(w, 512, 160, iters=4, spherical=spherical)
        assert torch.equal(a.perm, b.perm) and torch.equal(a.cluster, b.cluster)
        sizes = torch.bincount(a.cluster, minlength=512)
        assert int(sizes.max()) <= 160 and int(sizes.sum()) == w.shape[0]
        assert torch.equal(a.offsets[1:].long(), torch.cumsum(sizes, 0))
        assert torch.equal(a.cluster[a.perm], torch.sort(a.cluster).values)


def test_build_skewed_overflow() -> None:
    """Many identical rows (as untrained vocabulary rows are) overflow their cluster's cap."""
    w = _weights(20_000, 128)
    w[:6000] = w[0]
    idx = draft_ivf.build(w, 256, 100, iters=4)
    sizes = torch.bincount(idx.cluster, minlength=256)
    assert int(sizes.max()) <= 100 and int(sizes.sum()) == 20_000
    assert idx.stats["moved"] >= 6000 - 100


def test_all_clusters_is_exact_and_rank_matches_reference() -> None:
    w = _weights()
    idx = draft_ivf.build(w, 512, 160, iters=4)
    g = torch.Generator().manual_seed(1)
    y = (torch.randn((32, w.shape[1]), generator=g)).to(torch.bfloat16).to(DEV)
    logits = (y.float() @ w.float().T).to(torch.bfloat16).float()
    best, row = logits.max(1)
    ref_best, ref_row = draft_ivf.reference_draft(y, w, idx, 512)
    assert torch.equal(ref_row, row)
    rank = draft_ivf.search_rank(y, idx, row)
    for p in (1, 8, 64):
        _, r = draft_ivf.reference_draft(y, w, idx, p)
        assert torch.equal(r == row, rank < p), p


def test_kernels_match_reference() -> None:
    if draft_ivf.triton is None or (DEV == "cpu" and not INTERPRET):
        return
    w = _weights(6000, 256)
    idx = draft_ivf.build(w, 128, 64, iters=3)
    g = torch.Generator().manual_seed(2)
    y = torch.randn((3, w.shape[1]), generator=g).to(torch.bfloat16).to(DEV)
    for p in (4, 16):
        d = draft_ivf.Draft(w, idx, p, rank=1, offset=1000, max_rows=4,
                            centroid_gemv=lambda a, c: (a.float() @ c.float().T).to(torch.bfloat16))
        both = d.pack(y)
        ref_best, ref_row = draft_ivf.reference_draft(y, w, idx, p)
        assert torch.equal(both[:, 3].long(), ref_row + 1000), (both, ref_row)
        assert torch.allclose(both[:, 2], ref_best, rtol=1e-2)
        assert torch.equal(both[:, :2], torch.zeros_like(both[:, :2]))


def test_served_draft_graph_and_fallback() -> None:
    """The served path alone (no TP): eager and captured-graph drafts equal the reference
    rule (the IVF pick, or the full argmax where the gap is under the threshold), with the
    gap set so that both branches occur."""
    if DEV != "cuda" or draft_ivf.triton is None:
        return
    from types import SimpleNamespace

    from engine.fast import mfma

    torch.manual_seed(3)
    w = (torch.randn((40_000, 2048), device="cuda") * 0.05).to(torch.bfloat16)
    y = (torch.randn((8, 2048), device="cuda")).to(torch.bfloat16)
    probe = draft_ivf.Served(w, max_rows=8, p=8, clusters=1024, gap=0.0)
    # the reference: IVF pick and gap per row
    s = draft_ivf.cluster_scores(y, probe.indices[0])
    order = torch.sort(s, dim=1, descending=True, stable=True).indices
    nxt = s.gather(1, order[:, 8:9])[:, 0]
    logits = (y.float() @ w.float().T).to(torch.bfloat16).float()
    full = logits.argmax(1)
    _, ivf = draft_ivf.reference_draft(y, w, probe.indices[0], 8)
    top = logits.gather(1, ivf[:, None])[:, 0]
    gaps = top - nxt
    sg = torch.sort(gaps).values
    mid = float((sg[3] + sg[4]) / 2)  # half the rows under the gap, none on the edge
    model = SimpleNamespace(tp=None, rank=0, vocab_offset=0, lm_head=w,
                            fused=SimpleNamespace(gemv=lambda a, b: mfma.gemv(a, b)),
                            _argmax=lambda lg: lg.argmax(-1))
    # a rank falls back as a whole: any row under the gap puts every row on the full pass,
    # and the flag holds for the rest of the step's chain (until the next first link)
    probe.gap = mid
    assert torch.equal(probe.draft(y, model, first=True), full)
    probe.gap = -1e9
    assert torch.equal(probe.draft(y, model), full)
    assert torch.equal(probe.draft(y, model, first=True), ivf)
    assert torch.equal(probe.draft(y[:3], model), ivf[:3])
    probe.gap = float(sg[0]) - 1.0  # no row under the gap
    assert torch.equal(probe.draft(y, model, first=True), ivf) and int(probe.flag[0]) == 0
    st = probe.stats.tolist()
    assert st[0] == 8 + 8 + 8 + 3 + 8 and st[1] == 16, st
    # the same inside captured graphs: a falling-back chain and a clean one
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for gap, want in ((mid, full), (-1e9, ivf)):
        served = draft_ivf.Served(w, max_rows=8, p=8, clusters=1024, gap=gap)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            a = served.draft(y, model, first=True)
            b = served.draft(y, model)
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(a, want) and torch.equal(b, want), (gap, a, b, want)


def test_fast_search_matches_today() -> None:
    """Plan 71: ``Served._search_fast`` (both top-P algorithms) leaves the same selection, next
    score, shortlist maxima, drafts, flag and counters as ``_search``, at 1-8 rows, the flag
    starting at 0 or 1, first link or not, with tied scores (all-zero rows) included; and the
    whole draft through the fast path equals today's in a captured graph."""
    if DEV != "cuda" or draft_ivf.triton is None:
        return
    from types import SimpleNamespace

    from engine.fast import mfma

    torch.manual_seed(5)
    w = (torch.randn((40_000, 2048), device="cuda") * 0.05).to(torch.bfloat16)
    ys = (torch.randn((64, 2048), device="cuda")).to(torch.bfloat16)
    ys[:4] = 0
    served = draft_ivf.Served(w, max_rows=8, p=8, clusters=1024, gap=0.5)
    fast0 = served.fast
    for hist in (False, True):
        served.topp_hist = hist
        served.fast = True
        served._warm()
        for rows in (1, 2, 3, 8):
            for a in range(0, 64 - rows + 1, rows):
                y = ys[a:a + rows].contiguous()
                for f0, first in ((0, False), (1, False), (1, True)):
                    snaps = []
                    for which in ("old", "fast"):
                        served.flag.fill_(f0)
                        served.sel.zero_()
                        served.nxt.zero_()
                        served.loc.zero_()
                        st0 = served.stats.clone()
                        ids = torch.full((rows,), -7, dtype=torch.int64, device="cuda")
                        if which == "old":
                            if first:
                                served.flag.zero_()
                            served._search(y, ids)
                        else:
                            served._search_fast(y, ids, first)
                        snaps.append((served.sel[:, :rows].clone(), served.nxt[:, :rows].clone(),
                                      served.loc[:rows].clone(), ids.clone(), served.flag.clone(),
                                      served.stats - st0))
                    for x, z in zip(*snaps, strict=True):
                        assert torch.equal(x, z), (hist, rows, a, f0, first)
    served.topp_hist = draft_ivf.TOPP == "hist"
    model = SimpleNamespace(tp=None, rank=0, vocab_offset=0, lm_head=w,
                            fused=SimpleNamespace(gemv=lambda a, b: mfma.gemv(a, b)),
                            _argmax=lambda lg: lg.argmax(-1))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    outs = {}
    for fast in (False, True):
        served.fast = fast
        for gap in (-1e9, 1e9, 0.5):
            served.gap = gap
            graph = torch.cuda.CUDAGraph()
            buf = torch.zeros((8,), dtype=torch.int64, device="cuda")
            with torch.cuda.graph(graph, stream=stream):
                a = served.draft(ys[8:16], model, first=True)
                b = served.draft(ys[16:24], model, out=buf)
            graph.replay()
            torch.cuda.synchronize()
            outs[(fast, gap)] = (a.clone(), b.clone())
    served.fast = fast0
    for gap in (-1e9, 1e9, 0.5):
        assert all(torch.equal(x, z) for x, z in zip(outs[(False, gap)], outs[(True, gap)], strict=True)), gap


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
