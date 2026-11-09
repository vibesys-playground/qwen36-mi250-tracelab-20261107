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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
