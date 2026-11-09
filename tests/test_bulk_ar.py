"""Plan 52: the bulk in-graph all-reduce (``tp._bulk_ar_kernel``) gives the LL kernel's bits.

Needs two GPUs that can reach each other (the two GCDs of one MI250); skipped otherwise. Both
ranks capture the same calls through ``Group.allreduce`` (which picks bulk for BF16 tensors of
at least ``BULK_MIN_WORDS`` words) and through LL alone, then replay them on fresh random data
(the same seed on both GPUs, so each rank also holds the peer's partial and the torch formula
can be checked); every output must be bitwise LL's and the formula's, with no spin timeout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fast import tp  # noqa: E402

HID = 2048
SIZES = ((512, False), (576, True), (1792, True), (2048, False), (64, True), (511, False))

needs_two = pytest.mark.skipif(
    tp.triton is None or not torch.cuda.is_available() or torch.cuda.device_count() < 2
    or not tp.BULK_AR, reason="needs two GPUs, Triton and ENGINE_TP_BULK_AR=1")


@needs_two
def test_bulk_allreduce_is_bitwise_ll() -> None:
    devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    group = tp.Group(devs)
    ll = group.ll
    assert ll is not None and ll.bulk
    torch.cuda.manual_seed_all(52)
    bufs = [{n: [torch.empty(n, HID, dtype=torch.bfloat16, device=d) for _ in range(3)]
             for n, _ in SIZES} for d in devs]
    bad = [torch.zeros(3, dtype=torch.int64, device=d) for d in devs]

    def body(r: int) -> None:
        for n, residual in SIZES:
            a, b, c = bufs[r][n]
            a.normal_()
            b.normal_()
            c.normal_()
            x = a if r == 0 else b
            res = c if residual else None
            got = group.allreduce(r, x, res)
            ll.bulk = False
            want = group.allreduce(r, x, res)
            ll.bulk = True
            ref = (a.float() + b.float()).bfloat16()
            if residual:
                ref = (c.float() + ref.float()).bfloat16()
            bad[r][0] += (got.view(torch.int16) != want.view(torch.int16)).sum()
            bad[r][1] += (got.view(torch.int16) != ref.view(torch.int16)).sum()
            bad[r][2] += (want.view(torch.int16) != ref.view(torch.int16)).sum()

    graphs = []
    for r in (0, 1):
        with torch.cuda.device(devs[r]):
            s = torch.cuda.Stream(devs[r])
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
                body(r)
            torch.cuda.synchronize(devs[r])
        graphs.append(g)
    epochs = [e.clone() for e in ll.b_epoch]
    for _ in range(50):
        for r in (0, 1):
            graphs[r].replay()
    for d in devs:
        torch.cuda.synchronize(d)
    assert group.timeouts() == 0
    assert [t.tolist() for t in bad] == [[0, 0, 0], [0, 0, 0]]
    # The bulk calls ran (50 replays x the 4 sizes at or above the threshold on CTA 0) ...
    assert all(int(e[0] - e0[0]) == 50 * 4 for e, e0 in zip(ll.b_epoch, epochs))
    # ... and 64- and 511-row calls did not: the first CTA past 512 rows' worth saw 3 sizes.
    hi = 512 * 1024 // tp.BULK_BW
    if hi < len(ll.b_epoch[0]):
        assert all(int(e[hi] - e0[hi]) == 50 * 3 for e, e0 in zip(ll.b_epoch, epochs))
