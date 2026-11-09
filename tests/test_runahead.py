"""Plan 42: the run-ahead input kernel writes exactly what the host would build next.

``_chain_next_kernel`` turns a chained verify step's outputs and inputs into the next step's
packed inputs. Its reference is the host path: ``chain_step``'s bookkeeping (every pick
emitted, the ring advanced by the accepted count) followed by ``FastModel._chain_host``'s
layout, here written out on plain Python lists.
"""

from __future__ import annotations

import random

import pytest
import torch

from engine.fast import kernels

pytestmark = pytest.mark.skipif(
    kernels.triton is None or not torch.cuda.is_available(), reason="needs Triton on a GPU"
)


def _host_layout(tokens, drafts, slots, positions, banks, groups, size, width, dummy, ring, gdn_rows):
    host = [[0] * width for _ in range(4)]
    host[1] = [dummy] * width

    def ring_row(slot, bank, j):
        return slot + ((bank + j) % ring) * gdn_rows

    for j in range(groups):
        for b in range(size):
            host[3][j * size + b] = ring_row(dummy, 0, j)
        for b, slot in enumerate(slots):
            host[0][j * size + b] = tokens[b] if j == 0 else drafts[b][j - 1]
            host[1][j * size + b] = slot
            host[2][j * size + b] = positions[b] + j
            host[3][j * size + b] = ring_row(slot, banks[b], j)
    for r in range(groups * size, width):
        host[3][r] = dummy
    return host


@pytest.mark.parametrize("groups", [2, 4, 8])
@pytest.mark.parametrize("batch,size", [(1, 1), (2, 2), (3, 4), (5, 8)])
def test_chain_next_matches_host(groups: int, batch: int, size: int) -> None:
    from engine.fast.tp import _chain_next_kernel

    rng = random.Random(groups * 100 + batch)
    ring, gdn_rows, dummy = 8, 17, 16
    width = 8 * 8
    slots = rng.sample(range(16), batch)
    banks = [rng.randrange(ring) for _ in slots]
    positions = [rng.randrange(10, 5000) for _ in slots]
    tokens = [rng.randrange(1, 240_000) for _ in slots]
    drafts = [[rng.randrange(1, 240_000) for _ in range(groups - 1)] for _ in slots]
    packed = torch.tensor(
        _host_layout(tokens, drafts, slots, positions, banks, groups, size, width, dummy, ring, gdn_rows),
        dtype=torch.long, device="cuda",
    )
    # The step's outputs: picks per row, accepted counts, the head's next drafts.
    picks = [[rng.randrange(1, 240_000) for _ in range(groups)] for _ in slots]
    acc = [rng.randrange(groups) for _ in slots]
    nxt = [[rng.randrange(1, 240_000) for _ in range(groups - 1)] for _ in slots]
    out = [0] * (groups * size + size + (groups - 1) * size)
    for b in range(batch):
        for j in range(groups):
            out[j * size + b] = picks[b][j]
        out[groups * size + b] = acc[b]
        for j in range(groups - 1):
            out[groups * size + size + j * size + b] = nxt[b][j]
    out_t = torch.tensor(out, dtype=torch.long, device="cuda")
    _chain_next_kernel[(batch,)](out_t, packed, batch, size, width, ring, gdn_rows, G=groups, num_warps=1)
    want = _host_layout(
        [picks[b][acc[b]] for b in range(batch)], nxt, slots,
        [positions[b] + acc[b] + 1 for b in range(batch)],
        [(banks[b] + acc[b]) % ring for b in range(batch)],
        groups, size, width, dummy, ring, gdn_rows,
    )
    assert packed.cpu().tolist() == want
