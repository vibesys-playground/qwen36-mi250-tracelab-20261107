"""Plan 58: the slot-aware chain depth chooser (``engine.speculative.choose_k``) on host state only."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from engine import speculative as S
from engine.fast import mfma


def fast(n_rows: int = 256) -> NS:
    return NS(use_graphs=True, _rows=n_rows, bucket=lambda n: next(b for b in (1, 2, 4, 8, 16, 32) if b >= n))


def gen(p_depth=None, rate=None, need=200, steps=40.0, have=3) -> NS:
    g = NS(speculates=True, _max_tokens=need, _count=0, queue=[], spec_checked=0.0, spec_accepted=0.0,
           draft=11, drafts=[11] + [12] * (have - 1))
    if p_depth is not None:
        g.spec_depth_checked = [steps if j < len(p_depth) else 0.0 for j in range(max(S.KS))]
        g.spec_depth_accepted = [steps * p_depth[j] if j < len(p_depth) else 0.0 for j in range(max(S.KS))]
    if rate is not None:
        g.spec_checked, g.spec_accepted = 3 * steps, 3 * steps * rate
    return g


def choose(gens) -> int:
    return S.choose_k(NS(fast=fast()), gens)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("ENGINE_SPEC_K", raising=False)
    monkeypatch.delenv("ENGINE_SPEC_COST", raising=False)
    monkeypatch.setattr(mfma, "KMAX", 7)


REPLAY = [0.981, 0.959, 0.977]  # j268 1-slot k=3 conditionals, replay-like text
NATURAL = [0.835, 0.717, 0.794, 0.827, 0.713, 0.637, 0.692]  # j268 1-slot k=7, natural text


def test_depths_by_bucket():
    assert S.depths(1) == [1, 3, 4, 5, 7]
    assert S.depths(2) == [1, 3, 4, 5, 7]
    assert S.depths(4) == [1, 3]  # no measured cost row: plan 31's depths only
    assert S.depths(16) == [1, 3]
    assert S.depths(32) == [1]


def test_kmax3_is_plan31(monkeypatch):
    monkeypatch.setattr(mfma, "KMAX", 3)
    assert S.depths(1) == [1, 3]
    assert choose([gen(REPLAY, rate=0.97)]) == 3


def test_fresh_request_stays_at_3():
    assert choose([gen()]) == 3
    assert choose([gen(), gen()]) == 3


def test_high_acceptance_by_depth_goes_deep():
    assert choose([gen(REPLAY, rate=0.97)]) > 3
    assert choose([gen(REPLAY, rate=0.97), gen(REPLAY, rate=0.97)]) > 3


def test_falling_acceptance_stays_at_3():
    assert choose([gen(NATURAL, rate=0.79)]) == 3
    assert choose([gen(NATURAL[:3], rate=0.79)]) == 3
    assert choose([gen(NATURAL, rate=0.79), gen(NATURAL, rate=0.79)]) == 3


def test_few_tokens_left_stays_shallow():
    assert choose([gen(REPLAY, rate=0.97, need=4)]) <= 3


def test_four_slots_never_deep():
    assert choose([gen(REPLAY, rate=0.97) for _ in range(3)]) <= 3


def test_low_acceptance_k1():
    assert choose([gen([0.3, 0.3, 0.3], rate=0.3)]) == 1


def test_fixed_k(monkeypatch):
    monkeypatch.setenv("ENGINE_SPEC_K", "5")
    assert choose([gen()]) == 5
    assert choose([gen() for _ in range(3)]) == 3


def test_cost_table_forms(monkeypatch):
    monkeypatch.setenv("ENGINE_SPEC_COST", "1:9.2,3:13.7,7:23.2")  # the old 2-slot form
    assert S.cost(2, 3) == 13.7 and S.cost(1, 3) == 13.7 and S.cost(4, 1) == 9.2
    assert S.depths(1) == [1, 3]  # no 1-slot row: k <= 3 only... and 7 at 2 slots
    assert S.depths(2) == [1, 3, 7]


def test_depth_book_counts():
    g = gen()
    S._depth_book(g, 5, 2)  # drafts 1, 2 accepted, 3 rejected, 4-5 not checked
    assert g.spec_depth_checked[:5] == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert g.spec_depth_accepted[:5] == [1.0, 1.0, 0.0, 0.0, 0.0]
    S._depth_book(g, 3, 3)
    d = 1.0 * S.DECAY + 1.0
    assert g.spec_depth_checked[:4] == [d, d, d, 0.0]


def test_running_deep_stays_unless_beaten():
    # a batch whose last step ran k=4 keeps k=4 unless 5 or 7 beats it by the margin
    flat = [0.95] * 4
    assert choose([gen(flat, rate=0.95, have=4)]) in (4, 5, 7)
    assert choose([gen(NATURAL, rate=0.79, have=7)]) == 3  # no longer worth it: back to 3


def test_padding_positions_not_booked():
    g = gen()
    S._depth_book(g, 3, 5)  # a k=7 step with 3 real drafts, 5 accepted (padding happened to match)
    assert g.spec_depth_checked[:4] == [1.0, 1.0, 1.0, 0.0]
