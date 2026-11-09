"""Plan 67: session-carried acceptance, the slower decay and the served step price (host state only)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from engine import speculative as S
from engine.fast import mfma
from engine.generator import FastPrefixCache


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("ENGINE_SPEC_K", raising=False)
    monkeypatch.delenv("ENGINE_SPEC_COST", raising=False)
    monkeypatch.setattr(mfma, "KMAX", 7)
    monkeypatch.setattr(S, "CARRY", True)
    monkeypatch.setattr(S, "SERVED_COST", True)
    monkeypatch.setattr(S, "PRIOR", (0.8, 1.0))
    monkeypatch.setattr(S, "DECAY", 0.985)


def fast() -> NS:
    return NS(use_graphs=True, _rows=256, bucket=lambda n: next(b for b in (1, 2, 4, 8, 16, 32) if b >= n))


def gen(have: int = 3, need: int = 200) -> NS:
    return NS(speculates=True, _max_tokens=need, _count=0, queue=[], spec_checked=0.0,
              spec_accepted=0.0, draft=11, drafts=[11] + [12] * (have - 1))


def run(g: NS, steps: int, k: int, accepted: int) -> None:
    """Book ``steps`` steps of k drafts with ``accepted`` accepted, as ``_book`` does."""
    for _ in range(steps):
        checked = min(k, accepted + 1)
        g.spec_checked = g.spec_checked * S.DECAY + checked
        g.spec_accepted = g.spec_accepted * S.DECAY + accepted
        d = g.__dict__
        d["spec_recent_checked"] = d.get("spec_recent_checked", 0.0) * S.RECENT_DECAY + checked
        d["spec_recent_accepted"] = d.get("spec_recent_accepted", 0.0) * S.RECENT_DECAY + accepted
        S._depth_book(g, k, accepted)


def test_carry_round_trip():
    g = gen()
    run(g, 30, 5, 5)
    state = S.carry_state(g)
    h = gen()
    S.inherit(h, state)
    assert h.spec_checked == g.spec_checked and h.spec_accepted == g.spec_accepted
    assert h.spec_depth_checked == g.spec_depth_checked
    assert h.spec_depth_checked is not g.spec_depth_checked  # a copy: the rounds do not share it


def test_cold_request_keeps_prior():
    h = gen()
    S.inherit(h, None)
    assert h.spec_checked == 0.0 and "spec_depth_checked" not in h.__dict__
    assert S.acceptance(h) == pytest.approx(0.8)


def test_off_carries_nothing(monkeypatch):
    monkeypatch.setattr(S, "CARRY", False)
    g = gen()
    run(g, 30, 5, 5)
    assert S.carry_state(g) is None
    h = gen()
    S.inherit(h, {"checked": 9.0, "accepted": 9.0, "depth_checked": [1.0], "depth_accepted": [1.0]})
    assert h.spec_checked == 0.0


def test_carried_session_goes_deep_at_two_slots():
    a, b = gen(), gen()
    for g in (a, b):
        run(g, 40, 5, 5)
    fresh = [gen(), gen()]
    for g, src in zip(fresh, (a, b), strict=True):
        S.inherit(g, S.carry_state(src))
    assert S.choose_k(NS(fast=fast()), fresh) > 3


def test_rate_not_capped_near_0967():
    g = gen()
    run(g, 200, 7, 7)
    assert all(r > 0.98 for r in
               [(g.spec_depth_accepted[j] + S.PRIOR[0]) / (g.spec_depth_checked[j] + S.PRIOR[1]) for j in range(7)])


def test_one_miss_keeps_deep():
    a, b = gen(), gen()
    for g in (a, b):
        run(g, 40, 5, 5)
    run(a, 1, 5, 3)  # one rejection among good steps
    assert S.choose_k(NS(fast=fast()), [a, b]) > 3


def test_first_step_after_prefill_not_deep():
    g = gen(have=1)
    run(g, 40, 7, 7)
    assert S.choose_k(NS(fast=fast()), [g]) <= 3


def test_natural_text_falls_back_quickly():
    # a session that predicted well, then natural text: at most 5 steps before k <= 3
    g = gen()
    run(g, 60, 5, 5)
    steps = 0
    while S.choose_k(NS(fast=fast()), [g]) > 3:
        run(g, 1, 5, 1)
        steps += 1
        assert steps <= 3
    # and a cold request on natural text never goes deep
    h = gen()
    for _ in range(10):
        assert S.choose_k(NS(fast=fast()), [h]) <= 3
        run(h, 1, 3, 1)


def test_served_cost_ema():
    generator = NS()
    seed = S.cost(2, 5)
    for _ in range(200):
        S.learn_cost(generator, {"size": 2, "groups": 6, "gpu_ms": 13.0})
    assert S.cost(2, 5, generator) == pytest.approx(13.0, abs=0.01)
    assert S.cost(2, 3, generator) == S.cost(2, 3)  # untouched buckets keep the table
    S.learn_cost(generator, {"size": 2, "groups": 6, "gpu_ms": 10 * seed})  # a stall: dropped
    assert S.cost(2, 5, generator) == pytest.approx(13.0, abs=0.01)
    S.learn_cost(generator, {"size": 2, "groups": 6})  # untimed step
    assert generator._spec_cost_n[(2, 6 - 1)] == 200


class _Fast:
    def __init__(self) -> None:
        self.pool = NS(on_pressure=None)

    def detach(self, slot, n):
        return ("snap", slot, n)

    def snapshot(self, slot, n):
        return ("copy", slot, n)

    def snapshot_bytes(self, snap):
        return 1

    def release(self, snap):
        pass


def test_tail_carries_spec():
    cache = FastPrefixCache(_Fast(), chunk_tokens=4, max_tokens=10_000, max_entries=8)
    key = tuple(range(20))
    state = {"checked": 5.0, "accepted": 5.0, "depth_checked": [], "depth_accepted": []}
    cache.store_tail(key, 3, state)
    reused, snap, tail = cache.lookup(key + (99, 100))
    assert tail and reused == 20 and cache.last_spec is state
    cache.store(tuple(range(8)), 4)
    reused, snap, tail = cache.lookup(tuple(range(8)) + (50,))
    assert not tail and cache.last_spec is None
