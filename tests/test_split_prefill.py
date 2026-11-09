"""Split prefill's host side (plan 41): the pair gate, process grouping, CPU slices."""

from __future__ import annotations

import threading
import time

from engine import cpu_place, router, split_prefill


def test_borrow_waits_for_the_turn_and_blocks_the_next() -> None:
    gate = split_prefill.Gate()
    gate.enter()  # the worker is in a decode turn
    got = []
    t = threading.Thread(target=lambda: got.append(gate.borrow()))
    t.start()
    time.sleep(0.05)
    assert not got  # waits for the turn to end, does not preempt
    assert not gate.may_prefill()  # a waiting borrower holds back the worker's prefill
    gate.leave()
    t.join(2)
    assert got == [True]
    entered = threading.Event()
    w = threading.Thread(target=lambda: (gate.enter(), entered.set()))
    w.start()
    time.sleep(0.05)
    assert not entered.is_set()  # lent: the worker's next turn waits
    gate.give_back()
    w.join(2)
    assert entered.is_set()
    gate.leave()


def test_no_borrow_during_a_prefill_turn_or_twice() -> None:
    gate = split_prefill.Gate()
    gate.enter()
    assert gate.may_prefill()
    assert gate.borrow() is False  # the helper is prefilling: run unsplit
    gate.leave()
    assert gate.borrow() is True  # idle: at once
    assert gate.borrow() is False  # never two owners
    gate.give_back()


def test_hosts_pair_two_gpu_replicas_only_when_on(monkeypatch) -> None:
    layout = [(0, 1), (2, 3), (4, 5), (6, 7)]
    monkeypatch.setattr(split_prefill, "ENABLED", True)
    assert router.hosts(layout) == [[0, 1], [2, 3]]
    assert router.hosts([(0, 1, 2, 3), (4, 5, 6, 7)]) == [[0], [1]]
    assert router.hosts([(), ()]) == [[0], [1]]
    monkeypatch.setattr(split_prefill, "ENABLED", False)
    assert router.hosts(layout) == [[0], [1], [2], [3]]


def test_merge_slices_keeps_rank_cores_in_order() -> None:
    a = cpu_place.Slice([1, 2, 3], [1, 2], [3], 0)
    b = cpu_place.Slice([4, 5, 6], [4, 5], [6], 0)
    m = cpu_place.merge([a, b])
    assert m.ranks == [1, 2, 4, 5] and m.cpus == [1, 2, 3, 4, 5, 6] and m.rest == [3, 6]
    assert m.node == 0
