"""Host CPU placement (engine.cpu_place) on synthetic topologies; CPU only, no sysfs needed."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import cpu_place  # noqa: E402
from engine.cpu_place import Topology, plan_slices  # noqa: E402


def make(nodes: int, cores_per_node: int, smt: int = 1, l3: int = 8, allowed=None) -> Topology:
    """``nodes`` x ``cores_per_node`` physical cores; SMT sibling k of core c is CPU
    c + k * nodes * cores_per_node (Linux's usual numbering); ``l3`` cores share an L3."""
    total = nodes * cores_per_node
    cpus = list(range(total * smt))
    topo = Topology(sorted(allowed) if allowed is not None else cpus)
    for cpu in cpus:
        core = cpu % total
        topo.node[cpu] = core // cores_per_node
        topo.siblings[cpu] = tuple(core + k * total for k in range(smt))
        first = core // l3 * l3
        topo.l3[cpu] = tuple(c + k * total for k in range(smt) for c in range(first, first + l3))
    return topo


def check_disjoint(slices) -> None:
    seen: set[int] = set()
    for s in slices:
        assert not seen & set(s.cpus)
        seen |= set(s.cpus)
        assert len(set(s.ranks)) == len(s.ranks)
        assert set(s.ranks) <= set(s.cpus)
        assert not set(s.rest) & set(s.ranks) or s.rest == s.cpus


def test_two_nodes_smt_four_pairs() -> None:
    topo = make(2, 64, smt=2)
    slices, leftover = plan_slices(topo, [0, 0, 1, 1], ranks=2)
    check_disjoint(slices)
    assert [s.node for s in slices] == [0, 0, 1, 1]
    cpu0_l3 = set(topo.l3[0])
    for s in slices:
        assert not set(s.cpus) & cpu0_l3  # CPU 0's L3 group is left out
        a, b = s.ranks
        assert topo.l3[a] == topo.l3[b]  # the rank cores share an L3
        # Dedicated physical cores: neither rank core nor its sibling is in the process mask.
        assert not set(topo.with_siblings(s.ranks)) & set(s.rest)
    # node 0 lost one L3 group of 8 cores: 56 cores for two replicas
    assert len(slices[0].cpus) == 2 * 28 and len(slices[2].cpus) == 2 * 32
    assert cpu0_l3 <= set(leftover)


def test_single_node() -> None:
    topo = make(1, 32)
    slices, _ = plan_slices(topo, [0, 0, 0, 0], ranks=2)
    check_disjoint(slices)
    assert all(len(s.cpus) == 6 for s in slices)  # (32 - 8) / 4


def test_unknown_gpu_nodes_split_in_order() -> None:
    topo = make(2, 16, l3=4)
    slices, _ = plan_slices(topo, [None, None, None, None], ranks=2)
    check_disjoint(slices)
    assert all(len(s.cpus) == 7 for s in slices)


def test_unbalanced_nodes_fall_back_to_even_split() -> None:
    topo = make(2, 4, l3=4)  # node 0 has no core outside CPU 0's L3 group
    plan = plan_slices(topo, [0, 0, 1, 1], ranks=2, avoid_cpu0_l3=True)
    assert plan is not None
    check_disjoint(plan[0])


def test_tiny_cpuset_is_refused() -> None:
    topo = make(1, 64, allowed=[5, 6, 7])
    assert plan_slices(topo, [0, 0], ranks=2) is None
    assert plan_slices(make(1, 4), [0, 0, 0], ranks=2) is None


def test_small_cpuset_uses_cpu0_group_rather_than_refusing() -> None:
    topo = make(1, 9, l3=9)  # one L3 group: only CPU 0's own core is left out
    slices, leftover = plan_slices(topo, [0, 0], ranks=2)
    check_disjoint(slices)
    assert all(len(s.cpus) == 4 for s in slices) and leftover == [0]
    slices, _ = plan_slices(make(1, 4, l3=4), [0, 0], ranks=2)  # no room even for that
    assert sorted(c for s in slices for c in s.cpus) == [0, 1, 2, 3]


def test_without_avoidance_cpu0_group_is_used() -> None:
    topo = make(2, 64)
    slices, _ = plan_slices(topo, [0, 0, 1, 1], ranks=2, avoid_cpu0_l3=False)
    assert slices[0].ranks == [0, 1]


def test_child_env_keeps_operator_pools(monkeypatch) -> None:
    topo = make(2, 16, l3=4)
    s = plan_slices(topo, [0, 1], ranks=2)[0][0]
    env = cpu_place.child_env({"OMP_NUM_THREADS": "7"}, s)
    assert env["OMP_NUM_THREADS"] == "7" and env["RAYON_NUM_THREADS"] == cpu_place.POOL_THREADS
    assert cpu_place.parse_list(env["ENGINE_CPUS"]) == s.cpus
    assert [int(c) for c in env["ENGINE_RANK_CPUS"].split(",")] == s.ranks


def test_list_round_trip() -> None:
    cpus = [0, 1, 2, 5, 7, 8, 9, 64]
    assert cpu_place.format_list(cpus) == "0-2,5,7-9,64"
    assert cpu_place.parse_list("0-2,5,7-9,64") == cpus


def test_pin_rank_pins_only_the_calling_thread(monkeypatch) -> None:
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two CPUs")
    monkeypatch.setenv("ENGINE_RANK_CPUS", f"{allowed[0]},{allowed[1]}")
    seen = {}

    def run() -> None:
        seen["core"] = cpu_place.pin_rank(1)
        seen["mask"] = os.sched_getaffinity(0)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert seen["core"] == allowed[1] and seen["mask"] == {allowed[1]}
    assert sorted(os.sched_getaffinity(0)) == allowed  # this thread is untouched
    monkeypatch.setenv("ENGINE_CPU_PIN", "0")
    assert cpu_place.pin_rank(0) is None
    monkeypatch.setenv("ENGINE_CPU_PIN", "1")
    assert cpu_place.pin_rank(5) is None  # no core for that rank


def test_read_topology_here() -> None:
    topo = cpu_place.read_topology()
    assert topo is not None and topo.allowed and topo.cores()
