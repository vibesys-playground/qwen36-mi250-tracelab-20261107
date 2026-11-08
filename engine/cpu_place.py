"""Host CPU placement of the replicas and their rank threads.

The prefill pass is host-bound: two rank threads launch thousands of kernels each. Left to
the scheduler, those threads migrate between cores and sockets and share cores with the other
replicas' threads; pinned to dedicated cores, the pass is shorter and steadier (plan 36 step 0).

The router gives each replica process a disjoint slice of physical cores
(``plan_slices``), preferring its GPUs' NUMA node and avoiding the L3 group of CPU 0, which
also takes the system's interrupts and housekeeping. The child process starts with its
affinity set to the slice minus its rank cores (``preexec``); the environment carries the slice
(``ENGINE_CPUS``) and one dedicated core per tensor-parallel rank (``ENGINE_RANK_CPUS``). Each
thread (or process) that drives rank ``r`` calls ``pin_rank(r)`` when it starts. Thread pools
(OpenMP / torch intra-op, Rayon) are capped in the child's environment unless the operator set
them.

Everything is read from the running system: the allowed cpuset (``os.sched_getaffinity``),
``/sys/devices/system/node``, each CPU's SMT siblings and L3 sharing list, and the GPUs' PCI
addresses and NUMA nodes from the KFD topology. ``ENGINE_CPU_PIN=0`` disables placement; an
unreadable topology or a slice under ``MIN_CORES`` physical cores leaves everything unpinned.
``ENGINE_CPU_PIN_AVOID_CPU0=0`` lets slices use CPU 0's L3 group too.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MIN_CORES = 2
POOL_THREADS = "2"
SYS_CPU = Path("/sys/devices/system/cpu")
SYS_NODE = Path("/sys/devices/system/node")
KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def enabled() -> bool:
    return os.environ.get("ENGINE_CPU_PIN", "1") != "0"


def parse_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def format_list(cpus) -> str:
    cpus = sorted(set(cpus))
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


@dataclass
class Topology:
    """The CPUs this process may use, with their NUMA node, SMT siblings and L3 group."""

    allowed: list[int]
    node: dict[int, int] = field(default_factory=dict)  # cpu -> NUMA node (-1: unknown)
    siblings: dict[int, tuple[int, ...]] = field(default_factory=dict)  # cpu -> its SMT group
    l3: dict[int, tuple[int, ...]] = field(default_factory=dict)  # cpu -> CPUs sharing its L3

    def cores(self) -> list[int]:
        """One CPU (the lowest allowed sibling) per physical core."""
        allowed = set(self.allowed)
        return sorted({min(c for c in self.siblings.get(cpu, (cpu,)) if c in allowed) for cpu in self.allowed})

    def with_siblings(self, cores) -> list[int]:
        allowed = set(self.allowed)
        return sorted({c for core in cores for c in self.siblings.get(core, (core,)) if c in allowed})


def read_topology() -> Topology | None:
    """The running system's topology, or None when it cannot be read."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None
    if not allowed:
        return None
    topo = Topology(allowed)
    for d in sorted(SYS_NODE.glob("node[0-9]*")) if SYS_NODE.exists() else []:
        text = _read(d / "cpulist")
        if text is None:
            continue
        for cpu in parse_list(text):
            topo.node[cpu] = int(d.name[4:])
    for cpu in allowed:
        topo.node.setdefault(cpu, -1)
        sib = _read(SYS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list")
        topo.siblings[cpu] = tuple(parse_list(sib)) if sib else (cpu,)
        for index in sorted((SYS_CPU / f"cpu{cpu}" / "cache").glob("index[0-9]*")):
            if _read(index / "level") == "3":
                shared = _read(index / "shared_cpu_list")
                if shared:
                    topo.l3[cpu] = tuple(parse_list(shared))
                break
    if 0 not in topo.l3:  # CPU 0 may be outside the cpuset: read its L3 group anyway
        for index in sorted((SYS_CPU / "cpu0" / "cache").glob("index[0-9]*")):
            if _read(index / "level") == "3":
                shared = _read(index / "shared_cpu_list")
                if shared:
                    topo.l3[0] = tuple(parse_list(shared))
                break
    return topo


def gpu_nodes() -> list[int | None]:
    """The NUMA node of each GPU in HIP's enumeration order (the KFD topology's GPU nodes),
    narrowed by HIP/ROCR/CUDA_VISIBLE_DEVICES when set to plain indices; [] when unknown."""
    if not KFD_NODES.exists():
        return []
    gpus: list[int | None] = []
    for d in sorted(KFD_NODES.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1 << 30):
        props = _read(d / "properties")
        if not props:
            continue
        kv = {}
        for line in props.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                kv[parts[0]] = int(parts[1])
        if kv.get("simd_count", 0) <= 0 or "location_id" not in kv:
            continue
        loc = kv["location_id"]
        bdf = f"{kv.get('domain', 0):04x}:{loc >> 8 & 0xFF:02x}:{loc >> 3 & 0x1F:02x}.{loc & 0x7:x}"
        node = _read(Path("/sys/bus/pci/devices") / bdf / "numa_node")
        gpus.append(int(node) if node and node.lstrip("-").isdigit() and int(node) >= 0 else None)
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        current = os.environ.get(name, "").strip()
        if current:
            ids = [s.strip() for s in current.split(",") if s.strip()]
            if not all(s.isdigit() and int(s) < len(gpus) for s in ids):
                return []
            gpus = [gpus[int(s)] for s in ids]
    return gpus


@dataclass
class Slice:
    cpus: list[int]  # every CPU of the slice (SMT siblings included)
    ranks: list[int]  # one dedicated physical core per rank
    rest: list[int]  # the slice minus the rank cores and their siblings (the process's mask)
    node: int | None


def plan_slices(
    topo: Topology,
    replica_nodes: list[int | None],
    ranks: int,
    avoid_cpu0_l3: bool = True,
) -> tuple[list[Slice], list[int]] | None:
    """Disjoint core slices, one per replica, and the leftover CPUs; None when a replica would
    get fewer than ``max(MIN_CORES, ranks)`` physical cores.

    Cores are ordered by node, then L3 group, then number, so a slice is made of whole L3
    groups where it can be and its first cores (the rank cores) share one. Replicas whose GPUs'
    node is known take cores of that node, split evenly among them; if that leaves a slice too
    small, or no node is known, all cores are split evenly in order. CPU 0's L3 group is left out
    when ``avoid_cpu0_l3`` and enough cores remain without it, else CPU 0's core alone."""
    n = len(replica_nodes)
    if n == 0:
        return None
    need = max(MIN_CORES, ranks)
    cores = topo.cores()
    excluded: set[int] = set()
    if avoid_cpu0_l3 and 0 in topo.l3:
        excluded = set(topo.l3[0]) | {0}
    usable = [c for c in cores if c not in excluded]
    if len(usable) < need * n and avoid_cpu0_l3:  # too few: leave out CPU 0's own core only
        usable = [c for c in cores if c != 0]
    if len(usable) < need * n:
        usable = cores

    def key(c: int) -> tuple[int, int, int]:
        group = topo.l3.get(c, (c,))
        return (topo.node.get(c, -1), min(group), c)

    usable.sort(key=key)

    def split(pool: list[int], count: int) -> list[list[int]]:
        share = len(pool) // count
        return [pool[k * share : (k + 1) * share] for k in range(count)]

    chosen: list[list[int]] | None = None
    nodes_known = all(r is not None for r in replica_nodes)
    if nodes_known:
        by_node: dict[int, list[int]] = defaultdict(list)
        for r, node in enumerate(replica_nodes):
            by_node[node].append(r)  # type: ignore[index]
        chosen = [[] for _ in range(n)]
        for node, reps in by_node.items():
            pool = [c for c in usable if topo.node.get(c) == node]
            for r, part in zip(reps, split(pool, len(reps))):
                chosen[r] = part
        if any(len(s) < need for s in chosen):
            chosen = None
    if chosen is None:
        chosen = split(usable, n)
    if any(len(s) < need for s in chosen):
        return None
    slices = []
    for r, part in enumerate(chosen):
        rank_cores = part[:ranks]
        cpus = topo.with_siblings(part)
        rest = [c for c in cpus if c not in set(topo.with_siblings(rank_cores))] or cpus
        nodes = {topo.node.get(c, -1) for c in part}
        slices.append(Slice(cpus, rank_cores, rest, nodes.pop() if len(nodes) == 1 else None))
    used = {c for s in slices for c in s.cpus}
    leftover = [c for c in topo.allowed if c not in used]
    return slices, leftover


# ----------------------------------------------------------------------------- the router


def replica_slices(layout: list[tuple[int, ...]]) -> tuple[list[Slice], list[int]] | None:
    """The router's plan for ``layout`` (device groups), logged; None leaves all unpinned."""
    if not enabled():
        print("cpu_place: ENGINE_CPU_PIN=0, host threads unpinned", flush=True)
        return None
    topo = read_topology()
    if topo is None:
        print("cpu_place: topology unreadable, host threads unpinned", flush=True)
        return None
    gpus = gpu_nodes()
    nodes: list[int | None] = []
    for devices in layout:
        ns = {gpus[d] if d < len(gpus) else None for d in devices}
        nodes.append(ns.pop() if len(ns) == 1 else None)
    ranks = max((len(d) for d in layout), default=1)
    avoid = os.environ.get("ENGINE_CPU_PIN_AVOID_CPU0", "1") != "0"
    plan = plan_slices(topo, nodes, max(ranks, 1), avoid)
    if plan is None:
        print(f"cpu_place: {len(topo.cores())} cores allowed are too few for {len(layout)} replicas "
              f"of {ranks} ranks, host threads unpinned", flush=True)
        return None
    slices, leftover = plan
    for i, s in enumerate(slices):
        print(f"cpu_place: replica {i} (GPUs {','.join(map(str, layout[i]))}, GPU node {nodes[i]}) "
              f"cpus {format_list(s.cpus)} node {s.node} rank cores {s.ranks}", flush=True)
    print(f"cpu_place: router on {format_list(leftover) or 'all allowed CPUs'}", flush=True)
    return slices, leftover


def child_env(env: dict[str, str], s: Slice) -> dict[str, str]:
    out = dict(env)
    out["ENGINE_CPUS"] = format_list(s.cpus)
    out["ENGINE_RANK_CPUS"] = ",".join(map(str, s.ranks))
    for name in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS"):
        if not env.get(name):
            out[name] = POOL_THREADS
    return out


def preexec(s: Slice):
    """``Popen`` preexec part: the child's (and so every thread's) mask, before ``exec``."""
    cpus = set(s.rest)

    def apply() -> None:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            pass

    return apply


def pin_router(leftover: list[int]) -> None:
    if leftover:
        try:
            os.sched_setaffinity(0, set(leftover))
        except OSError:
            pass


# ----------------------------------------------------------------------------- a replica


def rank_core(rank: int) -> int | None:
    text = os.environ.get("ENGINE_RANK_CPUS", "").strip()
    if not text or not enabled():
        return None
    cores = [int(c) for c in text.split(",") if c.strip().isdigit()]
    return cores[rank] if 0 <= rank < len(cores) else None


def pin_rank(rank: int) -> int | None:
    """Pin the calling thread to rank ``rank``'s dedicated core (Linux applies a pid-0
    ``sched_setaffinity`` to the calling thread only). Works the same in a rank's own process.
    Returns the core, or None when unpinned."""
    core = rank_core(rank)
    if core is None:
        return None
    try:
        os.sched_setaffinity(0, {core})
    except OSError as exc:
        print(f"cpu_place: rank {rank} not pinned to core {core}: {exc}", flush=True)
        return None
    print(f"cpu_place: rank {rank} thread on core {core}", flush=True)
    return core
