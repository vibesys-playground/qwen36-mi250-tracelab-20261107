"""Plan 36 step 0: does placing a replica's host threads on its GCDs' CPU socket shorten the
host-bound TP2 prefill pass?

    python tools/cpu_place_probe.py all [--devices 0,1] [--reps 12]
    python tools/cpu_place_probe.py topo       # CPUs, NUMA nodes, each GCD's PCI address and node
    python tools/cpu_place_probe.py passes --topo <json>   # one process, ENGINE_TP=2

``topo`` records the CPU model, the visible core count and allowed cpuset, the NUMA nodes (sysfs
``/sys/devices/system/node``), the SMT siblings, and every visible GCD's PCI bus id (from HIP)
with its ``numa_node`` and ``local_cpulist`` (``/sys/bus/pci/devices/<bdf>``). From those it
slices each replica (GCD pairs 0-1, 2-3, 4-5, 6-7, as the router lays them out) a disjoint set of
physical cores on its pair's node: the first two are the rank threads' cores, the rest (with the
SMT siblings of the slice) host the process's other threads.

``passes`` loads a replica as the server does (``Generator.load``: warmup, graphs) on the two
visible GCDs and times one resumed prefill pass (3,600-token context, then N = 61, 561, 1,650
new tokens; ``TPModel.prefill_packed``, rank 1 on its thread as on the head), synchronized
wall, interleaved rep by rep, under five placements:

- ``unpinned``: every thread may run on any allowed CPU, default torch pool (today).
- ``local``: every thread of the process on the replica's slice of the GCDs' node; the rank-0
  thread (the caller) and the tp-rank1 thread each alone on one physical core of it; torch
  intra-op pool capped at 2.
- ``far``: the same on the other node.
- ``unpinned_bg``: as ``unpinned`` while 3 background processes (one per other GCD pair, two
  busy Python threads each, launching small kernels on their GCDs in a loop and synchronizing
  every 64 launches) stand in for the other replicas' host threads, all unpinned.
- ``local_bg``: as ``local`` with the background threads pinned to their own replicas' cores.

``all`` runs ``topo`` and ``passes`` in children and prints the table and the kill rules of
plan 36: park if local vs unpinned gains < 4% at 561 tokens both alone and under background
load, or if < 8 CPUs are allowed, or if there is a single NUMA node; NUMA placement is
irrelevant if far is < 3% slower than local.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CONTEXT = 3600
NEW = (61, 561, 1650)
MODES = ("unpinned", "local", "far", "unpinned_bg", "local_bg")
KILL_GAIN = 0.04
NUMA_IRRELEVANT = 0.03
MIN_CPUS = 8
POOL_CAP = 2
BG_LAUNCHES = 64


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


# =============================================================================== topology


def parse_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_topology() -> dict:
    allowed = sorted(os.sched_getaffinity(0))
    model = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    nodes: dict[int, list[int]] = {}
    base = Path("/sys/devices/system/node")
    for d in sorted(base.glob("node[0-9]*")) if base.exists() else []:
        cpus = parse_list(_read(str(d / "cpulist")) or "")
        nodes[int(d.name[4:])] = cpus
    siblings: dict[int, list[int]] = {}
    packages: dict[int, int] = {}
    for c in allowed:
        s = _read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        siblings[c] = parse_list(s) if s else [c]
        p = _read(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id")
        packages[c] = int(p) if p is not None else -1
    aset = set(allowed)
    node_allowed = {n: [c for c in cpus if c in aset] for n, cpus in nodes.items()}
    # Physical cores: one representative (the lowest allowed sibling) per sibling group.
    phys: dict[int, list[int]] = {}
    for n, cpus in node_allowed.items():
        reps = sorted({min(s for s in siblings[c] if s in aset) for c in cpus})
        phys[n] = reps
    return {
        "cpu_model": model,
        "os_cpu_count": os.cpu_count(),
        "allowed_count": len(allowed),
        "allowed": _compact(allowed),
        "smt_per_core": max((len(v) for v in siblings.values()), default=1),
        "packages": sorted(set(packages.values())),
        "nodes": {str(n): _compact(c) for n, c in nodes.items()},
        "nodes_allowed": {str(n): _compact(c) for n, c in node_allowed.items()},
        "phys_cores_allowed": {str(n): len(v) for n, v in phys.items()},
        "_phys": {str(n): v for n, v in phys.items()},
        "_siblings": {str(c): s for c, s in siblings.items()},
    }


def _compact(cpus: list[int]) -> str:
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def gcd_bus_ids() -> list[str]:
    """Each visible HIP device's PCI address (domain:bus:device.function), lower case."""
    import ctypes

    import torch

    n = torch.cuda.device_count()
    out = []
    hip = None
    for i in range(n):
        bdf = None
        try:
            if hip is None:
                lib = Path(torch.__file__).parent / "lib" / "libamdhip64.so"
                hip = ctypes.CDLL(str(lib) if lib.exists() else "libamdhip64.so")
            buf = ctypes.create_string_buffer(64)
            if hip.hipDeviceGetPCIBusId(buf, 64, i) == 0:
                bdf = buf.value.decode().lower()
        except (OSError, AttributeError):
            pass
        if bdf is None:
            p = torch.cuda.get_device_properties(i)
            bdf = f"{getattr(p, 'pci_domain_id', 0):04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        out.append(bdf)
    return out


def gcd_node(bdf: str) -> dict:
    dev = f"/sys/bus/pci/devices/{bdf}"
    node = _read(f"{dev}/numa_node")
    local = _read(f"{dev}/local_cpulist")
    return {"bdf": bdf, "numa_node": int(node) if node not in (None, "") else None, "local_cpulist": local}


def plan_slices(topo: dict, gcd_nodes: list[int | None]) -> dict:
    """Disjoint physical-core slices per replica (GCD pairs) on the pair's node; the rank cores
    are the first two of a slice. Also a slice for a replica-0-shaped process on the far node."""
    phys = {int(n): v for n, v in topo["_phys"].items()}
    pairs = [(2 * r, 2 * r + 1) for r in range(len(gcd_nodes) // 2)]
    rep_node = []
    for a, b in pairs:
        na, nb = gcd_nodes[a], gcd_nodes[b]
        rep_node.append(na if na is not None and na == nb else na if na is not None else nb)
    by_node: dict[int, list[int]] = defaultdict(list)
    for r, n in enumerate(rep_node):
        if n is not None:
            by_node[n].append(r)
    slices: dict[int, list[int]] = {}
    for n, reps in by_node.items():
        cores = phys.get(n, [])
        share = len(cores) // len(reps)
        for k, r in enumerate(reps):
            slices[r] = cores[k * share : (k + 1) * share]
    local = rep_node[0] if rep_node else None
    others = [n for n in phys if n != local and phys[n]]
    far = others[0] if others else None
    far_slice = None
    if far is not None:
        reps = by_node.get(far, [])
        far_slice = slices[reps[0]] if reps else phys[far][: max(2, len(phys[far]) // 2)]
    return {"replica_node": rep_node, "slices": {str(r): s for r, s in slices.items()},
            "local_node": local, "far_node": far, "far_slice": far_slice}


def with_siblings(topo: dict, cores: list[int]) -> list[int]:
    sib = topo["_siblings"]
    allowed = set(parse_list(topo["allowed"]))
    out = set()
    for c in cores:
        out.update(s for s in sib.get(str(c), [c]) if s in allowed)
    return sorted(out)


def topo_main(args) -> int:
    topo = cpu_topology()
    for cmd in ("lscpu", "numactl -H", "nproc", "cat /sys/fs/cgroup/cpuset.cpus.effective"):
        subprocess.run(["sh", "-c", f"echo '$ {cmd}'; {cmd} 2>&1 | head -40"], check=False)
    gcds = []
    try:
        gcds = [gcd_node(b) for b in gcd_bus_ids()]
    except Exception as exc:  # noqa: BLE001
        result("topo_gcd_error", error=f"{type(exc).__name__}: {exc}")
    plan = plan_slices(topo, [g["numa_node"] for g in gcds])
    out = {**topo, "gcds": gcds, "plan": plan}
    result("topo", **{k: v for k, v in out.items() if not k.startswith("_")})
    if args.out:
        Path(args.out).write_text(json.dumps(out))
    return 0


# =============================================================================== background


def _bg_proc(devices: str, mode, cores: list[int], allowed: list[int]) -> None:
    """One stand-in replica: two threads, each launching small kernels on its GCD in a loop."""
    import threading

    os.environ["HIP_VISIBLE_DEVICES"] = devices
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    import torch

    def loop(dev: int, core: int) -> None:
        torch.cuda.set_device(dev)
        x = torch.zeros(4096, device=f"cuda:{dev}")
        cur = -1
        while True:
            m = mode.value
            if m != cur:
                os.sched_setaffinity(0, {core} if m == 2 else set(allowed))
                cur = m
            if m == 3:
                return
            if m == 0:
                time.sleep(0.002)
                continue
            for _ in range(BG_LAUNCHES):
                x.add_(1.0)
            torch.cuda.synchronize(dev)

    n = min(2, torch.cuda.device_count())
    threads = [threading.Thread(target=loop, args=(i, cores[i]), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# =============================================================================== passes


def _set_all_threads(cpus: set[int]) -> None:
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except OSError:
            pass


def _last_cpu(tid: int) -> int | None:
    s = _read(f"/proc/self/task/{tid}/stat")
    if not s:
        return None
    fields = s.rsplit(")", 1)[1].split()
    return int(fields[36])  # field 39 'processor' (fields after the comm start at field 3)


def passes(args) -> int:
    import multiprocessing as mp
    import threading

    topo = json.loads(Path(args.topo).read_text())
    plan = topo["plan"]
    allowed = parse_list(topo["allowed"])
    slices = {int(r): s for r, s in plan["slices"].items()}
    cpu_node = {}
    for n, cpus in topo["nodes"].items():
        for c in parse_list(cpus):
            cpu_node[c] = int(n)
    modes = [m for m in args.modes.split(",") if m]
    if plan["far_node"] is None or not plan["far_slice"]:
        modes = [m for m in modes if m != "far"]
    # Background stand-ins for replicas 1-3 (started before any pinning: they inherit the cpuset).
    ctx = mp.get_context("spawn")
    bg_mode = ctx.RawValue("i", 0)
    bg = []
    n_gcd = len(topo.get("gcds", []))
    for r in range(1, n_gcd // 2):
        cores = slices.get(r, [])[:2]
        if len(cores) < 2:
            continue
        p = ctx.Process(target=_bg_proc, args=(f"{2 * r},{2 * r + 1}", bg_mode, cores, allowed), daemon=True)
        p.start()
        bg.append((r, cores, p))
    result("background", replicas=[{"replica": r, "cores": c} for r, c, _ in bg])
    if not bg:
        modes = [m for m in modes if not m.endswith("_bg")]

    import torch

    from engine.generator import CHECKPOINT, Generator
    from serve_probe import token_pool

    print(f"passes: ENGINE_TP={os.environ.get('ENGINE_TP')} visible={os.environ.get('HIP_VISIBLE_DEVICES')} "
          f"torch {torch.__version__} hip {torch.version.hip} gpus {torch.cuda.device_count()} "
          f"threads {torch.get_num_threads()}", flush=True)
    t0 = time.perf_counter()
    gen = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
                         prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30)
    fast = gen.fast
    if fast is None or not hasattr(fast, "ranks") or len(fast.ranks) != 2:
        raise RuntimeError("the TP2 fast path (rank 1 on a thread) did not load")
    result("load", seconds=round(time.perf_counter() - t0, 1), graphs=fast.use_graphs)
    here = [gcd_node(b) for b in gcd_bus_ids()]
    result("visible_gcds", gcds=here)
    default_threads = torch.get_num_threads()
    tid0 = threading.get_native_id()
    tid1 = fast._pool.submit(threading.get_native_id).result()
    slot = gen._free_slots[-1]
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 40_000, "synthetic")
    devices = list(range(torch.cuda.device_count()))
    serial = [0]

    def placement(mode: str) -> dict:
        if mode.startswith("unpinned"):
            return {"rest": allowed, "rank0": None, "rank1": None, "threads": default_threads}
        sl = plan["far_slice"] if mode == "far" else slices[0]
        return {"rest": with_siblings(topo, sl), "rank0": sl[0], "rank1": sl[1], "threads": POOL_CAP}

    def apply(mode: str) -> None:
        p = placement(mode)
        _set_all_threads(set(p["rest"]))
        torch.set_num_threads(p["threads"])
        if p["rank0"] is not None:
            os.sched_setaffinity(0, {p["rank0"]})
            fast._pool.submit(os.sched_setaffinity, 0, {p["rank1"]}).result()
        bg_mode.value = 0 if not mode.endswith("_bg") else (1 if mode.startswith("unpinned") else 2)

    for m in modes:
        p = placement(m)
        result("placement", mode=m, rank0=p["rank0"], rank1=p["rank1"], rest=_compact(p["rest"]),
               rest_nodes=sorted({cpu_node.get(c) for c in p["rest"]}, key=str), torch_threads=p["threads"])

    def sync() -> None:
        for d in devices:
            torch.cuda.synchronize(d)

    def context(new: int) -> list[int]:
        serial[0] += 1
        off = (serial[0] * 7919) % (len(pool) - CONTEXT - new - 1)
        ids = pool[off : off + CONTEXT + new]
        fast.reset(slot)
        fast.prefill_packed([(slot, ids[:2048], 0)])
        fast.prefill_packed([(slot, ids[2048:CONTEXT], 2048)])
        return ids[CONTEXT:]

    where: dict[str, list[int]] = defaultdict(list)

    def timed(mode: str, new: int) -> float:
        apply(mode)
        new_ids = context(new)  # also lets the background threads reach their new state
        sync()
        a = time.perf_counter()
        fast.prefill_packed([(slot, new_ids, CONTEXT)])
        sync()
        ms = (time.perf_counter() - a) * 1e3
        if mode.startswith("unpinned"):
            for tid in (tid0, tid1):
                c = _last_cpu(tid)
                if c is not None:
                    where[mode].append(cpu_node.get(c, -1))
        return ms

    try:
        for _ in range(2):  # warm every shape and placement
            for new in NEW:
                for mode in modes:
                    timed(mode, new)
        walls: dict[tuple[str, int], list[float]] = defaultdict(list)
        for rep in range(args.reps):
            order = modes[rep % len(modes):] + modes[: rep % len(modes)]  # rotate: no fixed order bias
            for new in NEW:
                for mode in order:
                    walls[(mode, new)].append(timed(mode, new))
    finally:
        bg_mode.value = 3
        _set_all_threads(set(allowed))
    for (mode, new), v in sorted(walls.items(), key=lambda kv: (kv[0][1], MODES.index(kv[0][0]))):
        s = sorted(v)
        p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
        result("pass", mode=mode, new_tokens=new, context=CONTEXT, reps=len(v),
               wall_ms_median=round(statistics.median(v), 2), wall_ms_p90=round(p90, 2),
               wall_ms_min=round(min(v), 2), wall_ms_all=[round(x, 1) for x in v])
    for mode, nodes in where.items():
        cnt = defaultdict(int)
        for n in nodes:
            cnt[n] += 1
        result("unpinned_rank_thread_nodes", mode=mode, local_node=plan["local_node"],
               samples={str(k): c for k, c in cnt.items()})
    for _, _, p in bg:
        p.join(timeout=10)
    sys.stdout.flush()
    os._exit(0)


# =============================================================================== all


def _child(cmd: list[str], env: dict, timeout: int) -> list[dict]:
    print(f"all: {' '.join(cmd)}", flush=True)
    rows = []
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    start = time.time()
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith("RESULT "):
            try:
                rows.append(json.loads(line[7:]))
            except json.JSONDecodeError:
                pass
        if time.time() - start > timeout:
            proc.kill()
            print("all: child timed out", flush=True)
            break
    proc.wait()
    print(f"all: exited {proc.returncode}", flush=True)
    return rows


def run_all(args) -> int:
    started = time.time()
    me = str(Path(__file__).resolve())
    out = os.path.join(tempfile.mkdtemp(prefix="cpu-place-"), "topo.json")
    env = dict(os.environ)
    rows = _child([sys.executable, me, "topo", "--out", out], env, 600)
    topo = json.loads(Path(out).read_text())
    plan = topo["plan"]
    n_nodes = sum(1 for v in topo["nodes_allowed"].values() if v)
    gate = []
    if topo["allowed_count"] < MIN_CPUS:
        gate.append(f"{topo['allowed_count']} CPUs allowed (< {MIN_CPUS})")
    if n_nodes < 2:
        gate.append(f"{n_nodes} NUMA node(s) with allowed CPUs")
    print(f"\n=== topology ===\ncpu {topo['cpu_model']}; allowed {topo['allowed_count']} ({topo['allowed']}); "
          f"smt {topo['smt_per_core']}; nodes {topo['nodes_allowed']}; phys cores {topo['phys_cores_allowed']}",
          flush=True)
    for i, g in enumerate(topo.get("gcds", [])):
        print(f"GCD {i}: {g['bdf']} node {g['numa_node']} local_cpulist {g['local_cpulist']}", flush=True)
    print(f"replica nodes {plan['replica_node']}; slices {plan['slices']}; far slice {plan['far_slice']}",
          flush=True)
    result("gate", ok=not gate, reasons=gate)
    if any(len(v) < 2 for v in plan["slices"].values()) or 0 not in {int(k) for k in plan["slices"]}:
        result("gate_slices", ok=False, why="a replica slice has < 2 physical cores; local placement impossible")
        gate.append("slices too small")
    if "0" in plan["slices"] and len(plan["slices"]["0"]) >= 2:
        penv = dict(env)
        penv["HIP_VISIBLE_DEVICES"] = args.devices
        penv.pop("CUDA_VISIBLE_DEVICES", None)
        penv["ENGINE_TP"] = "2"
        rows += _child([sys.executable, me, "passes", "--topo", out, "--reps", str(args.reps),
                        "--modes", args.modes], penv, 5400)
    walls = {(r["mode"], r["new_tokens"]): r for r in rows if r["kind"] == "pass"}
    print("\n=== prefill pass wall, ms: median (p90), 3,600-token context, TP2 threads ===", flush=True)
    present = [m for m in MODES if any(k[0] == m for k in walls)]
    print(f"{'new':>5} " + " ".join(f"{m:>17}" for m in present), flush=True)
    for new in NEW:
        cells = []
        for m in present:
            r = walls.get((m, new))
            cells.append(f"{r['wall_ms_median']:>8.1f} ({r['wall_ms_p90']:>6.1f})" if r else f"{'-':>17}")
        print(f"{new:>5} " + " ".join(cells), flush=True)

    def med(m, n):
        r = walls.get((m, n))
        return r["wall_ms_median"] if r else None

    def gain(a, b, n):
        x, y = med(a, n), med(b, n)
        return None if x is None or y is None else round(1 - y / x, 4)

    for new in NEW:
        result("gain", new_tokens=new, local_vs_unpinned=gain("unpinned", "local", new),
               local_bg_vs_unpinned_bg=gain("unpinned_bg", "local_bg", new),
               far_slower_than_local=None if med("far", new) is None or med("local", new) is None
               else round(med("far", new) / med("local", new) - 1, 4),
               bg_cost_unpinned=None if med("unpinned_bg", new) is None or med("unpinned", new) is None
               else round(med("unpinned_bg", new) / med("unpinned", new) - 1, 4))
    g_alone, g_bg = gain("unpinned", "local", 561), gain("unpinned_bg", "local_bg", 561)
    far = med("far", 561), med("local", 561)
    best = max((g for g in (g_alone, g_bg) if g is not None), default=None)
    verdict = ("KILL" if gate or best is None or best < KILL_GAIN else "PASS")
    result("kill", threshold=KILL_GAIN, gain_alone_561=g_alone, gain_bg_561=g_bg, gate=gate,
           numa_irrelevant=None if None in far else (far[0] / far[1] - 1) < NUMA_IRRELEVANT, verdict=verdict)
    print(f"all: done in {time.time() - started:.0f} s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_all = sub.add_parser("all")
    p_all.add_argument("--devices", default="0,1")
    p_all.add_argument("--reps", type=int, default=12)
    p_all.add_argument("--modes", default=",".join(MODES))
    p_topo = sub.add_parser("topo")
    p_topo.add_argument("--out", default="")
    p_pass = sub.add_parser("passes")
    p_pass.add_argument("--topo", required=True)
    p_pass.add_argument("--reps", type=int, default=12)
    p_pass.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args()
    try:
        if args.cmd == "all":
            return run_all(args)
        if args.cmd == "topo":
            return topo_main(args)
        return passes(args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
