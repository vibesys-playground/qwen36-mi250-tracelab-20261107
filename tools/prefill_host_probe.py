"""Plan 24 step 0: is a prefill pass bound by host work rather than GPU time?

    python tools/prefill_host_probe.py all [--devices 4,5] [--reps 5] [--tps 1,2] [--serve-settings ...]
    python tools/prefill_host_probe.py passes [--reps 5]      # one process, the visible GPUs
    python tools/prefill_host_probe.py serve [--settings default,ENGINE_TP=1] [serve_probe shape]

``passes`` loads the checkpoint as a replica does (``Generator.load``: warmup and captured
graphs included) on the visible GPUs and drives prefill through the server's own path:
``PackedPrefill.add`` then ``PackedPrefill.step``, as ``engine.server.Worker`` does. Cases
(synthetic token ids, the replay's synthetic code lines):

- ``resume N``: a 3,600-token context is prefilled cold, its request ends and is kept as a
  session tail (``Generation.retire``), then a prompt of that context plus N new tokens is
  admitted (tail hit, ``adopt``) and its one pass is the measured pass; N = 561, 1650, 2300.
- ``cold 1800``: a fresh 1,800-token prompt, one pass.

Per case: the synchronized wall time of the pass (median of ``--reps``), one pass under the
torch profiler (GPU kernel/copy count, summed kernel time and busy time per device and over
both, HIP runtime calls: launches, synchronizing calls and the time blocked in them, GPU time
by category, and which host ops ran while no GPU was busy), one pass with Python-level sync
counters (``Tensor.item/tolist/cpu/__bool__``, ``torch.cuda.synchronize``, event and stream
syncs: count and call site), and one under cProfile (the calling thread's top functions).

``serve`` runs ``tools/serve_probe.py``'s served run (``serve_and_run``) with a
``sitecustomize`` hook on ``PYTHONPATH`` that logs every ``PackedPrefill.step`` of every
replica (segments, their tokens and start positions, wall time) without changing what it
runs, then prints the share of passes with one segment, two, and more.

``all`` runs ``passes`` once per ``--tps`` value (ENGINE_TP) on ``--devices`` in a child
process each, then ``serve``. Prints ``RESULT`` lines (JSON) and a summary table with the kill
rule of plan 24: wall <= 1.3x GPU busy time under both TP settings -> park; single-segment
passes < 40% -> packs of two must be captured too.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import os
import pstats
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CONTEXT = 3600
RESUME = (561, 1650, 2300)
COLD = 1800
KILL_RATIO = 1.3
KILL_SINGLE = 0.40


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


# ============================================================================ kernel categories


def category(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("_gate_up_kernel", "_down_kernel", "moe", "topk", "_find_tile")):
        return "moe"
    if any(s in n for s in ("attn", "flash", "fmha", "sdpa", "attention")):
        return "attention"
    if any(
        s in n
        for s in (
            "chunk_", "gated_delta", "fused_recurrent", "l2norm", "solve_tril", "wy_", "kkt",
            "cumsum_", "recompute_w_u", "merge_16x16", "_gdn", "conv", "miopen", "naive_",
            "fwd_o", "fwd_h", "prepare_",
        )
    ):
        return "gdn"
    if any(s in n for s in ("cijk", "gemm", "gemv", "hipblaslt", "rocblas", "matmul", "_mt", "tensile")):
        return "gemm"
    if "ll_allreduce" in n or "bulk_ar" in n:
        return "allreduce"
    return "other"


# ============================================================================ trace analysis


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _length(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in intervals)


def _clip(intervals, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi]


def _complement(busy, lo, hi):
    gaps, at = [], lo
    for a, b in busy:
        if a > at:
            gaps.append((at, a))
        at = max(at, b)
    if at < hi:
        gaps.append((at, hi))
    return gaps


def is_sync_call(name: str) -> bool:
    """HIP runtime calls that block the host until the stream (or device) catches up: explicit
    synchronizes and the non-async copies (D2H, and H2D from pageable host memory)."""
    n = name.lower()
    if "async" in n:
        return False
    return "synchronize" in n or n in ("hipmemcpy", "hipmemcpywithstream", "hipmemcpydtoh", "cudamemcpy")


def analyze_trace(path: str, main_tid) -> dict:
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    window = [e for e in events if e.get("name") == "probe_pass" and e.get("ph") == "X"]
    # The CPU-side annotation (user_annotation); a GPU-side twin may exist too.
    cpu_windows = [e for e in window if e.get("cat") in ("user_annotation", "cpu_op")] or window
    w = cpu_windows[0]
    lo, hi = float(w["ts"]), float(w["ts"]) + float(w["dur"])
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    per_dev: dict[str, list[tuple[float, float]]] = defaultdict(list)
    ksum: Counter = Counter()
    kcount: Counter = Counter()
    ccount: Counter = Counter()
    csum: Counter = Counter()
    by_cat: Counter = Counter()
    by_name: Counter = Counter()
    name_count: Counter = Counter()
    for e in gpu:
        a, d = float(e["ts"]), float(e.get("dur", 0))
        dev = str((e.get("args") or {}).get("device", e.get("pid")))
        per_dev[dev].append((a, a + d))
        if e["cat"] == "kernel":
            ksum[dev] += d
            kcount[dev] += 1
            cat = category(e["name"])
            by_cat[cat] += d
            by_name[e["name"][:90]] += d
            name_count[e["name"][:90]] += 1
        else:
            ccount[dev] += 1
            csum[dev] += d
            by_cat["copy"] += d
    unions = {d: _union(v) for d, v in per_dev.items()}
    busy_any = _union([iv for v in per_dev.values() for iv in v])
    first_gpu = min((a for a, _ in busy_any), default=lo)
    last_gpu = max((b for _, b in busy_any), default=hi)
    # The pass: from its CPU start to the later of its CPU end and its last GPU work.
    end = max(hi, last_gpu)
    runtime = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")]
    rt_count: Counter = Counter()
    rt_time: Counter = Counter()
    for e in runtime:
        rt_count[e["name"]] += 1
        rt_time[e["name"]] += float(e.get("dur", 0))
    launches = sum(c for n, c in rt_count.items() if "launch" in n.lower())
    syncs = {n: c for n, c in rt_count.items() if is_sync_call(n)}
    sync_ms = sum(rt_time[n] for n in syncs) / 1e3
    # Host ops (outermost cpu_op per thread) running while no GPU was busy, inside the pass.
    idle = _complement(_clip(busy_any, lo, end), lo, end)
    ops = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"),
        key=lambda e: (e.get("tid"), float(e["ts"])),
    )
    outer: dict[object, list] = defaultdict(list)
    for e in ops:
        a, b = float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))
        lst = outer[e.get("tid")]
        if lst and a < lst[-1][1]:
            continue  # nested in the previous outermost op
        lst.append((a, b, e["name"]))
    tids = list(outer)
    main = main_tid if main_tid in outer else (max(tids, key=lambda t: len(outer[t])) if tids else None)
    gap_by_op: Counter = Counter()
    in_ops = 0.0
    if main is not None:
        for a, b, name in outer[main]:
            for g0, g1 in idle:
                o = min(b, g1) - max(a, g0)
                if o > 0:
                    gap_by_op[name] += o
                    in_ops += o
    idle_us = _length(idle)
    return {
        "prof_wall_ms": round((end - lo) / 1e3, 2),
        "cpu_return_ms": round((hi - lo) / 1e3, 2),
        "gpu_first_kernel_after_ms": round((first_gpu - lo) / 1e3, 2),
        "gpu_tail_after_cpu_ms": round(max(0.0, last_gpu - hi) / 1e3, 2),
        "kernels": dict(kcount),
        "kernels_total": sum(kcount.values()),
        "copies": dict(ccount),
        "kernel_sum_ms": {d: round(v / 1e3, 2) for d, v in ksum.items()},
        "copy_sum_ms": {d: round(v / 1e3, 2) for d, v in csum.items()},
        "busy_ms": {d: round(_length(_clip(u, lo, end)) / 1e3, 2) for d, u in unions.items()},
        "busy_any_ms": round(_length(_clip(busy_any, lo, end)) / 1e3, 2),
        "idle_any_ms": round(idle_us / 1e3, 2),
        "gpu_ms_by_category": {k: round(v / 1e3, 2) for k, v in by_cat.most_common()},
        "top_kernels": [
            {"name": n, "ms": round(t / 1e3, 2), "n": name_count[n], "cat": category(n)}
            for n, t in by_name.most_common(14)
        ],
        "runtime_launch_calls": launches,
        "runtime_sync_calls": syncs,
        "runtime_sync_blocked_ms": round(sync_ms, 2),
        "runtime_top": [
            {"name": n, "n": c, "ms": round(rt_time[n] / 1e3, 2)} for n, c in rt_count.most_common(12)
        ],
        "host_threads_with_ops": len(tids),
        "idle_overlap_main_thread_ops_ms": round(in_ops / 1e3, 2),
        "idle_main_thread_outside_ops_ms": round(max(0.0, idle_us - in_ops) / 1e3, 2),
        "idle_by_host_op": [{"op": n, "ms": round(t / 1e3, 2)} for n, t in gap_by_op.most_common(12)],
    }


# ============================================================================ python sync counters


class SyncCounter:
    """Counts host-synchronizing Python calls on CUDA tensors, by call site, while active."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.active = False
        self.count: Counter = Counter()
        self.sites: Counter = Counter()
        self._orig: list[tuple[object, str, object]] = []

    def _site(self) -> str:
        frames = traceback.extract_stack(limit=14)[:-2]
        for fr in reversed(frames):
            f = fr.filename
            if "/engine/" in f or "/fla/" in f or "/transformers/" in f:
                short = f.split("/engine/")[-1] if "/engine/" in f else f.split("site-packages/")[-1]
                return f"{short}:{fr.lineno} {fr.name}"
        return "other"

    def _wrap(self, owner, name: str, label: str, tensor_arg: bool) -> None:
        orig = getattr(owner, name)
        counter = self

        def wrapper(*args, **kwargs):
            if counter.active:
                t = args[0] if args else None
                if not tensor_arg or (getattr(t, "is_cuda", False)):
                    counter.count[label] += 1
                    counter.sites[f"{label} @ {counter._site()}"] += 1
            return orig(*args, **kwargs)

        self._orig.append((owner, name, orig))
        setattr(owner, name, wrapper)

    def install(self) -> None:
        torch = self.torch
        for name in ("item", "tolist", "cpu", "__bool__", "__int__", "__float__", "numpy"):
            self._wrap(torch.Tensor, name, f"Tensor.{name}", True)
        self._wrap(torch.cuda, "synchronize", "cuda.synchronize", False)
        self._wrap(torch.cuda.Event, "synchronize", "Event.synchronize", False)
        self._wrap(torch.cuda.Stream, "synchronize", "Stream.synchronize", False)

    def uninstall(self) -> None:
        for owner, name, orig in reversed(self._orig):
            setattr(owner, name, orig)
        self._orig.clear()


# ============================================================================ passes


def passes(args) -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from engine.generator import CHECKPOINT, Generator, PackedPrefill
    from engine.protocol import Sampling
    from serve_probe import token_pool

    tp = os.environ.get("ENGINE_TP", "2")
    print(
        f"passes: ENGINE_TP={tp} visible={os.environ.get('HIP_VISIBLE_DEVICES')} torch {torch.__version__} "
        f"hip {torch.version.hip} gpus {torch.cuda.device_count()}",
        flush=True,
    )
    t0 = time.perf_counter()
    gen = Generator.load(
        CHECKPOINT["model"],
        CHECKPOINT["revision"],
        max_model_len=262144,
        chunk_tokens=2048,
        prefix_cache_tokens=4194304,
        prefix_cache_entries=128,
        prefix_cache_bytes=16 * 2**30,
    )
    assert gen.fast is not None and gen.packing, "the fast path with packing did not load"
    result("load", tp=tp, seconds=round(time.perf_counter() - t0, 1), mtp=gen.fast.mtp is not None,
           graphs=gen.fast.use_graphs)
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 80_000, "synthetic")
    packer = PackedPrefill(gen)  # the server's own: one per worker loop
    one = Sampling(max_tokens=1, temperature=0.0, ignore_eos=True)
    devices = list(range(torch.cuda.device_count()))
    serial = [0]
    cached = [0]

    def sync() -> None:
        for d in devices:
            torch.cuda.synchronize(d)

    def run_one(ctx: int, new: int, wrap=None) -> tuple[float, float]:
        serial[0] += 1
        offset = (serial[0] * 7919) % (len(pool) - ctx - new - 1)
        ids = pool[offset : offset + ctx + new]
        if ctx:
            packer.add(tuple(ids[:ctx]), one)
            done = packer.step()
            g = done[0][1]
            assert not isinstance(g, Exception), g
            g.next_step()  # its one token: the request ends
            g.retire()  # kept as a session tail, as the server does at a request's end
        prompt = tuple(ids)
        sync()
        a = time.perf_counter()
        packer.add(prompt, one)
        b = time.perf_counter()
        if wrap is None:
            done = packer.step()
        else:
            done = wrap(packer.step)
        sync()
        c = time.perf_counter()
        g = done[0][1]
        assert not isinstance(g, Exception), g
        cached[0] = g.cached_tokens  # == ctx when resumed from the session tail
        g.next_step()
        gen._reclaim()
        return (b - a) * 1e3, (c - b) * 1e3

    cases = [("resume", CONTEXT, n) for n in RESUME] + [("cold", 0, COLD)]
    # Warm every case's shapes (autotune, MIOpen, allocator) before timing.
    for _ in range(2):
        for _, ctx, new in cases:
            run_one(ctx, new)
    sync()
    counter = SyncCounter()
    counter.install()
    tmp = tempfile.mkdtemp(prefix="pp-trace-")
    rows = []
    passes0 = gen.prefill_passes
    for label, ctx, new in cases:
        walls, adds = [], []
        for _ in range(args.reps):
            add_ms, step_ms = run_one(ctx, new)
            walls.append(step_ms)
            adds.append(add_ms)
        wall = statistics.median(walls)
        # One pass under the profiler.
        trace = os.path.join(tmp, f"tp{tp}-{label}-{new}.json")
        main_tid = threading.get_native_id()

        def profiled(step):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function("probe_pass"):
                    out = step()
                sync()
            prof.export_chrome_trace(trace)
            return out

        _, prof_step_ms = run_one(ctx, new, profiled)
        try:
            stats = analyze_trace(trace, main_tid)
        except Exception as exc:  # noqa: BLE001 - report and continue
            stats = {"error": f"{type(exc).__name__}: {exc}"}
        # One pass with the Python-level sync counters.
        counter.count.clear()
        counter.sites.clear()

        def counted(step):
            counter.active = True
            try:
                return step()
            finally:
                counter.active = False

        _, counted_ms = run_one(ctx, new, counted)
        py_syncs = dict(counter.count)
        sites = [{"site": s, "n": n} for s, n in counter.sites.most_common(12)]
        # One pass under cProfile (this thread only).
        prof_out = io.StringIO()

        def cprofiled(step):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return step()
            finally:
                pr.disable()
                ps = pstats.Stats(pr, stream=prof_out)
                ps.sort_stats("tottime")
                cprof_rows.clear()
                for (file, line, fn), (cc, nc, tt, ct, _) in sorted(
                    ps.stats.items(), key=lambda kv: -kv[1][2]
                )[:18]:
                    short = file.split("site-packages/")[-1].split("/engine/")[-1]
                    cprof_rows.append({"fn": f"{short}:{line} {fn}", "calls": nc, "tottime_ms": round(tt * 1e3, 2),
                                       "cumtime_ms": round(ct * 1e3, 2)})
                cprof_total[0] = sum(v[2] for v in ps.stats.values()) * 1e3

        cprof_rows: list = []
        cprof_total = [0.0]
        _, cprof_ms = run_one(ctx, new, cprofiled)
        busy_any = stats.get("busy_any_ms")
        ratio = round(wall / busy_any, 2) if busy_any else None
        row = {
            "tp": tp, "case": label, "context": ctx, "new_tokens": new,
            "wall_ms_median": round(wall, 2), "wall_ms_all": [round(x, 1) for x in walls],
            "add_ms_median": round(statistics.median(adds), 2), "cached_tokens": cached[0],
            "gpu_busy_any_ms": busy_any, "wall_over_gpu_busy": ratio,
            "profiled_pass_ms": round(prof_step_ms, 1), "counted_pass_ms": round(counted_ms, 1),
            "cprofile_pass_ms": round(cprof_ms, 1),
            "py_sync_calls": py_syncs, "py_sync_total": sum(py_syncs.values()), "py_sync_sites": sites,
            "cprofile_total_tottime_ms": round(cprof_total[0], 1), "cprofile_top": cprof_rows,
            **stats,
        }
        rows.append(row)
        result("pass", **row)
    counter.uninstall()
    result("passes_counted", tp=tp, passes=gen.prefill_passes - passes0,
           segments_per_pass=round((gen.prefill_segments) / max(1, gen.prefill_passes), 3))
    return 0


# ============================================================================ served run

HOOK = r'''
import importlib.abc, importlib.util, json, os, sys, time

_LOG = os.environ.get("PREFILL_PASS_LOG")


def _patch(module):
    cls = module.PackedPrefill
    orig = cls.step
    path = os.path.join(_LOG, f"passes-{os.environ.get('ENGINE_REPLICA_INDEX', 'x')}-{os.getpid()}.jsonl")
    sink = open(path, "a", buffering=1)

    def step(self):
        before = [(e, e.at) for e in self._pending]
        t0 = time.perf_counter()
        out = orig(self)
        dur = (time.perf_counter() - t0) * 1e3
        segs = []
        for e, at in before:
            if e.at != at:
                begin, end, _ = e.spans[at]
                segs.append([end - begin, begin])
        sink.write(json.dumps({"t": t0, "ms": round(dur, 2), "segs": segs, "waiting": len(before)}) + "\n")
        return out

    cls.step = step


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "engine.generator":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if _LOG:
    sys.meta_path.insert(0, _Finder())
'''


def summarize_passes(logdir: Path) -> dict:
    rows = []
    for f in sorted(logdir.glob("passes-*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return {"passes": 0}
    hist = Counter(len(r["segs"]) for r in rows)
    total = len(rows)
    tokens = [sum(n for n, _ in r["segs"]) for r in rows]
    ms_by = defaultdict(list)
    tok_by = defaultdict(list)
    for r, t in zip(rows, tokens):
        k = min(len(r["segs"]), 3)
        ms_by[k].append(r["ms"])
        tok_by[k].append(t)

    def p(v, q):
        v = sorted(v)
        return round(v[min(len(v) - 1, int(q * len(v)))], 1) if v else None

    resumed = sum(1 for r in rows for n, b in r["segs"] if b > 0)
    segs = sum(len(r["segs"]) for r in rows)
    return {
        "passes": total,
        "segments_hist": {str(k): v for k, v in sorted(hist.items())},
        "single_share": round(hist.get(1, 0) / total, 3),
        "two_share": round(hist.get(2, 0) / total, 3),
        "three_plus_share": round(sum(v for k, v in hist.items() if k >= 3) / total, 3),
        "single_share_of_segments": round(hist.get(1, 0) / max(1, segs), 3),
        "segments_resumed_share": round(resumed / max(1, segs), 3),
        "pass_ms_p50": {f"{k}{'+' if k == 3 else ''}seg": p(v, 0.5) for k, v in sorted(ms_by.items())},
        "pass_ms_p90": {f"{k}{'+' if k == 3 else ''}seg": p(v, 0.9) for k, v in sorted(ms_by.items())},
        "pass_tokens_p50": {f"{k}{'+' if k == 3 else ''}seg": p(v, 0.5) for k, v in sorted(tok_by.items())},
        "pass_tokens_p90": {f"{k}{'+' if k == 3 else ''}seg": p(v, 0.9) for k, v in sorted(tok_by.items())},
        "pass_seconds_total": round(sum(r["ms"] for r in rows) / 1e3, 1),
        "logs": len(list(logdir.glob("passes-*.jsonl"))),
    }


def serve(args) -> int:
    import serve_probe
    from engine.generator import CHECKPOINT

    hook = Path(tempfile.mkdtemp(prefix="pp-hook-"))
    (hook / "sitecustomize.py").write_text(HOOK)
    shape = argparse.Namespace(
        model=None, sessions=args.sessions, rounds=args.rounds, concurrency=args.concurrency,
        first=args.first, grow=args.grow, max_new=args.max_new, port=args.port, corpus="synthetic",
    )
    need = shape.sessions * (shape.first + math.ceil(shape.rounds) * shape.grow) + 1
    pool = serve_probe.token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], need, "synthetic")
    result("serve_shape", **vars(shape))
    out = {}
    for setting in args.settings.split(","):
        name = "".join(c if c.isalnum() else "_" for c in setting)
        logdir = Path(tempfile.mkdtemp(prefix=f"pp-log-{name}-"))
        saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "PREFILL_PASS_LOG")}
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(hook), saved["PYTHONPATH"]) if p)
        os.environ["PREFILL_PASS_LOG"] = str(logdir)
        try:
            serve_probe.serve_and_run(None if setting == "default" else setting, shape.port, shape, pool)
        except Exception as exc:  # noqa: BLE001 - report the pass log anyway
            print(f"serve: setting {setting} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        summary = summarize_passes(logdir)
        out[setting] = summary
        result("serve_passes", setting=setting, **summary)
    return 0


# ============================================================================ all


def run_all(args) -> int:
    started = time.time()
    subprocess.run(["sh", "-c", "rocm-smi --showuse --showmemuse 2>&1 | head -40"], check=False)
    rows: list[dict] = []
    serves: dict[str, dict] = {}
    for tp in args.tps.split(","):
        env = dict(os.environ)
        env["ENGINE_TP"] = tp
        env["HIP_VISIBLE_DEVICES"] = args.devices
        env.pop("CUDA_VISIBLE_DEVICES", None)
        cmd = [sys.executable, str(Path(__file__).resolve()), "passes", "--reps", str(args.reps)]
        print(f"all: {' '.join(cmd)} (ENGINE_TP={tp}, HIP_VISIBLE_DEVICES={args.devices})", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            if line.startswith("RESULT "):
                body = json.loads(line[7:])
                if body.get("kind") == "pass":
                    rows.append(body)
        proc.wait()
        print(f"all: passes ENGINE_TP={tp} exited {proc.returncode}", flush=True)
    if args.serve_settings:
        cmd = [sys.executable, str(Path(__file__).resolve()), "serve", "--settings", args.serve_settings]
        print(f"all: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            if line.startswith("RESULT "):
                body = json.loads(line[7:])
                if body.get("kind") == "serve_passes":
                    serves[body["setting"]] = body
        proc.wait()
        print(f"all: serve exited {proc.returncode}", flush=True)
    # Summary table.
    print("\n=== per-pass summary (wall: synchronized median; GPU: profiled pass) ===", flush=True)
    head = (f"{'TP':>2} {'case':>6} {'new':>5} {'wall ms':>8} {'GPU busy':>8} {'ratio':>5} {'krnls':>6} "
            f"{'launch':>6} {'rt sync':>7} {'py sync':>7} {'moe':>6} {'gdn':>6} {'attn':>6} {'gemm':>6} "
            f"{'other':>6} {'copy':>6}")
    print(head)
    for r in rows:
        c = r.get("gpu_ms_by_category", {})
        print(
            f"{r['tp']:>2} {r['case']:>6} {r['new_tokens']:>5} {r['wall_ms_median']:>8.1f} "
            f"{(r.get('gpu_busy_any_ms') or 0):>8.1f} {(r.get('wall_over_gpu_busy') or 0):>5.2f} "
            f"{r.get('kernels_total', 0):>6} {r.get('runtime_launch_calls', 0):>6} "
            f"{sum((r.get('runtime_sync_calls') or {}).values()):>7} {r.get('py_sync_total', 0):>7} "
            + " ".join(f"{c.get(k, 0):>6.1f}" for k in ("moe", "gdn", "attention", "gemm", "other", "copy")),
            flush=True,
        )
    by_tp = defaultdict(list)
    for r in rows:
        if r.get("wall_over_gpu_busy") is not None:
            by_tp[r["tp"]].append(r["wall_over_gpu_busy"])
    gpu_bound = bool(by_tp) and all(max(v) <= KILL_RATIO for v in by_tp.values())
    at561 = {r["tp"]: r.get("wall_over_gpu_busy") for r in rows if r["new_tokens"] == 561}
    result(
        "kill_ratio",
        threshold=KILL_RATIO,
        ratio_at_561=at561,
        max_ratio_by_tp={k: max(v) for k, v in by_tp.items()},
        verdict="PARK (GPU-bound under both TP)" if gpu_bound else "CONTINUE (host gap above 1.3x somewhere)",
    )
    for setting, s in serves.items():
        share = s.get("single_share")
        result(
            "kill_single",
            setting=setting,
            threshold=KILL_SINGLE,
            single_share=share,
            verdict=(None if share is None else
                     "packs of two must be captured too" if share < KILL_SINGLE else "single-segment majority"),
        )
    print(f"all: done in {time.time() - started:.0f} s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_all = sub.add_parser("all")
    p_all.add_argument("--devices", default="4,5")
    p_all.add_argument("--tps", default="1,2")
    p_all.add_argument("--reps", type=int, default=5)
    p_all.add_argument("--serve-settings", default="default,ENGINE_TP=1")
    p_pass = sub.add_parser("passes")
    p_pass.add_argument("--reps", type=int, default=5)
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--settings", default="default")
    p_serve.add_argument("--sessions", type=int, default=48)
    p_serve.add_argument("--concurrency", type=int, default=8)
    p_serve.add_argument("--rounds", type=float, default=6.3)
    p_serve.add_argument("--first", type=int, default=1800)
    p_serve.add_argument("--grow", type=int, default=60)
    p_serve.add_argument("--max-new", type=int, default=190)
    p_serve.add_argument("--port", type=int, default=18737)
    args = parser.parse_args()
    if args.cmd == "all":
        return run_all(args)
    if args.cmd == "passes":
        return passes(args)
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
