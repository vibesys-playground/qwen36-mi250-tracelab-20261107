"""Plan 61: build and check the prefill graphs' tuned BLAS table (engine/fast/blas_tune.py).

    python tools/blas_tune_probe.py record --devices 0,1 --dir /tmp/p61   # the GEMMs the graphs run
    python tools/blas_tune_probe.py tune --dir /tmp/p61 --gpus 0,1,2,3,4,5,6,7  # -> /tmp/p61/table.csv
    python tools/blas_tune_probe.py time --dir /tmp/p61 --device 0        # default vs tuned per shape

``record`` loads one TP2 replica as the server does (``Generator.load``, server defaults) with
``ENGINE_PREFILL_BLAS_TUNE=record``: while ``capture_prefill`` warms and captures the 26
prefill graphs, TunableOp runs the default solutions and writes every distinct GEMM it meets
(op signature, transposes, m, n, k, leading dimensions) to ``untuned*.csv``. Model shapes only;
the warm inputs are the engine's synthetic ids.

``tune`` deduplicates those lines, splits them over ``--gpus`` and in one process per GPU runs
``torch.cuda.tunable.tune_gemm_in_file`` (TunableOp's offline tuner: random inputs of each
recorded shape and layout, every hipBLASLt and rocBLAS solution plus the default timed on the
device, the fastest kept), then merges the per-GPU results under one validator header into
``table.csv`` and prints the wall time. The table is what ``engine/fast/blas_tuned_gfx90a.csv``
ships.

``time`` times each plain GEMM of the table (``GemmTunableOp`` rows) with torch.mm on tensors of
the recorded layout: TunableOp off (the default heuristic) against TunableOp on with the table
(tuning off), CUDA events, median of ``--reps`` batches; it says when a call missed the table.
Prints ``RESULT`` lines: per shape and per prefill bucket (sum over distinct shapes).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def record(args) -> int:
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("untuned*.csv"):
        f.unlink()
    env = dict(os.environ, HIP_VISIBLE_DEVICES=args.devices, ENGINE_PREFILL_BLAS_TUNE="record",
               PYTORCH_TUNABLEOP_UNTUNED_FILENAME=str(out / "untuned%d.csv"),
               PYTORCH_TUNABLEOP_RECORD_UNTUNED="1")
    code = (
        "import time, torch; t = time.perf_counter()\n"
        "from engine.generator import CHECKPOINT, Generator\n"
        "g = Generator.load(CHECKPOINT['model'], CHECKPOINT['revision'], max_model_len=262144, "
        "chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)\n"
        "f = g.fast\n"
        "print(f'RESULT record load_s={time.perf_counter() - t:.0f} prefill_graphs={len(f.pg)} "
        "pg_error={f.pg_error} blas={f.pg_blas}', flush=True)\n"
    )
    proc = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, env=env, timeout=args.timeout)
    lines = _untuned(out)
    ops = defaultdict(int)
    for op, _ in lines:
        ops[op.split(",")[0]] += 1
    print(f"RESULT record exit={proc.returncode} distinct_gemms={len(lines)} by_op={dict(ops)}", flush=True)
    return proc.returncode


def _untuned(out: Path) -> list[tuple[str, str]]:
    """Distinct (op, params) of the recorded files, each with its full original line."""
    seen: dict[str, str] = {}
    for f in sorted(out.glob("untuned*.csv")):
        for line in f.read_text().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 2 or not cells[0] or cells[0] == "Validator":
                continue
            seen.setdefault(cells[0] + "," + cells[1], line.strip())
    return sorted(seen.items())


INNER = r"""
import sys, time, torch
part, res = sys.argv[1], sys.argv[2]
tunable = torch.cuda.tunable
tunable.set_filename(res, False)
tunable.write_file_on_exit(False)
tunable.enable(True)
tunable.tuning_enable(True)
tunable.record_untuned_enable(False)
t = time.perf_counter()
tunable.tune_gemm_in_file(part)
tunable.write_file(res)
print(f"RESULT tune_part {part} solutions={len(tunable.get_results())} s={time.perf_counter() - t:.0f}", flush=True)
"""


def tune(args) -> int:
    out = Path(args.dir)
    lines = _untuned(out)
    gpus = [g for g in args.gpus.split(",") if g]
    parts = [[] for _ in gpus]
    for i, (_, line) in enumerate(lines):
        parts[i % len(gpus)].append(line)
    procs = []
    t0 = time.perf_counter()
    env0 = dict(os.environ)
    env0.setdefault("PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS", str(args.duration_ms))
    env0.setdefault("PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS", str(args.iterations))
    for g, part in zip(gpus, parts):
        pf = out / f"part{g}.csv"
        pf.write_text("".join(line + "\n" for line in part))
        rf = out / f"res{g}.csv"
        if rf.exists():
            rf.unlink()
        env = dict(env0, HIP_VISIBLE_DEVICES=g)
        procs.append((g, rf, subprocess.Popen([sys.executable, "-u", "-c", INNER, str(pf), str(rf)], cwd=ROOT, env=env)))
    bad = 0
    for g, rf, p in procs:
        try:
            p.wait(timeout=max(1.0, args.timeout - (time.perf_counter() - t0)))
        except subprocess.TimeoutExpired:
            p.kill()
            print(f"RESULT tune gpu {g} timed out", flush=True)
            bad += 1
            continue
        bad += int(p.returncode != 0)
    wall = time.perf_counter() - t0
    validators: dict[str, str] | None = None
    rows: dict[str, str] = {}
    for g, rf, _ in procs:
        if not rf.exists():
            print(f"RESULT tune gpu {g} wrote no results", flush=True)
            continue
        v: dict[str, str] = {}
        for line in rf.read_text().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) >= 3 and cells[0] == "Validator":
                v[cells[1]] = cells[2]
            elif len(cells) >= 4:
                rows[cells[0] + "," + cells[1]] = line.strip()
        if validators is None:
            validators = v
        elif v != validators:
            print(f"RESULT tune gpu {g} validators differ: {v} vs {validators}", flush=True)
            bad += 1
    table = out / "table.csv"
    body = [f"Validator,{k},{v}" for k, v in (validators or {}).items()] + [rows[k] for k in sorted(rows)]
    table.write_text("".join(line + "\n" for line in body))
    kinds = defaultdict(int)
    for line in rows.values():
        name = line.split(",")[2].lower()
        kinds["hipblaslt" if "hipblaslt" in name else "rocblas" if "rocblas" in name else name] += 1
    print(f"RESULT tune gemms={len(lines)} solutions={len(rows)} gpus={len(gpus)} wall_s={wall:.0f} "
          f"failed={bad} winners={dict(kinds)} table={table}", flush=True)
    return int(bad > 0)


SIG = re.compile(r"^([nt])([nt])_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)")


def _operands(sig: str, dev):
    """torch.mm operands whose BLAS call is the column-major GEMM ``sig``:
    C[m x n] = op(A) op(B), i.e. row-major C^T[n x m] = op(B)^T op(A)^T."""
    import torch

    mt = SIG.match(sig.lower())
    if mt is None:
        return None
    ta, tb = mt.group(1), mt.group(2)
    m, n, k, lda, ldb, ldc = (int(mt.group(i)) for i in range(3, 9))
    if ldc != m:
        return None
    def mat(rows, cols, ld):
        return (torch.randn(rows, ld, dtype=torch.bfloat16, device=dev) * 0.05)[:, :cols]
    # A col-major (m x k if 'n', k x m if 't') = row-major [k, m] / [m, k] with row stride lda.
    a = mat(k, m, lda) if ta == "n" else mat(m, k, lda).t()   # op(A)^T as a [k, m] view
    b = mat(n, k, ldb) if tb == "n" else mat(k, n, ldb).t()   # op(B)^T as an [n, k] view
    return b, a, (m, n, k)


def time_shapes(args) -> int:
    import torch

    from engine.fast import blas_tune

    out = Path(args.dir)
    table = args.table or str(out / "table.csv")
    validators, rows = blas_tune.read_table(table)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    tunable = torch.cuda.tunable
    tunable.set_filename("/tmp/p61_time_scratch.csv", False)
    tunable.write_file_on_exit(False)
    tunable.tuning_enable(False)
    for f in out.glob("time_missed*"):
        f.unlink()
    os.environ["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"] = str(out / "time_missed%d.csv")
    tunable.enable(True)
    ok = tunable.read_file(table)
    tunable.enable(False)
    print(f"RESULT time table={table} rows={len(rows)} read={ok} validators_match="
          f"{all(blas_tune.current_validators().get(k) == v for k, v in validators.items())}", flush=True)

    def bench(fn) -> float:
        """GPU us per call: ``--inner`` calls captured in one graph (as the prefill graphs run
        them, no host cost), median over ``--reps`` replays."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.inner):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        per = []
        for _ in range(args.reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            e.synchronize()
            per.append(s.elapsed_time(e) * 1000.0 / args.inner)
        del graph
        return statistics.median(per)

    def missed_bytes() -> int:
        return sum(f.stat().st_size for f in out.glob("time_missed*"))

    by_rows = defaultdict(lambda: [0.0, 0.0, 0])
    total = [0.0, 0.0]
    skipped = 0
    for cells in rows:
        op, sig, sol = cells[0], cells[1], cells[2]
        if not op.startswith("GemmTunableOp_BFloat16"):
            skipped += 1
            continue
        got = _operands(sig, dev)
        if got is None:
            skipped += 1
            continue
        x, w, (m, n, k) = got
        tunable.enable(False)
        t_def = bench(lambda: torch.mm(x, w))
        tunable.enable(True)
        tunable.record_untuned_enable(True)
        before = missed_bytes()
        t_tun = bench(lambda: torch.mm(x, w))
        tunable.record_untuned_enable(False)
        tunable.enable(False)
        missed = missed_bytes() > before
        print(f"RESULT shape {sig} rows={n} out={m} k={k} default_us={t_def:.1f} tuned_us={t_tun:.1f} "
              f"saving_pct={100 * (t_def - t_tun) / t_def:.1f} pick={sol}{' MISSED_TABLE' if missed else ''}",
              flush=True)
        b = by_rows[n]
        b[0] += t_def
        b[1] += t_tun
        b[2] += 1
        total[0] += t_def
        total[1] += t_tun
    for n in sorted(by_rows):
        d, t, c = by_rows[n]
        print(f"RESULT bucket rows={n} shapes={c} default_us={d:.0f} tuned_us={t:.0f} saving_us={d - t:.0f}",
              flush=True)
    print(f"RESULT time total default_us={total[0]:.0f} tuned_us={total[1]:.0f} skipped={skipped}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--devices", default="0,1")
    p.add_argument("--dir", default="/tmp/p61")
    p.add_argument("--timeout", type=float, default=1500)
    p = sub.add_parser("tune")
    p.add_argument("--dir", default="/tmp/p61")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--timeout", type=float, default=2400)
    p.add_argument("--duration-ms", type=int, default=30)
    p.add_argument("--iterations", type=int, default=100)
    p = sub.add_parser("time")
    p.add_argument("--dir", default="/tmp/p61")
    p.add_argument("--table", default=None)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--inner", type=int, default=20)
    args = parser.parse_args()
    return {"record": record, "tune": tune, "time": time_shapes}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
