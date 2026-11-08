"""Plan 34 step 0 (plan 25's probe, priorities inverted, plus CU-mask splits): can a TP2 replica
keep decoding while a prefill pass runs, with the pass barely slower?

    python tools/overlap_probe.py [--devices 0,1] [--ctx 3600] [--sizes 561,1650] [--passes 20]
                                  [--cu-splits 8,16,24]

Loads one 2-GCD replica as the server does (``Generator.load``: ``ENGINE_TP=2``, the MTP head
on, graphs captured, warmed up), then, with synthetic token ids:

Baselines (the serial engine today): on normal-priority streams, the served decode step
(``TPModel.verify``, k=1 MTP, graph replays) at B=1 and B=2 solo, and one packed prefill pass
(``TPModel.prefill_packed`` of one segment, ``--sizes`` new tokens on a ``--ctx``-token context
restored from a snapshot before each pass) solo.

Settings, each with its own per-GCD streams for decode (D) and for prefill (P):
  dechi   D at the greatest HIP priority, P at the least (plan 25's orientation, the control)
  equal   both at normal priority
  prehi   P at the greatest priority, D at the least (plan 34)
  cuN     ``hipExtStreamCreateWithCUMask``: D on the first N CUs of each GCD, P on the rest
          (disjoint; the KFD maps mask bits round-robin over shader engines, so a contiguous
          bit range is spread evenly). HIP gives a CU-masked stream no priority argument.
For each setting: D's and P's own solo numbers (a masked stream is slower alone), then the
overlap: a decode thread loops verify B=1 on D while a prefill thread runs ``--passes`` passes
per size on P (rank 1's half on TPModel's worker thread, its streams set the same way). Decode
steps whose midpoint falls inside a pass are "during". Reported:
  sd = mean decode step during a pass / baseline solo step,
  sp = mean overlapped pass / baseline solo pass,
the overlapped pass's logits against the solo pass's (each pass starts from the same restored
context), in-graph all-reduce timeouts (``Group.timeouts``), and a watchdog that reports a hang.
``prehi`` also runs verify B=2.

Whether captured decode graphs honour the stream priority: if they do, sd orders
dechi < equal < prehi (the graph replays sit on D; only D's priority changes between them).

Kill (plan 34): no setting has sp <= 1.03 and sd <= 2.5 at every size; any hang or timeout.
The TPOT projection is the planner's event simulation, fed with sd and sp from here.

Prints ``RESULT`` lines. A feasibility hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def _summ(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    return (f"n={len(xs)} mean={statistics.fmean(xs):.2f} p50={_pct(xs, 0.5):.2f} "
            f"p90={_pct(xs, 0.9):.2f} max={max(xs):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--ctx", type=int, default=3600)
    parser.add_argument("--sizes", default="561,1650")
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--solo-passes", type=int, default=6)
    parser.add_argument("--solo-steps", type=int, default=150)
    parser.add_argument("--cu-splits", default="8,16,24", help="decode CUs per GCD, disjoint split")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--hang-s", type=float, default=240.0, help="watchdog per run")
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_TP", "2")
    os.environ.setdefault("ENGINE_MTP", "1")
    os.environ.setdefault("ENGINE_TP_WAIT_S", "60")

    import torch

    from engine.fast import tp as tp_mod
    from engine.generator import CHECKPOINT, Generator

    started = time.perf_counter()
    g = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
        chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0,
    )
    fast = g.fast
    if fast is None or not hasattr(fast, "ranks"):
        raise SystemExit("RESULT error: the replica did not load TPModel (ENGINE_TP=2 on two GPUs)")
    devs = fast.devices_tp
    hip = tp_mod._hip()
    least, greatest = ctypes.c_int(), ctypes.c_int()
    with torch.cuda.device(devs[0]):
        hip.hipDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest))
    cus = [torch.cuda.get_device_properties(d).multi_processor_count for d in devs]
    print(f"RESULT load_s={time.perf_counter() - started:.0f} tp=2 devices={args.devices} "
          f"mtp={fast.mtp is not None} graphs={fast.use_graphs} graph_error={fast.graph_error} "
          f"capture_mode={fast.capture_mode} lockstep={tp_mod.LOCKSTEP} hip={torch.version.hip} "
          f"hip_priority_range least={least.value} greatest={greatest.value} "
          f"torch_priority_range={torch.cuda.Stream.priority_range()} cus={cus}", flush=True)
    if fast.mtp is None or not fast.use_graphs or fast.graph_error:
        raise SystemExit("RESULT error: need the MTP head and captured graphs")

    # ------------------------------------------------------------ streams
    def prio_stream(d: torch.device, prio: int):
        s = ctypes.c_void_p()
        with torch.cuda.device(d):
            rc = hip.hipStreamCreateWithPriority(ctypes.byref(s), ctypes.c_uint(1), ctypes.c_int(prio))
        if rc != 0:
            raise RuntimeError(f"hipStreamCreateWithPriority({prio}) failed: {rc}")
        return torch.cuda.ExternalStream(s.value, device=d)

    def mask_stream(d: torch.device, bits: list[int]):
        words = (max(bits) // 32) + 1
        arr = (ctypes.c_uint32 * words)()
        for b in bits:
            arr[b // 32] |= 1 << (b % 32)
        s = ctypes.c_void_p()
        with torch.cuda.device(d):
            rc = hip.hipExtStreamCreateWithCUMask(ctypes.byref(s), ctypes.c_uint32(words), arr)
        if rc != 0:
            raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {rc}")
        return torch.cuda.ExternalStream(s.value, device=d)

    def prio_of(stream) -> int:
        p = ctypes.c_int()
        hip.hipStreamGetPriority(ctypes.c_void_p(stream.cuda_stream), ctypes.byref(p))
        return p.value

    base = [prio_stream(d, 0) for d in devs]
    settings: list[tuple[str, list, list]] = [
        ("dechi", [prio_stream(d, greatest.value) for d in devs], [prio_stream(d, least.value) for d in devs]),
        ("equal", [prio_stream(d, 0) for d in devs], [prio_stream(d, 0) for d in devs]),
        ("prehi", [prio_stream(d, least.value) for d in devs], [prio_stream(d, greatest.value) for d in devs]),
    ]
    for n in [int(x) for x in args.cu_splits.split(",") if x]:
        try:
            dec = [mask_stream(d, list(range(n))) for d in devs]
            pre = [mask_stream(d, list(range(n, c))) for d, c in zip(devs, cus, strict=True)]
        except Exception as exc:  # noqa: BLE001
            print(f"RESULT cu{n} unavailable: {type(exc).__name__}: {exc}", flush=True)
            continue
        settings.append((f"cu{n}", dec, pre))
    for name, dec, pre in settings:
        print(f"RESULT streams {name} decode_priority={[prio_of(s) for s in dec]} "
              f"prefill_priority={[prio_of(s) for s in pre]}", flush=True)

    def use(streams: list, home: torch.device) -> None:
        """Make ``streams`` this thread's current streams (thread-local), ``home`` its device."""
        for s in streams:
            torch.cuda.set_stream(s)
        torch.cuda.set_device(home)

    def sync_all() -> None:
        for d in devs:
            torch.cuda.synchronize(d)

    # ------------------------------------------------------------ state
    vocab = int(fast.config.vocab_size)
    gen = torch.Generator().manual_seed(34)
    sizes = [int(s) for s in args.sizes.split(",")]
    total = args.ctx + max(sizes) + 1
    ids = torch.randint(1000, min(vocab, 150000), (total,), generator=gen).tolist()
    ctx_ids, new_ids = ids[: args.ctx], ids[args.ctx :]
    DEC = [0, 1]  # decode slots
    PRE = 2  # prefill slot
    SRC = 3  # context source
    with torch.inference_mode():
        fast.reset(SRC)
        pos = 0
        while pos < args.ctx:
            end = min(pos + args.chunk, args.ctx)
            fast.prefill(SRC, ctx_ids[pos:end], pos, ids[end])
            pos = end
        snap = fast.snapshot(SRC, args.ctx)
        fast.reset(SRC)
        for s in DEC:
            fast.reset(s)
            fast.restore(snap, s)
        for b in (1, 2):
            if b not in fast.ranks[0].vsegments[0].graphs:
                fast.capture(b, verify=True)
        sync_all()
    if fast.graph_error:
        raise SystemExit(f"RESULT error: capture failed: {fast.graph_error}")
    graphs_before = sorted(fast.ranks[0].vsegments[0].graphs)

    TOK, DRAFT = [ids[5], ids[6]], [ids[7], ids[8]]

    def step(b: int):
        """One served verify step: (t0, t1). The slots' state advances (timing only)."""
        t0 = time.perf_counter()
        fast.verify(TOK[:b], DRAFT[:b], DEC[:b], [args.ctx] * b)  # its picks .cpu() syncs D
        return t0, time.perf_counter()

    def decode_loop(streams, b: int, stop: threading.Event, limit: int, out: list, err: list) -> None:
        try:
            use(streams, devs[0])
            with torch.inference_mode():
                while not stop.is_set() and len(out) < limit:
                    out.append(step(b))
            for s in streams:
                s.synchronize()
        except BaseException as exc:  # noqa: BLE001
            err.append(f"decode: {type(exc).__name__}: {exc}")

    def pool_streams(streams) -> None:
        # rank 1's half of every eager pass runs on TPModel's worker thread: its streams too.
        fast._pool.submit(use, streams, devs[1]).result()

    def prefill_loop(streams, n: int, passes: int, out: list, err: list, logits_out: list) -> None:
        try:
            use(streams, devs[0])
            with torch.inference_mode():
                for _ in range(passes):
                    fast.reset(PRE)
                    fast.restore(snap, PRE)
                    for s in streams:
                        s.synchronize()
                    t0 = time.perf_counter()
                    logits = fast.prefill_packed([(PRE, new_ids[:n], args.ctx)], [None])
                    logits_out.append(logits.float().cpu())
                    streams[1].synchronize()
                    out.append((t0, time.perf_counter()))
                fast.reset(PRE)
                for s in streams:
                    s.synchronize()
        except BaseException as exc:  # noqa: BLE001
            err.append(f"prefill: {type(exc).__name__}: {exc}")

    def run_threads(targets: list[threading.Thread], label: str) -> None:
        for t in targets:
            if t.ident is None:
                t.daemon = True
                t.start()
        deadline = time.perf_counter() + args.hang_s
        for t in targets:
            t.join(max(0.0, deadline - time.perf_counter()))
        if any(t.is_alive() for t in targets):
            print(f"RESULT hang {label}: a thread did not finish in {args.hang_s:.0f}s "
                  f"(timeouts={fast.timeouts()}) -> KILL", flush=True)
            sys.stdout.flush()
            os._exit(3)

    def solo_decode(streams, b: int, steps: int, label: str) -> list[float]:
        stop = threading.Event()
        err: list = []
        run_threads([threading.Thread(target=decode_loop, args=(streams, b, stop, 10, [], err))], label)
        recs: list = []
        run_threads([threading.Thread(target=decode_loop, args=(streams, b, stop, steps, recs, err))], label)
        if err:
            raise SystemExit(f"RESULT error {label}: {err}")
        return [(r[1] - r[0]) * 1e3 for r in recs]

    def solo_prefill(streams, n: int, passes: int, label: str) -> tuple[list[float], list]:
        pool_streams(streams)
        err: list = []
        run_threads([threading.Thread(target=prefill_loop, args=(streams, n, 2, [], err, []))], label)
        recs: list = []
        lg: list = []
        run_threads([threading.Thread(target=prefill_loop, args=(streams, n, passes, recs, err, lg))], label)
        if err:
            raise SystemExit(f"RESULT error {label}: {err}")
        return [(r[1] - r[0]) * 1e3 for r in recs], lg

    # ------------------------------------------------------------ baselines (serial engine)
    sync_all()
    base_step: dict[int, float] = {}
    for b in (1, 2):
        w = solo_decode(base, b, args.solo_steps, f"base-decode B={b}")
        base_step[b] = statistics.fmean(w)
        print(f"RESULT base decode verify B={b} wall_ms {_summ(w)}", flush=True)
    base_pass: dict[int, float] = {}
    ref_logits: dict[int, torch.Tensor] = {}
    for n in sizes:
        w, lg = solo_prefill(base, n, args.solo_passes, f"base-prefill {n}")
        base_pass[n] = statistics.fmean(w)
        ref_logits[n] = lg[0]
        same = all(torch.equal(x, lg[0]) for x in lg)
        print(f"RESULT base prefill new={n} ctx={args.ctx} wall_ms {_summ(w)} repeatable_logits={same}",
              flush=True)

    # ------------------------------------------------------------ settings
    t_before = fast.timeouts()
    print(f"RESULT allreduce_timeouts_before={t_before}", flush=True)
    table: dict[tuple[str, int, int], tuple[float, float]] = {}
    overlapped = 0
    errors: list[str] = []
    for name, dec, pre in settings:
        w = solo_decode(dec, 1, args.solo_steps // 2, f"{name} solo-decode")
        print(f"RESULT {name} solo decode B=1 on D wall_ms {_summ(w)} "
              f"vs_base={statistics.fmean(w) / base_step[1]:.3f}", flush=True)
        for n in sizes:
            w, _ = solo_prefill(pre, n, 3, f"{name} solo-prefill {n}")
            print(f"RESULT {name} solo prefill new={n} on P wall_ms {_summ(w)} "
                  f"vs_base={statistics.fmean(w) / base_pass[n]:.3f}", flush=True)
        for b in ((1, 2) if name == "prehi" else (1,)):
            for n in sizes:
                pool_streams(pre)
                drecs: list = []
                precs: list = []
                lg: list = []
                err: list = []
                stop = threading.Event()
                dt = threading.Thread(target=decode_loop, args=(dec, b, stop, 10**9, drecs, err))
                pt = threading.Thread(target=prefill_loop, args=(pre, n, args.passes, precs, err, lg))
                dt.daemon = pt.daemon = True
                dt.start()
                time.sleep(0.05)  # decode in steady state before the first pass
                label = f"{name} B={b} new={n}"
                run_threads([pt], label)
                stop.set()
                run_threads([dt], label + " decode")
                if err:
                    errors.extend(err)
                    print(f"RESULT overlap {label} ERROR {err}", flush=True)
                    continue
                overlapped += len(precs)
                windows = [(p[0], p[1]) for p in precs]

                def inside(r, windows=windows) -> bool:
                    mid = 0.5 * (r[0] + r[1])
                    return any(a <= mid <= z for a, z in windows)

                dwall = [(r[1] - r[0]) * 1e3 for r in drecs if inside(r)]
                owall = [(r[1] - r[0]) * 1e3 for r in drecs if not inside(r)]
                pwall = [(p[1] - p[0]) * 1e3 for p in precs]
                busy = sum(z - a for a, z in windows)
                ldiff = max(float((x - ref_logits[n]).abs().max()) for x in lg)
                sd = statistics.fmean(dwall) / base_step[b] if dwall else float("inf")
                sp = statistics.fmean(pwall) / base_pass[n]
                table[(name, b, n)] = (sd, sp)
                print(f"RESULT overlap {label} passes={len(precs)} steps_during={len(dwall)} "
                      f"decode_rate_during={len(dwall) / busy if busy else 0:.1f}/s "
                      f"(base solo {1e3 / base_step[b]:.1f}/s)", flush=True)
                print(f"RESULT overlap {label} decode_during wall_ms {_summ(dwall)} | outside {_summ(owall)}",
                      flush=True)
                print(f"RESULT overlap {label} prefill wall_ms {_summ(pwall)}", flush=True)
                print(f"RESULT ratio {label} sd={sd:.3f} sp={sp:.3f} logits_maxdiff_vs_solo={ldiff:.3g}",
                      flush=True)
    sync_all()
    t_after = fast.timeouts()
    graphs_after = sorted(fast.ranks[0].vsegments[0].graphs)
    print(f"RESULT allreduce_timeouts_after={t_after} new={t_after - t_before} overlapped_passes={overlapped} "
          f"graphs_unchanged={graphs_before == graphs_after}", flush=True)

    # ------------------------------------------------------------ table and kill rules
    print("RESULT table setting B new sd sp ok(sd<=2.5,sp<=1.03)", flush=True)
    for (name, b, n), (sd, sp) in table.items():
        print(f"RESULT table {name} {b} {n} {sd:.3f} {sp:.3f} {sd <= 2.5 and sp <= 1.03}", flush=True)
    passing = [name for name, _, _ in settings
               if all((name, 1, n) in table and table[(name, 1, n)][0] <= 2.5 and table[(name, 1, n)][1] <= 1.03
                      for n in sizes)]
    kills = []
    if errors or t_after - t_before:
        kills.append(f"errors/timeouts ({errors}, {t_after - t_before})")
    if not passing:
        kills.append("no setting has sd <= 2.5 and sp <= 1.03 at every size")
    print(f"RESULT verdict settings_in_window={passing} -> "
          f"{'KILL: ' + '; '.join(kills) if kills else 'GO pending the TPOT projection'}", flush=True)


if __name__ == "__main__":
    main()
