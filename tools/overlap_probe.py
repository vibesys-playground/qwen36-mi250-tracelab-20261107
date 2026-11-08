"""Plan 25 step 0: can a replica run a prefill pass concurrently with its decode/verify loop?

    python tools/overlap_probe.py [--devices 0,1] [--ctx 3600] [--sizes 561,1650] [--passes 20]

Loads one 2-GCD replica as the server does (``Generator.load``: ``ENGINE_TP=2`` by default,
the MTP head on, graphs captured, warmed up), then, with synthetic token ids:

(a) solo decode: ``TPModel.decode`` and ``TPModel.verify`` (the served step) at B=1 and B=2,
    graph replays launched on high-priority streams (``torch.cuda.Stream(priority=-1)`` per
    GCD), each slot holding a ``--ctx``-token context. Per step: wall (launch to picks on the
    host), host launch time (until the last graph replay is enqueued: the GIL-sensitive part)
    and rank 0's device span (event before the input copy to event after its graph).
(b) solo prefill: ``TPModel.prefill`` of ``--sizes`` new tokens on top of the cached context
    (the context restored from a snapshot into the prefill slot before each pass), both ranks'
    threads on low-priority streams. Per pass: wall and rank 0's device span.
(c) both at once: a decode thread loops graph replays for B=1 (and B=2) slots on its
    high-priority streams while a prefill thread runs ``--passes`` passes per size into another
    slot on its low-priority streams (rank 1's half on the TP worker thread, its streams set the
    same way). Decode steps whose midpoint falls inside a pass are "during"; reported against
    (a). Passes against (b). In-graph all-reduce timeouts (``Group.timeouts``) before/after,
    and a watchdog that reports a hang. Correctness side check: each overlapped pass starts
    from the same restored context, so its logits are compared with the solo pass's.
    Variants: ``ENGINE_TP_SWITCH_S`` (Python's switch interval inside a prefill pass) 0.2 ms
    (the default) and 1 ms.
(d) projection with plan 25's arithmetic (see ``project``).

The two all-reduces cannot wait on each other: prefill (eager) rendezvouses its two host
threads through ``Group._box`` and a stream-ordered peer copy (no device spin); the captured
decode graphs use ``_LL``'s own fine-grained receive buffers and device epochs, which the eager
path never touches. A decode graph's spin waits only for the peer GCD's decode graph, which
waits only for CUs; prefill kernels never spin, so they always drain.

Prints ``RESULT`` lines. A feasibility hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
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


def project(solo_step: float, ov_step: float, solo_pass: float, ov_pass: float, share: float) -> float:
    """Plan 25's served gain (fraction of output tok/s).

    Today a stream stalls for its partner's whole pass: ``share`` of the wall (plan: 0.10-0.12,
    from j109's TTFT x passes per replica). Overlapped, during a partner pass of
    ``ov_pass`` (= solo_pass x k) the stream decodes at r = solo_step / ov_step of full speed,
    so its residual stall is k (1 - r) of a solo pass; its own passes take k - 1 longer (its
    TTFT), which it waits for. Wall cut = share (1 - k (1 - r) - (k - 1)).
    """
    k = ov_pass / solo_pass
    r = solo_step / ov_step
    cut = share * (1.0 - k * (1.0 - r) - (k - 1.0))
    return 1.0 / (1.0 - cut) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--ctx", type=int, default=3600)
    parser.add_argument("--sizes", default="561,1650")
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--solo-passes", type=int, default=8)
    parser.add_argument("--solo-steps", type=int, default=200)
    parser.add_argument("--switch", default="0.0002,0.001")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--hang-s", type=float, default=240.0, help="watchdog per overlap run")
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
    print(f"RESULT load_s={time.perf_counter() - started:.0f} tp=2 devices={args.devices} "
          f"mtp={fast.mtp is not None} graphs={fast.use_graphs} graph_error={fast.graph_error} "
          f"kv_pages={fast.pool.pages} priority_range={torch.cuda.Stream.priority_range()}", flush=True)
    if fast.mtp is None or not fast.use_graphs or fast.graph_error:
        raise SystemExit("RESULT error: need the MTP head and captured graphs")

    hi = [torch.cuda.Stream(d, priority=-1) for d in devs]
    lo = [torch.cuda.Stream(d, priority=0) for d in devs]
    print(f"RESULT streams hi_priority={[s.priority for s in hi]} lo_priority={[s.priority for s in lo]}",
          flush=True)

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
    gen = torch.Generator().manual_seed(25)
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
        # Plain decode graphs (load captured only verify, the served step) for (a).
        for b in (1, 2):
            fast.capture(b, verify=False)
        sync_all()
    if fast.graph_error:
        raise SystemExit(f"RESULT error: capture failed: {fast.graph_error}")
    graphs_before = {k: sorted(seg.graphs) for k, seg in (("d", fast.ranks[0].segments[0]),
                                                           ("v", fast.ranks[0].vsegments[0]))}

    # Instrument TPModel._run: the time the host finished enqueueing the replays.
    launch_end: list[float] = [0.0]
    end_events: list = [None]
    orig_run = fast._run

    def timed_run(size, rows, verify):
        out = orig_run(size, rows, verify)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream(devs[0]))
        end_events[0] = ev
        launch_end[0] = time.perf_counter()
        return out

    fast._run = timed_run
    TOK, DRAFT = [ids[5], ids[6]], [ids[7], ids[8]]

    def step(kind: str, b: int):
        """One decode/verify step: (t0, t1, host_ms, events, picks). The slots' state advances
        each step (timing only: the inputs stay fixed)."""
        ev0 = torch.cuda.Event(enable_timing=True)
        ev0.record(torch.cuda.current_stream(devs[0]))
        t0 = time.perf_counter()
        if kind == "verify":
            _, g1, g2, d1, d2 = fast.verify(TOK[:b], DRAFT[:b], DEC[:b], [args.ctx] * b)
            picks = (tuple(g1), tuple(g2), tuple(d1), tuple(d2))
        else:
            picks = tuple(fast.decode(TOK[:b], DEC[:b], [args.ctx] * b).argmax(-1).cpu().tolist())
        t1 = time.perf_counter()
        return t0, t1, (launch_end[0] - t0) * 1e3, (ev0, end_events[0]), picks

    def decode_loop(kind: str, b: int, stop: threading.Event, limit: int, out: list, err: list) -> None:
        try:
            use(hi, devs[0])
            with torch.inference_mode():
                while not stop.is_set() and len(out) < limit:
                    out.append(step(kind, b))
            torch.cuda.current_stream(devs[0]).synchronize()
        except BaseException as exc:  # noqa: BLE001
            err.append(f"decode: {type(exc).__name__}: {exc}")

    def lo_pool() -> None:
        # rank 1's half of every eager pass runs on TPModel's worker thread: its streams too.
        fast._pool.submit(use, lo, devs[1]).result()

    def prefill_loop(n: int, passes: int, out: list, err: list, logits_out: list) -> None:
        try:
            use(lo, devs[0])
            with torch.inference_mode():
                for _ in range(passes):
                    fast.reset(PRE)
                    fast.restore(snap, PRE)
                    for s in lo:
                        s.synchronize()
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record(lo[0])
                    t0 = time.perf_counter()
                    logits = fast.prefill(PRE, new_ids[:n], args.ctx)
                    ev1.record(lo[0])
                    logits_out.append(logits.float().cpu())
                    lo[1].synchronize()
                    t1 = time.perf_counter()
                    out.append((t0, t1, (ev0, ev1)))
                fast.reset(PRE)
                for s in lo:
                    s.synchronize()
        except BaseException as exc:  # noqa: BLE001
            err.append(f"prefill: {type(exc).__name__}: {exc}")

    def run_threads(targets: list[threading.Thread], label: str) -> None:
        for t in targets:
            t.daemon = True
            t.start()
        deadline = time.perf_counter() + args.hang_s
        for t in targets:
            t.join(max(0.0, deadline - time.perf_counter()))
        if any(t.is_alive() for t in targets):
            print(f"RESULT hang {label}: a thread did not finish in {args.hang_s:.0f}s "
                  f"(timeouts so far unknown) -> KILL", flush=True)
            os._exit(3)

    lo_pool()
    sync_all()

    # ------------------------------------------------------------ (a) solo decode
    solo: dict[tuple[str, int], dict] = {}
    for kind in ("decode", "verify"):
        for b in (1, 2):
            recs: list = []
            err: list = []
            stop = threading.Event()
            warm: list = []
            run_threads([threading.Thread(target=decode_loop, args=(kind, b, stop, 10, warm, err))],
                        "solo-warm")
            run_threads([threading.Thread(target=decode_loop,
                                          args=(kind, b, stop, args.solo_steps, recs, err))], "solo")
            if err:
                raise SystemExit(f"RESULT error: {err}")
            wall = [(r[1] - r[0]) * 1e3 for r in recs]
            host = [r[2] for r in recs]
            dev = [r[3][0].elapsed_time(r[3][1]) for r in recs]
            solo[(kind, b)] = {"wall": statistics.fmean(wall), "p50": _pct(wall, 0.5),
                               "host": statistics.fmean(host)}
            print(f"RESULT a_solo {kind} B={b} wall_ms {_summ(wall)} | host_launch_ms {_summ(host)} "
                  f"| dev0_span_ms {_summ(dev)}", flush=True)

    # ------------------------------------------------------------ (b) solo prefill
    solo_pass: dict[int, float] = {}
    ref_logits: dict[int, torch.Tensor] = {}
    for n in sizes:
        warm: list = []
        err: list = []
        lg: list = []
        run_threads([threading.Thread(target=prefill_loop, args=(n, 2, warm, err, lg))], "prefill-warm")
        recs: list = []
        lg = []
        run_threads([threading.Thread(target=prefill_loop, args=(n, args.solo_passes, recs, err, lg))],
                    "prefill-solo")
        if err:
            raise SystemExit(f"RESULT error: {err}")
        wall = [(r[1] - r[0]) * 1e3 for r in recs]
        dev = [r[2][0].elapsed_time(r[2][1]) for r in recs]
        solo_pass[n] = statistics.fmean(wall)
        ref_logits[n] = lg[0]
        same = all(torch.equal(x, lg[0]) for x in lg)
        print(f"RESULT b_solo prefill new={n} ctx={args.ctx} wall_ms {_summ(wall)} | dev0_span_ms "
              f"{_summ(dev)} | repeatable_logits={same}", flush=True)

    # ------------------------------------------------------------ (c) overlap
    timeouts0 = fast.timeouts()
    print(f"RESULT allreduce_timeouts_before_overlap={timeouts0}", flush=True)
    results: dict[tuple, dict] = {}
    configs = [("verify", 1, sw) for sw in [float(x) for x in args.switch.split(",")]]
    configs += [("decode", 1, configs[0][2]), ("verify", 2, configs[0][2])]
    any_err = []
    for kind, b, sw in configs:
        tp_mod.SWITCH_S = sw
        for n in sizes:
            drecs: list = []
            precs: list = []
            lg: list = []
            err: list = []
            stop = threading.Event()
            dt = threading.Thread(target=decode_loop, args=(kind, b, stop, 10**9, drecs, err))
            pt = threading.Thread(target=prefill_loop, args=(n, args.passes, precs, err, lg))
            dt.daemon = pt.daemon = True
            dt.start()
            time.sleep(0.05)  # decode in steady state before the first pass
            run_threads([pt], f"overlap {kind} B={b} new={n} switch={sw}")
            stop.set()
            run_threads([dt], f"overlap-decode {kind} B={b} new={n}")
            if err:
                any_err.extend(err)
                print(f"RESULT c_overlap {kind} B={b} new={n} switch_ms={sw * 1e3:.1f} ERROR {err}", flush=True)
                continue
            windows = [(p[0], p[1]) for p in precs]

            def inside(r) -> bool:
                mid = 0.5 * (r[0] + r[1])
                return any(a <= mid <= z for a, z in windows)

            during = [r for r in drecs if inside(r)]
            outside = [r for r in drecs if not inside(r)]
            dwall = [(r[1] - r[0]) * 1e3 for r in during]
            dhost = [r[2] for r in during]
            ddev = [r[3][0].elapsed_time(r[3][1]) for r in during]
            owall = [(r[1] - r[0]) * 1e3 for r in outside]
            pwall = [(p[1] - p[0]) * 1e3 for p in precs]
            pdev = [p[2][0].elapsed_time(p[2][1]) for p in precs]
            busy = sum(z - a for a, z in windows)
            steps_in = len(during)
            sref = solo[(kind, b)]
            ldiff = max(float((x - ref_logits[n]).abs().max()) for x in lg)
            res = {
                "dec_ratio": statistics.fmean(dwall) / sref["wall"] if dwall else float("inf"),
                "pass_ratio": statistics.fmean(pwall) / solo_pass[n],
                "ov_step": statistics.fmean(dwall) if dwall else float("inf"),
                "ov_pass": statistics.fmean(pwall),
                "host_ratio": statistics.fmean(dhost) / sref["host"] if dhost else float("inf"),
            }
            results[(kind, b, sw, n)] = res
            print(f"RESULT c_overlap {kind} B={b} new={n} switch_ms={sw * 1e3:.1f} passes={len(precs)} "
                  f"steps_total={len(drecs)} steps_during={steps_in} "
                  f"decode_rate_during={steps_in / busy if busy else 0:.1f}/s "
                  f"(solo {1e3 / sref['wall']:.1f}/s)", flush=True)
            print(f"RESULT c_overlap {kind} B={b} new={n} decode_during wall_ms {_summ(dwall)} | "
                  f"host_launch_ms {_summ(dhost)} | dev0_span_ms {_summ(ddev)}", flush=True)
            print(f"RESULT c_overlap {kind} B={b} new={n} decode_outside wall_ms {_summ(owall)}", flush=True)
            print(f"RESULT c_overlap {kind} B={b} new={n} prefill wall_ms {_summ(pwall)} | dev0_span_ms "
                  f"{_summ(pdev)}", flush=True)
            print(f"RESULT c_ratio {kind} B={b} new={n} switch_ms={sw * 1e3:.1f} "
                  f"decode_during/solo={res['dec_ratio']:.3f} prefill/solo={res['pass_ratio']:.3f} "
                  f"host_launch_during/solo={res['host_ratio']:.2f} "
                  f"prefill_logits_maxdiff_vs_solo={ldiff:.3g}", flush=True)
    sync_all()
    timeouts1 = fast.timeouts()
    graphs_after = {k: sorted(seg.graphs) for k, seg in (("d", fast.ranks[0].segments[0]),
                                                          ("v", fast.ranks[0].vsegments[0]))}
    print(f"RESULT allreduce_timeouts_after_overlap={timeouts1} new={timeouts1 - timeouts0} "
          f"graphs_unchanged={graphs_before == graphs_after}", flush=True)

    # ------------------------------------------------------------ (d) projection and kill rules
    sw0 = configs[0][2]
    prim = [results.get(("verify", 1, sw0, n)) for n in sizes]
    kills = []
    if any_err or timeouts1 - timeouts0:
        kills.append(f"errors/timeouts ({any_err}, {timeouts1 - timeouts0})")
    for n, r in zip(sizes, prim, strict=True):
        if r is None:
            kills.append(f"no overlap result for new={n}")
            continue
        if r["dec_ratio"] > 2.0:
            kills.append(f"decode during pass {r['dec_ratio']:.2f}x solo > 2.0 (new={n})")
        if r["pass_ratio"] > 1.6:
            kills.append(f"prefill {r['pass_ratio']:.2f}x solo > 1.6 (new={n})")
    gains = {}
    for share in (0.10, 0.12):
        per = []
        for n, r in zip(sizes, prim, strict=True):
            if r is None:
                continue
            gain = project(solo[("verify", 1)]["wall"], r["ov_step"], solo_pass[n], r["ov_pass"], share)
            per.append(gain)
            print(f"RESULT d_projection share={share:.2f} new={n} k=pass_ratio={r['pass_ratio']:.3f} "
                  f"r=solo/overlap_step={1 / r['dec_ratio']:.3f} gain={gain * 100:+.1f}%", flush=True)
        gains[share] = statistics.fmean(per) if per else float("nan")
        print(f"RESULT d_projection share={share:.2f} mean_over_sizes gain={gains[share] * 100:+.1f}%",
              flush=True)
    mid = 0.5 * (gains[0.10] + gains[0.12])
    if not mid >= 0.04:
        kills.append(f"projected gain {mid * 100:.1f}% < 4%")
    verdict = "KILL: " + "; ".join(kills) if kills else "GO"
    print(f"RESULT verdict projected_gain_mid={mid * 100:+.1f}% -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
