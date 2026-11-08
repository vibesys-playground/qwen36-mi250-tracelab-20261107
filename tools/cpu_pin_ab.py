"""Plan 36 step 1: served A/B of host CPU placement (engine.cpu_place).

    python tools/cpu_pin_ab.py [--settings S1,S2,...] [serve_probe shape flags]

Runs ``tools/serve_probe.py``'s served run (``serve_and_run``: the task's launch command, 8
concurrent multi-round sessions, temperature 0, exact lengths) once per setting, in order. The
default settings alternate ``ENGINE_CPU_PIN`` 0/1/0/1, then pinning that may use CPU 0's L3
group (``ENGINE_CPU_PIN_AVOID_CPU0=0``), then pinned plain decode (``ENGINE_MTP=0``). Prints
each setting's ``cpu_place`` log lines (the placement the router chose and each rank thread's
core), a table of output tok/s, TTFT and TPOT, and how many requests produced the same tokens
as the first setting (greedy: every setting must match; the ``ENGINE_MTP=0`` one is the
spec == plain check).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import serve_probe  # noqa: E402

DEFAULT = ",".join([
    "ENGINE_CPU_PIN=0",
    "ENGINE_CPU_PIN=1",
    "ENGINE_CPU_PIN=0",
    "ENGINE_CPU_PIN=1",
    "ENGINE_CPU_PIN=1+ENGINE_CPU_PIN_AVOID_CPU0=0",
    "ENGINE_CPU_PIN=1+ENGINE_MTP=0",
])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--rounds", type=float, default=4)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--first", type=int, default=2000)
    p.add_argument("--grow", type=int, default=300)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--port", type=int, default=18431)
    p.add_argument("--corpus", choices=("source", "synthetic"), default="synthetic")
    p.add_argument("--settings", default=DEFAULT)
    args = p.parse_args()
    from engine.generator import CHECKPOINT

    model = args.model or CHECKPOINT["model"]
    revision = None if args.model else CHECKPOINT["revision"]
    need = args.sessions * (args.first + math.ceil(args.rounds) * args.grow) + 1
    pool = serve_probe.token_pool(model, revision, need, args.corpus)
    print(json.dumps({"event": "shape", **vars(args)}), flush=True)
    runs = []
    for k, setting in enumerate(args.settings.split(",")):
        rows: list[dict] = []
        real_print = print

        def capture(*a, **kw):  # serve_and_run prints one JSON line per setting
            text = " ".join(map(str, a))
            if text.startswith("{"):
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError:
                    pass
            real_print(*a, **kw)

        serve_probe.print = capture  # type: ignore[attr-defined]
        try:
            ids = serve_probe.serve_and_run(setting, args.port + k, args, pool)
        except Exception as exc:  # noqa: BLE001 - report and go on with the next setting
            print(json.dumps({"event": "error", "setting": setting, "error": f"{type(exc).__name__}: {exc}"[-3000:]}),
                  flush=True)
            continue
        finally:
            serve_probe.print = real_print  # type: ignore[attr-defined]
        name = "".join(c if c.isalnum() else "_" for c in setting)
        log = Path(f"/tmp/serve-probe-{name}.log")
        place = [x for x in log.read_text().splitlines() if x.startswith("cpu_place:")] if log.exists() else []
        for line in place:
            print(f"[{k}] {line}", flush=True)
        runs.append((k, setting, rows[-1] if rows else {}, ids, len(place)))
    base_ids = runs[0][3] if runs else {}
    print("\n=== served A/B (synthetic sessions), in run order ===", flush=True)
    print(f"{'#':>2} {'setting':<46} {'tok/s':>8} {'ttft50':>8} {'ttft90':>8} {'tpot50':>7} {'tpot90':>7} "
          f"{'same/req':>9} {'place':>5}", flush=True)
    for k, setting, r, ids, nplace in runs:
        same = sum(ids.get(key) == v for key, v in base_ids.items())
        print(f"{k:>2} {setting:<46} {r.get('output_tok_per_s', 0):>8.1f} {r.get('ttft_ms_p50') or 0:>8.1f} "
              f"{r.get('ttft_ms_p90') or 0:>8.1f} {r.get('tpot_ms_p50') or 0:>7.2f} {r.get('tpot_ms_p90') or 0:>7.2f} "
              f"{same:>4}/{len(base_ids):<4} {nplace:>5}", flush=True)
        print("RESULT " + json.dumps({"kind": "ab", "run": k, "setting": setting,
                                      "same_as_first": same, "requests": len(base_ids),
                                      "cpu_place_lines": nplace,
                                      **{key: r.get(key) for key in (
                                          "output_tok_per_s", "ttft_ms_p50", "ttft_ms_p90", "tpot_ms_p50",
                                          "tpot_ms_p90", "ttft_first_round_ms_p50", "ttft_later_rounds_ms_p50",
                                          "load_s", "wall_s")}}), flush=True)


if __name__ == "__main__":
    main()
