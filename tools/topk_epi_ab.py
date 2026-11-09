"""Plan 37 step 1: served A/B of the router top-k in the K-sum kernel (``ENGINE_TOPK_EPI``),
with a count of Triton compiles after /health.

    python tools/topk_epi_ab.py [--settings ENGINE_TOPK_EPI=0,ENGINE_TOPK_EPI=1,...] [shape flags]

Runs ``tools/serve_probe.py``'s served run (the task's launch command, multi-round sessions,
temperature 0, ignore_eos, exact lengths) once per setting, in order. Each server gets its own
empty ``TRITON_CACHE_DIR`` under /tmp, so every kernel it compiles, in any process, leaves one
new entry there; the entries are counted when /health first answers (just before the first
session starts) and again after the last request. Any compile after /health is a live JIT
compile on the request path and fails the gate.

Prints a table (tok/s, TTFT, TPOT, compiles at /health and after), and for every pair of
settings how many requests produced identical tokens (greedy: on == off and cross-process
repeatability both require all of them). ``RESULT`` lines carry the numbers. A hint for the
lead, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import serve_probe  # noqa: E402


def entries(path: Path) -> int:
    return sum(1 for _ in path.iterdir()) if path.exists() else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--sessions", type=int, default=48)
    p.add_argument("--rounds", type=float, default=6.3)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--first", type=int, default=1800)
    p.add_argument("--grow", type=int, default=480)
    p.add_argument("--max-new", type=int, default=190)
    p.add_argument("--port", type=int, default=18531)
    p.add_argument("--corpus", choices=("source", "synthetic"), default="synthetic")
    p.add_argument("--settings", default="ENGINE_TOPK_EPI=0,ENGINE_TOPK_EPI=1,ENGINE_TOPK_EPI=0,ENGINE_TOPK_EPI=1")
    args = p.parse_args()
    from engine.generator import CHECKPOINT

    model = args.model or CHECKPOINT["model"]
    revision = None if args.model else CHECKPOINT["revision"]
    need = args.sessions * (args.first + math.ceil(args.rounds) * args.grow) + 1
    pool = serve_probe.token_pool(model, revision, need, args.corpus)
    print(json.dumps({"event": "shape", **vars(args)}), flush=True)
    real_session = serve_probe.session
    runs = []
    for k, setting in enumerate(args.settings.split(",")):
        cache = Path(f"/tmp/p37-triton-{k}")
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache)  # serve_and_run copies os.environ
        at_health: list[int] = []
        lock = threading.Lock()

        def session(url, pool_, index, a, _cache=cache, _at=at_health, _lock=lock):
            with _lock:  # the first session starts right after /health answered
                if not _at:
                    _at.append(entries(_cache))
            return real_session(url, pool_, index, a)

        serve_probe.session = session
        rows: list[dict] = []
        real_print = print

        def capture(*a, **kw):
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
            print(json.dumps({"event": "error", "setting": setting,
                              "error": f"{type(exc).__name__}: {exc}"[-3000:]}), flush=True)
            continue
        finally:
            serve_probe.print = real_print  # type: ignore[attr-defined]
            serve_probe.session = real_session
        after = entries(cache)
        health_n = at_health[0] if at_health else -1
        runs.append((k, setting, rows[-1] if rows else {}, ids, health_n, after))
    print("\n=== served A/B (synthetic sessions), in run order ===", flush=True)
    print(f"{'#':>2} {'setting':<24} {'tok/s':>8} {'ttft50':>8} {'ttft90':>8} {'tpot50':>7} {'tpot90':>7} "
          f"{'jit@health':>10} {'jit_after':>9}", flush=True)
    for k, setting, r, ids, health_n, after in runs:
        print(f"{k:>2} {setting:<24} {r.get('output_tok_per_s', 0):>8.1f} {r.get('ttft_ms_p50') or 0:>8.1f} "
              f"{r.get('ttft_ms_p90') or 0:>8.1f} {r.get('tpot_ms_p50') or 0:>7.2f} {r.get('tpot_ms_p90') or 0:>7.2f} "
              f"{health_n:>10} {after - health_n:>9}", flush=True)
        print("RESULT " + json.dumps({"kind": "ab", "run": k, "setting": setting,
                                      "triton_cache_entries_at_health": health_n,
                                      "triton_compiles_after_health": after - health_n,
                                      **{key: r.get(key) for key in (
                                          "output_tok_per_s", "ttft_ms_p50", "ttft_ms_p90", "tpot_ms_p50",
                                          "tpot_ms_p90", "load_s", "wall_s", "requests")}}), flush=True)
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            same = sum(b[3].get(key) == v for key, v in a[3].items())
            print("RESULT " + json.dumps({"kind": "identical", "runs": [a[0], b[0]],
                                          "settings": [a[1], b[1]], "same": same,
                                          "requests": len(a[3])}), flush=True)
    live = sum(after - health_n for _, _, _, _, health_n, after in runs)
    ok = bool(runs) and live == 0 and all(h > 0 for _, _, _, _, h, _ in runs)
    print("RESULT " + json.dumps({"kind": "jit_gate", "compiles_after_health": live,
                                  "verdict": "PASS" if ok else "FAIL"}), flush=True)


if __name__ == "__main__":
    main()
