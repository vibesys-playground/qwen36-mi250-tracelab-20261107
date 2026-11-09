"""Condense ``serve_probe`` output for plan 69: per setting the headline numbers, cached
prompt tokens, slot-parking counters, and each replica's request-path stamps (p50 ms).

    python tools/serve_probe.py ... | python tools/plan69_digest.py
"""

from __future__ import annotations

import json
import sys

KEYS = ("setting", "load_s", "wall_s", "requests", "output_tokens", "output_tok_per_s",
        "tpot_ms_p50", "tpot_ms_p90", "ttft_ms_p50", "ttft_ms_p90", "ttft_first_round_ms_p50",
        "ttft_later_rounds_ms_p50", "ttft_later_rounds_ms_p90", "cached_prompt_tokens")
COUNTERS = ("cached_tokens", "prefilled_tokens", "prefix_cache_tail_hits", "parked_tails",
            "parked_hits", "unparks", "prefix_cache_evictions", "tokenize_hits",
            "tokenize_misses")
SPANS = ("queue", "run", "add", "lookup", "drain", "wait_pass", "pass", "emit", "total",
         "tokenize", "adopt", "tail_detach", "tail_park", "unpark")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            if line.startswith(("Traceback", "RuntimeError", "AssertionError", "Error")):
                print(line, flush=True)
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("event")
        if kind == "agreement":
            print("AGREEMENT", json.dumps(event), flush=True)
        if kind != "setting":
            continue
        print("SETTING", json.dumps({k: event.get(k) for k in KEYS}), flush=True)
        health = event.get("health") or {}
        print("  counters", json.dumps({k: health.get(k) for k in COUNTERS}), flush=True)
        for index, rep in enumerate(health.get("replicas") or []):
            stamps = rep.get("req_stamps") or {}
            cells = {s: (stamps[s]["p50"], stamps[s]["n"]) for s in SPANS if s in stamps}
            extra = {k: rep.get(k) for k in ("parked_tails", "parked_hits", "unparks",
                                             "prefix_cache_tail_hits", "cached_tokens")}
            print(f"  replica{index} p50ms(n)", json.dumps(cells), json.dumps(extra), flush=True)
            if index == 0 and stamps:
                means = {s: stamps[s]["mean"] for s in SPANS if s in stamps}
                print("  replica0 mean ms", json.dumps(means), flush=True)


if __name__ == "__main__":
    main()
