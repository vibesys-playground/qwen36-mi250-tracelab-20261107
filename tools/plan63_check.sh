#!/bin/sh
# Plan 63 check 2 (on head c7c6727 + the pinned page-id fix): spec == plain on GCD 0,1, then
# three servers on all eight GCDs: moves off, moves forced (ENGINE_MIGRATE_GAP=0: every round
# that ends where another replica is no busier moves), and moves at the default gap 2.
# Replay-shaped (0.3 s mean tool waits) and uneven passes on each. Reports moved rounds, their
# TTFT, the engine-thread time of each export/import and the decode-loop period with a move in
# flight (source and target stall), and token identity of every run (moved rounds alone too)
# against the moves-off run. j302 ran past the job's limit before its identity lines.
export PYTHONUNBUFFERED=1
( timeout 1200 python -u tools/spec_step_bench.py --devices 0,1 --iters 5 --max-new 128 2>&1 \
    | grep -E "RESULT (parity|accept)|Traceback|Error" | cut -c1-300 )
M=ENGINE_MIGRATE
timeout 2400 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 \
  --first 1800 --grow 60 --max-new 190 --corpus synthetic --shapes replay,uneven \
  --replay-wait 0.3 --tool-wait 0.5 --seed 0 --pass-gap 6 \
  --settings $M=0,$M=1+ENGINE_MIGRATE_GAP=0,$M=1 2>&1 \
  | grep -v "^engine:" | python -u -c '
import json, sys
for line in sys.stdin:
    if not line.startswith("{"):
        print(line[:400], end="", flush=True)
        continue
    d = json.loads(line)
    if d.get("event") == "setting":
        keys = ("setting", "shape", "wall_s", "output_tok_per_s", "ttft_ms_p50", "ttft_ms_p90",
                "tpot_ms_p50", "tpot_ms_p90", "cached_prompt_tokens", "moved_rounds",
                "ttft_moved_ms_p50", "ttft_moved_ms_p90", "ttft_unmoved_later_ms_p50",
                "ttft_unmoved_later_ms_p90", "tpot_moved_ms_p50")
        r = d.get("router_pass") or {}
        print("SUMMARY", json.dumps({k: d.get(k) for k in keys} | {"moves": r.get("moves"),
              "share_over_2": r.get("share_over_2")}), flush=True)
        if d.get("replica_migrate_cumulative"):
            print("REPLICAS", json.dumps(d["replica_migrate_cumulative"]), flush=True)
    elif d.get("event") == "agreement":
        print(json.dumps(d), flush=True)
'
true
