#!/bin/sh
# Plan 63 gate: live moves of a resting session's state (KV pages + GDN conv/recurrent state)
# to the least-loaded replica during its tool wait, served on one MI250 node (4 x TP2).
#
#   sh tools/plan63_gate.sh
#
# 1. Unit tests (CPU): the move policy, the export/import round trip, and two CPU replicas
#    whose sessions move every round (token-identical to a front that never moves).
# 2. spec == plain on GCD 0,1 (tools/spec_step_bench.py), while Triton compiles after /health
#    are counted on GCD 2-7 (three replicas, moves on) under tools/live_compile_probe.py.
# 3. Served A/B, ENGINE_MIGRATE=0/1 x 4 (all eight GCDs, servers started off/on/off/on/...):
#    on each server a replay-shaped pass (48 sessions, 8 at a time, 6.3 rounds, 1800 first +
#    60 grow, 190 out, exponential tool waits of mean 0.3 s) and the generic uneven pass of
#    plan 62 (mean tool wait 0.5 s). Each pass prints tok/s, TTFT/TPOT, the router's in-flight
#    histogram, its move counts and times, moved rounds and their TTFT; then every run's tokens
#    are compared with the first (off) run's, overall and for moved rounds.
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
export PYTHONUNBUFFERED=1
F="_pytest|pluggy|Extension modules|^\s*$|amdgpu.ids"

echo "=== unit tests (CPU)"
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= timeout 900 python -m pytest -q -rf -p no:cacheprovider \
  tests/test_migrate.py tests/test_router.py 2>&1 | grep -vE "$F" | cut -c1-400 | tail -25

( timeout 2400 python -u tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-600 > /tmp/p63_spec.log ) &
( HIP_VISIBLE_DEVICES=2,3,4,5,6,7 timeout 2400 python tools/live_compile_probe.py 2>&1 \
    | grep -E "RESULT|Traceback|Error|live moves" | cut -c1-1500 > /tmp/p63_live.log ) &
wait
echo "=== spec == plain (defaults)"; cat /tmp/p63_spec.log
echo "=== live compiles after /health (3 replicas, moves on)"; cat /tmp/p63_live.log

echo "=== served A/B: ENGINE_MIGRATE off/on x 4"
M=ENGINE_MIGRATE
timeout 5400 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 \
  --first 1800 --grow 60 --max-new 190 --corpus synthetic --shapes replay,uneven \
  --replay-wait 0.3 --tool-wait 0.5 --seed 0 \
  --settings $M=0,$M=1,$M=0,$M=1,$M=0,$M=1,$M=0,$M=1 2>&1 \
  | grep -v "^engine:" | python -u -c '
import json, sys
for line in sys.stdin:
    if not line.startswith("{"):
        print(line[:400], end="", flush=True)
        continue
    d = json.loads(line)
    if d.get("event") == "setting":
        keys = ("setting", "shape", "wall_s", "requests", "output_tokens", "output_tok_per_s",
                "ttft_ms_p50", "ttft_ms_p90", "tpot_ms_p50", "tpot_ms_p90", "cached_prompt_tokens",
                "moved_rounds", "ttft_moved_ms_p50", "ttft_moved_ms_p90",
                "ttft_unmoved_later_ms_p50", "ttft_unmoved_later_ms_p90", "tpot_moved_ms_p50")
        print("SUMMARY", json.dumps({k: d.get(k) for k in keys} | {"router": d.get("router_pass")}),
              flush=True)
        if d.get("replica_migrate_cumulative"):
            print("REPLICAS", json.dumps(d["replica_migrate_cumulative"]), flush=True)
    elif d.get("event") in ("agreement", "shape"):
        print(json.dumps(d), flush=True)
'
true
