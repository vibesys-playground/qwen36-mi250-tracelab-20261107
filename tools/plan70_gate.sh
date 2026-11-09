#!/bin/sh
# Plan 70 stage 0 + gate: two-holder sessions (each round's saved state copied to a buddy
# replica during the tool wait; the next round runs on the less busy holder of the latest
# state), on one MI250 node (4 x TP2).
#
#   sh tools/plan70_gate.sh
#
# 1. Unit tests (CPU): holder versions, a copy in flight at arrival, stale refusal and
#    reroute, two CPU replicas serving alternate rounds token for token.
# 2. spec == plain on GCD 0,1, while Triton compiles after /health are counted on GCD 2-7
#    (three replicas, two holders on).
# 3. Served A/B on the trace shape (the recorded table's 48 sessions, 303 rounds, input and
#    output lengths and tool waits, 8 at a time; synthetic tokens), ENGINE_TWO_HOLDER
#    0/1 x 3 interleaved, then 1 with ties sent to the buddy (most rounds off home, for
#    token identity). Per pass: tok/s, TTFT/TPOT, per replica the share of decode time at
#    each batch size, syncs, buddy rounds, stale refusals. Then every run's tokens against
#    the first (off) run: repeatability (off vs off) and buddy-served rounds' identity.
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
export PYTHONUNBUFFERED=1
F="_pytest|pluggy|Extension modules|^\s*$|amdgpu.ids"

echo "=== unit tests (CPU) $(date +%T)"
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= timeout 900 python -m pytest -q -rf -p no:cacheprovider \
  tests/test_migrate.py tests/test_router.py 2>&1 | grep -vE "$F" | cut -c1-400 | tail -25

echo "=== spec == plain || live compiles $(date +%T)"
( timeout 1500 python -u tools/spec_step_bench.py --devices 0,1 --iters 5 --max-new 128 2>&1 \
    | grep -E "RESULT (parity|accept)|Traceback|Error" | cut -c1-300 > /tmp/p70_spec.log ) &
( HIP_VISIBLE_DEVICES=2,3,4,5,6,7 ENGINE_TWO_HOLDER=1 timeout 1500 python tools/live_compile_probe.py 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-1500 > /tmp/p70_live.log ) &
wait
echo "--- spec == plain"; cat /tmp/p70_spec.log
echo "--- live compiles after /health (3 replicas, two holders on)"; cat /tmp/p70_live.log

echo "=== served A/B on the trace shape $(date +%T)"
T=ENGINE_TWO_HOLDER
timeout 4200 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --corpus synthetic \
  --shapes trace --seed 0 \
  --settings $T=0,$T=1,$T=0,$T=1,$T=0,$T=1,$T=1+ENGINE_TWO_HOLDER_PREFER_BUDDY=1 2>&1 \
  | grep -v "^engine:" | python -u -c '
import json, sys
rows = []
for line in sys.stdin:
    if not line.startswith("{"):
        print(line[:400], end="", flush=True)
        continue
    d = json.loads(line)
    if d.get("event") == "setting":
        keys = ("setting", "shape", "load_s", "wall_s", "output_tok_per_s", "ttft_ms_p50",
                "ttft_ms_p90", "tpot_ms_p50", "tpot_ms_p90", "cached_prompt_tokens",
                "moved_rounds", "ttft_moved_ms_p50", "ttft_moved_ms_p90",
                "ttft_unmoved_later_ms_p50", "ttft_unmoved_later_ms_p90", "tpot_moved_ms_p50")
        r = d.get("router_pass") or {}
        syncs = (d.get("health") or {}).get("router", {}).get("syncs")
        print("SUMMARY", json.dumps({k: d.get(k) for k in keys} | {
            "share_over_2": r.get("share_over_2"), "inflight_seconds": r.get("inflight_seconds")}),
            flush=True)
        print("BATCH", json.dumps({"setting": d["setting"], "replicas": d.get("decode_by_batch")}),
              flush=True)
        if syncs:
            print("SYNCS", json.dumps(syncs)[:1500], flush=True)
        if d.get("replica_migrate_cumulative"):
            print("REPLICAS", json.dumps(d["replica_migrate_cumulative"])[:2000], flush=True)
        rows.append(d)
    elif d.get("event") == "agreement":
        print(json.dumps(d), flush=True)
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")
for flag in ("ENGINE_TWO_HOLDER=0", "ENGINE_TWO_HOLDER=1"):
    rs = [d for d in rows if d["setting"] == flag]
    if not rs:
        continue
    ge3 = [[x.get("share_ge3") for x in d.get("decode_by_batch") or []] for d in rs]
    b1 = [[(x.get("share") or {}).get("1") for x in d.get("decode_by_batch") or []] for d in rs]
    print("AB", json.dumps({"setting": flag, "runs": len(rs),
        **{k: round(mean([d[k] for d in rs if d.get(k) is not None]), 3) for k in (
            "output_tok_per_s", "ttft_ms_p50", "ttft_ms_p90", "tpot_ms_p50", "tpot_ms_p90")},
        "share_ge3_per_replica": ge3, "share_b1_per_replica": b1}), flush=True)
'
echo "=== done $(date +%T)"
true
