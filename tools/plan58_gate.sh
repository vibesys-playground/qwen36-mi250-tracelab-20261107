#!/bin/bash
# Plan 58 stage 1 gate: slot-aware chain depth (KS 1,3,4,5,7; KMAX 7; ENGINE_SLOTS 14; the
# (slots, k) cost table; the per-depth chooser). One MI250 node, 8 GCDs.
#
# Phase 1, in parallel:
#   GCD 0,1: chain_gate (defaults, RUNAHEAD=0 so steps are timed verify_chain calls): spec ==
#            plain at 1 and 2 slots for k = 1,3,4,5,7 and the chooser (adaptive), on replay-like
#            and natural text; per-depth acceptance; the adaptive run's steps by k.
#   GCD 2,3: spec_step_bench (defaults, run-ahead on): spec == plain; then unit tests on GCD 2.
#   GCD 4,5: live_compile_probe (defaults): own-kernel compiles after /health.
#   GCD 6,7: kv_pool_probe today (KMAX 3, SLOTS 32) and new defaults (KMAX 7, SLOTS 14).
# Phase 2, all 8 GCDs: served A/B off/on/off/on, off = KMAX 3 + SLOTS 32 (plan 31's depths),
#   on = the defaults: tok/s, TTFT, TPOT p50/p90, identical tokens, compiles after /health, KV
#   pages, prefix-cache evictions and cached tokens, steps by (requests in step : k).
L=/tmp/p58g
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT|Traceback|Error|error|passed|failed'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -v "RESULT bitwise " | sed -u "s/^/[$1] /"; }
OFF="ENGINE_SPEC_KMAX=3+ENGINE_SLOTS=32"
ON="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14"
(
  echo "=== [gate] chain_gate defaults $(date +%T)"
  ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=0,1 timeout 3000 python tools/chain_gate.py --devices 0,1 \
    --ks 1,3,4,5,7 --slots 1,2 --adaptive --drafts corpus --iters 50 --pairs 3 --max-new 192 2>&1 \
    | tee -a $L/gate.log | tag gate
  echo "=== [gate] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [spec] spec_step_bench defaults (run-ahead on) $(date +%T)"
  HIP_VISIBLE_DEVICES=2,3 timeout 1800 python tools/spec_step_bench.py --devices 2,3 --iters 10 \
    --max-new 192 2>&1 | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]} $(date +%T)"
  echo "=== [tests] $(date +%T)"
  HIP_VISIBLE_DEVICES=2 timeout 1800 python -m pytest -q -rf tests/test_spec_depth.py tests/test_speculative.py \
    tests/test_spec_chain.py tests/test_runahead.py 2>&1 | tail -25 | sed -u "s/^/[tests] /"
) &
(
  echo "=== [jit] live_compile_probe defaults $(date +%T)"
  HIP_VISIBLE_DEVICES=4,5 timeout 3000 python tools/live_compile_probe.py --max-new 192 2>&1 \
    | grep -vE "Loading weights|amdgpu.ids|^\s*$" | cut -c1-2000 | tail -16 | sed -u "s/^/[jit] /"
  grep -hE "engine: (fast path|gdn_hip)" /tmp/live-compile-server.log 2>/dev/null | head -4 | sed "s/^/[jit-server] /"
) &
(
  for cfg in "3 32" "7 14"; do
    set -- $cfg
    echo "=== [pool] KMAX=$1 SLOTS=$2 $(date +%T)"
    ENGINE_SPEC_KMAX=$1 ENGINE_SLOTS=$2 HIP_VISIBLE_DEVICES=6,7 timeout 900 python tools/kv_pool_probe.py \
      --devices 6,7 2>&1 | tee -a $L/pool.log | grep --line-buffered "RESULT pool" | sed -u "s/^/[pool] /"
  done
) &
wait
echo "=== tracebacks (phase 1)"; for f in $L/*.log; do grep -B2 -A25 Traceback $f 2>/dev/null | head -60; done
echo "=== [ab] served A/B off/on/off/on $(date +%T)"
timeout 5400 python -u tools/topk_epi_ab.py --settings "$OFF,$ON,$OFF,$ON" 2>&1 | tee $L/ab.log \
  | grep -v '^engine:' | grep -v '"event": "setting"' | grep -v '"event": "loop_timers"'
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
python - $L/ab.log <<'PY'
import json, re, sys
for line in open(sys.argv[1]):
    if not line.startswith("{"):
        continue
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("event") != "setting":
        continue
    h = e.get("health") or {}
    pages = [m.group(1) for x in e.get("engine", []) for m in [re.search(r"kv_pages=(\d+)", x)] if m]
    hist = {}
    for r in h.get("replicas", []):
        for key, v in (r.get("spec_k_hist") or {}).items():
            hist[key] = hist.get(key, 0) + v
    by_slots = {}
    for key, v in hist.items():
        n, k = key.split(":")
        by_slots.setdefault(f"{n} req", {})[f"k{k}"] = v
    keep = {k: h.get(k) for k in ("prefix_cache_evictions", "prefix_cache_entries", "prefix_cache_tail_hits",
                                  "cached_tokens", "largest_batch", "mtp_verify_steps", "mtp_proposed",
                                  "mtp_accepted")}
    print("CAPACITY", json.dumps({"setting": e["setting"], "kv_pages": pages, **keep,
                                  "cached_prompt_tokens": e.get("cached_prompt_tokens"),
                                  "tok_s": e.get("output_tok_per_s"), "tpot_p50": e.get("tpot_ms_p50"),
                                  "tpot_p90": e.get("tpot_ms_p90"), "ttft_p50": e.get("ttft_ms_p50"),
                                  "ttft_p90": e.get("ttft_ms_p90")}))
    print("KHIST", json.dumps({"setting": e["setting"],
                               "steps_by_requests_and_k": dict(sorted(by_slots.items(), key=lambda kv: int(kv[0].split()[0])))}))
PY
for f in /tmp/serve-probe-ENGINE_SPEC_KMAX_*.log; do
  echo "--- $f (the last run of that setting)"
  grep -E "kv_pages|Traceback|Error" $f | cut -c1-300 | head -8
done
echo "=== done $(date +%T)"
