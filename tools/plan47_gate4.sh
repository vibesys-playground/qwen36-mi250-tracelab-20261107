#!/bin/bash
# Plan 47 iteration 3 gate: the whole-vocabulary search on each rank (peer lm_head copy). One MI250
# node, 8 GCDs.
#
# Phase 1, in parallel:
#   GCD 0,1: spec_step_bench IVF=1 RUNAHEAD=1 (spec == plain), then chain_gate IVF=1 with the
#            audit (each draft also run in full: the IVF draft's agreement and fallback rate)
#   GCD 2,3: chain_gate IVF=0 RUNAHEAD=0: step times, accepted tokens per step, and the wall
#            time of each real verify step (serial steps: chain_gate times verify_chain)
#   GCD 4,5: chain_gate IVF=1 RUNAHEAD=0: the same (the k=3 step at the fallback rate of real
#            drafts); the audit run on 0,1 keeps run-ahead on
#   GCD 6,7: unit tests on GCD 6, then live_compile_probe (compiles after /health)
# Phase 2, all 8 GCDs: served A/B IVF 0/1/0/1 (TPOT p50/p90, identical tokens, compiles).
L=/tmp/p47d
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT (step|gen_step|accept|parity|fatal|error|load)|Traceback|IVF drafts|identical'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
gate() {  # name devices env...
  local name=$1 dev=$2; shift 2
  echo "=== [$name] chain_gate $* $(date +%T)"
  env "$@" HIP_VISIBLE_DEVICES=$dev timeout 2400 python tools/chain_gate.py --devices $dev --pairs 3 --iters 50 2>&1 \
    | tee -a $L/$name.log | tag $name
  echo "=== [$name] exit ${PIPESTATUS[0]} $(date +%T)"
}
(
  echo "=== [spec1] spec_step_bench IVF=1 $(date +%T)"
  ENGINE_DRAFT_IVF=1 ENGINE_RUNAHEAD=1 HIP_VISIBLE_DEVICES=0,1 timeout 1800 python tools/spec_step_bench.py \
    --devices 0,1 --iters 10 --max-new 128 2>&1 | tee -a $L/spec1.log | grep --line-buffered -E "RESULT|Traceback|IVF" \
    | sed -u "s/^/[spec1] /"
  echo "=== [spec1] exit ${PIPESTATUS[0]}"
  gate audit 0,1 ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_AUDIT=1
) &
gate off 2,3 ENGINE_DRAFT_IVF=0 ENGINE_RUNAHEAD=0 &
gate on 4,5 ENGINE_DRAFT_IVF=1 ENGINE_RUNAHEAD=0 &
(
  echo "=== [tests] pytest $(date +%T)"
  HIP_VISIBLE_DEVICES=6 timeout 1800 python -m pytest -q -rf tests/test_draft_ivf.py tests/test_runahead.py \
    tests/test_spec_chain.py tests/test_fused_gemv.py tests/test_chain_glue.py > $L/tests.log 2>&1
  echo "=== [tests] exit $?"
  echo "=== [jit] live_compile_probe $(date +%T)"
  HIP_VISIBLE_DEVICES=6,7 timeout 3000 python tools/live_compile_probe.py > $L/compile.log 2>&1
  echo "=== [jit] exit $?"
  cp /tmp/live-compile-server.log $L/live-server.log 2>/dev/null
) &
wait
echo "=== unit tests"; tail -12 $L/tests.log
echo "=== live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/compile.log | cut -c1-3000 | tail -30
grep -E "IVF|Traceback|Error" $L/live-server.log 2>/dev/null | tail -6
echo "=== tracebacks (phase 1)"; for f in $L/*.log; do grep -B2 -A25 Traceback $f 2>/dev/null | head -60; done
echo "=== [ab] served A/B IVF 0/1/0/1 $(date +%T)"
timeout 3600 python -u tools/topk_epi_ab.py \
  --settings ENGINE_DRAFT_IVF=0,ENGINE_DRAFT_IVF=1,ENGINE_DRAFT_IVF=0,ENGINE_DRAFT_IVF=1 2>&1 | tee $L/ab.log \
  | grep -v '^engine:' | grep -v '"event": "setting"'
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
echo "=== [ab] capacity per setting: KV pages, prefix-cache evictions, health"
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
    keep = {k: h.get(k) for k in ("prefix_cache_evictions", "prefix_cache_entries", "prefix_cache_bytes",
                                  "largest_batch", "mtp_verify_steps", "mtp_proposed", "mtp_accepted")}
    print("CAPACITY", json.dumps({"setting": e["setting"], "kv_pages": pages, **keep,
                                  "tpot_p50": e.get("tpot_ms_p50"), "tpot_p90": e.get("tpot_ms_p90")}))
PY
grep -h "kv_pages\|IVF drafts\|reserved" $L/*.log | sort | uniq -c | head -20
for f in /tmp/serve-probe-ENGINE_DRAFT_IVF_*.log; do
  echo "--- $f (the last run of that setting)"
  grep -E "kv_pages|IVF drafts|reserved|allocated|pressure|evict|Traceback|Error" $f | cut -c1-300 | head -12
done
