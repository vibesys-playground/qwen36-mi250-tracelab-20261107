#!/bin/bash
# Gate for plan 37 (router top-k in the K-sum kernel, ENGINE_TOPK_EPI) stacked on plan 42's
# run-ahead chained verify (ENGINE_RUNAHEAD). One MI250 node, 8 GCDs.
#
#   bash tools/combine37_gate.sh
#
# Phase 1, in parallel:
#   GCD 0,1: tools/topk_epi_check.py (routing bitwise vs _topk_kernel, 1000 graph replays x 3
#            layers x R 1-32), then tools/chain_gate.py EPI=1
#   GCD 2,3: tools/spec_step_bench.py EPI=1 RUNAHEAD=1 (spec == plain 8/8)
#   GCD 4,5: tools/live_compile_probe.py (defaults: EPI=1, RUNAHEAD=1), compiles after /health
#   GCD 6,7: unit tests on GCD 6 (run-ahead, chained verify, GEMV, router, fused MoE), then
#            tools/spec_step_bench.py EPI=0 RUNAHEAD=1 (spec == plain with the old top-k)
# Phase 2, all 8 GCDs: tools/topk_epi_ab.py served A/B EPI 0/1/0/1 (run-ahead on, the default):
#   identical-token counts per pair (0 vs 0 is the run-to-run floor), TPOT, compiles after /health.
L=/tmp/c37
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT|Traceback|Error|error|spec==plain|identical|after'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
(
  export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
  echo "=== [check] topk_epi_check $(date +%T)"
  HIP_VISIBLE_DEVICES=0,1 timeout 900 python tools/topk_epi_check.py --devices 0,1 --replays 1000 2>&1 \
    | tee -a $L/check.log | tag check
  echo "=== [check] exit ${PIPESTATUS[0]}"
  echo "=== [gate1] chain_gate EPI=1 $(date +%T)"
  ENGINE_TOPK_EPI=1 HIP_VISIBLE_DEVICES=0,1 timeout 900 python tools/chain_gate.py --devices 0,1 2>&1 \
    | tee -a $L/gate1.log | tag gate1
  echo "=== [gate1] exit ${PIPESTATUS[0]}"
) &
(
  export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
  echo "=== [spec1] spec_step_bench EPI=1 RUNAHEAD=1 $(date +%T)"
  ENGINE_TOPK_EPI=1 ENGINE_RUNAHEAD=1 HIP_VISIBLE_DEVICES=2,3 timeout 1800 python tools/spec_step_bench.py \
    --devices 2,3 --iters 10 --max-new 128 2>&1 | tee -a $L/spec1.log | tag spec1
  echo "=== [spec1] exit ${PIPESTATUS[0]}"
) &
(
  echo "=== [jit] live_compile_probe $(date +%T)"
  HIP_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 timeout 3000 python tools/live_compile_probe.py > $L/compile.log 2>&1
  echo "=== [jit] exit $?"
  cp /tmp/live-compile-server.log $L/live-server.log 2>/dev/null
) &
(
  echo "=== [tests] pytest $(date +%T)"
  HIP_VISIBLE_DEVICES=6 timeout 1800 python -m pytest -q -rf tests/test_runahead.py tests/test_spec_chain.py \
    tests/test_fused_gemv.py tests/test_router.py tests/test_fused_moe.py > $L/tests.log 2>&1
  echo "=== [tests] exit $?"
  export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
  echo "=== [spec0] spec_step_bench EPI=0 RUNAHEAD=1 $(date +%T)"
  ENGINE_TOPK_EPI=0 ENGINE_RUNAHEAD=1 HIP_VISIBLE_DEVICES=6,7 timeout 1800 python tools/spec_step_bench.py \
    --devices 6,7 --iters 10 --max-new 128 2>&1 | tee -a $L/spec0.log | tag spec0
  echo "=== [spec0] exit ${PIPESTATUS[0]}"
) &
wait
echo "=== spec == plain, EPI=1 RUNAHEAD=1 (TP2)"; grep -E "RESULT|Error|Traceback" $L/spec1.log | tail -30
echo "=== spec == plain, EPI=0 RUNAHEAD=1 (TP2)"; grep -E "RESULT|Error|Traceback" $L/spec0.log | tail -30
echo "=== topk_epi_check"; grep -E "RESULT|bitwise|mismatch|Error" $L/check.log | tail -30
echo "=== chain_gate EPI=1"; grep -E "RESULT|Error" $L/gate1.log | tail -20
echo "=== live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/compile.log | cut -c1-3000 | tail -40
grep -E "runahead|Traceback|Error" $L/live-server.log 2>/dev/null | tail -10
echo "=== unit tests"; tail -25 $L/tests.log
echo "=== tracebacks (phase 1)"; for f in $L/*.log; do grep -B2 -A25 Traceback $f 2>/dev/null | head -60; done
echo "=== [ab] served A/B EPI 0/1/0/1, run-ahead on $(date +%T)"
timeout 3000 python -u tools/topk_epi_ab.py 2>&1 | grep -v '^engine:'
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
