#!/bin/bash
# Plan 47 stage 1 gate: IVF-scored chained MTP drafts (ENGINE_DRAFT_IVF) on the run-ahead head.
# One MI250 node, 8 GCDs.
#
#   bash tools/plan47_gate.sh
#
# Phase 1, in parallel:
#   GCD 0,1: tools/spec_step_bench.py IVF=1 RUNAHEAD=1 (spec == plain 8/8, repeat in process)
#   GCD 2,3: tools/chain_gate.py IVF=0 then IVF=1 (chained step at 1 and 2 slots, k=1/3;
#            accepted tokens per step and per-depth acceptance on replay-like and natural text;
#            spec == plain per k)
#   GCD 4,5: tools/live_compile_probe.py (IVF on, the default): compiles after /health
#   GCD 6,7: unit tests on GCD 6 (IVF draft incl. the served path in a graph, run-ahead,
#            chained verify), then tools/spec_step_bench.py IVF=0 RUNAHEAD=1
# Phase 2, all 8 GCDs: served A/B IVF 0/1/0/1 (tools/topk_epi_ab.py with --settings): TPOT,
#   identical tokens per pair (on == off, and across processes), compiles after /health.
L=/tmp/p47
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT|Traceback|Error|error|IVF drafts|identical|after'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
(
  export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
  echo "=== [spec1] spec_step_bench IVF=1 $(date +%T)"
  ENGINE_DRAFT_IVF=1 ENGINE_RUNAHEAD=1 HIP_VISIBLE_DEVICES=0,1 timeout 1800 python tools/spec_step_bench.py \
    --devices 0,1 --iters 10 --max-new 128 2>&1 | tee -a $L/spec1.log | tag spec1
  echo "=== [spec1] exit ${PIPESTATUS[0]}"
) &
(
  export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
  for v in 0 1; do
    echo "=== [gate$v] chain_gate IVF=$v $(date +%T)"
    ENGINE_DRAFT_IVF=$v HIP_VISIBLE_DEVICES=2,3 timeout 1500 python tools/chain_gate.py --devices 2,3 --pairs 3 2>&1 \
      | tee -a $L/gate$v.log | tag gate$v
    echo "=== [gate$v] exit ${PIPESTATUS[0]}"
  done
) &
(
  echo "=== [jit] live_compile_probe $(date +%T)"
  HIP_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 timeout 3000 python tools/live_compile_probe.py > $L/compile.log 2>&1
  echo "=== [jit] exit $?"
  cp /tmp/live-compile-server.log $L/live-server.log 2>/dev/null
) &
(
  echo "=== [tests] pytest $(date +%T)"
  HIP_VISIBLE_DEVICES=6 timeout 1800 python -m pytest -q -rf tests/test_draft_ivf.py tests/test_runahead.py \
    tests/test_spec_chain.py > $L/tests.log 2>&1
  echo "=== [tests] exit $?"
  export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
  echo "=== [spec0] spec_step_bench IVF=0 $(date +%T)"
  ENGINE_DRAFT_IVF=0 ENGINE_RUNAHEAD=1 HIP_VISIBLE_DEVICES=6,7 timeout 1800 python tools/spec_step_bench.py \
    --devices 6,7 --iters 10 --max-new 128 2>&1 | tee -a $L/spec0.log | tag spec0
  echo "=== [spec0] exit ${PIPESTATUS[0]}"
) &
wait
echo "=== spec == plain, IVF=1"; grep -E "RESULT|Error|Traceback|IVF" $L/spec1.log | tail -30
echo "=== spec == plain, IVF=0"; grep -E "RESULT|Error|Traceback" $L/spec0.log | tail -30
echo "=== chain_gate IVF=0"; grep -E "RESULT|Error" $L/gate0.log | grep -v bitwise | tail -25
echo "=== chain_gate IVF=1"; grep -E "RESULT|Error|IVF" $L/gate1.log | grep -v bitwise | tail -25
echo "=== live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/compile.log | cut -c1-3000 | tail -40
grep -E "IVF|Traceback|Error" $L/live-server.log 2>/dev/null | tail -10
echo "=== unit tests"; tail -25 $L/tests.log
echo "=== tracebacks (phase 1)"; for f in $L/*.log; do grep -B2 -A25 Traceback $f 2>/dev/null | head -60; done
echo "=== [ab] served A/B IVF 0/1/0/1 $(date +%T)"
timeout 3600 python -u tools/topk_epi_ab.py \
  --settings ENGINE_DRAFT_IVF=0,ENGINE_DRAFT_IVF=1,ENGINE_DRAFT_IVF=0,ENGINE_DRAFT_IVF=1 2>&1 | grep -v '^engine:'
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
