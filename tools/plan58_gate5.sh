#!/bin/bash
# Plan 58 after merging head afb6be0 (plan 59: ROCm 7.0, torch 2.10, Triton 3.6). One MI250 node,
# 8 GCDs, in parallel (fits the command time limit):
#   GCD 0,1: chain_gate defaults (IVF on, RUNAHEAD=0): captured verify_chain step ms per (slots,
#            k), median of 200 (to re-seed speculative.COST if they moved), spec == plain at 1 and
#            2 slots for k = 1,3,4,5,7 and adaptive, replay-like and natural.
#   GCD 2,3: spec_step_bench defaults (run-ahead on): spec == plain; then unit tests.
#   GCD 4,5: live_compile_probe defaults: own-kernel compiles after /health (deep-k graphs and
#            the run-ahead kernel warmed before /health on the new stack).
#   GCD 6,7: kv_pool_probe defaults and KMAX 3 / 32 slots.
L=/tmp/p58g5
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT|Traceback|Error|error|passed|failed'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT (bitwise|gen_step) " | cut -c1-1200 | sed -u "s/^/[$1] /"; }
(
  echo "=== [gate] chain_gate defaults $(date +%T)"
  ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=0,1 timeout 1500 python tools/chain_gate.py --devices 0,1 \
    --ks 1,3,4,5,7 --slots 1,2 --adaptive --drafts corpus --iters 200 --pairs 3 --max-new 192 2>&1 \
    | tee -a $L/gate.log | tag gate
  echo "=== [gate] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [spec] spec_step_bench defaults $(date +%T)"
  HIP_VISIBLE_DEVICES=2,3 timeout 900 python tools/spec_step_bench.py --devices 2,3 --iters 10 --max-new 192 2>&1 \
    | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]}"
  HIP_VISIBLE_DEVICES=2 timeout 900 python -m pytest -q -rf tests/test_spec_depth.py tests/test_draft_ivf.py \
    tests/test_runahead.py 2>&1 | tail -6 | sed -u "s/^/[tests] /"
) &
(
  echo "=== [jit] live_compile_probe defaults $(date +%T)"
  HIP_VISIBLE_DEVICES=4,5 timeout 1500 python tools/live_compile_probe.py --max-new 192 2>&1 \
    | grep -E '"kind": "(compiles|load)"|Traceback|Error' | cut -c1-1500 | sed -u "s/^/[jit] /"
) &
(
  for cfg in "7 14" "3 32"; do
    set -- $cfg
    ENGINE_SPEC_KMAX=$1 ENGINE_SLOTS=$2 HIP_VISIBLE_DEVICES=6,7 timeout 900 python tools/kv_pool_probe.py \
      --devices 6,7 2>&1 | grep -E "RESULT pool|Traceback|Error" | cut -c1-400 | sed -u "s/^/[pool] /"
  done
) &
wait
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | head -40; done
echo "=== done $(date +%T)"
