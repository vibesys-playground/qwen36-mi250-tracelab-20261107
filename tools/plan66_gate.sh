#!/bin/bash
# Plan 66 exactness gate on one MI250 node (8 GCDs). New defaults: prefill graph sizes every 64
# rows to 4,096 and one segment per prefill pass. "HEAD" = c7c6727's bucket list and pack budget.
# Phase 1, in parallel:
#   GCD 0,1: spec_step_bench defaults (spec == plain), then the unit tests
#   GCD 2,3: tp_parity: teacher-forced top-1 of the new defaults against a HEAD-config reference
#            (32 synthetic prompts of 1,600..2,600 tokens, each one prefill pass, 96 greedy
#            steps), and in-process repeat
#   GCD 4,5: live_compile_probe defaults (own-kernel compiles after /health)
#   GCD 6,7: plan66_probe new defaults: capture time, memory, KV pages, graph vs eager and timing
#            at 1,650..4,000 tokens, packs
# Phase 2: serve_probe replay-like 4x2 (48 sessions, c8, first rounds 850..2,600), runs:
#   new; new; HEAD  -> repeatability (run 1 vs 0) and agreement / TTFT vs HEAD (run 2 vs 0)
L=/tmp/p66g
mkdir -p $L
export ENGINE_TP=2 PYTHONUNBUFFERED=1
HEAD="64,80,96,112,128,160,192,224,256,288,320,384,448,512,576,640,704,768,896,1024,1152,1280,1408,1536,1792,2048"
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1
F='^=== |RESULT|Traceback|Error|error|passed|failed|prefill graphs'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT (bitwise|gen_step) " | cut -c1-1500 | sed -u "s/^/[$1] /"; }
echo "=== phase 1 $(date +%T)"
(
  echo "=== [spec] spec_step_bench $(date +%T)"
  HIP_VISIBLE_DEVICES=0,1 timeout 1200 python tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 192 2>&1 \
    | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]} $(date +%T)"
  echo "=== [tests] $(date +%T)"
  HIP_VISIBLE_DEVICES=0,1 timeout 1800 python -m pytest -q -rf tests 2>&1 | tail -15 | sed -u "s/^/[tests] /"
  echo "=== [tests] done $(date +%T)"
) &
(
  E="HIP_VISIBLE_DEVICES=2,3"
  echo "=== [tf] ref, HEAD config $(date +%T)"
  env $E ENGINE_PREFILL_GRAPH_BUCKETS=$HEAD ENGINE_PREFILL_PACK_TOKENS=8192 timeout 1500 python -u tools/tp_parity.py \
    --mode ref --out $L/ref_head.pt --lengths 1600:2600:32 --one-pass 2>&1 | tee -a $L/tf.log | tag tf
  echo "=== [tf] check, new defaults $(date +%T)"
  env $E timeout 1500 python -u tools/tp_parity.py --mode check --ref $L/ref_head.pt --one-pass 2>&1 \
    | tee -a $L/tf.log | tag tf
  echo "=== [tf] repeat, new defaults $(date +%T)"
  env $E timeout 1500 python -u tools/tp_parity.py --mode repeat --lengths 1600:2600:32 --one-pass 2>&1 \
    | tee -a $L/tf.log | tag tf
  echo "=== [tf] done $(date +%T)"
) &
(
  echo "=== [jit] live_compile_probe $(date +%T)"
  HIP_VISIBLE_DEVICES=4,5 timeout 1800 python tools/live_compile_probe.py --max-new 192 2>&1 \
    | grep -E '"kind": "(compiles|load)"|Traceback|Error' | cut -c1-1500 | sed -u "s/^/[jit] /"
  echo "=== [jit] done $(date +%T)"
) &
(
  echo "=== [probe] plan66_probe new defaults $(date +%T)"
  HIP_VISIBLE_DEVICES=6,7 timeout 1500 python -u tools/plan66_probe.py --news 1650,1700,1900,2050,2300,2600,3000,4000 \
    --start 0 --reps 5 --packs 482+300 2>&1 | tee -a $L/probe.log | tag probe
  echo "=== [probe] done $(date +%T)"
) &
wait
echo "=== phase 2 served $(date +%T)"
timeout 3000 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 850 \
  --first-max 2600 --grow 290 --max-new 190 --corpus synthetic \
  --settings "ENGINE_LOOP_TIMERS=1;ENGINE_LOOP_TIMERS=1;ENGINE_LOOP_TIMERS=1+ENGINE_PREFILL_GRAPH_BUCKETS=$HEAD+ENGINE_PREFILL_PACK_TOKENS=8192" \
  2>&1 | grep -v "^engine:" | grep -vE "Loading weights|amdgpu.ids|^\s*$|it/s\]" | cut -c1-2500
echo "serve_probe exit ${PIPESTATUS[0]}"
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | head -40; done
echo "=== done $(date +%T)"
true
