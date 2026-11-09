#!/bin/bash
# Plan 37 step 1 gates on the merged head: the router top-k in the router|shared K-sum kernel
# (ENGINE_TOPK_EPI=1, the default) against the separate _topk_kernel (=0).
#
#   bash tools/topk_epi_step1.sh        # one MI250 node, 8 GCDs
#
# Phase 1, TP2 pairs at once:
#   GCD 0,1: tools/topk_epi_check.py (bitwise vs _topk_kernel, 1000 graph replays x 3 layers x
#            R 1-32, eager too), then tools/chain_gate.py EPI=1 (served verify step k=1,3 + parity)
#   GCD 2,3: tools/tp_parity.py teacher-forced EPI=1 against an EPI=0 reference (other process)
#   GCD 4,5: tools/tp_parity.py EPI=1 against an EPI=1 reference (cross-process repeatability)
#   GCD 6,7: tools/spec_step_bench.py EPI=1 (spec == plain 8/8), then chain_gate.py EPI=0
# Phase 2, all 8 GCDs: tools/topk_epi_ab.py served A/B EPI 0/1/0/1 with identical-token counts
#   per pair and the Triton compiles after /health (must be 0).
L=/tmp/p37
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
F='^=== |RESULT|Traceback|Error|error'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
(
  echo "=== [check] topk_epi_check $(date +%T)"
  HIP_VISIBLE_DEVICES=0,1 timeout 900 python tools/topk_epi_check.py --devices 0,1 --replays 1000 2>&1 \
    | tee -a $L/check.log | tag check
  echo "=== [check] exit ${PIPESTATUS[0]}"
  echo "=== [gate1] chain_gate ENGINE_TOPK_EPI=1 $(date +%T)"
  ENGINE_TOPK_EPI=1 HIP_VISIBLE_DEVICES=0,1 timeout 900 python tools/chain_gate.py --devices 0,1 2>&1 \
    | tee -a $L/gate1.log | tag gate1
  echo "=== [gate1] exit ${PIPESTATUS[0]}"
) &
(
  E="HIP_VISIBLE_DEVICES=2,3"
  echo "=== [onoff] tp_parity EPI=0 ref, EPI=1 check $(date +%T)"
  env $E ENGINE_TOPK_EPI=0 timeout 1200 python tools/tp_parity.py --mode ref --out /tmp/p37_off.pt 2>&1 \
    | tee -a $L/onoff.log | tag onoff-ref
  env $E ENGINE_TOPK_EPI=1 timeout 1200 python tools/tp_parity.py --mode check --ref /tmp/p37_off.pt 2>&1 \
    | tee -a $L/onoff.log | tag onoff
  echo "=== [onoff] exit ${PIPESTATUS[0]}"
) &
(
  E="HIP_VISIBLE_DEVICES=4,5"
  echo "=== [cross] tp_parity EPI=1 ref, EPI=1 check $(date +%T)"
  env $E ENGINE_TOPK_EPI=1 timeout 1200 python tools/tp_parity.py --mode ref --out /tmp/p37_on.pt 2>&1 \
    | tee -a $L/cross.log | tag cross-ref
  env $E ENGINE_TOPK_EPI=1 timeout 1200 python tools/tp_parity.py --mode check --ref /tmp/p37_on.pt 2>&1 \
    | tee -a $L/cross.log | tag cross
  echo "=== [cross] exit ${PIPESTATUS[0]}"
) &
(
  echo "=== [spec] spec_step_bench ENGINE_TOPK_EPI=1 $(date +%T)"
  ENGINE_TOPK_EPI=1 HIP_VISIBLE_DEVICES=6,7 timeout 1200 python tools/spec_step_bench.py --devices 6,7 \
    --iters 30 --max-new 128 2>&1 | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]}"
  echo "=== [gate0] chain_gate ENGINE_TOPK_EPI=0 $(date +%T)"
  ENGINE_TOPK_EPI=0 HIP_VISIBLE_DEVICES=6,7 timeout 900 python tools/chain_gate.py --devices 6,7 2>&1 \
    | tee -a $L/gate0.log | tag gate0
  echo "=== [gate0] exit ${PIPESTATUS[0]}"
) &
wait
echo "=== tracebacks (phase 1)"
for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | tail -30; done
echo "=== [ab] served A/B $(date +%T)"
unset ENGINE_TP ENGINE_MTP
timeout 2400 python tools/topk_epi_ab.py
echo "=== [ab] exit $? $(date +%T)"
