#!/bin/sh
# Plan 26 stage 1: the fused prefill path (ENGINE_PREFILL_FUSED) on one MI250 node, 8 GCDs.
#
#   sh tools/prefill_fused_check.sh [serve]
#
# Phase A, in parallel on separate GCD pairs (TP2 replicas):
#   GCD 0,1  teacher-forced top-1 of FUSED=1 against a FUSED=0 reference (tools/tp_parity.py)
#   GCD 2,3  greedy repeatability: FUSED=1 against a FUSED=1 reference (identical prefixes)
#   GCD 4,5  per-call-site GPU time, launches and pass wall, FUSED=0 then FUSED=1
#            (tools/prefill_op_split.py, resume 61/561/1,650 tokens on a 3,600-token context)
#   GCD 6,7  spec == plain greedy parity with FUSED=1 (tools/spec_step_bench.py)
# after the kernels' unit tests on GCD 0. Phase B (with "serve"): served A/B/A/B of
# ENGINE_PREFILL_FUSED=0/1 on the replay-like shape (tools/serve_probe.py, all 8 GCDs).
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/pf-logs
mkdir -p $L
HIP_VISIBLE_DEVICES=0 timeout 900 python -m pytest -q tests/test_prefill_kernels.py tests/test_moe_prefill.py 2>&1 | tail -15

( ENGINE_TP=2 ENGINE_PREFILL_FUSED=0 HIP_VISIBLE_DEVICES=0,1 timeout 1800 python tools/tp_parity.py --mode ref --out /tmp/ref0.pt \
  && ENGINE_TP=2 ENGINE_PREFILL_FUSED=1 HIP_VISIBLE_DEVICES=0,1 timeout 1800 python tools/tp_parity.py --mode check --ref /tmp/ref0.pt ) > $L/parity.log 2>&1 &
( ENGINE_TP=2 ENGINE_PREFILL_FUSED=1 HIP_VISIBLE_DEVICES=2,3 timeout 1800 python tools/tp_parity.py --mode ref --out /tmp/ref1.pt \
  && ENGINE_TP=2 ENGINE_PREFILL_FUSED=1 HIP_VISIBLE_DEVICES=2,3 timeout 1800 python tools/tp_parity.py --mode check --ref /tmp/ref1.pt ) > $L/repeat.log 2>&1 &
( ENGINE_PREFILL_FUSED=0 timeout 1800 python tools/prefill_op_split.py all --devices 4,5 ; \
  ENGINE_PREFILL_FUSED=1 timeout 1800 python tools/prefill_op_split.py all --devices 4,5 ) > $L/split.log 2>&1 &
( ENGINE_PREFILL_FUSED=1 timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
wait

echo "=== parity (FUSED=1 vs FUSED=0 reference)"; grep -E "RESULT|Error|error" $L/parity.log | tail -12
echo "=== repeatability (FUSED=1 vs FUSED=1 reference)"; grep -E "RESULT|Error|error" $L/repeat.log | tail -12
echo "=== spec == plain (FUSED=1)"; grep -E "RESULT|Error|Traceback" $L/spec.log | tail -40
echo "=== op split (first FUSED=0, then FUSED=1)"
grep -v "^RESULT\|Loading weights\|amdgpu.ids" $L/split.log | grep -v "^\s*$" | tail -200
grep -E "^RESULT .*\"kind\": \"(kill|host_log)\"" $L/split.log | cut -c1-600

if [ "$1" = "serve" ]; then
  timeout 4800 python tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 \
    --max-new 190 --corpus synthetic \
    --settings "ENGINE_PREFILL_FUSED=0,ENGINE_PREFILL_FUSED=1,ENGINE_PREFILL_FUSED=0+ENGINE_AB_REPEAT=2,ENGINE_PREFILL_FUSED=1+ENGINE_AB_REPEAT=2"
fi
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
