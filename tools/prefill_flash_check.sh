#!/bin/sh
# Plan 35 gates for the flash prefill attention kernel (ENGINE_PREFILL_FLASH, under
# ENGINE_PREFILL_FUSED) on one MI250 node, 8 GCDs.
#
#   sh tools/prefill_flash_check.sh [notune]
#
# Phase 0 on GCD 0: the unit tests (flash and fused prefill kernels), then the tile/split
# crossover sweep over T (tools/flash_prefill_probe.py --tune).
# Phase A, in parallel on separate GCD pairs (TP2 replicas):
#   GCD 0,1  teacher-forced top-1 of FLASH=1 against a FLASH=0 reference (FLASH=0 runs the
#            head's code, _prefill_attn_kernel), and FLASH=0 against it from another process
#            (the cross-process floor)
#   GCD 2,3  greedy repeatability with FLASH=1: in one process (tp_parity --mode repeat), then
#            across 2 processes (ref in one, check in another)
#   GCD 4,5  op split at 61/561/1,650 tokens on a 3,600-token context, FLASH=0 then FLASH=1
#   GCD 6,7  spec == plain greedy parity with FLASH=1 (tools/spec_step_bench.py)
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/pfl-logs
mkdir -p $L
export ENGINE_PREFILL_FUSED=1
HIP_VISIBLE_DEVICES=0 timeout 1200 python -m pytest -q tests/test_flash_prefill.py 2>&1 | tail -15
HIP_VISIBLE_DEVICES=0 timeout 900 python -m pytest -q tests/test_prefill_kernels.py 2>&1 | tail -5
if [ "$1" != "notune" ]; then
  HIP_VISIBLE_DEVICES=0 timeout 1500 python tools/flash_prefill_probe.py --tune --reps 10 > $L/tune.log 2>&1
  grep -v "^TUNE\|^@@" $L/tune.log | tail -20
fi

( P=tools/tp_parity.py; E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1"; \
  env $E ENGINE_PREFILL_FLASH=0 timeout 1800 python $P --mode ref --out /tmp/ref_rows.pt \
  && echo "--- FLASH=0 check against the FLASH=0 reference (another process: the floor)" \
  && env $E ENGINE_PREFILL_FLASH=0 timeout 1800 python $P --mode check --ref /tmp/ref_rows.pt \
  && echo "--- FLASH=1 check against the FLASH=0 reference (the head's kernel)" \
  && env $E ENGINE_PREFILL_FLASH=1 timeout 1800 python $P --mode check --ref /tmp/ref_rows.pt ) > $L/parity.log 2>&1 &
( P=tools/tp_parity.py; E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3"; \
  echo "--- FLASH=1 repeat in one process"; env $E ENGINE_PREFILL_FLASH=1 timeout 1800 python $P --mode repeat --out /tmp/rep1.pt; \
  echo "--- FLASH=1 ref (process 1)"; env $E ENGINE_PREFILL_FLASH=1 timeout 1800 python $P --mode ref --out /tmp/ref_flash.pt \
  && echo "--- FLASH=1 check against it (process 2)" \
  && env $E ENGINE_PREFILL_FLASH=1 timeout 1800 python $P --mode check --ref /tmp/ref_flash.pt ) > $L/repeat.log 2>&1 &
( ENGINE_PREFILL_FLASH=0 timeout 1800 python tools/prefill_op_split.py all --devices 4,5 ; \
  ENGINE_PREFILL_FLASH=1 timeout 1800 python tools/prefill_op_split.py all --devices 4,5 ) > $L/split.log 2>&1 &
( ENGINE_PREFILL_FLASH=1 timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
wait

echo "=== parity (FLASH=1 vs FLASH=0 reference; FLASH=0 floor first)"; grep -E "RESULT|Error|error|^---" $L/parity.log | tail -20
echo "=== repeatability (FLASH=1: in one process, then across 2 processes)"; grep -E "RESULT|Error|error|^---" $L/repeat.log | tail -20
echo "=== spec == plain (FLASH=1)"; grep -E "RESULT|Error|Traceback" $L/spec.log | tail -40
echo "=== op split (first FLASH=0, then FLASH=1)"
grep -v "^RESULT\|Loading weights\|amdgpu.ids" $L/split.log | grep -v "^\s*$" | tail -220
grep -E "^RESULT .*\"kind\": \"(kill|host_log)\"" $L/split.log | cut -c1-600
if [ "$1" != "notune" ]; then echo "=== tune runs"; grep "^TUNE" $L/tune.log; fi
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
