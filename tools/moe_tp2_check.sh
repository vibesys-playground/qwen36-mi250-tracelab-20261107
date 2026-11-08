#!/bin/sh
# Plan 38 step 1: token-count-bucketed MoE prefill tiles, tile table, BF16 pair buffer.
#
#   sh tools/moe_tp2_check.sh
#
# In parallel on separate GCD pairs (TP2 replicas, the server's default):
#   GCD 0   GPU unit tests (tests/test_moe_prefill.py against moe_prefill_reference and the
#           padded/loop paths, tile table, buckets; tests/test_prefill_kernels.py), then
#   GCD 0,1 this commit's prefill MoE block against the head's (6a89a83) bitwise: every call
#           and the last-token logits, 3,600-token context + 61/561/1,650 new tokens
#           (tools/moe_head_bitwise.py)
#   GCD 2,3 greedy repeatability in one process, 8 prompts (tools/tp_parity.py --mode repeat)
#   GCD 4,5 per-call-site GPU time (tools/prefill_op_split.py, 61/561/1,650 on 3,600)
#   GCD 6,7 spec == plain greedy, 8 sequences (tools/spec_step_bench.py)
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/m38-logs
mkdir -p $L
( HIP_VISIBLE_DEVICES=0 timeout 1200 python -m pytest -q tests/test_moe_prefill.py tests/test_prefill_kernels.py 2>&1 | tail -15; \
  echo "--- bitwise vs head"; \
  ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1 timeout 2400 python tools/moe_head_bitwise.py --news 61,561,1650 ) > $L/bitwise.log 2>&1 &
( ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 timeout 2400 python tools/tp_parity.py --mode repeat --out /tmp/rep.pt ) > $L/repeat.log 2>&1 &
( timeout 2400 python tools/prefill_op_split.py all --devices 4,5 ) > $L/split.log 2>&1 &
( timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
wait

echo "=== unit tests + bitwise vs head"; grep -E "passed|failed|FAILED|Error|RESULT|^---" $L/bitwise.log | cut -c1-600 | tail -40
echo "=== repeatability (one process)"; grep -E "RESULT|Error|error|^---" $L/repeat.log | tail -20
echo "=== spec == plain"; grep -E "RESULT|Error|Traceback" $L/spec.log | tail -40
echo "=== op split"
grep -v "^RESULT\|Loading weights\|amdgpu.ids" $L/split.log | grep -v "^\s*$" | tail -120
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
