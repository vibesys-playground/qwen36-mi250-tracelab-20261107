#!/bin/sh
# Plan 38 on head 0d89b27 (captured prefill graphs, plan 33): token-count-bucketed MoE prefill
# tiles, tile table, BF16 pair buffer.
#
#   sh tools/moe_tp2_check.sh
#
# In parallel on separate GCD pairs (TP2 replicas, the server's default):
#   GCD 0,1 this commit's prefill MoE block against the head's (0d89b27) bitwise: eager per call,
#           and last-token logits with prefill graphs captured with each block, 3,600-token
#           context + 61/561/1,650 new tokens (tools/moe_head_bitwise.py)
#   GCD 2,3 greedy repeatability in one process, 8 prompts (tools/tp_parity.py --mode repeat)
#   GCD 4   GPU unit tests (tests/test_moe_prefill.py, tests/test_prefill_kernels.py), then
#   GCD 4,5 a served replay-like load with every Triton compile logged: compiles after /health
#           (tools/live_compile_probe.py)
#   GCD 6,7 spec == plain greedy, 8 sequences (tools/spec_step_bench.py)
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/m38-logs
mkdir -p $L
( ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1 timeout 3000 python tools/moe_head_bitwise.py --news 61,561,1650 ) > $L/bitwise.log 2>&1 &
( ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 timeout 2400 python tools/tp_parity.py --mode repeat --out /tmp/rep.pt ) > $L/repeat.log 2>&1 &
( HIP_VISIBLE_DEVICES=4 timeout 1200 python -m pytest -q -rf tests/test_moe_prefill.py tests/test_prefill_kernels.py 2>&1 | tail -25; \
  echo "--- live compiles"; \
  HIP_VISIBLE_DEVICES=4,5 timeout 3000 python tools/live_compile_probe.py ) > $L/tests.log 2>&1 &
( timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
wait

echo "=== bitwise vs head"; grep -E "RESULT|Error|engine: prefill" $L/bitwise.log | cut -c1-700 | tail -30
echo "=== repeatability (one process)"; grep -E "RESULT|Error|error|^---|engine: prefill" $L/repeat.log | tail -12
echo "=== unit tests + live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/tests.log | cut -c1-3000 | tail -60
echo "=== spec == plain"; grep -E "RESULT|Error|Traceback|engine: prefill" $L/spec.log | tail -30
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
