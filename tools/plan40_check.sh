#!/bin/sh
# Plan 40 part A: merged prefill projection GEMMs (ENGINE_PREFILL_MERGED) on one MI250 node.
#
#   sh tools/plan40_check.sh
#
# In parallel on GCD pairs (TP2 replicas, the server's default):
#   GCD 0,1 per-op prefill split (tools/prefill_op_split.py) at 61/561/1,650 new tokens on a
#           3,600-token context, MERGED=0 then MERGED=1; then the captured pass wall per
#           bucket (tools/prefill_graph_bench.py) MERGED=0 then MERGED=1
#   GCD 2,3 teacher-forced top-1 of MERGED=1 against a MERGED=0 reference (tools/tp_parity.py)
#   GCD 4,5 MERGED=1 cross-process (8/8), then in-process repeat (8/8)
#   GCD 6   GPU unit tests (tests/test_prefill_kernels.py, tests/test_moe_prefill.py), then
#   GCD 6,7 spec == plain (tools/spec_step_bench.py), then Triton compiles after /health under
#           a served replay-like load (tools/live_compile_probe.py)
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p40-logs
mkdir -p $L
P=tools/tp_parity.py
( E="ENGINE_TP=2"; \
  echo "=== op split MERGED=0"; env $E ENGINE_PREFILL_MERGED=0 timeout 1800 python tools/prefill_op_split.py all --devices 0,1 --news 61,561,1650; \
  echo "=== op split MERGED=1"; env $E ENGINE_PREFILL_MERGED=1 timeout 1800 python tools/prefill_op_split.py all --devices 0,1 --news 61,561,1650; \
  echo "=== graph bench MERGED=0"; env $E HIP_VISIBLE_DEVICES=0,1 ENGINE_PREFILL_MERGED=0 timeout 1500 python tools/prefill_graph_bench.py --reps 12; \
  echo "=== graph bench MERGED=1"; env $E HIP_VISIBLE_DEVICES=0,1 ENGINE_PREFILL_MERGED=1 timeout 1500 python tools/prefill_graph_bench.py --reps 12 ) > $L/split.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3"; \
  env $E ENGINE_PREFILL_MERGED=0 timeout 1800 python $P --mode ref --out /tmp/p40_ref0.pt \
  && echo "--- MERGED=1 against the MERGED=0 reference" \
  && env $E ENGINE_PREFILL_MERGED=1 timeout 1800 python $P --mode check --ref /tmp/p40_ref0.pt ) > $L/parity.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=4,5 ENGINE_PREFILL_MERGED=1"; \
  env $E timeout 1800 python $P --mode ref --out /tmp/p40_a.pt \
  && echo "--- MERGED=1: a second process against the first" \
  && env $E timeout 1800 python $P --mode check --ref /tmp/p40_a.pt; \
  echo "--- MERGED=1 repeat in one process"; env $E timeout 1800 python $P --mode repeat --out /tmp/p40_rep.pt ) > $L/cross.log 2>&1 &
( HIP_VISIBLE_DEVICES=6 timeout 1200 python -m pytest -q -rf tests/test_prefill_kernels.py tests/test_moe_prefill.py 2>&1 | tail -25; \
  echo "--- spec == plain"; ENGINE_PREFILL_MERGED=1 timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128; \
  echo "--- live compiles"; \
  HIP_VISIBLE_DEVICES=6,7 ENGINE_PREFILL_MERGED=1 timeout 3000 python tools/live_compile_probe.py ) > $L/tests.log 2>&1 &
wait

echo "=== op split + graph bench"; grep -vE "Loading weights|amdgpu.ids|^\s*$|it/s\]" $L/split.log | cut -c1-400 | tail -260
echo "=== teacher-forced top-1, MERGED=1 vs MERGED=0"; grep -E "RESULT|Error|^---|engine: prefill|engine: fast path|failed in mode" $L/parity.log | tail -14
echo "=== cross-process + repeat"; grep -E "RESULT|Error|^---|engine: prefill|engine: fast path|failed in mode" $L/cross.log | tail -20
echo "=== unit tests, spec == plain, live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$|it/s\]" $L/tests.log | cut -c1-3000 | tail -80
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
