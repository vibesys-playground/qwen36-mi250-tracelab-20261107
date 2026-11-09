#!/bin/bash
# Plan 52 stage 1: the bulk in-graph all-reduce (ENGINE_TP_BULK_AR) on one MI250 node.
#
#   bash tools/plan52_gate.sh
#
# In parallel on GCD pairs (TP2 replicas, the server's default), every step under its own timeout:
#   GCD 0,1 captured pass wall per bucket at 61/561/1,650/2,048 new tokens (tools/prefill_graph_bench.py),
#           BULK_AR=0 (dumping logits, drafts, GDN state and KV), BULK_AR=1 (bitwise against that dump),
#           then BULK_AR=0 again (order check)
#   GCD 2,3 BULK_AR=1 teacher-forced greedy: a second process against the first (8/8), then repeat in one
#   GCD 4,5 per-op split of the captured pass at 561/1,650 (allreduce_kernel ms), BULK_AR=0 then 1
#   GCD 6,7 GPU unit test (tests/test_bulk_ar.py), spec == plain (BULK_AR=1), then Triton compiles after
#           /health under a served replay-like load (BULK_AR=1)
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p52-logs
mkdir -p $L
P=tools/tp_parity.py
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1"; \
  echo "=== graph bench BULK_AR=0"; env $E ENGINE_TP_BULK_AR=0 timeout 1500 python tools/prefill_graph_bench.py --news 61,561,1650,2048 --reps 15 --dump /tmp/p52_b0.pt; \
  echo "=== graph bench BULK_AR=1"; env $E ENGINE_TP_BULK_AR=1 timeout 1500 python tools/prefill_graph_bench.py --news 61,561,1650,2048 --reps 15 --against /tmp/p52_b0.pt; \
  echo "=== graph bench BULK_AR=0 again"; env $E ENGINE_TP_BULK_AR=0 timeout 1500 python tools/prefill_graph_bench.py --news 61,561,1650,2048 --reps 15 --against /tmp/p52_b0.pt ) > $L/bench.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 ENGINE_TP_BULK_AR=1"; \
  env $E timeout 1800 python $P --mode ref --out /tmp/p52_a.pt \
  && echo "--- BULK_AR=1: a second process against the first" \
  && env $E timeout 1800 python $P --mode check --ref /tmp/p52_a.pt; \
  echo "--- BULK_AR=1 repeat in one process"; env $E timeout 1800 python $P --mode repeat --out /tmp/p52_rep.pt ) > $L/cross.log 2>&1 &
( echo "=== op split (graph) BULK_AR=0"; env ENGINE_TP=2 ENGINE_TP_BULK_AR=0 timeout 1500 python tools/prefill_op_split.py all --devices 4,5 --news 561,1650 --reps 5; \
  echo "=== op split (graph) BULK_AR=1"; env ENGINE_TP=2 ENGINE_TP_BULK_AR=1 timeout 1500 python tools/prefill_op_split.py all --devices 4,5 --news 561,1650 --reps 5 ) > $L/split.log 2>&1 &
( HIP_VISIBLE_DEVICES=6,7 ENGINE_TP_BULK_AR=1 timeout 900 python -m pytest -q -rf tests/test_bulk_ar.py tests/test_tp.py 2>&1 | tail -25; \
  echo "--- spec == plain"; ENGINE_TP_BULK_AR=1 timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128; \
  echo "--- live compiles"; \
  HIP_VISIBLE_DEVICES=6,7 ENGINE_TP_BULK_AR=1 timeout 3000 python tools/live_compile_probe.py ) > $L/tests.log 2>&1 &
wait

F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"
echo "=== unit tests, spec == plain, live compiles"; grep -vE "$F" $L/tests.log | grep -v "^engine:" | cut -c1-4000 | tail -90
echo "=== cross-process + repeat"; grep -E "RESULT|Error|^---|failed in mode" $L/cross.log | cut -c1-600 | tail -20
echo "=== graph bench"; grep -vE "$F" $L/bench.log | grep -E "RESULT|^===|^ *[0-9]+ |N  *size|Error" | cut -c1-1500 | tail -60
echo "=== op split (graph)"; grep -vE "$F" $L/split.log | grep -vE '"kind": "(split|host_log)"' | grep -v "^engine:" | cut -c1-600 | tail -120
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
