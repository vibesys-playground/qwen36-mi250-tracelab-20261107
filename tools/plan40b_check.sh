#!/bin/sh
# Plan 40 part B: the fused GDN prefill chunk chain (ENGINE_GDN_CHUNK) on one MI250 node.
#
#   sh tools/plan40b_check.sh
#
# In parallel on GCD pairs (TP2 replicas, the server's default):
#   GCD 0,1 per-op prefill split on the captured path (kernel names) at 61/561/1,650 new tokens on a
#           3,600-token context, GDN_CHUNK=0 then 1; then the captured pass wall per bucket
#           (tools/prefill_graph_bench.py) GDN_CHUNK=0 then 1
#   GCD 2,3 teacher-forced top-1 of GDN_CHUNK=1 against a GDN_CHUNK=0 reference (tools/tp_parity.py),
#           then the eager per-call-site op split (--eager) GDN_CHUNK=0 then 1
#   GCD 4,5 GDN_CHUNK=1 cross-process (8/8), then in-process repeat (8/8)
#   GCD 6   GPU unit tests, then tools/gdn_chunk_check.py (fused vs fla vs fp32 rule, time, sweep); then
#   GCD 6,7 spec == plain (tools/spec_step_bench.py), then Triton compiles after /health under a served
#           replay-like load (tools/live_compile_probe.py)
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p40b-logs
mkdir -p $L
P=tools/tp_parity.py
( E="ENGINE_TP=2"; \
  echo "=== op split (graph) GDN_CHUNK=0"; env $E ENGINE_GDN_CHUNK=0 timeout 1500 python tools/prefill_op_split.py all --devices 0,1 --news 61,561,1650; \
  echo "=== op split (graph) GDN_CHUNK=1"; env $E ENGINE_GDN_CHUNK=1 timeout 1500 python tools/prefill_op_split.py all --devices 0,1 --news 61,561,1650; \
  echo "=== graph bench GDN_CHUNK=0"; env $E HIP_VISIBLE_DEVICES=0,1 ENGINE_GDN_CHUNK=0 timeout 1500 python tools/prefill_graph_bench.py --reps 12; \
  echo "=== graph bench GDN_CHUNK=1"; env $E HIP_VISIBLE_DEVICES=0,1 ENGINE_GDN_CHUNK=1 timeout 1500 python tools/prefill_graph_bench.py --reps 12 ) > $L/split.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3"; \
  env $E ENGINE_GDN_CHUNK=0 timeout 1800 python $P --mode ref --out /tmp/p40b_ref0.pt \
  && echo "--- GDN_CHUNK=1 against the GDN_CHUNK=0 reference" \
  && env $E ENGINE_GDN_CHUNK=1 timeout 1800 python $P --mode check --ref /tmp/p40b_ref0.pt; \
  echo "=== op split (eager) GDN_CHUNK=0"; env ENGINE_TP=2 ENGINE_GDN_CHUNK=0 timeout 1500 python tools/prefill_op_split.py all --eager --devices 2,3 --news 61,561,1650; \
  echo "=== op split (eager) GDN_CHUNK=1"; env ENGINE_TP=2 ENGINE_GDN_CHUNK=1 timeout 1500 python tools/prefill_op_split.py all --eager --devices 2,3 --news 61,561,1650 ) > $L/parity.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=4,5 ENGINE_GDN_CHUNK=1"; \
  env $E timeout 1800 python $P --mode ref --out /tmp/p40b_a.pt \
  && echo "--- GDN_CHUNK=1: a second process against the first" \
  && env $E timeout 1800 python $P --mode check --ref /tmp/p40b_a.pt; \
  echo "--- GDN_CHUNK=1 repeat in one process"; env $E timeout 1800 python $P --mode repeat --out /tmp/p40b_rep.pt ) > $L/cross.log 2>&1 &
( HIP_VISIBLE_DEVICES=6 timeout 1200 python -m pytest -q -rf tests/test_gdn_chunk.py tests/test_prefill_kernels.py 2>&1 | tail -25; \
  echo "--- gdn_chunk_check"; HIP_VISIBLE_DEVICES=6 timeout 1200 python tools/gdn_chunk_check.py; \
  echo "--- spec == plain"; ENGINE_GDN_CHUNK=1 timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128; \
  echo "--- live compiles"; \
  HIP_VISIBLE_DEVICES=6,7 ENGINE_GDN_CHUNK=1 timeout 3000 python tools/live_compile_probe.py ) > $L/tests.log 2>&1 &
wait

F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"
echo "=== unit tests, gdn_chunk_check, spec == plain, live compiles"; grep -vE "$F" $L/tests.log | cut -c1-4000 | tail -90
echo "=== teacher-forced top-1 GDN_CHUNK=1 vs 0, then eager op split"; grep -vE "$F" $L/parity.log | grep -vE '"kind": "(split|host_log)"' | cut -c1-600 | tail -150
echo "=== cross-process + repeat"; grep -E "RESULT|Error|^---|engine: prefill|engine: fast path|failed in mode" $L/cross.log | cut -c1-600 | tail -20
echo "=== op split (graph) + graph bench"; grep -vE "$F" $L/split.log | grep -vE '"kind": "(split|host_log)"' | cut -c1-600 | tail -200
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
