#!/bin/sh
# Plan 33 step 1: gates of the captured prefill passes (ENGINE_PREFILL_GRAPH) on one MI250 node.
#
#   sh tools/prefill_graph_check.sh [serve]
#
# Phase A, in parallel on GCD pairs (TP2 replicas, tools/tp_parity.py: 8 prompts, greedy):
#   GCD 0,1  reference with ENGINE_PREFILL_GRAPH=0 (the parent's eager prefill), then
#            teacher-forced top-1 of GRAPH=1 against it (gate >= 98.5%)
#   GCD 2,3  GRAPH=1 reference, then GRAPH=1 in a second process against it (8/8 identical)
#   GCD 4,5  GRAPH=1 repeatability in one process (tp_parity --mode repeat), then
#            tools/prefill_graph_bench.py: graph vs eager pass timing at 61/561/700/1650 new
#            tokens on a 3,600-token context, exactness and the padding leak test
#   GCD 6,7  spec == plain greedy with GRAPH=1 (tools/spec_step_bench.py)
# Phase B (with "serve"): served A/B of ENGINE_PREFILL_GRAPH=0/1 (tools/serve_probe.py, 8 GCDs).
set -x
L=/tmp/pg-logs
mkdir -p $L
P=tools/tp_parity.py
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1"; \
  env $E ENGINE_PREFILL_GRAPH=0 timeout 1800 python $P --mode ref --out /tmp/pg_ref0.pt \
  && echo "--- GRAPH=1 against the GRAPH=0 reference" \
  && env $E ENGINE_PREFILL_GRAPH=1 timeout 1800 python $P --mode check --ref /tmp/pg_ref0.pt ) > $L/parity.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 ENGINE_PREFILL_GRAPH=1"; \
  env $E timeout 1800 python $P --mode ref --out /tmp/pg_a.pt \
  && echo "--- GRAPH=1: a second process against the first" \
  && env $E timeout 1800 python $P --mode check --ref /tmp/pg_a.pt ) > $L/cross.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=4,5 ENGINE_PREFILL_GRAPH=1"; \
  echo "--- GRAPH=1 repeat in one process"; env $E timeout 1800 python $P --mode repeat --out /tmp/pg_rep.pt; \
  echo "--- bench"; env $E timeout 1800 python tools/prefill_graph_bench.py --reps 12 ) > $L/bench.log 2>&1 &
( ENGINE_PREFILL_GRAPH=1 timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
wait

echo "=== teacher-forced top-1, GRAPH=1 vs GRAPH=0 reference"; grep -E "RESULT|Error|^---|engine: prefill|engine: fast path|failed in mode" $L/parity.log | tail -14
echo "=== cross-process, GRAPH=1 vs GRAPH=1"; grep -E "RESULT|Error|^---|engine: prefill|engine: fast path|failed in mode" $L/cross.log | tail -14
echo "=== in-process repeat + bench"; grep -E "RESULT|Error|^---|engine: prefill|^ +[0-9]+ |^===|N +size" $L/bench.log | cut -c1-1500 | tail -40
echo "=== spec == plain (GRAPH=1)"; grep -E "RESULT|Error|Traceback|engine: prefill|engine: fast path|failed in mode" $L/spec.log | tail -30

if [ "$1" = "serve" ]; then
  timeout 5400 python tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 480 \
    --max-new 190 --corpus synthetic \
    --settings "ENGINE_PREFILL_GRAPH=0,ENGINE_PREFILL_GRAPH=1"
fi
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
