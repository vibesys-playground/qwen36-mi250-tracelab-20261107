#!/bin/sh
# Plan 35 on plan 33's captured prefill: the flash gates, plan 33's graph gates, prefill graph
# capture, and Triton compiles after /health (must be 0), on one MI250 node.
#
#   sh tools/plan35_graph_check.sh
echo "########## plan 35 gates (tools/prefill_flash_check.sh notune)"
sh tools/prefill_flash_check.sh notune 2>&1 | grep -v "^+ "
echo "########## plan 33 gates (tools/prefill_graph_check.sh)"
sh tools/prefill_graph_check.sh 2>&1 | grep -v "^+ "
echo "########## served run, ENGINE_LOG_JIT=1 (16 sessions, mixed prefill sizes)"
ENGINE_LOG_JIT=1 timeout 3600 python tools/serve_probe.py --sessions 16 --concurrency 8 --rounds 4 --first 1800 \
  --grow 480 --max-new 190 --corpus synthetic --settings "ENGINE_PREFILL_FLASH=1" 2>&1 | grep -v amdgpu.ids | cut -c1-900 | tail -5
S=/tmp/serve-probe-ENGINE_PREFILL_FLASH_1.log
echo "replica processes serving (engine-jit: serving): $(grep -c 'engine-jit: serving' $S)"
echo "compiles total: $(grep -c 'engine-jit: compile ' $S)  after serving: $(grep -c 'engine-jit: compile .*serving=True' $S)  post-compile hooks after serving: $(grep -c 'engine-jit: hook .*serving=True' $S)"
grep 'engine-jit: compile .*serving=True' $S | sed 's/.*fn=//' | sort | uniq -c | sort -rn | head -20
echo "########## prefill graph capture, every process"
for f in /tmp/pfl-logs/*.log /tmp/pg-logs/*.log $S; do
  echo "--- $f: $(grep -c 'engine: prefill graphs for' $f) captured, $(grep -c 'prefill graph capture failed' $f) failed"
  grep -m2 'engine: prefill graphs for\|prefill graph capture failed' $f | cut -c1-300
done
echo "########## tracebacks"; for f in /tmp/pfl-logs/*.log /tmp/pg-logs/*.log $S; do grep -B2 -A25 Traceback $f | head -40; done
true
