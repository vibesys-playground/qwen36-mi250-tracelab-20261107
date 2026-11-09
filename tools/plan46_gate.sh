#!/bin/sh
# Plan 46 gate: host request path off the engine thread, on one MI250 node (8 GCDs).
#
#   sh tools/plan46_gate.sh
#
# Phase A, in parallel:
#   CPU      incremental tokenization == full encode (tests + microbench with the checkpoint's
#            tokenizer)
#   GCD 0,1  spec == plain greedy, 8 sequences (tools/spec_step_bench.py), defaults
#   GCD 4,5  served replay-like load with every Triton compile logged: 0 after /health
# Phase B, whole node (4 x TP2), text prompts growing as the replay's (48 sessions, 8 at a
#   time, 1800 + 300 new, 190 out): A/B without stamps, tokenizing on the engine thread with
#   full encodes (ENGINE_REQ_FAST=0) then the candidate default, then the reverse order; then
#   the candidate with stamps and every prompt checked against the full encode. (The vocab
#   check fix in Generator._validate is on in both.)
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p46-logs
mkdir -p $L
( timeout 900 python -m pytest -q -rs tests/test_incremental_tokenize.py 2>&1 | tail -5; \
  timeout 600 python -u tools/tokenize_bench.py ) > $L/tok.log 2>&1 &
( timeout 2400 python tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
( HIP_VISIBLE_DEVICES=4,5 timeout 3000 python tools/live_compile_probe.py; \
  cp /tmp/live-compile-server.log $L/live-server.log ) > $L/compile.log 2>&1 &
wait
echo "=== tokenization"; cat $L/tok.log
echo "=== spec == plain"; grep -E "RESULT|Error|Traceback" $L/spec.log | tail -30
echo "=== live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/compile.log | cut -c1-3000 | tail -40
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done

P="timeout 1500 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 300 --max-new 190 --corpus synthetic --text"
OFF=ENGINE_REQ_FAST=0
ON=ENGINE_REQ_FAST=1
echo "=== served A/B: head path then candidate"
$P --settings $OFF,$ON 2>&1 | grep -v '^engine:' | grep -E '"event": "(setting|agreement)"' | sed 's/"health": .*"engine"/"engine"/'
echo "=== served A/B: candidate then head path"
$P --settings $ON,$OFF 2>&1 | grep -v '^engine:' | grep -E '"event": "(setting|agreement)"' | sed 's/"health": .*"engine"/"engine"/'
echo "=== served: candidate with stamps and the full-encode check"
$P --settings $ON+ENGINE_REQ_STAMPS=1+ENGINE_TOKENIZE_CHECK=1 2>&1 | grep -v '^engine:'
for f in /tmp/serve-probe-*.log; do grep -B2 -A20 Traceback "$f" | head -40; done
true
