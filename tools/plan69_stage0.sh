#!/bin/sh
# Plan 69 stage 0 + gates, on one MI250 node (8 GCDs). Progress is printed live; every
# section has its own timeout.
#
#   sh tools/plan69_stage0.sh
#
# Phase A, in parallel:
#   CPU      incremental tokenize == full encode (tests incl. natural and replay-like corpora,
#            multi-byte and boundary cases) + the tokenizer microbench; parking unit tests
#   GCD 0,1  spec == plain greedy (tools/spec_step_bench.py), defaults
#   GCD 4,5  replay-like served load, every Triton compile logged: 0 own compiles after /health
# Phase B, whole node (4 x TP2), text prompts growing as the replay's (48 sessions, 8 at a
#   time, 1800 + 480 new, 190 out), request-path stamps on (adopt / detach / park timed):
#   1. stage 0: ENGINE_REQ_FAST 0 vs 1 (tail parking off), both orders
#   2. parking: ENGINE_SLOT_PARK 0 vs 1 (handler tokenize on), both orders
#   3. the candidate's defaults with every prompt checked against the full encode
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest) >/dev/null 2>&1
L=/tmp/p69-logs
mkdir -p $L
( HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= timeout 1200 python -m pytest -q -rs -p no:cacheprovider \
    tests/test_incremental_tokenize.py 2>&1 | tail -5; \
  timeout 600 python -u tools/tokenize_bench.py ) > $L/tok.log 2>&1 &
( timeout 2400 python tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
( HIP_VISIBLE_DEVICES=4,5 timeout 3000 python tools/live_compile_probe.py; \
  cp /tmp/live-compile-server.log $L/live-server.log ) > $L/compile.log 2>&1 &
wait
echo "=== tokenization"; cat $L/tok.log
echo "=== spec == plain"; grep -E "RESULT|Error|Traceback" $L/spec.log | tail -30
echo "=== live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/compile.log | cut -c1-3000 | tail -40
echo "=== tracebacks (phase A)"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done

P="timeout 1200 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 480 --max-new 190 --corpus synthetic --text"
D="python -u tools/plan69_digest.py"
S=ENGINE_REQ_STAMPS=1
echo "=== stage 0: REQ_FAST 0 then 1 (parking off)"
$P --settings $S+ENGINE_SLOT_PARK=0+ENGINE_REQ_FAST=0,$S+ENGINE_SLOT_PARK=0+ENGINE_REQ_FAST=1 2>&1 | $D
echo "=== stage 0: REQ_FAST 1 then 0 (parking off)"
$P --settings $S+ENGINE_SLOT_PARK=0+ENGINE_REQ_FAST=1,$S+ENGINE_SLOT_PARK=0+ENGINE_REQ_FAST=0 2>&1 | $D
echo "=== parking: SLOT_PARK 0 then 1 (REQ_FAST 1)"
$P --settings $S+ENGINE_REQ_FAST=1+ENGINE_SLOT_PARK=0,$S+ENGINE_REQ_FAST=1+ENGINE_SLOT_PARK=1 2>&1 | $D
echo "=== parking: SLOT_PARK 1 then 0 (REQ_FAST 1)"
$P --settings $S+ENGINE_REQ_FAST=1+ENGINE_SLOT_PARK=1,$S+ENGINE_REQ_FAST=1+ENGINE_SLOT_PARK=0 2>&1 | $D
echo "=== candidate defaults: head path (REQ_FAST 0, park 0) vs defaults + full-encode check"
$P --settings ENGINE_REQ_FAST=0+ENGINE_SLOT_PARK=0,$S+ENGINE_TOKENIZE_CHECK=1 2>&1 | $D
echo "=== tracebacks (phase B)"
for f in /tmp/serve-probe-*.log; do grep -B2 -A20 Traceback "$f" | head -40; done
true
