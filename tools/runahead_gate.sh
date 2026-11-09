#!/bin/sh
# Plan 42 gate: run-ahead chained verify (ENGINE_RUNAHEAD) on one MI250 node, 8 GCDs.
#
#   sh tools/runahead_gate.sh
#
# Phase A, in parallel on separate GCD pairs:
#   GCD 0,1  spec == plain greedy, 8 sequences, ENGINE_RUNAHEAD=1 (tools/spec_step_bench.py)
#   GCD 2,3  the same with ENGINE_RUNAHEAD=0
#   GCD 4,5  served replay-like load with every Triton compile logged (live_compile_probe.py),
#            run-ahead on (the default): compiles after /health must be 0
#   GCD 6    the run-ahead input kernel against the host layout, chained verify unit tests
# Phase B, whole node (4 x TP2): serve probe on the replay-like shape (48 sessions, 8 at a
#   time, 1800 + 60, 190 out), with loop timers: RUNAHEAD=0 then 1 (GPU idle per step, outputs
#   identical); without timers: RUNAHEAD=1 then 0 (A/B); if either pair is not 302/302
#   identical, RUNAHEAD=0 twice (the floor of run-to-run batch numerics).
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/ra-logs
mkdir -p $L
( ENGINE_RUNAHEAD=1 timeout 2400 python tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 ) > $L/spec1.log 2>&1 &
( ENGINE_RUNAHEAD=0 timeout 2400 python tools/spec_step_bench.py --devices 2,3 --iters 10 --max-new 128 ) > $L/spec0.log 2>&1 &
( HIP_VISIBLE_DEVICES=4,5 timeout 3000 python tools/live_compile_probe.py; \
  cp /tmp/live-compile-server.log $L/live-server.log ) > $L/compile.log 2>&1 &
( HIP_VISIBLE_DEVICES=6 timeout 1800 python -m pytest -q -rf tests/test_runahead.py tests/test_spec_chain.py 2>&1 | tail -25 ) > $L/tests.log 2>&1 &
wait
echo "=== spec == plain, RUNAHEAD=1"; grep -E "RESULT|Error|Traceback" $L/spec1.log | tail -30
echo "=== spec == plain, RUNAHEAD=0"; grep -E "RESULT|Error|Traceback" $L/spec0.log | tail -30
echo "=== live compiles (RUNAHEAD default)"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/compile.log | cut -c1-3000 | tail -40
grep -E "runahead|Traceback|Error" $L/live-server.log | tail -10
echo "=== unit tests"; cat $L/tests.log
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done

P="timeout 1500 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 --max-new 190 --corpus synthetic"
echo "=== served, timers: RUNAHEAD 0 then 1"
$P --settings ENGINE_LOOP_TIMERS=1+ENGINE_RUNAHEAD=0,ENGINE_LOOP_TIMERS=1+ENGINE_RUNAHEAD=1 2>&1 | grep -v '^engine:' | tee $L/served_t.log
echo "=== served, no timers: RUNAHEAD 1 then 0"
$P --settings ENGINE_RUNAHEAD=1,ENGINE_RUNAHEAD=0 2>&1 | grep -v '^engine:' | tee $L/served.log
if ! grep -q '"identical_requests": 302' $L/served_t.log || ! grep -q '"identical_requests": 302' $L/served.log; then
  echo "=== served floor: RUNAHEAD 0 twice"
  $P --settings ENGINE_RUNAHEAD=0,ENGINE_RUNAHEAD=0+ENGINE_AB_REPEAT=2 2>&1 | grep -v '^engine:' | grep -E '"agreement"|"setting"' | cut -c1-600
fi
true
