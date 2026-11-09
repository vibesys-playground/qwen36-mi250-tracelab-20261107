#!/bin/sh
# Plan 51 gate: chain head glue in Triton (ENGINE_CHAIN_GLUE) and the out_proj/o_proj K-sum in
# the all-reduce (ENGINE_AR_KSUM) on one MI250 node, 8 GCDs.
#
#   sh tools/plan51_gate.sh
#
# Phase A, in parallel on separate GCD pairs (TP2 replicas):
#   GCD 0,1  glue_split chain k=3 B=1,2 + parity hashes, flags (GLUE,KSUM) 00 10 01 11
#   GCD 2,3  the same in the reverse order 11 01 10 00
#   GCD 4,5  chain_gate (step ms, acceptance, spec == plain on the chained path) 00 then 11
#   GCD 6,7  unit tests (tools/plan51_tests.sh, per file), spec == plain (spec_step_bench, defaults), then Triton compiles after
#            /health under a served replay-like load (live_compile_probe, defaults)
# Phase B, whole node (4 x TP2): served replay-like A/B 00 / 11 / 00 / 11.
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p51-logs
mkdir -p $L
G="timeout 900 python -u tools/glue_split.py --configs chain:1:3,chain:2:3 --parity"
( for f in 00 10 01 11; do echo "=== flags $f"; \
    env ENGINE_CHAIN_GLUE=${f%?} ENGINE_AR_KSUM=${f#?} $G --devices 0,1; done ) > $L/split01.log 2>&1 &
( for f in 11 01 10 00; do echo "=== flags $f"; \
    env ENGINE_CHAIN_GLUE=${f%?} ENGINE_AR_KSUM=${f#?} $G --devices 2,3; done ) > $L/split23.log 2>&1 &
( for f in 00 11; do echo "=== flags $f"; \
    env ENGINE_CHAIN_GLUE=${f%?} ENGINE_AR_KSUM=${f#?} timeout 1800 python -u tools/chain_gate.py --devices 4,5; done ) > $L/gate.log 2>&1 &
( sh tools/plan51_tests.sh; \
  echo "--- spec == plain (defaults)"; timeout 2400 python -u tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128; \
  echo "--- live compiles (defaults)"; HIP_VISIBLE_DEVICES=6,7 timeout 3000 python tools/live_compile_probe.py; \
  cp /tmp/live-compile-server.log $L/live-server.log ) > $L/tests.log 2>&1 &
wait

F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced|^engine:"
for f in split01 split23; do
  echo "=== glue_split $f"
  grep -E "^=== flags|RESULT (wall|step|parity_hash|glue|split chain3 B=2 gcd=1 item=(5_mtp|other))|RESULT top chain3 B=2 gcd=1 other|Traceback|Error" $L/$f.log | cut -c1-400
done
echo "=== chain_gate"; grep -E "^=== flags|RESULT|Traceback|Error" $L/gate.log | cut -c1-600
echo "=== unit tests, spec == plain, live compiles"; grep -vE "$F" $L/tests.log | cut -c1-2000 | tail -80
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done

P="timeout 1500 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 --max-new 190 --corpus synthetic"
echo "=== served A/B 00 / 11 / 00 / 11"
$P --settings ENGINE_CHAIN_GLUE=0+ENGINE_AR_KSUM=0,default,ENGINE_CHAIN_GLUE=0+ENGINE_AR_KSUM=0+ENGINE_AB_REPEAT=2,ENGINE_AB_REPEAT=2 2>&1 | grep -v '^engine:'
true
