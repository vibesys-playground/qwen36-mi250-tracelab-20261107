#!/bin/sh
# Plan 56 gate: the fused attention Q/K prologue (A, ENGINE_PREFILL_QK_FUSED) and the
# one-program MoE tile table (B, ENGINE_MOE_TILES_ONE) on one MI250 node, 8 GCDs.
#
#   sh tools/plan56_gate.sh
#
# Flags are written AB: 00 = both off (head 5a7e444's ops), 11 = both on (the default).
# Phase A, in parallel on GCD pairs (TP2 replicas, the server's default):
#   GCD 0,1  graph-mode per-kernel-name split (captured passes) at 61/561/1,650 new tokens on a
#            3,600-token context, flags 00 then 11; then the captured pass wall
#            (prefill_graph_bench) 00 then 11
#   GCD 2,3  the same split with 10 then 01 (each part alone)
#   GCD 4,5  teacher-forced top-1 of 11 against a 00 reference; 11 cross-process (8/8) and
#            in-process repeat (8/8)
#   GCD 6    GPU unit tests, then tools/qk_prologue_bench.py (kernel us, bitwise checks); then
#   GCD 6,7  spec == plain (11), then Triton compiles after /health under a served load (11)
# Phase B, whole node (4 x TP2): served replay-like A/B 00 / 11 / 00 / 11.
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p56-logs
mkdir -p $L
P=tools/tp_parity.py
S="timeout 1500 python -u tools/prefill_op_split.py all --news 61,561,1650"
fl() { echo "ENGINE_PREFILL_QK_FUSED=${1%?} ENGINE_MOE_TILES_ONE=${1#?}"; }
( for f in 00 11; do echo "=== op split (graph) flags $f"; env ENGINE_TP=2 $(fl $f) $S --devices 0,1; done; \
  for f in 00 11; do echo "=== graph bench flags $f"; \
    env ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1 $(fl $f) timeout 1500 python -u tools/prefill_graph_bench.py --reps 12; done \
) > $L/split01.log 2>&1 &
( for f in 10 01; do echo "=== op split (graph) flags $f"; env ENGINE_TP=2 $(fl $f) $S --devices 2,3; done \
) > $L/split23.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=4,5"; \
  env $E $(fl 00) timeout 1800 python $P --mode ref --out /tmp/p56_ref00.pt \
  && echo "--- 11 against the 00 reference" \
  && env $E $(fl 11) timeout 1800 python $P --mode check --ref /tmp/p56_ref00.pt; \
  env $E $(fl 11) timeout 1800 python $P --mode ref --out /tmp/p56_a.pt \
  && echo "--- 11: a second process against the first" \
  && env $E $(fl 11) timeout 1800 python $P --mode check --ref /tmp/p56_a.pt; \
  echo "--- 11 repeat in one process"; env $E $(fl 11) timeout 1800 python $P --mode repeat --out /tmp/p56_rep.pt \
) > $L/parity.log 2>&1 &
( HIP_VISIBLE_DEVICES=6 timeout 1500 python -X faulthandler -m pytest -q -rf -p no:cacheprovider \
    tests/test_prefill_kernels.py tests/test_moe_prefill.py 2>&1 | tail -25; \
  echo "--- qk_prologue_bench"; HIP_VISIBLE_DEVICES=6 timeout 900 python -u tools/qk_prologue_bench.py; \
  echo "--- spec == plain (11)"; env $(fl 11) timeout 2400 python -u tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128; \
  echo "--- live compiles (11)"; env HIP_VISIBLE_DEVICES=6,7 $(fl 11) timeout 3000 python tools/live_compile_probe.py; \
  cp /tmp/live-compile-server.log $L/live-server.log 2>/dev/null \
) > $L/tests.log 2>&1 &
wait

F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced|^engine:"
echo "=== unit tests, micro-bench, spec == plain, live compiles"; grep -vE "$F" $L/tests.log | cut -c1-2000 | tail -90
echo "=== teacher-forced top-1 11 vs 00, cross-process, repeat"; grep -E "RESULT|Error|^---|failed in mode" $L/parity.log | cut -c1-600 | tail -30
for f in split01 split23; do
  echo "=== $f"; grep -vE "$F" $L/$f.log | grep -vE '"kind": "(split|host_log)"' | cut -c1-260 | tail -230
done
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done

PR="timeout 1500 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 --max-new 190 --corpus synthetic"
echo "=== served A/B 00 / 11 / 00 / 11"
$PR --settings ENGINE_PREFILL_QK_FUSED=0+ENGINE_MOE_TILES_ONE=0,default,ENGINE_PREFILL_QK_FUSED=0+ENGINE_MOE_TILES_ONE=0+ENGINE_AB_REPEAT=2,ENGINE_AB_REPEAT=2 2>&1 | grep -v '^engine:'
true
