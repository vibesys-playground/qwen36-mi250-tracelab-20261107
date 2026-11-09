#!/bin/sh
# Plan 61 gate on one MI250 node (8 GCDs), with the shipped table
# (engine/fast/blas_tuned_gfx90a.csv, ENGINE_PREFILL_BLAS_TUNE=1 = the default).
#
#   sh tools/plan61_gate.sh
#
# Phase A (parallel, TP2 pairs):
#   GCD 0,1  graph-mode prefill op split OFF then ON at 61/561/1,650 new tokens
#   GCD 2,3  teacher-forced top-1 ON vs an OFF reference; ON cross-process (shipped table)
#   GCD 4,5  ON eager-prefill vs OFF eager-prefill (decode graphs bitwise); ON in-process repeat
#   GCD 6    GPU unit tests (blas_tune, prefill kernels); then
#   GCD 6,7  spec == plain (ON), then Triton compiles after /health under a served load (ON)
# Phase B, whole node (4 x TP2): served replay-like A/B OFF / ON / OFF / ON (TTFT, /health time).
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
D=/tmp/p61g
L=$D/logs
mkdir -p $L
export PYTHONUNBUFFERED=1 ENGINE_TP=2
OFF="ENGINE_PREFILL_BLAS_TUNE=0"
ON="ENGINE_PREFILL_BLAS_TUNE=1"
P=tools/tp_parity.py
S="timeout 1500 python -u tools/prefill_op_split.py all --devices 0,1 --news 61,561,1650 --reps 5"
G="RESULT|Traceback|^---|^===|prefill BLAS|capture failed|passed|failed"
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"

echo "=== phase A $(date +%T)"
( for f in "$OFF" "$ON"; do echo "=== op split (graph) $f"; env $f $S; done \
) 2>&1 | tee $L/split01.log | grep --line-buffered -E "$G" | cut -c1-400 &
( E="HIP_VISIBLE_DEVICES=2,3"; \
  env $E $OFF timeout 1800 python -u $P --mode ref --out $D/ref_off.pt \
  && echo "--- ON against the OFF reference (teacher-forced top-1)" \
  && env $E $ON timeout 1800 python -u $P --mode check --ref $D/ref_off.pt; \
  env $E $ON timeout 1800 python -u $P --mode ref --out $D/ref_on.pt \
  && echo "--- ON: a second process against the first (shipped table)" \
  && env $E $ON timeout 1800 python -u $P --mode check --ref $D/ref_on.pt \
) 2>&1 | tee $L/par23.log | grep --line-buffered -E "$G" | cut -c1-600 &
( E="HIP_VISIBLE_DEVICES=4,5"; \
  env $E $OFF timeout 1800 python -u $P --mode ref --eager-prefill --out $D/ref_off_eager.pt \
  && echo "--- ON eager-prefill against OFF eager-prefill (decode graphs bitwise)" \
  && env $E $ON timeout 1800 python -u $P --mode check --eager-prefill --ref $D/ref_off_eager.pt; \
  echo "--- ON repeat in one process"; env $E $ON timeout 1800 python -u $P --mode repeat --out $D/rep.pt \
) 2>&1 | tee $L/par45.log | grep --line-buffered -E "$G" | cut -c1-600 &
( HIP_VISIBLE_DEVICES=6 timeout 1500 python -X faulthandler -m pytest -q -rf -p no:cacheprovider \
    tests/test_blas_tune.py tests/test_prefill_kernels.py 2>&1 | tail -25; \
  echo "--- spec == plain (ON)"; env $ON timeout 2400 python -u tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128; \
  echo "--- live compiles (ON)"; env HIP_VISIBLE_DEVICES=6,7 $ON timeout 3000 python tools/live_compile_probe.py \
) 2>&1 | tee $L/tests.log | grep --line-buffered -E "$G" | cut -c1-900 &
wait

echo "=== phase A summary $(date +%T)"
echo "--- unit tests, spec == plain, live compiles"; grep -vE "$F" $L/tests.log | grep -E "$G" | cut -c1-2000 | tail -60
echo "--- top-1, cross-process, decode bitwise, repeat"; grep -hE "RESULT|Error|^---|failed in mode" $L/par23.log $L/par45.log | cut -c1-600
echo "--- op split"; grep -vE "$F" $L/split01.log | grep -vE '"kind": "(split|host_log)"' | cut -c1-260 | tail -230
echo "--- tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done

PR="timeout 1500 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 --max-new 190 --corpus synthetic"
echo "=== phase B served A/B OFF / ON / OFF / ON $(date +%T)"
$PR --settings ENGINE_PREFILL_BLAS_TUNE=0,default,ENGINE_PREFILL_BLAS_TUNE=0+ENGINE_AB_REPEAT=2,ENGINE_AB_REPEAT=2 2>&1 | grep -v '^engine:'
true
