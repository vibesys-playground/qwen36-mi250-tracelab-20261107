#!/bin/sh
# Plan 61 step 0 on one MI250 node (8 GCDs): tune the captured prefill graphs' GEMMs offline
# and check the tuned table (tools/blas_tune_probe.py, engine/fast/blas_tune.py).
#
#   sh tools/plan61_step0.sh
#
# OFF = ENGINE_PREFILL_BLAS_TUNE=0 (head's kernels); ON = the table this job tunes.
# Phase 1 (parallel): GCD 0,1 record the prefill graphs' GEMMs; GCD 2,3 teacher-forced
#   reference OFF; GCD 4,5 reference OFF with every prefill pass eager (decode graphs only).
# Phase 2: offline tuning of the recorded shapes on all 8 GCDs (nothing else running).
# Phase 3 (parallel): GCD 0 per-shape default vs tuned; then GCD 0,1 spec == plain ON;
#   GCD 2,3 top-1 ON vs OFF, then ON cross-process; GCD 4,5 ON eager-prefill vs OFF (decode
#   bitwise); GCD 6,7 graph-mode prefill op split OFF then ON at 61/561/1,650 new tokens.
D=/tmp/p61
L=$D/logs
mkdir -p $L
export PYTHONUNBUFFERED=1 ENGINE_TP=2
OFF="ENGINE_PREFILL_BLAS_TUNE=0"
ON="ENGINE_PREFILL_BLAS_TUNE=1 ENGINE_PREFILL_BLAS_TABLE=$D/table.csv"
P=tools/tp_parity.py
S="timeout 1500 python -u tools/prefill_op_split.py all --devices 6,7 --news 61,561,1650 --reps 5"
G="RESULT|Traceback|^---|^===|prefill BLAS|capture failed"
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"

echo "=== phase 1 $(date +%T)"
( timeout 1500 python -u tools/blas_tune_probe.py record --devices 0,1 --dir $D ) 2>&1 | tee $L/record.log | grep --line-buffered -E "$G" | cut -c1-400 &
( env HIP_VISIBLE_DEVICES=2,3 $OFF timeout 1800 python -u $P --mode ref --out $D/ref_off.pt ) 2>&1 | tee $L/par23.log | grep --line-buffered -E "$G" | cut -c1-400 &
( env HIP_VISIBLE_DEVICES=4,5 $OFF timeout 1800 python -u $P --mode ref --eager-prefill --out $D/ref_off_eager.pt ) 2>&1 | tee $L/par45.log | grep --line-buffered -E "$G" | cut -c1-400 &
wait
grep -E "RESULT|prefill BLAS|prefill graphs|Traceback|Error" $L/record.log | cut -c1-600
head -c 1500 $D/untuned*.csv 2>/dev/null; echo

echo "=== phase 2 tune $(date +%T)"
timeout 2700 python -u tools/blas_tune_probe.py tune --dir $D --gpus 0,1,2,3,4,5,6,7 2>&1 | tee $L/tune.log | grep -vE "$F" | cut -c1-600

echo "=== phase 3 $(date +%T)"
( HIP_VISIBLE_DEVICES=0 timeout 1200 python -u tools/blas_tune_probe.py time --dir $D; \
  echo "--- spec == plain ON"; env $ON timeout 2400 python -u tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 \
) 2>&1 | tee $L/time01.log | grep --line-buffered -E "$G" | cut -c1-400 &
( E="HIP_VISIBLE_DEVICES=2,3"; \
  echo "--- ON against the OFF reference (teacher-forced top-1)"; \
  env $E $ON timeout 1800 python -u $P --mode check --ref $D/ref_off.pt; \
  env $E $ON timeout 1800 python -u $P --mode ref --out $D/ref_on.pt \
  && echo "--- ON: a second process against the first" \
  && env $E $ON timeout 1800 python -u $P --mode check --ref $D/ref_on.pt \
) 2>&1 | tee -a $L/par23.log | grep --line-buffered -E "$G" | cut -c1-400 &
( echo "--- ON eager-prefill against OFF eager-prefill (decode graphs bitwise)"; \
  env HIP_VISIBLE_DEVICES=4,5 $ON timeout 1800 python -u $P --mode check --eager-prefill --ref $D/ref_off_eager.pt \
) 2>&1 | tee -a $L/par45.log | grep --line-buffered -E "$G" | cut -c1-400 &
( for f in "$OFF" "$ON"; do echo "=== op split (graph) $f"; env $f $S; done ) 2>&1 | tee $L/split67.log | grep --line-buffered -E "$G" | cut -c1-400 &
wait
echo "=== done $(date +%T)"

echo "=== summary (all RESULT lines again)"
echo "=== per-shape default vs tuned, spec == plain"; grep -E "RESULT|Traceback|Error|^---" $L/time01.log | cut -c1-400 | tail -500
echo "=== top-1 ON vs OFF, cross-process ON"; grep -E "RESULT|Error|^---|failed in mode|prefill BLAS" $L/par23.log | cut -c1-600
echo "=== decode bitwise (eager prefill) ON vs OFF"; grep -E "RESULT|Error|^---|prefill BLAS" $L/par45.log | cut -c1-600
echo "=== op split"; grep -vE "$F" $L/split67.log | grep -vE '"kind": "(split|host_log)"' | cut -c1-260 | tail -260
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
echo "=== table"; sed 's/^/TABLE|/' $D/table.csv
true
