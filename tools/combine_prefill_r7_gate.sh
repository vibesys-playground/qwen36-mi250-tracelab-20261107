#!/bin/sh
# Plans 54 + 60 + 61 on the ROCm 7.0 image (head afb6be0): regenerate plan 61's tuned prefill
# BLAS table for this stack, then gate the three together. One MI250 node, 8 GCDs.
#
#   sh tools/combine_prefill_r7_gate.sh
#
# OFF = ENGINE_GDN_WY=mono:8 ENGINE_MOE_DOWN_STREAM=0 ENGINE_PREFILL_BLAS_TUNE=0 (the head's
# kernels and BLAS); ON = the GDN and MoE defaults and ENGINE_PREFILL_BLAS_TABLE=$D/table.csv
# (the table this job tunes; it is printed at the end, TABLE| lines, to ship in the repository).
# Phase 1 (parallel): GCD 0,1 record the prefill graphs' GEMMs; GCD 2,3 teacher-forced
#   reference OFF; GCD 6 / 7 GPU unit tests (gdn_chunk / moe_prefill) on Triton 3.6.
# Phase 2: offline tuning of the recorded shapes on all 8 GCDs (nothing else running).
# Phase 3 (parallel): GCD 0,1 captured prefill op split at 61/561/1,650/2,300 new tokens, OFF then
#   ON; GCD 2,3 top-1 ON vs OFF (>= 98.5%); GCD 4,5 ON cross-process, then ON repeat in one
#   process; GCD 6 per-shape default vs tuned and the blas_tune tests on the new table, then
#   GCD 6,7 spec == plain (ON), then Triton compiles after /health (ON).
D=/tmp/cp7
L=$D/logs
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
export PYTHONUNBUFFERED=1
OFF="ENGINE_GDN_WY=mono:8 ENGINE_MOE_DOWN_STREAM=0 ENGINE_PREFILL_BLAS_TUNE=0"
ON="-u ENGINE_GDN_WY -u ENGINE_MOE_DOWN_STREAM ENGINE_PREFILL_BLAS_TUNE=1 ENGINE_PREFILL_BLAS_TABLE=$D/table.csv"
P=tools/tp_parity.py
S="timeout 1800 python -u tools/prefill_op_split.py all --devices 0,1 --news 61,561,1650,2300 --reps 5"
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"
PT="python -X faulthandler -m pytest -q -rf -p no:cacheprovider"

echo "=== stack"; python -c "import torch, triton; print('torch', torch.__version__, 'hip', torch.version.hip, 'triton', triton.__version__)"
echo "=== phase 1 $(date +%T)"
( ENGINE_TP=2 timeout 1500 python -u tools/blas_tune_probe.py record --devices 0,1 --dir $D ) > $L/record.log 2>&1 &
( env ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 $OFF timeout 1800 python -u $P --mode ref --out $D/ref_off.pt ) > $L/top1.log 2>&1 &
( HIP_VISIBLE_DEVICES=6 timeout 1800 $PT tests/test_gdn_chunk.py 2>&1 | tail -15 ) > $L/tests_gdn.log 2>&1 &
( HIP_VISIBLE_DEVICES=7 timeout 1800 $PT tests/test_moe_prefill.py 2>&1 | tail -15 ) > $L/tests_moe.log 2>&1 &
wait
grep -E "RESULT|prefill BLAS|prefill graphs|Traceback|Error" $L/record.log | cut -c1-600

echo "=== phase 2 tune $(date +%T)"
timeout 2700 python -u tools/blas_tune_probe.py tune --dir $D --gpus 0,1,2,3,4,5,6,7 2>&1 | tee $L/tune.log | grep -vE "$F" | cut -c1-600
if [ ! -s $D/table.csv ]; then echo "NO TABLE: tuning failed"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done; exit 1; fi

echo "=== phase 3 $(date +%T)"
( echo "=== op split (graph) OFF"; env ENGINE_TP=2 $OFF $S; \
  echo "=== op split (graph) ON"; env $ON ENGINE_TP=2 $S ) > $L/split.log 2>&1 &
( echo "--- ON against the OFF reference (teacher-forced top-1)"; \
  env $ON ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 timeout 1800 python -u $P --mode check --ref $D/ref_off.pt ) >> $L/top1.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=4,5"; \
  env $ON $E timeout 1800 python -u $P --mode ref --out $D/ref_on.pt \
  && echo "--- ON: a second process against the first" \
  && env $ON $E timeout 1800 python -u $P --mode check --ref $D/ref_on.pt; \
  echo "--- ON repeat in one process"; env $ON $E timeout 1800 python -u $P --mode repeat --out $D/rep.pt \
) > $L/repeat.log 2>&1 &
( HIP_VISIBLE_DEVICES=6 timeout 1200 python -u tools/blas_tune_probe.py time --dir $D; \
  echo "--- pytest tests/test_blas_tune.py (new table)"; \
  env HIP_VISIBLE_DEVICES=6 ENGINE_PREFILL_BLAS_TABLE=$D/table.csv timeout 900 $PT tests/test_blas_tune.py 2>&1 | tail -15; \
  echo "--- spec == plain (ON)"; env $ON timeout 2400 python -u tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128; \
  echo "--- live compiles (ON)"; env $ON HIP_VISIBLE_DEVICES=6,7 timeout 3000 python -u tools/live_compile_probe.py \
) > $L/misc.log 2>&1 &
wait
echo "=== done $(date +%T)"

echo "=== unit tests: test_gdn_chunk.py"; grep -vE "$F" $L/tests_gdn.log
echo "=== unit tests: test_moe_prefill.py"; grep -vE "$F" $L/tests_moe.log
echo "=== teacher-forced top-1 ON vs OFF"; grep -E "RESULT|Error|^---|failed in mode|prefill BLAS" $L/top1.log | cut -c1-600
echo "=== cross-process, repeat (ON)"; grep -E "RESULT|Error|^---|failed in mode|prefill BLAS" $L/repeat.log | cut -c1-600
echo "=== per-shape, blas_tune tests, spec == plain, live compiles"; grep -E "RESULT|Error|^---|passed|failed|prefill BLAS" $L/misc.log | grep -v "RESULT shape" | cut -c1-1500 | tail -80
echo "--- per shape where tuned is slower"; grep "RESULT shape" $L/misc.log | grep "saving_pct=-" | cut -c1-260
echo "=== op split (graph)"; grep -E "^=== op split|kill|prefill BLAS" $L/split.log | cut -c1-500
echo "--- per pass (ms per GCD)"
grep -E "^=== op split|\"kind\": \"split\"" $L/split.log | python -c "
import json, sys
for line in sys.stdin:
    if line.startswith('==='):
        print(line.strip()); continue
    d = json.loads(line.split('RESULT ', 1)[1])
    print(' new_tokens', d.get('new_tokens'), 'wall_ms_median', d.get('wall_ms_median'), 'busy_any_ms', d.get('busy_any_ms'))
    for b, names in d.get('bucket_top_names', {}).items():
        if any(s in b.lower() for s in ('gdn chunk', 'moe exp', 'moe_kernel', 'gemm')):
            print('   ', b[:48], [(n[:30], round(v / 2, 3)) for n, v in names][:6])
"
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
echo "=== table"; sed 's/^/TABLE|/' $D/table.csv
true
