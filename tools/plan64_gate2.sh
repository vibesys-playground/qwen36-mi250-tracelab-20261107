#!/bin/bash
# Plan 64 gate 2 (after j310: ORDER 1 was bitwise vs add_rms on each rank; the bench gave each
# rank its own norm weight, so its "ranks_differ" was an artefact, fixed). The decisive timing:
# glue_split chain B=2 k=3 and B=1 k=7 on GCD 0,1, OFF / ON / OFF / ON with
# ON = ENGINE_TOPK_SEL=1 ENGINE_AR_NORM=1 ENGINE_AR_NORM_ORDER=1 (nothing else runs), then the
# AR part of the bench (two GPUs, same weights) and tests/test_fused_glue.py with ON.
export PYTHONUNBUFFERED=1
ts() { echo "=== $1 $(date +%T)"; }
OFF="ENGINE_TOPK_SEL=0 ENGINE_AR_NORM=0"
ON="ENGINE_TOPK_SEL=1 ENGINE_AR_NORM=1 ENGINE_AR_NORM_ORDER=1"
GF='RESULT (wall|step|split chain(3|7) B=(1|2) gcd=(0|1) item=(1_add_rms|2_topk|allreduce))|Traceback|Error'
ts "1 glue_split OFF/ON/OFF/ON"
for s in OFF ON OFF ON; do
  echo "=== [$s] glue_split"
  env ${!s} timeout 900 python -u tools/glue_split.py --devices 0,1 --configs chain:2:3,chain:1:7 2>&1 | grep -E "$GF"
done
ts "2 AR bench (two GPUs) and test_fused_glue with ON"
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1
HIP_VISIBLE_DEVICES=0,1 timeout 900 python -u tools/plan64_bench.py --devices 0,1 --skip-router --iters 20 --stress 1000 2>&1 \
  | grep -E "RESULT|CHOICE|Traceback|Error" | cut -c1-600
env $ON HIP_VISIBLE_DEVICES=2,3 timeout 900 python -m pytest -q -p no:cacheprovider --tb=short tests/test_fused_glue.py 2>&1 | tail -15
ts "done"
true
