#!/bin/sh
# Plan 60 step 0: the N-streaming down kernel swept against today's (GCD 0,1, real TP2 shard
# and routing), with the GPU unit tests on GCD 2 in parallel.
#   sh tools/plan60_step0.sh
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p60-logs
mkdir -p $L
( HIP_VISIBLE_DEVICES=2 timeout 1500 python -X faulthandler -m pytest -q -rf -p no:cacheprovider \
    tests/test_moe_prefill.py 2>&1 | tail -40 ) > $L/tests.log 2>&1 &
ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1 timeout 3000 python -u tools/moe_down_stream_sweep.py 2>&1 \
  | grep -vE "Loading weights|amdgpu.ids|it/s\]|^engine:"
wait
echo "=== unit tests (GCD 2)"; cat $L/tests.log
true
