#!/bin/sh
# Plan 50 after merging head 5a7e444 (plan 51): GPU unit tests one file at a time (fault handler
# on, so a crash names its test) while spec == plain runs on another pair (ENGINE_GDN_HIP=1).
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
export PYTHONUNBUFFERED=1
F="_pytest|pluggy|Extension modules|^\s*$|amdgpu.ids"
run() {
  devs=$1; shift
  echo "=== HIP_VISIBLE_DEVICES=$devs $*"
  HIP_VISIBLE_DEVICES=$devs timeout 1500 python -X faulthandler -m pytest -v -rf -p no:cacheprovider "$@" 2>&1 \
    | grep -vE "$F" | cut -c1-400 | tail -60
}
( ENGINE_TP=2 ENGINE_MTP=1 ENGINE_GDN_HIP=1 HIP_VISIBLE_DEVICES=0,1 timeout 1500 python tools/spec_step_bench.py \
    --devices 0,1 --iters 20 --max-new 128 2>&1 | grep -E "RESULT|gdn_hip|Traceback|Error" > /tmp/p50_spec.log ) &
run 6 tests/test_gdn_hip.py
run 6,7 tests/test_chain_glue.py
run 6 tests/test_spec_chain.py
run 6 tests/test_runahead.py
run 6 tests/test_fused_verify.py tests/test_gdn_bank_fused.py tests/test_fused_decode.py
wait
echo "=== spec == plain (ENGINE_GDN_HIP=1)"
cat /tmp/p50_spec.log
true
