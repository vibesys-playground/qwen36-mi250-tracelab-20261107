#!/bin/sh
# Plan 56 after merging head 50f8f10 (plan 50, HIP GDN decode): GPU unit tests one file at a time
# (fault handler on) on GCD 6/6,7, while spec == plain (defaults) runs on GCD 0,1 and Triton
# compiles after /health under a served load are counted on GCD 2,3.
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
export PYTHONUNBUFFERED=1
F="_pytest|pluggy|Extension modules|^\s*$|amdgpu.ids"
run() {
  devs=$1; shift
  echo "=== HIP_VISIBLE_DEVICES=$devs $*"
  HIP_VISIBLE_DEVICES=$devs timeout 1500 python -X faulthandler -m pytest -q -rf -p no:cacheprovider "$@" 2>&1 \
    | grep -vE "$F" | cut -c1-400 | tail -30
}
( timeout 2400 python -u tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 2>&1 \
    | grep -E "RESULT|Traceback|Error" > /tmp/p56m_spec.log ) &
( HIP_VISIBLE_DEVICES=2,3 timeout 3000 python tools/live_compile_probe.py 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-900 > /tmp/p56m_live.log ) &
run 6 tests/test_prefill_kernels.py tests/test_moe_prefill.py
run 6 tests/test_gdn_hip.py
run 6 tests/test_fused_decode.py tests/test_fused_verify.py tests/test_gdn_bank_fused.py
run 6,7 tests/test_chain_glue.py
wait
echo "=== spec == plain (defaults)"; cat /tmp/p56m_spec.log
echo "=== live compiles after /health (defaults)"; cat /tmp/p56m_live.log
true
