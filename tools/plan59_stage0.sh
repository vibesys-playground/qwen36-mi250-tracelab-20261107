#!/bin/bash
# Plan 59 stage 0a on one MI250 node (8 GCDs): the image's ROCm 7.0 stack (torch 2.10.0+rocm7.0,
# triton-rocm 3.6.0) against the seed's ROCm 6.4 stack, engine unchanged.
#
#   bash tools/plan59_stage0.sh
#
# 1. env: versions, arch list, all 8 GCDs, driver, Triton + hipRTC smoke (tools/stack_probe.py env)
# 2. cold /health and a served replay-like run: j272's serve_probe command, Triton and hipRTC
#    caches empty
# 3. j273's command exactly (glue_split chain k=3 B=1,2 on GCD 0,1; prefill_op_split graph and
#    eager at 61/561/1650 on GCD 4,5), nothing else on the GPUs
# 4. stack_probe bench (streaming, small-kernel latency/launch, prefill GEMMs) on GCD 0
# 5. exactness: spec == plain (spec_step_bench, GCD 0,1), in-process repeat and teacher-forced
#    top-1 against a ROCm 6.4 reference made in this job (GCD 4,5), unit tests (GCD 6)
# 6. stack_probe bench on GCD 0 under the 6.4 stack, for a same-job A/B of (a)-(c)
# The 6.4 stack (torch 2.9.1+rocm6.4, pytorch-triton-rocm 3.5.1) is pip-installed under /tmp/t64
# in the background from step 3 on, and used through PYTHONPATH; steps that need it are skipped
# (and say so) when it is missing.
export PYTHONUNBUFFERED=1
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"
T64=/tmp/t64
ts() { echo "=== $1 $(date +%T)"; }

ts "1 env"
timeout 600 python -u tools/stack_probe.py env 2>&1 | grep -vE "$F"
echo "env exit ${PIPESTATUS[0]}"
df -h /tmp 2>&1 | tail -1
cat /sys/module/amdgpu/version 2>/dev/null | sed 's/^/amdgpu module version /'
pip list 2>/dev/null | grep -iE "^(torch|triton|fla|transformers|accelerate) " | sed 's/^/pip /'

ts "2 cold /health + served (j272 command), empty Triton/hipRTC caches"
export TRITON_CACHE_DIR=/tmp/triton-cold-$$ ENGINE_HIPMEGA_CACHE=/tmp/hipmega-cold-$$
timeout 2400 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 \
  --grow 60 --max-new 190 --corpus synthetic --settings ENGINE_LOOP_TIMERS=1 2>&1 | grep -v "^engine:" | grep -vE "$F" | cut -c1-1500
echo "serve_probe exit ${PIPESTATUS[0]}"
unset TRITON_CACHE_DIR ENGINE_HIPMEGA_CACHE

ts "6.4 stack install started in the background"
FREE=$(df -Pk /tmp | awk 'NR==2 {print $4}')
if [ "${FREE:-0}" -gt 41943040 ]; then
  (timeout 2400 pip install -q --no-cache-dir --no-deps --target $T64 'torch==2.9.1+rocm6.4' \
     'pytorch-triton-rocm==3.5.1' --index-url https://download.pytorch.org/whl/rocm6.4 > /tmp/t64.log 2>&1 \
   && touch $T64/.ok) &
else
  echo "skip 6.4 install: /tmp free ${FREE} KiB < 40 GiB"
fi

ts "3 j273 command"
echo "=== glue_split"; timeout 1500 python -u tools/glue_split.py --devices 0,1 --configs chain:1:3,chain:2:3 2>&1 | grep -v "^engine:" | grep -vE "$F"
echo "=== prefill_op_split graph"; timeout 1500 python -u tools/prefill_op_split.py all --devices 4,5 --news 61,561,1650 --reps 5 2>&1 | grep -vE "$F"
echo "=== prefill_op_split eager"; timeout 1500 python -u tools/prefill_op_split.py all --devices 4,5 --news 61,561,1650 --reps 5 --eager 2>&1 | grep -vE "$F"

ts "4 stack_probe bench (7.0)"
HIP_VISIBLE_DEVICES=0 timeout 900 python -u tools/stack_probe.py bench 2>&1 | grep -vE "$F"
echo "stack_probe exit ${PIPESTATUS[0]}"

ts "wait for the 6.4 install"
wait
if [ -f $T64/.ok ]; then
  PYTHONPATH=$T64 python -c "import torch, triton; print('t64', torch.__version__, torch.version.hip, triton.__version__, torch.__file__)" 2>&1 | grep -vE "$F"
else
  echo "6.4 stack missing"; tail -5 /tmp/t64.log 2>/dev/null
fi

ts "5 exactness"
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1
( echo "--- spec == plain (spec_step_bench, GCD 0,1)"
  HIP_VISIBLE_DEVICES=0,1 timeout 1800 python -u tools/spec_step_bench.py --devices 0,1 --iters 20 --max-new 128 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-600 ) > /tmp/p59-spec.log 2>&1 &
( E="HIP_VISIBLE_DEVICES=4,5"
  if [ -f $T64/.ok ]; then
    echo "--- tp_parity ref under 6.4"
    env $E PYTHONPATH=$T64 timeout 1500 python -u tools/tp_parity.py --mode ref --out /tmp/p59_ref64.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
    echo "--- tp_parity check under 7.0 against the 6.4 reference"
    env $E timeout 1500 python -u tools/tp_parity.py --mode check --ref /tmp/p59_ref64.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
  else
    echo "--- no 6.4 reference: tp_parity ref under 7.0, check under 7.0 (cross-process)"
    env $E timeout 1500 python -u tools/tp_parity.py --mode ref --out /tmp/p59_ref70.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
    env $E timeout 1500 python -u tools/tp_parity.py --mode check --ref /tmp/p59_ref70.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
  fi
  echo "--- tp_parity repeat (in process, 7.0)"
  env $E timeout 1500 python -u tools/tp_parity.py --mode repeat 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
) > /tmp/p59-tf.log 2>&1 &
( echo "--- unit tests (GCD 6)"
  HIP_VISIBLE_DEVICES=6 timeout 1500 python -m pytest -q -rf tests/test_gdn_hip.py tests/test_spec_chain.py \
    tests/test_fused_verify.py tests/test_gdn_bank_fused.py tests/test_fused_decode.py tests/test_runahead.py \
    tests/test_chain_glue.py 2>&1 | tail -25 ) > /tmp/p59-tests.log 2>&1 &
wait
cat /tmp/p59-spec.log /tmp/p59-tf.log /tmp/p59-tests.log

ts "6 stack_probe bench (6.4, same job)"
if [ -f $T64/.ok ]; then
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=$T64 timeout 900 python -u tools/stack_probe.py bench 2>&1 | grep -vE "$F"
  echo "--- stack_probe env under 6.4"
  PYTHONPATH=$T64 timeout 600 python -u tools/stack_probe.py env 2>&1 | grep -E "versions|env_verdict|hiprtc"
else
  echo "6.4 stack missing: no same-job A/B"
fi
ts "done"
true
