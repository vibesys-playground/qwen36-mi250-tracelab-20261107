#!/bin/bash
# Plan 59 stage 0, second job: the image's ROCm 7.0 stack (torch 2.10.0+rocm7.0, triton-rocm
# 3.6.0) with the peer-access compat fix, and the ROCm 7.1 stack (torch 2.13.0+rocm7.1,
# triton-rocm 3.7.1) pip-installed under /tmp/t71 and used through PYTHONPATH, engine unchanged.
#
#   bash tools/plan59_stage0b.sh
#
# 1. env under 7.0 (stack_probe env: arch list, 8 GCDs, driver, Triton + hipRTC on gfx90a)
# 2. cold /health and a served replay-like run under 7.0: j272's serve_probe command, empty
#    Triton and hipRTC caches
# 3. j273's command under 7.0 (glue_split chain k=3 B=1,2 on GCD 0,1; prefill_op_split graph and
#    eager at 61/561/1650 on GCD 4,5); the 7.1 and 6.4 stacks install in the background meanwhile
# 4. under 7.1: stack_probe env, j273's glue_split and prefill_op_split graph, stack_probe bench
# 5. exactness under 7.0: spec == plain (GCD 0,1), teacher-forced top-1 against a 6.4 reference
#    made in this job and in-process repeat (GCD 4,5), unit tests (GCD 6,7); then 7.1 against
#    the same 6.4 reference
export PYTHONUNBUFFERED=1
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"
T64=/tmp/t64 T71=/tmp/t71
ts() { echo "=== $1 $(date +%T)"; }
j273() {  # $1: label; env (PYTHONPATH) chooses the stack
  echo "=== [$1] glue_split"
  timeout 1500 python -u tools/glue_split.py --devices 0,1 --configs chain:1:3,chain:2:3 2>&1 | grep -v "^engine:" | grep -vE "$F"
  echo "=== [$1] prefill_op_split graph"
  timeout 1500 python -u tools/prefill_op_split.py all --devices 4,5 --news 61,561,1650 --reps 5 2>&1 | grep -vE "$F"
}

ts "1 env (7.0)"
timeout 600 python -u tools/stack_probe.py env 2>&1 | grep -vE "$F"

ts "2 cold /health + served (j272 command, 7.0), empty Triton/hipRTC caches"
TRITON_CACHE_DIR=/tmp/triton-cold-$$ ENGINE_HIPMEGA_CACHE=/tmp/hipmega-cold-$$ \
timeout 2400 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 \
  --grow 60 --max-new 190 --corpus synthetic --settings ENGINE_LOOP_TIMERS=1 2>&1 | grep -v "^engine:" | grep -vE "$F" | cut -c1-3000 | tail -40
echo "serve_probe exit ${PIPESTATUS[0]}"

ts "7.1 and 6.4 stacks install in the background"
(timeout 2400 pip install -q --no-cache-dir --no-deps --target $T71 'torch==2.13.0+rocm7.1' 'triton-rocm==3.7.1' \
   --index-url https://download.pytorch.org/whl/rocm7.1 > /tmp/t71.log 2>&1 && touch $T71/.ok) &
(timeout 2400 pip install -q --no-cache-dir --no-deps --target $T64 'torch==2.9.1+rocm6.4' 'pytorch-triton-rocm==3.5.1' \
   --index-url https://download.pytorch.org/whl/rocm6.4 > /tmp/t64.log 2>&1 && touch $T64/.ok) &

ts "3 j273 command (7.0)"
j273 "7.0"
echo "=== [7.0] prefill_op_split eager"
timeout 1500 python -u tools/prefill_op_split.py all --devices 4,5 --news 61,561,1650 --reps 5 --eager 2>&1 | grep -vE "$F"

ts "wait for the installs"
wait
for d in $T71 $T64; do
  if [ -f $d/.ok ]; then PYTHONPATH=$d python -c "import torch, triton; print('$d', torch.__version__, torch.version.hip, triton.__version__, torch.__file__)" 2>&1 | grep -vE "$F"
  else echo "$d missing"; tail -5 /tmp/$(basename $d).log; fi
done

ts "4 ROCm 7.1 stack"
if [ -f $T71/.ok ]; then
  export PYTHONPATH=$T71
  timeout 600 python -u tools/stack_probe.py env 2>&1 | grep -vE "$F"
  j273 "7.1"
  HIP_VISIBLE_DEVICES=0 timeout 900 python -u tools/stack_probe.py bench 2>&1 | grep -vE "$F"
  echo "stack_probe 7.1 exit ${PIPESTATUS[0]}"
  unset PYTHONPATH
fi

ts "5 exactness"
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1
( echo "--- [7.0] spec == plain (spec_step_bench, GCD 0,1)"
  HIP_VISIBLE_DEVICES=0,1 timeout 1800 python -u tools/spec_step_bench.py --devices 0,1 --iters 20 --max-new 128 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-600 ) > /tmp/p59-spec.log 2>&1 &
( E="HIP_VISIBLE_DEVICES=4,5"
  if [ -f $T64/.ok ]; then
    echo "--- [6.4] tp_parity ref"
    env $E PYTHONPATH=$T64 timeout 1500 python -u tools/tp_parity.py --mode ref --out /tmp/p59_ref64.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
    REF=/tmp/p59_ref64.pt
  else
    echo "--- no 6.4 reference: [7.0] tp_parity ref (cross-process check only)"
    env $E timeout 1500 python -u tools/tp_parity.py --mode ref --out /tmp/p59_ref70.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
    REF=/tmp/p59_ref70.pt
  fi
  echo "--- [7.0] tp_parity check against $REF"
  env $E timeout 1500 python -u tools/tp_parity.py --mode check --ref $REF 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
  echo "--- [7.0] tp_parity repeat (in process)"
  env $E timeout 1500 python -u tools/tp_parity.py --mode repeat 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
  if [ -f $T71/.ok ]; then
    echo "--- [7.1] tp_parity check against $REF"
    env $E PYTHONPATH=$T71 timeout 1500 python -u tools/tp_parity.py --mode check --ref $REF 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-600
  fi
) > /tmp/p59-tf.log 2>&1 &
( echo "--- [7.0] unit tests (GCD 6,7)"
  HIP_VISIBLE_DEVICES=6,7 timeout 1500 python -m pytest -q -rfE tests/test_gdn_hip.py tests/test_spec_chain.py \
    tests/test_fused_verify.py tests/test_gdn_bank_fused.py tests/test_fused_decode.py tests/test_runahead.py \
    tests/test_chain_glue.py tests/test_tp.py 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | cut -c1-400 ) > /tmp/p59-tests.log 2>&1 &
wait
cat /tmp/p59-spec.log /tmp/p59-tf.log /tmp/p59-tests.log
ts "done"
true
