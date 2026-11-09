#!/bin/bash
# Plan 59 stage 1 on the merge with cead343 (plans 56, 47), one MI250 node (8 GCDs): the image's
# ROCm 7.0 stack against the seed's ROCm 6.4 stack (torch 2.9.1+rocm6.4, pytorch-triton-rocm
# 3.5.1, pip-installed under /tmp/t64 and put first on PYTHONPATH), same engine, same job.
#
#   bash tools/plan59_stage1.sh
#
# 0. install the 6.4 stack (nothing else runs meanwhile)
# 1. timing, one thing at a time: glue_split chain k=3 B=1,2 on GCD 0,1 in the order
#    7.0 / 6.4 / 7.0 / 6.4; prefill_op_split graph at 61/561/1650 on GCD 4,5, 7.0 then 6.4
# 2. served, whole node: serve_probe (j272's shape) default (7.0) vs PYTHONPATH=6.4, then 6.4 vs
#    7.0 (TPOT/TTFT p50/p90 per run, identical requests per pair)
# 3. in parallel: spec == plain (spec_step_bench, GCD 0,1) and chain_gate with IVF on and its
#    audit (GCD 2,3) on 7.0; unit tests one file at a time with 2 GCDs, 7.0 on GCD 4,5 and 6.4 on
#    GCD 6,7 (-x: the first failing test per file)
# 4. every file that fails on 7.0: its first failing test again with AMD_SERIALIZE_KERNEL=3
#    (the faulting launch raises where it is made), the long traceback; the same under 6.4
# 5. teacher-forced top-1: 6.4 reference, 7.0 check (GCD 4,5)
export PYTHONUNBUFFERED=1
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced|Profiler clears|_warn_once"
T64=/tmp/t64
ts() { echo "=== $1 $(date +%T)"; }
on64() { env PYTHONPATH=$T64 "$@"; }

ts "0 install the 6.4 stack"
timeout 2400 pip install -q --no-cache-dir --no-deps --target $T64 'torch==2.9.1+rocm6.4' 'pytorch-triton-rocm==3.5.1' \
  --index-url https://download.pytorch.org/whl/rocm6.4 > /tmp/t64.log 2>&1 && touch $T64/.ok
[ -f $T64/.ok ] || { echo "6.4 stack missing"; tail -5 /tmp/t64.log; }
python -c "import torch, triton; print('7.0', torch.__version__, torch.version.hip, triton.__version__)"
on64 python -c "import torch, triton; print('6.4', torch.__version__, torch.version.hip, triton.__version__, torch.__file__)"
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1

ts "1 timing"
G="timeout 1500 python -u tools/glue_split.py --devices 0,1 --configs chain:1:3,chain:2:3"
GF='RESULT (load_s|wall|step|split chain3 B=2 gcd=0|glue)|Traceback|Error'
for run in "7.0 1" "6.4 1" "7.0 2" "6.4 2"; do
  set -- $run
  echo "=== [$1 run $2] glue_split"
  if [ "$1" = 6.4 ]; then on64 $G 2>&1 | grep -E "$GF"; else $G 2>&1 | grep -E "$GF"; fi
done
P="timeout 1500 python -u tools/prefill_op_split.py all --devices 4,5 --news 61,561,1650 --reps 5"
PF='kind": "(load|host_log)"|^by name|^TOTAL|^bucket|Traceback|Error'
echo "=== [7.0] prefill_op_split graph"; $P 2>&1 | grep -E "$PF" | cut -c1-260
echo "=== [6.4] prefill_op_split graph"; on64 $P 2>&1 | grep -E "$PF" | cut -c1-260

ts "2 served (j272 shape), 7.0 vs 6.4 then 6.4 vs 7.0"
S="timeout 2400 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 --max-new 190 --corpus synthetic"
SF='"event": "(setting|agreement)"|Traceback|Error'
$S --settings "default,PYTHONPATH=$T64" 2>&1 | grep -E "$SF" | cut -c1-900
$S --settings "PYTHONPATH=$T64,default" --port 18432 2>&1 | grep -E "$SF" | cut -c1-900

ts "3 exactness, chain_gate (IVF on), unit tests per file"
( HIP_VISIBLE_DEVICES=0,1 timeout 1800 python -u tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-500 | sed -u "s/^/[spec 7.0] /" ) > /tmp/p59-spec.log 2>&1 &
( ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_AUDIT=1 HIP_VISIBLE_DEVICES=2,3 timeout 2400 python -u tools/chain_gate.py \
    --devices 2,3 --pairs 3 --iters 30 2>&1 | grep -E "RESULT|IVF|Traceback|Error" | cut -c1-500 | sed -u "s/^/[chain_gate 7.0 ivf] /" ) > /tmp/p59-gate.log 2>&1 &
tests() {  # $1 label, $2 devices, $3 PYTHONPATH or ""
  for f in tests/test_*.py; do
    out=$(env HIP_VISIBLE_DEVICES=$2 ${3:+PYTHONPATH=$3} timeout 900 python -m pytest -q -x -p no:cacheprovider --tb=line "$f" 2>&1)
    rc=$?
    sum=$(echo "$out" | grep -E "passed|failed|error|no tests ran" | tail -1)
    first=$(echo "$out" | grep -E "^(FAILED|ERROR) |^E  |Error:" | head -2 | cut -c1-300 | tr '\n' ' ')
    echo "[$1] $f rc=$rc :: $sum :: $first"
    [ $rc -ne 0 ] && [ $rc -ne 5 ] && echo "$f" >> /tmp/p59-fail-$1
  done
}
rm -f /tmp/p59-fail-*
tests 7.0 4,5 "" > /tmp/p59-t70.log 2>&1 &
[ -f $T64/.ok ] && tests 6.4 6,7 $T64 > /tmp/p59-t64.log 2>&1 &
wait
cat /tmp/p59-spec.log /tmp/p59-gate.log /tmp/p59-t70.log /tmp/p59-t64.log 2>/dev/null

ts "4 failing files under AMD_SERIALIZE_KERNEL=3"
for f in $(cat /tmp/p59-fail-7.0 2>/dev/null | head -4); do
  for st in 7.0 6.4; do
    echo "--- [$st] $f"
    env HIP_VISIBLE_DEVICES=4,5 AMD_SERIALIZE_KERNEL=3 $([ $st = 6.4 ] && echo PYTHONPATH=$T64) timeout 900 \
      python -m pytest -q -x -p no:cacheprovider --tb=long "$f" 2>&1 \
      | grep -vE "^\s*$" | grep -E "^(FAILED|ERROR)|^E |\.py:[0-9]+:|_kernel|\[grid|launch|passed|failed|hip|HIP" | head -40 | cut -c1-300
  done
done

ts "5 teacher-forced top-1: 6.4 reference, 7.0 check"
if [ -f $T64/.ok ]; then
  E="HIP_VISIBLE_DEVICES=4,5"
  env $E PYTHONPATH=$T64 timeout 1500 python -u tools/tp_parity.py --mode ref --out /tmp/p59_ref64.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-400
  env $E timeout 1500 python -u tools/tp_parity.py --mode check --ref /tmp/p59_ref64.pt 2>&1 | grep -E "RESULT|Traceback|Error" | cut -c1-400
fi
ts "done"
true
