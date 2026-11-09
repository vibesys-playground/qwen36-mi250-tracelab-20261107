#!/bin/bash
# Plan 64 gate, one MI250 node (8 GCDs): router top-k selection (ENGINE_TOPK_SEL) and the
# all-reduce + RMSNorm tail (ENGINE_AR_NORM), both default off in this commit.
#
#   bash tools/plan64_gate.sh
#
# 1. tools/plan64_bench.py on GCD 0,1 alone: bitwise SEL 1/2 vs 0 and ORDER 0-3, per-call
#    times, AR stress. Picks ON = the fastest bitwise SEL + AR_NORM with the bitwise ORDER.
# 2. glue_split chain k=3 B=1,2 on GCD 0,1, OFF / ON / OFF / ON (nothing else runs).
# 3. in parallel with ON: spec_step_bench (spec == plain, GCD 0,1), chain_gate (GCD 2,3),
#    tp_parity OFF reference then ON check (teacher-forced top-1, GCD 4,5), unit tests (GCD 6,7).
# 4. served A/B OFF / ON / OFF / ON on the whole node: identical requests per pair, compiles
#    after /health, tok/s TTFT TPOT.
export PYTHONUNBUFFERED=1
L=/tmp/p64
mkdir -p $L
ts() { echo "=== $1 $(date +%T)"; }
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1

ts "1 bench"
HIP_VISIBLE_DEVICES=0,1 timeout 1200 python -u tools/plan64_bench.py --devices 0,1 2>&1 | tee $L/bench.log \
  | grep -E "RESULT|CHOICE|Traceback|Error" | cut -c1-700
SEL=$(grep -oP 'CHOICE SEL=\K-?[0-9]+' $L/bench.log | tail -1); SEL=${SEL:-0}
ORD=$(grep -oP 'CHOICE SEL=-?[0-9]+ ORDER=\K-?[0-9]+' $L/bench.log | tail -1); ORD=${ORD:--1}
OFF="ENGINE_TOPK_SEL=0 ENGINE_AR_NORM=0"
if [ "$ORD" -ge 0 ]; then ON="ENGINE_TOPK_SEL=$SEL ENGINE_AR_NORM=1 ENGINE_AR_NORM_ORDER=$ORD"; else ON="ENGINE_TOPK_SEL=$SEL ENGINE_AR_NORM=0"; fi
echo "=== ON = $ON"
[ "$SEL" = 0 ] && [ "$ORD" -lt 0 ] && { echo "=== nothing bitwise and faster: stop"; exit 0; }

ts "2 glue_split OFF/ON/OFF/ON"
GF='RESULT (wall|step|split chain3 B=(1|2) gcd=(0|1) item=(1_add_rms|2_topk|allreduce))|Traceback|Error'
for s in OFF ON OFF ON; do
  echo "=== [$s] glue_split"
  env ${!s} timeout 900 python -u tools/glue_split.py --devices 0,1 --configs chain:1:3,chain:2:3 2>&1 | grep -E "$GF"
done

ts "3 exactness with ON"
( env $ON HIP_VISIBLE_DEVICES=0,1 timeout 1500 python -u tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 128 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-500 | sed -u "s/^/[spec] /" ) > $L/spec.log 2>&1 &
( env $ON HIP_VISIBLE_DEVICES=2,3 timeout 1500 python -u tools/chain_gate.py --devices 2,3 --pairs 3 --iters 30 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-500 | sed -u "s/^/[chain_gate] /" ) > $L/gate.log 2>&1 &
( env $OFF HIP_VISIBLE_DEVICES=4,5 timeout 1500 python -u tools/tp_parity.py --mode ref --out /tmp/p64_ref.pt 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-400 | sed -u "s/^/[parity ref OFF] /"
  env $ON HIP_VISIBLE_DEVICES=4,5 timeout 1500 python -u tools/tp_parity.py --mode check --ref /tmp/p64_ref.pt 2>&1 \
    | grep -E "RESULT|Traceback|Error" | cut -c1-400 | sed -u "s/^/[parity ON] /" ) > $L/parity.log 2>&1 &
( for f in tests/test_fused_glue.py tests/test_tp.py tests/test_fused_verify.py tests/test_chain_glue.py \
           tests/test_spec_chain.py tests/test_fused_decode.py tests/test_mtp.py tests/test_fused_moe.py; do
    o=$(env $ON HIP_VISIBLE_DEVICES=6,7 timeout 900 python -m pytest -q -x -p no:cacheprovider --tb=short "$f" 2>&1)
    echo "[tests ON] $f rc=$? :: $(echo "$o" | grep -E "passed|failed|error|no tests ran" | tail -1) :: $(echo "$o" | grep -E "^(FAILED|ERROR) |^E  " | head -4 | cut -c1-300 | tr '\n' ' ')"
  done ) > $L/tests.log 2>&1 &
wait
cat $L/spec.log $L/gate.log $L/parity.log $L/tests.log

ts "4 served A/B OFF/ON/OFF/ON"
SOFF=$(echo $OFF | tr ' ' '+'); SON=$(echo $ON | tr ' ' '+')
timeout 2400 python -u tools/topk_epi_ab.py --grow 60 --settings "$SOFF,$SON,$SOFF,$SON" 2>&1 \
  | grep -E '^RESULT|^ ?[0-9] |=== served|"event": "error"|Traceback' | cut -c1-600
ts "done"
true
