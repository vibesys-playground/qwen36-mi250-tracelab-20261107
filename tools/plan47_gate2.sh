#!/bin/bash
# Plan 47 stage 1, gate 2: unit tests after the tiny-checkpoint guard, the chained step time
# split (IVF off; on; on with no fallback; on with a sorted shard copy) at 1 and 2 slots with
# each run's fallback rate, the acceptance runs with the fallback rate, then a served A/B 1/0
# on this commit. One MI250 node, 8 GCDs.
L=/tmp/p47b
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT (step|accept|parity|fatal|error)|Traceback|IVF drafts'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
run() {  # name devices env...
  local name=$1 dev=$2; shift 2
  echo "=== [$name] $* $(date +%T)"
  env "$@" HIP_VISIBLE_DEVICES=$dev timeout 1500 python tools/chain_gate.py --devices $dev --skip-accept --iters 50 2>&1 \
    | tee -a $L/$name.log | tag $name
}
( run off 0,1 ENGINE_DRAFT_IVF=0; run off2 0,1 ENGINE_DRAFT_IVF=0 ) &
( run on 2,3 ENGINE_DRAFT_IVF=1; run on2 2,3 ENGINE_DRAFT_IVF=1 ) &
( run nofb 4,5 ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_GAP=-1e9; run allfb 4,5 ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_GAP=1e9 ) &
(
  echo "=== [tests] pytest $(date +%T)"
  HIP_VISIBLE_DEVICES=6 timeout 1800 python -m pytest -q -rf tests/test_draft_ivf.py tests/test_runahead.py \
    tests/test_spec_chain.py tests/test_fused_gemv.py > $L/tests.log 2>&1
  echo "=== [tests] exit $?"
  run sorted 6,7 ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_SORTED=1
) &
wait
echo "=== unit tests"; tail -8 $L/tests.log
echo "=== [acc] chain_gate IVF=1 with fallback rates $(date +%T)"
ENGINE_DRAFT_IVF=1 HIP_VISIBLE_DEVICES=0,1 timeout 1500 python tools/chain_gate.py --devices 0,1 --pairs 3 2>&1 \
  | tee -a $L/acc.log | tag acc
echo "=== [ab] served A/B IVF 1/0 $(date +%T)"
timeout 2400 python -u tools/topk_epi_ab.py --settings ENGINE_DRAFT_IVF=1,ENGINE_DRAFT_IVF=0 2>&1 \
  | grep -E '^ *[0-9] |RESULT|^===|Traceback|Error'
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
