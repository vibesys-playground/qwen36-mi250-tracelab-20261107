#!/bin/bash
# Plan 71 stage 0 (tools only; the fast IVF path is off unless ENGINE_DRAFT_IVF_FAST=1). One MI250
# node, 8 GCDs, four groups in parallel, each section under its own timeout:
#   GCD 0,1: tools/link_bench.py: graph-of-N cost of one chained draft link at 1/2 rows (no
#            fallback / all fallback / IVF off; today and fast IVF path), per kernel of the head
#            layer and the IVF draft, the head layer at 4/8/10 rows, and the fast search's
#            bitwise equality with today's on random and real rows.
#   GCD 2,3: chain_gate step table (k 1,3,4,5,7 x 1,2 slots, 200 iters, corpus drafts), today's
#            path: no fallback (j323's nofb setting), then the served gap.
#   GCD 4,5: the same with ENGINE_DRAFT_IVF_FAST=1.
#   GCD 6,7: the same at all-fallback, today then fast; then the IVF unit tests (one GCD).
export ENGINE_MTP=1 PYTHONUNBUFFERED=1
F='^=== |RESULT|Traceback|Error|error|passed|failed|IVF drafts'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT bitwise " | cut -c1-1500 | sed -u "s/^/[$1] /"; }
gate() {  # gate <tag> <devices> <env...>
  local t=$1 dev=$2; shift 2
  echo "=== [$t] chain_gate $* $(date +%T)"
  env ENGINE_TP=2 ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=$dev "$@" timeout 1200 python tools/chain_gate.py \
    --devices $dev --ks 1,3,4,5,7 --slots 1,2 --drafts corpus --iters 200 --skip-accept 2>&1 | tag $t
  echo "=== [$t] exit ${PIPESTATUS[0]} $(date +%T)"
}
(
  echo "=== [link] link_bench $(date +%T)"
  ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1 timeout 3300 python tools/link_bench.py --devices 0,1 2>&1 | tag link
  echo "=== [link] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  gate old_nofb 2,3 ENGINE_DRAFT_IVF_GAP=-1e9
  gate old_served 2,3
) &
(
  gate fast_nofb 4,5 ENGINE_DRAFT_IVF_FAST=1 ENGINE_DRAFT_IVF_GAP=-1e9
  gate fast_served 4,5 ENGINE_DRAFT_IVF_FAST=1
) &
(
  gate old_allfb 6,7 ENGINE_DRAFT_IVF_GAP=1e9
  gate fast_allfb 6,7 ENGINE_DRAFT_IVF_FAST=1 ENGINE_DRAFT_IVF_GAP=1e9
  (python -c "import pytest" 2>/dev/null || pip install -q pytest)
  echo "=== [tests] test_draft_ivf (1 GCD) $(date +%T)"
  HIP_VISIBLE_DEVICES=6 timeout 900 python -m pytest -q -rf tests/test_draft_ivf.py 2>&1 | tail -8 | sed -u "s/^/[tests] /"
  ENGINE_DRAFT_IVF_FAST=1 HIP_VISIBLE_DEVICES=6 timeout 900 python -m pytest -q -rf tests/test_draft_ivf.py 2>&1 | tail -8 | sed -u "s/^/[tests-fast] /"
) &
wait
echo "=== done $(date +%T)"
