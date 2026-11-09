#!/bin/bash
# profile-ivf job 2: chained step time at k=1,3,4,5,7 x 1,2 slots with no draft fallback / every draft on the full pass / IVF off.
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
F='RESULT step|Traceback|IVF drafts'
run() { tag=$1; dev=$2; shift 2; ( env "$@" HIP_VISIBLE_DEVICES=$dev timeout 3000 python tools/chain_gate.py --devices $dev --pairs 3 --iters 30 --skip-accept --ks 1,3,4,5,7 --slots 1,2 2>&1 | grep --line-buffered -E "$F" | sed -u "s/^/[$tag] /"; ) & }
run nofb 0,1 ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_GAP=-1e9
run allfb 2,3 ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_GAP=1e9
run off 4,5 ENGINE_DRAFT_IVF=0
wait
