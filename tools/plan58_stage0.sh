#!/bin/bash
# Plan 58 stage 0: slot-aware chained-MTP depth. Tools only (ENGINE_SPEC_KMAX raised in these
# processes; no default changes). One MI250 node, 8 GCDs, HIP GDN on (the default).
#
# Phase 1, in parallel (pool probes: load, print KV pages and GDN state bytes, exit):
#   GCD 0,1: KMAX 3 SLOTS 32 (today)    GCD 2,3: KMAX 5 SLOTS 32
#   GCD 4,5: KMAX 5 SLOTS 16            GCD 6,7: KMAX 7 SLOTS 16
# Phase 2, in parallel:
#   GCD 0,1: chain_gate KMAX 7 SLOTS 32 (its load line is the KMAX 7 / 32 pool), RUNAHEAD=0 so
#            every step is a timed verify_chain: step ms at 1 and 2 slots x k 1,3,4,5,7 (median
#            of 200, corpus drafts), marginal ms per extra row, per-depth acceptance, tokens per
#            slot-step and spec == plain at 1 and 2 slots on replay-like and natural text,
#            tokens/ms per (slots, k) with the cheap-link (plan 47 IVF) estimate.
#   GCD 2,3: glue_split KMAX 7: per-kernel-family split (gdn, moe, gemv, attn, allreduce...) of
#            the captured chain step at 1 and 2 slots x k 1,3,4,5,7 (corpus drafts).
L=/tmp/p58
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
F='^=== |RESULT|Traceback|Error|error'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
pool() {  # name devices kmax slots
  echo "=== [$1] kv_pool_probe KMAX=$3 SLOTS=$4 $(date +%T)"
  ENGINE_SPEC_KMAX=$3 ENGINE_SLOTS=$4 HIP_VISIBLE_DEVICES=$2 timeout 900 python tools/kv_pool_probe.py \
    --devices $2 2>&1 | tee -a $L/$1.log | tag $1
  echo "=== [$1] exit ${PIPESTATUS[0]} $(date +%T)"
}
pool k3s32 0,1 3 32 &
pool k5s32 2,3 5 32 &
pool k5s16 4,5 5 16 &
pool k7s16 6,7 7 16 &
wait
(
  echo "=== [gate] chain_gate KMAX=7 $(date +%T)"
  ENGINE_SPEC_KMAX=7 ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=0,1 timeout 3000 python tools/chain_gate.py \
    --devices 0,1 --ks 1,3,4,5,7 --slots 1,2 --drafts corpus --iters 200 --pairs 4 --max-new 192 2>&1 \
    | tee -a $L/gate.log | grep --line-buffered -E "$F" | grep --line-buffered -v "RESULT bitwise " \
    | sed -u "s/^/[gate] /"
  echo "=== [gate] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [split] glue_split KMAX=7 $(date +%T)"
  C=chain:1:1,chain:1:3,chain:1:4,chain:1:5,chain:1:7,chain:2:1,chain:2:3,chain:2:4,chain:2:5,chain:2:7
  ENGINE_SPEC_KMAX=7 HIP_VISIBLE_DEVICES=2,3 timeout 2400 python tools/glue_split.py --devices 2,3 \
    --configs $C --drafts corpus --iters 50 --steps 5 2>&1 | tee -a $L/split.log \
    | grep --line-buffered -E "^=== |RESULT (load|wall|step|split|chain|warn)|Traceback|rror" \
    | sed -u "s/^/[split] /"
  echo "=== [split] exit ${PIPESTATUS[0]} $(date +%T)"
) &
wait
grep -h "RESULT bitwise_summary" $L/gate.log | sed "s/^/[gate] /"
echo "=== tracebacks"
for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | tail -30; done
echo "=== done $(date +%T)"
