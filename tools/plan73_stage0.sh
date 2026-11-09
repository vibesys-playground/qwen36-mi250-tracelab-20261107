#!/bin/bash
# Plan 73 stage 0 (tools only; default buckets unchanged in this commit). One MI250 node, in parallel:
#   GCD 0,1: plan73_stage0.py with ENGINE_DECODE_BUCKETS=1,2,3,4,5,6,8,16,32: step ms (synced call
#            and graph-of-N replay, 200 each) for 3 real slots on the 3- vs 4-slot graph (k 1,3,4,5),
#            5/6 real on 5/6 vs 8, bitwise verify_chain outputs and greedy generations exact vs
#            padded bucket, load/capture time, free memory, KV pages.
#   GCD 2,3: the same tool with today's buckets (load/capture time, memory, KV pages, timing cross-check).
#   GCD 4,5: chain_gate spec == plain at 3 slots (k 1,3 and adaptive) with the new buckets.
#   GCD 6,7: glue_split per-family split of chain 2:3, 3:3, 4:3 with the new buckets.
L=/tmp/p73s0
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
NEW=1,2,3,4,5,6,8,16,32
F='^=== |RESULT|Traceback|Error|error|engine: chained|engine: fast path'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT (gen_step) " | cut -c1-1500 | sed -u "s/^/[$1] /"; }
(
  echo "=== [new] plan73_stage0 buckets $NEW $(date +%T)"
  ENGINE_DECODE_BUCKETS=$NEW HIP_VISIBLE_DEVICES=0,1 timeout 2400 python tools/plan73_stage0.py --devices 0,1 2>&1 \
    | tee -a $L/new.log | tag new
  echo "=== [new] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [old] plan73_stage0 default buckets $(date +%T)"
  HIP_VISIBLE_DEVICES=2,3 timeout 1800 python tools/plan73_stage0.py --devices 2,3 --bitwise "" --skip-gen \
    --cases 2:2:3,3:4:1,3:4:3,3:4:4,3:4:5,4:4:3,5:8:3,6:8:3 2>&1 | tee -a $L/old.log | tag old
  echo "=== [old] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [gate] chain_gate 3 slots new buckets $(date +%T)"
  ENGINE_DECODE_BUCKETS=$NEW ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=4,5 timeout 2400 python tools/chain_gate.py \
    --devices 4,5 --ks 1,3 --slots 3 --adaptive --drafts corpus --iters 100 --pairs 3 --max-new 192 2>&1 \
    | tee -a $L/gate.log | grep --line-buffered -E "^=== |RESULT (load|step|accept|parity|done|fatal|error)|Traceback" \
    | cut -c1-800 | sed -u "s/^/[gate] /"
  echo "=== [gate] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [split] glue_split new buckets $(date +%T)"
  ENGINE_DECODE_BUCKETS=$NEW HIP_VISIBLE_DEVICES=6,7 timeout 2400 python tools/glue_split.py --devices 6,7 \
    --configs chain:2:3,chain:3:3,chain:4:3 --drafts corpus --iters 50 --steps 5 2>&1 | tee -a $L/split.log \
    | grep --line-buffered -E "^=== |RESULT (load|wall|split|chain|warn)|Traceback|rror" | cut -c1-1500 \
    | sed -u "s/^/[split] /"
  echo "=== [split] exit ${PIPESTATUS[0]} $(date +%T)"
) &
wait
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f 2>/dev/null | head -60; done
echo "=== done $(date +%T)"
