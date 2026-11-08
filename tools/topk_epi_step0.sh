#!/bin/bash
# Plan 37 step 0: the router top-k in the router|shared K-sum kernel (ENGINE_TOPK_EPI 1 = last
# program by atomic counter, 2 = one extra program per row) vs today's _topk_kernel (0).
#
#   bash tools/topk_epi_step0.sh        # one MI250 node, 8 GCDs, TP2 pairs at once
#
# GCD 0,1: tools/glue_split.py chain k=1, k=3 (B=2) and decode B=2 with EPI 0, 1, 2 in turn
#          (graph-replay step ms, per-kernel split) plus --parity logit hashes per setting.
# GCD 2,3: tools/topk_epi_check.py (bitwise over 1000 graph replays per layer and row count),
#          then tools/chain_gate.py with EPI 0, 1, 2 (served verify_chain step ms at k = 1, 3
#          and spec == plain parity per k).
# GCD 4,5 / 6,7: tools/spec_step_bench.py with EPI 1 / 2 (spec == plain 8/8).
# Every RESULT / === / error line streams to stdout as it is printed, tagged by its pair; full
# logs stay in /tmp/p37. Per-process timeouts make a hang show as "exit 124".
L=/tmp/p37
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
F='^=== |RESULT|Traceback|Error|error'
(
  for m in 0 1 2; do
    echo "=== [glue] ENGINE_TOPK_EPI=$m $(date +%T)"
    ENGINE_TOPK_EPI=$m HIP_VISIBLE_DEVICES=0,1 timeout 1200 python tools/glue_split.py --devices 0,1 \
      --configs chain:2:1,chain:2:3,decode:2 --iters 50 --parity 2>&1 | tee -a $L/glue.log \
      | grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT (top|split .*item=(1_add|3_|4|5|allreduce|gdn|attn|blas|other_torch))" \
      | sed -u "s/^/[glue$m] /"
    echo "=== [glue] exit ${PIPESTATUS[0]}"
  done
) &
(
  echo "=== [gate] topk_epi_check $(date +%T)"
  HIP_VISIBLE_DEVICES=2,3 timeout 900 python tools/topk_epi_check.py --devices 2,3 --replays 1000 2>&1 \
    | tee -a $L/check.log | grep --line-buffered -E "$F" | sed -u "s/^/[check] /"
  echo "=== [check] exit ${PIPESTATUS[0]}"
  for m in 0 1 2; do
    echo "=== [gate] chain_gate ENGINE_TOPK_EPI=$m $(date +%T)"
    ENGINE_TOPK_EPI=$m HIP_VISIBLE_DEVICES=2,3 timeout 900 python tools/chain_gate.py --devices 2,3 2>&1 \
      | tee -a $L/gate.log | grep --line-buffered -E "$F" | sed -u "s/^/[gate$m] /"
    echo "=== [gate] exit ${PIPESTATUS[0]}"
  done
) &
for m in 1 2; do
  (
    d=$((2 + 2 * m)),$((3 + 2 * m))
    echo "=== [spec$m] spec_step_bench ENGINE_TOPK_EPI=$m on $d $(date +%T)"
    ENGINE_TOPK_EPI=$m HIP_VISIBLE_DEVICES=$d timeout 1200 python tools/spec_step_bench.py --devices $d \
      --iters 30 --max-new 128 2>&1 | tee -a $L/spec$m.log | grep --line-buffered -E "$F" | sed -u "s/^/[spec$m] /"
    echo "=== [spec$m] exit ${PIPESTATUS[0]}"
  ) &
done
wait
echo "=== all done $(date +%T)"
for f in check gate glue spec1 spec2; do grep -B2 -A12 Traceback $L/$f.log 2>/dev/null | tail -30; done
