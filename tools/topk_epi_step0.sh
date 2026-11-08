#!/bin/bash
# Plan 37 step 0: the router top-k in the router|shared K-sum kernel (ENGINE_TOPK_EPI 1 = last
# program by atomic counter, 2 = one extra program per row) vs today's _topk_kernel (0).
#
#   bash tools/topk_epi_step0.sh        # one MI250 node, 8 GCDs, TP2 pairs at once
#
# GCD 0,1: tools/glue_split.py chain k=1, k=3 (B=2) and decode B=2 with EPI 0, 1, 2, 0 in turn
#          (graph-replay step ms, per-kernel split) plus --parity logit hashes per setting.
# GCD 2,3: tools/topk_epi_check.py (bitwise over 1000 graph replays per layer and row count),
#          then tools/chain_gate.py with EPI 0, 1, 2, 0 (served verify_chain step ms at k = 1, 3
#          and spec == plain parity per k).
# GCD 4,5 / 6,7: tools/spec_step_bench.py with EPI 1 / 2 (spec == plain 8/8).
L=/tmp/p37
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1
T="timeout 1800"
(
  for m in 0 1 2 0; do
    echo "=== glue_split ENGINE_TOPK_EPI=$m"
    ENGINE_TOPK_EPI=$m HIP_VISIBLE_DEVICES=0,1 $T python tools/glue_split.py --devices 0,1 \
      --configs chain:2:1,chain:2:3,decode:2 --iters 50 --parity
    echo "=== exit $?"
  done
) > $L/glue.log 2>&1 &
(
  echo "=== topk_epi_check"
  HIP_VISIBLE_DEVICES=2,3 $T python tools/topk_epi_check.py --devices 2,3 --replays 1000
  echo "=== exit $?"
  for m in 0 1 2 0; do
    echo "=== chain_gate ENGINE_TOPK_EPI=$m"
    ENGINE_TOPK_EPI=$m HIP_VISIBLE_DEVICES=2,3 $T python tools/chain_gate.py --devices 2,3
    echo "=== exit $?"
  done
) > $L/gate.log 2>&1 &
(
  echo "=== spec_step_bench ENGINE_TOPK_EPI=1"
  ENGINE_TOPK_EPI=1 HIP_VISIBLE_DEVICES=4,5 $T python tools/spec_step_bench.py --devices 4,5 --iters 30 --max-new 128
  echo "=== exit $?"
) > $L/spec1.log 2>&1 &
(
  echo "=== spec_step_bench ENGINE_TOPK_EPI=2"
  ENGINE_TOPK_EPI=2 HIP_VISIBLE_DEVICES=6,7 $T python tools/spec_step_bench.py --devices 6,7 --iters 30 --max-new 128
  echo "=== exit $?"
) > $L/spec2.log 2>&1 &
wait
for f in gate glue spec1 spec2; do
  echo "################ $f"
  grep -E "^=== |RESULT (load|bitwise|router_topk|step|accept|parity|tp_allreduce|wall|split .*item=(2_topk|gemv|other|moe) |glue |kill|done)|RESULT parity_hash|Error|Traceback|error" $L/$f.log
done
echo "################ tracebacks (tail)"
for f in gate glue spec1 spec2; do grep -B2 -A12 Traceback $L/$f.log | tail -40; done
