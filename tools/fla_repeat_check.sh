#!/bin/sh
# Plan 26: is cross-process greedy divergence fla's per-process autotuning, and do pins fix it?
#
#   sh tools/fla_repeat_check.sh [pinned-only [split]]   # (one MI250 node, 8 GCDs, TP2 replicas)
#
# 1. Four processes at once (GCD pairs), each autotuning fla on its own (FLA_CACHE_RESULTS=0,
#    ENGINE_FLA_PIN=0), FUSED=1: greedy references (tools/tp_parity.py) and their fla picks;
#    tools/fla_pins_pick.py compares picks and outputs and writes the majority pins.
# 2. With those pins (ENGINE_FLA_PINS), at once: two FUSED=1 processes in sequence on GCD 0,1
#    (ref, then check: 8/8 identical?) and a third concurrently on GCD 2,3 (compared with the
#    first); FUSED=1 against a FUSED=0 reference on GCD 4,5 (teacher-forced top-1); spec == plain
#    on GCD 6,7. With ENGINE_FLA_PINS unset the committed engine/fast/fla_configs.json is used.
# "pinned-only" skips step 1 (the committed pins); "split" then adds tools/prefill_op_split.py
# FUSED=0 and FUSED=1 on GCD 4,5 (GPU time and wall with the pinned fla configs).
set -x
L=/tmp/fr-logs
mkdir -p $L
P=tools/tp_parity.py
PINS=${PINS:-/tmp/pins.json}
if [ "$1" != "pinned-only" ]; then
  for i in 0 1 2 3; do
    ( env ENGINE_TP=2 HIP_VISIBLE_DEVICES=$((2*i)),$((2*i+1)) ENGINE_PREFILL_FUSED=1 FLA_CACHE_RESULTS=0 ENGINE_FLA_PIN=0 \
      timeout 1800 python $P --mode ref --out /tmp/auto$i.pt --dump-fla /tmp/snap$i.json ) > $L/auto$i.log 2>&1 &
  done
  wait
  for i in 0 1 2 3; do grep -E "RESULT|Error" $L/auto$i.log | tail -4; done
  python tools/fla_pins_pick.py --snapshots /tmp/snap0.json /tmp/snap1.json /tmp/snap2.json /tmp/snap3.json \
    --refs /tmp/auto0.pt /tmp/auto1.pt /tmp/auto2.pt /tmp/auto3.pt --out $PINS
  export ENGINE_FLA_PINS=$PINS
fi
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1 ENGINE_PREFILL_FUSED=1"; \
  env $E timeout 1800 python $P --mode ref --out /tmp/pin_a.pt \
  && echo "--- FUSED=1 pinned: a second process against the first" \
  && env $E timeout 1800 python $P --mode check --ref /tmp/pin_a.pt ) > $L/pin_ab.log 2>&1 &
( env ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 ENGINE_PREFILL_FUSED=1 timeout 1800 python $P --mode ref --out /tmp/pin_c.pt ) > $L/pin_c.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=4,5"; \
  env $E ENGINE_PREFILL_FUSED=0 timeout 1800 python $P --mode ref --out /tmp/pin_ref0.pt \
  && echo "--- FUSED=1 pinned against the FUSED=0 pinned reference" \
  && env $E ENGINE_PREFILL_FUSED=1 timeout 1800 python $P --mode check --ref /tmp/pin_ref0.pt ) > $L/pin_parity.log 2>&1 &
( ENGINE_PREFILL_FUSED=1 timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
wait
echo "=== pinned: two FUSED=1 processes in sequence"; grep -E "RESULT|Error|^---" $L/pin_ab.log | tail -10
echo "=== pinned: a concurrent third process against the first"
python tools/fla_pins_pick.py --refs /tmp/pin_a.pt /tmp/pin_c.pt
grep -E "RESULT|Error" $L/pin_c.log | tail -4
echo "=== pinned: FUSED=1 against FUSED=0"; grep -E "RESULT|Error|^---" $L/pin_parity.log | tail -10
echo "=== spec == plain (FUSED=1, pinned)"; grep -E "RESULT|Error|Traceback" $L/spec.log | tail -20
if [ "$2" = "split" ]; then
  ( ENGINE_PREFILL_FUSED=0 timeout 1800 python tools/prefill_op_split.py all --devices 4,5 ; \
    ENGINE_PREFILL_FUSED=1 timeout 1800 python tools/prefill_op_split.py all --devices 4,5 ) > $L/split.log 2>&1
  echo "=== op split (FUSED=0, then FUSED=1)"
  grep -E "^N=|ALL FUSIBLE|TOTAL|gdn chunk \(fla\)" $L/split.log | cut -c1-330
fi
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A20 Traceback $f | head -40; done
true
