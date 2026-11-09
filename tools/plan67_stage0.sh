#!/bin/bash
# Plan 67 stage 0 on head c7c6727 + the chooser changes behind ENGINE_SPEC_CARRY=1 (default 0 =
# the head). One MI250 node, 8 GCDs:
#   (a) GCD 0,1 / 2,3 in parallel: chain_gate (RUNAHEAD=0) with carry off / on: step ms per
#       (slots 1/2, k) with corpus drafts, gen_step = the real verify steps of MTP's own greedy
#       drafts (served IVF fallback rate) per (slots, k), spec == plain for k 1-7 + adaptive,
#       and (c) the natural-text adaptive rows (k histogram from gen_step n). GCD 4: unit tests.
#   (b) served A/B on all 8 GCDs, replay-like sessions (48 x 6.3 rounds, 8 concurrent):
#       off/on x 3, with k histogram, E, need-capped 2:3 steps, served step prices, the 2-request
#       k=3 time share, TTFT p50/p90 and first-round TTFT, compiles after /health, repeatability.
L=/tmp/p67s0
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT|Traceback|Error|error|passed|failed'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT bitwise" | cut -c1-900 | sed -u "s/^/[$1] /"; }
for c in 0 1; do
  (
    d=$((2 * c)),$((2 * c + 1))
    echo "=== [gate$c] chain_gate ENGINE_SPEC_CARRY=$c $(date +%T)"
    ENGINE_SPEC_CARRY=$c ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=$d timeout 1200 python tools/chain_gate.py \
      --devices $d --ks 1,3,4,5,7 --slots 1,2 --adaptive --drafts corpus --iters 100 --pairs 3 \
      --max-new 192 2>&1 | tee -a $L/gate$c.log | tag gate$c
    echo "=== [gate$c] exit ${PIPESTATUS[0]} $(date +%T)"
  ) &
done
(
  HIP_VISIBLE_DEVICES=4 timeout 900 python -m pytest -q -rf tests/test_spec_carry.py tests/test_spec_depth.py \
    tests/test_draft_ivf.py tests/test_runahead.py 2>&1 | tail -6 | sed -u "s/^/[tests] /"
) &
wait
S="ENGINE_SPEC_CARRY=0,ENGINE_SPEC_CARRY=1,ENGINE_SPEC_CARRY=0,ENGINE_SPEC_CARRY=1,ENGINE_SPEC_CARRY=0,ENGINE_SPEC_CARRY=1"
echo "=== [ab] served A/B $(date +%T)"
timeout 1800 python -u tools/topk_epi_ab.py --settings "$S" 2>&1 | tee $L/ab.log \
  | grep --line-buffered -E '^RESULT \{"kind": "(ab|jit_gate)"|"event": "error"|Traceback' | cut -c1-500
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
python tools/plan67_summary.py $L/ab.log
for f in /tmp/serve-probe-ENGINE_SPEC_CARRY_*.log; do
  echo "--- $f"; grep -E "Traceback|Error" $f | head -5
done
echo "=== tracebacks"; for f in $L/gate*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | head -40; done
echo "=== done $(date +%T)"
