#!/bin/sh
# Plan 41 stage 2: the served split prefill (engine/split_prefill.py, engine/pair_host.py).
#
#   sh tools/plan41_check.sh
#
# Phase 1, in parallel:
#   GCD 0-3  numerics and wall of split vs single in one pair-host process
#            (tools/split_prefill_check.py): top-1 / MTP draft agreement, max logit diff vs the
#            padded-size floor, split repeatability, both owners
#   GCD 4-7  unit tests, then Triton compiles after /health under a served load with split on
#            (tools/live_compile_probe.py; 2 replicas = 1 pair-host process)
# Phase 2, all 8 GCDs: served A/B (tools/serve_probe.py), replay-like shape (first prompts
#   800..2,600 tokens, growing rounds, 8 sessions in flight): SPLIT 0/1/0/1, then pair hosts
#   with no split (min tokens 10^6: the cost of two pairs sharing a process and its GIL), then
#   split on with MTP off (spec == plain against run 2). Token agreement between all runs:
#   repeated settings are cross-process repeatability.
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/p41-logs
mkdir -p $L
( HIP_VISIBLE_DEVICES=0,1,2,3 timeout 2700 python -u tools/split_prefill_check.py --reps 5 ) > $L/check.log 2>&1 &
( HIP_VISIBLE_DEVICES=4 timeout 600 python -m pytest -q -rf tests/test_split_prefill.py tests/test_router.py tests/test_cpu_place.py 2>&1 | tail -15; \
  echo "--- live compiles, split on"; \
  HIP_VISIBLE_DEVICES=4,5,6,7 ENGINE_SPLIT_PREFILL=1 timeout 3000 python tools/live_compile_probe.py; \
  echo "--- server log (engine/split lines)"; grep -hE "split|pair host|Traceback|Error" /tmp/live-compile*.log 2>/dev/null | head -40 ) > $L/tests.log 2>&1 &
wait
echo "=== split vs single"; grep -vE "Loading weights|amdgpu.ids|^\s*$|it/s\]" $L/check.log | cut -c1-1500 | tail -120
echo "=== tests, live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$|it/s\]" $L/tests.log | cut -c1-3000 | tail -80
echo "=== served A/B"
S="${PLAN41_SETTINGS:-ENGINE_SPLIT_PREFILL=0,ENGINE_SPLIT_PREFILL=1,ENGINE_SPLIT_PREFILL=0,ENGINE_SPLIT_PREFILL=1}"
S="$S,${PLAN41_EXTRA:-ENGINE_SPLIT_PREFILL=1+ENGINE_SPLIT_MIN_TOKENS=1000000,ENGINE_SPLIT_PREFILL=1+ENGINE_MTP=0}"
timeout 5400 python -u tools/serve_probe.py --sessions 32 --rounds 4 --concurrency 8 \
  --first-spread 800:2600 --grow 250 --max-new 200 --wait 0.5 --corpus synthetic --settings "$S" 2>&1 \
  | grep --line-buffered -vE "Loading weights|amdgpu.ids|it/s\]" | cut -c1-6000
echo "=== server logs (split lines)"; for f in /tmp/serve-probe-*.log; do echo "--- $f"; grep -E "split|pair host|router: starting|Traceback|Error|timeout" $f | head -20; done
echo "=== tracebacks"; for f in $L/*.log /tmp/serve-probe-*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
