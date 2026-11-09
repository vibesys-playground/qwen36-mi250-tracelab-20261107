#!/bin/sh
# Plan 54: forms of the GDN prefill WY stage (ENGINE_GDN_WY) on one MI250 node.
#
#   sh tools/plan54_check.sh
#
# A (parallel): GCD 6 the form sweep (tools/gdn_wy_bench.py at 61/561/1,650 tokens; picks the
#   fastest form at 561 = BEST, and the fastest stacked-only form = STACK); GCD 7 unit tests.
# B (parallel, TP2 pairs):
#   GCD 0,1 captured prefill pass op split (tools/prefill_op_split.py) at 61/561/1,650 new
#           tokens: ENGINE_GDN_WY=mono:8 (head), BEST
#   GCD 2,3 teacher-forced top-1 of BEST, then of the default, against a mono:8 reference
#           (tools/tp_parity.py)
#   GCD 4,5 spec == plain with BEST (tools/spec_step_bench.py), then Triton compiles after /health
#           under a served replay-like load (tools/live_compile_probe.py) with BEST
#   GCD 6,7 BEST greedy repeat in one process, then across processes
# C: served A/B on all eight GCDs (4 x TP2), replay-like shape, mono:8 / BEST / mono:8 / BEST.
L=/tmp/p54-logs
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"
P=tools/tp_parity.py

echo "=== A: form sweep (GCD 6) + unit tests (GCD 7)"
( HIP_VISIBLE_DEVICES=6 timeout 1500 python -u tools/gdn_wy_bench.py --best-out $L/best.txt ) > $L/bench.log 2>&1 &
( HIP_VISIBLE_DEVICES=7 timeout 1500 python -m pytest -q -rf tests/test_gdn_chunk.py tests/test_prefill_kernels.py 2>&1 | tail -25 ) > $L/tests.log 2>&1 &
wait
grep -vE "$F" $L/tests.log
grep -vE "$F" $L/bench.log | cut -c1-1500
BEST=$(cat $L/best.txt 2>/dev/null || echo split:4:4:128:2)
STACK=$(grep '"kind": "best"' $L/bench.log | python -c "import json,sys; print(json.loads(sys.stdin.read().split('RESULT ',1)[1])['best_stack'])" 2>/dev/null || echo stack:8)
echo "BEST=$BEST STACK=$STACK"

echo "=== B"
( for v in mono:8 $BEST; do echo "=== op split (graph) ENGINE_GDN_WY=$v"; \
    env ENGINE_TP=2 ENGINE_GDN_WY=$v timeout 900 python -u tools/prefill_op_split.py all --devices 0,1 --news 61,561,1650 --reps 5; done ) > $L/split.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3"; \
  env $E ENGINE_GDN_WY=mono:8 timeout 1200 python -u $P --mode ref --out /tmp/p54_ref.pt \
  && echo "--- top-1 ENGINE_GDN_WY=$BEST against mono:8" \
  && env $E ENGINE_GDN_WY=$BEST timeout 1200 python -u $P --mode check --ref /tmp/p54_ref.pt \
  && echo "--- top-1 default ENGINE_GDN_WY (unset) against mono:8" \
  && env $E timeout 1200 python -u $P --mode check --ref /tmp/p54_ref.pt ) > $L/parity.log 2>&1 &
( echo "--- spec == plain ENGINE_GDN_WY=$BEST"; ENGINE_GDN_WY=$BEST timeout 1500 python -u tools/spec_step_bench.py --devices 4,5 --iters 10 --max-new 128; \
  echo "--- live compiles ENGINE_GDN_WY=$BEST"; HIP_VISIBLE_DEVICES=4,5 ENGINE_GDN_WY=$BEST timeout 1800 python -u tools/live_compile_probe.py ) > $L/live.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=6,7 ENGINE_GDN_WY=$BEST"; \
  echo "--- repeat in one process"; env $E timeout 1200 python -u $P --mode repeat --out /tmp/p54_rep.pt; \
  env $E timeout 1200 python -u $P --mode ref --out /tmp/p54_a.pt \
  && echo "--- a second process against the first" && env $E timeout 1200 python -u $P --mode check --ref /tmp/p54_a.pt ) > $L/repeat.log 2>&1 &
wait
echo "=== top-1 vs mono:8"; grep -vE "$F" $L/parity.log | grep -E "RESULT|top|agree|match|Error|^---|failed" | cut -c1-600 | tail -40
echo "=== repeat"; grep -E "RESULT|Error|^---|failed in mode|agree|match" $L/repeat.log | cut -c1-600 | tail -20
echo "=== spec == plain, live compiles"; grep -vE "$F" $L/live.log | grep -vE "^engine:" | cut -c1-1200 | tail -60
echo "=== op split (graph)"; grep -E "^===|wall_ms_median|gdn_chunk_prep|kill" $L/split.log | grep -vE '"kind": "split"' | cut -c1-700
echo "--- per-kernel (gdn chunk bucket, per GCD summed over both)"
grep -E "^===|\"kind\": \"split\"" $L/split.log | python -c "
import json, sys
for line in sys.stdin:
    if line.startswith('==='):
        print(line.strip()); continue
    d = json.loads(line.split('RESULT ', 1)[1])
    names = d.get('bucket_top_names', {}).get('gdn chunk (fused, ENGINE_GDN_CHUNK)', [])
    print(' new_tokens', d['new_tokens'], 'wall_ms_median', d.get('wall_ms_median'), 'busy_any_ms', d.get('busy_any_ms'),
          [(n[:28], round(v / 2, 3)) for n, v in names])
"
echo "=== tracebacks"; for f in $L/*.log; do grep -l Traceback $f; grep -B2 -A25 Traceback $f | head -50; done

echo "=== C: served A/B, 4 x TP2, ENGINE_GDN_WY=mono:8 / $BEST / mono:8 / $BEST"
timeout 2700 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 --max-new 190 \
  --corpus synthetic --settings ENGINE_GDN_WY=mono:8,ENGINE_GDN_WY=$BEST,ENGINE_GDN_WY=mono:8,ENGINE_GDN_WY=$BEST 2>&1 \
  | grep -vE "^engine:|$F" | python -c "
import json, sys
for line in sys.stdin:
    try:
        d = json.loads(line)
    except ValueError:
        print(line.rstrip()[:400]); continue
    if d.get('event') == 'setting':
        d.pop('health', None); d.pop('engine', None)
    print(json.dumps(d)[:1500], flush=True)
"
true
