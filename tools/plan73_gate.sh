#!/bin/bash
# Plan 73 gate: exact 3/5/6-slot chained verify buckets (default ENGINE_DECODE_BUCKETS) and cost rows 3, 4.
# OLD = ENGINE_DECODE_BUCKETS=1,2,4,8,16,32 with the plan-58 cost table (head 2a86fd3's behaviour).
# Phase 1, in parallel:
#   GCD 0,1: chain_gate spec == plain at 1, 2, 3 slots, k 1,3,4,5,7 and adaptive (run-ahead off);
#            step ms per (slots, k), 200 each, for the cost table (same job)
#   GCD 2,3: plan73_stage0 (RUNAHEAD=0): per-verify-step outputs exact vs padded bucket (3:4, 5:8,
#            6:8) and greedy tokens; step ms of the new buckets; then spec_step_bench (run-ahead on)
#   GCD 4,5: live_compile_probe (own-kernel compiles after /health)
#   GCD 6,7: unit tests, one file per process (GCD 6 / 7, single GCD: ENGINE_TP unset)
# Phase 2: serve_probe replay-like 4x2 (48 sessions, c8, 6.3 rounds = 302 requests): NEW;OLD;NEW;OLD
#   with loop timers (pure_b3/b5/b6 gpu_step), agreement across runs (repeatability, >= 300/302).
L=/tmp/p73g
mkdir -p $L
export PYTHONUNBUFFERED=1
OLD="ENGINE_DECODE_BUCKETS=1,2,4,8,16,32+ENGINE_SPEC_COST=1:1:7.07,1:3:8.86,1:4:10.29,1:5:11.62,1:7:14.74,2:1:8.28,2:3:11.54,2:4:13.42,2:5:15.26,2:7:19.46"
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1
PT="python -X faulthandler -m pytest -q -rf -p no:cacheprovider"
F='^=== |RESULT|Traceback|Error|error|passed|failed|engine: chained'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT (bitwise|gen_step|bitwise_nonrepeat) " | cut -c1-1200 | sed -u "s/^/[$1] /"; }
echo "=== phase 1 $(date +%T)"
(
  echo "=== [gate] chain_gate $(date +%T)"
  ENGINE_TP=2 ENGINE_MTP=1 ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=0,1 timeout 2700 python tools/chain_gate.py --devices 0,1 \
    --ks 1,3,4,5,7 --slots 1,2,3 --adaptive --drafts corpus --iters 200 --pairs 3 --max-new 192 2>&1 \
    | tee -a $L/gate.log | grep --line-buffered -E "^=== |RESULT (load|step|accept|parity|done|fatal|error)|Traceback|engine: chained" \
    | cut -c1-600 | sed -u "s/^/[gate] /"
  echo "=== [gate] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [s0] plan73_stage0 per-step $(date +%T)"
  ENGINE_TP=2 ENGINE_MTP=1 ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=2,3 timeout 1800 python tools/plan73_stage0.py --devices 2,3 \
    --iters 100 --cases 2:2:1,2:2:3,3:3:1,3:3:3,3:4:3,4:4:1,4:4:3,5:5:1,5:5:3,6:6:1,6:6:3,8:8:3 2>&1 \
    | tee -a $L/s0.log | tag s0
  echo "=== [s0] exit ${PIPESTATUS[0]} $(date +%T)"
  echo "=== [spec] spec_step_bench $(date +%T)"
  ENGINE_TP=2 ENGINE_MTP=1 HIP_VISIBLE_DEVICES=2,3 timeout 1200 python tools/spec_step_bench.py --devices 2,3 --iters 10 --max-new 192 2>&1 \
    | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [jit] live_compile_probe $(date +%T)"
  ENGINE_TP=2 HIP_VISIBLE_DEVICES=4,5 timeout 1800 python tools/live_compile_probe.py --max-new 192 2>&1 \
    | grep -E '"kind": "(compiles|load)"|Traceback|Error' | cut -c1-1500 | sed -u "s/^/[jit] /"
  echo "=== [jit] done $(date +%T)"
) &
(
  echo "=== [tests] one file per process $(date +%T)"
  for d in 6 7; do
    (
      i=0
      for f in tests/test_*.py; do
        i=$((i + 1))
        [ $((i % 2)) -eq $((d - 6)) ] || continue
        out=$(HIP_VISIBLE_DEVICES=$d timeout 900 $PT $f 2>&1); rc=$?
        echo "[tests] $f exit $rc: $(echo "$out" | grep -E "passed|failed|error|skipped" | tail -1)"
        [ $rc -eq 0 ] || echo "$out" | grep -E "FAILED|Error|Fatal|Segmentation|core" | head -8 | sed "s/^/[tests]   /"
      done
    ) &
  done
  wait
  echo "=== [tests] done $(date +%T)"
) &
wait
echo "=== phase 2 served $(date +%T)"
T="ENGINE_LOOP_TIMERS=1"
timeout 3600 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 850 \
  --first-max 2600 --grow 290 --max-new 190 --corpus synthetic \
  --settings "$T;$T+$OLD;$T;$T+$OLD" \
  2>&1 | grep -v "^engine:" | grep -vE "Loading weights|amdgpu.ids|^\s*$|it/s\]" | python3 -c "
import json, re, sys
keys = ['load_s', 'output_tok_per_s', 'tpot_ms_p50', 'tpot_ms_p90', 'ttft_ms_p50', 'ttft_ms_p90', 'cached_prompt_tokens']
for line in sys.stdin:
    if line.startswith('{\"event\": \"loop_timers\"'):
        d = json.loads(line)
        name = 'OLD' if 'DECODE_BUCKETS' in d.get('setting', '') else 'NEW'
        cls = {k: (v.get('n'), (v.get('gpu_step') or [None])[0]) for k, v in d.items()
               if isinstance(v, dict) and re.match(r'pure_b[1-8]_k[1-7]$', k)}
        print('TIMERS', name, 'replica', d.get('replica'), json.dumps(cls, sort_keys=True), flush=True)
    elif line.startswith('{\"event\": \"setting\"'):
        name = 'OLD' if 'DECODE_BUCKETS' in line[:3000] else 'NEW'
        vals = []
        for k in keys:
            m = re.search('\"' + k + '\": ([0-9.]+)', line)
            vals.append(k + '=' + (m.group(1) if m else '?'))
        print('SERVED', name, ' '.join(vals), flush=True)
    elif '\"agreement\"' in line:
        print('AGREE', line.rstrip()[:400], flush=True)
    else:
        print(line.rstrip()[:400], flush=True)
"
echo "serve_probe exit ${PIPESTATUS[0]}"
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | head -40; done
echo "=== done $(date +%T)"
true
