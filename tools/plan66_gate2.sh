#!/bin/bash
# Plan 66 gate on head eceea30 (prefill stack: GDN WY split, streaming MoE down, tuned BLAS table
# for the head's prefill graph sizes) merged. One MI250 node, 8 GCDs.
# NEW  = the defaults: graph sizes every 64 rows 1,600..4,096, one segment per pass.
# HEAD = eceea30's bucket list (to 2,048) and pack budget 8192.
# MIX  = HEAD's sizes to 2,048 (all in the tuned BLAS table) + 64-row sizes 2,112..4,096, one segment.
# Phase 1, in parallel:
#   GCD 0,1: spec_step_bench (spec == plain, NEW)
#   GCD 2,3: tp_parity: HEAD reference (32 prompts 1,600..2,600, one pass each, 96 greedy steps);
#            NEW and MIX teacher-forced against it; NEW in-process repeat
#   GCD 4,5: live_compile_probe (own-kernel compiles after /health, NEW)
#   GCD 6,7: plan66_probe NEW (timing own size vs HEAD size vs eager, capture, memory, KV pages),
#            HEAD load only (memory, KV pages); then unit tests, one file per process (GCD 6 / 7)
# Phase 2: serve_probe replay-like 4x2 (48 sessions, c8, first rounds 850..2,600 shuffled, grow
#   290, 190 out): NEW;HEAD;NEW;MIX;HEAD;NEW (runs 2 and 5 vs 0 = repeatability).
L=/tmp/p66g2
mkdir -p $L
export ENGINE_TP=2 PYTHONUNBUFFERED=1
HEAD="64,80,96,112,128,160,192,224,256,288,320,384,448,512,576,640,704,768,896,1024,1152,1280,1408,1536,1792,2048"
MIX="$HEAD,$(python3 -c "print(','.join(str(n) for n in range(2112, 4097, 64)))")"
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1
PT="python -X faulthandler -m pytest -q -rf -p no:cacheprovider"
F='^=== |RESULT|Traceback|Error|error|passed|failed|prefill graphs|prefill BLAS'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -vE "RESULT (bitwise|gen_step) " | cut -c1-1500 | sed -u "s/^/[$1] /"; }
echo "MIX=$MIX"
echo "=== phase 1 $(date +%T)"
(
  echo "=== [spec] spec_step_bench $(date +%T)"
  HIP_VISIBLE_DEVICES=0,1 timeout 1200 python tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 192 2>&1 \
    | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  E="HIP_VISIBLE_DEVICES=2,3"
  echo "=== [tf] ref, HEAD config $(date +%T)"
  env $E ENGINE_PREFILL_GRAPH_BUCKETS=$HEAD ENGINE_PREFILL_PACK_TOKENS=8192 timeout 1200 python -u tools/tp_parity.py \
    --mode ref --out $L/ref_head.pt --lengths 1600:2600:32 --one-pass 2>&1 | tee -a $L/tf.log | tag tf
  echo "=== [tf] check NEW vs HEAD $(date +%T)"
  env $E timeout 1200 python -u tools/tp_parity.py --mode check --ref $L/ref_head.pt --one-pass 2>&1 | tee -a $L/tf.log | tag tf
  echo "=== [tf] check MIX vs HEAD $(date +%T)"
  env $E ENGINE_PREFILL_GRAPH_BUCKETS=$MIX timeout 1200 python -u tools/tp_parity.py --mode check --ref $L/ref_head.pt \
    --one-pass 2>&1 | tee -a $L/tf.log | tag tf
  echo "=== [tf] repeat NEW $(date +%T)"
  env $E timeout 1200 python -u tools/tp_parity.py --mode repeat --lengths 1600:2600:32 --one-pass 2>&1 | tee -a $L/tf.log | tag tf
  echo "=== [tf] done $(date +%T)"
) &
(
  echo "=== [jit] live_compile_probe $(date +%T)"
  HIP_VISIBLE_DEVICES=4,5 timeout 1800 python tools/live_compile_probe.py --max-new 192 2>&1 \
    | grep -E '"kind": "(compiles|load)"|Traceback|Error' | cut -c1-1500 | sed -u "s/^/[jit] /"
  echo "=== [jit] done $(date +%T)"
) &
(
  echo "=== [probe] plan66_probe NEW $(date +%T)"
  HIP_VISIBLE_DEVICES=6,7 timeout 1500 python -u tools/plan66_probe.py \
    --news 1600,1650,1700,1780,1850,1900,1980,2050,2300,2600,3000,4000 --start 0 --reps 5 --packs 482+300 2>&1 \
    | tee -a $L/probe.log | tag probe
  echo "=== [probe] HEAD load $(date +%T)"
  ENGINE_PREFILL_GRAPH_BUCKETS=$HEAD HIP_VISIBLE_DEVICES=6,7 timeout 900 python -u tools/plan66_probe.py --news "" --packs "" 2>&1 \
    | tee -a $L/probe.log | tag probe-head
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
  --settings "$T;$T+ENGINE_PREFILL_GRAPH_BUCKETS=$HEAD+ENGINE_PREFILL_PACK_TOKENS=8192;$T;$T+ENGINE_PREFILL_GRAPH_BUCKETS=$MIX;$T+ENGINE_PREFILL_GRAPH_BUCKETS=$HEAD+ENGINE_PREFILL_PACK_TOKENS=8192;$T" \
  2>&1 | grep -v "^engine:" | grep -vE "loop_timers|Loading weights|amdgpu.ids|^\s*$|it/s\]" | python3 -c "
import re, sys
keys = ['load_s', 'output_tok_per_s', 'tpot_ms_p50', 'tpot_ms_p90', 'ttft_ms_p50', 'ttft_ms_p90',
        'ttft_first_round_ms_p50', 'ttft_first_round_ms_p90', 'ttft_later_rounds_ms_p50', 'ttft_later_rounds_ms_p90',
        'prefill_passes_per_request']
for line in sys.stdin:
    if line.startswith('{\"event\": \"setting\"'):
        head = line[:3000]
        name = 'HEAD' if 'PACK_TOKENS=8192' in head else ('MIX' if 'GRAPH_BUCKETS' in head else 'NEW')
        vals = []
        for k in keys:
            m = re.search('\"' + k + '\": ([0-9.]+)', line)
            vals.append(k + '=' + (m.group(1) if m else '?'))
        print('SERVED', name, ' '.join(vals), flush=True)
    elif '\"agreement\"' in line:
        m = re.search('\"runs\": \[([0-9, ]+)\].*\"identical_requests\": ([0-9]+), \"requests\": ([0-9]+)', line)
        print('AGREE runs', m.group(1) if m else line[:200], 'identical', m.group(2) if m else '', '/', m.group(3) if m else '', flush=True)
    else:
        print(line.rstrip()[:600], flush=True)
"
echo "serve_probe exit ${PIPESTATUS[0]}"
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | head -40; done
echo "=== done $(date +%T)"
true
