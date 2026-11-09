#!/bin/bash
# Plan 66 step 0 on one MI250 node (8 GCDs), env vars only (engine unchanged):
#   (a-c) GCD 4,5: tools/plan66_probe.py with the extended bucket list (64-row sizes to 4,096):
#         first-round passes at 1,700/1,900/2,050/2,300/2,600 as graph (own size), graph at the
#         head's bucket, eager; packs 482+300 and 1,200+482 eager vs two single replays; capture
#         time, memory, pg_error, KV pages. Then the head's default list (load/memory only + the
#         same passes) for the memory and KV-pool delta.
#   (d)   serve_probe replay-like (48 sessions, c8, 4x2, first rounds 850..2,600, ~480 new
#         tokens per later round, 190 out), A/B/A/B of default vs extended buckets +
#         ENGINE_PREFILL_PACK_TOKENS=1, loop timers on.
export PYTHONUNBUFFERED=1 ENGINE_TP=2
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced"
EXT=$(python3 -c "print(','.join(map(str, [64,80,96,112,128,160,192,224,256,288,320,384,448,512,576,640,704,768,896,1024,1152,1280,1408,1536] + list(range(1600, 4097, 64)))))")
ts() { echo "=== $1 $(date +%T)"; }
echo "EXT=$EXT"

ts "a-c probe, extended buckets, GCD 4,5"
ENGINE_PREFILL_GRAPH_BUCKETS=$EXT HIP_VISIBLE_DEVICES=4,5 timeout 1500 python -u tools/plan66_probe.py \
  --news 1700,1900,2050,2300,2600 --start 0 --reps 7 --packs 482+300,1200+482 2>&1 \
  | grep -E "RESULT|prefill graphs|Traceback|Error|error" | cut -c1-1500
echo "probe ext exit ${PIPESTATUS[0]}"

ts "a-c probe, head buckets, GCD 4,5"
HIP_VISIBLE_DEVICES=4,5 timeout 1200 python -u tools/plan66_probe.py \
  --news 1900,2300 --start 0 --reps 5 --packs "" 2>&1 \
  | grep -E "RESULT|prefill graphs|Traceback|Error|error" | cut -c1-1500
echo "probe head exit ${PIPESTATUS[0]}"

ts "d served A/B/A/B"
B="ENGINE_LOOP_TIMERS=1+ENGINE_PREFILL_GRAPH_BUCKETS=$EXT+ENGINE_PREFILL_PACK_TOKENS=1"
echo "B=$B"
timeout 3000 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 850 \
  --first-max 2600 --grow 290 --max-new 190 --corpus synthetic \
  --settings "ENGINE_LOOP_TIMERS=1;$B;ENGINE_LOOP_TIMERS=1;$B" 2>&1 | grep -v "^engine:" | grep -vE "$F" | cut -c1-2500
echo "serve_probe exit ${PIPESTATUS[0]}"
ts "done"
true
