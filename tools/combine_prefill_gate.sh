#!/bin/sh
# Gate for plans 54 + 60 combined: the split GDN prefill WY stage (ENGINE_GDN_WY, default
# split:2:4:128:2) and the N-streaming MoE down kernel (ENGINE_MOE_DOWN_STREAM, default on above
# 1,024 tokens). One MI250 node, 8 GCDs.
#
#   sh tools/combine_prefill_gate.sh
#
# off = ENGINE_GDN_WY=mono:8 ENGINE_MOE_DOWN_STREAM=0 (the head's kernels); on = both unset.
# In parallel:
#   GCD 0,1  captured prefill op split at 61/561/1,650/2,300 new tokens, off then on
#   GCD 2,3  prefill logits on vs off, bitwise (tools/prefill_flags_bitwise.py), eager and graphs
#   GCD 4,5  spec == plain (on), then Triton compiles after /health under a served load (on)
#   GCD 6    GPU unit tests, tests/test_gdn_chunk.py
#   GCD 7    GPU unit tests, tests/test_moe_prefill.py
L=/tmp/cp-logs
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
OFF="ENGINE_GDN_WY=mono:8 ENGINE_MOE_DOWN_STREAM=0"
ON="-u ENGINE_GDN_WY -u ENGINE_MOE_DOWN_STREAM"
S="timeout 1800 python -u tools/prefill_op_split.py all --news 61,561,1650,2300 --reps 5 --devices 0,1"
( echo "=== op split (graph) off"; env ENGINE_TP=2 $OFF $S; \
  echo "=== op split (graph) on"; env $ON ENGINE_TP=2 $S ) > $L/split.log 2>&1 &
( env $ON ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3 timeout 2400 python -u tools/prefill_flags_bitwise.py \
    --news 61,561,1650,2300; echo "exit $?" ) > $L/bitwise.log 2>&1 &
( echo "--- spec == plain (on)"; env $ON timeout 2400 python -u tools/spec_step_bench.py --devices 4,5 --iters 10 --max-new 128; \
  echo "--- live compiles (on)"; env $ON HIP_VISIBLE_DEVICES=4,5 timeout 3000 python -u tools/live_compile_probe.py; \
  cp /tmp/live-compile-server.log $L/live-server.log 2>/dev/null ) > $L/live.log 2>&1 &
( HIP_VISIBLE_DEVICES=6 timeout 1800 python -X faulthandler -m pytest -q -rf -p no:cacheprovider \
    tests/test_gdn_chunk.py 2>&1 | tail -30 ) > $L/tests_gdn.log 2>&1 &
( HIP_VISIBLE_DEVICES=7 timeout 1800 python -X faulthandler -m pytest -q -rf -p no:cacheprovider \
    tests/test_moe_prefill.py 2>&1 | tail -30 ) > $L/tests_moe.log 2>&1 &
wait

F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced|^engine:"
echo "=== unit tests: test_gdn_chunk.py (GCD 6)"; grep -vE "$F" $L/tests_gdn.log
echo "=== unit tests: test_moe_prefill.py (GCD 7)"; grep -vE "$F" $L/tests_moe.log
echo "=== bitwise on vs off"; grep -E "RESULT|Error|^exit" $L/bitwise.log | cut -c1-500
echo "=== spec == plain, live compiles"; grep -vE "$F" $L/live.log | cut -c1-1500 | tail -60
echo "=== op split (graph)"; grep -E "^===|wall_ms_median|kill" $L/split.log | grep -vE '"kind": "(split|host_log)"' | cut -c1-700
echo "--- per-kernel (gdn chunk and MoE buckets, ms per GCD)"
grep -E "^===|\"kind\": \"split\"" $L/split.log | python -c "
import json, sys
for line in sys.stdin:
    if line.startswith('==='):
        print(line.strip()); continue
    d = json.loads(line.split('RESULT ', 1)[1])
    print(' new_tokens', d.get('new_tokens'), 'wall_ms_median', d.get('wall_ms_median'), 'busy_any_ms', d.get('busy_any_ms'))
    for b, names in d.get('bucket_top_names', {}).items():
        if 'gdn' in b.lower() or 'moe' in b.lower():
            print('   ', b[:48], [(n[:30], round(v / 2, 3)) for n, v in names][:6])
"
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
