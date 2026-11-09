#!/bin/bash
# Plan 75 stage 0 (feasibility, tools only): tile-outer all-rows MoE down kernel variants vs
# today's on one gfx90a GCD (tools/moe_down_rows_bench.py), R = 2..24, graph-of-N, bitwise.
#
#   bash tools/plan75_stage0.sh
export PYTHONUNBUFFERED=1
ts() { echo "=== $1 $(date +%T)"; }
F='RESULT|Traceback|Error|error:'
ts "env"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-unset} host=$(hostname)"
python -c "import torch; print('torch', torch.__version__, torch.version.hip, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
ts "rows bench"
timeout -k 30 2400 python -u tools/moe_down_rows_bench.py --rows 2,4,6,8,12,16,24 2>&1 | grep -E "$F" | cut -c1-700
echo "rc=${PIPESTATUS[0]}"
ts "done"
true
