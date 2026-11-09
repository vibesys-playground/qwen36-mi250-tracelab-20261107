#!/bin/bash
# Plan 72 stage 0 (feasibility, tools only): channel-aware weight layout (row pad, staggered
# expert bases, chunk swizzle) on one gfx90a GCD (tools/layout_bw_bench.py).
#
#   bash tools/plan72_stage0.sh
#
# (a) bare read probe: today's access shapes vs pad / stagger / swizzle / contiguous / stream
# (b) real kernels with a stride argument: MoE up/down at R=2/4/8, MFMA GEMV at the TP2 shapes,
#     bitwise vs the engine's kernels. Each section has its own timeout.
export PYTHONUNBUFFERED=1
ts() { echo "=== $1 $(date +%T)"; }
F='RESULT|Traceback|Error|error:'
ts "env"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-unset} host=$(hostname)"
python -c "import torch; print('torch', torch.__version__, torch.version.hip, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
ts "a probe"
timeout -k 30 1500 python -u tools/layout_bw_bench.py --part a 2>&1 | grep -E "$F" | cut -c1-500
echo "rc=${PIPESTATUS[0]}"
ts "b moe"
timeout -k 30 2400 python -u tools/layout_bw_bench.py --part moe 2>&1 | grep -E "$F" | cut -c1-700
echo "rc=${PIPESTATUS[0]}"
ts "b dense"
timeout -k 30 1800 python -u tools/layout_bw_bench.py --part dense 2>&1 | grep -E "$F" | cut -c1-500
echo "rc=${PIPESTATUS[0]}"
ts "done"
true
