#!/bin/bash
# Plan 65 stage 0 (feasibility, tools only): wave-specialized GEMV with HBM -> LDS DMA producers
# on one gfx90a GCD (tools/wsdecode_probe.py, engine/fast/hip/wsdecode.hip).
#
#   bash tools/plan65_stage0.sh
#
# (a) hipRTC compile of every variant + disassembly of the DMA (no GPU needed)
# (b) streaming TB/s at 4 and 8 rows vs today's kernels, a quick pass first (one variant, no
#     lm_head) so a protocol bug shows before the full sweep; repeatability and spin errors
# (c) GEMV -> 20 us -> GEMV chain, WS (prefetch on/off) vs today's three-node graph
# Each section has its own timeout; every spin in the kernel is capped (err counts, no hang).
export PYTHONUNBUFFERED=1
ts() { echo "=== $1 $(date +%T)"; }
F='^RESULT|^ASM|Traceback|Error|error:'
ts "env"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-unset} host=$(hostname)"
python -c "import torch; print('torch', torch.__version__, torch.version.hip, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
ts "a compile + disassembly"
timeout 600 python -u tools/wsdecode_probe.py --part a 2>&1 | grep -E "$F" | cut -c1-600
ts "b quick (p2d3, no lm_head)"
timeout 600 python -u tools/wsdecode_probe.py --part b --quick 2>&1 | grep -E "$F" | cut -c1-600
echo "rc=${PIPESTATUS[0]}"
ts "b full"
timeout 2400 python -u tools/wsdecode_probe.py --part b 2>&1 | grep -E "$F" | cut -c1-600
echo "rc=${PIPESTATUS[0]}"
ts "c chain"
timeout 1200 python -u tools/wsdecode_probe.py --part c 2>&1 | grep -E "$F" | cut -c1-700
echo "rc=${PIPESTATUS[0]}"
ts "done"
true
