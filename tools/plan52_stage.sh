#!/bin/bash
# Plan 52: stage 0 (tools/bulk_ar_probe.py) and, only if it passes, stage 1 (tools/plan52_gate.sh)
# with the probe's fastest bulk layout at 1,792 rows (ENGINE_TP_BULK_BW / ENGINE_TP_BULK_WARPS).
#
#   bash tools/plan52_stage.sh
echo "=== stage 0: bulk all-reduce probe"
timeout 2600 python -u tools/bulk_ar_probe.py 2>&1 | grep -vE "^engine:|amdgpu.ids" | tee /tmp/p52_probe.log
echo "=== probe exit ${PIPESTATUS[0]}"
if ! grep -q "RESULT verdict PASS" /tmp/p52_probe.log; then
  echo "=== stage 0 did not pass: stage 1 gate skipped"
  exit 0
fi
best=$(grep "RESULT best_1792" /tmp/p52_probe.log)
export ENGINE_TP_BULK_BW=$(echo "$best" | sed -E 's/.* bw=([0-9]+) .*/\1/')
export ENGINE_TP_BULK_WARPS=$(echo "$best" | sed -E 's/.* warps=([0-9]+) .*/\1/')
echo "=== stage 1 gate with ENGINE_TP_BULK_BW=$ENGINE_TP_BULK_BW ENGINE_TP_BULK_WARPS=$ENGINE_TP_BULK_WARPS"
bash tools/plan52_gate.sh 2>&1 | grep -v "^+"
