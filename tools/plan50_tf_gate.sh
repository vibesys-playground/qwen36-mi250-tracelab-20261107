#!/bin/bash
# Plan 50: explain the teacher-forced misses of the HIP GDN path (j250: one at a ref margin of
# 10.4). One MI250 node:
#   GCD 0,1  tp_parity ref, ENGINE_GDN_HIP=0 (today's Triton GDN) -> /tmp/p50_off.pt
#   then in parallel, all fed that reference's tokens:
#   GCD 2,3  tp_parity check ENGINE_GDN_HIP=1 (every miss printed with both margins)
#   GCD 4,5  tp_parity check ENGINE_GDN_HIP=0 (Triton against a second process)
#   GCD 6,7  tools/plan50_tf_probe.py: Triton, HIP and float64 GDN side by side in one process
#   GCD 0,1  tp_parity check ENGINE_GDN_HIP=0 again (third Triton process)
export ENGINE_TP=2 ENGINE_MTP=0 PYTHONUNBUFFERED=1
F='RESULT|Traceback|Error'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
echo "=== ref HIP=0 $(date +%T)"
ENGINE_GDN_HIP=0 HIP_VISIBLE_DEVICES=0,1 timeout 1200 python tools/tp_parity.py --mode ref --out /tmp/p50_off.pt 2>&1 | tag ref-off
( ENGINE_GDN_HIP=1 HIP_VISIBLE_DEVICES=2,3 timeout 1200 python tools/tp_parity.py --mode check --ref /tmp/p50_off.pt 2>&1 | tag hip-vs-off ) &
( ENGINE_GDN_HIP=0 HIP_VISIBLE_DEVICES=4,5 timeout 1200 python tools/tp_parity.py --mode check --ref /tmp/p50_off.pt 2>&1 | tag off-vs-off ) &
( ENGINE_GDN_HIP=0 HIP_VISIBLE_DEVICES=0,1 timeout 1200 python tools/tp_parity.py --mode check --ref /tmp/p50_off.pt 2>&1 | tag off-vs-off2 ) &
( ENGINE_GDN_HIP=1 HIP_VISIBLE_DEVICES=6,7 timeout 3000 python tools/plan50_tf_probe.py --ref /tmp/p50_off.pt 2>&1 | tag probe ) &
wait
echo "=== done $(date +%T)"
