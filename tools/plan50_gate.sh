#!/bin/bash
# Plan 50 stage 1 gate: the HIP GDN step (ENGINE_GDN_HIP=1, the default) against today's Triton
# kernels (=0) on one MI250 node, 8 GCDs, TP2 pairs in parallel.
#
#   bash tools/plan50_gate.sh
#
#   GCD 0,1: chain_gate HIP=1 (served k=1,3,7 verify steps, acceptance, spec == plain), then
#            glue_split HIP=1 (k=3 chained step split at 1 and 2 slots, decode B=2)
#   GCD 2,3: the same with HIP=0
#   GCD 4,5: tp_parity teacher-forced: HIP=0 ref -> HIP=1 check (top-1 >= 98.5%), then
#            HIP=1 ref -> HIP=1 check (cross-process repeatability), then repeat mode (in process)
#   GCD 6,7: spec_step_bench HIP=1 (spec == plain 8/8), unit tests (GCD 6), live_compile_probe
#            (own-kernel compiles after /health, HIP compiles logged too)
L=/tmp/p50
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT|Traceback|Error|error|gdn_hip'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
run_pair() {  # $1 = HIP flag, $2 = devices, $3 = tag
  echo "=== [$3] chain_gate ENGINE_GDN_HIP=$1 $(date +%T)"
  ENGINE_GDN_HIP=$1 HIP_VISIBLE_DEVICES=$2 timeout 1500 python tools/chain_gate.py --devices $2 2>&1 \
    | tee -a $L/$3-gate.log | tag $3-gate
  echo "=== [$3] chain_gate exit ${PIPESTATUS[0]}"
  echo "=== [$3] glue_split ENGINE_GDN_HIP=$1 $(date +%T)"
  ENGINE_GDN_HIP=$1 HIP_VISIBLE_DEVICES=$2 timeout 1500 python tools/glue_split.py --devices $2 \
    --configs chain:2:3,chain:1:3,decode:2 --iters 30 2>&1 | tee -a $L/$3-split.log | tag $3-split
  echo "=== [$3] glue_split exit ${PIPESTATUS[0]}"
}
run_pair 1 0,1 on &
run_pair 0 2,3 off &
(
  E="HIP_VISIBLE_DEVICES=4,5"
  echo "=== [tf] tp_parity HIP=0 ref, HIP=1 check $(date +%T)"
  env $E ENGINE_GDN_HIP=0 timeout 1200 python tools/tp_parity.py --mode ref --out /tmp/p50_off.pt 2>&1 | tee -a $L/tf.log | tag tf-ref0
  env $E ENGINE_GDN_HIP=1 timeout 1200 python tools/tp_parity.py --mode check --ref /tmp/p50_off.pt 2>&1 | tee -a $L/tf.log | tag tf-on-vs-off
  echo "=== [rep] tp_parity HIP=1 ref, HIP=1 check (cross-process) $(date +%T)"
  env $E ENGINE_GDN_HIP=1 timeout 1200 python tools/tp_parity.py --mode ref --out /tmp/p50_on.pt 2>&1 | tee -a $L/rep.log | tag rep-ref1
  env $E ENGINE_GDN_HIP=1 timeout 1200 python tools/tp_parity.py --mode check --ref /tmp/p50_on.pt 2>&1 | tee -a $L/rep.log | tag rep-cross
  echo "=== [rep] tp_parity repeat (in process) $(date +%T)"
  env $E ENGINE_GDN_HIP=1 timeout 1200 python tools/tp_parity.py --mode repeat 2>&1 | tee -a $L/rep.log | tag rep-in
) &
(
  echo "=== [spec] spec_step_bench ENGINE_GDN_HIP=1 $(date +%T)"
  ENGINE_GDN_HIP=1 HIP_VISIBLE_DEVICES=6,7 timeout 1500 python tools/spec_step_bench.py --devices 6,7 \
    --iters 20 --max-new 128 2>&1 | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]}"
  echo "=== [tests] $(date +%T)"
  HIP_VISIBLE_DEVICES=6 timeout 1500 python -m pytest -q -rf tests/test_gdn_hip.py tests/test_spec_chain.py \
    tests/test_fused_verify.py tests/test_gdn_bank_fused.py tests/test_fused_decode.py tests/test_runahead.py 2>&1 \
    | tail -25 | sed -u "s/^/[tests] /"
  echo "=== [compile] live_compile_probe $(date +%T)"
  ENGINE_GDN_HIP=1 HIP_VISIBLE_DEVICES=6,7 timeout 2400 python tools/live_compile_probe.py --sessions 16 2>&1 \
    | grep -vE "Loading weights|amdgpu.ids|^\s*$" | cut -c1-2000 | tail -12 | sed -u "s/^/[compile] /"
  grep -h "gdn_hip" /tmp/live-compile-server.log 2>/dev/null | head -4 | sed "s/^/[compile-server] /"
) &
wait
echo "=== tracebacks"
for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | tail -30; done
echo "=== done $(date +%T)"
