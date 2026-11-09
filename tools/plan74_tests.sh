#!/bin/bash
# Plan 74: each listed unit-test file alone, full tracebacks, a timeout per file, threads capped
# per process (OMP_NUM_THREADS) so the parallel lanes do not oversubscribe the CPUs.
#
#   bash tools/plan74_tests.sh stage0     # the 8 files, pair + one GCD (+ abort files serialized)
#   PROBES=1 REAL=1 REPEAT=5 bash tools/plan74_tests.sh verify
#
# verify: the 8 files on a GCD pair (0,1 / 2,3) and on one GCD (4 / 5), test_gdn on 6,7 and 6;
# PROBES=1 then runs tools/plan74_tp_probe.py variants on 6,7; REAL=1 then runs, on the real
# model, spec == plain (tools/spec_step_bench.py, GCD 0,1) and the Triton compiles after
# /health under a replay-like load (tools/live_compile_probe.py, GCD 4,5). REPEAT=n then reruns
# test_session_tail_cache n more times on one GCD (4) and on a pair (6,7): its served
# warm-versus-cold token checks run concurrent sessions, so a flake shows up here.
# Progress is printed live; a failing file's whole log is printed at the end, a passing one's
# summary.
MODE=${1:-stage0}
L=/tmp/p74
mkdir -p $L
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${OMP_NUM_THREADS:-12}
unset ENGINE_TP ENGINE_FAST
(python -c "import pytest" 2>/dev/null || pip install -q pytest) > /dev/null 2>&1
PT="python -X faulthandler -m pytest -q -rfEs -p no:cacheprovider --tb=long --durations=10"
TMO=${TMO:-900}
A="test_router test_session_tail_cache test_prefill_packed test_fast_model"
B="test_spec_chain test_speculative test_tp"
lane() {  # name, devices, extra env, files...
  local name=$1 dev=$2 extra=$3; shift 3
  for f in "$@"; do
    local t0=$(date +%s)
    echo "=== [$name] start $f $(date +%T)"
    env HIP_VISIBLE_DEVICES=$dev $extra timeout $TMO $PT tests/$f.py > $L/$name.$f.log 2>&1; rc=$?
    echo "$rc" > $L/$name.$f.rc
    echo "=== [$name] $f exit $rc in $(( $(date +%s) - t0 ))s: $(grep -E '(passed|failed|error|skipped)' $L/$name.$f.log | tail -1)" | tee -a $L/progress
  done
}
echo "=== start $MODE $(date +%T) cpus $(nproc) omp $OMP_NUM_THREADS"
if [ "$MODE" = stage0 ]; then
  lane pair01 0,1 "" $A &
  lane pair23 2,3 "" $B &
  lane one4 4 "" $A &
  lane one5 5 "" $B &
  ( lane pair67 6,7 "" test_gdn; lane one6 6 "" test_gdn ) &
  wait
  lane pair01-ser 0,1 "AMD_SERIALIZE_KERNEL=3" test_session_tail_cache test_prefill_packed test_fast_model &
  lane pair23-ser 2,3 "AMD_SERIALIZE_KERNEL=3" test_tp &
  wait
  for f in $L/*.log; do echo; echo "################ $f"; grep -vE "Loading weights|amdgpu.ids" $f | cut -c1-2000 | tail -400; done
  echo "=== summary"; sort $L/progress; echo "=== done $(date +%T)"; exit 0
fi
lane pair01 0,1 "" $A &
lane pair23 2,3 "" $B &
lane one4 4 "" $A &
lane one5 5 "" $B &
(
  lane pair67 6,7 "" test_gdn; lane one6 6 "" test_gdn
  if [ -n "$PROBES" ]; then
    for v in "4096|" "4096|AMD_SERIALIZE_KERNEL=3" "1025|AMD_SERIALIZE_KERNEL=3" \
             "4096|AMD_SERIALIZE_KERNEL=3 ENGINE_PREFILL_GRAPH_BUCKETS=64,128,256" \
             "4096|AMD_SERIALIZE_KERNEL=3 ENGINE_PREFILL_GRAPH=0"; do
      ml=${v%%|*}; ex=${v#*|}
      echo "=== [probe] tp max_len=$ml env='$ex' $(date +%T)"
      env HIP_VISIBLE_DEVICES=6,7 $ex timeout 600 python tools/plan74_tp_probe.py tp $ml 2>&1 \
        | grep -vE "Loading weights|amdgpu.ids|Writing model" | cut -c1-600 | tail -80
      echo "=== [probe] rc ${PIPESTATUS[0]}"
    done
    echo "=== [probe] margin (Transformers path, batched vs solo) GCD 6 $(date +%T)"
    HIP_VISIBLE_DEVICES=6 timeout 600 python tools/plan74_tp_probe.py margin 2>&1 | grep -vE "Loading weights|amdgpu.ids|Writing model" | tail -30
  fi
) > $L/probes.out 2>&1 &
wait
cat $L/probes.out
if [ -n "$REAL" ]; then
  echo "=== real model $(date +%T)"
  ( HIP_VISIBLE_DEVICES=0,1 timeout 2400 python tools/spec_step_bench.py --devices 0,1 --iters 10 --max-new 192 2>&1 \
      | grep -E "RESULT|engine:|Traceback|Error" | cut -c1-400 ) > $L/real.spec.out 2>&1 &
  ( HIP_VISIBLE_DEVICES=4,5 timeout 2400 python tools/live_compile_probe.py 2>&1 \
      | grep -vE "Loading weights|amdgpu.ids" | cut -c1-3000 | tail -60; echo "live_compile_probe rc ${PIPESTATUS[0]}" ) > $L/real.compile.out 2>&1 &
  wait
  echo "=== [real] spec == plain (GCD 0,1)"; cat $L/real.spec.out
  echo "=== [real] compiles after /health (GCD 4,5)"; cat $L/real.compile.out
fi
if [ -n "$REPEAT" ]; then
  echo "=== repeats $(date +%T)"
  ( for i in $(seq $REPEAT); do lane one4-r$i 4 "" test_session_tail_cache; done ) &
  ( for i in $(seq $REPEAT); do lane pair67-r$i 6,7 "" test_session_tail_cache; done ) &
  wait
fi
for f in $L/*.test_*.log; do
  b=${f%.log}
  if [ "$(cat $b.rc)" != 0 ]; then
    echo; echo "################ FAILED $f"; grep -vE "Loading weights|amdgpu.ids" $f | cut -c1-1500
  else
    echo "################ ok $f: $(grep -E '(passed|failed|error)' $f | tail -1)"
  fi
done
echo "=== summary"; sort $L/progress
echo "=== done $(date +%T)"
