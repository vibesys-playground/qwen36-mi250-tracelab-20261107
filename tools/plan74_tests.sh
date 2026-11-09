#!/bin/bash
# Plan 74: each listed unit-test file alone, full tracebacks, per-file timeout, threads capped
# per process (OMP_NUM_THREADS / torch threads) so parallel lanes do not oversubscribe the CPUs.
#
#   bash tools/plan74_tests.sh [stage0|verify]
#
# Lanes run in parallel, each on its own GCDs; progress lines are printed live, every lane's
# full log is printed at the end.
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
    echo "=== [$name] start $f $(date +%T)" | tee -a $L/progress
    env HIP_VISIBLE_DEVICES=$dev $extra timeout $TMO $PT tests/$f.py > $L/$name.$f.log 2>&1; rc=$?
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
  ( sleep 5; echo "=== [probe] mtp device map on GCD 2,3 $(date +%T)";
    HIP_VISIBLE_DEVICES=2,3 timeout 600 python tools/plan74_mtp_probe.py > $L/probe.pair.log 2>&1;
    HIP_VISIBLE_DEVICES=4 timeout 600 python tools/plan74_mtp_probe.py > $L/probe.one.log 2>&1;
    echo "=== [probe] done $(date +%T)" ) &
  wait
  # The abort files once more on a pair with kernels serialized, to name the first fault.
  lane pair01-ser 0,1 "AMD_SERIALIZE_KERNEL=3" test_session_tail_cache test_prefill_packed test_fast_model &
  lane pair23-ser 2,3 "AMD_SERIALIZE_KERNEL=3" test_tp &
  wait
else
  lane pair01 0,1 "" $A &
  lane pair23 2,3 "" $B &
  lane one4 4 "" $A &
  lane one5 5 "" $B &
  ( lane pair67 6,7 "" test_gdn; lane one6 6 "" test_gdn ) &
  wait
fi
for f in $L/*.log; do
  echo; echo "################ $f"; grep -vE "Loading weights|amdgpu.ids" $f | cut -c1-2000 | tail -400
done
echo "=== summary"; grep -E "exit [0-9]+ in" $L/progress | sort
echo "=== done $(date +%T)"
