#!/bin/sh
# Plans 38 (bucketed MoE prefill tiles) + 35 (flash prefill attention) merged on head f4b2962
# (plan 36 host placement over plan 33's captured prefill), on one MI250 node, 8 GCDs.
#
#   sh tools/combine_k_check.sh
#
# Phase A, in parallel on separate GCD pairs (TP2 replicas, the server's default):
#   GCD 0,1  teacher-forced top-1 (tools/tp_parity.py, 8 prompts, prefill graphs on):
#            reference with ENGINE_PREFILL_FLASH=0 (the head's attention kernel; the MoE block
#            is bitwise the head's, see phase B), then FLASH=1 (as served) against it
#            (gate >= 98.5%), then FLASH=0 from another process (the floor)
#   GCD 2,3  greedy repeatability as served: in one process, then across 2 processes (8/8)
#   GCD 4    GPU unit tests of both plans; then GCD 4,5: a served replay-like load with every
#            Triton compile logged (tools/live_compile_probe.py), this commit, then the head
#            f4b2962's tree if git can produce it, to see which compiles the head shows too
#   GCD 6,7  spec == plain greedy, 8 sequences (tools/spec_step_bench.py)
# Phase B on GCD 0,1: this commit's prefill MoE block against 0d89b27's (== f4b2962's; plan 36
#            changes no kernel) bitwise, eager per call and last-token logits through the
#            captured prefill graphs (tools/moe_head_bitwise.py), as served (FLASH=1).
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
L=/tmp/ck-logs
mkdir -p $L
P=tools/tp_parity.py
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=0,1"; \
  env $E ENGINE_PREFILL_FLASH=0 timeout 1800 python $P --mode ref --out /tmp/ck_ref0.pt \
  && echo "--- FLASH=1 (served) against the FLASH=0 reference (head attention, head-bitwise MoE)" \
  && env $E ENGINE_PREFILL_FLASH=1 timeout 1800 python $P --mode check --ref /tmp/ck_ref0.pt \
  && echo "--- FLASH=0 against the FLASH=0 reference (another process: the floor)" \
  && env $E ENGINE_PREFILL_FLASH=0 timeout 1800 python $P --mode check --ref /tmp/ck_ref0.pt ; \
  echo "--- moe_head_bitwise (FLASH=1)"; \
  env $E timeout 3000 python tools/moe_head_bitwise.py --news 61,561,1650 ) > $L/parity.log 2>&1 &
( E="ENGINE_TP=2 HIP_VISIBLE_DEVICES=2,3"; \
  echo "--- repeat in one process"; env $E timeout 1800 python $P --mode repeat --out /tmp/ck_rep.pt; \
  echo "--- ref (process 1)"; env $E timeout 1800 python $P --mode ref --out /tmp/ck_a.pt \
  && echo "--- check against it (process 2)" \
  && env $E timeout 1800 python $P --mode check --ref /tmp/ck_a.pt ) > $L/repeat.log 2>&1 &
( HIP_VISIBLE_DEVICES=4 timeout 1800 python -m pytest -q -rf tests/test_moe_prefill.py tests/test_prefill_kernels.py \
    tests/test_flash_prefill.py tests/test_cpu_place.py 2>&1 | tail -25; \
  echo "--- live compiles, this commit"; \
  HIP_VISIBLE_DEVICES=4,5 timeout 3000 python tools/live_compile_probe.py; \
  cp /tmp/live-compile-server.log $L/live-server-this.log; \
  H=/tmp/head-f4b2962; rm -rf $H; mkdir -p $H; \
  if git archive f4b296276fe1ecb84303a5a1227d04a0b262d0a6 2>/dev/null | tar -x -C $H; then \
    cp -r tools/live_compile_probe.py tools/serve_probe.py tools/compile_hook $H/tools/; \
    echo "--- live compiles, head f4b2962 (same probe, head's engine)"; \
    HIP_VISIBLE_DEVICES=4,5 timeout 3000 python $H/tools/live_compile_probe.py --port 18534; \
  else echo "--- head f4b2962 tree unavailable (no git objects in this checkout)"; fi ) > $L/tests.log 2>&1 &
( timeout 2400 python tools/spec_step_bench.py --devices 6,7 --iters 10 --max-new 128 ) > $L/spec.log 2>&1 &
wait

echo "=== top-1 vs FLASH=0 reference, floor, MoE bitwise vs head"; grep -E "RESULT|Error|error|^---|engine: prefill" $L/parity.log | cut -c1-700 | tail -45
echo "=== repeatability (one process, then across 2 processes)"; grep -E "RESULT|Error|error|^---|engine: prefill" $L/repeat.log | tail -20
echo "=== unit tests + live compiles"; grep -vE "Loading weights|amdgpu.ids|^\s*$" $L/tests.log | cut -c1-3000 | tail -70
echo "=== spec == plain"; grep -E "RESULT|Error|Traceback|engine: prefill" $L/spec.log | tail -30
echo "=== tracebacks"; grep -l Traceback $L/*.log; for f in $L/*.log; do grep -B2 -A25 Traceback $f | head -60; done
true
