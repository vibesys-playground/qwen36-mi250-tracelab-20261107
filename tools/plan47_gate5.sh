#!/bin/bash
# Plan 47 on the plan 50 merge (HIP GDN): a short check. One MI250 node.
#   GCD 0,1: spec_step_bench IVF=1 RUNAHEAD=1 (spec == plain)
#   GCD 2,3: chain_gate IVF=1 with the audit (parity, accepted tokens, per-rank fallback)
#   GCD 4,5: chain_gate IVF=0 (accepted tokens off)
#   GCD 6:   unit tests (IVF draft, run-ahead, chained verify, chain glue, GEMV)
L=/tmp/p47e
mkdir -p $L
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
F='^=== |RESULT (step|accept|parity|fatal|error|load|tp_)|RESULT (load_s|step|parity|accept|tp_)|Traceback|IVF drafts|kv_pages'
tag() { grep --line-buffered -E "$F" | sed -u "s/^/[$1] /"; }
(
  ENGINE_DRAFT_IVF=1 ENGINE_RUNAHEAD=1 HIP_VISIBLE_DEVICES=0,1 timeout 1800 python tools/spec_step_bench.py \
    --devices 0,1 --iters 10 --max-new 128 2>&1 | tee $L/spec1.log | tag spec1
  echo "=== [spec1] exit ${PIPESTATUS[0]}"
) &
(
  ENGINE_DRAFT_IVF=1 ENGINE_DRAFT_IVF_AUDIT=1 HIP_VISIBLE_DEVICES=2,3 timeout 2400 python tools/chain_gate.py \
    --devices 2,3 --pairs 3 --iters 30 2>&1 | tee $L/audit.log | tag audit
  echo "=== [audit] exit ${PIPESTATUS[0]}"
) &
(
  ENGINE_DRAFT_IVF=0 HIP_VISIBLE_DEVICES=4,5 timeout 2400 python tools/chain_gate.py \
    --devices 4,5 --pairs 3 --iters 30 2>&1 | tee $L/off.log | tag off
  echo "=== [off] exit ${PIPESTATUS[0]}"
) &
(
  HIP_VISIBLE_DEVICES=6 timeout 1800 python -m pytest -q -rf tests/test_draft_ivf.py tests/test_runahead.py \
    tests/test_spec_chain.py tests/test_fused_gemv.py tests/test_chain_glue.py > $L/tests.log 2>&1
  echo "=== [tests] exit $?"
) &
wait
echo "=== unit tests"; tail -12 $L/tests.log
echo "=== tracebacks"; for f in $L/*.log; do grep -B2 -A25 Traceback $f 2>/dev/null | head -60; done
