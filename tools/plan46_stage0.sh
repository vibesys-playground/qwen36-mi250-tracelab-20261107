#!/bin/sh
# Plan 46 step 0: where a round's TTFT goes outside its prefill pass, on one MI250 node.
#
#   sh tools/plan46_stage0.sh
#
# 1. CPU: the tokenizer microbenchmark (full vs incremental encode at replay sizes, with the
#    checkpoint's tokenizer) and the exactness tests of the incremental encode.
# 2. Served, 4 x TP2, text prompts growing as the replay's (prompt + output text + new text;
#    48 sessions, 1800 + 300 new, 190 out), per-request stage stamps (ENGINE_REQ_STAMPS=1):
#    concurrency 8: ENGINE_REQ_FAST=0 (head's path) vs 1 (handler-side incremental tokenize);
#    then 1 + a 1 ms GIL switch interval vs 1 + every prompt checked against the full encode;
#    concurrency 4: ENGINE_REQ_FAST=0.
set -x
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
echo "=== tokenizer microbench"
timeout 600 python -u tools/tokenize_bench.py
echo "=== incremental tokenize exactness tests"
timeout 600 python -m pytest -q -rs tests/test_incremental_tokenize.py 2>&1 | tail -5
P="timeout 1500 python -u tools/serve_probe.py --sessions 48 --rounds 6.3 --first 1800 --grow 300 --max-new 190 --corpus synthetic --text"
echo "=== served c8: REQ_FAST 0 vs 1 (stamps)"
$P --concurrency 8 --settings ENGINE_REQ_STAMPS=1+ENGINE_REQ_FAST=0,ENGINE_REQ_STAMPS=1+ENGINE_REQ_FAST=1 2>&1 | grep -v '^engine:'
echo "=== served c8: REQ_FAST 1 + switch 1 ms vs REQ_FAST 1 + tokenize check"
$P --concurrency 8 --settings ENGINE_REQ_STAMPS=1+ENGINE_REQ_FAST=1+ENGINE_SWITCH_INTERVAL_US=1000,ENGINE_REQ_STAMPS=1+ENGINE_REQ_FAST=1+ENGINE_TOKENIZE_CHECK=1 2>&1 | grep -v '^engine:'
echo "=== served c4: REQ_FAST 0 (stamps)"
$P --concurrency 4 --settings ENGINE_REQ_STAMPS=1+ENGINE_REQ_FAST=0 2>&1 | grep -v '^engine:'
for f in /tmp/serve-probe-*.log; do grep -B2 -A20 Traceback "$f" | head -40; done
true
