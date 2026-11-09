#!/bin/sh
# Plan 51: the GPU unit tests one file at a time (verbose, Python fault handler on), so a crash
# names its test: the new chain glue / AR K-sum tests on two GCDs (the LL all-reduce needs a
# pair), the chained verify and run-ahead tests on one (as tools/runahead_gate.sh runs them).
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F="_pytest|pluggy|Extension modules|^\s*$|amdgpu.ids"
run() {
  devs=$1; shift
  echo "=== HIP_VISIBLE_DEVICES=$devs $*"
  HIP_VISIBLE_DEVICES=$devs timeout 1500 python -X faulthandler -m pytest -v -rf -p no:cacheprovider "$@" 2>&1 \
    | grep -vE "$F" | cut -c1-400 | tail -70
}
run 6,7 tests/test_chain_glue.py
run 6 tests/test_chain_glue.py -k "not allreduce"
run 6 tests/test_spec_chain.py
run 6 tests/test_runahead.py
true
