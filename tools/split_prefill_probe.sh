#!/bin/sh
# Plan 41 step 0: split a first-round prefill across two GCD pairs (tools/split_prefill_probe.py)
# on GPUs 0-3: pair A (0,1) owner, pair B (2,3) helper. Topology first, then the probe with a
# per-process timeout; output streams as it runs.
#
#   sh tools/split_prefill_probe.sh [probe args...]
echo "=== topology"
timeout 60 rocm-smi --showtopo 2>&1 | head -80 || echo "rocm-smi unavailable"
echo "=== probe"
HIP_VISIBLE_DEVICES=0,1,2,3 ENGINE_PREFILL_GRAPH_BUCKETS=640,896,1280,1792,2048 \
  timeout 2700 python -u tools/split_prefill_probe.py "$@" 2>&1
rc=$?
echo "=== probe exit $rc"
exit $rc
