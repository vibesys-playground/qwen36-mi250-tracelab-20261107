#!/bin/sh
# Plan 45 stage 0 (no served-path change): per-slot chained verify "lanes" running
# concurrently on GCDs 0,1 (TP2) against today's batched graph; bitwise checks, hang guard.
#
#   sh tools/lane_probe.sh [extra tools/lane_probe.py args]
#
# The measuring part runs in its own process under `timeout` (tools/lane_probe.py --timeout).
set -x
cd "$(dirname "$0")/.." || exit 1
rocm-smi --showtopo 2>/dev/null | sed -n '1,30p' || true
timeout -k 60 3300 python -u tools/lane_probe.py --part all --timeout 3000 "$@"
