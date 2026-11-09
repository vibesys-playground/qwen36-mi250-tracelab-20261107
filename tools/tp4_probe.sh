#!/bin/sh
# Plan 43 stage 0 (no served-path change): TP4 chained verify step on GCDs 0-3 against today's
# TP2 step on GCDs 0,1, parity, in-graph 4-way all-reduce, quarter-width per-op times.
#
#   sh tools/tp4_probe.sh [extra tools/tp4_probe.py args]
#
# Each part runs in its own process under `timeout` (tools/tp4_probe.py --timeout, seconds).
set -x
cd "$(dirname "$0")/.." || exit 1
rocm-smi --showtopo 2>/dev/null | sed -n '1,40p' || true
timeout -k 60 6000 python tools/tp4_probe.py --part all --timeout 2700 "$@"
