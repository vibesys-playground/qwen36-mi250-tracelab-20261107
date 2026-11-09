#!/bin/bash
# Plan 68 stage 0 on one MI250 node (8 GCDs): speculative two-chunk pipeline over a layer-split
# pair, feasibility only (tools/pp_spec_probe.py; no engine change).
#
#   bash tools/plan68_stage0.sh
#
# Run 2 (after j319: the per-op capture died at B=1 k=5 and took the rest of the layer-split
# configs with it; the dual-layout load stopped at KV sizing under the 6 GB reserve; the
# layer split's head looked dominated by full-vocabulary draft fallbacks).
# Phase 1, four processes, one GCD pair each, output prefixed and streamed live:
#   [tp1]   (a) layer split, every (B, k), class split, IVF fallback rate             GCD 0,1
#   [tp1p]  (a) layer split per-op graphs and the real-input link, B=1,2 x k=1,3     GCD 6,7
#   [tp1c]  (a) layer split with ENGINE_DRAFT_IVF_CLUSTERS=8192 (the index of one GCD
#               at TP2's rows per cluster), every (B, k), IVF fallback rate             GCD 2,3
#   [tp2]   (a) TP2 every (B, k), IVF fallback rate (same job)                          GCD 4,5
# Phase 2: [mem] (c) dual layout with ENGINE_RESERVE_GB=2 so the pool sizes (GCD 0,1);
#          [tp1cp] the 8192-cluster layer split's per-op graphs and link (GCD 2,3).
# Phase 3: (d) projection for both layer-split variants (with j319's acceptance, offline too).
export PYTHONUNBUFFERED=1
F="Loading weights|amdgpu.ids|^\s*$|it/s\]|ROCTracer produced|Warning: Tensor|STAGE:"
D=/tmp/p68
mkdir -p $D
ts() { echo "=== $1 $(date +%T)"; }
run() {  # tag timeout_s logfile cmd...
  local tag=$1 to=$2 logf=$3; shift 3
  ( timeout "$to" "$@" 2>&1; echo "exit $?" ) | grep --line-buffered -vE "$F" | cut -c1-6000 \
    | tee "$logf" | sed -u "s/^/[$tag] /"
}
ALL=1:1,1:3,1:5,1:7,2:1,2:3,2:5,2:7

ts "phase 1"
run tp1 2400 $D/tp1.log python -u tools/pp_spec_probe.py stage --tp 1 --devices 0,1 --configs $ALL &
run tp1p 2400 $D/tp1p.log python -u tools/pp_spec_probe.py stage --tp 1 --devices 6,7 --quick \
  --configs 1:1,1:3,2:1,2:3 --perop 1:1,1:3,2:1,2:3 &
run tp1c 2400 $D/tp1c.log env ENGINE_DRAFT_IVF_CLUSTERS=8192 P68_VARIANT=c8192 \
  python -u tools/pp_spec_probe.py stage --tp 1 --devices 2,3 --configs $ALL &
run tp2 2400 $D/tp2.log python -u tools/pp_spec_probe.py stage --tp 2 --devices 4,5 --quick --configs $ALL &
wait
ts "phase 2"
run mem 2400 $D/mem.log env ENGINE_RESERVE_GB=2 python -u tools/pp_spec_probe.py mem --devices 0,1 --split 22 &
run tp1cp 2400 $D/tp1cp.log env ENGINE_DRAFT_IVF_CLUSTERS=8192 P68_VARIANT=c8192 \
  python -u tools/pp_spec_probe.py stage --tp 1 --devices 2,3 --quick --configs 1:1,1:3,2:1,2:3 --perop 1:1,1:3,2:1,2:3 &
wait
ts "phase 3 projection (acceptance comes from j319's deep run: pasted offline)"
for v in base c8192; do
  timeout 300 python -u tools/pp_spec_probe.py project --variant $v $D/tp2.log $D/tp1.log $D/tp1p.log \
    $D/tp1c.log $D/tp1cp.log $D/mem.log 2>&1 | grep -vE "corpus="
done
ts "done"
true
