#!/bin/bash
# Plan 68 stage 0 on one MI250 node (8 GCDs): speculative two-chunk pipeline over a layer-split
# pair, feasibility only (tools/pp_spec_probe.py; no engine change).
#
#   bash tools/plan68_stage0.sh
#
# Phase 1, four processes at once, one GCD pair each, output prefixed and streamed live:
#   [tp2]  (a) TP2 chained verify at B=1,2 x k=1,3,5,7: call, GPU replay, class split  GCD 0,1
#   [tp1]  (a) the layer split (ENGINE_TP=1): per stage, overlap, per-op, link, classes GCD 6,7
#   [deep] (b) chain depth 1..15 acceptance, replay-like and natural, IVF on, layer split GCD 2,3
#   [mem]  (c) TP2 + the stages' other halves + second GDN rings, KV pool, free HBM      GCD 4,5
# Phase 2: (d) projection and kill rules from phase 1's RESULT json lines.
# Phase 3: same-pair cross-check, B=1,2 k=3 only: tp1 on GCD 0,1 and tp2 on GCD 6,7.
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

ts "env"
python -c "import torch, triton; print('torch', torch.__version__, 'hip', torch.version.hip, 'triton', triton.__version__, 'gpus', torch.cuda.device_count())" 2>&1 | grep -vE "$F"
rocm-smi --showtopo 2>/dev/null | grep -A10 -i "weight between" | head -12

ts "phase 1"
run tp2 3000 $D/tp2.log python -u tools/pp_spec_probe.py stage --tp 2 --devices 0,1 &
run tp1 3600 $D/tp1.log python -u tools/pp_spec_probe.py stage --tp 1 --devices 6,7 &
run deep 3600 $D/deep.log python -u tools/pp_spec_probe.py deep --devices 2,3 &
run mem 2400 $D/mem.log python -u tools/pp_spec_probe.py mem --devices 4,5 --split 22 &
wait
ts "phase 1 done"

ts "phase 2 projection"
timeout 300 python -u tools/pp_spec_probe.py project $D/tp2.log $D/tp1.log $D/deep.log $D/mem.log 2>&1

ts "phase 3 same-pair cross-check (B=1,2 k=3)"
run tp1x 2400 $D/tp1x.log python -u tools/pp_spec_probe.py stage --tp 1 --devices 0,1 --quick --configs 1:3,2:3 &
run tp2x 2400 $D/tp2x.log python -u tools/pp_spec_probe.py stage --tp 2 --devices 6,7 --quick --configs 1:3,2:3 &
wait
ts "done"
true
