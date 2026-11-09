#!/bin/bash
# Plan 58 after merging head cead343 (plan 47 IVF drafts): deep chains (up to 7 links) on the IVF
# draft path. One MI250 node, 8 GCDs; must fit the command time limit (~30 min).
# Phase 1, in parallel:
#   GCD 0,1: chain_gate defaults (IVF on, RUNAHEAD=0): spec == plain at 1 and 2 slots for k =
#            1,3,4,5,7 and adaptive, replay-like and natural; step ms per (slots, k) with IVF
#            and its fallback rate; per-depth acceptance.
#   GCD 2,3: chain_gate ENGINE_DRAFT_IVF_AUDIT=1 at k = 3,7 (each IVF draft also scored in full:
#            drafts unlike the full-vocabulary draft, per depth up to 7 links), parity.
#   GCD 4,5: spec_step_bench defaults (run-ahead on): spec == plain; then unit tests.
#   GCD 6,7: live_compile_probe defaults: own-kernel compiles after /health.
# Phase 2: short served A/B off/on/off/on (off = KMAX 3, 32 slots), counters as plan58_gate3.sh.
L=/tmp/p58g4
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
(python -c "import pytest" 2>/dev/null || pip install -q pytest)
F='^=== |RESULT|Traceback|Error|error|passed|failed'
tag() { grep --line-buffered -E "$F" | grep --line-buffered -v "RESULT bitwise " | cut -c1-900 | sed -u "s/^/[$1] /"; }
(
  echo "=== [gate] chain_gate defaults (IVF) $(date +%T)"
  ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=0,1 timeout 1200 python tools/chain_gate.py --devices 0,1 \
    --ks 1,3,4,5,7 --slots 1,2 --adaptive --drafts corpus --iters 50 --pairs 3 --max-new 192 2>&1 \
    | tee -a $L/gate.log | tag gate
  echo "=== [gate] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [audit] chain_gate IVF audit k=3,7 $(date +%T)"
  ENGINE_DRAFT_IVF_AUDIT=1 ENGINE_RUNAHEAD=0 HIP_VISIBLE_DEVICES=2,3 timeout 1200 python tools/chain_gate.py \
    --devices 2,3 --ks 3,7 --slots 1,2 --drafts corpus --iters 10 --pairs 2 --max-new 192 2>&1 \
    | tee -a $L/audit.log | tag audit
  echo "=== [audit] exit ${PIPESTATUS[0]} $(date +%T)"
) &
(
  echo "=== [spec] spec_step_bench defaults $(date +%T)"
  HIP_VISIBLE_DEVICES=4,5 timeout 900 python tools/spec_step_bench.py --devices 4,5 --iters 10 --max-new 192 2>&1 \
    | tee -a $L/spec.log | tag spec
  echo "=== [spec] exit ${PIPESTATUS[0]}"
  HIP_VISIBLE_DEVICES=4 timeout 900 python -m pytest -q -rf tests/test_spec_depth.py tests/test_draft_ivf.py \
    tests/test_spec_chain.py tests/test_runahead.py 2>&1 | tail -12 | sed -u "s/^/[tests] /"
) &
(
  echo "=== [jit] live_compile_probe defaults $(date +%T)"
  HIP_VISIBLE_DEVICES=6,7 timeout 1200 python tools/live_compile_probe.py --max-new 192 2>&1 \
    | grep -E '"kind": "compiles"|Traceback|Error' | cut -c1-1500 | sed -u "s/^/[jit] /"
) &
wait
echo "=== tracebacks (phase 1)"; for f in $L/*.log; do grep -B2 -A20 Traceback $f 2>/dev/null | head -40; done
OFF="ENGINE_SPEC_KMAX=3+ENGINE_SLOTS=32"
ON="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14"
echo "=== [ab] served A/B off/on/off/on $(date +%T)"
timeout 840 python -u tools/topk_epi_ab.py --settings "$OFF,$ON,$OFF,$ON" 2>&1 | tee $L/ab.log \
  | grep -v '^engine:' | grep -v '"kind": "identical"' | grep -v '"event": "setting"' | cut -c1-420
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
python - $L/ab.log <<'PY'
import json, sys
from collections import defaultdict
short = {"ENGINE_SPEC_KMAX=3+ENGINE_SLOTS=32": "off", "ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14": "on"}
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("RESULT ") and '"identical"' in line:
        e = json.loads(line[7:])
        print("IDENTICAL", [short.get(x, x) for x in e["settings"]], f"{e['same']}/{e['requests']}")
        continue
    if not line.startswith("{"):
        continue
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("event") != "setting":
        continue
    h = e.get("health") or {}
    hist, tok, ahead, ch = defaultdict(int), defaultdict(int), defaultdict(int), 0
    for r in h.get("replicas", []):
        for key, v in (r.get("spec_k_hist") or {}).items():
            hist[key] += v
        for key, v in (r.get("spec_k_tokens") or {}).items():
            tok[key] += v
        for key, v in (r.get("spec_ahead") or {}).items():
            ahead[key] += v
        ch += r.get("spec_k_changes", 0)
    by = {key: {"steps": n, "E": round(tok[key] / max(1, n * int(key.split(":")[0])), 3)}
          for key, n in sorted(hist.items(), key=lambda kv: tuple(map(int, kv[0].split(":"))))}
    print("STEPS", json.dumps({"setting": short.get(e["setting"], e["setting"]), "tok_s": e.get("output_tok_per_s"),
                               "tpot_p50": e.get("tpot_ms_p50"), "tpot_p90": e.get("tpot_ms_p90"),
                               "by_requests_k": by, "k_changes": ch, "runahead": dict(ahead),
                               "evictions": h.get("prefix_cache_evictions")}))
PY
echo "=== done $(date +%T)"
