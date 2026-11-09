#!/bin/bash
# Plan 58 stage 1, gate 3: the padding / hysteresis fix (1f1a052) on head 0f17705. One served
# A/B on 8 GCDs, in run order off/on/nodeep/on1/off/on/off/on (three rounds of off and on, one of
# nodeep and on1: twelve runs overran the command job's time limit, j281), then:
#   - mean tok/s, TPOT p50/p90, TTFT per setting over the rounds and on vs off;
#   - per setting: steps and emitted tokens per slot-step by (requests : k), k changes,
#     run-ahead eligibility (all steps and deep steps), KV pages, prefix-cache evictions;
#   - identical tokens for every same-setting pair (cross-process greedy repeatability) and for
#     every off-vs-other pair (spec == plain-depth).
L=/tmp/p58g3
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1
OFF="ENGINE_SPEC_KMAX=3+ENGINE_SLOTS=32"
NODEEP="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14+ENGINE_SPEC_DEEP_MARGIN=1000"
ON="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14"
ON1="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14+ENGINE_SPEC_DEEP_SIZES=1"
S="$OFF,$ON,$NODEEP,$ON1,$OFF,$ON,$OFF,$ON"
echo "=== [ab] served A/B $(date +%T)"
timeout 1680 python -u tools/topk_epi_ab.py --settings "$S" 2>&1 | tee $L/ab.log \
  | grep -v '^engine:' | grep -v '"kind": "identical"' | grep -v '"event": "loop_timers"' | cut -c1-420
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
python - $L/ab.log <<'PY'
import json, re, statistics, sys
from collections import defaultdict
short = {"ENGINE_SPEC_KMAX=3+ENGINE_SLOTS=32": "off",
         "ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14+ENGINE_SPEC_DEEP_MARGIN=1000": "nodeep",
         "ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14": "on",
         "ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14+ENGINE_SPEC_DEEP_SIZES=1": "on1"}
runs = defaultdict(list)
ident = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("RESULT "):
        e = json.loads(line[7:])
        if e.get("kind") == "identical":
            ident.append(e)
        continue
    if not line.startswith("{"):
        continue
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("event") != "setting":
        continue
    name = short.get(e["setting"], e["setting"])
    h = e.get("health") or {}
    agg = {"hist": defaultdict(int), "tok": defaultdict(int), "ahead": defaultdict(int), "changes": 0}
    for r in h.get("replicas", []):
        for key, v in (r.get("spec_k_hist") or {}).items():
            agg["hist"][key] += v
        for key, v in (r.get("spec_k_tokens") or {}).items():
            agg["tok"][key] += v
        for key, v in (r.get("spec_ahead") or {}).items():
            agg["ahead"][key] += v
        agg["changes"] += r.get("spec_k_changes", 0)
    pages = [m.group(1) for x in e.get("engine", []) for m in [re.search(r"kv_pages=(\d+)", x)] if m]
    by = {}
    for key, n in sorted(agg["hist"].items(), key=lambda kv: tuple(map(int, kv[0].split(":")))):
        req, k = map(int, key.split(":"))
        by[key] = {"steps": n, "E_per_slot_step": round(agg["tok"][key] / max(1, n * req), 3)}
    rnd = len(runs[name])
    runs[name].append(e)
    print("STEPS", json.dumps({"setting": name, "round": rnd, "by_requests_k": by, "k_changes": agg["changes"],
                               "runahead": dict(agg["ahead"]), "kv_pages": pages,
                               "evictions": h.get("prefix_cache_evictions"), "cached_tokens": h.get("cached_tokens"),
                               "tok_s": e.get("output_tok_per_s"), "tpot_p50": e.get("tpot_ms_p50"),
                               "tpot_p90": e.get("tpot_ms_p90")}))
mean = {}
for name, es in runs.items():
    mean[name] = {k: round(statistics.mean(x[k] for x in es if x.get(k) is not None), 3)
                  for k in ("output_tok_per_s", "tpot_ms_p50", "tpot_ms_p90", "ttft_ms_p50", "ttft_ms_p90")}
    mean[name]["rounds"] = len(es)
    print("MEAN", json.dumps({"setting": name, **mean[name]}))
if "off" in mean:
    for name in mean:
        if name == "off":
            continue
        print("VS_OFF", json.dumps({"setting": name, **{k: round(mean[name][k] / mean["off"][k], 4)
                                                       for k in ("output_tok_per_s", "tpot_ms_p50", "tpot_ms_p90",
                                                                 "ttft_ms_p50", "ttft_ms_p90")}}))
pairs = defaultdict(list)
for e in ident:
    a, b = (short.get(x, x) for x in e["settings"])
    pairs[" vs ".join(sorted([a, b]))].append(f"{e['same']}/{e['requests']}")
for key, v in sorted(pairs.items()):
    print("IDENTICAL", json.dumps({"pair": key, "results": v}))
PY
for f in /tmp/serve-probe-ENGINE_SPEC_KMAX_*.log; do
  echo "--- $f"; grep -E "Traceback|Error" $f | head -5
done
echo "=== done $(date +%T)"
