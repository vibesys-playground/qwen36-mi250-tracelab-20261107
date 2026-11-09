#!/bin/bash
# Plan 58 stage 1, gate 2 (diagnosis of j271's served A/B: on was 2-3% slower in tok/s): one
# served A/B on 8 GCDs with the loop timers on in every setting, two rounds of
#   off    : ENGINE_SPEC_KMAX=3, ENGINE_SLOTS=32 (plan 31's depths, ring of 4)
#   nodeep : KMAX 7, SLOTS 14, ENGINE_SPEC_DEEP_MARGIN=1000 (the ring of 8, k <= 3 always)
#   on     : KMAX 7, SLOTS 14 (the defaults: deep k at 1 and 2 requests)
#   on1    : the defaults with ENGINE_SPEC_DEEP_SIZES=1 (deep k at 1 request only)
# Prints tok/s, TTFT, TPOT, identical tokens, compiles after /health, then per setting: KV pages,
# evictions, steps by (requests : k), run-ahead eligibility counts, and the loop timers of
# steady-state steps by (batch, k): host gap, GPU step and idle, tokens per step.
L=/tmp/p58g2
mkdir -p $L
export ENGINE_TP=2 ENGINE_MTP=1 PYTHONUNBUFFERED=1 ENGINE_LOOP_TIMERS=1
OFF="ENGINE_SPEC_KMAX=3+ENGINE_SLOTS=32"
NODEEP="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14+ENGINE_SPEC_DEEP_MARGIN=1000"
ON="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14"
ON1="ENGINE_SPEC_KMAX=7+ENGINE_SLOTS=14+ENGINE_SPEC_DEEP_SIZES=1"
echo "=== [ab] served A/B $(date +%T)"
timeout 7200 python -u tools/topk_epi_ab.py --settings "$OFF,$NODEEP,$ON,$ON1,$OFF,$NODEEP,$ON,$ON1" 2>&1 \
  | tee $L/ab.log | grep -v '^engine:' | grep -v '"event": "setting"' | grep -v '"event": "loop_timers"'
echo "=== [ab] exit ${PIPESTATUS[0]} $(date +%T)"
python - $L/ab.log <<'PY'
import json, re, sys
timers = {}
run = -1
names = []
for line in open(sys.argv[1]):
    if not line.startswith("{"):
        continue
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("event") == "loop_timers":
        key = (e["setting"], len([n for n in names if n == e["setting"]]))
        timers.setdefault(key, []).append(e)
        continue
    if e.get("event") != "setting":
        continue
    rnd = len([n for n in names if n == e["setting"]])
    names.append(e["setting"])
    h = e.get("health") or {}
    pages = [m.group(1) for x in e.get("engine", []) for m in [re.search(r"kv_pages=(\d+)", x)] if m]
    hist, ahead = {}, {}
    for r in h.get("replicas", []):
        for key, v in (r.get("spec_k_hist") or {}).items():
            hist[key] = hist.get(key, 0) + v
        for key, v in (r.get("spec_ahead") or {}).items():
            ahead[key] = ahead.get(key, 0) + v
    by = {}
    for key, v in hist.items():
        n, k = key.split(":")
        by.setdefault(int(n), {})[f"k{k}"] = v
    keep = {k: h.get(k) for k in ("prefix_cache_evictions", "cached_tokens", "largest_batch",
                                  "mtp_verify_steps", "mtp_proposed", "mtp_accepted")}
    print("CAPACITY", json.dumps({"setting": e["setting"], "round": rnd, "kv_pages": pages, **keep,
                                  "tok_s": e.get("output_tok_per_s"), "tpot_p50": e.get("tpot_ms_p50"),
                                  "tpot_p90": e.get("tpot_ms_p90"), "ttft_p90": e.get("ttft_ms_p90")}))
    print("KHIST", json.dumps({"setting": e["setting"], "round": rnd, "steps_by_requests": dict(sorted(by.items())),
                               "runahead": ahead}))
for (setting, rnd), evs in timers.items():
    agg = {}
    for ev in evs:
        for key, ent in ev.items():
            if not isinstance(ent, dict) or not re.match(r"pure_b[12]_k\d", key):
                continue
            a = agg.setdefault(key, {"n": 0})
            n = ent.get("n", 0)
            a["n"] += n
            for stage in ("gap", "gpu_step", "gpu_idle", "wait", "choosek", "book", "adv"):
                if stage in ent:
                    a[stage] = a.get(stage, 0.0) + ent[stage][2] * n  # mean x n
            a["tok"] = a.get("tok", 0.0) + ent.get("tokens_p50", 0) * n
    out = {}
    for key, a in sorted(agg.items()):
        n = max(1, a["n"])
        out[key] = {"n": a["n"], **{s: round(a[s] / n, 3) for s in a if s not in ("n",)}}
    print("TIMERS", json.dumps({"setting": setting, "round": rnd, "steady_steps_ms_mean": out}))
PY
for f in /tmp/serve-probe-ENGINE_SPEC_KMAX_*.log; do
  echo "--- $f"; grep -E "Traceback|Error" $f | head -5
done
echo "=== done $(date +%T)"
