"""Plan 67: summarize a tools/topk_epi_ab.py log of ENGINE_SPEC_CARRY=0/1 served runs.

    python tools/plan67_summary.py ab.log

Per run: steps, emitted tokens per slot-step and need-capped steps by (requests : k), the
served step price per (bucket : k) (ENGINE_SPEC_CARRY=1 only), the share of 2-request decode
time by k (steps x price: the served price where there is one, else the table), tok/s, TPOT,
TTFT (all rounds and first rounds). Then means per setting, on / off, and identical-token pairs.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import speculative  # noqa: E402

NAMES = {"ENGINE_SPEC_CARRY=0": "off", "ENGINE_SPEC_CARRY=1": "on"}
KEYS = ("output_tok_per_s", "tpot_ms_p50", "tpot_ms_p90", "ttft_ms_p50", "ttft_ms_p90",
        "ttft_first_round_ms_p50", "ttft_first_round_ms_p90")


def name_of(setting: str) -> str:
    return NAMES.get(setting, setting)


def main() -> None:
    runs = defaultdict(list)
    ident = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if line.startswith("RESULT "):
            e = json.loads(line[7:])
            if e.get("kind") == "identical":
                ident.append(e)
            elif e.get("kind") in ("ab", "jit_gate"):
                print(line)
            continue
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") != "setting":
            continue
        name = name_of(e["setting"])
        h = e.get("health") or {}
        hist, tok, capped, ahead = (defaultdict(int) for _ in range(4))
        price: dict[str, list[float]] = defaultdict(list)
        price_n: dict[str, int] = defaultdict(int)
        changes = 0
        for r in h.get("replicas", []) or [h]:
            for src, dst in (("spec_k_hist", hist), ("spec_k_tokens", tok), ("spec_k_capped", capped),
                             ("spec_ahead", ahead)):
                for key, v in (r.get(src) or {}).items():
                    dst[key] += v
            for key, v in (r.get("spec_cost_served") or {}).items():
                price[key].append(v)
            for key, v in (r.get("spec_cost_n") or {}).items():
                price_n[key] += v
            changes += r.get("spec_k_changes", 0)
        served = {key: round(statistics.mean(v), 3) for key, v in price.items()}
        by = {}
        time2: dict[int, float] = {}
        for key, n in sorted(hist.items(), key=lambda kv: tuple(map(int, kv[0].split(":")))):
            req, k = map(int, key.split(":"))
            by[key] = {"steps": n, "E_per_slot_step": round(tok[key] / max(1, n * req), 3),
                       "need_capped": capped.get(key, 0)}
            if req == 2:
                ms = served.get(f"2:{k}", speculative.cost(2, k))
                time2[k] = n * ms
        total2 = sum(time2.values()) or 1.0
        share2 = {k: round(t / total2, 4) for k, t in sorted(time2.items())}
        e["_share2"] = share2
        rnd = len(runs[name])
        runs[name].append(e)
        print("STEPS", json.dumps({
            "setting": name, "round": rnd, "by_requests_k": by, "k_changes": changes,
            "two_request_time_share_by_k": share2,
            "two_request_k3_capped_share": round(capped.get("2:3", 0) / max(1, hist.get("2:3", 0)), 4),
            "served_price_ms": served, "served_price_n": dict(price_n), "runahead": dict(ahead),
            "evictions": h.get("prefix_cache_evictions"), "tail_hits": h.get("prefix_cache_tail_hits"),
            **{k: e.get(k) for k in KEYS}}))
    mean = {}
    for name, es in runs.items():
        mean[name] = {k: round(statistics.mean(x[k] for x in es if x.get(k) is not None), 3)
                      for k in KEYS if any(x.get(k) is not None for x in es)}
        mean[name]["k3_share_of_2req_time"] = round(statistics.mean(x["_share2"].get(3, 0.0) for x in es), 4)
        mean[name]["rounds"] = len(es)
        print("MEAN", json.dumps({"setting": name, **mean[name]}))
    if "off" in mean and "on" in mean:
        print("ON_VS_OFF", json.dumps({k: round(mean["on"][k] / mean["off"][k], 4)
                                       for k in KEYS if k in mean["on"] and k in mean["off"]}))
        tok = mean["on"]["output_tok_per_s"] / mean["off"]["output_tok_per_s"] - 1
        ttft = max(mean["on"][k] / mean["off"][k] - 1 for k in ("ttft_ms_p50", "ttft_ms_p90"))
        k3 = mean["on"]["k3_share_of_2req_time"]
        kill = []
        if tok < 0.02:
            kill.append(f"tok/s +{tok:.2%} < +2.0%")
        if ttft > 0.03:
            kill.append(f"TTFT +{ttft:.2%} > +3%")
        if k3 > 0.35:
            kill.append(f"2-request k=3 time share {k3:.1%} > 35%")
        print("STAGE0", json.dumps({"tok_s_gain": round(tok, 4), "ttft_worst": round(ttft, 4),
                                    "k3_share_on": k3, "verdict": "KILL" if kill else "GO", "why": kill}))
    pairs = defaultdict(list)
    for e in ident:
        a, b = (name_of(x) for x in e["settings"])
        pairs[" vs ".join(sorted([a, b]))].append(f"{e['same']}/{e['requests']}")
    for key, v in sorted(pairs.items()):
        print("IDENTICAL", json.dumps({"pair": key, "results": v}))


if __name__ == "__main__":
    main()
