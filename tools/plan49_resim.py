"""Plan 49: the planner's event simulation of the replay, fed with a measured migration cost.

    python tools/plan49_resim.py .vibesys/tasks/mi2508x-tracelab/recorded_sessions.csv \
        --cost-ms 12.3 --stall-ms 4.5 [--cost-ms ... --stall-ms ...]

The model is the planner's (plan 49; scratch copy plan49-sim.py), unchanged: 8 closed-loop
sessions over the recorded rounds, 4 serial replicas, a prefill pass of
30 ms + 0.0725 ms/token, a verify step of 11.8 / 13.9 / +2.2 ms per extra slot, 3.3 tokens
per step; round 0 goes to the replica with the fewest in-flight requests; later rounds stay
home (``home``) or go to the fewest-in-flight replica, ties home (``free``). A move adds
``cost`` to the target's pass (the planner's term). Added here: ``stall``, time the move
holds the home replica (its export) before its next pass or step.

Prints one ``RESULT`` line per configuration: tok/s, TTFT p50/p90, TPOT p50/p90 (ms) and the
TPOT p90 change against ``home``.
"""

from __future__ import annotations

import argparse
import csv
import json


def load(path: str) -> dict[int, list[dict]]:
    sessions: dict[int, list[dict]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sessions.setdefault(int(r["session_id"]), []).append(r)
    for s in sessions.values():
        s.sort(key=lambda r: int(r["round_idx"]))
    return sessions


def P(n: int) -> float:
    return 0.030 + 0.0000725 * n


def step(b: int) -> float:
    return {1: 0.0118, 2: 0.0139}.get(b, 0.0139 + 0.0022 * (b - 2))


TPS = 3.3


def run(S, policy: str, mig: float = 0.008, stall: float = 0.0):
    dt = 0.0005
    t = 0.0
    R = 4
    queue = sorted(S)
    home: dict[int, int] = {}
    state: dict[int, dict] = {}
    ttft, tpot = [], []
    out = 0
    moves = rounds = 0
    pending = [[] for _ in range(R)]
    decoding = [set() for _ in range(R)]
    for _ in range(8):
        state[queue.pop(0)] = dict(r=0, phase="wait", ready=0.0)
    rep_next = [0.0] * R
    while state or queue:
        for sid, s in list(state.items()):
            if s["phase"] == "wait" and s["ready"] <= t:
                rr = S[sid][s["r"]]
                n = int(rr["input_len"])
                cnt = [len(decoding[i]) + len(pending[i]) for i in range(R)]
                cost = 0.0
                if s["r"] == 0 or policy == "free":
                    h = home.get(sid)
                    if s["r"] == 0 or h is None:
                        rep = min(range(R), key=lambda i: (cnt[i], i))
                    else:
                        rounds += 1
                        rep = min(range(R), key=lambda i: (cnt[i], i != h))
                        if rep != h:
                            moves += 1
                            cost = mig
                            # The export holds the home replica before its next pass or step.
                            rep_next[h] = max(rep_next[h], t) + stall
                else:
                    rounds += 1
                    rep = home[sid]
                home[sid] = rep
                s.update(phase="q", rep=rep, t0=t, n=n, cost=cost, out=int(rr["output_len"]))
                pending[rep].append(sid)
        for i in range(R):
            if rep_next[i] > t:
                continue
            if pending[i]:
                sid = pending[i].pop(0)
                s = state[sid]
                d = P(s["n"]) + s["cost"]
                rep_next[i] = t + d
                s.update(phase="dec", rem=s["out"] - 1, first=t + d)
                ttft.append(t + d - s["t0"])
                decoding[i].add(sid)
                s["start_at"] = t + d
            elif decoding[i]:
                act = [x for x in decoding[i] if state[x]["start_at"] <= t]
                if not act:
                    continue
                d = step(len(act))
                rep_next[i] = t + d
                for x in act:
                    s = state[x]
                    s["rem"] -= TPS
                    if s["rem"] <= 0:
                        end = t + d
                        out += s["out"]
                        if s["out"] > 1:
                            tpot.append((end - s["first"]) / (s["out"] - 1))
                        decoding[i].discard(x)
                        s["r"] += 1
                        rr = S[x][s["r"] - 1]
                        if s["r"] >= len(S[x]):
                            del state[x]
                            if queue:
                                state[queue.pop(0)] = dict(r=0, phase="wait", ready=end)
                        else:
                            s.update(phase="wait", ready=end + int(rr["tool_wait_after_ms"]) / 1000)
        t += dt

    def q(a, p):
        return sorted(a)[int(p * len(a))]

    return {
        "tok_s": round(out / t, 1), "ttft_p50": round(q(ttft, 0.5) * 1e3, 2),
        "ttft_p90": round(q(ttft, 0.9) * 1e3, 2), "tpot_p50": round(q(tpot, 0.5) * 1e3, 3),
        "tpot_p90": round(q(tpot, 0.9) * 1e3, 3), "moves": moves, "continuing_rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--cost-ms", type=float, action="append", default=[])
    parser.add_argument("--stall-ms", type=float, action="append", default=[])
    parser.add_argument("--label", action="append", default=[])
    args = parser.parse_args()
    S = load(args.csv)
    base = run(S, "home")
    print("RESULT " + json.dumps({"kind": "sim", "policy": "home", **base}), flush=True)
    planner = run(S, "free", mig=0.008)
    print("RESULT " + json.dumps({"kind": "sim", "policy": "free", "label": "planner 8 ms, no stall",
                                  "cost_ms": 8.0, "stall_ms": 0.0, **planner,
                                  "tpot_p90_change": round(planner["tpot_p90"] / base["tpot_p90"] - 1, 4)}), flush=True)
    for i, cost in enumerate(args.cost_ms):
        stall = args.stall_ms[i] if i < len(args.stall_ms) else 0.0
        label = args.label[i] if i < len(args.label) else ""
        r = run(S, "free", mig=cost / 1e3, stall=stall / 1e3)
        print("RESULT " + json.dumps({"kind": "sim", "policy": "free", "label": label, "cost_ms": cost,
                                      "stall_ms": stall, **r,
                                      "tpot_p90_change": round(r["tpot_p90"] / base["tpot_p90"] - 1, 4)}), flush=True)


if __name__ == "__main__":
    main()
