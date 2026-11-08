"""Plan 26: compare fla's autotune picks and greedy outputs across processes; write pins.

    python tools/fla_pins_pick.py --snapshots a.json b.json ... [--refs a.pt b.pt ...] --out pins.json

``--snapshots``: ``tools/tp_parity.py --dump-fla`` outputs of processes that autotuned on their
own (FLA_CACHE_RESULTS=0). Prints, per kernel and tuned key, whether every process picked the
same config. Writes the pins (``engine/fast/fla_pin``): per kernel, the config picked most
often over all processes and keys (ties to the smallest string, so the choice is fixed).
``--refs``: those processes' ``tp_parity --mode ref`` records; prints per prompt the number
of identical greedy picks of each against the first.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="*", default=[])
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    snaps = [json.loads(Path(p).read_text()) for p in args.snapshots]
    names = sorted(set().union(*snaps)) if snaps else []
    pins = {}
    differ = 0
    for name in names:
        votes: Counter = Counter()
        keys = sorted(set().union(*(s.get(name, {}).get("picked", {}).keys() for s in snaps)))
        for key in keys:
            picks = [s.get(name, {}).get("picked", {}).get(key) for s in snaps]
            for p in picks:
                if p is not None:
                    votes[p] += 1
            same = len({p for p in picks if p is not None}) <= 1
            differ += not same
            print(f"RESULT fla_pick kernel={name} key={key} same={same} picks={picks}", flush=True)
        if votes:
            top = max(votes.values())
            pins[name] = min(c for c, n in votes.items() if n == top)
    if args.out and snaps:
        Path(args.out).write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
        print(f"RESULT fla_pins kernels={len(pins)} keys_with_differing_picks={differ} written={args.out}",
              flush=True)
        print("FLA_PINS_JSON " + json.dumps(pins, sort_keys=True), flush=True)
    if len(args.refs) >= 2:
        import torch

        recs = [torch.load(p) for p in args.refs]
        base = recs[0]["picks"]
        for path, rec in zip(args.refs[1:], recs[1:]):
            same = []
            for k in sorted(base):
                a, b = base[k], rec["picks"][k]
                n = 0
                while n < len(a) and a[n] == b[n]:
                    n += 1
                same.append(n)
            print(f"RESULT greedy_vs_first {path} identical_prefix={same} of {len(base[0])}", flush=True)


if __name__ == "__main__":
    main()
