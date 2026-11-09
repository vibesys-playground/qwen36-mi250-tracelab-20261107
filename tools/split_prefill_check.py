"""Plan 41 stage 2: numerics and wall of the served split prefill, in one pair-host process.

    HIP_VISIBLE_DEVICES=0,1,2,3 python -u tools/split_prefill_check.py [--lens 1500,...] [--reps 5]

Loads two pairs exactly as a split-prefill server process does (``engine.pair_host.load``:
pair A on GPUs 0,1, pair B on 2,3, both ``Generator``s, the chains, the warm split), then for
each prompt (replay-like synthetic coding lines and Transformers source text, ``--lens``
tokens, uncached) and each owner (A, then B):

- single: the served one-pair pass (``TPModel.prefill_packed``, captured graph of the padded
  size, eager above 2,048), last-token logits and MTP draft;
- split: the same request through the pair's ``SplitFast`` (owner half 1, helper half 2);
- floor: where a larger captured size exists, the single pass replayed at that padded size
  (``_pg_run``), the size-only perturbation the split is measured against (profile j203).

Per prompt: top-1 and draft agreement split vs single, max |logit diff| split vs single and
floor vs single, the median walls (``--reps``) of single and split, and the owner's slot
state after split vs single. Prints RESULT lines and a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="1500,1580,1650,1720,1790,1850,1930,2010,2100,2250,2400,2700,3100,3600")
    ap.add_argument("--natural", default="1550,1800,2300")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    import torch

    from engine import pair_host, split_prefill
    from engine.generator import CHECKPOINT
    from serve_probe import token_pool

    t0 = time.perf_counter()
    load_args = SimpleNamespace(
        model=CHECKPOINT["model"], experts_implementation=None, max_model_len=262144,
        prefill_chunk_tokens=2048, prefix_cache_tokens=4194304, prefix_cache_entries=128,
        prefix_cache_gib=16.0,
    )
    generators, _ = pair_host.load(load_args, CHECKPOINT["revision"])
    result("loaded", seconds=round(time.perf_counter() - t0, 1),
           chains=[sorted(g.fast._pair.chains.graphs) for g in generators],
           chain_errors=[g.fast._pair.chains.error for g in generators],
           reserved_gib=[round(torch.cuda.memory_reserved(d) / 2**30, 2) for d in range(4)])
    lens = [int(x) for x in args.lens.split(",") if x]
    nat = [int(x) for x in args.natural.split(",") if x]
    synth = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 3 * max(lens) + 100, "synthetic")
    source = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 3 * max(nat + [1]) + 100, "source")
    prompts = [("synthetic", n, synth[n % 997: n % 997 + n]) for n in lens]
    prompts += [("source", n, source[n: 2 * n]) for n in nat]

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def state(tp, slot: int, n: int) -> dict:
        out = {}
        for r, m in enumerate(tp.ranks):
            row = m._g(slot)
            for i, (conv, rec) in m.gdn_state.items():
                out[f"r{r}.{i}.rec"] = rec[row].float().cpu()
                out[f"r{r}.{i}.conv"] = conv[row].float().cpu()
            table = torch.tensor(m.slot_pages[slot], dtype=torch.long)
            pos = torch.arange(0, n)
            page, off = table[pos // 64], pos % 64
            for i, (kp, vp) in m.kv.items():
                out[f"r{r}.{i}.k"] = kp[page.to(kp.device), :, off.to(kp.device)].float().cpu()
        return out

    rows = []
    with torch.inference_mode():
        for owner_index, g in enumerate(generators):
            fast = g.fast
            tp, pair = fast._tp, fast._pair
            slot = g._free_slots[-1]
            for corpus, n, ids in prompts:
                ids = list(ids)

                def single():
                    tp.reset(slot)
                    out = tp.prefill_packed([(slot, ids, 0)], [None])
                    return out[0].float().cpu(), int(tp.last_drafts[0])

                def split():
                    tp.reset(slot)
                    before = pair.splits_owned
                    out = fast.prefill_packed([(slot, ids, 0)], [None])
                    if pair.splits_owned != before + 1:
                        raise RuntimeError("the pass did not split")
                    return out[0].float().cpu(), int(tp.last_drafts[0])

                ref = single(); sync(); ref_state = state(tp, slot, n)
                spl = split(); sync(); spl_state = state(tp, slot, n)
                rep = split(); sync()
                floor = None
                size = tp._pg_size(n)
                bigger = [s for s in sorted(tp.pg) if size is not None and s > size]
                if bigger:
                    tp.reset(slot)
                    fl = tp._pg_run(bigger[0], slot, ids, 0, None)
                    sync()
                    floor = float((fl.float().cpu() - ref[0]).abs().max())

                def timed(fn) -> float:
                    walls = []
                    for _ in range(args.reps):
                        sync()
                        a = time.perf_counter()
                        fn()
                        sync()
                        walls.append((time.perf_counter() - a) * 1e3)
                    return round(statistics.median(walls), 2)

                single_ms, split_ms = timed(single), timed(split)
                st = {}
                for kind in ("rec", "conv", "k"):
                    keys = [k for k in ref_state if k.endswith("." + kind)]
                    st[kind] = round(max(float((spl_state[k] - ref_state[k]).abs().max()) for k in keys), 5)
                rec = {
                    "owner": "AB"[owner_index], "corpus": corpus, "n": n, "size": size,
                    "top1_equal": int(spl[0].argmax()) == int(ref[0].argmax()),
                    "draft_equal": spl[1] == ref[1],
                    "max_logit_diff": round(float((spl[0] - ref[0]).abs().max()), 4),
                    "floor_logit_diff": None if floor is None else round(floor, 4),
                    "floor_size": bigger[0] if bigger else None,
                    "split_repeat_bitwise": bool(torch.equal(spl[0], rep[0]) and spl[1] == rep[1]),
                    "single_ms": single_ms, "split_ms": split_ms,
                    "ratio": round(split_ms / single_ms, 3), "state_max_diff": st,
                }
                result("prompt", **rec)
                rows.append(rec)
                tp.reset(slot)
            result("timeouts", pair=owner_index, value=tp.timeouts())
    top1 = sum(r["top1_equal"] for r in rows)
    drafts = sum(r["draft_equal"] for r in rows)
    floors = [r["floor_logit_diff"] for r in rows if r["floor_logit_diff"] is not None]
    summary = {
        "prompts": len(rows), "top1_equal": top1, "draft_equal": drafts,
        "split_repeat_bitwise": sum(r["split_repeat_bitwise"] for r in rows),
        "max_logit_diff_split": max(r["max_logit_diff"] for r in rows),
        "median_logit_diff_split": statistics.median(r["max_logit_diff"] for r in rows),
        "max_logit_diff_floor": max(floors) if floors else None,
        "median_logit_diff_floor": statistics.median(floors) if floors else None,
        "ratio_median": statistics.median(r["ratio"] for r in rows),
        "min_tokens": split_prefill.MIN_TOKENS,
    }
    result("summary", **summary)
    print("\n=== plan 41 split vs single (owner, corpus, n): top1 draft maxdiff floor single split ratio")
    for r in rows:
        print(f"  {r['owner']} {r['corpus']:9s} {r['n']:5d} {r['top1_equal']!s:5} {r['draft_equal']!s:5} "
              f"{r['max_logit_diff']:7.3f} {str(r['floor_logit_diff']):>7} {r['single_ms']:8.2f} "
              f"{r['split_ms']:8.2f} {r['ratio']:6.3f}")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
