"""Time Generator.start with the old per-expert loop vs the padded-bmm experts path.

Run from the project root on a GPU node: python scripts/time_prefill_experts.py
Loads the model as engine.server does, times a full prefill (generator.start) at each
length for both paths (warm-up + repeats, synchronized), and compares last-token logits.
"""

import argparse
import json
import statistics
import time

import torch

from engine import experts
from engine.generator import CHECKPOINT, Generator, PrefixCache
from engine.protocol import Sampling

NEW = experts._padded_experts_forward
OLD = experts._sorted_experts_forward


def prompt_ids(vocab, n, seed):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randint(1000, min(vocab, 100000), (n,), generator=g).tolist())


def run(gen, impl, prompt):
    experts._padded_experts_forward = impl  # experts_forward looks this global up per call
    gen._prefix_cache = PrefixCache(gen._chunk, 10**7, 4)  # fresh: never reuse a snapshot
    torch.cuda.synchronize()
    t = time.perf_counter()
    g = gen.start(prompt, Sampling(max_tokens=1))
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000, g.logits.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[900, 2048])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    gen = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=8192,
        chunk_tokens=a.chunk, prefix_cache_tokens=10**7, prefix_cache_entries=4,
    )
    vocab = gen.model.config.vocab_size
    report = {}
    for n in a.lengths:
        row = {}
        run(gen, NEW, prompt_ids(vocab, n, 99))  # warm-up both paths
        run(gen, OLD, prompt_ids(vocab, n, 98))
        times = {"old": [], "new": []}
        diffs, agree = [], []
        for r in range(a.repeats):
            p = prompt_ids(vocab, n, r)  # same prompt for both paths -> comparable logits
            t_old, l_old = run(gen, OLD, p)
            t_new, l_new = run(gen, NEW, p)
            times["old"].append(t_old)
            times["new"].append(t_new)
            diffs.append((l_old - l_new).abs().max().item())
            agree.append(int(l_old.argmax() == l_new.argmax()))
        mo, mn = statistics.median(times["old"]), statistics.median(times["new"])
        row.update(old_ms=times["old"], new_ms=times["new"], old_median_ms=mo, new_median_ms=mn,
                   speedup_pct=100 * (mo - mn) / mo, max_abs_logit_diff=max(diffs),
                   top1_agree=agree)
        report[n] = row
        print(f"RESULT {n}: {json.dumps(row)}", flush=True)
    print("REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
