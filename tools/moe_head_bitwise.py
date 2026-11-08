"""Plan 38: the prefill MoE block of this commit against the head's (6a89a83), bitwise.

    HIP_VISIBLE_DEVICES=0,1 python tools/moe_head_bitwise.py [--news 61,561,1650]

Loads the replica as the server does (ENGINE_TP default 2). ``tools/plan38_head`` holds verbatim
copies of 6a89a83's ``moe_prefill.py`` and ``prefill_kernels.py`` (only the import of
``moe_prefill`` points at the copy), i.e. the head's tiles, binary search, fp32 pair buffer and
combine. For each corpus (the replay's synthetic coding lines, Transformers source) and N in
--news, a 3,600-token context is prefilled in 2,048-token chunks, then N new tokens:

* ``per_call``: every ``prefill_kernels.moe`` call (both ranks, every layer and chunk, MTP head
  included) runs this commit's block and the head's on the same input; outputs compared with
  ``torch.equal``; the pass continues with this commit's output.
* ``logits``: the whole sequence prefilled once with the head's block everywhere and once with
  this commit's; the last-token logits compared with ``torch.equal``.

Prints ``RESULT`` lines (JSON). Evidence for the lead, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CONTEXT = 3600


def result(kind: str, **kw) -> None:
    print("RESULT " + json.dumps({"kind": kind, **kw}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", default="61,561,1650")
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1")

    import torch

    from engine.fast import moe_prefill as mp
    from engine.fast import prefill_kernels as pk
    from engine.generator import CHECKPOINT, Generator
    from plan38_head import prefill_kernels as head_pk
    from serve_probe import token_pool

    assert head_pk.moe_prefill is not mp
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    result("load", tp=os.environ.get("ENGINE_TP", "default"), kind_fast=type(fast).__name__,
           graphs=fast.use_graphs, buckets=[[b, str(c)] for b, c in mp.BUCKETS])
    new_moe = pk.moe

    def prefill(prompt: list[int]) -> torch.Tensor:
        fast.reset(0)
        position = 0
        # The context in chunks, then the N new tokens as one pass (N <= the chunk).
        while position < CONTEXT:
            end = min((position // args.chunk + 1) * args.chunk, CONTEXT)
            fast.prefill(0, prompt[position:end], position, prompt[end])
            position = end
        out = fast.prefill(0, prompt[CONTEXT:], CONTEXT).float().cpu()
        fast.reset(0)
        return out

    stats = {"calls": 0, "equal": 0, "rows": set()}

    def both(x, mlp, top_k):
        y = new_moe(x, mlp, top_k)
        ref = head_pk.moe(x, mlp, top_k)
        stats["calls"] += 1
        stats["rows"].add(int(x.shape[0]))
        if torch.equal(y, ref):
            stats["equal"] += 1
        else:
            d = (y.float() - ref.float()).abs()
            result("per_call_mismatch", rows=int(x.shape[0]), differing=int((d > 0).sum()), max_abs=float(d.max()))
        return y

    all_ok = True
    pools = {
        "replay": token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 40_000, "synthetic"),
        "natural": token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 40_000, "source"),
    }
    news = [int(n) for n in args.news.split(",")]
    with torch.inference_mode():
        for corpus, pool in pools.items():
            for k, n in enumerate(news):
                offset = (k * 7919 + 101) % (len(pool) - CONTEXT - n - 1)
                ids = pool[offset : offset + CONTEXT + n]
                stats.update(calls=0, equal=0, rows=set())
                pk.moe = both
                try:
                    prefill(ids)
                finally:
                    pk.moe = new_moe
                calls, equal, rows = stats["calls"], stats["equal"], sorted(stats["rows"])
                pk.moe = head_pk.moe
                try:
                    head_logits = prefill(ids)
                finally:
                    pk.moe = new_moe
                new_logits = prefill(ids)
                same = bool(torch.equal(head_logits, new_logits))
                ok = calls > 0 and equal == calls and same
                all_ok = all_ok and ok
                result("bitwise", corpus=corpus, new_tokens=n, moe_calls=calls, moe_calls_equal=equal,
                       call_rows=rows, last_logits_equal=same,
                       max_logit_diff=float((head_logits - new_logits).abs().max()),
                       argmax_head=int(head_logits.argmax()), argmax_new=int(new_logits.argmax()), ok=ok)
    result("bitwise_verdict", all_equal=all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
