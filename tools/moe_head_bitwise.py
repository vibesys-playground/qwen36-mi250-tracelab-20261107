"""Plan 38: the prefill MoE block of this commit against the head's (0d89b27), bitwise.

    HIP_VISIBLE_DEVICES=0,1 python tools/moe_head_bitwise.py [--news 61,561,1650]

Loads the replica as the server does (ENGINE_TP default 2, captured prefill graphs per padded
size, plan 33). ``tools/plan38_head`` holds verbatim copies of 0d89b27's ``moe_prefill.py`` and
``prefill_kernels.py`` (only the import of ``moe_prefill`` points at the copy): the head's
tiles, binary search, fp32 pair buffer and combine. For each corpus (the replay's synthetic
coding lines, Transformers source) and N in --news, a 3,600-token context is prefilled in
2,048-token chunks, then N new tokens as one pass; last-token logits are kept.

1. ``eager``: the captured graphs set aside (eager prefill), every ``prefill_kernels.moe`` call
   (both ranks, every layer and chunk, MTP head included) runs this commit's block and the
   head's on the same input, compared with ``torch.equal``.
2. ``graphs``: as loaded (this commit's block inside every captured size), logits per case.
3. The graphs dropped and recaptured with the head's ``moe`` (same process, same Gated DeltaNet
   autotune picks): logits per case, compared bitwise with 2.
4. Recaptured again with this commit's block: logits equal to 2 (recapture repeatability).

Prints ``RESULT`` lines (JSON). Evidence for the lead, not the task's metric.
"""

from __future__ import annotations

import argparse
import gc
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
    pg = getattr(fast, "pg", None)
    result("load", tp=os.environ.get("ENGINE_TP", "default"), kind_fast=type(fast).__name__,
           graphs=fast.use_graphs, prefill_graph_sizes=sorted(pg) if pg else None,
           pg_error=getattr(fast, "pg_error", None), buckets=[[b, str(c)] for b, c in mp.BUCKETS])
    if not pg:
        result("bitwise_verdict", all_equal=False, error="no captured prefill graphs")
        return 1
    new_moe = pk.moe

    def prefill(prompt: list[int]) -> torch.Tensor:
        fast.reset(0)
        position = 0
        while position < CONTEXT:
            end = min((position // args.chunk + 1) * args.chunk, CONTEXT)
            fast.prefill(0, prompt[position:end], position, prompt[end])
            position = end
        out = fast.prefill(0, prompt[CONTEXT:], CONTEXT).float().cpu()
        fast.reset(0)
        return out

    pools = {
        "replay": token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 40_000, "synthetic"),
        "natural": token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 40_000, "source"),
    }
    news = [int(n) for n in args.news.split(",")]
    cases = []
    for corpus, pool in pools.items():
        for k, n in enumerate(news):
            offset = (k * 7919 + 101) % (len(pool) - CONTEXT - n - 1)
            cases.append((corpus, n, pool[offset : offset + CONTEXT + n]))

    def all_logits() -> tuple[dict, int]:
        before = fast.pg_replays
        out = {(c, n): prefill(ids) for c, n, ids in cases}
        return out, fast.pg_replays - before

    def recapture(moe_fn) -> None:
        fast.pg.clear()
        gc.collect()
        for d in fast.devices_tp:
            torch.cuda.synchronize(d)
        torch.cuda.empty_cache()
        pk.moe = moe_fn
        try:
            fast.pg_error = None
            fast.capture_prefill()
        finally:
            pk.moe = new_moe
        result("recaptured", moe="head" if moe_fn is not new_moe else "this commit",
               sizes=len(fast.pg), pg_error=fast.pg_error)

    all_ok = True
    # No outer inference mode: TPModel.prefill has its own, and capture_prefill must run outside it.
    # 1. Eager, per call.
    saved = dict(fast.pg)
    fast.pg.clear()
    stats = {"calls": 0, "equal": 0}

    def both(x, mlp, top_k):
        y = new_moe(x, mlp, top_k)
        ref = head_pk.moe(x, mlp, top_k)
        stats["calls"] += 1
        if torch.equal(y, ref):
            stats["equal"] += 1
        else:
            d = (y.float() - ref.float()).abs()
            result("per_call_mismatch", rows=int(x.shape[0]), differing=int((d > 0).sum()), max_abs=float(d.max()))
        return y

    pk.moe = both
    try:
        for c, n, ids in cases:
            stats.update(calls=0, equal=0)
            prefill(ids)
            ok = stats["calls"] > 0 and stats["equal"] == stats["calls"]
            all_ok = all_ok and ok
            result("eager_per_call", corpus=c, new_tokens=n, moe_calls=stats["calls"],
                   moe_calls_equal=stats["equal"], ok=ok)
    finally:
        pk.moe = new_moe
    fast.pg.update(saved)
    del saved
    # 2. Graphs as loaded.
    new_logits, replays_new = all_logits()
    # 3. Head's block captured.
    recapture(head_pk.moe)
    head_logits, replays_head = all_logits()
    # 4. This commit's block captured again.
    recapture(new_moe)
    new2_logits, replays_new2 = all_logits()
    result("graph_replays", new=replays_new, head=replays_head, new_again=replays_new2,
           expected_per_run=len(cases) * 3)
    for c, n, _ in cases:
        a, b, a2 = new_logits[(c, n)], head_logits[(c, n)], new2_logits[(c, n)]
        same = bool(torch.equal(a, b))
        again = bool(torch.equal(a, a2))
        ok = same and again
        all_ok = all_ok and ok
        result("graphs_bitwise", corpus=c, new_tokens=n, logits_equal_head=same, max_logit_diff=float((a - b).abs().max()),
               recapture_equal=again, argmax_head=int(b.argmax()), argmax_new=int(a.argmax()), ok=ok)
    all_ok = all_ok and replays_new > 0 and replays_head > 0
    result("bitwise_verdict", all_equal=all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
