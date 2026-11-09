"""Plans 54 + 60 combined: prefill logits with both flags on (the defaults) against both off, bitwise.

    HIP_VISIBLE_DEVICES=0,1 python tools/prefill_flags_bitwise.py [--news 61,561,1650,2300]

Off: ``ENGINE_GDN_WY=mono:8`` (the fused WY kernel of the head) and the MoE down projection
through today's ``_down_kernel`` (``moe_prefill._DOWN_STREAM = False``). On: ``ENGINE_GDN_WY``
unset (the split form) and ``_DOWN_STREAM`` as loaded (unset: the stream kernel above 1,024
tokens). Both are read at call time, so one process holds one model and the same GDN autotune
picks. Loads the replica as the server does (ENGINE_TP default 2, captured prefill graphs).
For each corpus (replay-like synthetic coding lines, Transformers source) and N in --news, a
3,600-token context is prefilled in 2,048-token chunks, then N new tokens as one pass (graphs up
to 2,048, eager above); the pass's logits are kept.

1. ``eager``: graphs set aside, off then on.
2. ``graphs``: as loaded (on), recaptured off, recaptured on again.
Every on result must equal the off one bit for bit, and the recaptured on the loaded on.
Prints ``RESULT`` lines (JSON); exits 1 on any difference. Evidence, not the task's metric.
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
    parser.add_argument("--news", default="61,561,1650,2300")
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    loaded_wy = os.environ.get("ENGINE_GDN_WY")

    import torch

    from engine.fast import gdn_chunk
    from engine.fast import moe_prefill as mp
    from engine.generator import CHECKPOINT, Generator
    from serve_probe import token_pool

    loaded_stream = mp._DOWN_STREAM

    def flags(on: bool) -> None:
        if on:
            if loaded_wy is None:
                os.environ.pop("ENGINE_GDN_WY", None)
            else:
                os.environ["ENGINE_GDN_WY"] = loaded_wy
            mp._DOWN_STREAM = loaded_stream
        else:
            os.environ["ENGINE_GDN_WY"] = "mono:8"
            mp._DOWN_STREAM = False

    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    pg = getattr(fast, "pg", None)
    result("load", tp=os.environ.get("ENGINE_TP", "default"), graphs=getattr(fast, "use_graphs", None),
           prefill_graph_sizes=sorted(pg) if pg else None, pg_error=getattr(fast, "pg_error", None),
           wy_on=list(gdn_chunk.wy_variant()), down_stream_on=loaded_stream,
           stream_min_tokens=mp.STREAM_MIN_TOKENS)
    if not pg:
        result("bitwise_verdict", all_equal=False, error="no captured prefill graphs")
        return 1

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

    def recapture(on: bool) -> None:
        fast.pg.clear()
        gc.collect()
        for d in fast.devices_tp:
            torch.cuda.synchronize(d)
        torch.cuda.empty_cache()
        flags(on)
        fast.pg_error = None
        fast.capture_prefill()
        result("recaptured", flags="on" if on else "off", sizes=len(fast.pg), pg_error=fast.pg_error)

    all_ok = True

    def judge(kind: str, a: dict, b: dict, **extra) -> None:
        nonlocal all_ok
        for c, n, _ in cases:
            x, y = a[(c, n)], b[(c, n)]
            same = bool(torch.equal(x, y))
            all_ok = all_ok and same
            result(kind, corpus=c, new_tokens=n, equal=same, max_logit_diff=float((x - y).abs().max()),
                   argmax_a=int(x.argmax()), argmax_b=int(y.argmax()), **extra)

    # 1. Eager.
    saved = dict(fast.pg)
    fast.pg.clear()
    flags(False)
    eager_off, _ = all_logits()
    flags(True)
    eager_on, _ = all_logits()
    judge("eager_on_vs_off", eager_on, eager_off)
    del eager_off, eager_on
    fast.pg.update(saved)
    del saved
    # 2. Graphs: as loaded (on), recaptured off, recaptured on.
    flags(True)
    on_logits, r_on = all_logits()
    recapture(False)
    off_logits, r_off = all_logits()
    recapture(True)
    on2_logits, r_on2 = all_logits()
    result("graph_replays", on=r_on, off=r_off, on_again=r_on2)
    judge("graphs_on_vs_off", on_logits, off_logits)
    judge("graphs_on_recapture", on_logits, on2_logits)
    all_ok = all_ok and r_on > 0 and r_off > 0
    result("bitwise_verdict", all_equal=all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
