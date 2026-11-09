"""Plan 46 microbenchmark: full vs incremental prompt tokenization at replay sizes.

    python tools/tokenize_bench.py [--tokenizer DIR]

Builds replay-like growing prompts from the replay's synthetic code lines
(previous prompt + new text), and times per round: the full encode
(``tokenizer.encode``, what ``completion_job`` runs on the engine thread),
``PromptEncoder.encode`` (incremental), and the full encode while another
Python thread keeps the GIL busy (as the engine thread does). Checks every
incremental result against the full encode.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--first", type=int, default=3100, help="first prompt, tokens")
    parser.add_argument("--grow", type=int, default=482, help="new tokens per round")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--sessions", type=int, default=8)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from engine.generator import CHECKPOINT
    from engine.inctok import PromptEncoder

    if args.tokenizer:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tok = AutoTokenizer.from_pretrained(CHECKPOINT["model"], revision=CHECKPOINT["revision"])
    enc = PromptEncoder(lambda t: tok.encode(t), Tokenizer.from_str(tok.backend_tokenizer.to_str()))
    print(json.dumps({"incremental": enc.incremental, "why_full": enc.why_full}), flush=True)
    corpus = "".join(
        f"def case_{i}(items):\n    return sorted(set(items))  # {i} 6c1f0a9b2d\n"
        for i in range(200_000)
    )
    full_ids = tok.backend_tokenizer.encode(corpus[:4_000_000])
    offsets = full_ids.offsets

    def text_upto(tokens: int) -> str:
        return corpus[: offsets[tokens][0]]

    rows = {"full": [], "incremental": [], "full_busy": [], "prompt_tokens": [], "new_tokens": []}
    mismatches = 0
    for s in range(args.sessions):
        base = s * (args.first + args.rounds * args.grow + 1000)
        start = offsets[base][0]
        end_tok = base + args.first
        for r in range(args.rounds):
            text = corpus[start : offsets[end_tok][0]]
            t0 = time.perf_counter()
            want = tuple(tok.encode(text))
            t1 = time.perf_counter()
            got = enc.encode(text)
            t2 = time.perf_counter()
            mismatches += got != want
            if r:
                rows["full"].append(t1 - t0)
                rows["incremental"].append(t2 - t1)
                rows["prompt_tokens"].append(len(want))
            end_tok += args.grow
    # The full encode with a busy Python thread (GIL contention as on the engine thread).
    stop = threading.Event()

    def spin() -> None:
        x = 0
        while not stop.is_set():
            x += 1

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    sample = text_upto(args.first + 4 * args.grow)
    for _ in range(20):
        t0 = time.perf_counter()
        tok.encode(sample)
        rows["full_busy"].append(time.perf_counter() - t0)
    stop.set()
    th.join()

    def p(values, q):
        v = sorted(values)
        return round(1e3 * v[min(len(v) - 1, int(q * len(v)))], 3)

    out = {
        "event": "tokenize_bench",
        "rounds_timed": len(rows["full"]),
        "prompt_tokens_p50": sorted(rows["prompt_tokens"])[len(rows["prompt_tokens"]) // 2],
        "full_ms_p50": p(rows["full"], 0.5), "full_ms_p90": p(rows["full"], 0.9),
        "incremental_ms_p50": p(rows["incremental"], 0.5),
        "incremental_ms_p90": p(rows["incremental"], 0.9),
        "full_with_busy_thread_ms_p50": p(rows["full_busy"], 0.5),
        "mismatches": mismatches, **enc.stats(),
    }
    print(json.dumps(out), flush=True)
    if mismatches:
        raise SystemExit("incremental tokenization differs from the full encode")


if __name__ == "__main__":
    main()
