"""Step costs and greedy parity of MTP self-speculation on the fast path, on one 2-GCD replica.

    python tools/spec_step_bench.py [--devices 0,1] [--prompt-len 1800] [--iters 40] [--max-new 128]

Loads the checkpoint as one replica of the server does (``Generator.load`` with
the MTP head, ``ENGINE_MTP`` on, sharded over ``--devices``), then:

1. Step time, every device synchronized, median over ``--iters`` after warmup,
   with B slots prefilled with ``--prompt-len``-token replay-like prompts:
   - ``decode`` B=1,2,4: one captured plain decode step (1 token per slot) and
     the host's read of the picks;
   - ``verify`` B=1,2,4: one ``FastModel.verify`` (2 tokens per slot, captured,
     with the MTP head on both rows inside the same graphs) and its host read;
   - ``mtp`` B=1,2,4: the MTP head alone on 2B rows, as its own captured graph.
   The plan's kill test: verify(B=2) / decode(B=2) > 1.6 (verify includes the
   head, so this is (verify2 + mtp) / decode1 of the plan).
2. Parity and acceptance: replay-like prompts (Request Factory's synthetic
   coding lines, lengths of recorded_sessions.csv round 0) and natural chat
   prompts, 4 at a time, ``--max-new`` greedy tokens with ignore_eos: plain
   captured decode against the speculative ``Generator`` path. Per request:
   identical or the first divergence, with plain decode's top-2 logit margin
   there (a near-tie when small); and the draft acceptance.

Prints ``RESULT`` lines. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--model", default=None, help="a local checkpoint (CPU smoke test)")
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")

    import torch

    from engine.generator import CHECKPOINT, Generator
    from engine.protocol import Sampling

    def sync() -> None:
        for d in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            torch.cuda.synchronize(d)

    def median_ms(fn, iters: int) -> float:
        for _ in range(3):
            fn()
        times = []
        for _ in range(iters):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - start) * 1e3)
        return statistics.median(times)

    started = time.perf_counter()
    g = Generator.load(
        args.model or CHECKPOINT["model"],
        None if args.model else CHECKPOINT["revision"],
        max_model_len=262144 if args.model is None else 65536,
        chunk_tokens=args.chunk,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )
    fast = g.fast
    print(
        f"RESULT load_s={time.perf_counter() - started:.0f} gpus={torch.cuda.device_count()} "
        f"tp={os.environ.get('ENGINE_TP', '1')} "
        f"mtp={fast.mtp is not None} graphs={fast.use_graphs} graph_error={fast.graph_error} "
        f"kv_pages={fast.pool.pages}",
        flush=True,
    )
    if fast.mtp is None:
        raise SystemExit("no MTP head")
    tok = g.tokenizer
    chunk = args.chunk

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(prompt) - 1
        position = 0
        while position < body:
            end = min((position // chunk + 1) * chunk, body)
            fast.prefill(slot, prompt[position:end], position, prompt[end])
            position = end
        return fast.prefill(slot, prompt[body:], body)

    # Replay-like pool: Request Factory's synthetic corpus lines (see tools/spec_serve_probe.py).
    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    corpus = pool(9973 * 8 + 20000)

    # ---------------------------------------------------------------- 1. step time
    slots = [0, 1, 2, 3]
    prompts = [corpus[9973 * k : 9973 * k + args.prompt_len] for k in range(4)]
    lasts = [int(torch.argmax(prefill(s, p))) for s, p in zip(slots, prompts, strict=True)]
    length = args.prompt_len
    drafts = [int(d) for d in [0, 0, 0, 0]]
    for batch in (1, 2, 4):
        s = slots[:batch]
        dec = median_ms(
            lambda: fast.decode(lasts[:batch], s, [length] * batch).argmax(-1).cpu(), args.iters
        )
        ver = median_ms(lambda: fast.verify(lasts[:batch], drafts[:batch], s, [length] * batch), args.iters)
        # The MTP head alone on 2B rows (the layout the last verify left in the step inputs).
        if not torch.cuda.is_available() or hasattr(fast, "ranks"):
            # (Under ENGINE_TP=2 the head alone would wait for rank 1 inside its all-reduces.)
            print(f"RESULT step B={batch} decode_ms={dec:.2f} verify_ms={ver:.2f} "
                  f"verify/decode={ver / dec:.3f}", flush=True)
            if batch == 2:
                print(f"RESULT kill_test B=2 verify_incl_mtp/decode={ver / dec:.3f}", flush=True)
            continue
        size = fast.bucket(batch)
        rows = 2 * size
        dev = fast.lm_head.device
        meta = fast.devices[dev]
        hidden = torch.randn((rows, fast.config.hidden_size), dtype=fast.embed.dtype, device=dev)
        toks = meta.tokens[:rows].clone()
        with torch.cuda.device(dev):
            stream = torch.cuda.Stream(dev)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):
                    fast._mtp_decode(hidden, toks, meta)
            torch.cuda.current_stream(dev).wait_stream(stream)
            sync()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
                fast._mtp_decode(hidden, toks, meta)
        mtp = median_ms(graph.replay, args.iters)
        del graph
        print(
            f"RESULT step B={batch} decode_ms={dec:.2f} verify_ms={ver:.2f} mtp_ms={mtp:.2f} "
            f"verify/decode={ver / dec:.3f}",
            flush=True,
        )
        if batch == 2:
            ratio = ver / dec
            verdict = "KILL" if ratio > 1.6 else "GO"
            print(f"RESULT kill_test B=2 verify_incl_mtp/decode={ratio:.3f} threshold=1.6 -> {verdict}", flush=True)
    for s in slots:
        fast.reset(s)

    # ---------------------------------------------------------------- 2. parity and acceptance
    with (ROOT / ".vibesys/tasks/mi2508x-tracelab/recorded_sessions.csv").open(newline="") as f:
        rows_csv = list(csv.DictReader(f))
    first_round = sorted(
        (r for r in rows_csv if int(r["round_idx"]) == 0), key=lambda r: int(r["session_id"])
    )
    groups = {
        "replay": [
            corpus[9973 * k : 9973 * k + int(r["input_len"])] for k, r in enumerate(first_round[:4])
        ],
        "natural": [
            tok.encode(
                tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                ),
                add_special_tokens=False,
            )
            for text in (
                "Write a Python function that parses an ISO-8601 timestamp into a timezone-aware "
                "datetime, with validation, a docstring and three doctests.",
                "Implement a thread-safe LRU cache class in Python and explain the design.",
                "Explain the difference between TCP and UDP to a new backend engineer.",
                "Write a short story (about 200 words) about a lighthouse keeper.",
            )
        ],
    }
    n = args.max_new
    for name, group in groups.items():
        # Plain captured decode, all four together.
        plain: list[list[int]] = []
        margins: list[list[float]] = []
        lasts = []
        for s, p in enumerate(group):
            logits = prefill(s, p)
            top = torch.topk(logits.float(), 2).values
            margins.append([float(top[0] - top[1])])
            lasts.append(int(torch.argmax(logits)))
            plain.append([lasts[-1]])
        for step in range(n - 1):
            out = fast.decode(lasts, list(range(len(group))), [len(p) + step for p in group]).float()
            top = torch.topk(out, 2, dim=-1).values
            lasts = out.argmax(-1).tolist()
            for r in range(len(group)):
                plain[r].append(lasts[r])
                margins[r].append(float(top[r, 0] - top[r, 1]))
        for s in range(len(group)):
            fast.reset(s)
        # The speculative Generator path, all four together.
        before = (g.verify_steps, g.mtp_proposed, g.mtp_accepted)
        gens = [g.start(tuple(p), Sampling(max_tokens=n, temperature=0.0, ignore_eos=True)) for p in group]
        spec: list[list[int]] = [[] for _ in group]
        live = list(range(len(group)))
        while live:
            waiting = []
            for i in live:
                while True:
                    step = gens[i].next_step()
                    spec[i].append(step.token_id)
                    if step.finish is not None:
                        gens[i].retire()
                        break
                    if not gens[i].has_queued():
                        waiting.append(i)
                        break
            live = waiting
            if live:
                g.decode([gens[i] for i in live])
        g._reclaim()
        steps = g.verify_steps - before[0]
        proposed = g.mtp_proposed - before[1]
        accepted = g.mtp_accepted - before[2]
        same = 0
        for r in range(len(group)):
            if spec[r] == plain[r]:
                same += 1
                print(f"RESULT parity {name}[{r}] identical tokens={n}", flush=True)
            else:
                at = next(i for i, (a, b) in enumerate(zip(spec[r], plain[r])) if a != b)
                print(
                    f"RESULT parity {name}[{r}] diverges_at={at} plain_top2_margin={margins[r][at]:.4f} "
                    f"min_margin_before={min(margins[r][:at] or [0]):.4f}",
                    flush=True,
                )
        print(
            f"RESULT accept {name} identical={same}/{len(group)} verify_steps={steps} "
            f"proposed={proposed} accepted={accepted} rate={accepted / max(1, proposed):.3f} "
            f"tokens_per_slot_step={(n - 1) / max(1, steps):.2f}",
            flush=True,
        )
    if hasattr(fast, "timeouts"):
        print(f"RESULT tp_allreduce_timeouts={fast.timeouts()}", flush=True)


if __name__ == "__main__":
    main()
