"""Plan 28 step 0: the MTP head's static draft vocabulary (ENGINE_MTP_VOCAB) on one replica.

    ENGINE_TP=2 python tools/mtp_vocab_probe.py [--devices 0,1] [--sizes 0,16384,32768,65536]
        [--iters 100] [--rounds 2] [--max-new 256]

Loads the checkpoint once as one replica of the server does (``Generator.load``, MTP head
on, the full draft vocabulary), builds each size's draft subset with
``engine.fast.draft_vocab`` (generic in-image text only), and for each size switches the
head to it (``set_draft_vocab``) and drops the verify graphs, which are recaptured on the
next step:

1. Step time: the captured verify step (2 rows per slot, the MTP head inside the graph) at
   B=1,2,4, ``--prompt-len`` replay-like prompts prefilled, median of ``--iters`` with every
   device synchronized; ``--rounds`` passes over the sizes, alternating the order, so drift
   shows. Also the captured plain decode step once, for reference.
2. Acceptance and parity: 4 replay-like (Request Factory's synthetic coding lines, lengths of
   round 0 of recorded_sessions.csv) and 4 natural chat prompts, ``--max-new`` greedy tokens
   with ignore_eos through the speculative ``Generator`` path, against plain captured decode
   (computed once: it does not involve the head). Per size and group: identical/4, proposed,
   accepted, rate, tokens per slot-step.

Prints ``RESULT`` lines. A probe for the plan, not the task's metric.
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
    parser.add_argument("--sizes", default="0,16384,32768,65536")
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")
    os.environ["ENGINE_MTP_VOCAB"] = "0"  # load with the full head; the probe switches sizes

    import torch

    from engine.fast import draft_vocab
    from engine.generator import CHECKPOINT, Generator
    from engine.protocol import Sampling

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def median_ms(fn, iters: int) -> float:
        for _ in range(5):
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
        CHECKPOINT["model"],
        CHECKPOINT["revision"],
        max_model_len=262144,
        chunk_tokens=args.chunk,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )
    fast = g.fast
    ranks = list(fast.ranks) if hasattr(fast, "ranks") else [fast]
    print(
        f"RESULT load_s={time.perf_counter() - started:.0f} tp={len(ranks)} mtp={fast.mtp is not None} "
        f"graphs={fast.use_graphs} graph_error={fast.graph_error} vocab={fast.vocab}",
        flush=True,
    )
    if fast.mtp is None:
        raise SystemExit("no MTP head")
    tok = g.tokenizer

    sizes = [int(s) for s in args.sizes.split(",")]
    subsets: dict[int, list[int] | None] = {}
    for n in sizes:
        t0 = time.perf_counter()
        subsets[n] = draft_vocab.build(tok, n, fast.vocab)
        ids = subsets[n]
        if ids is None:
            print(f"RESULT subset N={n} full vocab", flush=True)
            continue
        split = [sum(1 for i in ids if r.vocab_offset <= i < r.vocab_offset + r.vocab) for r in ranks]
        print(
            f"RESULT subset N={n} ids={len(ids)} build_s={time.perf_counter() - t0:.1f} "
            f"max_id={max(ids)} owned_per_shard={split} rows_per_rank={[len(ids) - len(ids) // 2, len(ids) // 2] if len(ranks) == 2 else [len(ids)]}",
            flush=True,
        )
    print(f"RESULT corpus files={len(draft_vocab.corpus_files())}", flush=True)

    def use(n: int) -> None:
        """Switch the head's draft vocabulary and drop the verify graphs (recaptured lazily)."""
        sync()
        for m in ranks:
            for seg in m.vsegments:
                seg.graphs.clear()
                seg.inputs.clear()
                seg.outputs.clear()
            m.__dict__.pop("_verify_static", None)
            # Dropping a pool's last graphs frees the pool: capturing into its handle again
            # trips the HIP caching allocator's assert, so the next capture takes a new pool.
            m.__dict__.pop("_pools", None)
        fast.set_draft_vocab(subsets[n])
        sync()
        if fast.graph_error is not None:
            raise SystemExit(f"graphs are off ({fast.graph_error}): eager step times mean nothing")

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        logits = None
        for begin, end, _ in g._chunks(tuple(prompt), 0):
            follow = prompt[end] if end < len(prompt) else None
            logits = fast.prefill(slot, prompt[begin:end], begin, follow)
        return logits

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
    drafts = [0, 0, 0, 0]
    for batch in (1, 2, 4):
        dec = median_ms(lambda: fast.decode(lasts[:batch], slots[:batch], [length] * batch).argmax(-1).cpu(), args.iters)
        print(f"RESULT decode B={batch} decode_ms={dec:.3f}", flush=True)
    if fast.graph_error is not None:
        raise SystemExit(f"graphs are off ({fast.graph_error}): eager step times mean nothing")
    times: dict[tuple[int, int], list[float]] = {}
    for rnd in range(args.rounds):
        order = sizes if rnd % 2 == 0 else list(reversed(sizes))
        for n in order:
            use(n)
            for batch in (1, 2, 4):
                s = slots[:batch]
                ver = median_ms(lambda: fast.verify(lasts[:batch], drafts[:batch], s, [length] * batch), args.iters)
                times.setdefault((n, batch), []).append(ver)
                print(f"RESULT verify round={rnd} N={n} B={batch} verify_ms={ver:.3f} graph_error={fast.graph_error}", flush=True)
    for batch in (1, 2, 4):
        base = statistics.median(times[(sizes[0], batch)])
        for n in sizes:
            v = statistics.median(times[(n, batch)])
            print(
                f"RESULT step_table B={batch} N={n} verify_ms={v:.3f} rounds={['%.3f' % t for t in times[(n, batch)]]} "
                f"saving_ms={base - v:+.3f} vs N={sizes[0]}",
                flush=True,
            )
    for s in slots:
        fast.reset(s)

    # ---------------------------------------------------------------- 2. acceptance and parity
    with (ROOT / ".vibesys/tasks/mi2508x-tracelab/recorded_sessions.csv").open(newline="") as f:
        rows_csv = list(csv.DictReader(f))
    first_round = sorted((r for r in rows_csv if int(r["round_idx"]) == 0), key=lambda r: int(r["session_id"]))
    groups = {
        "replay": [corpus[9973 * k : 9973 * k + int(r["input_len"])] for k, r in enumerate(first_round[:4])],
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
    n_new = args.max_new
    plains: dict[str, list[list[int]]] = {}
    for name, group in groups.items():
        plain: list[list[int]] = []
        last = []
        for s, p in enumerate(group):
            last.append(int(torch.argmax(prefill(s, p))))
            plain.append([last[-1]])
        for step in range(n_new - 1):
            out = fast.decode(last, list(range(len(group))), [len(p) + step for p in group])
            last = out.argmax(-1).tolist()
            for r in range(len(group)):
                plain[r].append(last[r])
        for s in range(len(group)):
            fast.reset(s)
        plains[name] = plain

    accept: dict[tuple[int, str], tuple[float, float]] = {}
    all_same = 0
    all_total = 0
    for n in sizes:
        use(n)
        n_same = n_total = 0
        for name, group in groups.items():
            before = (g.verify_steps, g.mtp_proposed, g.mtp_accepted)
            gens = [g.start(tuple(p), Sampling(max_tokens=n_new, temperature=0.0, ignore_eos=True)) for p in group]
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
                if spec[r] == plains[name][r]:
                    same += 1
                else:
                    at = next((i for i, (a, b) in enumerate(zip(spec[r], plains[name][r])) if a != b), None)
                    print(f"RESULT parity N={n} {name}[{r}] diverges_at={at} len={len(spec[r])}", flush=True)
            all_same += same
            all_total += len(group)
            n_same += same
            n_total += len(group)
            rate = accepted / max(1, proposed)
            tps = (n_new - 1) / max(1, steps)
            accept[(n, name)] = (rate, tps)
            print(
                f"RESULT accept N={n} {name} identical={same}/{len(group)} verify_steps={steps} "
                f"proposed={proposed} accepted={accepted} rate={rate:.4f} tokens_per_slot_step={tps:.3f}",
                flush=True,
            )
        print(f"RESULT parity N={n} spec==plain {n_same}/{n_total}", flush=True)
    base = sizes[0]
    for n in sizes:
        v2 = statistics.median(times[(n, 2)])
        parts = []
        for name in groups:
            rate, tps = accept[(n, name)]
            parts.append(
                f"{name}_rate={rate:.4f} ({(rate - accept[(base, name)][0]) * 100:+.2f} pt) "
                f"{name}_tok_per_ms_B2={tps / v2:.4f}"
            )
        print(f"RESULT summary N={n} verify_B2_ms={v2:.3f} " + " ".join(parts), flush=True)
    print(f"RESULT parity_total identical={all_same}/{all_total}", flush=True)
    if hasattr(fast, "timeouts"):
        print(f"RESULT tp_allreduce_timeouts={fast.timeouts()}", flush=True)


if __name__ == "__main__":
    main()
