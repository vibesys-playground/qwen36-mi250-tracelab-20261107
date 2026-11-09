"""Plan 73 stage 0: exact slot buckets for the chained verify step (3, 5, 6 slots).

    ENGINE_DECODE_BUCKETS=1,2,3,4,5,6,8,16,32 ENGINE_TP=2 python tools/plan73_stage0.py

One TP2 replica loaded as the server loads it (``Generator.load``), then for each case
(real slots, graph bucket, k):

* ``call_ms``: median of ``--iters`` synced ``verify_chain`` calls (host build, H2D, both
  ranks' graph replays, the picks' read), as j303's cost table;
* ``replay_ms``: graph-of-N, the step's inputs loaded once, then ``--iters`` back-to-back replays
  of both ranks' graphs with one sync at the end, divided by N (pure GPU step).

The bucket is forced by overriding rank 0's ``bucket`` (``TPModel`` and the chooser ask rank 0),
so "3 real slots on the 4-slot graph" is today's B=3 path in the same process.

Bitwise: 3 (5, 6) real slots on the exact bucket against the next larger bucket: every
``verify_chain`` output (logits after the last tokens, picks, accepted counts, next drafts) at
``--bitwise-steps`` positions with corpus and random drafts; then greedy generations through the
``Generator``'s chained path (k fixed at 3 and adaptive, run-ahead as configured) on both
buckets: identical tokens.

Also: load time, chained-graph capture (from the engine's log line), free memory and KV pages.
Prints ``RESULT`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--cases", default="1:1:3,2:2:3,3:3:1,3:4:1,3:3:3,3:4:3,3:3:4,3:4:4,3:3:5,3:4:5,"
                        "4:4:3,5:5:3,5:8:3,6:6:3,6:8:3,5:5:1,5:8:1",
                        help="real:bucket:k items")
    parser.add_argument("--bitwise", default="3:4,5:8,6:8", help="exact:larger bucket pairs (real = exact)")
    parser.add_argument("--bitwise-ks", default="1,3,5")
    parser.add_argument("--bitwise-steps", type=int, default=50)
    parser.add_argument("--gen-new", type=int, default=192)
    parser.add_argument("--skip-gen", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("ENGINE_MTP", "1")

    import torch

    from engine import speculative
    from engine.fast import mfma
    from engine.generator import CHECKPOINT, Generator
    from engine.protocol import Sampling

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def mem() -> list[dict]:
        res = []
        for d in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(d)
            res.append({"dev": d, "free_gb": round(free / 2**30, 2), "reserved_gb":
                        round(torch.cuda.memory_reserved(d) / 2**30, 2)})
        return res

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    tok = g.tokenizer
    r0 = fast.ranks[0] if hasattr(fast, "ranks") else fast
    out("load", seconds=round(time.perf_counter() - started, 1), buckets=r0.buckets, kmax=mfma.KMAX,
        ring=fast.ring, graph_error=fast.graph_error, kv_pages=fast.pool.pages, slots=fast.slots,
        max_tokens_one_seq=(fast.pool.pages - 1) * 64, mem=mem(),
        depths={s: speculative.depths(s) for s in r0.buckets})
    real_bucket = r0.bucket

    def force(b: int | None) -> None:
        if b is None:
            r0.__dict__.pop("bucket", None)
        else:
            r0.bucket = lambda batch, _b=b: _b if batch <= _b else real_bucket(batch)

    # ------------------------------------------------------------ prompts
    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    corpus = pool(9973 * 8 + 20000)

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        logits = None
        for begin, end, _ in g._chunks(tuple(prompt), 0):
            follow = prompt[end] if end < len(prompt) else None
            logits = fast.prefill(slot, prompt[begin:end], begin, follow)
        return logits

    # ------------------------------------------------------------ step time
    cases = [tuple(int(x) for x in c.split(":")) for c in args.cases.split(",") if c]
    nmax = max(max(c[0] for c in cases), 6)
    starts = [9973 * s for s in range(nmax)]
    prompts = [corpus[a : a + args.prompt_len] for a in starts]
    lasts = [int(torch.argmax(prefill(s, p))) for s, p in enumerate(prompts)]
    length = args.prompt_len
    step: dict = {}
    for real, bucket, k in cases:
        if k > mfma.KMAX or (k + 1) * bucket > 64 or bucket not in r0.buckets:
            out("skip", real=real, bucket=bucket, k=k)
            continue
        slots = list(range(real))
        drafts = [corpus[a + length + 1 : a + length + 1 + k] for a in starts[:real]]
        try:
            force(bucket)
            t_cap = time.perf_counter()
            fn = lambda: fast.verify_chain(lasts[:real], drafts, slots, [length] * real)  # noqa: E731
            fn()
            first_s = time.perf_counter() - t_cap
            for _ in range(3):
                fn()
            times = []
            for _ in range(args.iters):
                sync()
                t0 = time.perf_counter()
                fn()
                sync()
                times.append((time.perf_counter() - t0) * 1e3)
            # graph-of-N: inputs of this step loaded (by the last call), replays back to back
            sync()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fast._run(bucket, (k + 1) * bucket, verify=False, chain=k + 1)
            sync()
            replay = (time.perf_counter() - t0) * 1e3 / args.iters
            times.sort()
            step[(real, bucket, k)] = (statistics.median(times), replay)
            out("step", real=real, bucket=bucket, k=k, rows=(k + 1) * bucket, real_rows=(k + 1) * real,
                call_ms=round(statistics.median(times), 3), p10=round(times[len(times) // 10], 3),
                p90=round(times[(9 * len(times)) // 10], 3), replay_ms=round(replay, 3),
                first_call_s=round(first_s, 2), graph_error=fast.graph_error)
        except Exception:  # noqa: BLE001
            out("error", what=f"step {real}:{bucket}:{k}", error=traceback.format_exc().replace("\n", " | ")[-1500:])
        finally:
            force(None)
    for (real, bucket, k), (c, r) in sorted(step.items()):
        if bucket != real:
            exact = step.get((real, real, k))
            if exact:
                out("saving", real=real, k=k, padded_bucket=bucket, call_ms_padded=round(c, 3),
                    call_ms_exact=round(exact[0], 3), call_saving_ms=round(c - exact[0], 3),
                    replay_saving_ms=round(r - exact[1], 3),
                    kill_rule=("PASS" if real == 3 and k == 3 and c - exact[0] >= 0.9 else
                               ("FAIL" if real == 3 and k == 3 else None)))
    out("mem_after_steps", mem=mem())

    # ------------------------------------------------------------ bitwise, verify_chain outputs
    rng = random.Random(73)
    ks = [k for k in (int(x) for x in args.bitwise_ks.split(",")) if k <= mfma.KMAX]
    bit_total = bit_same = 0
    for pair in [p_ for p_ in args.bitwise.split(",") if p_]:
        exact, larger = (int(x) for x in pair.split(":"))
        if exact not in r0.buckets:
            continue
        slots = list(range(exact))
        for k in ks:
            if (k + 1) * larger > 64:
                continue
            same = n = 0
            first_diff = None
            for it in range(args.bitwise_steps):
                pos = length - 400 + 8 * it  # positions inside the prefilled prompt (rewritten)
                if it % 2 == 0:
                    drafts = [corpus[a + pos + 1 : a + pos + 1 + k] for a in starts[:exact]]
                else:
                    drafts = [[rng.randrange(1000, 150000) for _ in range(k)] for _ in slots]
                toks = [corpus[a + pos] for a in starts[:exact]]
                res = {}
                for b in (exact, larger, exact):
                    force(b)
                    lg, picks, acc, nxt = fast.verify_chain(toks, drafts, slots, [pos] * exact)
                    res.setdefault(b, []).append((lg.clone(), picks, acc, nxt))
                force(None)
                a_, b_ = res[exact][0], res[larger][0]
                ok = (torch.equal(a_[0], b_[0]) and a_[1:] == b_[1:])
                ok_rep = torch.equal(a_[0], res[exact][1][0]) and a_[1:] == res[exact][1][1:]
                n += 1
                same += ok
                if not ok and first_diff is None:
                    first_diff = {"it": it, "picks_eq": a_[1] == b_[1], "acc_eq": a_[2] == b_[2],
                                  "nxt_eq": a_[3] == b_[3],
                                  "logits_max_abs": float((a_[0] - b_[0]).abs().max())}
                if not ok_rep:
                    out("bitwise_nonrepeat", exact=exact, k=k, it=it)
            bit_total += n
            bit_same += same
            out("bitwise", exact=exact, larger=larger, k=k, identical=f"{same}/{n}", first_diff=first_diff)
    for s in range(nmax):
        fast.reset(s)
    out("bitwise_summary", identical=f"{bit_same}/{bit_total}",
        verdict="PASS" if bit_same == bit_total else "FAIL")

    # ------------------------------------------------------------ generations, exact vs padded bucket
    if not args.skip_gen:
        natural = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                                      tokenize=False, enable_thinking=False), add_special_tokens=False)
                   for t in ("Write a Python function that parses an ISO-8601 timestamp, with tests.",
                             "Explain the difference between TCP and UDP to a new backend engineer.",
                             "Implement a thread-safe LRU cache class in Python.",
                             "Write a Go HTTP handler that returns the median of a JSON list of integers.",
                             "Explain Rust's ownership rules with three small examples.",
                             "Write a short story about a lighthouse keeper.")]
        sets = {"replay": [corpus[9973 * i + 50 : 9973 * i + 50 + args.prompt_len] for i in range(6)],
                "natural": natural}

        def run(group: list[list[int]]) -> tuple[list[list[int]], int]:
            before = g.verify_steps
            gens = [g.start(tuple(p), Sampling(max_tokens=args.gen_new, temperature=0.0, ignore_eos=True))
                    for p in group]
            seqs: list[list[int]] = [[] for _ in group]
            live = list(range(len(group)))
            while live:
                waiting = []
                for i in live:
                    while True:
                        st = gens[i].next_step()
                        seqs[i].append(st.token_id)
                        if st.finish is not None:
                            gens[i].retire()
                            break
                        if not gens[i].has_queued():
                            waiting.append(i)
                            break
                live = waiting
                if live:
                    g.decode([gens[i] for i in live])
            g._reclaim()
            return seqs, g.verify_steps - before

        gen_total = gen_same = 0
        for pair in [p_ for p_ in args.bitwise.split(",") if p_]:
            exact, larger = (int(x) for x in pair.split(":"))
            if exact not in r0.buckets:
                continue
            for kname in ("3", ""):
                os.environ["ENGINE_SPEC_K"] = kname
                for name, group in sets.items():
                    group = group[:exact]
                    try:
                        force(larger)
                        want, st_l = run(group)
                        force(None)
                        got, st_e = run(group)
                    except Exception:  # noqa: BLE001
                        force(None)
                        out("error", what=f"gen {exact} {kname} {name}",
                            error=traceback.format_exc().replace("\n", " | ")[-1500:])
                        continue
                    same = sum(a == b for a, b in zip(got, want, strict=True))
                    gen_total += len(group)
                    gen_same += same
                    out("gen", exact=exact, larger=larger, k=kname or "adaptive", set=name,
                        identical=f"{same}/{len(group)}", steps_exact=st_e, steps_padded=st_l)
        os.environ["ENGINE_SPEC_K"] = ""
        out("gen_summary", identical=f"{gen_same}/{gen_total}", verdict="PASS" if gen_same == gen_total else "FAIL")
    if hasattr(fast, "timeouts"):
        out("tp_allreduce_timeouts", n=fast.timeouts())
    out("done", seconds=round(time.perf_counter() - started))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        out("fatal", error=traceback.format_exc().replace("\n", " | "))
        raise
