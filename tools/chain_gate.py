"""Plan 31 stage 1 gate: the real chained verify step (``ENGINE_SPEC_CHAIN=1``) under TP2.

    ENGINE_SPEC_CHAIN=1 python tools/chain_gate.py [--devices 0,1] [--today today.txt]

One 2-GCD replica loaded as the server loads it (``Generator.load``: TP2, the MTP head, the
chained verify's graphs captured at load), then:

1. GPU bitwise checks of the engine's MFMA GEMV (``engine.fast.mfma.gemv``) on every dense
   weight shape of rank 0 (random 16-row inputs): each row's output equals its output with
   1, 2, 4 and 8 rows and after the rows are rolled or reversed in the tile.
2. Step time: 2 slots prefilled with replay-like prompts (Request Factory's synthetic coding
   lines, as tools/spec_step_bench.py builds them), one ``verify_chain`` with k = 1, 3, 7
   drafts per slot (2 (k + 1) rows, captured graphs, the host's read of the picks included),
   median of ``--iters`` after warm-up, every device synchronized.
3. Acceptance and parity on that path: ``--pairs`` pairs of replay-like prompts and of
   natural chat prompts, ``--max-new`` greedy tokens with ignore_eos, 2 at a time, through
   the ``Generator``'s chained path with k fixed (``ENGINE_SPEC_K`` = 1, 3, 7) and adaptive,
   against plain captured decode (``FastModel.decode``, the same MFMA kernels): identical or
   the first divergence; emitted tokens per slot per step (E_k) and drafts accepted.
4. Tokens/ms per k = 2 E_k / step_ms(k) on replay-like text, against today's verify at k = 1
   from ``--today`` (tools/spec_step_bench.py's RESULT lines with ENGINE_SPEC_CHAIN off: its
   B=2 ``verify_ms`` and replay ``tokens_per_slot_step``). Kill if the best k is under 1.35x.

Prints ``RESULT`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NATURAL = [
    "Write a Python function that parses an ISO-8601 timestamp into a timezone-aware datetime, "
    "with validation, a docstring and three doctests.",
    "Implement a thread-safe LRU cache class in Python and explain the design.",
    "Explain the difference between TCP and UDP to a new backend engineer.",
    "Write a short story (about 200 words) about a lighthouse keeper.",
    "Write a Go HTTP handler that returns the median of a JSON list of integers, with tests.",
    "Explain Rust's ownership and borrowing rules with three small code examples.",
]


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--max-new", type=int, default=192)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--today", default=None, help="spec_step_bench RESULT lines, chain off")
    parser.add_argument("--baseline-ms", type=float, default=None,
                        help="a k=1 verify B=2 step time given by the lead (another commit)")
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")

    import torch

    from engine.fast import mfma
    from engine.generator import CHECKPOINT, Generator
    from engine.protocol import Sampling

    if not mfma.CHAIN:
        raise SystemExit("run with ENGINE_SPEC_CHAIN=1")

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    tok = g.tokenizer
    out("load", seconds=round(time.perf_counter() - started), tp=hasattr(fast, "ranks"),
        graphs=fast.use_graphs, graph_error=fast.graph_error, ring=fast.ring, kmax=mfma.KMAX,
        max_rows=fast.fused.max_rows, mtp=fast.mtp is not None)

    # ------------------------------------------------------------ 1. GPU bitwise checks
    r0 = fast.ranks[0] if hasattr(fast, "ranks") else fast
    fd = r0.fused
    shapes = {}
    for i in range(len(r0.layers)):
        layer = r0.layers[i]
        for w in (fd.w_in[i], fd.w_moe[i],
                  layer.linear_attn.out_proj.weight if r0.layer_types[i] == "linear_attention"
                  else layer.self_attn.o_proj.weight):
            shapes.setdefault(tuple(w.shape), w)
    shapes.setdefault(tuple(r0.lm_head.shape), r0.lm_head)
    shapes.setdefault(tuple(r0.mtp.fc.weight.shape), r0.mtp.fc.weight)
    all_ok = True
    with torch.inference_mode(), torch.cuda.device(r0.embed.device):
        for shape, w in shapes.items():
            torch.manual_seed(shape[0])
            x = (torch.randn((16, shape[1]), device=w.device) * 0.5).to(torch.bfloat16)
            ref = mfma.gemv(x, w).clone()
            checks = {}
            for rows in (1, 2, 4, 8):
                checks[f"rows{rows}"] = bool(torch.equal(mfma.gemv(x[:rows].contiguous(), w), ref[:rows]))
            for name, perm in (("roll5", [(i + 5) % 16 for i in range(16)]), ("reverse", list(range(15, -1, -1)))):
                p = torch.tensor(perm, device=w.device)
                checks[name] = bool(torch.equal(mfma.gemv(x[p].contiguous(), w), ref[p]))
            exact = x.float() @ w.float().T
            ok = all(checks.values())
            all_ok = all_ok and ok
            out("bitwise", shape=shape, tiles=mfma.tiles_for(*shape), invariant=ok, checks=checks,
                max_abs_vs_fp32=float((ref.float() - exact).abs().max()))
    out("bitwise_summary", all_invariant=all_ok)

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

    # ------------------------------------------------------------ 2. step time
    slots = [0, 1]
    prompts = [corpus[9973 * k : 9973 * k + args.prompt_len] for k in range(2)]
    lasts = [int(torch.argmax(prefill(s, p))) for s, p in zip(slots, prompts, strict=True)]
    length = args.prompt_len
    step_ms = {}
    for k in (1, 3, 7):
        drafts = [[lasts[b]] * k for b in range(2)]
        try:
            fn = lambda: fast.verify_chain(lasts, drafts, slots, [length] * 2)  # noqa: E731
            for _ in range(3):
                fn()
            times = []
            for _ in range(args.iters):
                sync()
                t0 = time.perf_counter()
                fn()
                sync()
                times.append((time.perf_counter() - t0) * 1e3)
            step_ms[k] = statistics.median(times)
            out("step", k=k, slots=2, rows=2 * (k + 1), verify_chain_ms=round(step_ms[k], 3),
                p10=round(sorted(times)[len(times) // 10], 3), graph_error=fast.graph_error)
        except Exception:  # noqa: BLE001
            out("error", what=f"step k={k}", error=traceback.format_exc().replace("\n", " | ")[-1500:])
    for s in slots:
        fast.reset(s)

    # ------------------------------------------------------------ 3. acceptance and parity
    groups = {
        "replay": [corpus[9973 * k + 50 : 9973 * k + 50 + args.prompt_len] for k in range(2 * args.pairs)],
        "natural": [tok.encode(tok.apply_chat_template([{"role": "user", "content": NATURAL[k % len(NATURAL)]}],
                                                       add_generation_prompt=True, tokenize=False,
                                                       enable_thinking=False), add_special_tokens=False)
                    for k in range(2 * args.pairs)],
    }
    n = args.max_new
    plain_cache: dict[tuple[str, int], list[int]] = {}

    def plain(name: str, idx: list[int]) -> list[list[int]]:
        need = [i for i in idx if (name, i) not in plain_cache]
        if need:
            group = [groups[name][i] for i in need]
            lasts_ = [int(torch.argmax(prefill(s, p))) for s, p in enumerate(group)]
            seqs = [[t] for t in lasts_]
            for step in range(n - 1):
                o = fast.decode(lasts_, list(range(len(group))), [len(p) + step for p in group])
                lasts_ = o.float().argmax(-1).tolist()
                for r in range(len(group)):
                    seqs[r].append(lasts_[r])
            for s in range(len(group)):
                fast.reset(s)
            for i, sq in zip(need, seqs, strict=True):
                plain_cache[(name, i)] = sq
        return [plain_cache[(name, i)] for i in idx]

    accept: dict[tuple[str, str], dict] = {}
    parity_total = parity_same = 0
    for kname in ("1", "3", "7", ""):
        os.environ["ENGINE_SPEC_K"] = kname
        for name in groups:
            tot_tokens = tot_steps = 0
            same = 0
            proposed = accepted = 0
            for p0 in range(0, 2 * args.pairs, 2):
                idx = [p0, p0 + 1]
                want = plain(name, idx)
                before = (g.verify_steps, g.mtp_proposed, g.mtp_accepted)
                gens = [g.start(tuple(groups[name][i]), Sampling(max_tokens=n, temperature=0.0, ignore_eos=True))
                        for i in idx]
                spec: list[list[int]] = [[] for _ in idx]
                live = list(range(len(idx)))
                while live:
                    waiting = []
                    for i in live:
                        while True:
                            st = gens[i].next_step()
                            spec[i].append(st.token_id)
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
                steps = g.verify_steps - before[0]
                proposed += g.mtp_proposed - before[1]
                accepted += g.mtp_accepted - before[2]
                tot_steps += steps
                tot_tokens += n - 1
                for r in range(2):
                    parity_total += 1
                    if spec[r] == want[r]:
                        same += 1
                        parity_same += 1
                    else:
                        at = next(i for i, (a, b) in enumerate(zip(spec[r], want[r])) if a != b)
                        out("parity_diverges", k=kname or "adaptive", set=name, prompt=idx[r], at=at)
            e = tot_tokens / max(1, tot_steps)
            accept[(kname, name)] = {"E": e, "rate": accepted / max(1, proposed)}
            out("accept", k=kname or "adaptive", set=name, identical=f"{same}/{2 * args.pairs}",
                verify_steps=tot_steps, tokens_per_slot_step=round(e, 3),
                drafts_checked=proposed, drafts_accepted=accepted,
                per_draft_rate=round(accepted / max(1, proposed), 4))
    os.environ["ENGINE_SPEC_K"] = ""
    out("parity_summary", identical=f"{parity_same}/{parity_total}", verdict="PASS" if parity_same == parity_total else "FAIL")

    # ------------------------------------------------------------ 4. tokens/ms
    today_ms = today_e = None
    if args.today and Path(args.today).exists():
        text = Path(args.today).read_text()
        m = re.search(r"RESULT step B=2 decode_ms=\S+ verify_ms=([\d.]+)", text)
        today_ms = float(m.group(1)) if m else None
        m = re.search(r"RESULT accept replay .*?tokens_per_slot_step=([\d.]+)", text)
        today_e = float(m.group(1)) if m else None
    out("today", verify_B2_ms=today_ms, replay_tokens_per_slot_step=today_e, source=args.today)
    for name in groups:
        rows = {}
        for k in (1, 3, 7):
            if k in step_ms and (str(k), name) in accept:
                e = accept[(str(k), name)]["E"]
                rows[k] = {"step_ms": round(step_ms[k], 3), "E_k": round(e, 3),
                           "tok_per_ms": round(2 * e / step_ms[k], 4), "tpot_ms": round(step_ms[k] / e, 3)}
        if not rows:
            continue
        best = max(rows, key=lambda k: rows[k]["tok_per_ms"])
        line = dict(set=name, by_k=rows, best_k=best)
        if 1 in rows:
            line["best_vs_chain_k1"] = round(rows[best]["tok_per_ms"] / rows[1]["tok_per_ms"], 3)
        if today_ms and today_e:
            today = 2 * today_e / today_ms
            line["today_k1_tok_per_ms"] = round(today, 4)
            line["best_vs_today_k1"] = round(rows[best]["tok_per_ms"] / today, 3)
            if name == "replay":
                line["gate"] = "GO" if rows[best]["tok_per_ms"] / today >= 1.35 else "KILL"
        if args.baseline_ms and today_e:
            base = 2 * today_e / args.baseline_ms
            line["baseline_ms"] = args.baseline_ms
            line["best_vs_baseline_k1"] = round(rows[best]["tok_per_ms"] / base, 3)
            if name == "replay":
                line["gate_vs_baseline"] = "GO" if rows[best]["tok_per_ms"] / base >= 1.35 else "KILL"
        out("tok_per_ms", **line)
    if hasattr(fast, "timeouts"):
        out("tp_allreduce_timeouts", n=fast.timeouts())
    out("done", seconds=round(time.perf_counter() - started))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        out("fatal", error=traceback.format_exc().replace("\n", " | "))
        raise
