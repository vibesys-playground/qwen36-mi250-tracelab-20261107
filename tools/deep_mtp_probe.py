"""Plan 14 step 0: does drafting k > 1 tokens per step with the checkpoint's own MTP head pay?

    python tools/deep_mtp_probe.py [--devices 0,1] [--sessions 16] [--gen 288] [--iters 20]

One 2-GCD replica, loaded as the server loads it (``Generator.load``, MTP head on). No
server, no n-gram drafter: every draft comes from the MTP head, applied autoregressively.

(a) Acceptance per draft depth. ``--sessions`` replay-like prompts (the replay's synthetic
    coding lines, ``tools/serve_probe.py --corpus synthetic``: 1800 tokens each, sessions
    ``--stride`` tokens apart) and as many natural chat prompts are greedy-decoded for
    ``--gen`` tokens, 8 at a time, by an eager plain-decode loop that keeps each step's final
    normed hidden ``H[t]``. After every step the head drafts a chain from ``(H[t], x[t+1])``:
    draft ``d1`` at position ``t``, then ``d_j`` from (the head's own output hidden,
    ``d_{j-1}``) at ``t + j - 1``, up to ``--depth`` (vLLM/SGLang's multi-step MTP with one
    MTP layer; their hidden is the head's normed output, variant ``normed``; ``prenorm``
    feeds the pre-norm output instead, a wiring control, to depth 4). The head's KV at the
    chain's draft positions is overwritten by the next step's true pair before it is read,
    so every chain sees exactly the KV a real k-deep speculative decoder would. Reported per
    set: per-position conditional acceptance ``p_j``, mean accepted tokens per start, and
    ``E_k`` = emitted tokens per verify step of a simulated k-draft decoder (accept the
    longest correct prefix, emit it plus the target's own pick, restart there). Natural
    sequences are cut at their first end-of-sequence token.
(b) Expert overlap. The same decode steps record each row's routed top-8 per layer (the
    reference router: softmax over the BF16 router logits). Per set: mean distinct routed
    experts per layer among R rows that are R consecutive tokens of one session (S sessions
    x R/S consecutive tokens), against R rows from R unrelated sessions at the same step.
(c) Verify step time from real tokens. Every session's slot is prefilled with its real
    context (prompt + 32 generated tokens); a verify of k drafts feeds the next k+1 true
    tokens at consecutive positions (paged attention is causal by position; the Gated
    DeltaNet recurrence runs group after group, ``FastModel.verify_groups = k+1``, each
    group's state in its own row). Timed as captured graphs (median of ``--iters`` replays,
    every device synchronized, recurrent state restored between replays outside the timed
    region), lm_head + argmax included, the MTP head not. Paths: ``fused`` (plan 11 layers,
    ``ENGINE_FUSED_MAX_ROWS`` raised to 32, the recurrence group by group with state copies),
    ``chain`` (the same, the recurrence of a slot's whole chain in one launch per layer:
    ``fused._gdn_chain_*``), ``perop`` (per-op kernels). Each verify's picks are compared
    with the true next tokens (exact greedy check). Also timed: the MTP head on the same
    rows (``mtp_all``: the first draft pass over a slot's k+1 accepted positions) and on
    one row per slot (``mtp_one``), and plain decode of R unrelated sessions (``unrelated``).
    Prediction per (set, S slots, k): tokens/ms = S * E_k / (verify + k * mtp_one) (the
    brief's formula) and S * E_k / (verify + mtp_all + (k-1) * mtp_one) (``real``), each as
    a ratio to k = 1 and to plain decode.

Prints ``RESULT`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NATURAL = [
    # tools/mtp_probe.py (plan 5)
    "Write a Python function that parses an ISO-8601 timestamp string into a timezone-aware "
    "datetime, with input validation, a docstring and three doctest examples.",
    "Implement a thread-safe LRU cache class in Python with get, put and a max size, and "
    "briefly explain the design choices.",
    "Here is a buggy function:\n\n```python\ndef mean(xs):\n    total = 0\n    for i in range(1, len(xs)):\n"
    "        total += xs[i]\n    return total / len(xs)\n```\n\nFind the bugs, fix them, and add "
    "unit tests with pytest.",
    "Explain the difference between TCP and UDP to a new backend engineer, with concrete "
    "examples of when to use each.",
    "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
    # more, code and prose
    "Write a Go HTTP handler that accepts a JSON body with a list of integers and returns their "
    "median, with error handling and a table-driven test.",
    "Explain Rust's ownership and borrowing rules with three small code examples, including one "
    "that does not compile and why.",
    "Write a SQL query that finds, for each customer, their three most recent orders and the "
    "running total of their spend, and explain how the window functions work.",
    "Refactor this JavaScript into modern async/await style and explain the changes:\n\n"
    "function load(url, cb) {\n  fetch(url).then(function (r) { return r.json(); })\n"
    "    .then(function (d) { cb(null, d); })\n    .catch(function (e) { cb(e); });\n}",
    "Describe how a B-tree index speeds up lookups in a relational database, and when an index "
    "can make writes slower.",
    "Write a bash script that finds the ten largest files under a directory, prints their sizes "
    "in human-readable form, and skips hidden directories.",
    "Summarize the causes and consequences of the 2008 financial crisis in about 250 words.",
    "Implement Dijkstra's shortest-path algorithm in C++ with a priority queue, and state its "
    "time complexity.",
    "Write a unit-tested Python class for a bank account with deposits, withdrawals, an "
    "overdraft limit, and a transaction history.",
    "Explain gradient descent and the role of the learning rate to a high-school student, "
    "using an analogy.",
    "Write a Dockerfile for a small Flask app with a non-root user, a health check and a "
    "multi-stage build, then explain each stage.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--sessions", type=int, default=16, help="per set (replay-like, natural)")
    parser.add_argument("--first", type=int, default=1800)
    parser.add_argument("--stride", type=int, default=2400)
    parser.add_argument("--gen", type=int, default=288)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--context-gen", type=int, default=32)
    parser.add_argument("--max-k", type=int, default=7)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--skip-accept", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")

    import torch
    import torch.nn.functional as F

    from engine.fast.model import _rms
    from engine.generator import CHECKPOINT, Generator

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def out(tag: str, **fields) -> None:
        print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":")), flush=True)

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
    tok = g.tokenizer
    if fast.mtp is None or fast.fused is None:
        raise SystemExit(f"needs the MTP head and the fused layers: mtp={fast.mtp is not None} "
                         f"fused_error={fast.fused_error}")
    out("load", seconds=round(time.perf_counter() - started), gpus=torch.cuda.device_count(),
        slots=fast.slots, gdn_rows=fast.gdn_rows, kv_pages=fast.pool.pages,
        graphs=fast.use_graphs, graph_error=fast.graph_error, max_rows=fast.fused.max_rows)
    n_layers = len(fast.layers)
    head_dev = fast.lm_head.device
    hmeta = fast.devices[head_dev]
    eos = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                       tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(t, int) and t >= 0}
    chunk = args.chunk

    def prefill(slot: int, ids: list[int], follow: int | None = None) -> torch.Tensor:
        """Prefill ``ids`` into ``slot`` (reset) with the MTP head; logits after the last."""
        fast.reset(slot)
        body = len(ids) - 1
        position = 0
        while position < body:
            end = min((position // chunk + 1) * chunk, body)
            fast.prefill(slot, ids[position:end], position, ids[end])
            position = end
        return fast.prefill(slot, ids[body:], body, follow)

    # ------------------------------------------------------------------ prompts
    S = args.sessions
    pool: list[int] = []
    index = 0
    while len(pool) < S * args.stride + args.first:
        for line in (f"def case_{index}(items):", f"    return sorted(set(items))  # {index} 0f3a9c"):
            pool.extend(tok.encode(line, add_special_tokens=False))
        index += 1
    replay = [pool[s * args.stride : s * args.stride + args.first] for s in range(S)]
    natural = []
    for s in range(S):
        text = NATURAL[s % len(NATURAL)]
        rendered = tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        natural.append(tok.encode(rendered, add_special_tokens=False))
    sets = {"replay": replay, "natural": natural}

    # ------------------------------------------------------------------ routing recorder
    record = {"on": False, "step": {}}
    fused_moe = fast.fused._moe

    def moe_rec(i, x, h):
        if record["on"] and i < n_layers:
            mlp = fast.layers[i].mlp
            probs = torch.softmax(F.linear(h, mlp.gate.weight), dim=-1, dtype=torch.float)
            record["step"][i] = torch.topk(probs, fast.top_k, dim=-1).indices
        return fused_moe(i, x, h)

    fast.fused._moe = moe_rec

    def decode_eager(tokens: list[int], slots: list[int], positions: list[int]):
        """One plain decode step, eagerly: (fp32 logits [n, V], final normed hidden [n, D])."""
        n = len(tokens)
        for slot, position in zip(slots, positions, strict=True):
            fast._ensure_pages(slot, position + args.depth + 2)
        host = torch.zeros((4, fast._rows), dtype=torch.long)
        host[1].fill_(fast.dummy)
        host[3].fill_(fast.dummy)
        host[0, :n] = torch.tensor(tokens)
        host[1, :n] = torch.tensor(slots)
        host[2, :n] = torch.tensor(positions)
        host[3, :n] = torch.tensor([fast._g(s) for s in slots])
        for meta in fast.devices.values():
            meta.packed.copy_(host)
        x = fast.devices[fast.embed.device].tokens[:n]
        hidden = None
        for seg in fast.segments:
            x = x.to(seg.device)
            meta = fast.devices[seg.device]
            for op in seg.ops:
                if op == fast._head_op:
                    hidden = x
                x = op(x, meta)
        return x, hidden

    def head_step(hidden, tokens, slots_t, positions_t, n):
        """The MTP head on one row per slot: (draft ids, normed output, pre-norm output)."""
        hmeta.packed[1, :n] = slots_t
        hmeta.packed[2, :n] = positions_t
        slots, positions = hmeta.slots[:n], hmeta.positions[:n]
        z_in = fast._mtp_in(hidden, tokens)
        if fast._fused_rows(n, verify=True):
            z = fast.fused.layer(fast.mtp_index, z_in, slots, positions, hmeta)
        else:
            z = fast._mtp_layer(z_in, lambda h: fast._attn_decode(fast.mtp_index, h, slots, positions, hmeta),
                                decode=True)
        y = _rms(z, fast.mtp.norm.weight, fast.eps)
        return F.linear(y, fast.lm_head).argmax(-1), y, z

    # ------------------------------------------------------------------ (a) + (b): decode and draft
    variants = {"normed": args.depth, "prenorm": min(4, args.depth)}
    seqs: dict[str, list[list[int]]] = {}
    plen: dict[str, list[int]] = {}
    drafts: dict[str, dict[str, list[list[list[int]]]]] = {}  # set -> variant -> session -> step -> chain
    routes: dict[str, list[torch.Tensor]] = {}  # set -> session -> [steps, layers, top_k]
    slot_of: dict[tuple[str, int], int] = {}
    for si, (name, prompts) in enumerate(sets.items()):
        seqs[name], plen[name], routes[name] = [], [], []
        drafts[name] = {v: [] for v in variants}
        for b0 in range(0, S, 8):
            batch = list(range(b0, min(S, b0 + 8)))
            slots = [si * S + s for s in batch]
            for s, slot in zip(batch, slots, strict=True):
                slot_of[(name, s)] = slot
            t0 = time.perf_counter()
            gen = []
            for s, slot in zip(batch, slots, strict=True):
                logits = prefill(slot, prompts[s])
                gen.append([int(logits.argmax())])
            chains = {v: [[] for _ in batch] for v in variants}
            rts = [[] for _ in batch]
            n = len(batch)
            for step in range(args.gen):
                positions = [len(prompts[s]) + step for s in batch]
                record["on"], record["step"] = True, {}
                logits, hidden = decode_eager([gl[-1] for gl in gen], slots, positions)
                record["on"] = False
                picks = logits.argmax(-1)
                layers = torch.stack([record["step"][i].to("cpu") for i in range(n_layers)], dim=1)
                hidden = hidden.to(head_dev)
                picks_h = picks.to(head_dev)
                slots_t = torch.tensor(slots, device=head_dev)
                pos_t = torch.tensor(positions, device=head_dev)
                vdrafts = {}
                for v, depth in variants.items():
                    h, x_in, ds = hidden, picks_h, []
                    for j in range(depth):
                        d, y, z = head_step(h, x_in, slots_t, pos_t + j, n)
                        ds.append(d)
                        h, x_in = (y if v == "normed" else z), d
                    vdrafts[v] = torch.stack(ds, dim=1).tolist()
                picks = picks.tolist()
                for r in range(n):
                    gen[r].append(picks[r])
                    rts[r].append(layers[r])
                    for v in variants:
                        chains[v][r].append(vdrafts[v][r])
            for r, s in enumerate(batch):
                seqs[name].append(gen[r])
                plen[name].append(len(prompts[s]))
                routes[name].append(torch.stack(rts[r]))  # [steps, layers, top_k]
                for v in variants:
                    drafts[name][v].append(chains[v][r])
            out("generated", set=name, sessions=batch, steps=args.gen,
                seconds=round(time.perf_counter() - t0, 1),
                head=[tok.decode(gen[r][:24]) for r in range(min(2, n))])

    # Acceptance. gen[r][n] is the token fed at step n; step n's pick is gen[r][n+1]; its chain
    # drafts gen[r][n+1+j] for j = 1..depth.
    def cut(name: str, s: int) -> int:
        """Usable generated tokens of a session: natural ones end at their first EOS."""
        gl = seqs[name][s]
        if name == "natural":
            for i, t in enumerate(gl):
                if t in eos:
                    return i + 1
        return len(gl)

    accept: dict[str, dict[str, dict[int, float]]] = {}
    for name in sets:
        accept[name] = {}
        for v, depth in variants.items():
            hits = [0] * (depth + 1)  # hits[j]: starts whose first j drafts are all correct
            starts = 0
            sum_acc = {k: 0 for k in range(1, depth + 1)}
            sim = {}
            for s in range(S):
                gl, ch, end = seqs[name][s], drafts[name][v][s], cut(name, s)
                # starts with every drafted position known
                lens = []
                for step in range(len(ch)):
                    if step + 1 + depth >= end:
                        break
                    a = 0
                    while a < depth and ch[step][a] == gl[step + 2 + a]:
                        a += 1
                    lens.append(a)
                starts += len(lens)
                for a in lens:
                    for j in range(a + 1):
                        hits[j] += 1
                    for k in sum_acc:
                        sum_acc[k] += min(a, k)
                for k in range(1, depth + 1):
                    tokens = steps = 0
                    step = 0
                    while step < len(lens):
                        a = min(lens[step], k)
                        tokens += a + 1
                        steps += 1
                        step += a + 1
                    tk, st = sim.get(k, (0, 0))
                    sim[k] = (tk + tokens, st + steps)
            p = [hits[j] / max(1, hits[j - 1]) for j in range(1, depth + 1)]
            ek = {k: sim[k][0] / max(1, sim[k][1]) for k in sim}
            accept[name][v] = ek
            out("accept", set=name, variant=v, starts=starts,
                p_j=[round(x, 4) for x in p],
                mean_accepted={k: round(sum_acc[k] / max(1, starts), 3) for k in sum_acc},
                E_k={k: round(e, 3) for k, e in ek.items()})

    # (b) distinct routed experts per layer.
    experts = fast.layers[0].mlp.experts.gate_up_proj.shape[0]
    rng = random.Random(0)
    overlap: dict[str, dict] = {}
    for name in sets:
        hot = []
        for s in range(S):
            r = routes[name][s][: cut(name, s)]  # [T, L, K]
            h = torch.zeros((r.shape[0], r.shape[1], experts), dtype=torch.bool)
            h.scatter_(2, r.long(), True)
            hot.append(h)
        res = {}
        for slots_n in (1, 2, 4):
            for k in range(0, args.max_k + 1):
                rows = slots_n * (k + 1)
                same, unrel = [], []
                for _ in range(64):
                    group = rng.sample(range(S), slots_n)
                    T = min(hot[s].shape[0] for s in group) - k
                    if T <= 0:
                        continue
                    t = rng.randrange(T)
                    u = torch.zeros_like(hot[group[0]][0])
                    for s in group:
                        u |= hot[s][t : t + k + 1].any(0)
                    same.append(u.sum(-1).float().mean().item())
                    if rows <= S:
                        others = rng.sample(range(S), rows)
                        T2 = min(hot[s].shape[0] for s in others)
                        t2 = rng.randrange(T2)
                        u2 = torch.zeros_like(u)
                        for s in others:
                            u2 |= hot[s][t2]
                        unrel.append(u2.sum(-1).float().mean().item())
                res[(slots_n, k)] = (statistics.mean(same) if same else None,
                                     statistics.mean(unrel) if unrel else None)
                out("experts", set=name, slots=slots_n, k=k, rows=rows,
                    same_session=None if not same else round(statistics.mean(same), 2),
                    unrelated=None if not unrel else round(statistics.mean(unrel), 2),
                    per_layer_top_k=fast.top_k)
        overlap[name] = res

    # ------------------------------------------------------------------ (c) step times
    fast.fused.max_rows = 32
    ctx = {}
    for name in sets:
        for s in range(S):
            slot = slot_of[(name, s)]
            full = sets[name][s] + seqs[name][s]
            L = plen[name][s] + args.context_gen
            prefill(slot, full[:L], full[L])
            ctx[(name, s)] = (slot, L, full)
    sync()
    bank1 = fast.gdn_rows  # rows gdn_rows .. 2 * gdn_rows - 2 are free scratch (the last is the dummy's)
    scratch_rows = list(range(bank1, 2 * fast.gdn_rows - 1))

    def set_flags(path: str) -> None:
        fast.triton_kernels["fused_decode"] = path in ("fused", "chain")
        fast.triton_kernels["fused_verify"] = path in ("fused", "chain")
        fast.gdn_chain = path == "chain"

    def capture(ops_by_dev, rows):
        meta0 = fast.devices[fast.embed.device]
        x = meta0.tokens[:rows]
        parts = []
        for dev, ops in ops_by_dev:
            with torch.cuda.device(dev):
                meta = fast.devices[dev]
                static_in = x if not parts else x.to(dev).clone()
                stream = torch.cuda.Stream(dev)
                stream.wait_stream(torch.cuda.current_stream(dev))

                def run(inp, ops=ops, meta=meta):
                    for op in ops:
                        inp = op(inp, meta)
                    return inp

                with torch.cuda.stream(stream), torch.inference_mode():
                    for _ in range(2):
                        run(static_in)
                torch.cuda.current_stream(dev).wait_stream(stream)
                torch.cuda.synchronize(dev)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"), \
                        torch.inference_mode():
                    o = run(static_in)
                parts.append((dev, static_in, o, graph))
                x = o
        return parts

    def replay(parts):
        x = None
        for dev, sin, o, graph in parts:
            with torch.cuda.device(dev):
                if x is not None:
                    sin.copy_(x)
                graph.replay()
                x = o
        return x

    def make_restore(dst: list[int], src: list[int]):
        idx = {}
        for i, (conv, _) in fast.gdn_state.items():
            if conv.device not in idx:
                idx[conv.device] = (torch.tensor(dst, device=conv.device), torch.tensor(src, device=conv.device))

        def go():
            for i, (conv, rec) in fast.gdn_state.items():
                d, s = idx[conv.device]
                conv.index_copy_(0, d, conv.index_select(0, s))
                rec.index_copy_(0, d, rec.index_select(0, s))

        return go

    def time_parts(parts, restore) -> float:
        for _ in range(2):
            restore()
            replay(parts)
        times = []
        for _ in range(args.iters):
            restore()
            sync()
            t0 = time.perf_counter()
            replay(parts)
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(times)

    store: dict[str, torch.Tensor] = {}

    def head_pick(x, meta):
        store["h"] = x
        return F.linear(x, fast.lm_head).argmax(-1)

    def verify_ops(verify: bool):
        ops = [(fast.embed.device, fast._embed_op)]
        for i in range(n_layers):
            ops.append((fast.layers[i].input_layernorm.weight.device, fast._layer_op(i, verify)))
        ops.append((fast.final_norm.device, fast._norm_op))
        ops.append((head_dev, head_pick))
        grouped = []
        for dev, op in ops:
            if not grouped or grouped[-1][0] != dev:
                grouped.append((dev, []))
            grouped[-1][1].append(op)
        return grouped

    def load_rows(rows_spec):
        """rows_spec: list of (token, slot, position, state row) -> packed on every device."""
        host = torch.zeros((4, fast._rows), dtype=torch.long)
        host[1].fill_(fast.dummy)
        host[3].fill_(fast.dummy)
        for r, (t, sl, p, st) in enumerate(rows_spec):
            host[:, r] = torch.tensor([t, sl, p, st])
        for meta in fast.devices.values():
            meta.packed.copy_(host)

    def run_verify(name, sessions, k, path):
        """Time one verify of ``k`` drafts for ``sessions`` (k+1 real tokens each); returns
        (ms, picks correct, rows, hidden, picks, rows_spec)."""
        set_flags(path)
        n = len(sessions)
        groups = k + 1
        fast.verify_groups = groups
        backup = scratch_rows[:n]
        spec, expect = [], []
        for j in range(groups):
            for c, s in enumerate(sessions):
                slot, L, full = ctx[(name, s)]
                fast._ensure_pages(slot, L + groups + 1)
                state = slot if j == 0 else scratch_rows[n + (j - 1) * n + c]
                spec.append((full[L + j], slot, L + j, state))
                expect.append(full[L + j + 1])
        assert n * groups <= len(scratch_rows)
        snap = make_restore(backup, [fast._g(ctx[(name, s)][0]) for s in sessions])
        snap()  # bank-0 rows -> backup rows
        restore = make_restore([fast._g(ctx[(name, s)][0]) for s in sessions], backup)
        load_rows(spec)
        parts = capture(verify_ops(verify=groups > 1), n * groups)
        ms = time_parts(parts, restore)
        restore()
        picks = replay(parts).tolist()
        hidden = store["h"].clone()
        restore()
        del parts
        ok = sum(int(a == b) for a, b in zip(picks, expect, strict=True))
        return ms, ok, n * groups, hidden, picks, spec

    def run_unrelated(keys, path):
        set_flags(path)
        fast.verify_groups = 2
        n = len(keys)
        spec, expect = [], []
        for name, s in keys:
            slot, L, full = ctx[(name, s)]
            fast._ensure_pages(slot, L + 2)
            spec.append((full[L], slot, L, fast._g(slot)))
            expect.append(full[L + 1])
        backup = scratch_rows[:n]
        make_restore(backup, [sp[3] for sp in spec])()
        restore = make_restore([sp[3] for sp in spec], backup)
        load_rows(spec)
        parts = capture(verify_ops(verify=False), n)
        ms = time_parts(parts, restore)
        restore()
        picks = replay(parts).tolist()
        restore()
        del parts
        return ms, sum(int(a == b) for a, b in zip(picks, expect, strict=True))

    def run_mtp(hidden, picks, spec, path):
        set_flags(path)
        n = len(spec)
        host = torch.zeros((4, fast._rows), dtype=torch.long)
        host[1].fill_(fast.dummy)
        host[3].fill_(fast.dummy)
        for r, (t, sl, p, st) in enumerate(spec):
            host[:, r] = torch.tensor([t, sl, p, sl])
        hmeta.packed.copy_(host.to(head_dev))
        hid = hidden[:n].clone()
        toks = torch.tensor(picks[:n], device=head_dev)
        with torch.cuda.device(head_dev):
            stream = torch.cuda.Stream(head_dev)
            stream.wait_stream(torch.cuda.current_stream(head_dev))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):
                    fast._mtp_decode(hid, toks, hmeta)
            torch.cuda.current_stream(head_dev).wait_stream(stream)
            sync()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"), \
                    torch.inference_mode():
                fast._mtp_decode(hid, toks, hmeta)
        ms = time_parts([(head_dev, hid, None, graph)], lambda: None)
        del graph
        return ms

    def guarded(label, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001 - one bad config must not end the job
            where = traceback.format_exc().strip().splitlines()[-3:]
            out("error", what=label, error=f"{type(exc).__name__}: {str(exc)[:300]}", where=where)
            try:
                sync()
            except Exception:  # noqa: BLE001
                pass
            return None

    paths = ("chain", "fused", "perop")
    times: dict = {}
    # Unrelated rows (plain decode of R sessions, one row each): the k = 0 reference.
    for name in sets:
        for R in (1, 2, 4, 8, 16):
            if R > S:
                continue
            for path in ("fused", "perop"):
                res = guarded(f"unrelated {name} R={R} {path}", run_unrelated,
                              [(name, s) for s in range(R)], path)
                if res:
                    times[("unrel", name, R, path)] = res[0]
                    out("unrelated", set=name, rows=R, path=path, ms=round(res[0], 2),
                        picks_ok=f"{res[1]}/{R}")
    mixed = [(nm, s) for s in range(S) for nm in sets][:32]
    for path in ("fused", "perop"):
        res = guarded(f"unrelated mixed R={len(mixed)} {path}", run_unrelated, mixed, path)
        if res:
            times[("unrel", "mixed", len(mixed), path)] = res[0]
            out("unrelated", set="mixed", rows=len(mixed), path=path, ms=round(res[0], 2),
                picks_ok=f"{res[1]}/{len(mixed)}")

    mtp_one: dict = {}
    for name in sets:
        for slots_n in (1, 2, 4):
            sessions = list(range(slots_n))
            for k in range(1, args.max_k + 1):
                for path in paths:
                    res = guarded(f"verify {name} S={slots_n} k={k} {path}", run_verify, name,
                                  sessions, k, path)
                    if not res:
                        continue
                    ms, ok, rows, hidden, picks, spec = res
                    times[("verify", name, slots_n, k, path)] = ms
                    line = dict(set=name, slots=slots_n, k=k, rows=rows, path=path,
                                ms=round(ms, 2), picks_ok=f"{ok}/{rows}")
                    if path in ("chain", "perop"):
                        mpath = "fused" if path == "chain" else "perop"
                        m_all = guarded(f"mtp_all {name} S={slots_n} k={k} {path}", run_mtp,
                                        hidden, picks, spec, mpath)
                        if m_all is not None:
                            times[("mtp_all", name, slots_n, k, mpath)] = m_all
                            line["mtp_all_ms"] = round(m_all, 2)
                        if (name, slots_n, mpath) not in mtp_one:
                            m1 = guarded(f"mtp_one {name} S={slots_n} {path}", run_mtp,
                                         hidden[:slots_n], picks[:slots_n], spec[:slots_n], mpath)
                            if m1 is not None:
                                mtp_one[(name, slots_n, mpath)] = m1
                        if (name, slots_n, mpath) in mtp_one:
                            line["mtp_one_ms"] = round(mtp_one[(name, slots_n, mpath)], 2)
                    out("verify", **line)
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ predictions
    def predict(name, slots_n, path, k, real):
        mpath = "perop" if path == "perop" else "fused"
        ek = accept[name]["normed"].get(k)
        v = times.get(("verify", name, slots_n, k, path))
        m1 = mtp_one.get((name, slots_n, mpath))
        if ek is None or v is None or m1 is None:
            return None
        if real:
            ma = times.get(("mtp_all", name, slots_n, k, mpath))
            if ma is None:
                return None
            t = v + ma + (k - 1) * m1
        else:
            t = v + k * m1
        return slots_n * ek / t

    summary = {}
    for path in paths:
        for real in (False, True):
            for slots_n in (1, 2, 4):
                gains = {}
                for name in sets:
                    base = predict(name, slots_n, path, 1, real)
                    dec = times.get(("unrel", name, slots_n, "perop" if path == "perop" else "fused"))
                    row = {}
                    for k in range(1, args.max_k + 1):
                        r = predict(name, slots_n, path, k, real)
                        if r is None or base is None:
                            continue
                        row[k] = {"tok_per_ms": round(r, 4), "vs_k1": round(r / base, 3),
                                  "vs_decode": None if not dec else round(r / (slots_n / dec), 3)}
                    gains[name] = row
                    out("predict", set=name, path=path, formula="real" if real else "brief",
                        slots=slots_n, by_k=row)
                ks = [k for k in range(1, args.max_k + 1) if all(k in gains[n] for n in sets)]
                if ks:
                    avg = {k: statistics.mean(gains[n][k]["vs_k1"] for n in sets) for k in ks}
                    best = max(avg, key=avg.get)
                    summary[(path, real, slots_n)] = (best, avg[best])
                    out("predict_avg", path=path, formula="real" if real else "brief",
                        slots=slots_n, avg_vs_k1={k: round(a, 3) for k, a in avg.items()},
                        best_k=best, best_gain=round(avg[best], 3),
                        kill_1=("PARK" if avg[best] < 1.2 else "GO"))

    # Kill 2: rows beyond 8 vs unrelated rows.
    for name in sets:
        for path, upath in (("chain", "fused"), ("fused", "fused"), ("perop", "perop")):
            for slots_n, k, R in ((2, 7, 16), (1, 7, 8), (2, 3, 8)):
                same = times.get(("verify", name, slots_n, k, path))
                unrel = times.get(("unrel", name, R, upath))
                if same is None or unrel is None:
                    continue
                out("rows_vs_unrelated", set=name, path=path, slots=slots_n, k=k, rows=R,
                    same_ms=round(same, 2), unrelated_ms=round(unrel, 2),
                    ratio=round(same / unrel, 3))
            a8 = times.get(("verify", name, 2, 3, path))
            a16 = times.get(("verify", name, 2, 7, path))
            u8 = times.get(("unrel", name, 8, upath))
            u16 = times.get(("unrel", name, 16, upath))
            if None not in (a8, a16, u8, u16):
                out("beyond8", set=name, path=path,
                    same_per_row_ms=round((a16 - a8) / 8, 3),
                    unrelated_per_row_ms=round((u16 - u8) / 8, 3),
                    ratio=round((a16 - a8) / max(1e-6, u16 - u8), 3),
                    kill_2=("PARK" if (a16 - a8) >= 0.9 * (u16 - u8) else "GO"))
    out("done", seconds=round(time.perf_counter() - started))


if __name__ == "__main__":
    main()
