"""Plan 47 stage 0: retrieval-scored MTP drafts, feasibility on one GCD (the MI210 slice).

    python -u tools/ivf_probe.py [--reserve-gb 12] [--gen 160] [--clusters 2048,4096,8192]

1. Load the BF16 checkpoint on the CPU, then put the embedding, final norm, lm_head, the MTP
   head and as many trunk layers as fit on cuda:0; the other layers stay in pinned host memory
   and are copied in for each forward (the whole model does not fit one 64 GB GCD). Same
   weights and Transformers modules as the baseline engine; slow but exact.
2. Prompts: replay-like (Request Factory-style synthetic coding lines, as tools/chain_gate.py
   builds them; not the benchmark's sessions) and natural chat prompts (non-thinking
   template), greedy-continued ``--gen`` tokens (ignore_eos) in one left-padded batch; plus
   English prose from the interpreter's own pydoc topics (teacher-forced only).
3. Per sequence, teacher-forced through the trunk: the final normed hidden ``H`` and the verify
   greedy pick at every position; the MTP head prefilled over the pairs ``(H[t], x[t+1])``
   (its KV at every position, as the engine's ``_mtp_prefill``) gives link 1's normed output
   ``y1[t]``; links 2 and 3 run as the engine's ``_chain_head_op`` does (the head on its own
   previous normed output and draft, at the next position, attending to the prefix and to the
   chain's own earlier links), batched over positions with an explicit mask (checked against
   one-at-a-time decode). Drafts are "accepted" where every link so far equals the verify pick.
4. The IVF index per TP2 vocabulary shard (built from the lm_head weights only) at every
   ``--clusters`` x {euclidean, spherical} k-means; build seconds and cluster-size spread.
   The IVF draft (combined over the two shards like ``Group.argmax``) equals the full-vocab
   draft iff the full draft's cluster ranks < P on its shard; agreement per depth, text kind
   and P, on all drafts and on accepted ones. The low-rank shortlist fallback (rank-r
   projection, top-T rescored exactly) the same way.
5. Kernel time on this GPU, graph-captured, for 1, 2 and 8 rows of real ``y``: today's draft
   (``mfma.gemv(y, lm_head shard)`` + ``tp.argmax_pack`` + ``tp.argmax_pick``; the all-reduce
   between them is the same for both and not run here) vs the IVF draft (centroid GEMV + top-P
   + gathered GEMV/argmax + pick), its parts, a gather-read ceiling (same kernel, loads only)
   and a contiguous read of the same bytes; the low-rank shortlist path. GPU IVF vs the torch
   reference draft (id equality).
6. The kill rule of plan 47 stage 0.

Prints ``RESULT`` lines. A plan hint, not the task's metric. Nothing measured here feeds the
index: it is built before any text is read and only from weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

NATURAL = [
    "Write a Python function that parses an ISO-8601 timestamp into a timezone-aware datetime, "
    "with validation, a docstring and three doctests.",
    "Implement a thread-safe LRU cache class in Python and explain the design.",
    "Explain the difference between TCP and UDP to a new backend engineer.",
    "Write a short story (about 200 words) about a lighthouse keeper.",
    "Write a Go HTTP handler that returns the median of a JSON list of integers, with tests.",
    "Explain Rust's ownership and borrowing rules with three small code examples.",
    "Summarize the causes and consequences of the French Revolution in five paragraphs.",
    "Give me a week-long vegetarian meal plan with a shopping list.",
    "Write a SQL query that finds the top three customers by revenue per region, and explain it.",
    "What are the trade-offs between microservices and a monolith for a five-person startup?",
    "Write a bash script that rotates log files older than seven days and compresses them.",
    "Describe how a transformer language model generates text, for a high-school student.",
]


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def time_us(fns, min_launches: int = 64, reps: int = 7) -> float:
    """us per call of a list of launch closures, captured back to back as one graph
    (tools/moe_decode_bench.py's helper)."""
    seq = list(fns) * max(1, math.ceil(min_launches / len(fns)))
    for f in fns:
        f()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fns[0]()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for f in seq:
            f()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(seq))
    del graph
    return statistics.median(times)


# ============================================================================ offload


class Offload:
    """Keep a module's tensors in pinned host memory; copy them in before its forward and
    drop the GPU copies after."""

    def __init__(self, module: torch.nn.Module, dev: torch.device) -> None:
        self.dev = dev
        self.items = []
        for m in module.modules():
            for k, t in m._parameters.items():
                if t is not None:
                    host = t.data.to("cpu").pin_memory()
                    t.data = host
                    self.items.append((t, host))
            for k, t in list(m._buffers.items()):
                if t is not None:
                    m._buffers[k] = t.to(dev)  # small buffers stay on the GPU
        self.bytes = sum(h.numel() * h.element_size() for _, h in self.items)
        module.register_forward_pre_hook(self.pre)
        module.register_forward_hook(self.post)

    def pre(self, module, args):
        for t, host in self.items:
            t.data = host.to(self.dev, non_blocking=True)

    def post(self, module, args, output):
        for t, host in self.items:
            t.data = host


# ============================================================================ main


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve-gb", type=float, default=12.0)
    ap.add_argument("--gen", type=int, default=160)
    ap.add_argument("--replay", type=int, default=8)
    ap.add_argument("--replay-len", type=int, default=1024)
    ap.add_argument("--natural", type=int, default=12)
    ap.add_argument("--doc-len", type=int, default=1536)
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--clusters", default="2048,4096,8192")
    ap.add_argument("--ps", default="8,16,32,64,128")
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--chain-max", type=int, default=400, help="chain positions per sequence")
    ap.add_argument("--rows", default="1,2,8")
    args = ap.parse_args()
    t_start = time.perf_counter()

    os.environ.setdefault("ENGINE_SPEC_CHAIN", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from transformers.masking_utils import create_causal_mask

    from engine import experts, gdn
    from engine.fast import draft_ivf, mfma, tp
    from engine.generator import CHECKPOINT, experts_implementation
    from engine.mtp import MTPHead

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    out("device", name=props.name, total_GB=round(props.total_memory / 1e9, 1),
        cus=props.multi_processor_count, gcn=getattr(props, "gcnArchName", "?"),
        gpus=torch.cuda.device_count(), torch=torch.__version__,
        placement=os.environ.get("VIBESYS_PLACEMENT", "?"))
    ps = [int(p) for p in args.ps.split(",")]
    cls_list = [int(c) for c in args.clusters.split(",")]

    # ------------------------------------------------------------------ 1. load
    model_id, rev = CHECKPOINT["model"], CHECKPOINT["revision"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    experts.register()
    chosen = experts_implementation(torch.version.hip)
    t0 = time.perf_counter()
    lm = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.bfloat16,
                                              device_map="cpu",
                                              **({"experts_implementation": chosen} if chosen else {}))
    lm.eval()
    text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
    tcfg = text.config
    log(f"loaded on cpu in {time.perf_counter() - t0:.0f}s")
    text.embed_tokens.to(dev)
    text.norm.to(dev)
    text.rotary_emb.to(dev)
    lm.lm_head.to(dev)
    head = MTPHead.load(model_id, rev, getattr(lm.config, "text_config", lm.config), dev)
    embed = text.embed_tokens
    W = lm.lm_head.weight.detach()  # [V, K] bf16 on dev
    V, K = W.shape
    resident, offloaded, off_bytes = 0, 0, 0
    for layer in text.layers:
        need = sum(p.numel() * p.element_size() for p in layer.parameters())
        free = torch.cuda.mem_get_info(dev)[0]
        if free - need > args.reserve_gb * 1e9:
            layer.to(dev)
            resident += 1
        else:
            off_bytes += Offload(layer, dev).bytes
            offloaded += 1
    gdn_info = gdn.install(lm)
    out("load", load_s=round(time.perf_counter() - t0), resident_layers=resident,
        offloaded_layers=offloaded, offloaded_GB=round(off_bytes / 1e9, 1), gdn=gdn_info.get("mode"),
        experts=chosen, vocab=V, hidden=K, free_GB=round(torch.cuda.mem_get_info(dev)[0] / 1e9, 1))

    @torch.inference_mode()
    def trunk(ids, mask=None, pos=None, cache=None):
        return text(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                    use_cache=True).last_hidden_state

    def greedy(h: torch.Tensor) -> torch.Tensor:
        return torch.cat([(h[a:a + 512] @ W.T).argmax(-1) for a in range(0, h.shape[0], 512)])

    # ------------------------------------------------------------------ 2. prompts
    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    corpus = pool(9973 * args.replay + args.replay_len + 100)
    prompts = [("replay", corpus[9973 * k + 50: 9973 * k + 50 + args.replay_len]) for k in range(args.replay)]
    for k in range(args.natural):
        msg = [{"role": "user", "content": NATURAL[k % len(NATURAL)]}]
        s = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompts.append(("natural", tok.encode(s, add_special_tokens=False)))

    # Greedy continuation, one left-padded batch, ignore_eos.
    t0 = time.perf_counter()
    B = len(prompts)
    L0 = max(len(p) for _, p in prompts)
    ids = torch.zeros((B, L0), dtype=torch.long)
    mask = torch.zeros((B, L0), dtype=torch.long)
    for b, (_, p) in enumerate(prompts):
        ids[b, L0 - len(p):] = torch.tensor(p)
        mask[b, L0 - len(p):] = 1
    ids, mask = ids.to(dev), mask.to(dev)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    cache = DynamicCache(config=tcfg)
    gen = []
    with torch.inference_mode():
        h = trunk(ids, mask, pos, cache)[:, -1]
        nxt = (h @ W.T).argmax(-1)
        gen.append(nxt)
        for step in range(args.gen - 1):
            mask = torch.cat((mask, torch.ones((B, 1), dtype=mask.dtype, device=dev)), 1)
            pos = pos[:, -1:] + 1
            h = trunk(nxt[:, None], mask, pos, cache)[:, -1]
            nxt = (h @ W.T).argmax(-1)
            gen.append(nxt)
            if step % 40 == 0:
                log(f"gen step {step} {(time.perf_counter() - t0) / (step + 1):.2f}s/step")
    gen = torch.stack(gen, 1).tolist()
    del cache
    seqs = [(kind, p, p + gen[b]) for b, (kind, p) in enumerate(prompts)]
    out("generate", sequences=B, gen=args.gen, seconds=round(time.perf_counter() - t0))
    for kind, p, s in seqs[:: max(1, B // 4)]:
        out("sample", kind=kind, text=tok.decode(s[len(p):len(p) + 60]))
    try:
        import pydoc_data.topics as topics

        prose = "\n\n".join(topics.topics[k] for k in sorted(topics.topics))
        prose_ids = tok.encode(prose, add_special_tokens=False)
        for d in range(args.docs):
            a = 20000 + d * 7919
            seqs.append(("doc", [], prose_ids[a: a + args.doc_len]))
    except Exception as exc:  # noqa: BLE001
        out("error", what="pydoc prose", error=repr(exc))

    # ------------------------------------------------------------------ 3. hidden states
    hcfg = head.config
    lay = head.layers[0]

    def mtp_in(hid: torch.Tensor, toks: torch.Tensor) -> torch.Tensor:
        e = head.pre_fc_norm_embedding(embed(toks))
        return head.fc(torch.cat([e, head.pre_fc_norm_hidden(hid)], dim=-1))

    def rope(x: torch.Tensor, positions: torch.Tensor):
        return head.rotary_emb(x, positions[None, None].expand(3, 1, -1))

    def link(hid, toks, positions, cache_, amask):
        x = mtp_in(hid, toks)[None]
        z = lay(x, position_embeddings=rope(x, positions), attention_mask=amask,
                position_ids=positions[None], past_key_values=cache_, use_cache=True)
        return head.norm(z)[0]

    def seeded(keys, values):
        c = DynamicCache(config=hcfg)
        c.update(keys, values, 0)
        return c

    samples: dict[tuple[str, int], list] = {}  # (kind, depth) -> [(y, accepted)]

    def add(kind, depth, y, acc):
        samples.setdefault((kind, depth), []).append((y.to(torch.bfloat16), acc.bool()))

    t0 = time.perf_counter()
    agree_seq = []
    checked = False
    with torch.inference_mode():
        for si, (kind, p, s) in enumerate(seqs):
            x = torch.tensor(s, device=dev)
            L = x.numel()
            H = trunk(x[None], cache=DynamicCache(config=tcfg))[0]
            g = greedy(H)  # g[t]: verify pick after token t
            P0 = len(p)
            gen_lo = max(P0 - 1, 0) if kind != "doc" else 0
            if kind != "doc":
                agree_seq.append(float((g[P0 - 1:L - 1] == x[P0:L]).float().mean()))
            # MTP prefill over (H[t], x[t+1]), t = 0..L-2
            positions = torch.arange(L - 1, device=dev)
            hc = DynamicCache(config=hcfg)
            xin = mtp_in(H[:-1], x[1:])[None]
            amask = create_causal_mask(config=hcfg, inputs_embeds=xin, attention_mask=None,
                                       past_key_values=hc, position_ids=positions[None])
            z = lay(xin, position_embeddings=rope(xin, positions), attention_mask=amask,
                    position_ids=positions[None], past_key_values=hc, use_cache=True)
            y1 = head.norm(z)[0]  # y1[t] -> draft of token t+2
            d1 = greedy(y1)
            keys, values = hc.layers[0].keys, hc.layers[0].values
            # follow: the trajectory at t is greedy (text == verify picks) for the chain's span
            tf = torch.zeros(L + 3, dtype=torch.bool, device=dev)
            tf[: L - 1] = g[: L - 1] == x[1:]
            if kind == "replay":  # prompt-region depth-1 drafts too
                t_pr = torch.arange(0, max(0, P0 - 3), device=dev)
                acc = (d1[t_pr] == g[t_pr + 1]) & tf[t_pr] & tf[t_pr + 1]
                add("replay_prompt", 1, y1[t_pr], acc)
            # chain positions: t in the generated region (doc: everywhere) with t+3 <= L-1
            T = torch.arange(gen_lo, L - 4, device=dev)
            if T.numel() > args.chain_max:
                T = T[torch.linspace(0, T.numel() - 1, args.chain_max, device=dev).long()]
            n = T.numel()
            if n == 0:
                continue
            acc1 = (d1[T] == g[T + 1]) & tf[T] & tf[T + 1]
            add(kind, 1, y1[T], acc1)
            cache_len = L - 1
            ks = torch.arange(cache_len, device=dev)
            neg = torch.finfo(y1.dtype).min
            m2 = torch.full((n, cache_len + n), neg, dtype=y1.dtype, device=dev)
            m2[:, :cache_len] = torch.where(ks[None] <= T[:, None], 0.0, neg).to(y1.dtype)
            m2[torch.arange(n), cache_len + torch.arange(n)] = 0
            c2 = seeded(keys, values)
            y2 = link(y1[T], d1[T], T + 1, c2, m2[None, None])
            d2 = greedy(y2)
            acc2 = acc1 & (d2 == g[T + 2]) & tf[T + 2]
            add(kind, 2, y2, acc2)
            m3 = torch.full((n, cache_len + 2 * n), neg, dtype=y1.dtype, device=dev)
            m3[:, : cache_len + n] = m2
            m3[torch.arange(n), cache_len + n + torch.arange(n)] = 0
            y3 = link(y2, d2, T + 2, c2, m3[None, None])
            d3 = greedy(y3)
            acc3 = acc2 & (d3 == g[T + 3]) & tf[T + 3]
            add(kind, 3, y3, acc3)
            if not checked:  # batched chain == one position at a time (plain decode, no mask)
                checked = True
                errs, same = [], 0
                for i in range(min(4, n)):
                    t = int(T[i])
                    c = seeded(keys[:, :, : t + 1], values[:, :, : t + 1])
                    a2 = link(y1[t:t + 1], d1[t:t + 1], T[i:i + 1] + 1, c, None)
                    b2 = greedy(a2)
                    a3 = link(a2, b2, T[i:i + 1] + 2, c, None)
                    b3 = greedy(a3)
                    same += int(b2[0] == d2[i]) + int(b3[0] == d3[i])
                    errs.append(float((a2[0].float() - y2[i].float()).abs().max()))
                    errs.append(float((a3[0].float() - y3[i].float()).abs().max()))
                out("chain_check", positions=min(4, n), drafts_equal=same, of=2 * min(4, n),
                    max_abs_y=max(errs))
            out("seq", i=si, kind=kind, len=L, prompt=P0, chain_positions=n,
                accept_rate_d1=round(float(acc1.float().mean()), 4),
                accept_rate_d3=round(float(acc3.float().mean()), 4))
    out("hidden", seconds=round(time.perf_counter() - t0),
        greedy_consistency_min=round(min(agree_seq), 4) if agree_seq else None,
        greedy_consistency_mean=round(statistics.mean(agree_seq), 4) if agree_seq else None)

    # free the trunk and the head
    text.layers = torch.nn.ModuleList()
    del head, lay, lm
    gc.collect()
    torch.cuda.empty_cache()
    out("mem", free_GB=round(torch.cuda.mem_get_info(dev)[0] / 1e9, 1))

    # ------------------------------------------------------------------ 4. indexes
    bounds = [math.ceil(V * r / 2) for r in range(3)]
    shards = [W[bounds[r]: bounds[r + 1]] for r in range(2)]
    indexes = {}  # (variant, C) -> [idx rank0, idx rank1]
    for C in cls_list:
        cap = max(64, 2 * math.ceil(shards[0].shape[0] / C))
        for variant in ("euclid", "spherical"):
            pair = []
            for r in range(2):
                idx = draft_ivf.build(shards[r], C, cap, iters=args.iters, spherical=variant == "spherical")
                pair.append(idx)
                out("index", variant=variant, C=C, rank=r, build_s=round(idx.build_s, 2), **idx.stats)
            indexes[(variant, C)] = pair
    lowrank = {}
    for rk in (128, 256):
        pair = []
        for r in range(2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wf = shards[r].float()
            evals, evecs = torch.linalg.eigh(wf.T @ wf)
            vr = evecs[:, -rk:].contiguous()  # top-rk right singular directions
            wr = (wf @ vr).to(torch.bfloat16)
            torch.cuda.synchronize()
            pair.append((vr, wr))
            energy = float(evals[-rk:].sum() / evals.sum())
            out("lowrank_build", rank=rk, shard=r, build_s=round(time.perf_counter() - t0, 2),
                energy=round(energy, 4), MB=round(wr.numel() * 2 / 1e6, 1))
            del wf
        lowrank[rk] = pair

    # ------------------------------------------------------------------ 5. agreement
    TS = (256, 512, 1024)
    table = {}  # (kind, depth) -> dict
    for (kind, depth), lst in sorted(samples.items()):
        Y = torch.cat([y for y, _ in lst])
        A = torch.cat([a for _, a in lst])
        vstar = torch.cat([(Y[a:a + 512] @ W.T).argmax(-1) for a in range(0, Y.shape[0], 512)])
        shard = (vstar >= bounds[1]).long()
        local = vstar - torch.tensor(bounds[:2], device=dev)[shard]
        res = {"n": int(Y.shape[0]), "accepted": int(A.sum())}
        for (variant, C), pair in indexes.items():
            rank = torch.empty_like(vstar)
            for r in range(2):
                sel = shard == r
                if sel.any():
                    rank[sel] = draft_ivf.search_rank(Y[sel], pair[r], local[sel])
            for P in ps:
                ok = rank < P
                res[f"{variant}:C{C}:P{P}"] = (float(ok.float().mean()),
                                               float(ok[A].float().mean()) if A.any() else None)
        for rk, pair in lowrank.items():
            rank = torch.empty_like(vstar)
            for r in range(2):
                sel = shard == r
                if not sel.any():
                    continue
                vr, wr = pair[r]
                ys, ls = Y[sel], local[sel]
                part = []
                for a in range(0, ys.shape[0], 512):
                    approx = ((ys[a:a + 512].float() @ vr).to(torch.bfloat16) @ wr.T).float()
                    mine = approx.gather(1, ls[a:a + 512][:, None])
                    part.append((approx > mine).sum(1))
                rank[sel] = torch.cat(part)
            for T_ in TS:
                ok = rank < T_
                res[f"lowrank:r{rk}:T{T_}"] = (float(ok.float().mean()),
                                               float(ok[A].float().mean()) if A.any() else None)
        table[(kind, depth)] = res
        out("agreement", kind=kind, depth=depth, n=res["n"], accepted=res["accepted"],
            **{k: [round(v[0], 5), None if v[1] is None else round(v[1], 5)]
               for k, v in res.items() if isinstance(v, tuple)})

    # ------------------------------------------------------------------ 6. timing
    rows_list = [int(r) for r in args.rows.split(",")]
    ysrc = torch.cat([y for y, _ in samples.get(("natural", 1), samples[next(iter(samples))])])
    w0 = shards[0].contiguous()
    timing = {}
    for R in rows_list:
        y = ysrc[:R].contiguous()
        try:
            t_g = time_us([lambda: mfma.gemv(y, w0)])
            t_today = time_us([lambda: tp.argmax_pick(tp.argmax_pack(mfma.gemv(y, w0), 0, 0))])
            timing[("today", R)] = t_today
            out("time_today", rows=R, gemv_us=round(t_g, 1), gemv_argmax_us=round(t_today, 1),
                GBps=round(w0.numel() * 2 / t_g / 1e3, 1))
        except Exception:  # noqa: BLE001
            out("error", what=f"today R={R}", error=traceback.format_exc()[-1500:])
    for C in cls_list:
        idx = indexes[("euclid", C)][0]
        for P in (16, 32, 64):
            try:
                d = draft_ivf.Draft(w0, idx, P, rank=0, offset=0, max_rows=max(rows_list),
                                    centroid_gemv=lambda a, c: mfma.gemv(a, c))
            except Exception:  # noqa: BLE001
                out("error", what=f"Draft C={C} P={P}", error=traceback.format_exc()[-1500:])
                continue
            # GPU kernel vs reference draft (shard 0) on 256 real queries
            try:
                q = ysrc[:256].contiguous()
                gid = []
                for a in range(0, q.shape[0], max(rows_list)):
                    gid.append(tp.argmax_pick(d.pack(q[a:a + max(rows_list)])))
                gid = torch.cat(gid)
                _, rid = draft_ivf.reference_draft(q, w0, idx, P)
                kmatch = float((gid == rid).float().mean())
            except Exception:  # noqa: BLE001
                kmatch = None
                out("error", what=f"kernel check C={C} P={P}", error=traceback.format_exc()[-1500:])
            for R in rows_list:
                y = ysrc[:R].contiguous()
                try:
                    sel = d.select(y)
                    cs = (idx.offsets[1:] - idx.offsets[:-1]).long()
                    touched = int(cs[sel.flatten()].sum())
                    t_ivf = time_us([lambda: tp.argmax_pick(d.pack(y))])
                    t_sel = time_us([lambda: d.select(y)])
                    t_gat = time_us([lambda: d.gather(y, sel)])
                    t_ro = time_us([lambda: d.gather(y, sel, read_only=True)])
                    t_pick = time_us([lambda: d.pick(R)])
                    t_cg = time_us([lambda: mfma.gemv(y, d.cent)])
                    nrow = max(16, touched)
                    t_contig = time_us([lambda: mfma.gemv(y[:1], d.ws[:nrow])])
                    timing[(C, P, R)] = t_ivf
                    gbytes = touched * K * 2
                    out("time_ivf", C=C, P=P, rows=R, total_us=round(t_ivf, 1),
                        centroid_gemv_us=round(t_cg, 1), select_us=round(t_sel, 1),
                        gather_us=round(t_gat, 1), gather_readonly_us=round(t_ro, 1),
                        pick_us=round(t_pick, 1), contiguous_read_us=round(t_contig, 1),
                        rows_touched=touched, gather_GBps=round(gbytes / t_gat / 1e3, 1),
                        contig_GBps=round(nrow * K * 2 / t_contig / 1e3, 1),
                        saving_us=round(timing.get(("today", R), float("nan")) - t_ivf, 1),
                        kernel_eq_reference=kmatch)
                except Exception:  # noqa: BLE001
                    out("error", what=f"time C={C} P={P} R={R}", error=traceback.format_exc()[-1500:])
            del d
            torch.cuda.empty_cache()
    for rk, pair in lowrank.items():
        vr, wr = pair[0]
        vrb = vr.to(torch.bfloat16)
        for T_ in (512, 1024):
            for R in rows_list:
                y = ysrc[:R].contiguous()

                def lr():
                    s = mfma.gemv((y @ vrb).contiguous(), wr)
                    top = torch.topk(s.float(), T_, dim=1, sorted=False).indices
                    cand = w0.index_select(0, top.flatten()).view(R, T_, K)
                    lg = torch.bmm(cand, y[:, :, None])[:, :, 0]
                    return top.gather(1, lg.float().argmax(1, keepdim=True))

                try:
                    t_lr = time_us([lr])
                    out("time_lowrank", rank=rk, T=T_, rows=R, total_us=round(t_lr, 1),
                        saving_us=round(timing.get(("today", R), float("nan")) - t_lr, 1))
                except Exception:  # noqa: BLE001
                    out("error", what=f"lowrank r={rk} T={T_} R={R}", error=traceback.format_exc()[-1500:])

    # ------------------------------------------------------------------ 7. kill rule
    build_max = max(i.build_s for pair in indexes.values() for i in pair if i.clusters == 4096)
    best = None
    for (variant, C) in indexes:
        for P in ps:
            per_depth = []
            for depth in (1, 2, 3):
                vals = [table[(k, depth)][f"{variant}:C{C}:P{P}"][0] for k in ("replay", "natural")
                        if (k, depth) in table]
                if vals:
                    per_depth.append(statistics.mean(vals))
            if not per_depth:
                continue
            sav = {R: timing[("today", R)] - timing[(C, P, R)] for R in rows_list
                   if ("today", R) in timing and (C, P, R) in timing}
            row = dict(variant=variant, C=C, P=P, agree_mean=round(statistics.mean(per_depth), 5),
                       agree_min_depth=round(min(per_depth), 5),
                       saving_us={R: round(v, 1) for R, v in sav.items()})
            out("config", **row)
            s2 = sav.get(2)
            if s2 is not None and s2 >= 250 and min(per_depth) >= 0.993:
                if best is None or row["agree_min_depth"] > best["agree_min_depth"]:
                    best = row
    out("verdict", go=best is not None and build_max <= 60, best=best, index_build_max_s_C4096=round(build_max, 2),
        rule="go iff some (C, P) saves >= 0.25 ms/draft at 2 rows with agreement >= 99.3% at every depth "
             "1-3 (mean of replay and natural), and the C=4096 build <= 60 s",
        total_s=round(time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
