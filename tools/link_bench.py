"""Plan 71 stage 0: what one chained MTP draft link costs, kernel by kernel, under graph replay.

    ENGINE_TP=2 python tools/link_bench.py [--devices 0,1] [--n 64] [--reps 30]

One TP2 replica loaded as the server loads it (``Generator.load``), 2 slots prefilled with
replay-like prompts (``--ctx``; a second pair at ``--long-ctx``), then:

1. Real head outputs ``y`` (the IVF search's input): the MTP head's layer output over every
   prefill position (replay-like prompts and stdlib docstrings as natural prose), normed by
   ``_mtp_norm``, captured on rank 0 while the prefill runs eagerly.
2. Graph-of-N timing (plan 64's lesson: a profiler's per-call time overstates a graph node):
   each item is captured ``--n`` times back to back into one graph per rank, both ranks'
   graphs replayed together, ``--reps`` replays; the per-call time is the median GPU time of a
   replay (events on each GCD, the slower GCD) over ``n``. Items:
   - ``link``: the served link j >= 2 (``_mtp_in`` + fc, the head's decoder layer, ``_mtp_norm``,
     the IVF draft, the copy into the chain output), chained (each link's draft and output
     feed the next), at 1 and 2 rows, with no fallback (gap -1e9), every draft falling back
     (gap +1e9) and IVF off (the full lm_head pass), old and fast (``ENGINE_DRAFT_IVF_FAST``)
     paths; ``head`` (the head layer alone) and ``draft`` (the IVF draft alone).
   - per-kernel items of the head layer (mtp_in, fc, add_rms, q|k|v, the three attention
     kernels, o_proj, all-reduces, post add_rms, router, MoE, mtp_norm) and of the IVF draft
     (centroid GEMV, top-P per part, gather per part, local, flag, the predicated full pass
     nodes with the flag at 0 and at 1, the final pick), the full-pass draft, and the fast
     path's kernels (both-part top-P radix / histogram, both-part gather, local+flag,
     persistent predicated GEMV). Bytes and the floor at ``--bw`` TB/s for the big readers.
   - the head layer at 4, 8 and 10 rows (the all-rows head pass of a chained step).
   - ``empty``: a graph of n one-element fills (the per-node floor).
3. Bitwise: the fast search (``Served._search_fast``, both top-P algorithms) against today's
   (``_search``) on ``--random`` random rows (plus all-zero rows: every centroid score tied)
   and every real row, at 1, 2, 3, 8 and 16 rows per call, flag starting 0 and 1, first link
   or not: sel, nxt, loc, ids, flag and counters equal. Then the whole draft (both ranks,
   eager, the predicated full pass and its all-reduce included) old vs fast on real rows at
   the served gap and with every row falling back.

Prints ``RESULT`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--ctx", type=int, default=1800)
    parser.add_argument("--long-ctx", type=int, default=12000)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--bw", type=float, default=1.05, help="TB/s for the floors")
    parser.add_argument("--random", type=int, default=10000)
    parser.add_argument("--real", type=int, default=2000, help="real rows per kind for the bitwise checks")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--skip", default="", help="comma list of sections to skip: time,kernels,rows,bitwise")
    args = parser.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")

    import torch

    from engine.fast import draft_ivf, fused as fused_mod, kernels, mfma, tp as tp_mod
    from engine.fast.model import FastModel
    from engine.generator import CHECKPOINT, Generator

    triton = kernels.triton
    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    tok = g.tokenizer
    ranks = list(fast.ranks)
    r0, r1 = ranks
    out("load", seconds=round(time.perf_counter() - started), graphs=fast.use_graphs, graph_error=fast.graph_error,
        ivf=r0.ivf is not None, ivf_fast_default=draft_ivf.FAST, arrms=tp_mod.GLUE_ARRMS, ar_ksum=tp_mod.AR_KSUM,
        attn_splits=kernels.ATTN_SPLITS, chain_glue=os.environ.get("ENGINE_CHAIN_GLUE", "1"))
    if r0.ivf is None:
        raise SystemExit("no IVF draft on this replica")

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    # ------------------------------------------------------------ prompts and real head outputs
    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    corpus = pool(9973 * 4 + args.long_ctx + 100)
    import argparse as _a, collections as _c, json as _j, pathlib as _p, re as _r, subprocess as _s, threading as _t  # noqa: E401
    prose = "\n\n".join(m.__doc__ or "" for m in (_a, _c, _j, _p, _r, _s, _t, statistics, os, time))
    natural = tok.encode(prose, add_special_tokens=False)[: max(args.real + 8, 256)]

    stash: list = []

    def grab(z, attend, decode):
        res = FastModel._mtp_layer(r0, z, attend, decode)
        stash.append(res.detach().clone())
        return res

    def prefill(slot: int, prompt: list[int], keep: bool = False):
        fast.reset(slot)
        saved = fast.pg
        if keep:
            fast.pg = {}  # eager prefill, so the head's layer runs through Python on rank 0
            r0._mtp_layer = grab
        try:
            logits = None
            for begin, end, _ in g._chunks(tuple(prompt), 0):
                follow = prompt[end] if end < len(prompt) else None
                logits = fast.prefill(slot, prompt[begin:end], begin, follow)
        finally:
            fast.pg = saved
            r0.__dict__.pop("_mtp_layer", None)
        return logits

    real: dict[str, torch.Tensor] = {}
    with torch.inference_mode(), torch.cuda.device(r0.embed.device):
        for name, prompts in (("replay", [corpus[0:args.ctx], corpus[9973:9973 + args.ctx]]), ("natural", [natural])):
            stash.clear()
            for k, p in enumerate(prompts):
                prefill(4 + k, p, keep=True)
                fast.reset(4 + k)
            z = torch.cat(stash)[: args.real]
            real[name] = r0._mtp_norm(z.contiguous()).contiguous()
            out("real_rows", kind=name, rows=real[name].shape[0],
                rms=round(float(real[name].float().pow(2).mean().sqrt()), 4))
    sync()

    # the timing slots: (0, 1) at --ctx, (2, 3) at --long-ctx
    ctx_slots = {args.ctx: [0, 1], args.long_ctx: [2, 3]}
    for ctx, sl in ctx_slots.items():
        for s, a in zip(sl, (0, 9973), strict=True):
            prefill(s, corpus[a:a + ctx])
            r0._ensure_pages(s, ctx + 32)
    sync()
    out("prefilled", ctx=list(ctx_slots), seconds=round(time.perf_counter() - started))

    K = r0.lm_head.shape[1]

    # ------------------------------------------------------------ graph-of-N harness
    def on(m):
        return torch.cuda.device(m.embed.device)

    def per_rank(fn):
        """fn(m) on both ranks (rank 1 on its thread, as the engine captures it)."""
        with torch.inference_mode(), on(r0):
            a = fn(r0)
        b = fast._pool.submit(fast._on_rank1, lambda: _inf(fn, r1)).result()
        return [a, b]

    def _inf(fn, m):
        with torch.inference_mode():
            return fn(m)

    def capture(m, call, n):
        dev = m.embed.device
        stream = torch.cuda.Stream(dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            for _ in range(n):
                call()
        torch.cuda.synchronize(dev)
        return graph

    def bench(name: str, make, n: int | None = None, setup=None, **info) -> float | None:
        """Per-call us of ``make(m)()`` (both ranks), from graphs of n back-to-back calls."""
        n = n or args.n
        try:
            fast._both(lambda m: make(m)())  # compile any new variant eagerly, before capture
            sync()
            graphs = per_rank(lambda m: capture(m, make(m), n))
            if setup is not None:
                per_rank(setup)
            sync()
            devs = [m.embed.device for m in ranks]
            for _ in range(3):
                for gr, d in zip(graphs, devs, strict=True):
                    with torch.cuda.device(d):
                        gr.replay()
                sync()
            ts = []
            for _ in range(args.reps):
                if setup is not None:
                    per_rank(setup)
                sync()
                evs = []
                for gr, d in zip(graphs, devs, strict=True):
                    with torch.cuda.device(d):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        gr.replay()
                        e1.record()
                        evs.append((e0, e1))
                sync()
                ts.append(max(e0.elapsed_time(e1) for e0, e1 in evs) * 1e3 / n)
            us = statistics.median(ts)
            ts.sort()
            fl = {}
            if "bytes" in info and info["bytes"]:
                fl["floor_us"] = round(info["bytes"] / (args.bw * 1e6), 1)
            out("t", item=name, us=round(us, 2), p10=round(ts[len(ts) // 10], 2),
                p90=round(ts[(9 * len(ts)) // 10], 2), n=n, **info, **fl)
            del graphs
            return us
        except Exception:  # noqa: BLE001
            out("error", item=name, error=traceback.format_exc().replace("\n", " | ")[-1500:])
            sync()
            return None

    def same_on_ranks(t: torch.Tensor) -> list[torch.Tensor]:
        return [t.to(m.embed.device).clone() for m in ranks]

    def inputs(rows: int, ctx: int, kind: str = "replay"):
        """Per-rank (y, d, slots, positions, meta) of a link at ``rows`` rows over ``ctx``."""
        sl = ctx_slots[ctx]
        y = real[kind][100:100 + rows]
        d = torch.tensor([corpus[ctx + 1 + r] for r in range(rows)], dtype=torch.long)
        slots = torch.tensor([sl[r % 2] for r in range(rows)], dtype=torch.long)
        pos = torch.full((rows,), ctx + 4, dtype=torch.long)
        res = []
        for m in ranks:
            dev = m.embed.device
            res.append(SimpleNamespace(y=y.to(dev).clone(), d=d.to(dev), slots=slots.to(dev), pos=pos.to(dev),
                                       meta=m.devices[dev]))
        return res

    def head(m, st, y, d):
        z = m._fused_layer(m.mtp_index, m._mtp_in(y, d, True), st.slots, st.pos, st.meta)
        return m._mtp_norm(z)

    def set_ivf(gap: float | None = None, fast_: bool | None = None, flag: int | None = None):
        def f(m):
            if gap is not None:
                m.ivf.gap = gap
            if fast_ is not None:
                m.ivf.fast = fast_
            if flag is not None:
                m.ivf.flag.fill_(flag)
        per_rank(f)

    GAP0 = r0.ivf.gap
    results: dict = {}

    # ------------------------------------------------------------ 2a. whole links
    if "time" not in skip:
        bench("empty", lambda m: (lambda t=torch.zeros(1, device=m.embed.device): t.fill_(1.0)))
        for ctx in (args.ctx, args.long_ctx):
            for rows in (1, 2):
                sts = inputs(rows, ctx)
                for mode in ("nofb", "allfb", "off"):
                    for fast_ in ((False, True) if mode != "off" else (False,)):
                        gap = {"nofb": -1e9, "allfb": 1e9, "off": GAP0}[mode]
                        set_ivf(gap=gap, fast_=fast_, flag=0)

                        def make(m, mode=mode, fast_=fast_, rows=rows, sts=sts):
                            st = sts[ranks.index(m)]
                            outb = torch.zeros((8 * rows,), dtype=torch.long, device=m.embed.device)
                            cur = SimpleNamespace(y=st.y, d=st.d, j=0)

                            def call():
                                y = head(m, st, cur.y, cur.d)
                                dst = outb[(cur.j % 8) * rows:(cur.j % 8 + 1) * rows]
                                if mode == "off":
                                    d = m._argmax(m.fused.gemv(y, m.lm_head))
                                    dst.copy_(d)
                                elif fast_:
                                    d = m.ivf.draft(y, m, out=dst)
                                else:
                                    d = m.ivf.draft(y, m)
                                    dst.copy_(d)
                                cur.y, cur.d, cur.j = y, d, cur.j + 1
                            return call

                        tag = f"link_{mode}" + ("_fast" if fast_ else "")
                        results[(tag, rows, ctx)] = bench(tag, make, rows=rows, ctx=ctx)
                set_ivf(gap=GAP0, fast_=False, flag=0)
                # the head layer alone, chained on its own output
                def make_head(m, sts=sts):
                    st = sts[ranks.index(m)]
                    cur = SimpleNamespace(y=st.y)

                    def call():
                        cur.y = head(m, st, cur.y, st.d)
                    return call
                results[("head", rows, ctx)] = bench("head", make_head, rows=rows, ctx=ctx)
                # the draft alone on a fixed y
                for mode, gap in (("nofb", -1e9), ("allfb", 1e9)):
                    for fast_ in (False, True):
                        set_ivf(gap=gap, fast_=fast_, flag=0)

                        def make_draft(m, sts=sts, fast_=fast_, rows=rows):
                            st = sts[ranks.index(m)]
                            dst = torch.zeros((rows,), dtype=torch.long, device=m.embed.device)
                            return (lambda: m.ivf.draft(st.y, m, out=dst)) if fast_ else (lambda: m.ivf.draft(st.y, m))
                        tag = f"draft_{mode}" + ("_fast" if fast_ else "")
                        results[(tag, rows, ctx)] = bench(tag, make_draft, rows=rows, ctx=ctx)
                set_ivf(gap=GAP0, fast_=False, flag=0)

                def make_full(m, sts=sts):
                    st = sts[ranks.index(m)]
                    return lambda: m._argmax(m.fused.gemv(st.y, m.lm_head))
                bench("draft_full_pass", make_full, rows=rows, ctx=ctx,
                      bytes=r0.lm_head.numel() * 2)
        # summary per (rows, ctx)
        for ctx in (args.ctx, args.long_ctx):
            for rows in (1, 2):
                gv = lambda t: results.get((t, rows, ctx))  # noqa: E731
                out("link_summary", rows=rows, ctx=ctx, **{t: (round(gv(t), 1) if gv(t) else None) for t in (
                    "link_nofb", "link_nofb_fast", "link_allfb", "link_allfb_fast", "link_off", "head",
                    "draft_nofb", "draft_nofb_fast", "draft_allfb", "draft_allfb_fast")})

    # ------------------------------------------------------------ 2b. per kernel
    if "kernels" not in skip:
        for ctx in (args.ctx, args.long_ctx):
            for rows in (1, 2):
                sts = inputs(rows, ctx)
                # eager pass of the head layer on both ranks for every intermediate
                def eager(m, sts=sts):
                    st = sts[ranks.index(m)]
                    fd = m.fused
                    mi = m.mtp_index
                    layer = fd._layer(mi)
                    I = SimpleNamespace()
                    mt = m.mtp
                    I.x2 = kernels.mtp_in(st.d, m.mtp_embed, st.y, mt.pre_fc_norm_embedding.weight,
                                          mt.pre_fc_norm_hidden.weight, m.eps)
                    I.z = fd.gemv(I.x2, mt.fc.weight)
                    _, I.h1 = kernels.add_rms(I.z, None, layer.input_layernorm.weight, m.eps)
                    I.qkv = fd._gemv(I.h1, fd.w_in[mi])
                    I.o = fd._attn(mi, I.h1, st.slots, st.pos, st.meta)
                    I.a = m._reduce(I.o)
                    I.z2, I.h2 = kernels.add_rms(I.z, I.a, layer.post_attention_layernorm.weight, m.eps)
                    I.moe = fd._moe(mi, None, I.h2)
                    I.z3 = m._reduce(I.moe, I.z2)
                    I.y = m._mtp_norm(I.z3)
                    torch.cuda.synchronize(m.embed.device)
                    return I
                Is = fast._both(eager)
                sync()
                fd0 = r0.fused
                mi = r0.mtp_index
                att = fd0._layer(mi).self_attn
                kp, _ = r0.kv[mi]
                kvh, page, dim = kp.shape[1], kp.shape[2], kp.shape[3]
                qh = att.o_proj.weight.shape[1] // dim
                ex = fd0._layer(mi).mlp.experts
                per_expert = (ex.gate_up_proj[0].numel() + ex.down_proj[0].numel()) * 2
                sh = fd0._layer(mi).mlp.shared_expert
                shared_b = (sh.gate_proj.weight.numel() + sh.up_proj.weight.numel() + sh.down_proj.weight.numel()) * 2
                kv_b = rows * ctx * kvh * dim * 2 * 2

                def k_item(name, fn, **info):
                    def make(m):
                        I = Is[ranks.index(m)]
                        st = sts[ranks.index(m)]
                        return lambda: fn(m, I, st)
                    bench("k_" + name, make, rows=rows, ctx=ctx, **info)

                lw = lambda m: m.fused._layer(m.mtp_index)  # noqa: E731
                k_item("mtp_in", lambda m, I, st: kernels.mtp_in(st.d, m.mtp_embed, st.y, m.mtp.pre_fc_norm_embedding.weight,
                                                                 m.mtp.pre_fc_norm_hidden.weight, m.eps))
                k_item("fc", lambda m, I, st: m.fused.gemv(I.x2, m.mtp.fc.weight), bytes=r0.mtp.fc.weight.numel() * 2,
                       tiles=mfma.tiles_for(*r0.mtp.fc.weight.shape))
                k_item("add_rms_in", lambda m, I, st: kernels.add_rms(I.z, None, lw(m).input_layernorm.weight, m.eps))
                k_item("qkv", lambda m, I, st: m.fused._gemv(I.h1, m.fused.w_in[m.mtp_index]),
                       bytes=fd0.w_in[mi].numel() * 2, tiles=mfma.tiles_for(*fd0.w_in[mi].shape))
                k_item("attn_all", lambda m, I, st: m.fused._attn(m.mtp_index, I.h1, st.slots, st.pos, st.meta, defer=True),
                       bytes=fd0.w_in[mi].numel() * 2 + att.o_proj.weight.numel() * 2 + kv_b)

                def attn_parts(m, I, st, which):
                    fd = m.fused
                    a = fd._layer(m.mtp_index).self_attn
                    k_pool, v_pool = m.kv[m.mtp_index]
                    cos, sin = fd.rope[I.h1.device]
                    q = torch.empty((rows, qh, dim), dtype=I.h1.dtype, device=I.h1.device)
                    lengths = torch.empty((rows,), dtype=torch.int32, device=I.h1.device)
                    if which in ("prep", "core"):
                        fused_mod._attn_prep_kernel[(rows, qh + 2 * kvh)](
                            I.qkv, q, k_pool, v_pool, st.meta.page_table, st.slots, st.pos, lengths, cos, sin,
                            a.q_norm.weight, a.k_norm.weight, a.q_norm.eps, a.k_norm.eps,
                            I.qkv.stride(0), st.meta.page_table.stride(0), k_pool.stride(0), k_pool.stride(1),
                            k_pool.stride(2), QH=qh, KVH=kvh, D=dim, HALF=cos.shape[1], PAGE=page, num_warps=4)
                    if which == "prep":
                        return
                    splits = kernels.ATTN_SPLITS
                    acc = torch.empty((rows, qh, splits, dim), dtype=torch.float32, device=I.h1.device)
                    ml = torch.empty((rows, qh, splits, 2), dtype=torch.float32, device=I.h1.device)
                    if which == "core":
                        kernels._paged_attn_kernel[(rows, kvh, splits)](
                            q, k_pool, v_pool, st.meta.page_table, st.slots, lengths, acc, ml, a.scaling, qh,
                            st.meta.page_table.stride(0), k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
                            GROUP=qh // kvh, GROUP_PAD=max(16, triton.next_power_of_2(qh // kvh)), D=dim, PAGE=page,
                            SPLITS=splits, DOT_F32=kernels.INTERPRET)
                        gated = torch.empty((rows, qh * dim), dtype=I.h1.dtype, device=I.h1.device)
                        fused_mod._attn_combine_kernel[(rows, qh)](
                            acc, ml, I.qkv, gated, I.qkv.stride(0), QH=qh, D=dim, SPLITS=splits, num_warps=4)

                k_item("attn_prep", lambda m, I, st: attn_parts(m, I, st, "prep"))
                k_item("attn_prep_core_combine", lambda m, I, st: attn_parts(m, I, st, "core"), bytes=kv_b)
                k_item("o_proj_defer", lambda m, I, st: m.fused._gemv(I.o, lw(m).self_attn.o_proj.weight, defer=True),
                       bytes=att.o_proj.weight.numel() * 2)
                k_item("ar_attn", lambda m, I, st: m._reduce(I.o))
                k_item("add_rms_post", lambda m, I, st: kernels.add_rms(I.z, I.a, lw(m).post_attention_layernorm.weight, m.eps))
                w3 = fd0.w_moe[mi]

                def router(m, I, st):
                    fd = m.fused
                    w = fd.w_moe[m.mtp_index]
                    idx = torch.empty((rows, m.top_k), dtype=torch.int32, device=I.h2.device)
                    wt = torch.empty((rows, m.top_k), dtype=torch.float32, device=I.h2.device)
                    e = fd._layer(m.mtp_index).mlp.experts.gate_up_proj.shape[0]
                    r = mfma.gemv_topk(I.h2, w, idx, wt, e, m.top_k) if mfma.TOPK_EPI else None
                    if r is None:
                        fd._gemv(I.h2, w)
                k_item("router", router, bytes=w3.numel() * 2)
                k_item("moe", lambda m, I, st: m.fused._moe(m.mtp_index, None, I.h2),
                       bytes=w3.numel() * 2 + rows * r0.top_k * per_expert + shared_b, experts_bytes_row=per_expert)
                k_item("ar_moe", lambda m, I, st: m._reduce(I.moe, I.z2))
                k_item("mtp_norm", lambda m, I, st: m._mtp_norm(I.z3))

                # IVF draft kernels (today's), y = the eager head output
                iv0 = r0.ivf
                C, P = iv0.c, iv0.p
                part_b = rows * P * 64 * K * 2

                def ivf_item(name, fn, flag=0, **info):
                    def make(m):
                        I = Is[ranks.index(m)]
                        iv = m.ivf
                        S = SimpleNamespace(s=mfma.gemv(I.y, iv.cent),
                                            ids=torch.empty((rows,), dtype=torch.int64, device=I.y.device),
                                            pack=torch.zeros((rows, 4), dtype=torch.float32, device=I.y.device))
                        if not hasattr(iv, "_lb_logits"):
                            iv._lb_logits = torch.empty((16, m.lm_head.shape[0]), dtype=torch.bfloat16, device=I.y.device)
                        return lambda: fn(m, iv, I, S)
                    bench("ivf_" + name, make, setup=lambda m: m.ivf.flag.fill_(flag), rows=rows, ctx=ctx, flag=flag, **info)

                dI = draft_ivf
                ivf_item("cgemv", lambda m, iv, I, S: mfma.gemv(I.y, iv.cent), bytes=iv0.cent.numel() * 2,
                         tiles=mfma.tiles_for(*iv0.cent.shape))
                for part in range(iv0.np):
                    ivf_item(f"radix_p{part}", lambda m, iv, I, S, part=part: dI._radix_topp_kernel[(rows,)](
                        S.s[:, part * iv.c:], iv.sel[part], iv.nxt[part], iv.np * iv.c, C=iv.c, P=iv.p, num_warps=4))
                    ivf_item(f"gather_p{part}", lambda m, iv, I, S, part=part: dI._ivf_gather_kernel[(rows, iv.p, iv.ch)](
                        I.y, iv.weights[part], iv.ids[part], iv.offsets[part], iv.sel[part], iv.outv[part],
                        iv.outi[part], iv.p, K=K, CH=iv.ch, BR=iv.BR, BK=iv.BK, READ_ONLY=False, PERM=True,
                        num_warps=4), bytes=part_b)
                ivf_item("local", lambda m, iv, I, S: dI._ivf_local_kernel[(rows,)](
                    iv.outv, iv.outi, iv.nxt, iv.loc, S.ids, iv.m, iv.max_rows, iv.part_rows, NP=iv.np,
                    BM=min(1024, triton.next_power_of_2(iv.m)), num_warps=4))
                ivf_item("flag", lambda m, iv, I, S: dI._ivf_flag_kernel[(1,)](
                    iv.loc, iv.flag, iv.stats, rows, -1e9, RB=iv.rb, num_warps=1))
                ivf_item("zero_flag", lambda m, iv, I, S: iv.flag.zero_())
                for flag in (0, 1):
                    def pg(m, iv, I, S, persistent):
                        n_ = m.lm_head.shape[0]
                        bn, bk, warps, _ = iv.tiles_w
                        tiles = triton.cdiv(n_, bn)
                        lg = iv._lb_logits[:rows]
                        if persistent:
                            dI._pred_gemv_p_kernel[(min(tiles, max(1, iv.pred_grid) * iv.cus),)](
                                iv.flag, I.y, iv.w, lg, rows, n_, n_, tiles, K=K, BM=mfma.BM, BN=bn, BK=bk,
                                DOT_F32=kernels.INTERPRET, num_warps=warps)
                        else:
                            dI._pred_gemv_kernel[(tiles,)](
                                iv.flag, I.y, iv.w, lg, rows, n_, n_, K=K, BM=mfma.BM, BN=bn, BK=bk,
                                DOT_F32=kernels.INTERPRET, num_warps=warps)
                    lmb = r0.lm_head.numel() * 2 if flag else 0
                    ivf_item("pred_gemv", lambda m, iv, I, S: pg(m, iv, I, S, False), flag=flag, bytes=lmb)
                    ivf_item("pred_gemv_persistent", lambda m, iv, I, S: pg(m, iv, I, S, True), flag=flag, bytes=lmb,
                             grid=f"{max(1, iv0.pred_grid)} x {iv0.cus}")
                    ivf_item("pred_argmax_pack", lambda m, iv, I, S: dI._pred_argmax_pack_kernel[(rows,)](
                        iv.flag, iv._lb_logits, S.pack, m.lm_head.shape[0], m.lm_head.shape[0], 0, RANK=m.rank,
                        SOLO=False, BLOCK=iv.ARGMAX_BLOCK, num_warps=8), flag=flag)
                    ivf_item("ar_pred", lambda m, iv, I, S: m.tp.allreduce_pred(m.rank, S.pack, iv.flag), flag=flag)
                    ivf_item("final", lambda m, iv, I, S: dI._ivf_final_kernel[(1,)](
                        iv.flag, S.pack, S.ids, rows, RB=iv.rb, num_warps=1), flag=flag)
                # fast path kernels
                for hist in (False, True):
                    ivf_item("topp2_" + ("hist" if hist else "radix"), lambda m, iv, I, S, hist=hist: dI._topp2_kernel[(rows, iv.np)](
                        S.s, iv.sel, iv.nxt, iv.np * iv.c, iv.max_rows * iv.p, iv.max_rows, C=iv.c, P=iv.p,
                        HIST=hist, num_warps=4))
                for warps in (2, 8):
                    ivf_item(f"topp2_radix_w{warps}", lambda m, iv, I, S, warps=warps: dI._topp2_kernel[(rows, iv.np)](
                        S.s, iv.sel, iv.nxt, iv.np * iv.c, iv.max_rows * iv.p, iv.max_rows, C=iv.c, P=iv.p,
                        HIST=False, num_warps=warps))
                ivf_item("gather2", lambda m, iv, I, S: dI._ivf_gather2_kernel[(rows, iv.np * iv.p, iv.ch)](
                    I.y, iv.weights[0], iv.wdelta, iv.ids_cat, iv.off_cat, iv.sel, iv.outv, iv.outi, iv.p,
                    iv.max_rows * iv.p, iv.max_rows * iv.m, iv.part_rows, iv.c + 1, K=K, CH=iv.ch, BR=iv.BR,
                    BK=iv.BK, num_warps=4), bytes=2 * part_b)
                for first in (False, True):
                    rb = max(2, triton.next_power_of_2(rows))
                    ivf_item("local_flag" + ("_first" if first else ""), lambda m, iv, I, S, first=first:
                             dI._ivf_local_flag_kernel[(1,)](
                                 iv.outv, iv.outi, iv.nxt, iv.loc, S.ids, iv.flag, iv.stats, rows, iv.m, iv.max_rows,
                                 iv.part_rows, -1e9, NP=iv.np, RB=rb,
                                 BM=min(1024, max(16, 4096 // rb), triton.next_power_of_2(iv.m)), FIRST=first,
                                 num_warps=4))
                for fast_ in (False, True):
                    def srch(m, iv, I, S, fast_=fast_):
                        g0 = iv.gap
                        iv.gap = -1e9
                        try:
                            if fast_:
                                iv._search_fast(I.y, S.ids)
                            else:
                                iv._search(I.y, S.ids)
                        finally:
                            iv.gap = g0
                    ivf_item("search" + ("_fast" if fast_ else ""), srch,
                             bytes=iv0.cent.numel() * 2 + 2 * part_b)
                del Is
                sync()

    # ------------------------------------------------------------ 2c. the all-rows head pass
    if "rows" not in skip:
        for rows in (4, 8, 10):
            sts = inputs(rows, args.ctx)

            def make_rows(m, sts=sts):
                st = sts[ranks.index(m)]
                return lambda: head(m, st, st.y, st.d)
            bench("head_rows", make_rows, rows=rows, ctx=args.ctx)

    # ------------------------------------------------------------ 3. bitwise, fast vs today
    if "bitwise" not in skip:
        torch.manual_seed(71)
        with torch.inference_mode(), on(r0):
            iv = r0.ivf
            dev = iv.w.device
            scale = float(real["replay"].float().std())
            rnd = (torch.randn((args.random, K), device=dev) * scale).to(torch.bfloat16)
            rnd[:16] = 0  # every centroid score tied
            rnd[16:32] = (torch.randn((16, K), device=dev) * 1e-3).to(torch.bfloat16)  # many ties
            sets = {"random": rnd, "replay": real["replay"], "natural": real["natural"]}
            stats = {}
            for hist in (False, True):
                iv.topp_hist = hist
                try:
                    iv._search_fast(rnd[:1].contiguous(), torch.empty((1,), dtype=torch.int64, device=dev))
                except Exception:  # noqa: BLE001
                    out("error", item=f"search_fast topp={'hist' if hist else 'radix'}",
                        error=traceback.format_exc().replace("\n", " | ")[-1500:])
                    continue
                for name, Y in sets.items():
                    for rows in (1, 2, 3, 8, 16):
                        total = bad = 0
                        first_bad = None
                        for a in range(0, Y.shape[0] - rows + 1, rows):
                            y = Y[a:a + rows].contiguous()
                            for f0, first in ((0, False), (1, False), (1, True)):
                                snaps = []
                                for which in ("old", "fast"):
                                    iv.flag.fill_(f0)
                                    iv.sel.zero_()
                                    iv.nxt.zero_()
                                    iv.loc.zero_()
                                    st0 = iv.stats.clone()
                                    ids = torch.full((rows,), -7, dtype=torch.int64, device=dev)
                                    if which == "old":
                                        if first:
                                            iv.flag.zero_()
                                        iv._search(y, ids)
                                    else:
                                        iv._search_fast(y, ids, first)
                                    snaps.append((iv.sel[:, :rows].clone(), iv.nxt[:, :rows].clone(), iv.loc[:rows].clone(),
                                                  ids.clone(), iv.flag.clone(), iv.stats - st0))
                                total += 1
                                eq = all(torch.equal(x, y_) for x, y_ in zip(snaps[0], snaps[1], strict=True))
                                if not eq:
                                    bad += 1
                                    if first_bad is None:
                                        first_bad = dict(row=a, f0=f0, first=first,
                                                         which=[k for k, (x, y_) in enumerate(zip(snaps[0], snaps[1], strict=True))
                                                                if not torch.equal(x, y_)])
                            if name == "random" and a >= 2000 and rows > 2:
                                break  # random rows: all of them at 1-2 rows, the first 2000 above
                        stats[(hist, name, rows)] = (total, bad)
                        out("bitwise_search", topp="hist" if hist else "radix", kind=name, rows=rows, calls=total,
                            mismatches=bad, first_bad=first_bad)
            iv.topp_hist = draft_ivf.TOPP == "hist"
            iv.flag.zero_()
            out("bitwise_search_summary", all_equal=all(b == 0 for _, b in stats.values()),
                calls=sum(t for t, _ in stats.values()))
        sync()
        # the whole draft on both ranks, eager (predicated full pass + its all-reduce included)
        for gap_name, gap in (("served", GAP0), ("allfb", 1e9)):
            bad = total = 0
            for name in ("replay", "natural"):
                Y = real[name]
                for rows in (1, 2):
                    for a in range(0, min(Y.shape[0], 400) - rows + 1, rows):
                        y = Y[a:a + rows]
                        res = {}
                        for fast_ in (False, True):
                            def run(m, fast_=fast_):
                                iv = m.ivf
                                iv.gap, iv.fast = gap, fast_
                                d = iv.draft(y.to(m.embed.device).contiguous(), m, first=True)
                                return d.cpu(), int(iv.flag.item())
                            res[fast_] = fast._both(run)
                        total += 1
                        a0, a1 = res[False], res[True]  # per rank (drafts, flag), today / fast
                        same = all(torch.equal(a0[k][0], a1[k][0]) and a0[k][1] == a1[k][1] for k in (0, 1))
                        same = same and torch.equal(a0[0][0], a0[1][0])
                        bad += not same
            set_ivf(gap=GAP0, fast_=draft_ivf.FAST, flag=0)
            out("bitwise_draft", gap=gap_name, calls=total, mismatches=bad)
    out("done", seconds=round(time.perf_counter() - started))


if __name__ == "__main__":
    main()
