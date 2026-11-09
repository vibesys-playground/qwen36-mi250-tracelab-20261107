"""Plan 43 stage 0: a 4-GCD tensor-parallel (TP4) chained verify step against today's TP2 one.

    python tools/tp4_probe.py [--part all|ref|tp4] [--prompt-len 4000] [--steps 16] [--iters 30]

No served-path change: ``engine/fast/tp.py`` runs exactly two ranks, so this tool builds its
own W-rank composition from the engine's pieces (``TPWModel``): the same ``FastModel`` per
rank with the same fused / MFMA / chained-verify kernels, sharded at quarter width
(GDN k-heads 16/4, v-heads 32/4; attention q-heads 16/4 with the 2 KV heads each replicated
on 2 ranks; routed and shared expert intermediate 512/4; lm_head vocabulary /4), and an
in-graph one-shot LL all-reduce over W peers (``_ll_w_kernel``: each rank writes its partial
as (epoch, 4 data bytes) words into every peer's receive slot, spins on its own slots, then
sums the W partials in rank order in fp32, so every rank gets the same bits). Prefill runs
``TPModel._lockstep`` from one thread (``_pair`` summed over W ranks in rank order).

Parts (``--part all`` runs ``ref`` then ``tp4`` as subprocesses, then prints the verdict):

- ``ref`` (GCDs 0,1): one replica loaded as the server loads it (``Generator.load``, TP2,
  chained verify graphs). 4 slots prefilled with ``--prompt-len``-token replay-like prompts
  (Request Factory's synthetic coding lines), then ``--steps`` teacher-forced chained verify
  steps (k = 3: each slot's next 4 corpus tokens, ``advance(slot, 3)``): per row the top-2
  logits over the whole vocabulary. Twice: prefill through the served path and through
  ``_lockstep`` (the noise floor of the prefill path alone). Then the step time of
  ``verify_chain`` at 2 slots (8 rows) and 4 slots (16 rows), captured graphs, picks read
  by the host, every GCD synchronized, median; and per-op times on rank 0.
- ``tp4`` (GCDs 0-3): the checkpoint loaded over 4 GCDs, sharded into ``TPWModel``; the same
  prefill (lockstep) and teacher-forced steps, compared with ``ref``'s; step time at 2/3/4
  slots (8/12/16 rows), the same steps with the all-reduces off (``Group.solo``: the AR
  share), the in-graph all-reduce alone at 4-64 KB (bitwise check against an fp32 rank-order
  reference), and per-op times of the quarter-width GDN, attention, MoE and GEMVs on rank 0.

Prints ``RESULT <tag> <json>`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUPS = 4  # k = 3 drafts + the last token per slot
MARGIN = 0.5  # a top-1 disagreement where the reference's top-2 margin is below this is a near-tie

# Kernel tiles of the TP4 shard's dense shapes for the MFMA GEMV (``engine.fast.mfma.TILES``
# has the TP2 and whole shapes only): picked like the TP2 entries (a few hundred programs per
# launch), not swept. ``ENGINE_MFMA_TILES`` set by the caller wins.
TP4_TILES = ("3088x2048=32,128,4,2;2048x1024=16,128,4,2;2560x2048=16,128,4,2;"
             "513x2048=16,128,4,8;62080x2048=32,256,4,1")


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


# ================================================================================ driver


def run_part(argv: list[str], env: dict, timeout: int, lines: list[str]) -> int:
    print(f"===== part: {' '.join(argv)} (timeout {timeout}s)", flush=True)
    t0 = time.time()
    # ``timeout`` kills a part that hangs without output (a spinning all-reduce, say).
    proc = subprocess.Popen(["timeout", "-k", "30", str(timeout), sys.executable, *argv], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            cwd=str(ROOT))
    assert proc.stdout is not None
    for line in proc.stdout:
        if "Loading weights" in line or "amdgpu.ids" in line:
            continue
        print(line, end="", flush=True)
        if line.startswith("RESULT "):
            lines.append(line.strip())
    rc = proc.wait()
    print(f"===== part done rc={rc}{' (TIMEOUT)' if rc in (124, 137) else ''} "
          f"s={time.time() - t0:.0f}", flush=True)
    return rc


def driver(args: argparse.Namespace) -> None:
    common = ["--prompt-len", str(args.prompt_len), "--steps", str(args.steps),
              "--iters", str(args.iters), "--ref-file", args.ref_file]
    lines: list[str] = []
    env = dict(os.environ)
    env.update(ENGINE_MTP="1", HIP_VISIBLE_DEVICES="0,1", ENGINE_TP="2")
    rc_ref = run_part([__file__, "--part", "ref", *common], env, args.timeout, lines)
    env = dict(os.environ)
    env.update(ENGINE_MTP="1", HIP_VISIBLE_DEVICES="0,1,2,3", ENGINE_PREFILL_GRAPH="0")
    env.setdefault("ENGINE_MFMA_TILES", TP4_TILES)
    rc_tp4 = run_part([__file__, "--part", "tp4", *common], env, args.timeout, lines)

    res: dict[str, list[dict]] = {}
    for line in lines:
        _, tag, payload = line.split(" ", 2)
        try:
            res.setdefault(tag, []).append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    steps = {(r["impl"], r["rows"], r.get("ar", True)): r["p50_ms"] for r in res.get("step", [])}
    tp4_16 = steps.get(("tp4", 16, True))
    tp4_8 = steps.get(("tp4", 8, True))
    ar = {r["kb"]: r for r in res.get("ar", [])}
    ar_32_64 = [ar[kb]["p50_us"] for kb in (32, 64) if kb in ar]
    par = {r["what"]: r for r in res.get("parity", [])}
    tp4_par = par.get("tp4_vs_ref_lockstep")
    floor = par.get("noise_floor_tp2_served_vs_tp2_lockstep")
    if tp4_par and floor:
        out("parity_vs_floor", tp4_agree=tp4_par["agree"], floor_agree=floor["agree"],
            rows=tp4_par["rows"], tp4_mismatches=tp4_par["mismatches"],
            floor_mismatches=floor["mismatches"])
    failed = []
    if rc_ref != 0 or rc_tp4 != 0:
        failed.append(f"part rc ref={rc_ref} tp4={rc_tp4}")
    if tp4_16 is None or tp4_8 is None:
        failed.append("no TP4 step time")
    checks = res.get("check", [])
    for c in checks:
        if not c.get("ok"):
            failed.append(f"check {c}")
    if tp4_par is None:
        failed.append("no TP4 parity")
    elif tp4_par["mismatches"] > 0:
        failed.append(f"TP4 top-1 mismatches beyond near-ties: {tp4_par['mismatches']}")
    kill = bool(failed) or (tp4_16 is not None and tp4_16 > 12.5) or any(v > 12.0 for v in ar_32_64)
    passed = (not kill and tp4_16 is not None and tp4_8 is not None
              and tp4_16 <= 11.5 and tp4_8 <= 9.0)
    out("verdict", tp4_8_ms=tp4_8, tp4_12_ms=steps.get(("tp4", 12, True)), tp4_16_ms=tp4_16,
        tp2_8_ms=steps.get(("tp2", 8, True)), tp2_16_ms=steps.get(("tp2", 16, True)),
        ar_p50_us_32_64kb=ar_32_64,
        pass_rule="16 rows <= 11.5 ms and 8 rows <= 9.0 ms",
        kill_rule="16 rows > 12.5 ms or in-graph AR > 12 us at 32-64 KB or any mismatch",
        failed=failed, verdict="pass" if passed else ("kill" if kill else "neither"))


# ================================================================================ shared


def make_corpus(tok, limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
    """Replay-like token pool: Request Factory's synthetic coding lines (as spec_step_bench)."""
    ids: list[int] = []
    i = 0
    while len(ids) < limit:
        for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1
    return ids[:limit]


SLOT_STRIDE = 9973


def sync_all(torch) -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def median_ms(torch, fn, iters: int) -> tuple[float, float]:
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        sync_all(torch)
        t0 = time.perf_counter()
        fn()
        sync_all(torch)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return statistics.median(times), times[min(len(times) - 1, int(0.9 * len(times)))]


def prefill_all(fast, corpus, plen: int, chunk: int, fn) -> None:
    for s in range(4):
        fast.reset(s)
    for s in range(4):
        prompt = corpus[SLOT_STRIDE * s : SLOT_STRIDE * s + plen]
        for begin in range(0, plen, chunk):
            end = min(plen, begin + chunk)
            follow = prompt[end] if end < plen else None
            fn(s, prompt[begin:end], begin, follow)


def full_logits(torch, fast, size: int):
    """The last chained step's logits of every row over the whole vocabulary (rank 0's device)."""
    outs = [m._segs(False, GROUPS)[0].outputs[size] for m in fast.ranks]
    dev = outs[0].device
    return torch.cat([o.to(dev).float() for o in outs], dim=-1)


def teacher_forced(torch, fast, corpus, plen: int, steps: int) -> dict:
    """``steps`` chained verify steps over 4 slots (16 rows), each slot fed its next 4 corpus
    tokens; per row the top-2 (values, ids) and the step's picks; whether all ranks' picks agree."""
    slots = [0, 1, 2, 3]
    vals, ids, picks = [], [], []
    ranks_agree = True
    for t in range(steps):
        pos = plen + GROUPS * t
        toks = [corpus[SLOT_STRIDE * s + pos] for s in slots]
        drafts = [[corpus[SLOT_STRIDE * s + pos + j] for j in range(1, GROUPS)] for s in slots]
        _, pk, _, _ = fast.verify_chain(toks, drafts, slots, [pos] * 4)
        size = fast.bucket(4)
        lg = full_logits(torch, fast, size)[: GROUPS * size]
        top = torch.topk(lg, 2, dim=-1)
        vals.append(top.values.cpu())
        ids.append(top.indices.cpu())
        picks.append(torch.tensor(pk))
        statics = [m._chain_static[(size, GROUPS)].cpu() for m in fast.ranks]
        ranks_agree = ranks_agree and all(torch.equal(statics[0], s) for s in statics[1:])
        for s in slots:
            fast.advance(s, GROUPS - 1)
    return {"vals": torch.stack(vals), "ids": torch.stack(ids), "picks": torch.stack(picks),
            "ranks_agree": ranks_agree}


def compare(torch, a: dict, b: dict, what: str) -> dict:
    """Top-1 agreement of every row of ``b`` against reference ``a``; a disagreement is a
    mismatch unless ``a``'s top-2 margin there is below ``MARGIN`` (a near-tie)."""
    ia, ib = a["ids"][..., 0], b["ids"][..., 0]
    margin = a["vals"][..., 0] - a["vals"][..., 1]
    diff = ia != ib
    n = ia.numel()
    dis_margins = sorted(round(float(v), 4) for v in margin[diff])
    same = ~diff
    drift = (a["vals"][..., 0] - b["vals"][..., 0]).abs()[same]
    r = dict(what=what, rows=n, agree=int(n - int(diff.sum())), rate=round(1 - float(diff.float().mean()), 5),
             disagree_ref_margins=dis_margins[:20],
             mismatches=int((diff & (margin >= MARGIN)).sum()),
             top1_logit_drift_max=round(float(drift.max()) if drift.numel() else 0.0, 4),
             top1_logit_drift_mean=round(float(drift.mean()) if drift.numel() else 0.0, 5))
    out("parity", **r)
    return r


def op_times(torch, m, rows_list, plen: int, iters: int, label: str) -> None:
    """Per-op time of one rank's fused kernels at a chained step's rows (G = 4 rows per slot),
    each op over all its layers in one captured graph on that rank alone (no all-reduce);
    random hidden states (so the MoE's routing is near-uniform: more unique experts than real
    text, an upper bound)."""
    from engine.fast import kernels, mfma

    fd = m.fused
    dev = m.embed.device
    meta = m.devices[dev]
    hidden = m.config.hidden_size
    gdn = [i for i in range(len(m.layers)) if m.layer_types[i] == "linear_attention"]
    attn = [i for i in range(len(m.layers)) if m.layer_types[i] != "linear_attention"]
    allx = list(range(len(m.layers)))
    shapes = {}
    for i in (gdn[0], attn[0]):
        layer = m.layers[i]
        o = layer.linear_attn.out_proj.weight if i in gdn else layer.self_attn.o_proj.weight
        for w in (fd.w_in[i], fd.w_moe[i], o):
            shapes.setdefault(tuple(w.shape), w)
    shapes.setdefault(tuple(m.lm_head.shape), m.lm_head)
    torch.manual_seed(0)
    with torch.cuda.device(dev), torch.inference_mode():
        stream = torch.cuda.Stream(dev)
        for rows in rows_list:
            b = rows // GROUPS
            order = [(j, s) for j in range(GROUPS) for s in range(b)]
            slots = torch.tensor([s for _, s in order], dtype=torch.long, device=dev)
            positions = torch.tensor([plen + j for j, _ in order], dtype=torch.long, device=dev)
            states = torch.tensor([m._ring_row(s, j) for j, s in order], dtype=torch.long, device=dev)
            h = (torch.randn((rows, hidden), device=dev) * 0.5).to(torch.bfloat16)
            x = (torch.randn((rows, hidden), device=dev) * 2.0).to(torch.bfloat16)
            w0 = m.layers[0].post_attention_layernorm.weight
            ops = {
                "gdn": (gdn, lambda i: fd._gdn_verify(i, h, states, groups=GROUPS)),
                "attn": (attn, lambda i: fd._attn(i, h, slots, positions, meta)),
                "moe": (allx, lambda i: fd._moe(i, None, h)),
                "add_rms": (allx, lambda i: kernels.add_rms(x, h, w0, m.eps)),
            }
            for (n, k), w in shapes.items():
                xin = (torch.randn((rows, k), device=dev) * 0.5).to(torch.bfloat16)
                ops[f"gemv_{n}x{k}"] = ([0], lambda i, w=w, xin=xin: mfma.gemv(xin, w))
            for name, (layers, fn) in ops.items():
                stream.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        for i in layers:
                            fn(i)
                torch.cuda.current_stream(dev).wait_stream(stream)
                torch.cuda.synchronize(dev)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    for i in layers:
                        fn(i)
                for _ in range(3):
                    g.replay()
                times = []
                for _ in range(iters):
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize(dev)
                    times.append((time.perf_counter() - t0) * 1e3)
                p50 = statistics.median(times)
                out("op", impl=label, rows=rows, op=name, layers=len(layers),
                    per_layer_us=round(p50 * 1e3 / len(layers), 1), total_ms=round(p50, 4))
                del g


def step_times(torch, fast, corpus, plen: int, batches, iters: int, label: str, ar: bool = True) -> None:
    for batch in batches:
        slots = list(range(batch))
        pos = plen + 200
        toks = [corpus[SLOT_STRIDE * s + pos] for s in slots]
        drafts = [[corpus[SLOT_STRIDE * s + pos + j] for j in range(1, GROUPS)] for s in slots]
        p50, p90 = median_ms(torch, lambda: fast.verify_chain(toks, drafts, slots, [pos] * batch), iters)
        size = fast.bucket(batch)
        out("step", impl=label, ar=ar, slots=batch, rows=GROUPS * batch, padded_rows=GROUPS * size,
            p50_ms=round(p50, 3), p90_ms=round(p90, 3))


# ================================================================================ ref (TP2)


def part_ref(args: argparse.Namespace) -> None:
    import torch

    from engine.generator import CHECKPOINT, Generator

    t0 = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    if not hasattr(fast, "ranks") or not fast.chain:
        raise SystemExit(f"expected the TP2 chained path, got {type(fast).__name__}")
    out("load", impl="tp2", seconds=round(time.perf_counter() - t0), graph_error=fast.graph_error,
        pg=len(fast.pg), kv_pages=fast.pool.pages, buckets=fast.buckets)
    corpus = make_corpus(g.tokenizer, SLOT_STRIDE * 4 + 1000)
    plen = args.prompt_len

    prefill_all(fast, corpus, plen, args.chunk, lambda s, ids, b, f: fast.prefill(s, ids, b, f))
    served = teacher_forced(torch, fast, corpus, plen, args.steps)

    def lock(s, ids, b, f):
        fast.ranks[0]._ensure_pages(s, b + len(ids))
        return fast._lockstep([(s, ids, b)], [f])[0]

    prefill_all(fast, corpus, plen, args.chunk, lock)
    locked = teacher_forced(torch, fast, corpus, plen, args.steps)
    out("check", what="tp2_ranks_agree", ok=bool(served["ranks_agree"] and locked["ranks_agree"]))
    compare(torch, served, locked, "noise_floor_tp2_served_vs_tp2_lockstep")
    torch.save({"served": served, "lockstep": locked}, args.ref_file)

    step_times(torch, fast, corpus, plen, (2, 4), args.iters, "tp2")
    out("check", what="tp2_ar_timeouts", ok=fast.timeouts() == 0, timeouts=fast.timeouts())
    op_times(torch, fast.ranks[0], (8, 16), plen, args.iters, "tp2")


# ================================================================================ TP4


def build_tpw():
    """The W-rank composition: ``TPWModel`` and its group, from the engine's own pieces."""
    import torch
    from torch.nn import functional as F  # noqa: F401

    from engine.fast import kernels, tp
    from engine.fast.model import FastModel

    triton, tl = kernels.triton, kernels.tl
    CHUNK = tp.CHUNK
    CAP = 1 << 16  # words per (source, parity) slot: 64 rows x 2048 BF16
    MAXIT = 4_000_000

    @triton.jit(do_not_specialize=["NW", "maxit"])
    def _ll_w_kernel(x_ptr, res_ptr, out_ptr, b0, b1, b2, b3, epoch_ptr, err_ptr, NW, CAPW, maxit,
                     RANK: tl.constexpr, W: tl.constexpr, CHUNK: tl.constexpr, BF16: tl.constexpr,
                     RES: tl.constexpr):
        """out = [res +] sum over ranks q = 0..W-1 (in that order, fp32) of x_q, as ``tp``'s
        two-rank kernel: LL words (epoch << 32 | 4 data bytes), write-through to every peer's
        slot for this rank, uncached spins on this rank's slots, double-buffered by epoch."""
        p = tl.program_id(0)
        e = tl.load(epoch_ptr + p) + 1
        par = (e % 2).to(tl.int64)
        offs = p * CHUNK + tl.arange(0, CHUNK)
        m = offs < NW
        xi = tl.load(x_ptr + offs, mask=m, other=0)
        word = (e.to(tl.int64) << 32) | (xi.to(tl.int64) & 0xFFFFFFFF)
        mine = (RANK * 2 + par) * CAPW + offs
        if RANK != 0:
            tl.store(b0 + mine, word, mask=m, cache_modifier=".wt")
        if RANK != 1:
            tl.store(b1 + mine, word, mask=m, cache_modifier=".wt")
        if W > 2:
            if RANK != 2:
                tl.store(b2 + mine, word, mask=m, cache_modifier=".wt")
            if RANK != 3:
                tl.store(b3 + mine, word, mask=m, cache_modifier=".wt")
        if RANK == 0:
            own = b0
        elif RANK == 1:
            own = b1
        elif RANK == 2:
            own = b2
        else:
            own = b3
        s_lo = tl.zeros([CHUNK], dtype=tl.float32)
        s_hi = tl.zeros([CHUNK], dtype=tl.float32)
        bad = 0
        for q in tl.static_range(W):
            if q == RANK:
                yi = xi
            else:
                src = own + (q * 2 + par) * CAPW + offs
                w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
                pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
                it = pending * 0
                while (pending > 0) & (it < maxit):
                    w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
                    pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
                    it += 1
                bad += pending
                yi = (w & 0xFFFFFFFF).to(tl.int32)
            if BF16:
                s_lo += ((yi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
                s_hi += (yi & -65536).to(tl.float32, bitcast=True)
            else:
                s_lo += yi.to(tl.float32, bitcast=True)
        if bad > 0:
            tl.atomic_add(err_ptr, 1)
        if BF16:
            s_lo = s_lo.to(tl.bfloat16).to(tl.float32)
            s_hi = s_hi.to(tl.bfloat16).to(tl.float32)
            if RES:
                ri = tl.load(res_ptr + offs, mask=m, other=0)
                s_lo = (((ri & 0xFFFF) << 16).to(tl.float32, bitcast=True) + s_lo).to(tl.bfloat16).to(tl.float32)
                s_hi = ((ri & -65536).to(tl.float32, bitcast=True) + s_hi).to(tl.bfloat16).to(tl.float32)
            lo = s_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            hi = s_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
            res = lo | hi
        else:
            s = s_lo
            if RES:
                s = tl.load(res_ptr + offs, mask=m, other=0).to(tl.float32, bitcast=True) + s
            res = s.to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, res, mask=m)
        tl.store(epoch_ptr + p, e)

    @triton.jit
    def _argmax_pack_w(l_ptr, both_ptr, V, stride, offset, RANK: tl.constexpr, W2: tl.constexpr,
                       BLOCK: tl.constexpr):
        """``tp._argmax_pack_kernel`` over W ranks: (max, first index + offset) in columns
        2 RANK, 2 RANK + 1 of 2W, zeros elsewhere."""
        row = tl.program_id(0)
        base = l_ptr + row.to(tl.int64) * stride
        lanes = tl.arange(0, BLOCK)
        best = tl.full([BLOCK], float("-inf"), tl.float32)
        bidx = tl.zeros([BLOCK], dtype=tl.int32)
        for v0 in range(0, V, BLOCK):
            cols = v0 + lanes
            x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
            upd = x > best
            best = tl.where(upd, x, best)
            bidx = tl.where(upd, cols, bidx)
        mx = tl.max(best, axis=0)
        i = tl.min(tl.where(best == mx, bidx, V), axis=0)
        i = tl.where(mx == float("-inf"), 0, i)
        k = tl.arange(0, W2)
        val = tl.where(k == 2 * RANK, mx, tl.where(k == 2 * RANK + 1, (i + offset).to(tl.float32), 0.0))
        tl.store(both_ptr + row * W2 + k, val)

    @triton.jit
    def _argmax_pick_w(both_ptr, out_ptr, R, W: tl.constexpr, RB: tl.constexpr):
        """The larger max over the W ranks' (max, index) pairs, ties to the lower rank."""
        r = tl.arange(0, RB)
        ok = r < R
        best = tl.load(both_ptr + r * (2 * W), mask=ok, other=0.0)
        idx = tl.load(both_ptr + r * (2 * W) + 1, mask=ok, other=0.0)
        for q in tl.static_range(1, W):
            v = tl.load(both_ptr + r * (2 * W) + 2 * q, mask=ok, other=0.0)
            j = tl.load(both_ptr + r * (2 * W) + 2 * q + 1, mask=ok, other=0.0)
            take = v > best
            best = tl.where(take, v, best)
            idx = tl.where(take, j, idx)
        tl.store(out_ptr + r, idx.to(tl.int64), mask=ok)

    class LLW:
        """Fine-grained receive buffers ``[W sources][2 parities][CAP]`` of 8-byte words per
        rank, per-CTA epochs, and the launch."""

        def __init__(self, devices) -> None:
            self.devices = list(devices)
            self.w = len(self.devices)
            assert 2 <= self.w <= 4
            self.hip = hip = tp._hip()
            for a in range(self.w):
                for b in range(self.w):
                    if a == b:
                        continue
                    if hip.hipSetDevice(self.devices[a].index) != 0:
                        raise RuntimeError("hipSetDevice failed")
                    rc = hip.hipDeviceEnablePeerAccess(self.devices[b].index, 0)
                    if rc not in (0, 704):
                        raise RuntimeError(f"hipDeviceEnablePeerAccess({a}->{b}) failed: {rc}")
            torch.cuda.set_device(self.devices[0])
            self.nbytes = self.w * 2 * CAP * 8
            self.recv = [self._fine(d) for d in self.devices]
            self.epoch = [torch.zeros(CAP // CHUNK, dtype=torch.int32, device=d) for d in self.devices]
            self.err = [torch.zeros(1, dtype=torch.int32, device=d) for d in self.devices]
            # Compile and load every variant on every GPU with no spin (maxit 0), then clear the
            # buffers, epochs and error counts those launches left.
            for rank, dev in enumerate(self.devices):
                for dtype, residual in ((torch.bfloat16, False), (torch.bfloat16, True), (torch.float32, False)):
                    x = torch.zeros((1, 256), dtype=dtype, device=dev)
                    self.launch(rank, x, x if residual else None, maxit=0)
            for dev in self.devices:
                torch.cuda.synchronize(dev)
            self.clear()

        def _fine(self, dev):
            ptr = ctypes.c_void_p()
            hip = self.hip
            if hip.hipSetDevice(dev.index) != 0:
                raise RuntimeError("hipSetDevice failed")
            rc = hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(self.nbytes), ctypes.c_uint(1))
            if rc != 0:
                raise RuntimeError(f"hipExtMallocWithFlags failed: {rc}")
            torch.cuda.set_device(self.devices[0])
            return tp._Raw(ptr.value, torch.int64)

        def clear(self) -> None:
            for dev, buf in zip(self.devices, self.recv, strict=True):
                self.hip.hipSetDevice(dev.index)
                if self.hip.hipMemset(ctypes.c_void_p(buf.ptr), 0, ctypes.c_size_t(self.nbytes)) != 0:
                    raise RuntimeError("hipMemset failed")
                self.hip.hipDeviceSynchronize()
            torch.cuda.set_device(self.devices[0])
            for e in self.epoch + self.err:
                e.zero_()
            for dev in self.devices:
                torch.cuda.synchronize(dev)

        def launch(self, rank: int, x, residual=None, maxit: int = MAXIT):
            x = x.contiguous()
            nbytes = x.numel() * x.element_size()
            nw = nbytes // 4
            if nbytes % 4 or nw > CAP or x.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"all-reduce of {tuple(x.shape)} {x.dtype} is not supported")
            out = torch.empty_like(x)
            res = residual.contiguous() if residual is not None else x
            bufs = list(self.recv) + [self.recv[rank]] * (4 - self.w)
            with torch.cuda.device(x.device):
                _ll_w_kernel[(triton.cdiv(nw, CHUNK),)](
                    x.view(-1).view(torch.int32), res.view(-1).view(torch.int32),
                    out.view(-1).view(torch.int32), *bufs, self.epoch[rank], self.err[rank],
                    nw, CAP, maxit, RANK=rank, W=self.w, CHUNK=CHUNK,
                    BF16=x.dtype == torch.bfloat16, RES=residual is not None, num_warps=1,
                )
            return out

        def timeouts(self) -> int:
            return sum(int(e.item()) for e in self.err)

    class GroupW:
        """``tp.Group``'s interface over W ranks: in-graph all-reduce (``LLW``) and the greedy
        pick; ``solo`` returns the local partial (warm-up, or a step without all-reduces)."""

        def __init__(self, devices) -> None:
            self.devices = list(devices)
            self.w = len(self.devices)
            self.solo = False
            self.ll = LLW(self.devices)

        def start(self) -> None:
            pass

        def abort(self, why: str) -> None:
            pass

        def _exchange(self, rank, x):
            raise RuntimeError("TPWModel has no eager all-reduce (prefill runs in lock-step)")

        def allreduce(self, rank, x, residual=None):
            x = x.contiguous()
            if self.solo:
                return x if residual is None else residual + x
            if x.is_cuda and torch.cuda.is_current_stream_capturing():
                return self.ll.launch(rank, x, residual)
            raise RuntimeError("TPWModel all-reduces only inside a captured graph")

        def allreduce_rms(self, rank, x, residual, weight, eps):
            return kernels.add_rms(residual, self.allreduce(rank, x), weight, eps)

        def argmax(self, rank, logits, offset):
            logits = logits.contiguous()
            rows, v = logits.shape
            both = torch.empty((rows, 2 * self.w), dtype=torch.float32, device=logits.device)
            with kernels.on_device(logits):
                _argmax_pack_w[(rows,)](logits, both, v, logits.stride(0), offset, RANK=rank,
                                        W2=2 * self.w, BLOCK=tp.ARGMAX_BLOCK, num_warps=8)
            both = self.allreduce(rank, both)
            picks = torch.empty((rows,), dtype=torch.long, device=logits.device)
            with kernels.on_device(both):
                _argmax_pick_w[(1,)](both, picks, rows, W=self.w,
                                     RB=max(16, triton.next_power_of_2(rows)), num_warps=1)
            return picks

        def timeouts(self) -> int:
            return self.ll.timeouts()

    def attn_shard_w(m, q_heads, kv_heads, rank, ranks, dev):
        """Query heads ``[q2 r, q2 (r + 1))`` and the KV heads they read (replicated when
        there are more ranks than KV heads: rank r reads KV head ``q2 r // (q_heads / kv_heads)``)."""
        d = m.head_dim
        q2 = q_heads // ranks
        kv2 = max(1, kv_heads // ranks)
        kv0 = (rank * q2) // (q_heads // kv_heads)
        return tp.NS(
            q_proj=tp.NS(weight=tp._put(m.q_proj.weight[rank * q2 * 2 * d : (rank + 1) * q2 * 2 * d], dev)),
            k_proj=tp.NS(weight=tp._put(m.k_proj.weight[kv0 * d : (kv0 + kv2) * d], dev)),
            v_proj=tp.NS(weight=tp._put(m.v_proj.weight[kv0 * d : (kv0 + kv2) * d], dev)),
            o_proj=tp.NS(weight=tp._put(m.o_proj.weight[:, rank * q2 * d : (rank + 1) * q2 * d], dev)),
            q_norm=tp._norm(m.q_norm, dev),
            k_norm=tp._norm(m.k_norm, dev),
            scaling=m.scaling,
            head_dim=d,
        )

    def layer_shard_w(layer, kind, cfg, rank, ranks, dev):
        o = tp.NS(
            input_layernorm=tp._norm(layer.input_layernorm, dev),
            post_attention_layernorm=tp._norm(layer.post_attention_layernorm, dev),
            mlp=tp._moe_shard(layer.mlp, rank, ranks, dev),
        )
        if kind == "linear_attention":
            o.linear_attn = tp._gdn_shard(layer.linear_attn, rank, ranks, dev)
        else:
            o.self_attn = attn_shard_w(layer.self_attn, cfg.num_attention_heads,
                                       cfg.num_key_value_heads, rank, ranks, dev)
        return o

    def shard_model_w(lm, mtp, devices):
        """``tp.shard_model`` for W ranks (KV heads replicated when W exceeds them)."""
        ranks = len(devices)
        text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
        cfg = text.config
        rank_cfg = copy.copy(cfg)
        rank_cfg.num_attention_heads = cfg.num_attention_heads // ranks
        rank_cfg.num_key_value_heads = max(1, cfg.num_key_value_heads // ranks)
        vocab = lm.lm_head.weight.shape[0]
        bounds = [math.ceil(vocab * r / ranks) for r in range(ranks + 1)]
        layers = [[] for _ in devices]
        for i, layer in enumerate(text.layers):
            for r, dev in enumerate(devices):
                layers[r].append(layer_shard_w(layer, cfg.layer_types[i], cfg, r, ranks, dev))
            tp._free(layer)
            torch.cuda.empty_cache()
        heads = [None] * ranks
        if mtp is not None:
            for r, dev in enumerate(devices):
                heads[r] = tp.NS(
                    fc=tp.NS(weight=tp._put(mtp.fc.weight, dev)),
                    pre_fc_norm_embedding=tp._norm(mtp.pre_fc_norm_embedding, dev),
                    pre_fc_norm_hidden=tp._norm(mtp.pre_fc_norm_hidden, dev),
                    layers=[layer_shard_w(mtp.layers[0], "full_attention", cfg, r, ranks, dev)],
                    norm=tp._norm(mtp.norm, dev),
                )
            tp._free(mtp)
        shards = []
        for r, dev in enumerate(devices):
            shards.append(tp.NS(
                model=tp.NS(language_model=tp.NS(
                    config=rank_cfg,
                    embed_tokens=tp.NS(weight=tp._put(text.embed_tokens.weight, dev)),
                    norm=tp.NS(weight=tp._put(text.norm.weight, dev)),
                    layers=layers[r],
                    rotary_emb=tp.NS(inv_freq=text.rotary_emb.inv_freq.detach().clone(),
                                     attention_scaling=text.rotary_emb.attention_scaling),
                )),
                lm_head=tp.NS(weight=tp._put(lm.lm_head.weight[bounds[r] : bounds[r + 1]], dev)),
                vocab_offset=bounds[r],
            ))
        for t in (text.embed_tokens.weight, text.norm.weight, lm.lm_head.weight):
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        torch.cuda.empty_cache()
        return list(zip(shards, heads, strict=True))

    class TPWModel(tp.TPModel):
        """``tp.TPModel`` over W ranks: decode/verify graphs replayed per rank from one thread
        (the ranks rendezvous inside them), lock-step prefill, warm-up rank by rank alone."""

        def __init__(self, lm, *, slots, max_len, mtp, devices, kv_tokens, buckets) -> None:
            self.devices_tp = list(devices)
            self.config = (lm.model.language_model if hasattr(lm.model, "language_model") else lm.model).config
            kernels.ATTN_SPLITS = int(os.environ.get("ENGINE_TP_ATTN_SPLITS", "64"))
            shards = shard_model_w(lm, mtp, self.devices_tp)
            self.group = GroupW(self.devices_tp)
            self.ranks = []
            for r, (shard, head) in enumerate(shards):
                model = FastModel(shard, slots=slots, max_len=max_len, kv_tokens=kv_tokens,
                                  buckets=buckets, mtp=head)
                model.tp, model.rank, model.vocab_offset = self.group, r, shard.vocab_offset
                self.ranks.append(model)
            r0 = self.ranks[0]
            for m in self.ranks[1:]:
                if m.pool.pages != r0.pool.pages or len(m.segments) != 1:
                    raise RuntimeError("ranks differ in KV pages or segments")
                m.pool, m.slot_pages, m.bank, m._lock = r0.pool, r0.slot_pages, r0.bank, r0._lock
            for m in self.ranks:
                m._peers = [o for o in self.ranks if o is not m]
            r0._push_table(r0.dummy)
            self.vocab = sum(r.vocab for r in self.ranks)
            self.last_drafts = None
            self.graph_error = None
            self.capture_mode = None
            self._pool = None
            self.pg = {}
            self.pg_error = "off (TPWModel)"
            self.pg_replays = 0

        def _pair(self, parts, residuals=None):
            """The W-rank sum in rank order in fp32 (the in-graph kernel's order), per rank."""
            o = []
            for r, part in enumerate(parts):
                acc = parts[0].to(part.device).float()
                for q in range(1, len(parts)):
                    acc = acc + parts[q].to(part.device).float()
                total = acc.to(part.dtype)
                o.append(total if residuals is None else residuals[r] + total)
            return o

        @torch.inference_mode()
        def prefill(self, slot, ids, start, follow=None):
            self.ranks[0]._ensure_pages(slot, start + len(ids))
            return self._lockstep([(slot, ids, start)], [follow])[0]

        def capture(self, size, verify=False, chain=0):
            group = self.group
            group.solo = True
            try:
                for m in self.ranks:
                    with tp._on(m), torch.inference_mode():
                        self._warm(m, size, verify, chain)
            finally:
                group.solo = getattr(self, "_solo_steps", False)
            modes = getattr(self, "_capture_modes", None) or ["global", "thread_local", "relaxed"]
            errors = []
            for mode in modes:
                results = []
                for m in self.ranks:
                    with tp._on(m):
                        results.append(m._capture(size, mode, verify, chain))
                if all(r is None for r in results):
                    self._capture_modes = [mode]
                    self.capture_mode = mode
                    return
                self.clear_graphs()
                errors.append(f"{mode}: {[r for r in results if r]}")
            self.graph_error = " | ".join(errors)

        def clear_graphs(self) -> None:
            for m in self.ranks:
                for seg in m._all_segments():
                    seg.graphs.clear()
                    seg.inputs.clear()
                    seg.outputs.clear()
                m.__dict__.pop("_capture_streams", None)
            for dev in self.devices_tp:
                torch.cuda.synchronize(dev)

    return TPWModel


def ar_bench(torch, fast, rows_list, calls: int, reps: int) -> None:
    """The in-graph W-rank all-reduce alone (with the residual, as a layer's second one): one
    graph of ``calls`` calls per rank, replayed on every rank; and one captured call's result
    against the fp32 rank-order reference, bitwise, on every rank."""
    group = fast.group
    ll = group.ll
    devs = fast.devices_tp
    for rows in rows_list:
        torch.manual_seed(rows)
        xs_cpu = [(torch.randn((rows, 2048)) * 0.5).to(torch.bfloat16) for _ in devs]
        res_cpu = (torch.randn((rows, 2048)) * 2).to(torch.bfloat16)
        acc = xs_cpu[0].float()
        for x in xs_cpu[1:]:
            acc = acc + x.float()
        ref = (res_cpu.float() + acc.to(torch.bfloat16).float()).to(torch.bfloat16)
        xs = [x.to(d) for x, d in zip(xs_cpu, devs, strict=True)]
        rs = [res_cpu.to(d) for d in devs]
        # one call, checked
        graphs, outs = [], []
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                s = torch.cuda.Stream(d)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    outs.append(ll.launch(r, xs[r], rs[r]))
                graphs.append(g)
        for _ in range(3):
            for g, d in zip(graphs, devs, strict=True):
                with torch.cuda.device(d):
                    g.replay()
            sync_all(torch)
        exact = all(torch.equal(o.cpu(), ref) for o in outs)
        # many calls, timed
        graphs = []
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                s = torch.cuda.Stream(d)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    for _ in range(calls):
                        ll.launch(r, xs[r], rs[r])
                graphs.append(g)

        def replay(graphs=graphs):
            for g, d in zip(graphs, devs, strict=True):
                with torch.cuda.device(d):
                    g.replay()

        p50, p90 = median_ms(torch, replay, reps)
        out("ar", world=len(devs), rows=rows, kb=rows * 2048 * 2 // 1024, p50_us=round(p50 * 1e3 / calls, 2),
            p90_us=round(p90 * 1e3 / calls, 2), calls=calls, bitwise_vs_ref=exact, timeouts=ll.timeouts())
        out("check", what=f"ar_bitwise_{rows}rows", ok=bool(exact))


def part_tp4(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine import experts, gdn, speculative
    from engine.fast import mfma
    from engine.generator import CHECKPOINT, experts_implementation

    if torch.cuda.device_count() != 4:
        raise SystemExit(f"need 4 visible GPUs, have {torch.cuda.device_count()}")
    t0 = time.perf_counter()
    name, rev = CHECKPOINT["model"], CHECKPOINT["revision"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    experts.register()
    chosen = experts_implementation(torch.version.hip)
    lm = AutoModelForCausalLM.from_pretrained(
        name, revision=rev, dtype=torch.bfloat16, device_map="auto",
        **({"experts_implementation": chosen} if chosen else {}))
    lm.eval()
    gdn.install(lm)
    head = speculative.load_head(lm, name, rev)
    if head is None:
        raise SystemExit("no MTP head")
    TPWModel = build_tpw()
    devices = [torch.device("cuda", i) for i in range(4)]
    fast = TPWModel(lm, slots=8, max_len=32768 + 1, mtp=head, devices=devices,
                    kv_tokens=8 * 8192, buckets=(1, 2, 3, 4, 8))
    del lm
    torch.cuda.empty_cache()
    if not fast.chain:
        raise SystemExit("the chained verify is off (needs ENGINE_SPEC_CHAIN=1 and Triton)")
    r0 = fast.ranks[0]
    out("load", impl="tp4", seconds=round(time.perf_counter() - t0), kmax=mfma.KMAX,
        mem_gb=[round(torch.cuda.memory_allocated(d) / 2**30, 1) for d in devices],
        shard=dict(gdn_in=list(r0.fused.w_in[0].shape), moe_in=list(r0.fused.w_moe[0].shape),
                   lm_head=list(r0.lm_head.shape), kv_heads=r0.config.num_key_value_heads,
                   q_heads=r0.config.num_attention_heads),
        mfma_tiles=os.environ.get("ENGINE_MFMA_TILES", ""))
    corpus = make_corpus(tok, SLOT_STRIDE * 4 + 1000)
    plen = args.prompt_len

    # The in-graph all-reduce alone first (epochs advance identically on every rank).
    ar_bench(torch, fast, (1, 4, 8, 12, 16), calls=50, reps=args.iters)

    t1 = time.perf_counter()
    prefill_all(fast, corpus, plen, args.chunk, lambda s, ids, b, f: fast.prefill(s, ids, b, f))
    out("prefill", impl="tp4_lockstep", seconds=round(time.perf_counter() - t1, 1))
    mine = teacher_forced(torch, fast, corpus, plen, args.steps)
    out("check", what="tp4_ranks_agree", ok=bool(mine["ranks_agree"]))
    out("check", what="tp4_graphs", ok=fast.graph_error is None, graph_error=fast.graph_error)
    ref = torch.load(args.ref_file)
    compare(torch, ref["lockstep"], mine, "tp4_vs_ref_lockstep")
    compare(torch, ref["served"], mine, "tp4_vs_ref_served")

    step_times(torch, fast, corpus, plen, (2, 3, 4), args.iters, "tp4")
    out("check", what="tp4_ar_timeouts", ok=fast.timeouts() == 0, timeouts=fast.timeouts())

    # The same steps without the all-reduces: each rank alone (wrong numbers, right kernels).
    # Per-op first (j209: the solo capture below failed and fell back to tp.py's 2-rank eager path).
    op_times(torch, r0, (8, 12, 16), plen, args.iters, "tp4")

    fast.clear_graphs()
    fast._solo_steps = True
    fast.group.solo = True
    try:
        for batch in (2, 4):
            fast.capture(fast.bucket(batch), False, GROUPS)
            if fast.graph_error is not None:
                raise RuntimeError(fast.graph_error)
        step_times(torch, fast, corpus, plen, (2, 4), args.iters, "tp4", ar=False)
    except Exception as exc:  # noqa: BLE001 - informational only
        out("noar_failed", error=f"{type(exc).__name__}: {str(exc)[:600]}")
    fast.group.solo = False
    fast._solo_steps = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all", choices=("all", "ref", "tp4"))
    parser.add_argument("--prompt-len", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=2400, help="seconds per part")
    parser.add_argument("--ref-file", default="/tmp/tp4_probe_ref.pt")
    args = parser.parse_args()
    if args.part == "all":
        driver(args)
    elif args.part == "ref":
        part_ref(args)
    else:
        part_tp4(args)


if __name__ == "__main__":
    main()
