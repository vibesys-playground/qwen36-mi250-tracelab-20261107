"""Plan 41 step 0: can a long first-round prefill be split across two GCD pairs?

    HIP_VISIBLE_DEVICES=0,1,2,3 ENGINE_PREFILL_GRAPH_BUCKETS=640,896,1280,1792,2048 \
        python -u tools/split_prefill_probe.py [--news 875,1200,1750,2400] [--splits 1200,1750,2400]
        [--groups 1,10] [--reps 12]

Builds two TP2 replicas in one process, pair A on GPUs (0,1) and pair B on (2,3), from one load
of the BF16 checkpoint (as ``Generator.load`` does, minus the decode graphs), with the server's
captured prefill passes (``TPModel.capture_prefill``) for the sizes above. Then:

(a) single: the served one-pair pass (``prefill_packed`` of one uncached segment, a captured
    graph per padded size up to 2048, the eager pass above it, as served) of N tokens on pair A,
    median of ``--reps`` synchronized walls.
(b) split: the same N tokens as two halves. Pair A (owner) runs half 1 (positions 0..n1) on its
    slot; pair B (helper) runs half 2 (n1..N) as a continuation segment on its own slot, its
    layers one group behind A's. Each half pass is ``FastModel._pg_body`` captured as a chain of
    graphs of ``G`` layers (G=1: one layer behind). After A's group k, rank r of A copies its
    layer state of that group to rank r of B over the inter-card link (GDN conv + recurrent
    rows; the attention KV pages of half 1; after the last group the MTP head's KV): a side
    stream on each GPU, cross-device events, no host wait. After B's group k, B copies its
    final rows and half-2 KV pages of that group back into A's slot, so A ends holding the state
    of the whole prompt (as after the single pass). The wall runs from filling the inputs to
    every GPU idle and the logits on GPU 0. Also: each half chain alone, both halves at once
    with no handoff (the floor), the served continuation (half 1 then half 2 on pair A).
(c) P2P copies of 0.004/1/10/40 MB: GPU 0->2, 0->2 with 1->3 at once (the rank-local
    handoff), 2->0, and 0->1 (in-card) for reference.
(d) parity: the split's last-token logits, MTP draft and A's final state (GDN rows, KV of all
    N positions) against the single pass, against the served continuation on pair A, and the
    same chains run back to back on pair A (must be bitwise: the same kernels and values).

No engine code changes: the chained half pass is the engine's own ``_pg_body`` cut at layer
boundaries. Every line starting with RESULT is one JSON record.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

os.environ.setdefault("ENGINE_TP", "2")
os.environ.setdefault("ENGINE_PREFILL_GRAPH_BUCKETS", "640,896,1280,1792,2048")


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def med(xs):
    return round(statistics.median(xs), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="875,1200,1750,2400")
    ap.add_argument("--splits", default="1200,1750,2400")
    ap.add_argument("--groups", default="1,10")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--mb", default="0.004,1,10,40")
    ap.add_argument("--max-len", type=int, default=16384)
    args = ap.parse_args()
    news = [int(n) for n in args.news.split(",")]
    splits = [int(n) for n in args.splits.split(",")]
    group_sizes = [int(g) for g in args.groups.split(",")]
    reps = args.reps

    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM

    from engine import experts, gdn, speculative
    from engine.fast import model as fm
    from engine.fast import prefill_kernels
    from engine.fast import tp as tpmod
    from engine.generator import CHECKPOINT, experts_implementation
    from serve_probe import token_pool

    n_dev = torch.cuda.device_count()
    result("env", devices=n_dev, names=[torch.cuda.get_device_name(i) for i in range(n_dev)],
           torch=torch.__version__, hip=torch.version.hip, buckets=list(tpmod.PG_BUCKETS),
           peer={f"{a}->{b}": torch.cuda.can_device_access_peer(a, b)
                 for a in range(n_dev) for b in range(n_dev) if a != b})
    if n_dev < 4:
        print("need 4 visible GPUs", flush=True)
        return 1
    devs = [torch.device("cuda", i) for i in range(4)]

    # ------------------------------------------------------------------ load: one checkpoint
    # load on GPUs 2,3 (Generator.load's path), pair A sharded from it without freeing, then
    # pair B sharded from it (freeing it layer by layer, as the server does).
    t0 = time.perf_counter()
    experts.register()
    chosen = experts_implementation(torch.version.hip)
    gib = 2**30
    lm = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT["model"], revision=CHECKPOINT["revision"], dtype=torch.bfloat16,
        device_map="auto", max_memory={0: 1 * gib, 1: 1 * gib, 2: 62 * gib, 3: 62 * gib},
        **({"experts_implementation": chosen} if chosen else {}),
    )
    lm.eval()
    installed = gdn.install(lm)
    head = speculative.load_head(lm, CHECKPOINT["model"], CHECKPOINT["revision"])
    placed = sorted({str(p.device) for p in lm.parameters()})
    log(f"loaded in {time.perf_counter() - t0:.1f}s on {placed}; gdn {installed}; mtp {head is not None}")
    opts = dict(slots=4, max_len=args.max_len + 1, mtp=head)
    # ``shard_model`` frees what it shards (``_free``; the embedding, final norm and lm_head
    # directly): keep all of it for the second pair.
    text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
    kept = [(t, t.data) for t in (text.embed_tokens.weight, text.norm.weight, lm.lm_head.weight)]
    real_free = tpmod._free
    tpmod._free = lambda module: None
    try:
        A = tpmod.TPModel(lm, devices=devs[0:2], **opts)
    finally:
        tpmod._free = real_free
        for t, d in kept:
            t.data = d
    del kept, text
    B = tpmod.TPModel(lm, devices=devs[2:4], **opts)
    del lm
    torch.cuda.empty_cache()
    torch.cuda.set_device(devs[0])
    log(f"pairs built at {time.perf_counter() - t0:.1f}s")
    for name, pair in (("A", A), ("B", B)):
        pair.capture_all(decode=False, verify=False)
        result("pair", name=name, devices=[str(d) for d in pair.devices_tp], pg_sizes=sorted(pair.pg),
               pg_error=pair.pg_error, kv_pages=pair.pool.pages,
               reserved_gib=[round(torch.cuda.memory_reserved(d) / gib, 2) for d in pair.devices_tp])
        if not pair.pg:
            return 1
    log(f"prefill graphs captured at {time.perf_counter() - t0:.1f}s")

    L = len(A.ranks[0].layers)
    mtp_index = A.ranks[0].mtp_index if A.ranks[0].mtp is not None else None

    def sync_all() -> None:
        for d in devs:
            torch.cuda.synchronize(d)

    # ------------------------------------------------------------------ chained half pass
    def chain_part(m, buf, size: int, lo: int, hi: int, carry):
        """``FastModel._pg_body`` over layers lo..hi (the embedding before layer 0, the head
        after the last): the same operations in the same order."""
        dev = m.embed.device
        if lo == 0:
            m0 = 2 * size
            ids, nxt = buf[:size], buf[size:m0]
            meta = buf[m0: m0 + 2]
            count, start, pick = buf[m0], buf[m0 + 1], buf[m0 + 4]
            slot, grow, last = buf[m0 + 2: m0 + 3], buf[m0 + 3: m0 + 4], buf[m0 + 5: m0 + 6]
            rows = torch.arange(size, device=dev)
            keep = rows < count
            positions = start + torch.minimum(rows, count - 1)
            table = m.devices[dev].page_table.index_select(0, slot)[0].long()
            scratch = m.slot_pages[m.dummy][0]
            from types import SimpleNamespace
            st = SimpleNamespace(
                meta=meta, keep=keep, grow=grow, table=table, positions=positions,
                page=torch.where(keep, table[positions // fm.PAGE], scratch),
                offset=torch.where(keep, positions % fm.PAGE, rows % fm.PAGE),
                rows=rows, nxt=nxt, pick=pick, last=last,
            )
            x = F.embedding(ids, m.embed)
            h = prefill_kernels.reduce_rms(x, None, None, m.layers[0].input_layernorm.weight, m.eps)[1]
        else:
            st, x, h = carry
        for i in range(lo, hi):
            layer = m.layers[i]
            h = m._pg_gdn(i, h, st) if m.layer_types[i] == "linear_attention" else m._pg_attn(i, h, st)
            x, h = m._reduce_rms(h, x, layer.post_attention_layernorm.weight)
            nxt_w = m.layers[i + 1].input_layernorm.weight if i + 1 < len(m.layers) else m.final_norm
            x, h = m._reduce_rms(m._moe(layer.mlp, h), x, nxt_w)
        if hi < len(m.layers):
            return (st, x, h), None
        hidden = h
        logits = F.linear(hidden.index_select(0, st.last), m.lm_head).float()
        drafts = None
        if m.mtp is not None:
            tokens = torch.where((st.rows == st.last) & (st.pick != 0), m._argmax(logits), st.nxt)
            z = m._mtp_layer(m._mtp_in(hidden, tokens), lambda h: m._pg_attn(m.mtp_index, h, st), decode=False)
            drafts = m._argmax(m._mtp_logits(z.index_select(0, st.last)))
        return (st, x, h), (logits[0], drafts)

    def bounds_for(g: int):
        return [(lo, min(lo + g, L)) for lo in range(0, L, g)]

    chains: dict[tuple[str, int, int], dict] = {}

    def capture_chain(name: str, pair, size: int, g: int) -> None:
        r0, r1 = pair.ranks
        dummy = r0.dummy
        scratch = r0.slot_pages[dummy][0]
        bufs = [m._pg_inputs(size) for m in pair.ranks]
        host = fm.FastModel.pg_host(size, [(7 * j) % 1000 + 100 for j in range(size - 3)], 0, None,
                                    dummy, r0._g(dummy))
        for b in bufs:
            b.copy_(host)
        bounds = bounds_for(g)
        if not hasattr(pair, "_chain_pools"):
            pair._chain_pools = {m.embed.device: torch.cuda.graph_pool_handle() for m in pair.ranks}
            pair._chain_streams = {m.embed.device: torch.cuda.Stream(m.embed.device) for m in pair.ranks}
        pools, streams = pair._chain_pools, pair._chain_streams
        for m in pair.ranks:
            for meta in m.devices.values():
                meta.page_table[dummy].fill_(scratch)
        try:
            # Warm: the chain eagerly on both ranks (the kernels at this size), dummy slot only.
            def warm(m) -> None:
                dev = m.embed.device
                s = streams[dev]
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):
                    carry = None
                    for lo, hi in bounds:
                        carry, _ = chain_part(m, bufs[m.rank], size, lo, hi, carry)
                torch.cuda.current_stream(dev).wait_stream(s)
                torch.cuda.synchronize(dev)

            pair._both(warm)

            def cap(m, mode):
                dev = m.embed.device
                graphs, carries, out = [], [], None
                carry = None
                with torch.cuda.device(dev), torch.no_grad():
                    for lo, hi in bounds:
                        gr = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(gr, pool=pools[dev], stream=streams[dev], capture_error_mode=mode):
                            carry, out = chain_part(m, bufs[m.rank], size, lo, hi, carry)
                        graphs.append(gr)
                        carries.append(carry)
                return graphs, carries, out

            errors = []
            for mode in ("global", "thread_local", "relaxed"):
                try:
                    c0 = cap(r0, mode)
                    c1 = pair._pool.submit(pair._on_rank1, lambda: cap(r1, mode)).result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{mode}: {type(exc).__name__}: {exc}")
                    for d in pair.devices_tp:
                        torch.cuda.synchronize(d)
                    continue
                chains[(name, size, g)] = dict(graphs=[c0[0], c1[0]], carries=[c0[1], c1[1]],
                                               outs=[c0[2], c1[2]], bufs=bufs, bounds=bounds, mode=mode)
                return
            raise RuntimeError(" | ".join(errors))
        finally:
            r0._push_table(dummy)
            for m in pair.ranks:
                for conv, rec in m.gdn_state.values():
                    for row in (dummy, dummy + m.gdn_rows):
                        if row < conv.shape[0]:
                            conv[row].zero_()
                            rec[row].zero_()
            for d in pair.devices_tp:
                torch.cuda.synchronize(d)

    half_sizes = sorted({A._pg_size(n // 2) for n in splits} | {A._pg_size(n - n // 2) for n in splits})
    t1 = time.perf_counter()
    for size in half_sizes:
        for g in group_sizes:
            for name, pair in (("A", A), ("B", B)):
                try:
                    capture_chain(name, pair, size, g)
                except Exception as exc:  # noqa: BLE001
                    result("chain_capture_error", pair=name, size=size, group=g, error=str(exc)[:2000])
    result("chains", captured=sorted(f"{k[0]}:{k[1]}:G{k[2]}" for k in chains),
           seconds=round(time.perf_counter() - t1, 1),
           reserved_gib=[round(torch.cuda.memory_reserved(d) / gib, 2) for d in devs])
    log(f"chains captured at {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------ inputs, slots
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], max(news + splits) * 3 + 100, "synthetic")
    sA, sB = 0, 0
    nmax = max(news + splits)
    for pair, s in ((A, sA), (B, sB)):
        pair.reset(s)
        pair.ranks[0]._ensure_pages(s, nmax)
    zeros = {"A": A._gdn_copies(sA), "B": B._gdn_copies(sB)}
    for name, pair, s in (("A", A, sA), ("B", B, sB)):
        pages = pair.ranks[0].slot_pages[s]
        result("slot_pages", pair=name, first=pages[0], count=len(pages),
               contiguous=pages == list(range(pages[0], pages[0] + len(pages))))

    def zero(name: str) -> None:
        pair, s = (A, sA) if name == "A" else (B, sB)
        pair._gdn_load(s, zeros[name])

    def prompt(n: int):
        return pool[n: 2 * n]  # a fixed window per length (synthetic coding lines)

    def state(pair, slot: int, n: int) -> dict:
        out = {}
        for r, m in enumerate(pair.ranks):
            row = m._g(slot)
            for i, (conv, rec) in m.gdn_state.items():
                out[f"r{r}.{i}.conv"] = conv[row].float().cpu()
                out[f"r{r}.{i}.rec"] = rec[row].float().cpu()
            table = torch.tensor(m.slot_pages[slot], dtype=torch.long)
            pos = torch.arange(0, n)
            page, off = table[pos // fm.PAGE], pos % fm.PAGE
            for i, (kp, vp) in m.kv.items():
                out[f"r{r}.{i}.k"] = kp[page.to(kp.device), :, off.to(kp.device)].float().cpu()
                out[f"r{r}.{i}.v"] = vp[page.to(vp.device), :, off.to(vp.device)].float().cpu()
        return out

    def compare(a, b) -> dict:
        la, da, sa = a
        lb, db, sb = b
        out = {
            "top1_equal": int(la.argmax()) == int(lb.argmax()),
            "logits_max_abs_diff": float((la - lb).abs().max()),
            "drafts_equal": None if da is None else bool(torch.equal(da.cpu(), db.cpu())),
            "bitwise_all": bool(torch.equal(la, lb) and (da is None or torch.equal(da.cpu(), db.cpu()))
                                and all(torch.equal(sa[k], sb[k]) for k in sa)),
        }
        for kind in ("conv", "rec", "k", "v"):
            keys = [k for k in sa if k.endswith("." + kind)]
            diffs = [float((sa[k] - sb[k]).abs().max()) for k in keys]
            scale = max(float(sb[k].abs().max()) for k in keys)
            out[kind] = {"max_abs_diff": max(diffs), "max_abs_ref": scale}
        return out

    # ------------------------------------------------------------------ (a) single
    def run_single(pair, slot: int, ids):
        logits = pair.prefill_packed([(slot, ids, 0)], [None])
        return logits[0], pair.last_drafts

    singles = {}
    with torch.inference_mode():
        for n in news:
            ids = prompt(n)
            walls = []
            for _ in range(reps + 1):
                zero("A")
                sync_all()
                a = time.perf_counter()
                run_single(A, sA, ids)
                sync_all()
                walls.append((time.perf_counter() - a) * 1e3)
            walls = walls[1:]
            singles[n] = med(walls)
            result("single", pair="A", n=n, size=A._pg_size(n), wall_ms=med(walls),
                   min_max=[round(min(walls), 2), round(max(walls), 2)])
        # pair symmetry: B at 1750 (or the largest split)
        n = 1750 if 1750 in news else news[-1]
        walls = []
        for _ in range(reps + 1):
            zero("B")
            sync_all()
            a = time.perf_counter()
            run_single(B, sB, prompt(n))
            sync_all()
            walls.append((time.perf_counter() - a) * 1e3)
        result("single", pair="B", n=n, size=B._pg_size(n), wall_ms=med(walls[1:]))

    # ------------------------------------------------------------------ (c) P2P
    def p2p(routes, nbytes: int, n: int = 20) -> dict:
        srcs = [torch.empty(nbytes, dtype=torch.uint8, device=devs[a]).fill_(1) for a, _ in routes]
        dsts = [torch.empty(nbytes, dtype=torch.uint8, device=devs[b]) for _, b in routes]
        ss = {i: torch.cuda.Stream(devs[i]) for i in range(4)}
        walls, evms = [], []
        for it in range(n + 2):
            sync_all()
            a = time.perf_counter()
            evs = []
            for (sa, sb), src, dst in zip(routes, srcs, dsts):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(ss[sb]), torch.cuda.stream(ss[sa]):
                    e0.record(ss[sa])
                    dst.copy_(src, non_blocking=True)
                    e1.record(ss[sa])
                evs.append((e0, e1))
            sync_all()
            if it >= 2:
                walls.append((time.perf_counter() - a) * 1e3)
                evms.append(max(e0.elapsed_time(e1) for e0, e1 in evs))
        ok = all(bool(d.eq(1).all()) for d in dsts)
        w = med(walls)
        return {"wall_ms": w, "event_ms": med(evms), "gbps_per_route": round(nbytes / 1e6 / max(med(evms), 1e-6), 2),
                "ok": ok}

    for mb in [float(x) for x in args.mb.split(",")]:
        nbytes = int(mb * 1e6)
        for label, routes in (("0->2", [(0, 2)]), ("0->2+1->3", [(0, 2), (1, 3)]), ("2->0", [(2, 0)]),
                              ("0->1", [(0, 1)])):
            try:
                result("p2p", mb=mb, route=label, **p2p(routes, nbytes))
            except Exception as exc:  # noqa: BLE001
                result("p2p_error", mb=mb, route=label, error=f"{type(exc).__name__}: {exc}"[:500])

    # ------------------------------------------------------------------ (b, d) split
    copy_s = {i: torch.cuda.Stream(devs[i]) for i in range(4)}
    back_s = {i: torch.cuda.Stream(devs[i]) for i in range(4)}

    def fill(key, ids, start: int, follow, pair, slot: int) -> None:
        c = chains[key]
        size = key[1]
        host = fm.FastModel.pg_host(size, ids, start, follow, slot, pair.ranks[0]._g(slot))
        for b in c["bufs"]:
            b.copy_(host)

    def page_runs(src_pages, dst_pages):
        """(src lo, dst lo, length) runs of consecutive page ids in both lists."""
        runs = []
        for s, d in zip(src_pages, dst_pages, strict=True):
            if runs and runs[-1][0] + runs[-1][2] == s and runs[-1][1] + runs[-1][2] == d:
                runs[-1][2] += 1
            else:
                runs.append([s, d, 1])
        return runs

    def plan_copies(n: int, g: int):
        """Per group, per rank: (src, dst) views forward (A->B) and back (B->A)."""
        n1 = n // 2
        bounds = bounds_for(g)
        pa, pb = A.ranks[0].slot_pages[sA], B.ranks[0].slot_pages[sB]
        p1 = (n1 + fm.PAGE - 1) // fm.PAGE  # pages holding half 1 (the boundary page in full)
        p_lo, p_hi = n1 // fm.PAGE, (n + fm.PAGE - 1) // fm.PAGE
        fwd_runs = page_runs(pa[:p1], pb[:p1])
        back_runs = page_runs(pb[p_lo:p_hi], pa[p_lo:p_hi])
        fwd, back = [], []
        nbytes_f = nbytes_b = 0
        for gi, (lo, hi) in enumerate(bounds):
            layers = list(range(lo, hi)) + ([mtp_index] if hi == L and mtp_index is not None else [])
            f_r, b_r = [], []
            for r in range(2):
                ma, mb_ = A.ranks[r], B.ranks[r]
                ra, rb = ma._g(sA), mb_._g(sB)
                f, b = [], []
                for i in layers:
                    if i in ma.gdn_state:
                        for ta, tb in zip(ma.gdn_state[i], mb_.gdn_state[i], strict=True):
                            f.append((ta[ra], tb[rb]))
                            b.append((tb[rb], ta[ra]))
                    if i in ma.kv:
                        for ta, tb in zip(ma.kv[i], mb_.kv[i], strict=True):
                            for s, d, k in fwd_runs:
                                f.append((ta[s: s + k], tb[d: d + k]))
                            for s, d, k in back_runs:
                                b.append((tb[s: s + k], ta[d: d + k]))
                if r == 0:
                    nbytes_f += sum(x.numel() * x.element_size() for x, _ in f)
                    nbytes_b += sum(x.numel() * x.element_size() for x, _ in b)
                f_r.append(f)
                b_r.append(b)
            fwd.append(f_r)
            back.append(b_r)
        return fwd, back, nbytes_f, nbytes_b, len(fwd_runs), len(back_runs)

    def copy_group(pairs, src_dev: int, dst_dev: int, streams, after: torch.cuda.Event) -> torch.cuda.Event:
        s_src, s_dst = streams[src_dev], streams[dst_dev]
        s_src.wait_event(after)
        with torch.cuda.stream(s_dst), torch.cuda.stream(s_src):
            for src, dst in pairs:
                dst.copy_(src, non_blocking=True)
        done = torch.cuda.Event()
        done.record(s_dst)
        return done

    def replay(key, k: int, pair) -> list:
        c = chains[key]
        evs = []
        for r, m in enumerate(pair.ranks):
            dev = m.embed.device
            with torch.cuda.device(dev):
                c["graphs"][r][k].replay()
                e = torch.cuda.Event()
                e.record(torch.cuda.current_stream(dev))
                evs.append(e)
        return evs

    def run_split(n: int, g: int, ids, copies):
        """One split pass; returns (logits on GPU 0, drafts, host issue ms)."""
        n1 = n // 2
        ka, kb = ("A", A._pg_size(n1), g), ("B", B._pg_size(n - n1), g)
        fwd, back = copies[0], copies[1]
        a = time.perf_counter()
        fill(ka, ids[:n1], 0, ids[n1], A, sA)
        fill(kb, ids[n1:], n1, None, B, sB)
        for k in range(len(chains[ka]["bounds"])):
            ea = replay(ka, k, A)
            for r in range(2):
                done = copy_group(fwd[k][r], r, 2 + r, copy_s, ea[r])
                torch.cuda.current_stream(devs[2 + r]).wait_event(done)
            eb = replay(kb, k, B)
            for r in range(2):
                copy_group(back[k][r], 2 + r, r, back_s, eb[r])
        for r in range(2):
            # A's slot is whole once both back streams finished.
            ev = torch.cuda.Event()
            ev.record(back_s[r])
            torch.cuda.current_stream(devs[r]).wait_event(ev)
        cb = chains[kb]
        logits = B._gather([o[0] for o in [cb["outs"][0], cb["outs"][1]]]).to(devs[0])
        drafts = cb["outs"][0][1]
        issue = (time.perf_counter() - a) * 1e3
        return logits, drafts, issue

    def run_chain_alone(key, pair, ids, start, follow, slot):
        fill(key, ids, start, follow, pair, slot)
        for k in range(len(chains[key]["bounds"])):
            replay(key, k, pair)

    def run_both_nohandoff(ka, kb, ids, n1):
        fill(ka, ids[:n1], 0, ids[n1], A, sA)
        fill(kb, ids[n1:], n1, None, B, sB)
        for k in range(len(chains[ka]["bounds"])):
            replay(ka, k, A)
            replay(kb, k, B)

    def timed(fn, pre=None) -> tuple[float, list]:
        walls = []
        for _ in range(reps + 1):
            if pre:
                pre()
            sync_all()
            a = time.perf_counter()
            fn()
            sync_all()
            walls.append((time.perf_counter() - a) * 1e3)
        walls = walls[1:]
        return med(walls), [round(min(walls), 2), round(max(walls), 2)]

    def split_case(n, g, ids, n1, ka, kb, ref, sc, half_ms):
        copies = plan_copies(n, g)
        # Same chains back to back on pair A (the split's exact math on one pair).
        zero("A"); sync_all()
        run_chain_alone(ka, A, ids[:n1], 0, ids[n1], sA)
        run_chain_alone(ka, A, ids[n1:], n1, None, sA)
        sync_all()
        ca = chains[ka]
        seqc = (A._gather([ca["outs"][0][0], ca["outs"][1][0]]).float().cpu(),
                ca["outs"][0][1].clone() if ca["outs"][0][1] is not None else None, state(A, sA, n))
        # The split.
        zero("A"); zero("B"); sync_all()
        lg, dr, _ = run_split(n, g, ids, copies); sync_all()
        spl = (lg.float().cpu(), dr.clone() if dr is not None else None, state(A, sA, n))
        issues = []

        def split_once():
            issues.append(run_split(n, g, ids, copies)[2])

        split_ms, split_mm = timed(split_once, lambda: (zero("A"), zero("B")))
        owner_ms = timed(lambda: run_chain_alone(ka, A, ids[:n1], 0, ids[n1], sA), lambda: zero("A"))[0]
        helper_ms = timed(lambda: run_chain_alone(kb, B, ids[n1:], n1, None, sB), lambda: zero("B"))[0]
        floor_ms = timed(lambda: run_both_nohandoff(ka, kb, ids, n1), lambda: (zero("A"), zero("B")))[0]
        rec = {
            "n": n, "group": g, "sync_points": len(chains[ka]["bounds"]), "capture_mode": chains[ka]["mode"],
            "single_ms": singles[n], "split_ms": split_ms, "split_min_max": split_mm,
            "split_over_single": round(split_ms / singles[n], 3),
            "owner_chain_ms": owner_ms, "helper_chain_ms": helper_ms,
            "both_no_handoff_ms": floor_ms, "half_pass_ms": half_ms,
            "handoff_and_lag_ms": round(split_ms - floor_ms, 3),
            "host_issue_ms": med(issues[1:]) if len(issues) > 1 else None,
            "fwd_mb_per_rank": round(copies[2] / 1e6, 2), "back_mb_per_rank": round(copies[3] / 1e6, 2),
            "page_runs": [copies[4], copies[5]],
            "split_vs_chain_seq": compare(spl, seqc),
            "chain_seq_vs_served_cont": compare(seqc, sc),
            "split_vs_single": compare(spl, ref),
        }
        return rec

    def report(summary) -> None:
        print("\n=== plan 41 step 0: single-pair pass (pair A, median of %d; 2400 runs eager as served)" % reps)
        for n in news:
            print(f"  N={n:5d}  {singles[n]:8.2f} ms")
        print("=== split (owner A half 1, helper B half 2 one group behind, state shipped back)")
        print(f"{'N':>5} {'G':>3} {'single':>8} {'split':>8} {'ratio':>6} {'floor':>8} {'half':>8} "
              f"{'hand+lag':>8} {'issue':>7} {'fwdMB':>6} {'backMB':>6}  top1 bitwise(seq) drafts")
        for r in summary:
            sv = r["split_vs_single"]
            print(f"{r['n']:>5} {r['group']:>3} {r['single_ms']:>8} {r['split_ms']:>8} {r['split_over_single']:>6} "
                  f"{r['both_no_handoff_ms']:>8} {r['half_pass_ms']:>8} {r['handoff_and_lag_ms']:>8} "
                  f"{str(r['host_issue_ms']):>7} {r['fwd_mb_per_rank']:>6} {r['back_mb_per_rank']:>6}  "
                  f"{sv['top1_equal']!s:5} {r['split_vs_chain_seq']['bitwise_all']!s:5} {sv['drafts_equal']}")

    summary = []
    with torch.inference_mode():
        for n in splits:
            ids = prompt(n)
            n1 = n // 2
            size1, size2 = A._pg_size(n1), A._pg_size(n - n1)
            if size1 != size2:
                result("split_skip", n=n, why=f"half sizes {size1} != {size2}")
                continue
            # References on pair A: the single pass, its repeat, the served continuation.
            if n not in singles:
                singles[n] = timed(lambda: run_single(A, sA, ids), lambda: zero("A"))[0]
            zero("A"); sync_all()
            ref = run_single(A, sA, ids); sync_all()
            ref = (ref[0].float().cpu(), ref[1].clone() if ref[1] is not None else None, state(A, sA, n))
            zero("A"); sync_all()
            rep = run_single(A, sA, ids); sync_all()
            rep = (rep[0].float().cpu(), rep[1].clone() if rep[1] is not None else None, state(A, sA, n))

            def served_cont():
                A.prefill_packed([(sA, ids[:n1], 0)], [ids[n1]])
                return A.prefill_packed([(sA, ids[n1:], n1)], [None])[0], A.last_drafts

            zero("A"); sync_all()
            sc = served_cont(); sync_all()
            sc = (sc[0].float().cpu(), sc[1].clone() if sc[1] is not None else None, state(A, sA, n))
            cont_ms = timed(served_cont, lambda: zero("A"))[0]
            half_ms = timed(lambda: A.prefill_packed([(sA, ids[:n1], 0)], [ids[n1]]), lambda: zero("A"))[0]
            base = {"n": n, "n1": n1, "half_size": size1, "single_ms": singles[n],
                    "half_pass_ms": half_ms, "served_continuation_ms": cont_ms,
                    "single_repeat_bitwise": compare(ref, rep)["bitwise_all"],
                    "continuation_vs_single": compare(sc, ref)}
            result("split_ref", **base)
            for g in group_sizes:
                ka, kb = ("A", size1, g), ("B", size2, g)
                if ka not in chains or kb not in chains:
                    result("split_skip", n=n, group=g, why="chain not captured")
                    continue
                try:
                    rec = split_case(n, g, ids, n1, ka, kb, ref, sc, half_ms)
                except Exception as exc:  # noqa: BLE001 - report and go on
                    import traceback
                    traceback.print_exc()
                    result("split_error", n=n, group=g, error=f"{type(exc).__name__}: {exc}"[:2000])
                    continue
                result("split", **rec)
                summary.append(rec)
    result("timeouts", A=A.timeouts(), B=B.timeouts())
    report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
