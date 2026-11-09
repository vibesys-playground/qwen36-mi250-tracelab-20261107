"""Plan 33 step 1: captured prefill passes (ENGINE_PREFILL_GRAPH=1) against the eager pass.

    ENGINE_PREFILL_GRAPH=1 HIP_VISIBLE_DEVICES=0,1 python tools/prefill_graph_bench.py
        [--news 61,561,700,1650] [--start 3600] [--reps 12]

Loads a TP2 replica as the server does (``Generator.load``: decode/verify graphs, the prefill
graphs of every bucket, warmup), prefills a ``--start``-token context of synthetic ids into one
slot, then per N runs the resumed single-segment pass through the server's call
(``prefill_packed([(slot, ids, start)], [None])``), with the slot's Gated DeltaNet rows restored
between runs:

- timing: graph (the size's replay) and eager (the same call with the graphs hidden), median
  of ``--reps`` synchronized runs each, interleaved;
- graph vs eager: top-1 of the logits, max |logit diff|, the MTP draft, max |diff| of the GDN
  conv and recurrent rows (a padding bug shows up here as an O(1) conv-state error);
- graph vs graph: bitwise (logits, drafts, GDN rows, KV of the new positions) on a repeat;
- padding leak: the same pass with the padding rows' token ids random instead of 0, bitwise
  against the pass with zeros (padding must change no real row and no state).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="61,561,700,1650")
    ap.add_argument("--start", type=int, default=3600)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--dump", default="", help="save each N's graph pass (logits, drafts, state)")
    ap.add_argument("--against", default="", help="compare each N's graph pass bitwise with a --dump")
    args = ap.parse_args()
    news = [int(n) for n in args.news.split(",")]
    start = args.start
    os.environ.setdefault("ENGINE_TP", "2")
    import torch

    from engine.fast import model as fm
    from engine.fast import tp as tpmod
    from engine.generator import CHECKPOINT, Generator
    from serve_probe import token_pool

    t0 = time.perf_counter()
    gen = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
        prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30,
    )
    fast = gen.fast
    assert isinstance(fast, tpmod.TPModel)
    devs = list(fast.devices_tp)
    result("load", seconds=round(time.perf_counter() - t0, 1), prefill_graph=tpmod.PREFILL_GRAPH,
           sizes=sorted(fast.pg), pg_error=fast.pg_error, capacity_words=tpmod.CAPACITY,
           reserved_gib=[round(torch.cuda.memory_reserved(d) / 2**30, 2) for d in devs],
           allocated_gib=[round(torch.cuda.memory_allocated(d) / 2**30, 2) for d in devs],
           graph_error=fast.graph_error, replays_during_load=fast.pg_replays)
    if not fast.pg:
        print("no prefill graphs: nothing to compare", flush=True)
        return 1

    def sync() -> None:
        for d in devs:
            torch.cuda.synchronize(d)

    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], start + 9 * max(news) + 100, "synthetic")
    slot = 0
    fast.reset(slot)
    with torch.inference_mode():
        for lo in range(0, start, 2048):
            fast.prefill(slot, pool[lo: min(start, lo + 2048)], lo)
    sync()
    saved = fast._gdn_copies(slot)
    graphs = dict(fast.pg)

    def restore() -> None:
        fast._gdn_load(slot, saved)

    def state(n: int) -> dict:
        out = {}
        for r, m in enumerate(fast.ranks):
            row = m._g(slot)
            for i, (conv, rec) in m.gdn_state.items():
                out[f"r{r}.{i}.conv"] = conv[row].float().cpu()
                out[f"r{r}.{i}.rec"] = rec[row].float().cpu()
            table = torch.tensor(m.slot_pages[slot], dtype=torch.long)
            pos = torch.arange(start, start + n)
            page, off = table[pos // fm.PAGE], pos % fm.PAGE
            for i, (kp, vp) in m.kv.items():
                out[f"r{r}.{i}.k"] = kp[page.to(kp.device), :, off.to(kp.device)].float().cpu()
                out[f"r{r}.{i}.v"] = vp[page.to(vp.device), :, off.to(vp.device)].float().cpu()
        return out

    def run(ids, graph: bool):
        fast.pg = graphs if graph else {}
        try:
            logits = fast.prefill_packed([(slot, ids, start)], [None])
            drafts = fast.last_drafts.clone() if fast.last_drafts is not None else None
        finally:
            fast.pg = graphs
        return logits[0].float().cpu(), drafts

    def diff(a: dict, b: dict, kinds) -> dict:
        out = {}
        for kind in kinds:
            keys = [k for k in a if k.endswith("." + kind)]
            d = max(float((a[k] - b[k]).abs().max()) for k in keys)
            same = all(torch.equal(a[k], b[k]) for k in keys)
            out[kind] = {"bitwise": same, "max_abs_diff": d}
        return out

    rows = []
    dumped = {}
    ref = torch.load(args.against) if args.against else None
    with torch.inference_mode():
        for n in news:
            ids = pool[start + 5 * n: start + 6 * n]
            size = fast._pg_size(n)
            rec = {"new": n, "size": size}
            # Exactness.
            restore(); lg, dg = run(ids, True); sync(); sg = state(n)
            restore(); lg2, dg2 = run(ids, True); sync(); sg2 = state(n)
            restore(); le, de = run(ids, False); sync(); se = state(n)
            rec["graph_vs_eager"] = {
                "top1_equal": int(lg.argmax()) == int(le.argmax()),
                "logits_max_abs_diff": float((lg - le).abs().max()),
                "drafts_equal": bool(torch.equal(dg.cpu(), de.cpu())) if dg is not None else None,
                **diff(sg, se, ("conv", "rec", "k", "v")),
            }
            if args.dump:
                dumped[n] = {"logits": lg, "drafts": None if dg is None else dg.cpu(), "state": sg}
            if ref is not None and n in ref:
                o = ref[n]
                bad = [k for k in sg if not torch.equal(sg[k], o["state"][k])]
                rec["graph_vs_dump_bitwise"] = {
                    "logits": bool(torch.equal(lg, o["logits"])),
                    "drafts": None if dg is None else bool(torch.equal(dg.cpu(), o["drafts"])),
                    "state": not bad, "state_mismatch": bad[:8]}
            rec["graph_vs_graph_bitwise"] = bool(
                torch.equal(lg, lg2) and (dg is None or torch.equal(dg, dg2))
                and all(torch.equal(sg[k], sg2[k]) for k in sg))
            # Padding leak: the same pass with random padding token ids.
            if size is not None and size > n:
                r0 = fast.ranks[0]
                host = fm.FastModel.pg_host(size, ids, start, None, slot, r0._g(slot))
                rnd = random.Random(n)
                host[n:size] = torch.tensor([rnd.randrange(1000, 100000) for _ in range(size - n)])
                host[size + n: 2 * size] = torch.tensor([rnd.randrange(1000, 100000) for _ in range(size - n)])
                g, bufs, outs, drafts = graphs[size]
                restore()
                for buf in bufs:
                    buf.copy_(host)
                for gr, buf in zip(g, bufs):
                    with torch.cuda.device(buf.device):
                        gr.replay()
                lp = fast._gather(outs).float().cpu()
                dp = drafts[0].clone() if drafts[0] is not None else None
                sync()
                sp = state(n)
                rec["padding_ids_leak_free"] = bool(
                    torch.equal(lp, lg) and (dp is None or torch.equal(dp, dg))
                    and all(torch.equal(sp[k], sg[k]) for k in sg))
            # Timing.
            walls = {"graph": [], "eager": []}
            for _ in range(args.reps):
                for key in walls:
                    restore()
                    sync()
                    a = time.perf_counter()
                    run(ids, key == "graph")
                    sync()
                    walls[key].append((time.perf_counter() - a) * 1e3)
            med = {k: round(statistics.median(v), 2) for k, v in walls.items()}
            rec["wall_ms_median"] = med
            rec["wall_ms_min_max"] = {k: [round(min(v), 2), round(max(v), 2)] for k, v in walls.items()}
            rec["graph_over_eager"] = round(med["graph"] / med["eager"], 3)
            result("case", **rec)
            rows.append(rec)
    if args.dump:
        torch.save(dumped, args.dump)
    result("timeouts", ll=fast.timeouts())
    print(f"\n=== plan 33 step 1: captured vs eager prefill (TP2, start {start}, median of {args.reps})")
    print(f"{'N':>6} {'size':>5} {'eager ms':>9} {'graph ms':>9} {'ratio':>6}  top1  conv/rec max diff  repeat  pad-leak-free")
    for r in rows:
        ge = r["graph_vs_eager"]
        print(f"{r['new']:>6} {str(r['size']):>5} {r['wall_ms_median']['eager']:>9} {r['wall_ms_median']['graph']:>9} "
              f"{r['graph_over_eager']:>6}  {ge['top1_equal']!s:5} {ge['conv']['max_abs_diff']:.3g}/{ge['rec']['max_abs_diff']:.3g}"
              f"  {r['graph_vs_graph_bitwise']!s:6}  {r.get('padding_ids_leak_free')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
