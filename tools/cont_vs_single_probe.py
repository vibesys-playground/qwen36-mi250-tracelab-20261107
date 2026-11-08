"""Why does a continuation prefill differ from a single pass? (profile-cont, plan 41 context)

    HIP_VISIBLE_DEVICES=0,1 python -u tools/cont_vs_single_probe.py [--n 1750] [--splits 875,832,1024]

One TP2 pair, loaded as the server does. Runs the engine's own ``_pg_body`` eagerly (same
kernels as the captured graphs) with recording hooks (no change to values): the residual
stream after every attention/GDN and MoE block (rank 0), the router's top-8 expert ids and
the 8th/9th logit gap per MoE layer. Passes: S single at the served bucket, S2 single at a
larger bucket (a noise floor: only the padded M differs), S3 = S repeated (determinism), and
C = half 1 then half 2 on the same slot for each split. Then per layer: relative error of the
hidden states, of the final GDN conv/recurrent state, of K/V by position range, routing flips
and the error of flipped vs never-flipped rows. Also served graph replay vs eager, and the
non-graph engine path. Every line starting with RESULT is one JSON record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("ENGINE_TP", "2")
os.environ.setdefault("ENGINE_PREFILL_GRAPH_BUCKETS", "768,896,1024,1792,2048")


def result(kind, **f):
    print("RESULT " + json.dumps({"kind": kind, **f}), flush=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def r4(x):
    return [round(float(v), 5) for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1750)
    ap.add_argument("--splits", default="875,832,1024")
    args = ap.parse_args()
    N = args.n
    splits = [int(s) for s in args.splits.split(",")]

    import torch
    from transformers import AutoModelForCausalLM

    from engine import experts, gdn, speculative
    from engine.fast import model as fm
    from engine.fast import prefill_kernels as pk
    from engine.fast import tp as tpmod
    from engine.generator import CHECKPOINT, experts_implementation
    from serve_probe import token_pool

    experts.register()
    t0 = time.perf_counter()
    lm = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT["model"], revision=CHECKPOINT["revision"], dtype=torch.bfloat16, device_map="auto",
        **({"experts_implementation": experts_implementation(torch.version.hip)}
           if experts_implementation(torch.version.hip) else {}))
    lm.eval()
    gdn.install(lm)
    head = speculative.load_head(lm, CHECKPOINT["model"], CHECKPOINT["revision"])
    A = tpmod.TPModel(lm, slots=4, max_len=4097, mtp=head)
    del lm
    torch.cuda.empty_cache()
    log(f"loaded {time.perf_counter() - t0:.0f}s; mtp {head is not None}")
    try:
        A.capture_all(decode=False, verify=False)
    except Exception as exc:  # noqa: BLE001
        log(f"capture failed {exc}")
    result("env", torch=torch.__version__, hip=torch.version.hip, pg=sorted(A.pg), pg_error=A.pg_error,
           buckets=list(tpmod.PG_BUCKETS), page=fm.PAGE, mtp_index=A.ranks[0].mtp_index,
           layer_types=A.ranks[0].layer_types, gdn_impl=str(fm.hf.torch_chunk_gated_delta_rule))
    r0 = A.ranks[0]
    L = len(r0.layers)
    types = ["g" if t == "linear_attention" else "a" for t in r0.layer_types]
    mybuckets = [768, 896, 1024, 1792, 2048]

    def bucket(c):
        return next(b for b in mybuckets if b >= c)

    # ---------------------------------------------------------------- hooks
    class H:
        on = False
        count = 0
        x: list = []
        route: list = []

    orig_rr = r0._reduce_rms

    def rr(h, res, w):
        x, hn = orig_rr(h, res, w)
        if H.on:
            H.x.append(x[: H.count].cpu())
        return x, hn

    r0._reduce_rms = rr
    orig_route = pk.route

    def route(logits, top_k):
        out = orig_route(logits, top_k)
        if H.on and logits.device.index == 0:
            c = H.count
            v = torch.topk(logits[:c].float(), top_k + 1, dim=-1).values
            H.route.append((out[1][:c].sort(dim=-1).values.cpu(), (v[:, top_k - 1] - v[:, top_k]).cpu()))
        return out

    pk.route = route

    # ---------------------------------------------------------------- passes
    def snap(slot, n):
        out = {}
        for r, m in enumerate(A.ranks):
            row = m._g(slot)
            for i, (conv, rec) in m.gdn_state.items():
                out[(i, "conv", r)] = conv[row].float().cpu()
                out[(i, "rec", r)] = rec[row].float().cpu()
            table = torch.tensor(m.slot_pages[slot], dtype=torch.long)
            pos = torch.arange(0, n)
            page, off = table[pos // fm.PAGE], pos % fm.PAGE
            for i, (kp, vp) in m.kv.items():
                out[(i, "k", r)] = kp[page.to(kp.device), :, off.to(kp.device)].float().cpu()
                out[(i, "v", r)] = vp[page.to(vp.device), :, off.to(vp.device)].float().cpu()
        return out

    def eager(slot, ids, start, follow, size):
        r0._ensure_pages(slot, start + len(ids))
        host = fm.FastModel.pg_host(size, ids, start, follow, slot, r0._g(slot))
        bufs = [m._pg_inputs(size) for m in A.ranks]
        for b in bufs:
            b.copy_(host)
        H.count, H.x, H.route, H.on = len(ids), [], [], True
        try:
            outs = A._both(lambda m: m._pg_body(bufs[m.rank], size))
        finally:
            H.on = False
        for d in A.devices_tp:
            torch.cuda.synchronize(d)
        logits = A._gather([o[0] for o in outs]).float().cpu()
        drafts = outs[0][1].clone().cpu() if outs[0][1] is not None else None
        return {"logits": logits, "drafts": drafts, "x": H.x[: 2 * L], "route": H.route[:L]}

    def single(ids, size, slot=0):
        A.reset(slot)
        p = eager(slot, ids, 0, None, size)
        p["snap"] = snap(slot, len(ids))
        return p

    def cont(ids, n1, slot=0):
        A.reset(slot)
        p1 = eager(slot, ids[:n1], 0, ids[n1], bucket(n1))
        s1 = snap(slot, n1)
        p2 = eager(slot, ids[n1:], n1, None, bucket(len(ids) - n1))
        p2["snap"] = snap(slot, len(ids))
        p2["h1"] = p1
        p2["s1"] = s1
        p2["n1"] = n1
        p2["x_full"] = [torch.cat([a, b]) for a, b in zip(p1["x"], p2["x"])]
        p2["route_full"] = [(torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]])) for a, b in
                            zip(p1["route"], p2["route"])]
        return p2

    # ---------------------------------------------------------------- metrics
    def rel(a, b):
        a, b = a.float(), b.float()
        return float((a - b).norm() / (b.norm() + 1e-12))

    def logit_cmp(a, b):
        d = (a - b).abs()
        k = int(d.argmax())
        return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "at_token": k,
                "logit_a": float(a[k]), "logit_b": float(b[k]), "ref_max": float(b.max()),
                "ref_min": float(b.min()), "ref_std": float(b.std()),
                "top1_equal": int(a.argmax()) == int(b.argmax()),
                "top5_diff": float((a[b.topk(5).indices] - b.topk(5).values).abs().max()),
                "bitwise": bool(torch.equal(a, b))}

    def hidden_cmp(tag, xa, xb, n1):
        """per record: rel error rows [0,n1) and [n1,N); max abs diff and ref max."""
        lo_rel, hi_rel, hi_max, hi_ref = [], [], [], []
        for a, b in zip(xa, xb):
            lo_rel.append(rel(a[:n1], b[:n1]))
            hi_rel.append(rel(a[n1:], b[n1:]))
            hi_max.append(float((a[n1:].float() - b[n1:].float()).abs().max()))
            hi_ref.append(float(b[n1:].float().abs().max()))
        last = (xa[-1][n1:].float() - xb[-1][n1:].float()).norm(dim=-1) / xb[-1][n1:].float().norm(dim=-1)
        bins = [float(c.mean()) for c in last.chunk(8)]
        result("hidden", pair=tag, n1=n1, types=types,
               rel_rows_lt_n1_per_record=r4(lo_rel), rel_rows_ge_n1_per_record=r4(hi_rel),
               maxabs_ge_n1=r4(hi_max), maxref_ge_n1=r4(hi_ref),
               final_row_rel_8bins_after_n1=r4(bins),
               final_first16rows_rel=r4(last[:16]))

    def state_cmp(tag, sa, sb, n, n1):
        keys = sorted({(i, k) for (i, k, _) in sa}, key=lambda t: (t[0], t[1]))
        rows = []
        for i, k in keys:
            na = sum(float((sa[(i, k, r)] - sb[(i, k, r)]).float().pow(2).sum()) for r in range(2))
            nb = sum(float(sb[(i, k, r)].float().pow(2).sum()) for r in range(2))
            mx = max(float((sa[(i, k, r)] - sb[(i, k, r)]).abs().max()) for r in range(2))
            ref = max(float(sb[(i, k, r)].abs().max()) for r in range(2))
            row = {"layer": i, "kind": k, "rel": round((na / (nb + 1e-30)) ** 0.5, 5), "maxabs": round(mx, 4),
                   "maxref": round(ref, 4)}
            if k in ("k", "v"):
                d = torch.cat([(sa[(i, k, r)] - sb[(i, k, r)]).float() for r in range(2)], dim=1)  # [n,h,d]
                b = torch.cat([sb[(i, k, r)].float() for r in range(2)], dim=1)
                pm = d.abs().amax(dim=(1, 2))
                p = int(pm.argmax())
                row["argmax_pos"] = p
                row["rel_pos_lt_n1"] = round(rel(d[:n1] + b[:n1], b[:n1]), 5)
                row["rel_pos_ge_n1"] = round(rel(d[n1:] + b[n1:], b[n1:]), 5)
                row["maxabs_pos_lt_n1"] = round(float(pm[:n1].max()), 4)
                row["maxabs_pos_ge_n1"] = round(float(pm[n1:].max()), 4)
                row["ref_at_argmax_maxabs"] = round(float(b[p].abs().max()), 4)
            rows.append(row)
        result("state", pair=tag, n1=n1, rows=rows)

    def route_cmp(tag, ra, rb, xa, xb, n1):
        """routing flips per MoE layer on rows >= n1 and error of ever-flipped vs never-flipped rows."""
        ever = torch.zeros(N - n1, dtype=torch.bool)
        out = []
        for l in range(L):
            ia, ga = ra[l]
            ib, gb = rb[l]
            fl = (ia[n1:] != ib[n1:]).any(dim=-1)
            ever |= fl
            xl_a, xl_b = xa[2 * l + 1][n1:].float(), xb[2 * l + 1][n1:].float()
            err = (xl_a - xl_b).norm(dim=-1) / xl_b.norm(dim=-1)
            out.append({"l": l, "flip_frac": round(float(fl.float().mean()), 4),
                        "flip_gap_median": round(float(gb[n1:][fl].median()), 4) if fl.any() else None,
                        "all_gap_median": round(float(gb[n1:].median()), 4),
                        "ever_flipped_frac": round(float(ever.float().mean()), 4),
                        "err_mean_ever": round(float(err[ever].mean()), 5) if ever.any() else None,
                        "err_mean_never": round(float(err[~ever].mean()), 5) if (~ever).any() else None})
        result("route", pair=tag, n1=n1, layers=out)

    # ---------------------------------------------------------------- experiments
    pool_syn = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 3 * N + 100, "synthetic")
    pool_src = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 8000 + N, "source")
    prompts = {"synthetic": pool_syn[N: 2 * N], "source": pool_src[4000: 4000 + N]}
    sizeS, size2 = bucket(N), 2048
    with torch.inference_mode():
        for name, ids in prompts.items():
            result("prompt", name=name, n=N, size_single=sizeS, size_noise=size2)
            S = single(ids, sizeS)
            S2 = single(ids, size2)
            S3 = single(ids, sizeS)
            result("determinism", prompt=name, S_vs_S_repeat=logit_cmp(S3["logits"], S["logits"]),
                   x_bitwise=all(torch.equal(a, b) for a, b in zip(S["x"], S3["x"])),
                   state_bitwise=all(torch.equal(S["snap"][k], S3["snap"][k]) for k in S["snap"]))
            result("logits", prompt=name, pair="S2_vs_S", **logit_cmp(S2["logits"], S["logits"]),
                   drafts_equal=bool(torch.equal(S2["drafts"], S["drafts"])))
            n1s = splits if name == "synthetic" else splits[:1]
            for n1 in n1s:
                try:
                    C = cont(ids, n1)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    result("error", what=f"cont {n1}", error=str(exc)[:500])
                    continue
                tag = f"{name}.C{n1}"
                result("logits", prompt=name, pair="C_vs_S", n1=n1, **logit_cmp(C["logits"], S["logits"]),
                       drafts_equal=bool(torch.equal(C["drafts"], S["drafts"])), sizes=[bucket(n1), bucket(N - n1)])
                result("logits", prompt=name, pair="C_vs_S2", n1=n1, **logit_cmp(C["logits"], S2["logits"]))
                hidden_cmp(tag + "_vs_S", C["x_full"], S["x"], n1)
                hidden_cmp(f"{name}.S2_vs_S", S2["x"], S["x"], n1)
                hidden_cmp(tag + "_vs_S2", C["x_full"], S2["x"], n1)
                # half 1 alone vs the single pass's prefix: same start, only M differs
                state_cmp(tag + "_vs_S", C["snap"], S["snap"], N, n1)
                state_cmp(f"{name}.S2_vs_S", S2["snap"], S["snap"], N, n1)
                route_cmp(tag + "_vs_S", C["route_full"], S["route"], C["x_full"], S["x"], n1)
                route_cmp(f"{name}.S2_vs_S", S2["route"], S["route"], S2["x"], S["x"], n1)
                del C
            if name == "synthetic":
                # served graph replay vs eager; continuation on the served graphs
                try:
                    A.reset(0)
                    g = A.prefill_packed([(0, ids, 0)], [None])[0].float().cpu()
                    gd = A.last_drafts.clone().cpu()
                    result("served_graph", what="single graph vs eager S", **logit_cmp(g, S["logits"]),
                           drafts_equal=bool(torch.equal(gd, S["drafts"])))
                    n1 = splits[0]
                    A.reset(0)
                    A.prefill_packed([(0, ids[:n1], 0)], [ids[n1]])
                    gc = A.prefill_packed([(0, ids[n1:], n1)], [None])[0].float().cpu()
                    result("served_graph", what="graph continuation vs graph single", n1=n1,
                           **logit_cmp(gc, g))
                    # the non-graph engine path (unfused per-rank prefill)
                    saved = A.pg
                    A.pg = {}
                    try:
                        A.reset(0)
                        e1 = A.prefill_packed([(0, ids, 0)], [None])[0].float().cpu()
                        result("nograph", what="nograph single vs graph single", **logit_cmp(e1, g))
                        A.reset(0)
                        A.prefill_packed([(0, ids[:n1], 0)], [ids[n1]])
                        e2 = A.prefill_packed([(0, ids[n1:], n1)], [None])[0].float().cpu()
                        result("nograph", what="nograph continuation vs nograph single", n1=n1, **logit_cmp(e2, e1))
                        result("nograph", what="nograph continuation vs graph single", n1=n1, **logit_cmp(e2, g))
                    finally:
                        A.pg = saved
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    result("error", what="served", error=str(exc)[:500])
            del S, S2, S3
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
