"""Plan 47 stage 0b: the confidence-gated hybrid draft (called by ``tools/ivf_probe.py --gate``).

The IVF draft (spherical k-means, 4,096 clusters per TP2 shard, top P = 32 or 64) is kept
unless a gate signal computed from the weights and the query alone says it may be wrong, in
which case the draft falls back to today's full-vocabulary pass. Signals, per query:

- ``top``: the shortlist's top logit (the IVF draft's own logit, max over both shards);
- ``gap``: that logit minus the best *unscanned* centroid score on either shard (the
  (P+1)-th centroid; a centroid score is its cluster's mean logit);
- ``cmargin``: best minus next-best centroid score on the winning shard;
- ``top|gap``: fall back when either is below its threshold.

Thresholds are chosen on the calibration half of the natural prompts (``natural_cal``: even
prompt index) as the smallest fallback rate whose agreement with the full draft is >= the
target at every depth 1-3, then applied unchanged to ``natural_test`` (odd index), ``doc``
(pydoc prose, teacher-forced) and ``replay`` (replay-like). The index itself is built from the
lm_head weights only.

Times (graph-captured, this GPU): today's full draft (``mfma.gemv`` + ``argmax_pack`` +
``argmax_pick``), the IVF draft with ``torch.topk`` selection and with a fused one-kernel
top-P (``tl.sort`` of packed (score, index) keys), and the expected per-draft time
``search + f x full`` per text kind (``f`` the fallback rate).
"""

from __future__ import annotations

import math
import statistics

import torch

from engine.fast import draft_ivf, kernels, mfma, tp

triton, tl = kernels.triton, kernels.tl

if triton is not None:

    @triton.jit
    def _topp_kernel(s_ptr, out_ptr, C: tl.constexpr, P: tl.constexpr, LOG2C: tl.constexpr):
        """out[r, :P] = indices of the P largest of s[r, :C] (ties to the lower index)."""
        r = tl.program_id(0)
        i = tl.arange(0, C)
        s = tl.load(s_ptr + r * C + i).to(tl.float32)
        b = s.to(tl.int32, bitcast=True)
        key = tl.where(b >= 0, b.to(tl.int64) + 2147483648, (~b).to(tl.int64))
        packed = (key << LOG2C) | (C - 1 - i).to(tl.int64)
        srt = tl.sort(packed, 0, descending=True)
        idx = (C - 1) - (srt & (C - 1))
        tl.store(out_ptr + r * P + i, idx, mask=i < P)


    @triton.jit
    def _radix_topp_kernel(s_ptr, out_ptr, nxt_ptr, C: tl.constexpr, P: tl.constexpr):
        """out[r, :P] = indices of the P largest bf16 scores of s[r, :C] (ties to the lower
        index; in index order), nxt[r] = the best unselected score (the gate's (P+1)-th).
        A 16-step radix select on the scores' 16-bit order keys (bf16 has 16 bits), then a
        compaction by prefix sums; one program per row."""
        r = tl.program_id(0)
        i = tl.arange(0, C)
        b = tl.load(s_ptr + r * C + i)
        bi = b.to(tl.int16, bitcast=True).to(tl.int32)
        key = tl.where(bi >= 0, bi + 32768, ~bi)
        t = tl.zeros([], dtype=tl.int32)
        for j in tl.static_range(16):
            cand = t | (1 << (15 - j))
            cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
            t = tl.where(cnt >= P, cand, t)
        gt = key > t
        ngt = tl.sum(gt.to(tl.int32), axis=0)
        eq = key == t
        ceq = tl.cumsum(eq.to(tl.int32), axis=0)
        sel = gt | (eq & (ceq <= P - ngt))
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(out_ptr + r * P + pos, i.to(tl.int64), mask=sel)
        rest = tl.where(sel, float("-inf"), b.to(tl.float32))
        tl.store(nxt_ptr + r, tl.max(rest, axis=0))

    @triton.jit
    def _flag_kernel(flag_ptr, out_ptr):
        """A predicated launch that is not taken: read a flag, return (served-path extra)."""
        f = tl.load(flag_ptr)
        if f != 0:
            tl.store(out_ptr, f)


class RadixSelect:
    def __init__(self, d: draft_ivf.Draft, max_rows: int, num_warps: int = 4) -> None:
        self.d, self.warps = d, num_warps
        self.c = d.cent.shape[0]
        self.buf = torch.empty((max_rows, d.p), dtype=torch.int64, device=d.ws.device)
        self.nxt = torch.empty((max_rows,), dtype=torch.float32, device=d.ws.device)

    def select_scores(self, s: torch.Tensor) -> torch.Tensor:
        rows = s.shape[0]
        sel = self.buf[:rows]
        _radix_topp_kernel[(rows,)](s, sel, self.nxt, C=self.c, P=self.d.p, num_warps=self.warps)
        return sel

    def select(self, y: torch.Tensor) -> torch.Tensor:
        return self.select_scores(mfma.gemv(y, self.d.cent))

    def pack(self, y: torch.Tensor) -> torch.Tensor:
        sel = self.select(y)
        self.d.gather(y, sel)
        return self.d.pick(y.shape[0])


class FusedSelect:
    def __init__(self, d: draft_ivf.Draft, max_rows: int) -> None:
        self.d = d
        c = d.cent.shape[0]
        self.c, self.log2c = c, int(math.log2(c))
        self.buf = torch.empty((max_rows, d.p), dtype=torch.int64, device=d.ws.device)

    def select(self, y: torch.Tensor) -> torch.Tensor:
        s = mfma.gemv(y, self.d.cent)
        sel = self.buf[: y.shape[0]]
        _topp_kernel[(y.shape[0],)](s, sel, C=self.c, P=self.d.p, LOG2C=self.log2c, num_warps=8)
        return sel

    def pack(self, y: torch.Tensor) -> torch.Tensor:
        sel = self.select(y)
        self.d.gather(y, sel)
        return self.d.pick(y.shape[0])


def features(Y: torch.Tensor, W: torch.Tensor, bounds, pair, ps, chunk: int = 256) -> dict:
    """Per query: correct[P] (IVF draft == full draft) and the gate signals per P."""
    feats = {P: {"correct": [], "top": [], "gap": [], "cmargin": []} for P in ps}
    for a in range(0, Y.shape[0], chunk):
        y = Y[a:a + chunk]
        n = y.shape[0]
        lg = y @ W.T  # bf16 logits, as the engine's GEMV rounds them
        vstar = lg.argmax(-1)
        shard = (vstar >= bounds[1]).long()
        local = vstar - torch.tensor(bounds[:2], device=Y.device)[shard]
        per = []
        for r in range(2):
            idx = pair[r]
            lgr = lg[:, bounds[r]:bounds[r + 1]].float()
            cmax = torch.full((n, idx.clusters), float("-inf"), device=Y.device)
            cmax.scatter_reduce_(1, idx.cluster[None].expand(n, -1), lgr, "amax", include_self=True)
            s = draft_ivf.cluster_scores(y, idx)
            order = torch.sort(s, dim=1, descending=True, stable=True).indices
            per.append((idx, cmax, s, order))
            del lgr
        rank = torch.empty_like(vstar)
        for r in range(2):
            sel = shard == r
            if sel.any():
                rank[sel] = draft_ivf.search_rank(y[sel], pair[r], local[sel])
        for P in ps:
            m = torch.stack([cmax.gather(1, order[:, :P]).amax(1) for _, cmax, _, order in per], 1)
            nxt = torch.stack([s.gather(1, order[:, P:P + 1])[:, 0] for _, _, s, order in per], 1)
            c1 = torch.stack([s.gather(1, order[:, 0:1])[:, 0] for _, _, s, order in per], 1)
            c2 = torch.stack([s.gather(1, order[:, 1:2])[:, 0] for _, _, s, order in per], 1)
            top, win = m.max(1)
            f = feats[P]
            f["correct"].append(rank < P)
            f["top"].append(top)
            f["gap"].append(top - nxt.max(1).values)
            f["cmargin"].append((c1 - c2).gather(1, win[:, None])[:, 0])
    return {P: {k: torch.cat(v) for k, v in f.items()} for P, f in feats.items()}


def run(samples, W, args, out, time_us) -> None:
    dev = W.device
    V = W.shape[0]
    bounds = [math.ceil(V * r / 2) for r in range(3)]
    shards = [W[bounds[r]: bounds[r + 1]] for r in range(2)]
    C = 4096
    pair = []
    for r in range(2):
        idx = draft_ivf.build(shards[r], C, 64, iters=args.iters, spherical=True)
        out("index", variant="spherical", C=C, rank=r, build_s=round(idx.build_s, 2), **idx.stats)
        pair.append(idx)
    ps = (32, 64)

    # ---------------------------------------------------------------- features
    data = {}  # (kind, depth) -> (feats, accepted)
    for (kind, depth), lst in sorted(samples.items()):
        if kind == "replay_prompt":
            continue
        Y = torch.cat([y for y, _ in lst])
        A = torch.cat([a for _, a in lst])
        data[(kind, depth)] = (features(Y, W, bounds, pair, ps), A)
        f = data[(kind, depth)][0]
        out("ungated", kind=kind, depth=depth, n=int(Y.shape[0]), accepted=int(A.sum()),
            **{f"P{P}": [round(float(f[P]["correct"].float().mean()), 5),
                         round(float(f[P]["correct"][A].float().mean()), 5) if A.any() else None]
               for P in ps})
    kinds = sorted({k for k, _ in data})

    # ---------------------------------------------------------------- times
    rows_list = (1, 2)
    ysrc = torch.cat([y for y, _ in samples[("natural_cal", 1)]])
    w0 = shards[0].contiguous()
    t_full, t_search, t_fused, t_radix = {}, {}, {}, {}
    flag = torch.zeros(1, dtype=torch.int32, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    t_noop = time_us([lambda: _flag_kernel[(1,)](flag, sink)])
    out("time_noop", predicated_launch_us=round(t_noop, 2))
    for R in rows_list:
        y = ysrc[:R].contiguous()
        t_full[R] = time_us([lambda: tp.argmax_pick(tp.argmax_pack(mfma.gemv(y, w0), 0, 0))])
    for P in ps:
        d = draft_ivf.Draft(w0, pair[0], P, max_rows=8, centroid_gemv=lambda a, c: mfma.gemv(a, c))
        fz = FusedSelect(d, 8)
        try:  # fused top-P == torch.topk as sets, on real queries
            q = ysrc[:8].contiguous()
            sc = mfma.gemv(q, d.cent).float()  # same scores; ties may pick other ids
            a_ = torch.sort(sc.gather(1, fz.select(q)), 1).values
            b_ = torch.sort(sc.gather(1, d.select(q)), 1).values
            fused_ok = bool(torch.equal(a_, b_))
        except Exception as exc:  # noqa: BLE001
            fused_ok = f"error {exc!r}"[:300]
        rx_best = None
        for warps in (4, 8, 16):
            rx = RadixSelect(d, 8, warps)
            try:  # exact: the first P of a stable descending sort (ties to the lower index)
                q = ysrc[:8].contiguous()
                sc = mfma.gemv(q, d.cent)
                ref = torch.sort(sc.float(), dim=1, descending=True, stable=True).indices
                got = torch.sort(rx.select_scores(sc), 1).values
                exact = bool(torch.equal(got, torch.sort(ref[:, :P], 1).values))
                nxt_ok = bool(torch.equal(rx.nxt[:8], sc.float().gather(1, ref[:, P:P + 1])[:, 0]))
                tk = torch.sort(sc.float().gather(1, torch.topk(sc.float(), P, dim=1).indices), 1).values
                topk_ok = bool(torch.equal(torch.sort(sc.float().gather(1, got), 1).values, tk))
                y2 = ysrc[:2].contiguous()
                t_sel = time_us([lambda: rx.select(y2)])
                out("radix_check", P=P, warps=warps, exact_vs_stable_sort=exact, next_score_ok=nxt_ok,
                    scores_match_topk=topk_ok, select_us_2rows=round(t_sel, 1))
                if exact and topk_ok and (rx_best is None or t_sel < rx_best[1]):
                    rx_best = (rx, t_sel)
            except Exception as exc:  # noqa: BLE001
                out("error", what=f"radix P={P} warps={warps}", error=repr(exc)[:400])
        for R in rows_list:
            y = ysrc[:R].contiguous()
            if rx_best is not None:
                rx = rx_best[0]
                t_radix[(P, R)] = time_us([lambda: tp.argmax_pick(rx.pack(y))])
                out("time_radix", P=P, rows=R, warps=rx.warps, search_us=round(t_radix[(P, R)], 1),
                    select_us=round(time_us([lambda: rx.select(y)]), 1),
                    radix_kernel_us=round(time_us([lambda: rx.select_scores(mfma.gemv(y, d.cent))]) -
                                          time_us([lambda: mfma.gemv(y, d.cent)]), 1))
            t_search[(P, R)] = time_us([lambda: tp.argmax_pick(d.pack(y))])
            if fused_ok is True:
                t_fused[(P, R)] = time_us([lambda: tp.argmax_pick(fz.pack(y))])
                t_sel = time_us([lambda: fz.select(y)])
            else:
                t_sel = None
            out("time", P=P, rows=R, full_us=round(t_full[R], 1), search_us=round(t_search[(P, R)], 1),
                fused_search_us=round(t_fused[(P, R)], 1) if (P, R) in t_fused else None,
                fused_select_us=None if t_sel is None else round(t_sel, 1), fused_matches_topk=fused_ok)
        del d, fz, rx_best
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- calibrate
    def agree(kind, P, fb):
        """Per depth: agreement (all, accepted) and fallback rate under mask fn ``fb``."""
        res = []
        for depth in (1, 2, 3):
            if (kind, depth) not in data:
                continue
            f, A = data[(kind, depth)]
            fall = fb(f[P])
            ok = f[P]["correct"] | fall
            res.append(dict(depth=depth, n=int(ok.numel()), agree=float(ok.float().mean()),
                            agree_acc=float(ok[A].float().mean()) if A.any() else None,
                            fallback=float(fall.float().mean())))
        return res

    def cal_pick(P, sig, target):
        """Smallest calibration fallback whose min-depth agreement >= target."""
        cal = [data[("natural_cal", d)][0][P] for d in (1, 2, 3)]
        pool = torch.cat([f[sig] for f in cal]) if sig != "top|gap" else None
        best = None
        if sig != "top|gap":
            for q in torch.linspace(0, 0.9, 901).tolist():
                tau = float(torch.quantile(pool.float(), q)) if q > 0 else float("-inf")
                fb = (lambda t: (lambda f: f[sig] < t))(tau)
                r = agree("natural_cal", P, fb)
                if min(x["agree"] for x in r) >= target:
                    fr = statistics.mean(x["fallback"] for x in r)
                    return dict(sig=sig, tau=tau, fb=fb, cal_fallback=fr)
            return None
        tops = torch.cat([f["top"] for f in cal]).float()
        gaps = torch.cat([f["gap"] for f in cal]).float()
        qs = torch.linspace(0, 0.8, 41).tolist()
        ta = [float(torch.quantile(tops, q)) if q > 0 else float("-inf") for q in qs]
        tg = [float(torch.quantile(gaps, q)) if q > 0 else float("-inf") for q in qs]
        for a_ in ta:
            for g_ in tg:
                fb = (lambda a1, g1: (lambda f: (f["top"] < a1) | (f["gap"] < g1)))(a_, g_)
                r = agree("natural_cal", P, fb)
                if min(x["agree"] for x in r) >= target:
                    fr = statistics.mean(x["fallback"] for x in r)
                    if best is None or fr < best["cal_fallback"]:
                        best = dict(sig=sig, tau=(a_, g_), fb=fb, cal_fallback=fr)
                    break  # larger g_ only adds fallback
        return best

    chosen = {}
    for P in ps:
        for target in (0.993, 0.995, 0.997):
            rules = []
            for sig in ("top", "gap", "cmargin", "top|gap"):
                rule = cal_pick(P, sig, target)
                if rule is None:
                    out("rule", P=P, target=target, sig=sig, found=False)
                    continue
                rules.append(rule)
                for kind in kinds:
                    r = agree(kind, P, rule["fb"])
                    fr = statistics.mean(x["fallback"] for x in r)
                    exp = {R: t_search[(P, R)] + fr * t_full[R] for R in rows_list}
                    expf = {R: t_fused[(P, R)] + fr * t_full[R] for R in rows_list if (P, R) in t_fused}
                    expr = {R: t_radix[(P, R)] + fr * t_full[R] for R in rows_list if (P, R) in t_radix}
                    srv = {(R, ar): t_radix[(P, R)] + 3 * t_noop + fr * (t_full[R] + ar)
                           for R in rows_list for ar in (10, 20) if (P, R) in t_radix}
                    out("gated", P=P, target=target, sig=sig, tau=rule["tau"], kind=kind,
                        cal_fallback=round(rule["cal_fallback"], 4),
                        agree=[round(x["agree"], 5) for x in r],
                        agree_accepted=[None if x["agree_acc"] is None else round(x["agree_acc"], 5) for x in r],
                        fallback=[round(x["fallback"], 4) for x in r], n=[x["n"] for x in r],
                        expected_us={R: round(v, 1) for R, v in exp.items()},
                        saving_us={R: round(t_full[R] - v, 1) for R, v in exp.items()},
                        fused_saving_us={R: round(t_full[R] - v, 1) for R, v in expf.items()},
                        radix_saving_us={R: round(t_full[R] - v, 1) for R, v in expr.items()},
                        served_saving_us={f"{R}rows_ar{ar}": round(t_full[R] - v, 1) for (R, ar), v in srv.items()})
            if rules:  # the calibration's own choice: least calibration fallback
                chosen[(P, target)] = min(rules, key=lambda x: x["cal_fallback"])

    # ---------------------------------------------------------------- gate
    verdicts = []
    for (P, target), rule in sorted(chosen.items()):
        ok_agree, savings = True, {}
        for kind in ("natural_test", "doc"):
            r = agree(kind, P, rule["fb"])
            if not r:
                continue
            ok_agree &= min(x["agree"] for x in r) >= 0.993
            fr = statistics.mean(x["fallback"] for x in r)
            rad = t_radix.get((P, 2))
            savings[kind] = (round(t_full[2] - (t_search[(P, 2)] + fr * t_full[2]), 1),
                             None if rad is None else round(t_full[2] - (rad + fr * t_full[2]), 1),
                             None if rad is None else round(t_full[2] - (rad + 3 * t_noop + fr * (t_full[2] + 20)), 1))
        ok_save = all(s[1] is not None and s[1] >= 150 for s in savings.values())
        ok_save_f = all(s[2] is not None and s[2] >= 150 for s in savings.values())
        v = dict(P=P, target=target, sig=rule["sig"], tau=rule["tau"], agree_ok=ok_agree,
                 saving_ok_radix=ok_save, saving_ok_radix_served_ar20=ok_save_f,
                 saving_us_2rows_topk_radix_served=savings)
        verdicts.append(v)
        out("gate_choice", **v)
    out("verdict_gate", go_radix=any(v["agree_ok"] and v["saving_ok_radix"] for v in verdicts),
        go_served_ar20=any(v["agree_ok"] and v["saving_ok_radix_served_ar20"] for v in verdicts),
        rule="calibrated on natural_cal; pass iff natural_test and doc agreement >= 99.3% at every depth "
             "and expected saving >= 150 us per draft at 2 rows on both (radix selection; served adds 3 "
             "predicated launches per draft and a 20 us rank all-reduce per fallback)")
