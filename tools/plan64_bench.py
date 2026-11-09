"""Plan 64 microbenchmarks and bitwise checks (no model load), on two GCDs.

    python tools/plan64_bench.py [--devices 0,1] [--replays 1000] [--iters 40] [--stress 2000]

(a) router top-k selection (``mfma.TOPK_SEL`` 0/1/2) in ``_ksum_router_kernel``: a random
    router|shared weight of the TP2 shape (769 x 2048, some router rows duplicated so experts
    tie), fresh inputs per graph replay; idx/wt/logits of SEL 1 and 2 against SEL 0 bitwise,
    then per-call time of ``mfma.gemv`` (K-sum only) and ``gemv_topk`` per SEL (a graph of 200
    calls) at R = 2, 4, 8, 16.
(b) the all-reduce + RMSNorm tail (``tp.AR_NORM``): per ``AR_NORM_ORDER`` 0-3, on two GPUs,
    rows whose norm differs from the all-reduce followed by ``kernels.add_rms`` (both forms the
    layers use; plain and K-split inputs) over random rows (R 1-64); then the per-call time of
    AR + add_rms against the NORM all-reduce (loopback, graph of 200) and a stress run of the
    best order on two GPUs (bitwise every 50 iterations, timeouts).

Prints ``RESULT`` lines; ``RESULT choice`` names the fastest bitwise SEL and the bitwise ORDER.
A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def graph_time(torch, dev, call, n: int = 200) -> float:
    stream = torch.cuda.Stream(dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.current_stream(dev).wait_stream(stream)
    torch.cuda.synchronize(dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(n):
            call()
    graph.replay()
    torch.cuda.synchronize(dev)
    times = []
    for _ in range(9):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        torch.cuda.synchronize(dev)
        times.append(e0.elapsed_time(e1) * 1e3 / n)
    del graph
    return round(sorted(times)[4], 2)


def router(torch, args) -> int | None:
    from engine.fast import fused, mfma

    dev = torch.device("cuda", 0)
    n, hidden, experts, top_k = 769, 2048, 256, 8
    g = torch.Generator(device="cpu").manual_seed(64)
    w = torch.randn(n, hidden, generator=g) * 0.02
    w[17] = w[3]  # tied experts: identical logits whatever the input
    w[200] = w[3]
    w[101] = w[100]
    w3 = w.to(torch.bfloat16).to(dev)
    sels = [0, 1, 2]
    ok = {s: True for s in sels}
    with torch.cuda.device(dev), torch.inference_mode():
        for rows in (1, 2, 4, 8, 16):
            h = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=dev)
            bufs = {s: (torch.empty((rows, top_k), dtype=torch.int32, device=dev),
                        torch.empty((rows, top_k), dtype=torch.float32, device=dev)) for s in sels}
            res = {}

            def step():
                for s in sels:
                    mfma.TOPK_SEL = s
                    idx, wt = bufs[s]
                    idx.fill_(-1)
                    wt.fill_(float("nan"))
                    res[s] = mfma.gemv_topk(h, w3, idx, wt, experts, top_k)
                mfma.TOPK_SEL = 0
                # and the separate kernel (eager path) on SEL 0's logits
                ri, rw = bufs["k"]
                fused._topk_kernel[(rows,)](res[0], ri, rw, n, E=experts, TOPK=top_k,
                                            TOPKP=fused._pow2(top_k), num_warps=1)

            bufs["k"] = (torch.empty((rows, top_k), dtype=torch.int32, device=dev),
                         torch.empty((rows, top_k), dtype=torch.float32, device=dev))
            h.copy_(torch.randn((rows, hidden), device=dev))
            stream = torch.cuda.Stream(dev)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                step()
                step()
            torch.cuda.current_stream(dev).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                step()
            mism = torch.zeros((3, 3), dtype=torch.int64, device=dev)
            for rep in range(args.replays):
                scale = (0.25, 1.0, 4.0, 16.0)[rep % 4] if rep % 5 == 0 else 1.0
                h.copy_(torch.randn((rows, hidden), device=dev) * scale)
                if rep % 50 == 0 and rows > 1:
                    h[-1].copy_(h[0])
                graph.replay()
                ri, rw = bufs[0]
                for s in sels:
                    mi, mw = bufs[s]
                    mism[s, 0] += (mi != ri).any(dim=1).sum()
                    mism[s, 1] += (mw.view(torch.int32) != rw.view(torch.int32)).any(dim=1).sum()
                    mism[s, 2] += (res[s].view(torch.int16) != res[0].view(torch.int16)).any(dim=1).sum()
            ki, kw = bufs["k"]
            ri, rw = bufs[0]
            sep_ok = torch.equal(ki, ri) and torch.equal(kw.view(torch.int32), rw.view(torch.int32))
            torch.cuda.synchronize(dev)
            m = mism.cpu().tolist()
            for s in sels:
                ok[s] = ok[s] and sum(m[s]) == 0
            out("router_bitwise", rows=rows, replays=args.replays,
                rows_bad={s: {"idx": m[s][0], "wt": m[s][1], "logits": m[s][2]} for s in (1, 2)},
                sel0_equals_topk_kernel_last=sep_ok)
            del graph
        for rows in (2, 4, 8, 16):
            h = torch.randn((rows, hidden), device=dev).to(torch.bfloat16)
            idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
            wt = torch.empty((rows, top_k), dtype=torch.float32, device=dev)
            t = {"gemv_ksum_only": graph_time(torch, dev, lambda: mfma.gemv(h, w3))}
            for s in sels:
                mfma.TOPK_SEL = s
                t[f"sel{s}"] = graph_time(torch, dev, lambda: mfma.gemv_topk(h, w3, idx, wt, experts, top_k))
            mfma.TOPK_SEL = 0
            if rows == 8:
                t8 = t
            out("router_us", rows=rows, per_call_us=t,
                topk_overhead_us={k: round(v - t["gemv_ksum_only"], 2) for k, v in t.items()})
    good = [s for s in (1, 2) if ok[s] and t8[f"sel{s}"] < t8["sel0"]]
    best = min(good, key=lambda s: t8[f"sel{s}"]) if good else 0
    out("router_summary", bitwise={s: ok[s] for s in sels}, best_sel=best)
    return best


def ar_norm(torch, args):
    from engine.fast import kernels, mfma, tp

    devs = [torch.device("cuda", int(i)) for i in range(2)]
    for d in devs:  # both contexts up before the peer-access setup (j306: hipErrorInvalidDevice)
        torch.zeros(1, device=d)
        torch.cuda.synchronize(d)
    torch.cuda.set_device(devs[0])
    group = tp.Group(devs)
    ll = group.ll
    H = tp.RMS_WIDTH
    eps = 1e-6
    g = torch.Generator(device="cpu").manual_seed(7)
    w = [(torch.randn(H, generator=g) * 0.3).to(torch.bfloat16).to(d) for d in devs]

    def hidden(R):
        x = torch.randn(R, H, generator=g)
        x[:, 7] *= 300.0
        x[:, H // 2 + 7] *= -80.0
        return x.to(torch.bfloat16)

    def one(R, it, order, ksplit):
        tp.AR_NORM_ORDER = order
        res = hidden(R)
        res_d = [res.to(d) for d in devs]
        if ksplit > 1:
            parts = [(torch.randn(ksplit * mfma.BM * H, generator=g) * (0.3 + it % 5)).to(d) for d in devs]
            mk = lambda r: mfma.KParts(parts[r], torch.empty((R, H), dtype=torch.bfloat16, device=devs[r]), ksplit)  # noqa: E731
        else:
            parts = [(torch.randn(R, H, generator=g) * (0.5 + it % 7)).to(torch.bfloat16).to(d) for d in devs]
            mk = lambda r: parts[r]  # noqa: E731
        s = [ll._launch(r, mk(r), None) for r in range(2)]
        ref_a = [kernels.add_rms(res_d[r], s[r], w[r], eps) for r in range(2)]
        t = [ll._launch(r, mk(r), res_d[r]) for r in range(2)]
        ref_b = [kernels.add_rms(t[r], None, w[r], eps)[1] for r in range(2)]
        got = [ll._launch(r, mk(r), res_d[r], norm=(w[r], eps)) for r in range(2)]
        for d in devs:
            torch.cuda.synchronize(d)
        bad_sum = bad_a = bad_b = 0
        for r in range(2):
            bad_sum += int((got[r][0] != ref_a[r][0]).any(dim=1).sum()) + int((got[r][0] != t[r]).any(dim=1).sum())
            bad_a += int((got[r][1] != ref_a[r][1]).any(dim=1).sum())
            bad_b += int((got[r][1] != ref_b[r]).any(dim=1).sum())
        ranks = torch.equal(got[0][1].cpu(), got[1][1].cpu())
        return bad_sum, bad_a, bad_b, ranks

    good = []
    for order in (1, 0, 2, 3):
        tot = {"rows": 0, "sum": 0, "norm_a": 0, "norm_b": 0, "ranks_differ": 0}
        for R in (1, 2, 8, 16, 64, 112):
            for it in range(args.iters):
                ksplit = 4 if (it % 2 and R <= mfma.BM) else 1
                bs, ba, bb, rk = one(R, it, order, ksplit)
                tot["rows"] += R
                tot["sum"] += bs
                tot["norm_a"] += ba
                tot["norm_b"] += bb
                tot["ranks_differ"] += 0 if rk else 1
        ok = tot["sum"] == 0 and tot["norm_a"] == 0 and tot["norm_b"] == 0 and tot["ranks_differ"] == 0
        out("ar_norm_bitwise", order=order, ok=ok, bad=tot, timeouts=ll.timeouts())
        if ok:
            good.append(order)
            if order == 1:
                break
    order = good[0] if good else 1
    tp.AR_NORM_ORDER = order
    dev = devs[0]
    with torch.cuda.device(dev):
        for R in (2, 8, 16):
            x = (torch.randn(R, H, device=dev)).to(torch.bfloat16)
            res = (torch.randn(R, H, device=dev)).to(torch.bfloat16)
            t = {
                "ar_res": graph_time(torch, dev, lambda: ll._launch(0, x, res, loop=True)),
                "add_rms": graph_time(torch, dev, lambda: kernels.add_rms(res, None, w[0], eps)),
                "ar_res_then_add_rms": graph_time(
                    torch, dev, lambda: kernels.add_rms(ll._launch(0, x, res, loop=True), None, w[0], eps)),
                "ar_norm": graph_time(torch, dev, lambda: ll._launch(0, x, res, loop=True, norm=(w[0], eps))),
            }
            out("ar_norm_us_loopback", rows=R, order=order, per_call_us=t,
                saved_us=round(t["ar_res_then_add_rms"] - t["ar_norm"], 2))
    # stress on two GPUs
    bad = 0
    for it in range(args.stress):
        if it % 50 == 0:
            bs, ba, bb, rk = one(8, it, order, 4 if it % 100 else 1)
            bad += bs + ba + bb + (0 if rk else 1)
        else:
            tp.AR_NORM_ORDER = order
            x = [torch.randn(8, H, device=d).to(torch.bfloat16) for d in devs]
            [ll._launch(r, x[r], x[r], norm=(w[r], eps)) for r in range(2)]
    for d in devs:
        torch.cuda.synchronize(d)
    out("ar_norm_stress", iters=args.stress, order=order, bad_rows=bad, timeouts=ll.timeouts())
    return good


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default="0,1")
    p.add_argument("--replays", type=int, default=1000)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--stress", type=int, default=2000)
    p.add_argument("--skip-router", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ["ENGINE_AR_NORM"] = "1"
    import torch

    sel = order = None
    try:
        sel = None if args.skip_router else router(torch, args)
    except Exception as exc:  # noqa: BLE001
        out("router_error", error=repr(exc)[:2000])
    try:
        order = ar_norm(torch, args)
    except Exception as exc:  # noqa: BLE001
        out("ar_norm_error", error=repr(exc)[:2000])
    out("choice", sel=sel if sel is not None else 0,
        order=order[0] if order else -1)
    print(f"CHOICE SEL={sel if sel is not None else 0} ORDER={order[0] if order else -1}", flush=True)


if __name__ == "__main__":
    main()
