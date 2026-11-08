"""Plan 37 step 0 (a): the router top-k fused into the router|shared K-sum kernel is bitwise
``fused._topk_kernel``, over many CUDA-graph replays with fresh inputs.

    python tools/topk_epi_check.py [--devices 0,1] [--replays 1000] [--rows 1,2,4,8,16,32]

Loads one TP2 replica as the server does (``Generator.load``) and takes rank 0's packed
router|shared weights (``FusedDecode.w_moe``, real checkpoint shards) of three MoE layers (the
first, the last, the MTP head's). Per layer and row count R it captures one graph that runs,
on the same input h [R, hidden]:

- today's path: ``mfma.gemv(h, w3)`` then ``fused._topk_kernel`` -> r, idx, wt;
- ``mfma.gemv_topk`` mode 1 (last program by atomic counter) and mode 2 (one extra program
  per row), each into idx/wt buffers first filled with -1 / NaN inside the graph, so a missing
  or stale write shows as a mismatch.

Each of ``--replays`` replays draws a fresh h (``torch.randn`` scaled like a post-RMSNorm
hidden state, written into the graph's static input outside it) and compares r, idx and wt
of both modes bitwise with today's, accumulating mismatch counts on the GPU. Also counts how
many distinct expert sets appeared (the inputs do move the routing) and checks the counter is
back to 0. Ends with an eager (no graph) pass at the same sizes, and the kernel timings of
the three paths (graph of 200 back-to-back calls, R = 4 and 8).

Prints ``RESULT`` lines; ``RESULT bitwise_summary`` has the verdict. A plan hint, not the
task's metric.
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


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--rows", default="1,2,4,8,16,32")
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")

    import torch

    from engine.fast import fused, mfma
    from engine.generator import CHECKPOINT, Generator

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    r0 = fast.ranks[0] if hasattr(fast, "ranks") else fast
    fd = r0.fused
    out("load", seconds=round(time.perf_counter() - started), tp=hasattr(fast, "ranks"),
        mfma_gemv=mfma.GEMV, topk_epi_env=mfma.TOPK_EPI, top_k=r0.top_k)

    layers = [0, len(r0.layers) - 1]
    if getattr(r0, "mtp", None) is not None:
        layers.append(r0.mtp_index)
    rows_list = [int(v) for v in args.rows.split(",")]
    all_ok = True
    with torch.inference_mode():
        for li in layers:
            w3 = fd.w_moe[li]
            dev = w3.device
            experts = fd._layer(li).mlp.experts.gate_up_proj.shape[0]
            top_k = r0.top_k
            n, hidden = w3.shape
            tiles = mfma.tiles_for(n, hidden)
            with torch.cuda.device(dev):
                cnt = torch.zeros(1, dtype=torch.int32, device=dev)
                stream = torch.cuda.Stream(dev)
                for rows in rows_list:
                    h = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=dev)
                    bufs = {name: (torch.empty((rows, top_k), dtype=torch.int32, device=dev),
                                   torch.empty((rows, top_k), dtype=torch.float32, device=dev))
                            for name in ("ref", "m1", "m2")}
                    res = {}

                    def step():
                        r = mfma.gemv(h, w3)
                        idx, wt = bufs["ref"]
                        idx.fill_(-1)
                        wt.fill_(float("nan"))
                        fused._topk_kernel[(rows,)](r, idx, wt, n, E=experts, TOPK=top_k,
                                                    TOPKP=fused._pow2(top_k), num_warps=1)
                        res["ref"] = r
                        for mode in (1, 2):
                            idx, wt = bufs[f"m{mode}"]
                            idx.fill_(-1)
                            wt.fill_(float("nan"))
                            res[f"m{mode}"] = mfma.gemv_topk(h, w3, idx, wt, experts, top_k, cnt,
                                                             mode=mode)

                    torch.manual_seed(1000 * li + rows)
                    h.copy_(torch.randn_like(h, dtype=torch.float32) * 1.0)
                    stream.wait_stream(torch.cuda.current_stream(dev))
                    with torch.cuda.stream(stream):
                        step()
                        step()
                    torch.cuda.current_stream(dev).wait_stream(stream)
                    torch.cuda.synchronize(dev)
                    if res["m1"] is None or res["m2"] is None:
                        out("bitwise", layer=li, rows=rows, error="gemv_topk returned None",
                            tiles=tiles)
                        all_ok = False
                        continue
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        step()
                    mism = torch.zeros(6, dtype=torch.int64, device=dev)
                    sets = set()
                    eager_bad = 0
                    for rep in range(args.replays):
                        # Fresh inputs; every 7th replay uses a different scale (sharper or
                        # flatter softmax), and every 50th an exactly repeated row (ties in
                        # row content, not in logits).
                        scale = (0.25, 1.0, 4.0)[rep % 3] if rep % 7 == 0 else 1.0
                        h.copy_(torch.randn((rows, hidden), device=dev) * scale)
                        if rep % 50 == 0 and rows > 1:
                            h[-1].copy_(h[0])
                        graph.replay()
                        ref_r = res["ref"]
                        ref_i, ref_w = bufs["ref"]
                        for j, mode in enumerate((1, 2)):
                            mi, mw = bufs[f"m{mode}"]
                            mism[3 * j] += (mi != ref_i).any(dim=1).sum()
                            mism[3 * j + 1] += (mw.view(torch.int32) != ref_w.view(torch.int32)).any(dim=1).sum()
                            mism[3 * j + 2] += (res[f"m{mode}"].view(torch.int16)
                                                != ref_r.view(torch.int16)).any(dim=1).sum()
                        if rep % 100 == 0:
                            sets.update(tuple(v) for v in ref_i.sort(dim=1).values.cpu().tolist())
                    # The eager path (no graph) at the same size, a few fresh inputs.
                    for rep in range(20):
                        h.copy_(torch.randn((rows, hidden), device=dev))
                        step()
                        for mode in (1, 2):
                            mi, mw = bufs[f"m{mode}"]
                            ri, rw = bufs["ref"]
                            if not (torch.equal(mi, ri) and torch.equal(mw.view(torch.int32), rw.view(torch.int32))
                                    and torch.equal(res[f"m{mode}"], res["ref"])):
                                eager_bad += 1
                    torch.cuda.synchronize(dev)
                    m = mism.cpu().tolist()
                    ok = sum(m) == 0 and eager_bad == 0 and int(cnt.item()) == 0
                    all_ok = all_ok and ok
                    out("bitwise", layer=li, rows=rows, replays=args.replays, ok=ok,
                        m1_rows_bad={"idx": m[0], "wt": m[1], "logits": m[2]},
                        m2_rows_bad={"idx": m[3], "wt": m[4], "logits": m[5]},
                        eager_bad=eager_bad, counter_after=int(cnt.item()),
                        distinct_expert_sets_sampled=len(sets), tiles=tiles)
                    del graph

            # Kernel timing of the three paths (graph of 200 calls back to back), rank 0 GCD.
            if li != layers[0]:
                continue
            with torch.cuda.device(dev):
                for rows in (2, 4, 8):
                    h = torch.randn((rows, hidden), device=dev).to(torch.bfloat16)
                    idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
                    wt = torch.empty((rows, top_k), dtype=torch.float32, device=dev)
                    timings = {}
                    for name in ("ref", "m1", "m2"):
                        def call():
                            if name == "ref":
                                r = mfma.gemv(h, w3)
                                fused._topk_kernel[(rows,)](r, idx, wt, n, E=experts, TOPK=top_k,
                                                            TOPKP=fused._pow2(top_k), num_warps=1)
                            else:
                                mfma.gemv_topk(h, w3, idx, wt, experts, top_k, cnt,
                                               mode=int(name[1]))
                        stream.wait_stream(torch.cuda.current_stream(dev))
                        with torch.cuda.stream(stream):
                            call()
                            call()
                        torch.cuda.current_stream(dev).wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            for _ in range(200):
                                call()
                        graph.replay()
                        torch.cuda.synchronize(dev)
                        times = []
                        for _ in range(7):
                            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0.record()
                            graph.replay()
                            e1.record()
                            torch.cuda.synchronize(dev)
                            times.append(e0.elapsed_time(e1) * 1e3 / 200)
                        timings[name] = round(sorted(times)[3], 2)
                        del graph
                    out("router_topk_us", rows=rows, per_call_us=timings,
                        saved_m1_us=round(timings["ref"] - timings["m1"], 2),
                        saved_m2_us=round(timings["ref"] - timings["m2"], 2),
                        counter_after=int(cnt.item()))
    out("bitwise_summary", all_ok=all_ok, verdict="PASS" if all_ok else "FAIL")
    if hasattr(fast, "timeouts"):
        out("tp_allreduce_timeouts", n=fast.timeouts())


if __name__ == "__main__":
    main()
