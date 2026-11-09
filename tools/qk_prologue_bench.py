"""Plan 56 micro-bench on one GCD: the attention Q/K prologue and the MoE tile table.

    HIP_VISIBLE_DEVICES=0 python tools/qk_prologue_bench.py [--news 61,561,1650] [--iters 200]

Shapes of one TP2 rank of Qwen3.6-35B-A3B: 8 query heads (q|gate interleaved per head), one KV
head, head dim 256, 64 rotary dims, the merged q|k|v projection output as column views; 256
experts, top 8. Each variant is captured in a CUDA graph of ``--iters`` calls and replayed, so
the time is GPU time per call without host gaps (as in the captured prefill pass).

- prologue: the eager ops of ``FastModel._qkv`` (cos/sin per layer, the two RMSNorms, RoPE, the
  gate copy) and the two K/V page writes, against ``prefill_kernels.qk_prologue`` (+ the
  per-pass cos/sin, reported apart). Also the max BF16 ulp between them and the bitwise share.
- tiles: ``_tiles_kernel`` (grid of 32-tile programs) against ``_tiles_one_kernel`` (one
  program) at pairs = 8 x N with random routing, and whether the buffers are bitwise equal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


def graph_us(fn, iters: int) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / iters)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="61,561,1650")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    import torch

    from engine.fast import moe_prefill as mp
    from engine.fast import prefill_kernels as pk
    from engine.fast.model import _rms, _rope

    dev = torch.device("cuda")
    heads, kv_heads, dim, rot, page_size = 8, 1, 256, 64, 64
    n_q, n_kv = heads * 2 * dim, kv_heads * dim
    inv_freq = (1.0 / (10_000_000 ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))).to(dev)
    q_w = (torch.randn(dim) * 0.2).bfloat16().to(dev)
    k_w = (torch.randn(dim) * 0.2).bfloat16().to(dev)
    pages = 400
    for n in (int(v) for v in args.news.split(",")):
        g = torch.Generator().manual_seed(n)
        out = torch.randn(n, n_q + 2 * n_kv, generator=g).bfloat16().to(dev)
        qg, k, v = torch.split(out, [n_q, n_kv, n_kv], dim=-1)
        start = 3600
        positions = torch.arange(start, start + n, device=dev)
        table = torch.randperm(pages, generator=g).to(dev)
        page = table[positions // page_size]
        offset = positions % page_size
        pools = [tuple(torch.zeros(pages, kv_heads, page_size, dim, dtype=torch.bfloat16, device=dev)
                       for _ in range(2)) for _ in range(2)]

        def eager():
            qv = qg.view(n, -1, 2 * dim)
            q, gate = torch.chunk(qv, 2, dim=-1)
            gate = gate.reshape(n, -1)
            q = _rms(q, q_w, 1e-6)
            kk = _rms(k.view(n, -1, dim), k_w, 1e-6)
            vv = v.view(n, -1, dim)
            freqs = positions.float()[:, None] * inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
            q, kk = _rope(q, cos, sin), _rope(kk, cos, sin)
            pools[0][0][page, :, offset] = kk
            pools[0][1][page, :, offset] = vv
            return q, gate

        cos, sin = pk.rope_tables(positions, inv_freq, torch.bfloat16)

        def fused():
            return pk.qk_prologue(qg, k, v, heads, q_w, k_w, 1e-6, 1e-6, cos, sin, *pools[1], page, offset)

        def tables():
            return pk.rope_tables(positions, inv_freq, torch.bfloat16)

        rq, rg = eager()
        fq, fg = fused()
        torch.cuda.synchronize()

        def ulp(a, b):
            return int((a.contiguous().view(torch.int16).int() - b.contiguous().view(torch.int16).int()).abs().max())

        result("prologue_check", new_tokens=n, gate_bitwise=bool(torch.equal(rg, fg)),
               q_max_ulp=ulp(fq, rq), q_bitwise_share=round(float((fq == rq).float().mean()), 5),
               k_max_ulp=ulp(pools[1][0], pools[0][0]),
               k_bitwise_share=round(float((pools[1][0] == pools[0][0]).float().mean()), 6),
               v_bitwise=bool(torch.equal(pools[0][1], pools[1][1])))
        result("prologue_us", new_tokens=n, eager_us=round(graph_us(eager, args.iters), 2),
               fused_us=round(graph_us(fused, args.iters), 2),
               rope_tables_once_us=round(graph_us(tables, args.iters), 2))

        # Tile table, the TP2 layer shapes: 256 experts, top 8, BLOCK_M from the config.
        e, top_k = 256, 8
        pairs = n * top_k
        bm = mp.config(pairs, e).bm
        max_tiles = -(-pairs // bm) + e
        ids = torch.stack([torch.randperm(e, generator=g)[:top_k] for _ in range(n)])
        counts = torch.bincount(ids.reshape(-1), minlength=e).to(torch.int32).to(dev)
        bufs = []
        for one in (False, True):
            meta = torch.full((2, e), -1, dtype=torch.int32, device=dev)
            tab = torch.full((3 * max_tiles + 1,), -7, dtype=torch.int32, device=dev)

            def build(one=one, meta=meta, tab=tab):
                mp.build_tiles(counts, meta[0], meta[1], tab, max_tiles, bm, one=one)

            us = graph_us(build, args.iters)
            bufs.append((meta.clone(), tab.clone(), us))
        result("tiles", new_tokens=n, pairs=pairs, bm=bm, max_tiles=max_tiles,
               grid_us=round(bufs[0][2], 2), one_us=round(bufs[1][2], 2),
               bitwise=bool(torch.equal(bufs[0][0], bufs[1][0]) and torch.equal(bufs[0][1], bufs[1][1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
