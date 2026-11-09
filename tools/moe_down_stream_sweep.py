"""Plan 60 step 0: the N-streaming prefill down projection against today's, on one GCD.

    HIP_VISIBLE_DEVICES=0,1 python tools/moe_down_stream_sweep.py [--news 61,561,1650,2300]

1. Loads the server's model (``Generator.load``, ENGINE_TP default = 2 over the two visible
   GCDs) with eager prefill (ENGINE_PREFILL_GRAPH=0, so Python sees every MoE call; the math
   is the captured pass's) and, as ``tools/prefill_op_split.py`` does, prefills a
   3,600-token context, retires it as a session tail and resumes it with N new tokens, for
   two corpora: ``replay`` (the TraceLab replay's synthetic coding lines) and ``natural``
   (Transformers source). Every ``prefill_kernels.moe`` call of the resumed pass on rank 0
   (GCD 0, the TP2 shard: down [256, 2048, 256]) is recorded with its input, the 40 decoder
   layers and the MTP layer. Routing is the model's router (``prefill_kernels.route``).
2. Per case, on GCD 0, for every recorded call: today's tile table (bucket BM), order and
   act (today's ``_gate_up_kernel``); then the reference pair buffer from today's
   ``_down_kernel`` at the bucket's config. Timed per pass (all calls in one CUDA graph,
   median of --reps replays): today's down, and ``_down_stream_kernel`` over BN {64, 128,
   256} x GROUP*BN {256, 512, 1024, 2048} x warps {4, 8} x stages {1, 2} x nonkdim {0, 16},
   then the 4 fastest with waves_per_eu 2. Each config: ms, effective down-weight TB/s
   (distinct experts touched x H x I x 2 B), VGPR / AGPR / spill counts from the compiled
   kernel, and bitwise equality of the whole pair buffer with today's.
3. The best bitwise-equal config per N is picked on ``replay`` and applied to ``natural``;
   the whole MoE block (``prefill_kernels.moe``) with it is checked bitwise against today's.
   Kill (plan 60): best / today > 0.75 at 1,650 or > 0.85 at 561 (replay) -> park.

Every line starting with RESULT is JSON. Numbers are evidence for the lead, not the metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CONTEXT = 3600
KILL = {561: 0.85, 1650: 0.75}


def result(kind: str, **kw) -> None:
    print("RESULT " + json.dumps({"kind": kind, **kw}), flush=True)


def capture(args):
    import torch

    from engine.fast import prefill_kernels as pk
    from engine.generator import CHECKPOINT, Generator, PackedPrefill
    from engine.protocol import Sampling
    from serve_probe import token_pool

    t0 = time.perf_counter()
    gen = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
        prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30,
    )
    assert gen.fast is not None and gen.packing, "the fast path with packing did not load"
    ranks = getattr(gen.fast, "ranks", None) or [gen.fast]
    r0 = ranks[0]
    mlps = [layer.mlp for layer in r0.layers]
    if getattr(r0, "mtp", None) is not None:
        mlps += [layer.mlp for layer in r0.mtp.layers]
    names = {id(m): i for i, m in enumerate(mlps)}
    dev0 = mlps[0].experts.gate_up_proj.device
    merged = bool(r0.triton_kernels.get("prefill_merged"))
    result("load", seconds=round(time.perf_counter() - t0, 1), ranks=len(ranks), tp=os.environ.get("ENGINE_TP", "default"),
           moe_layers=len(mlps), top_k=r0.top_k, device=str(dev0), merged=merged,
           gate_up=list(mlps[0].experts.gate_up_proj.shape), down=list(mlps[0].experts.down_proj.shape),
           triton_kernels={k: bool(v) for k, v in r0.triton_kernels.items()})
    news = [int(n) for n in args.news.split(",")]
    pools = {
        "replay": token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 60_000, "synthetic"),
        "natural": token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 60_000, "source"),
    }
    packer = PackedPrefill(gen)
    one = Sampling(max_tokens=1, temperature=0.0, ignore_eos=True)
    calls: list = []
    on = [False]
    original = pk.moe

    def recording(x, mlp, top_k, *a, **kw):
        if on[0] and x.device == dev0 and id(mlp) in names:
            calls.append((names[id(mlp)], x.detach().clone()))
        return original(x, mlp, top_k, *a, **kw)

    pk.moe = recording
    cases = {}
    serial = 0
    try:
        for corpus, pool in pools.items():
            for n in news:
                serial += 1
                offset = (serial * 7919) % (len(pool) - CONTEXT - n - 1)
                ids = pool[offset : offset + CONTEXT + n]
                packer.add(tuple(ids[:CONTEXT]), one)
                done = packer.step()
                g = done[0][1]
                assert not isinstance(g, Exception), g
                g.next_step()
                g.retire()
                packer.add(tuple(ids), one)
                calls.clear()
                on[0] = True
                done = packer.step()
                on[0] = False
                torch.cuda.synchronize(dev0)
                g = done[0][1]
                assert not isinstance(g, Exception), g
                assert g.cached_tokens == CONTEXT, (g.cached_tokens, CONTEXT)
                g.next_step()
                gen._reclaim()
                cases[(corpus, n)] = list(calls)
                result("captured", corpus=corpus, new_tokens=n, calls=len(calls),
                       rows=sorted({int(x.shape[0]) for _, x in calls}), layers=len({li for li, _ in calls}))
    finally:
        pk.moe = original
    top_k = r0.top_k
    del packer, gen, ranks, r0
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(dev0)
    result("freed", free_gb=round(free / 2**30, 1), total_gb=round(total / 2**30, 1))
    return mlps, cases, top_k, dev0, merged


def regs(compiled) -> dict:
    """VGPR / AGPR / spills of a compiled Triton kernel (attributes, else the AMDGCN text)."""
    out = {}
    for name in ("n_regs", "n_spills"):
        v = getattr(compiled, name, None)
        if v is not None:
            out[name] = int(v)
    asm = getattr(compiled, "asm", {}) or {}
    text = asm.get("amdgcn") if isinstance(asm, dict) else None
    if text:
        for key in ("vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count"):
            m = re.search(r"\." + key + r":\s*(\d+)", text)
            if m:
                out[key] = int(m.group(1))
        m = re.search(r";\s*Occupancy:\s*(\d+)", text)
        if m:
            out["occupancy"] = int(m.group(1))
    return out


class Case:
    def __init__(self, calls, mlps, top_k, dev):
        import torch
        from torch.nn import functional as F

        from engine.fast import moe_prefill as mp
        from engine.fast import prefill_kernels as pk
        from engine.fast.kernels import triton

        self.items = []
        gu0 = mlps[0].experts.gate_up_proj
        E, two_i, H = gu0.shape
        self.E, self.I, self.H = E, two_i // 2, H
        self.top_k = top_k
        with torch.inference_mode():
            for li, x in calls:
                mlp = mlps[li]
                x = x.contiguous()
                T = x.shape[0]
                pairs = T * top_k
                cfg = mp.config(pairs, E)
                _, top_i, counts = pk.route(F.linear(x, mlp.gate.weight), top_k)
                meta = torch.empty((2, E), dtype=torch.int32, device=dev)
                max_tiles = triton.cdiv(pairs, cfg.bm) + E
                table = torch.empty(3 * max_tiles + 1, dtype=torch.int32, device=dev)
                order = torch.empty(pairs, dtype=torch.int32, device=dev)
                act = torch.empty((pairs, self.I), dtype=torch.bfloat16, device=dev)
                gu = mlp.experts.gate_up_proj
                mp.build_tiles(counts, meta[0], meta[1], table, max_tiles, cfg.bm)
                pk._expert_order_kernel[(triton.cdiv(pairs, 1024),)](top_i, meta[1], order, pairs, BLOCK=1024)
                bn1, bk1 = min(cfg.gate_up.bn, self.I), min(cfg.gate_up.bk, H)
                mp._gate_up_kernel[(max_tiles, triton.cdiv(self.I, bn1))](
                    x, gu, order, table, act, gu.stride(0), gu.stride(1), max_tiles,
                    H=H, I=self.I, TOP_K=top_k, BLOCK_M=cfg.bm, BLOCK_N=bn1, BLOCK_K=bk1, WIDEN=False,
                    **mp.launch_options(cfg.gate_up),
                )
                ref = torch.zeros((pairs, H), dtype=torch.bfloat16, device=dev)
                out = torch.zeros((pairs, H), dtype=torch.bfloat16, device=dev)
                mp.launch_down(act, mlp.experts.down_proj, order, table, ref, max_tiles, cfg, pairs, stream=False)
                self.items.append(dict(layer=li, x=x, mlp=mlp, T=T, pairs=pairs, cfg=cfg, table=table, order=order,
                                       act=act, ref=ref, out=out, max_tiles=max_tiles,
                                       used=int((counts > 0).sum()), max_rows=int(counts.max())))
        torch.cuda.synchronize(dev)
        self.bm = self.items[0]["cfg"].bm
        self.today = self.items[0]["cfg"].down

    def today_down(self, it):
        from engine.fast import moe_prefill as mp

        mp.launch_down(it["act"], it["mlp"].experts.down_proj, it["order"], it["table"], it["out"], it["max_tiles"],
                       it["cfg"], it["pairs"], stream=False)

    def stream_down(self, it, s):
        from engine.fast import moe_prefill as mp

        dn = it["mlp"].experts.down_proj
        bk = min(it["cfg"].down.bk, self.I)
        opts = mp.launch_options(mp.KernelConfig(s.bn, bk, s.warps, s.stages, s.nonkdim, s.waves))
        return mp._down_stream_kernel[(it["max_tiles"], self.H // (s.group * s.bn))](
            it["act"], dn, it["order"], it["table"], it["out"], dn.stride(0), dn.stride(1), it["max_tiles"],
            H=self.H, I=self.I, BLOCK_M=it["cfg"].bm, BLOCK_N=s.bn, BLOCK_K=bk, GROUP=s.group, WIDEN=False, **opts,
        )


def timed(fn, reps: int, dev) -> tuple[float, str]:
    """Median ms of ``fn`` replayed as one CUDA graph (eager launches if capture fails)."""
    import torch

    fn()
    torch.cuda.synchronize(dev)
    mode = "graph"
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        run = graph.replay
    except Exception as exc:  # noqa: BLE001
        result("graph_failed", error=repr(exc)[:200])
        graph = None
        run = fn
        mode = "eager"
    run()
    torch.cuda.synchronize(dev)
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    del graph
    return statistics.median(times), mode


def grid(H: int):
    from engine.fast import moe_prefill as mp

    out = []
    for bn in (64, 128, 256):
        for gbn in (256, 512, 1024, 2048):
            if gbn < bn or gbn > H or H % gbn:
                continue
            for warps in (4, 8):
                for stages in (1, 2):
                    for nonk in (0, 16):
                        out.append(mp.StreamConfig(bn, gbn // bn, warps, stages, nonk, 0))
    return out


def cfg_str(s) -> str:
    return f"BN{s.bn}/G{s.group}/w{s.warps}/s{s.stages}/nk{s.nonkdim}/wpe{s.waves}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--news", default="61,561,1650,2300")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    os.environ["ENGINE_PREFILL_GRAPH"] = "0"
    started = time.time()
    import torch

    print(f"torch {torch.__version__} hip {torch.version.hip} visible={os.environ.get('HIP_VISIBLE_DEVICES')} "
          f"gpus {torch.cuda.device_count()}", flush=True)
    try:
        import triton

        print(f"triton {triton.__version__}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    from engine.fast import moe_prefill as mp
    from engine.fast import prefill_kernels as pk

    mlps, cases, top_k, dev, merged = capture(args)
    torch.cuda.set_device(dev)
    failed: dict[str, str] = {}
    reg_cache: dict = {}
    chosen: dict[int, object] = {}
    summary = {}
    ns = sorted({n for _, n in cases})
    for corpus, n in [("replay", n) for n in ns] + [("natural", n) for n in ns]:
        calls = cases.pop((corpus, n))
        t_case = time.time()
        with torch.inference_mode():
            case = Case(calls, mlps, top_k, dev)
            items = case.items
            used = [it["used"] for it in items]
            bytes_w = sum(u * case.H * case.I * 2 for u in used)
            result("case", corpus=corpus, new_tokens=n, calls=len(items), rows=sorted({it["T"] for it in items}),
                   bm=case.bm, today_cfg=list(case.today), experts_touched_mean=round(sum(used) / len(used), 1),
                   experts_touched_min=min(used), experts_touched_max=max(used),
                   max_rows_per_expert_max=max(it["max_rows"] for it in items),
                   down_weight_gb_per_pass=round(bytes_w / 1e9, 3))
            t_today, mode = timed(lambda: [case.today_down(it) for it in items], args.reps, dev)
            today_eq = all(torch.equal(it["out"], it["ref"]) for it in items)
            result("today", corpus=corpus, new_tokens=n, ms=round(t_today, 3), mode=mode,
                   tb_s=round(bytes_w / (t_today * 1e-3) / 1e12, 3), repeat_bitwise=today_eq)

            res = {}

            def run(cfgs):
                for s in cfgs:
                    key = (case.bm, s)
                    if str(key) in failed:
                        continue
                    try:
                        for it in items:
                            it["out"].zero_()
                        compiled = None
                        for it in items:
                            compiled = case.stream_down(it, s)
                        torch.cuda.synchronize(dev)
                        eq = all(torch.equal(it["out"], it["ref"]) for it in items)
                        if key not in reg_cache:
                            reg_cache[key] = regs(compiled)
                        ms, _ = timed(lambda s=s: [case.stream_down(it, s) for it in items], args.reps, dev)
                        res[s] = (ms, eq)
                        result("cfg", corpus=corpus, new_tokens=n, cfg=cfg_str(s), ms=round(ms, 3),
                               ratio=round(ms / t_today, 3), tb_s=round(bytes_w / (ms * 1e-3) / 1e12, 3),
                               bitwise=eq, **reg_cache[key])
                    except Exception as exc:  # noqa: BLE001
                        failed[str(key)] = repr(exc)[:300]
                        result("cfg_failed", corpus=corpus, new_tokens=n, cfg=cfg_str(s), error=failed[str(key)])
                        torch.cuda.synchronize(dev)

            if corpus == "replay":
                run(grid(case.H))
                top = [s for s, (ms, eq) in sorted(res.items(), key=lambda kv: kv[1][0]) if eq][:4]
                run([s._replace(waves=2) for s in top])
                ok = [(ms, s) for s, (ms, eq) in res.items() if eq]
                chosen[n] = min(ok, key=lambda v: v[0])[1] if ok else None
            else:
                cands = [chosen.get(n)] if chosen.get(n) else []
                run([s for s in cands])
            best = chosen.get(n)
            n_neq = sum(1 for _, (ms, eq) in res.items() if not eq)
            row = dict(corpus=corpus, new_tokens=n, today_ms=round(t_today, 3),
                       today_tb_s=round(bytes_w / (t_today * 1e-3) / 1e12, 3), configs=len(res), configs_not_bitwise=n_neq)
            if best is not None and best in res:
                ms, eq = res[best]
                row.update(chosen=cfg_str(best), chosen_ms=round(ms, 3), ratio=round(ms / t_today, 3),
                           saved_ms=round(t_today - ms, 3), chosen_tb_s=round(bytes_w / (ms * 1e-3) / 1e12, 3),
                           chosen_bitwise=eq, **reg_cache.get((case.bm, best), {}))
                # The whole MoE block with the chosen config against today's (prefill_kernels.moe).
                os.environ["ENGINE_MOE_DOWN_STREAM_CFG"] = ",".join(str(v) for v in best)
                block_eq = True
                for it in items:
                    mp._DOWN_STREAM = False
                    y0 = pk.moe(it["x"], it["mlp"], top_k, merged=merged)
                    mp._DOWN_STREAM = True
                    y1 = pk.moe(it["x"], it["mlp"], top_k, merged=merged)
                    y2 = pk.moe(it["x"], it["mlp"], top_k, merged=merged)
                    block_eq = block_eq and torch.equal(y0, y1) and torch.equal(y1, y2)
                mp._DOWN_STREAM = False
                del os.environ["ENGINE_MOE_DOWN_STREAM_CFG"]
                row["moe_block_bitwise"] = block_eq
            if corpus == "replay":
                top = sorted(((ms, s) for s, (ms, eq) in res.items() if eq), key=lambda v: v[0])[:8]
                result("top", corpus=corpus, new_tokens=n, today=round(t_today, 3),
                       best=[[cfg_str(s), round(ms, 3), round(ms / t_today, 3)] for ms, s in top])
            row["case_s"] = round(time.time() - t_case, 1)
            summary[(corpus, n)] = row
            result("summary", **row)
        del case, items, calls
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== plan 60 step 0: down projection per pass, rank 0 / GCD 0, ms", flush=True)
    print(f"{'corpus':8} {'N':>5} {'today':>8} {'TB/s':>6} {'stream':>8} {'TB/s':>6} {'ratio':>6} {'bitwise':>8} "
          f"{'block':>6}  config", flush=True)
    for (corpus, n), r in summary.items():
        print(f"{corpus:8} {n:>5} {r['today_ms']:>8.3f} {r['today_tb_s']:>6.2f} {r.get('chosen_ms', float('nan')):>8.3f} "
              f"{r.get('chosen_tb_s', float('nan')):>6.2f} {r.get('ratio', float('nan')):>6.3f} "
              f"{str(r.get('chosen_bitwise')):>8} {str(r.get('moe_block_bitwise')):>6}  {r.get('chosen')}", flush=True)
    go = True
    for n, limit in KILL.items():
        r = summary.get(("replay", n))
        go = go and r is not None and r.get("ratio", 9.0) <= limit and bool(r.get("chosen_bitwise"))
    result("verdict", go=go, rule={str(k): v for k, v in KILL.items()},
           ratios_replay={str(n): summary.get(("replay", n), {}).get("ratio") for n in ns},
           ratios_natural={str(n): summary.get(("natural", n), {}).get("ratio") for n in ns},
           chosen={str(n): cfg_str(s) if s else None for n, s in chosen.items()},
           configs_failed=len(failed), seconds=round(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
