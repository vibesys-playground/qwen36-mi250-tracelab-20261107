"""Plan 38 step 0: TP2-shaped grouped MoE prefill kernels, swept on one GCD with real routing.

    HIP_VISIBLE_DEVICES=0,1 python tools/moe_prefill_tp2_sweep.py [--news 61,561,1650,2300]

1. Loads the server's model (``Generator.load``, ENGINE_TP default = 2 over the two visible
   GCDs) and, as ``tools/prefill_op_split.py`` does, prefills a 3,600-token context, retires
   it as a session tail and resumes it with N new tokens, for two corpora: ``replay`` (the
   TraceLab replay's synthetic coding lines, ``serve_probe.token_pool(..., "synthetic")``)
   and ``natural`` (Transformers source text). During the resumed pass every call of
   ``prefill_kernels.moe`` on rank 0 (GCD 0) is recorded: the layer and its input x. Those
   are the real per-GCD inputs; rank 0's experts are the TP2 shard ([256, 512, 2048] and
   [256, 2048, 256]); routing is the model's router through ``prefill_kernels.route``.
2. Per case (corpus, N), on GCD 0 only, over all recorded calls of the pass (one per MoE
   layer per chunk), it times with CUDA events (median of --reps after 2 warm runs):
     today: bookkeeping (offsets + order kernels) + ``moe_prefill._gate_up_kernel`` +
            ``_down_kernel`` (fp32 pair buffer) at DEFAULT_CONFIG + ``_moe_combine_kernel``;
     new:   bookkeeping + per-tile table (``moe_prefill_tp2._tile_table_kernel``) + the table
            kernels ``_gate_up_kernel_t`` / ``_down_kernel_t`` (BF16 pair buffer) at every
            swept config + ``_combine_kernel_bf16``.
   Stage A sweeps BM {16,32,64,128} x BN x BK x warps {4,8} x stages {1,2} per kernel
   (gate_up and down independently); stage B retries the 4 fastest of each kernel per
   replay case with matrix_instr_nonkdim {16,32} and waves_per_eu {0,2}.
3. Bitwise: today's decomposed path == ``prefill_kernels.moe`` (harness sanity); the table +
   BF16 pair buffer at DEFAULT_CONFIG == ``prefill_kernels.moe`` (must be bitwise); the
   chosen per-N config vs ``prefill_kernels.moe`` (equal, or max |diff| and share differing).
4. Per N the config is chosen on the replay case only (a function of the token count) and
   applied to the natural case too. Kill rule (plan 38): new/today <= 0.80 at 561 AND <= 0.75
   at 1,650 (bookkeeping + gate_up + down + combine), else park.

Every line starting with RESULT is JSON. Numbers are evidence for the lead, not the metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CONTEXT = 3600
KILL = {561: 0.80, 1650: 0.75}


def result(kind: str, **kw) -> None:
    print("RESULT " + json.dumps({"kind": kind, **kw}), flush=True)


# ============================================================================ capture


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
    index = {id(m): i for i, m in enumerate(mlps)}
    dev0 = mlps[0].experts.gate_up_proj.device
    result("load", seconds=round(time.perf_counter() - t0, 1), ranks=len(ranks), tp=os.environ.get("ENGINE_TP", "default"),
           layers=len(mlps), top_k=r0.top_k, device=str(dev0),
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

    def recording(x, mlp, top_k):
        if on[0] and x.device == dev0 and id(mlp) in index:
            calls.append((index[id(mlp)], x.detach().clone()))
        return original(x, mlp, top_k)

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
                       rows=[int(x.shape[0]) for _, x in calls[:3]], layers=sorted({li for li, _ in calls})[:5])
    finally:
        pk.moe = original
    top_k = r0.top_k
    del packer, gen, ranks, r0, g, done
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(dev0)
    result("freed", free_gb=round(free / 2**30, 1), total_gb=round(total / 2**30, 1))
    return mlps, cases, top_k, dev0


# ============================================================================ kernels


class Case:
    """One prefill pass's recorded MoE calls on GCD 0, with routing and per-BM bookkeeping."""

    def __init__(self, calls, mlps, top_k, dev, bms):
        import torch
        from torch.nn import functional as F

        from engine.fast import moe_prefill as mp
        from engine.fast import moe_prefill_tp2 as m2
        from engine.fast import prefill_kernels as pk
        from engine.fast.kernels import triton

        self.top_k = top_k
        self.items = []
        self.max_pairs = max(int(x.shape[0]) for _, x in calls) * top_k
        gu0 = mlps[0].experts.gate_up_proj
        E, two_i, H = gu0.shape
        self.E, self.I, self.H = E, two_i // 2, H
        self.out32 = torch.empty((self.max_pairs, H), dtype=torch.float32, device=dev)
        self.out16 = torch.empty((self.max_pairs, H), dtype=torch.bfloat16, device=dev)
        with torch.inference_mode():
            for li, x in calls:
                mlp = mlps[li]
                x = x.contiguous()
                T = x.shape[0]
                it = {"layer": li, "x": x, "mlp": mlp, "T": T, "pairs": T * top_k}
                it["y_ref"] = pk.moe(x, mlp, top_k)
                shared = mlp.shared_expert
                g = F.linear(x, shared.gate_proj.weight)
                u = F.linear(x, shared.up_proj.weight)
                act = torch.empty_like(g)
                pk._silu_mul_kernel[(triton.cdiv(g.numel(), 1024),)](g, u, act, g.numel(), BLOCK=1024, WIDEN=False)
                it["shared"] = F.linear(act, shared.down_proj.weight).contiguous()
                it["sg"] = F.linear(x, mlp.shared_expert_gate.weight)
                top_w, top_i, counts = pk.route(F.linear(x, mlp.gate.weight), top_k)
                it.update(top_w=top_w, top_w_flat=top_w.reshape(-1), top_i=top_i, counts=counts)
                it["used"] = int((counts > 0).sum())
                it["max_rows"] = int(counts.max())
                pairs = T * top_k
                meta = torch.empty((3, E), dtype=torch.int32, device=dev)
                it["starts"], it["cursor"], it["te_today"] = meta[0], meta[1], meta[2]
                it["order"] = torch.empty(pairs, dtype=torch.int32, device=dev)
                # One order for every BM (rows of an expert run in this order for gate_up and down).
                pk._expert_offsets_kernel[(1,)](counts, it["starts"], it["cursor"], it["te_today"], E=E, BLOCK_M=64)
                pk._expert_order_kernel[(triton.cdiv(pairs, 1024),)](top_i, it["cursor"], it["order"], pairs, BLOCK=1024)
                it["bm"] = {}
                for bm in bms:
                    scratch = torch.empty((3, E), dtype=torch.int32, device=dev)
                    pk._expert_offsets_kernel[(1,)](counts, scratch[0], scratch[1], scratch[2], E=E, BLOCK_M=bm)
                    max_tiles = triton.cdiv(pairs, bm) + E
                    table = torch.empty(3 * max_tiles, dtype=torch.int32, device=dev)
                    m2._tile_table_kernel[(E,)](counts, it["starts"], scratch[2], table, max_tiles, BLOCK_M=bm)
                    act_r = torch.empty((pairs, self.I), dtype=torch.bfloat16, device=dev)
                    b = {"te": scratch[2], "scratch": scratch, "table": table, "total": scratch[2][E - 1 :],
                         "max_tiles": max_tiles, "act": act_r}
                    it["bm"][bm] = b
                    # act for this BM (an input to the down timings): today's gate_up kernel.
                    mp._gate_up_kernel[(max_tiles, triton.cdiv(self.I, 64))](
                        x, mlp.experts.gate_up_proj, it["order"], b["te"], counts, it["starts"], act_r,
                        mlp.experts.gate_up_proj.stride(0), mlp.experts.gate_up_proj.stride(1),
                        H=H, I=self.I, E=E, LOG_E=max(1, (E - 1).bit_length()), TOP_K=top_k,
                        BLOCK_M=bm, BLOCK_N=64, BLOCK_K=128, WIDEN=False, num_warps=4, num_stages=2,
                    )
                self.items.append(it)
        torch.cuda.synchronize(dev)

    # ------------------------------------------------------------------ launches

    @staticmethod
    def opts(c):
        bm, bn, bk, warps, stages, nonk, wpe = c
        d = {"num_warps": warps, "num_stages": stages}
        if nonk:
            d["matrix_instr_nonkdim"] = nonk
        if wpe:
            d["waves_per_eu"] = wpe
        return d

    def book_today(self, it):
        from engine.fast import prefill_kernels as pk
        from engine.fast.kernels import triton

        pk._expert_offsets_kernel[(1,)](it["counts"], it["starts"], it["cursor"], it["te_today"], E=self.E, BLOCK_M=64)
        pk._expert_order_kernel[(triton.cdiv(it["pairs"], 1024),)](it["top_i"], it["cursor"], it["order"], it["pairs"], BLOCK=1024)

    def book_new(self, it, bm_g, bm_d):
        """offsets (BM of gate_up) + order + table; a second offsets + table when down's BM differs."""
        from engine.fast import moe_prefill_tp2 as m2
        from engine.fast import prefill_kernels as pk
        from engine.fast.kernels import triton

        b = it["bm"][bm_g]
        pk._expert_offsets_kernel[(1,)](it["counts"], it["starts"], it["cursor"], b["te"], E=self.E, BLOCK_M=bm_g)
        pk._expert_order_kernel[(triton.cdiv(it["pairs"], 1024),)](it["top_i"], it["cursor"], it["order"], it["pairs"], BLOCK=1024)
        m2._tile_table_kernel[(self.E,)](it["counts"], it["starts"], b["te"], b["table"], b["max_tiles"], BLOCK_M=bm_g)
        if bm_d != bm_g:
            d = it["bm"][bm_d]
            s2 = d["scratch"]
            pk._expert_offsets_kernel[(1,)](it["counts"], s2[0], s2[1], d["te"], E=self.E, BLOCK_M=bm_d)
            m2._tile_table_kernel[(self.E,)](it["counts"], it["starts"], d["te"], d["table"], d["max_tiles"], BLOCK_M=bm_d)

    def gu_old(self, it, c, te=None):
        from engine.fast import moe_prefill as mp
        from engine.fast.kernels import triton

        bm, bn, bk = c[0], min(c[1], self.I), min(c[2], self.H)
        b = it["bm"][bm]
        gu = it["mlp"].experts.gate_up_proj
        mp._gate_up_kernel[(b["max_tiles"], triton.cdiv(self.I, bn))](
            it["x"], gu, it["order"], b["te"] if te is None else te, it["counts"], it["starts"], b["act"],
            gu.stride(0), gu.stride(1), H=self.H, I=self.I, E=self.E, LOG_E=max(1, (self.E - 1).bit_length()),
            TOP_K=self.top_k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, WIDEN=False, **self.opts(c),
        )

    def dn_old(self, it, c):
        from engine.fast import moe_prefill as mp
        from engine.fast.kernels import triton

        bm, bn, bk = c[0], min(c[1], self.H), min(c[2], self.I)
        b = it["bm"][bm]
        dn = it["mlp"].experts.down_proj
        mp._down_kernel[(b["max_tiles"], triton.cdiv(self.H, bn))](
            b["act"], dn, it["top_w_flat"], it["order"], b["te"], it["counts"], it["starts"], self.out32,
            dn.stride(0), dn.stride(1), H=self.H, I=self.I, E=self.E, LOG_E=max(1, (self.E - 1).bit_length()),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, WIDEN=False, **self.opts(c),
        )

    def gu_new(self, it, c, act=None):
        from engine.fast import moe_prefill_tp2 as m2
        from engine.fast.kernels import triton

        bm, bn, bk = c[0], min(c[1], self.I), min(c[2], self.H)
        b = it["bm"][bm]
        gu = it["mlp"].experts.gate_up_proj
        m2._gate_up_kernel_t[(b["max_tiles"], triton.cdiv(self.I, bn))](
            it["x"], gu, it["order"], b["table"], b["total"], b["act"] if act is None else act,
            gu.stride(0), gu.stride(1), b["max_tiles"], H=self.H, I=self.I, TOP_K=self.top_k,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, WIDEN=False, **self.opts(c),
        )

    def dn_new(self, it, c, act=None):
        from engine.fast import moe_prefill_tp2 as m2
        from engine.fast.kernels import triton

        bm, bn, bk = c[0], min(c[1], self.H), min(c[2], self.I)
        b = it["bm"][bm]
        dn = it["mlp"].experts.down_proj
        m2._down_kernel_t[(b["max_tiles"], triton.cdiv(self.H, bn))](
            b["act"] if act is None else act, dn, it["order"], b["table"], b["total"], self.out16,
            dn.stride(0), dn.stride(1), b["max_tiles"], H=self.H, I=self.I,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, WIDEN=False, **self.opts(c),
        )

    def comb32(self, it, y):
        from engine.fast import prefill_kernels as pk
        from engine.fast.kernels import triton

        block = min(1024, triton.next_power_of_2(self.H))
        pk._moe_combine_kernel[(it["T"], triton.cdiv(self.H, block))](
            self.out32, it["sg"], it["shared"], y, H=self.H, K=self.top_k, BLOCK=block, WIDEN=False,
        )

    def comb16(self, it, y):
        from engine.fast import moe_prefill_tp2 as m2
        from engine.fast.kernels import triton

        block = min(1024, triton.next_power_of_2(self.H))
        m2._combine_kernel_bf16[(it["T"], triton.cdiv(self.H, block))](
            self.out16, it["top_w_flat"], it["sg"], it["shared"], y, H=self.H, K=self.top_k, BLOCK=block, WIDEN=False,
        )

    # ------------------------------------------------------------------ full paths (bitwise)

    def full_today(self, it):
        """Today's routed path decomposed as in ``prefill_kernels.moe`` (fresh bookkeeping)."""
        import torch

        y = torch.empty_like(it["x"])
        self.book_today(it)
        b64 = it["bm"][64]
        self.gu_old(it, (64, 64, 128, 4, 2, 0, 0), te=it["te_today"])
        # down reads the act that gate_up just wrote: point the BM=64 act at it.
        from engine.fast import moe_prefill as mp
        from engine.fast.kernels import triton

        dn = it["mlp"].experts.down_proj
        mp._down_kernel[(b64["max_tiles"], triton.cdiv(self.H, 64))](
            b64["act"], dn, it["top_w_flat"], it["order"], it["te_today"], it["counts"], it["starts"], self.out32,
            dn.stride(0), dn.stride(1), H=self.H, I=self.I, E=self.E, LOG_E=max(1, (self.E - 1).bit_length()),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, WIDEN=False, num_warps=4, num_stages=2,
        )
        self.comb32(it, y)
        return y

    def full_new(self, it, cg, cd):
        import torch

        y = torch.empty_like(it["x"])
        act = torch.empty((it["pairs"], self.I), dtype=torch.bfloat16, device=it["x"].device)
        self.book_new(it, cg[0], cd[0])
        self.gu_new(it, cg, act=act)
        self.dn_new(it, cd, act=act)
        self.comb16(it, y)
        return y


# ============================================================================ timing


def timed(fn, reps: int, dev) -> float:
    import torch

    fn()
    fn()
    torch.cuda.synchronize(dev)
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def sweep_grid():
    gu, dn = [], []
    for bm in (16, 32, 64, 128):
        for warps in (4, 8):
            for stages in (1, 2):
                for bn in (32, 64, 128):
                    for bk in (64, 128):
                        gu.append((bm, bn, bk, warps, stages, 0, 0))
                for bn in (64, 128, 256):
                    for bk in (64, 128, 256):
                        dn.append((bm, bn, bk, warps, stages, 0, 0))
    return gu, dn


def stage_b(cfgs_ms: dict, top: int = 4):
    ranked = [c for c, t in sorted(cfgs_ms.items(), key=lambda kv: kv[1]) if t == t and t != float("inf")][:top]
    out = []
    for c in ranked:
        bm, bn, bk, warps, stages, _, _ = c
        for nonk in (0, 16, 32):
            for wpe in (0, 2):
                if nonk == 32 and (bm < 32 or bn < 32):
                    continue
                v = (bm, bn, bk, warps, stages, nonk, wpe)
                if v not in cfgs_ms:
                    out.append(v)
    return out


def cfg_str(c) -> str:
    return "BM{}/BN{}/BK{}/w{}/s{}/nk{}/wpe{}".format(*c)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--news", default="61,561,1650,2300")
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    started = time.time()
    import torch

    print(f"torch {torch.__version__} hip {torch.version.hip} visible={os.environ.get('HIP_VISIBLE_DEVICES')} "
          f"gpus {torch.cuda.device_count()}", flush=True)
    try:
        import triton

        print(f"triton {triton.__version__}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    mlps, cases, top_k, dev = capture(args)
    torch.cuda.set_device(dev)
    bms = (16, 32, 64, 128)
    today_cfg = (64, 64, 128, 4, 2, 0, 0)
    gu_grid, dn_grid = sweep_grid()
    gu_grid = [today_cfg] + [c for c in gu_grid if c != today_cfg]
    dn_grid = [today_cfg] + [c for c in dn_grid if c != today_cfg]
    extra_gu: list = []
    extra_dn: list = []
    chosen: dict[int, tuple] = {}
    summary = {}
    failed: dict[str, str] = {}
    order = [("replay", n) for n in sorted({n for _, n in cases})] + [("natural", n) for n in sorted({n for _, n in cases})]
    for corpus, n in order:
        calls = cases.pop((corpus, n))
        t_case = time.time()
        with torch.inference_mode():
            case = Case(calls, mlps, top_k, dev, bms)
            items = case.items
            pairs_total = sum(it["pairs"] for it in items)
            used = [it["used"] for it in items]
            bytes_w = sum(it["used"] * (2 * case.I * case.H + case.H * case.I) * 2 for it in items)
            flops = sum(it["pairs"] * 2 * (2 * case.I * case.H + case.H * case.I) for it in items)
            result("case", corpus=corpus, new_tokens=n, calls=len(items), pairs_per_call=items[0]["pairs"],
                   experts_used_mean=round(sum(used) / len(used), 1), experts_used_min=min(used),
                   max_rows_per_expert_mean=round(sum(it["max_rows"] for it in items) / len(items), 1),
                   weight_gb_per_pass=round(bytes_w / 1e9, 2), real_tflop_per_pass=round(flops / 1e12, 3))

            def over(f):
                return lambda: [f(it) for it in items]

            # Bookkeeping and combine.
            t_book_today = timed(over(case.book_today), args.reps, dev)
            t_book_new = {bm: timed(over(lambda it, bm=bm: case.book_new(it, bm, bm)), args.reps, dev) for bm in bms}
            ys = [torch.empty_like(it["x"]) for it in items]
            t_c32 = timed(lambda: [case.comb32(it, y) for it, y in zip(items, ys)], args.reps, dev)
            t_c16 = timed(lambda: [case.comb16(it, y) for it, y in zip(items, ys)], args.reps, dev)
            del ys
            # The timed bookkeeping rewrote ``order`` (another valid permutation; atomics) and the
            # tables identically; the timed GEMMs only need valid inputs.

            def sweep(kind, grid):
                res = {}
                fn_new = case.gu_new if kind == "gu" else case.dn_new
                for c in grid:
                    key = f"{kind}:{c}"
                    if key in failed:
                        res[c] = float("inf")
                        continue
                    try:
                        res[c] = timed(over(lambda it, c=c: fn_new(it, c)), args.reps, dev)
                    except Exception as exc:  # noqa: BLE001
                        failed[key] = repr(exc)[:200]
                        res[c] = float("inf")
                        result("cfg_failed", kernel=kind, cfg=cfg_str(c), error=failed[key])
                        torch.cuda.synchronize(dev)
                return res

            t0 = time.time()
            gu_ms = sweep("gu", gu_grid + extra_gu)
            dn_ms = sweep("dn", dn_grid + extra_dn)
            if corpus == "replay":
                b_gu = stage_b(gu_ms)
                b_dn = stage_b(dn_ms)
                gu_ms.update(sweep("gu", b_gu))
                dn_ms.update(sweep("dn", b_dn))
                extra_gu += [c for c in b_gu if c not in extra_gu]
                extra_dn += [c for c in b_dn if c not in extra_dn]
            sweep_s = time.time() - t0
            # Today's kernels.
            t_gu_old = timed(over(lambda it: case.gu_old(it, today_cfg)), args.reps, dev)
            t_dn_old = timed(over(lambda it: case.dn_old(it, today_cfg)), args.reps, dev)
            today_total = t_book_today + t_gu_old + t_dn_old + t_c32
            # Best per kernel, and the best same-BM pair.
            best_gu = min(gu_ms, key=gu_ms.get)
            best_dn = min(dn_ms, key=dn_ms.get)
            same = {}
            for bm in bms:
                g = min((c for c in gu_ms if c[0] == bm), key=gu_ms.get)
                d = min((c for c in dn_ms if c[0] == bm), key=dn_ms.get)
                same[bm] = (g, d, t_book_new[bm] + gu_ms[g] + dn_ms[d] + t_c16)
            bm_same = min(same, key=lambda b: same[b][2])
            # Mixed BM: time its bookkeeping (two tables).
            if best_gu[0] != best_dn[0]:
                t_book_mixed = timed(over(lambda it: case.book_new(it, best_gu[0], best_dn[0])), args.reps, dev)
            else:
                t_book_mixed = t_book_new[best_gu[0]]
            mixed_total = t_book_mixed + gu_ms[best_gu] + dn_ms[best_dn] + t_c16
            if mixed_total < same[bm_same][2]:
                pick = (best_gu, best_dn, mixed_total, t_book_mixed)
            else:
                pick = (same[bm_same][0], same[bm_same][1], same[bm_same][2], t_book_new[bm_same])
            if corpus == "replay":
                chosen[n] = (pick[0], pick[1])
            cg, cd = chosen.get(n, (pick[0], pick[1]))
            # The replay-chosen config on this case (natural: transferred).
            if cg[0] != cd[0]:
                t_book_ch = timed(over(lambda it: case.book_new(it, cg[0], cd[0])), args.reps, dev)
            else:
                t_book_ch = t_book_new[cg[0]]
            chosen_total = t_book_ch + gu_ms.get(cg, float("nan")) + dn_ms.get(cd, float("nan")) + t_c16
            # Separations: (b)+(c) alone at today's config; (a) alone (today's kernels, chosen cfg).
            bc_only = t_book_new[64] + gu_ms[today_cfg] + dn_ms[today_cfg] + t_c16
            try:
                t_gu_old_ch = timed(over(lambda it: case.gu_old(it, cg)), args.reps, dev)
                t_dn_old_ch = timed(over(lambda it: case.dn_old(it, cd)), args.reps, dev)
                a_only = t_book_today + t_gu_old_ch + t_dn_old_ch + t_c32
            except Exception as exc:  # noqa: BLE001
                a_only = float("nan")
                result("a_only_failed", error=repr(exc)[:200])
            # Bitwise.
            harness_eq = all(torch.equal(case.full_today(it), it["y_ref"]) for it in items)
            bc_eq = all(torch.equal(case.full_new(it, today_cfg, today_cfg), it["y_ref"]) for it in items)
            n_diff = n_el = 0
            max_diff = 0.0
            for it in items:
                y = case.full_new(it, cg, cd)
                d = (y.float() - it["y_ref"].float()).abs()
                n_diff += int((d > 0).sum())
                n_el += d.numel()
                max_diff = max(max_diff, float(d.max()))
            ch_eq = n_diff == 0
            repeat_eq = all(torch.equal(case.full_new(it, cg, cd), case.full_new(it, cg, cd)) for it in items[:8])
            kern_today = t_gu_old + t_dn_old
            kern_ch = gu_ms.get(cg, float("nan")) + dn_ms.get(cd, float("nan"))
            row = dict(
                corpus=corpus, new_tokens=n, calls=len(items),
                today_ms=round(today_total, 3), today_book=round(t_book_today, 3), today_gate_up=round(t_gu_old, 3),
                today_down=round(t_dn_old, 3), today_combine=round(t_c32, 3),
                chosen_gu=cfg_str(cg), chosen_dn=cfg_str(cd), chosen_ms=round(chosen_total, 3),
                chosen_book=round(t_book_ch, 3), chosen_gate_up=round(gu_ms.get(cg, float("nan")), 3),
                chosen_down=round(dn_ms.get(cd, float("nan")), 3), chosen_combine=round(t_c16, 3),
                ratio_total=round(chosen_total / today_total, 3),
                ratio_expert_kernels=round(kern_ch / kern_today, 3),
                own_best_ms=round(pick[2], 3), own_best_ratio=round(pick[2] / today_total, 3),
                own_best_gu=cfg_str(pick[0]), own_best_dn=cfg_str(pick[1]),
                table_bf16_at_today_cfg_ms=round(bc_only, 3), ratio_table_bf16_only=round(bc_only / today_total, 3),
                tuned_today_kernels_ms=round(a_only, 3), ratio_tuning_only=round(a_only / today_total, 3),
                today_tb_s=round(bytes_w / (kern_today * 1e-3) / 1e12, 2),
                today_tf_s=round(flops / (kern_today * 1e-3) / 1e12, 1),
                chosen_tb_s=round(bytes_w / (kern_ch * 1e-3) / 1e12, 2),
                chosen_tf_s=round(flops / (kern_ch * 1e-3) / 1e12, 1),
                bitwise_harness_today_eq_moe=harness_eq,
                bitwise_table_bf16_today_cfg_eq_moe=bc_eq,
                bitwise_chosen_eq_moe=ch_eq, chosen_elems_differing=n_diff, chosen_elems=n_el,
                chosen_max_abs_diff=max_diff, chosen_repeatable=repeat_eq,
                sweep_s=round(sweep_s, 1), case_s=round(time.time() - t_case, 1),
            )
            summary[(corpus, n)] = row
            result("summary", **row)
            for kind, ms in (("gu", gu_ms), ("dn", dn_ms)):
                top = sorted(((t, c) for c, t in ms.items() if t != float("inf")))[:6]
                result("top", corpus=corpus, new_tokens=n, kernel=kind, today=round(ms[today_cfg], 3),
                       best=[[cfg_str(c), round(t, 3)] for t, c in top])
                per_bm = {}
                for bm in bms:
                    vals = [t for c, t in ms.items() if c[0] == bm and t != float("inf")]
                    per_bm[bm] = round(min(vals), 3) if vals else None
                result("best_per_bm", corpus=corpus, new_tokens=n, kernel=kind, ms=per_bm)
        del case, items, calls
        gc.collect()
        torch.cuda.empty_cache()
    # Table and verdict.
    print("\n=== plan 38 step 0: per pass (all MoE calls of rank 0 / GCD 0), ms", flush=True)
    print(f"{'corpus':8} {'N':>5} {'today':>8} {'chosen':>8} {'ratio':>6} {'kern_r':>6} {'tbl+bf16':>8} "
          f"{'tune':>8} {'bitwise(b+c)':>12} {'bitwise(ch)':>11}  chosen gate_up | down", flush=True)
    for (corpus, n), r in summary.items():
        print(f"{corpus:8} {n:>5} {r['today_ms']:>8.2f} {r['chosen_ms']:>8.2f} {r['ratio_total']:>6.3f} "
              f"{r['ratio_expert_kernels']:>6.3f} {r['table_bf16_at_today_cfg_ms']:>8.2f} {r['tuned_today_kernels_ms']:>8.2f} "
              f"{str(r['bitwise_table_bf16_today_cfg_eq_moe']):>12} {str(r['bitwise_chosen_eq_moe']):>11}  "
              f"{r['chosen_gu']} | {r['chosen_dn']}", flush=True)
    ok = True
    for n, limit in KILL.items():
        r = summary.get(("replay", n))
        if r is None:
            ok = False
            continue
        ok = ok and r["ratio_total"] <= limit
    bc_all = all(r["bitwise_table_bf16_today_cfg_eq_moe"] for r in summary.values())
    result("verdict", go=ok, rule={str(k): v for k, v in KILL.items()},
           ratios_replay={str(n): summary[("replay", n)]["ratio_total"] for n in KILL if ("replay", n) in summary},
           ratios_natural={str(n): summary[("natural", n)]["ratio_total"] for n in KILL if ("natural", n) in summary},
           table_bf16_bitwise_all=bc_all, configs_failed=len(failed), seconds=round(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
