"""Plan 11 stage 0: can a persistent per-GCD decode megakernel beat the per-kernel fast path?

    python tools/mega_stage0.py [--layers 4] [--batches 1,2,4]

Runs on one GPU (an MI210 is one gfx90a GCD). Three parts, all printed as ``RESULT`` lines:

(a) Grid barrier. A persistent Triton kernel with one or two CTAs per CU (all resident) runs
    ITERS software grid barriers (an atomic counter that only grows, then a spin on it), each
    checked by a neighbour-read: every CTA writes a value before the barrier and reads its
    neighbour's after it. Microseconds per barrier = (time with ITERS - time with 0) / ITERS.
    Also the per-node cost of a HIP graph of tiny kernels (1 CTA and one CTA per CU).
(b) One-layer prototype on real weights of layer 0 (a Gated DeltaNet layer) from the HF cache:
    one persistent launch doing (1) the four GDN input projections as one concatenated GEMV,
    (2) router | shared gate | shared up | shared-expert gate as one concatenated GEMV,
    (3) softmax + top-8 + renormalize, (4) routed experts' gate_up, (5) routed down + combine
    + shared expert down + gate, with a grid barrier between phases. Against the current
    fast-path kernels for the same work (``FastModel._gdn_project`` + ``FastModel._moe``),
    both captured in a HIP graph, at each batch; output parity (top-8 sets, max abs diff) and
    run-to-run determinism. The same kernel launched as five phase kernels (no barrier) is
    reported as the fusion-only variant.
(c) The existing fast-path decode of one GDN layer and one attention layer at B=2 (graph
    replay and eager), the cost of each of its ops alone (graph of repeats, with GB/s), and
    a torch.profiler kernel breakdown of the GDN layer (eager and graph replay).

Kill criteria (plan 11): barrier > 15 us, or prototype < 1.8x faster than the current
kernels at B=2; narrow to fusion-only if kernels are busy >= 70% of the layer at >= 0.7 TB/s.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
MAX_SPIN = tl.constexpr(1_000_000)

# --------------------------------------------------------------------------------------------
# Grid barrier
# --------------------------------------------------------------------------------------------


@triton.jit
def _grid_barrier(bar_ptr, target, err_ptr, SPIN_VOLATILE: tl.constexpr):
    """Arrive (acq_rel atomic add, agent scope), then wait until the counter reaches
    ``target``. The counter only grows, so no reset is needed between barriers or launches."""
    tl.debug_barrier()  # every thread of this CTA has issued its phase's stores
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    cnt = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    spins = 0
    while (cnt < target) & (spins < MAX_SPIN):
        if SPIN_VOLATILE:
            cnt = tl.load(bar_ptr, volatile=True)
        else:
            cnt = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        spins += 1
    if SPIN_VOLATILE:  # order the phase's later loads after the counter (acquire)
        cnt = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    if spins >= MAX_SPIN:
        tl.atomic_add(err_ptr, 1)
    tl.debug_barrier()


@triton.jit
def _epoch_base(bar_ptr, per_launch):
    """The counter value at this launch's start: previous launches finished all their
    barriers, and this launch cannot have completed one before this CTA arrives."""
    c0 = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    return (c0 // per_launch) * per_launch


@triton.jit
def _arrive_once(bar_ptr):
    tl.atomic_add(bar_ptr, 1)


@triton.jit
def _barrier_bench(bar_ptr, err_ptr, buf_ptr, bad_ptr, iters, per_barrier,
                   SPIN_VOLATILE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    base = _epoch_base(bar_ptr, tl.maximum(per_barrier * iters, 1))
    nb = (pid + 1) % nprog
    bad = 0
    for it in range(iters):
        tl.store(buf_ptr + (it % 2) * nprog + pid, it * 7 + pid)
        _grid_barrier(bar_ptr, base + (it + 1) * per_barrier, err_ptr, SPIN_VOLATILE)
        v = tl.load(buf_ptr + (it % 2) * nprog + nb, volatile=True)
        bad += (v != it * 7 + nb).to(tl.int32)
    tl.atomic_add(bad_ptr, bad)


@triton.jit
def _tiny(out_ptr):
    tl.store(out_ptr + tl.program_id(0), 1.0)


# --------------------------------------------------------------------------------------------
# One-layer prototype: input projections, router/shared GEMV, top-k, routed + shared experts
# --------------------------------------------------------------------------------------------


@triton.jit
def _gemv_tile(x_ptr, w_ptr, out_ptr, N, tile, B: tl.constexpr, BP: tl.constexpr,
               H: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[b, rows] = bf16(W[rows, :] . x[b, :]) for one tile of BN rows and all B rows of x."""
    rows = tile * BN + tl.arange(0, BN)
    rok = rows < N
    bs = tl.arange(0, BP)
    bok = bs < B
    acc = tl.zeros([BN, BP], dtype=tl.float32)
    for k0 in range(0, H, BK):
        cols = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rows[:, None].to(tl.int64) * H + cols[None, :], mask=rok[:, None],
                    other=0.0).to(tl.float32)
        x = tl.load(x_ptr + bs[:, None] * H + cols[None, :], mask=bok[:, None],
                    other=0.0).to(tl.float32)
        acc += tl.sum(w[:, None, :] * x[None, :, :], axis=2)
    tl.store(out_ptr + bs[None, :] * N + rows[:, None], acc.to(tl.bfloat16),
             mask=rok[:, None] & bok[None, :])


@triton.jit
def _mega_proto(
    h1_ptr, w1_ptr, out1_ptr, h2_ptr, w3_ptr, out3_ptr, idx_ptr, wt_ptr,
    gu_ptr, act_ptr, dn_ptr, sd_ptr, out_ptr, bar_ptr, err_ptr, per_barrier,
    stride_gue, stride_dne,
    B: tl.constexpr, BP: tl.constexpr, H: tl.constexpr, N1: tl.constexpr, N3: tl.constexpr,
    E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr, I: tl.constexpr,
    IS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BKI: tl.constexpr, BNE: tl.constexpr,
    PHASE: tl.constexpr, SPIN_VOLATILE: tl.constexpr,
):
    """PHASE 0: all five phases in one persistent launch, a grid barrier between phases.
    PHASE p in 1..5: only phase p (one ordinary launch per phase, no barrier)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if PHASE == 0:
        base = _epoch_base(bar_ptr, 4 * per_barrier)

    # (1) GDN input projections qkv | z | b | a: one concatenated GEMV of h1.
    if (PHASE == 0) | (PHASE == 1):
        for t in range(pid, tl.cdiv(N1, BN), nprog):
            _gemv_tile(h1_ptr, w1_ptr, out1_ptr, N1, t, B, BP, H, BN, BK)
    if PHASE == 0:
        _grid_barrier(bar_ptr, base + per_barrier, err_ptr, SPIN_VOLATILE)

    # (2) router | shared gate | shared up | shared-expert gate: one concatenated GEMV of h2.
    if (PHASE == 0) | (PHASE == 2):
        for t in range(pid, tl.cdiv(N3, BN), nprog):
            _gemv_tile(h2_ptr, w3_ptr, out3_ptr, N3, t, B, BP, H, BN, BK)
    if PHASE == 0:
        _grid_barrier(bar_ptr, base + 2 * per_barrier, err_ptr, SPIN_VOLATILE)

    # (3) softmax over the bf16 router logits in fp32, top-k, renormalize, round to bf16.
    if (PHASE == 0) | (PHASE == 3):
        for r in range(pid, B, nprog):
            ee = tl.arange(0, E)
            logit = tl.load(out3_ptr + r * N3 + ee, volatile=True).to(tl.float32)
            p = tl.exp(logit - tl.max(logit, axis=0))
            p = p / tl.sum(p, axis=0)
            kk = tl.arange(0, TOPKP)
            sel_w = tl.zeros([TOPKP], dtype=tl.float32)
            sel_i = tl.zeros([TOPKP], dtype=tl.int32)
            for j in tl.static_range(TOPK):
                mx = tl.max(p, axis=0)
                i = tl.min(tl.where(p == mx, ee, E), axis=0)
                sel_w = tl.where(kk == j, mx, sel_w)
                sel_i = tl.where(kk == j, i, sel_i)
                p = tl.where(ee == i, -1.0, p)
            sel_w = sel_w / tl.sum(sel_w, axis=0)
            sel_w = sel_w.to(tl.bfloat16).to(tl.float32)
            ok = kk < TOPK
            tl.store(idx_ptr + r * TOPK + kk, sel_i, mask=ok)
            tl.store(wt_ptr + r * TOPK + kk, sel_w, mask=ok)
    if PHASE == 0:
        _grid_barrier(bar_ptr, base + 3 * per_barrier, err_ptr, SPIN_VOLATILE)

    # (4) routed experts' gate_up: work items (pair, row tile), act = silu(gate) * up.
    if (PHASE == 0) | (PHASE == 4):
        ntile = tl.cdiv(I, BNE)
        for t in range(pid, B * TOPK * ntile, nprog):
            pair = t // ntile
            tile = t % ntile
            tok = pair // TOPK
            expert = tl.load(idx_ptr + pair, volatile=True).to(tl.int64)
            rows = tile * BNE + tl.arange(0, BNE)
            wbase = gu_ptr + expert * stride_gue
            acc_g = tl.zeros([BNE], dtype=tl.float32)
            acc_u = tl.zeros([BNE], dtype=tl.float32)
            for k0 in range(0, H, BK):
                cols = k0 + tl.arange(0, BK)
                xv = tl.load(h2_ptr + tok * H + cols).to(tl.float32)
                wg = tl.load(wbase + rows[:, None] * H + cols[None, :]).to(tl.float32)
                wu = tl.load(wbase + (rows[:, None] + I) * H + cols[None, :]).to(tl.float32)
                acc_g += tl.sum(wg * xv[None, :], axis=1)
                acc_u += tl.sum(wu * xv[None, :], axis=1)
            g = acc_g.to(tl.bfloat16).to(tl.float32)
            u = acc_u.to(tl.bfloat16).to(tl.float32)
            a = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
            tl.store(act_ptr + pair * I + rows, (a * u).to(tl.bfloat16))
    if PHASE == 0:
        _grid_barrier(bar_ptr, base + 4 * per_barrier, err_ptr, SPIN_VOLATILE)

    # (5) routed down + weighted combine, shared expert down, sigmoid gate, sum.
    if (PHASE == 0) | (PHASE == 5):
        ntile = tl.cdiv(H, BNE)
        for t in range(pid, B * ntile, nprog):
            tok = t // ntile
            rows = (t % ntile) * BNE + tl.arange(0, BNE)
            acc = tl.zeros([BNE], dtype=tl.float32)
            for j in tl.static_range(TOPK):
                pair = tok * TOPK + j
                expert = tl.load(idx_ptr + pair, volatile=True).to(tl.int64)
                weight = tl.load(wt_ptr + pair, volatile=True)
                wbase = dn_ptr + expert * stride_dne
                part = tl.zeros([BNE], dtype=tl.float32)
                for k0 in range(0, I, BKI):
                    cols = k0 + tl.arange(0, BKI)
                    hv = tl.load(act_ptr + pair * I + cols).to(tl.float32)
                    w = tl.load(wbase + rows[:, None] * I + cols[None, :]).to(tl.float32)
                    part += tl.sum(w * hv[None, :], axis=1)
                acc += (part.to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
            routed = acc.to(tl.bfloat16).to(tl.float32)
            sc = tl.arange(0, IS)
            g = tl.load(out3_ptr + tok * N3 + E + sc).to(tl.float32)
            u = tl.load(out3_ptr + tok * N3 + E + IS + sc).to(tl.float32)
            sa = ((g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16)
            sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :]).to(tl.float32)
            shared = tl.sum(sw * sa.to(tl.float32)[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
            sg = tl.load(out3_ptr + tok * N3 + E + 2 * IS, volatile=True).to(tl.float32)
            sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
            term = (sg * shared).to(tl.bfloat16).to(tl.float32)
            tl.store(out_ptr + tok * H + rows, (routed + term).to(tl.bfloat16))


class MegaProto:
    """Weights of one layer's prototype (concatenated copies of the small projections)."""

    def __init__(self, m, mlp, top_k: int, nprog: int, bn: int, bk: int, bne: int,
                 warps: int, spin_volatile: bool, per_barrier: int, bar: torch.Tensor,
                 err: torch.Tensor):
        self.w1 = torch.cat([m.in_proj_qkv.weight, m.in_proj_z.weight, m.in_proj_b.weight,
                             m.in_proj_a.weight]).contiguous()
        self.splits = [m.in_proj_qkv.weight.shape[0], m.in_proj_z.weight.shape[0],
                       m.in_proj_b.weight.shape[0], m.in_proj_a.weight.shape[0]]
        sh = mlp.shared_expert
        self.w3 = torch.cat([mlp.gate.weight, sh.gate_proj.weight, sh.up_proj.weight,
                             mlp.shared_expert_gate.weight]).contiguous()
        self.gu = mlp.experts.gate_up_proj
        self.dn = mlp.experts.down_proj
        self.sd = sh.down_proj.weight.contiguous()
        self.E = mlp.gate.weight.shape[0]
        self.IS = sh.gate_proj.weight.shape[0]
        self.I = self.gu.shape[1] // 2
        self.H = self.w1.shape[1]
        assert self.gu.stride(1) == self.H and self.dn.stride(1) == self.I
        self.top_k = top_k
        self.nprog, self.bn, self.bk, self.bne, self.warps = nprog, bn, bk, bne, warps
        self.spin_volatile = spin_volatile
        self.per_barrier = per_barrier
        self.bar, self.err = bar, err
        self._bufs: dict[int, tuple] = {}

    def bufs(self, B: int, dev):
        if B not in self._bufs:
            bf = torch.bfloat16
            self._bufs[B] = (
                torch.empty((B, self.w1.shape[0]), dtype=bf, device=dev),
                torch.empty((B, self.w3.shape[0]), dtype=bf, device=dev),
                torch.empty((B, self.top_k), dtype=torch.int32, device=dev),
                torch.empty((B, self.top_k), dtype=torch.float32, device=dev),
                torch.empty((B * self.top_k, self.I), dtype=bf, device=dev),
                torch.empty((B, self.H), dtype=bf, device=dev),
            )
        return self._bufs[B]

    def run(self, h1: torch.Tensor, h2: torch.Tensor, phase: int = 0):
        """phase 0: one persistent launch; -1: five launches (phases 1..5, fusion only)."""
        B = h1.shape[0]
        out1, out3, idx, wt, act, out = self.bufs(B, h1.device)
        phases = [0] if phase == 0 else [1, 2, 3, 4, 5]
        for p in phases:
            _mega_proto[(self.nprog,)](
                h1, self.w1, out1, h2, self.w3, out3, idx, wt, self.gu, act, self.dn, self.sd,
                out, self.bar, self.err, self.per_barrier, self.gu.stride(0), self.dn.stride(0),
                B=B, BP=max(2, triton.next_power_of_2(B)), H=self.H, N1=self.w1.shape[0],
                N3=self.w3.shape[0], E=self.E, TOPK=self.top_k,
                TOPKP=triton.next_power_of_2(self.top_k), I=self.I, IS=self.IS, BN=self.bn,
                BK=self.bk, BKI=min(self.bk, self.I), BNE=self.bne, PHASE=p, SPIN_VOLATILE=self.spin_volatile,
                num_warps=self.warps,
            )
        return out1, idx, wt, out

    def bytes(self, idx: torch.Tensor) -> int:
        unique = len(set(idx.flatten().tolist()))
        per_expert = self.gu[0].numel() * 2 + self.dn[0].numel() * 2
        return (self.w1.numel() + self.w3.numel() + self.sd.numel()) * 2 + unique * per_expert


def reference_moe(mlp, x, top_k):
    """The fast path's ``_moe`` math with the PyTorch reference experts (CPU tests)."""
    from engine.fast import kernels

    shared = mlp.shared_expert
    shared_out = F.linear(F.silu(F.linear(x, shared.gate_proj.weight))
                          * F.linear(x, shared.up_proj.weight), shared.down_proj.weight)
    router = F.linear(x, mlp.gate.weight)
    probs = torch.softmax(router, dim=-1, dtype=torch.float)
    top_w, top_i = torch.topk(probs, top_k, dim=-1)
    top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(router.dtype)
    routed = kernels.moe_decode_reference(x, top_i, top_w, mlp.experts.gate_up_proj,
                                          mlp.experts.down_proj)
    return routed + torch.sigmoid(F.linear(x, mlp.shared_expert_gate.weight)) * shared_out, top_i


# --------------------------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------------------------


def _sync():
    torch.cuda.synchronize()


def graph_of(fn, reps: int = 1) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    _sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    _sync()
    return g


def time_ms(fn, n: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    _sync()
    return a.elapsed_time(b) / n


def graph_ms(fn, reps: int = 10, n: int = 30) -> float:
    """ms per call of ``fn`` inside a graph of ``reps`` back-to-back calls."""
    g = graph_of(fn, reps)
    return time_ms(g.replay, n) / reps


# --------------------------------------------------------------------------------------------
# (a) barrier
# --------------------------------------------------------------------------------------------


def calibrate(dev, nprog: int) -> int:
    """Counter increment of one arrival by every CTA (1 per CTA when a scalar atomic runs once)."""
    c = torch.zeros(1, dtype=torch.int32, device=dev)
    _arrive_once[(nprog,)](c, num_warps=4)
    _sync()
    total = int(c.item())
    assert total % nprog == 0, (total, nprog)
    return total


def part_a(dev, ncu: int) -> dict:
    out = {}
    for mult in (1, 2):
        nprog = ncu * mult
        per_barrier = calibrate(dev, nprog)
        print(f"barrier calib grid={nprog} counter_per_barrier={per_barrier}")
        # The volatile-load spin timed out in j63 even on a fresh counter (the scalar load
        # likely never re-reads L2); only the atomic spin is measured.
        for vol in (False,):
            bar = torch.zeros(1, dtype=torch.int32, device=dev)
            err = torch.zeros(1, dtype=torch.int32, device=dev)
            bad = torch.zeros(1, dtype=torch.int32, device=dev)
            buf = torch.zeros(2 * nprog, dtype=torch.int32, device=dev)

            def launch(iters):
                # A fresh counter per launch: the epoch base assumes every earlier launch ran
                # the same number of barriers (j63 mixed 0, 10 and 2000 on one counter, so
                # its barriers waited for a target already passed: invalid, 75% mismatches).
                bar.zero_()
                _barrier_bench[(nprog,)](bar, err, buf, bad, iters, per_barrier,
                                         SPIN_VOLATILE=vol, num_warps=4)

            launch(10)  # residency probe: a CTA that never runs times its barriers out
            _sync()
            if int(err.item()):
                print(f"RESULT barrier grid={nprog} vol={vol} not_resident timeouts={int(err.item())}")
                continue
            iters = 2000
            t0 = time_ms(lambda: launch(0), 20)
            t1 = time_ms(lambda: launch(iters), 10, 2)
            us = (t1 - t0) * 1e3 / iters
            label = f"grid={nprog} spin={'volatile' if vol else 'atomic'}"
            print(f"RESULT barrier {label} us_per_barrier={us:.2f} launch_ms_0iters={t0:.4f} "
                  f"timeouts={int(err.item())} neighbour_mismatches={int(bad.item())}")
            if int(err.item()) == 0 and int(bad.item()) == 0:
                out[(mult, vol)] = us
    # Per-node cost of a graph of tiny kernels.
    y = torch.zeros(4 * ncu, device=dev)
    for grid in (1, ncu):
        ms = graph_ms(lambda grid=grid: _tiny[(grid,)](y, num_warps=4), reps=200, n=20)
        print(f"RESULT graph_node_us tiny_kernel grid={grid} us_per_node={ms * 1e3:.2f}")
    return out


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def load_fast(layers: int):
    from transformers import AutoConfig

    from engine.fast.model import FastModel
    from engine.generator import CHECKPOINT, Generator

    model, revision = CHECKPOINT["model"], CHECKPOINT["revision"]
    config = AutoConfig.from_pretrained(model, revision=revision)
    for cfg in {id(c): c for c in (config, config.get_text_config())}.values():
        if hasattr(cfg, "layer_types"):
            cfg.layer_types = list(cfg.layer_types)[:layers]
            cfg.num_hidden_layers = layers
        if hasattr(cfg, "mtp_num_hidden_layers"):
            cfg.mtp_num_hidden_layers = 0
    t0 = time.perf_counter()
    gen = Generator.load(model, revision, config=config, max_model_len=8192, chunk_tokens=2048,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = FastModel(gen.model, slots=8, max_len=8192, kv_tokens=16384, use_graphs=False)
    print(f"RESULT load_s={time.perf_counter() - t0:.1f} layer_types={fast.layer_types}")
    return gen, fast


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--configs",
                    default="16x512x16x4x1,8x512x8x8x1,8x512x8x8x2,8x1024x8x4x2,32x256x16x8x1",
                    help="BN x BK x BNE x warps x CTAs-per-CU of the prototype")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    ncu = props.multi_processor_count
    print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__,
          "gpu", props.name, getattr(props, "gcnArchName", "?"), "CUs", ncu,
          "mem_GB", round(props.total_memory / 1e9, 1))
    verdict = {}

    # ---------------------------------------------------------------- (a)
    try:
        bars = part_a(dev, ncu)
        if bars:
            best1 = min((v for (mult, _), v in bars.items() if mult == 1), default=None)
            verdict["barrier_us"] = best1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("RESULT barrier error")

    torch.set_grad_enabled(False)
    gen, fast = load_fast(args.layers)
    from engine.fast import kernels

    li = fast.layer_types.index("linear_attention")
    ai = fast.layer_types.index("full_attention") if "full_attention" in fast.layer_types else None
    layer = fast.layers[li]
    m, mlp = layer.linear_attn, layer.mlp
    eps = fast.eps
    meta = fast.devices[layer.input_layernorm.weight.device]

    def inputs(B: int):
        """Real-ish activations: embeddings of fixed tokens through layer li's own ops."""
        ids = torch.tensor([100, 2000, 31337, 777, 4242, 9001, 123, 50000][:B], device=dev)
        x = F.embedding(ids, fast.embed).to(dev)
        _, h1 = kernels.add_rms(x, None, layer.input_layernorm.weight, eps)
        slots = torch.arange(B, device=dev)
        h = fast._gdn_decode(li, h1, slots)
        _, h2 = kernels.add_rms(x, h, layer.post_attention_layernorm.weight, eps)
        return x, h1.contiguous(), h2.contiguous(), slots

    err = torch.zeros(1, dtype=torch.int32, device=dev)
    batches = [int(b) for b in args.batches.split(",")]

    # ---------------------------------------------------------------- (b)
    speedups = {}
    for cfg in args.configs.split(","):
        bn, bk, bne, warps, per_cu = (int(v) for v in cfg.split("x"))
        try:
            nprog = ncu * per_cu
            proto = MegaProto(m, mlp, fast.top_k, nprog, bn, bk, bne, warps, False,
                              calibrate(dev, nprog),
                              torch.zeros(1, dtype=torch.int32, device=dev), err)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            continue
        for B in batches:
            try:
                x, h1, h2, slots = inputs(B)

                def base():
                    p = fast._gdn_project(m, h1)
                    return p, fast._moe(mlp, h2, decode=True)

                (qkv, z, b, a), moe_ref = base()
                ref_cat = torch.cat([qkv, z, b, a], dim=-1)
                router = F.linear(h2, mlp.gate.weight)
                ref_idx = torch.topk(torch.softmax(router, -1, dtype=torch.float), fast.top_k)[1]
                errs0 = int(err.item())
                out1, idx, wt, out = proto.run(h1, h2)
                _sync()
                if int(err.item()) != errs0:
                    print(f"RESULT proto cfg={cfg} B={B} not_resident (barrier timed out); skipped")
                    break
                d1 = float((out1.float() - ref_cat.float()).abs().max())
                same_sets = all(sorted(idx[r].tolist()) == sorted(ref_idx[r].tolist())
                                for r in range(B))
                dm = float((out.float() - moe_ref.float()).abs().max())
                scale = float(moe_ref.float().abs().max())
                first = out.clone()
                repeat_ok = True
                for _ in range(20):
                    proto.run(h1, h2)
                    repeat_ok &= bool(torch.equal(proto.bufs(B, dev)[5], first))
                out1m, idxm, _, outm = proto.run(h1, h2, phase=-1)
                _sync()
                dmulti = float((outm.float() - out.float()).abs().max())
                timeouts = int(err.item()) - errs0
                t_base_graph = graph_ms(base)
                t_base_eager = time_ms(base, 20)
                t_mega = graph_ms(lambda: proto.run(h1, h2))
                t_multi = graph_ms(lambda: proto.run(h1, h2, phase=-1))
                gb = proto.bytes(idx) / 1e9
                sp = t_base_graph / t_mega
                print(
                    f"RESULT proto cfg={cfg} B={B} base_graph_us={t_base_graph * 1e3:.1f} "
                    f"base_eager_us={t_base_eager * 1e3:.1f} mega_us={t_mega * 1e3:.1f} "
                    f"five_launch_us={t_multi * 1e3:.1f} speedup_vs_base_graph={sp:.2f}x "
                    f"fusion_only_speedup={t_base_graph / t_multi:.2f}x bytes_MB={gb * 1e3:.1f} "
                    f"mega_GBps={gb / (t_mega * 1e-3):.0f} base_GBps={gb / (t_base_graph * 1e-3):.0f}"
                )
                print(
                    f"RESULT parity cfg={cfg} B={B} inproj_max_abs={d1:.4f} topk_sets_equal={same_sets} "
                    f"moe_max_abs={dm:.4f} moe_ref_absmax={scale:.3f} repeat_identical={repeat_ok} "
                    f"five_launch_vs_mega_max_abs={dmulti:.4f} barrier_timeouts={timeouts}"
                )
                # The current path's own Triton-vs-PyTorch spread for the same MoE output.
                fast.triton_kernels["moe"] = False
                try:
                    moe_torch = fast._moe(mlp, h2, decode=True)
                finally:
                    fast.triton_kernels["moe"] = True
                spread = float((moe_ref.float() - moe_torch.float()).abs().max())
                tol = max(2 * spread, 0.03 * scale)
                print(f"RESULT parity_tol cfg={cfg} B={B} base_triton_vs_torch_max_abs={spread:.4f} "
                      f"tol={tol:.4f} within={dm <= tol}")
                ok = same_sets and repeat_ok and timeouts == 0 and dm <= tol
                if ok:
                    speedups.setdefault(B, []).append((sp, cfg, t_mega, t_base_graph, t_multi))
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                print(f"RESULT proto cfg={cfg} B={B} error")
    for B, rows in sorted(speedups.items()):
        sp, cfg, tm, tb, t5 = max(rows)
        print(f"RESULT proto_best B={B} cfg={cfg} mega_us={tm * 1e3:.1f} base_graph_us={tb * 1e3:.1f} "
              f"speedup={sp:.2f}x five_launch_us={t5 * 1e3:.1f}")
    if 2 in speedups:
        verdict["proto_speedup_b2"] = max(speedups[2])[0]

    # ---------------------------------------------------------------- (c)
    B = 2
    x, h1, h2, slots = inputs(B)
    positions = torch.full((B,), 1000, dtype=torch.long, device=dev)
    meta.packed[1, :B] = slots
    meta.packed[2, :B] = positions
    try:
        def gdn_layer():
            return fast.layer_decode(li, x, slots, positions, meta)

        t_layer_graph = graph_ms(gdn_layer)
        t_layer_eager = time_ms(gdn_layer, 20)
        print(f"RESULT layer_ms kind=gdn B={B} graph={t_layer_graph:.4f} eager={t_layer_eager:.4f}")
        verdict["layer_graph_ms"] = t_layer_graph
        if ai is not None:
            for s in range(B):
                fast._ensure_pages(s, 1001)

            def attn_layer():
                return fast.layer_decode(ai, x, slots, positions, meta)

            print(f"RESULT layer_ms kind=attn B={B} graph={graph_ms(attn_layer):.4f} "
                  f"eager={time_ms(attn_layer, 20):.4f}")

        # Each op alone (graph of 20 repeats): us and GB/s.
        conv_state, rec_state = fast.gdn_state[li]
        qkv, z, b, a = fast._gdn_project(m, h1)
        mixed = kernels.gdn_conv_(conv_state, slots, qkv, m.conv1d.weight)
        core = kernels.gdn_fused_(rec_state, slots, mixed, a, b, z, m)
        sh = mlp.shared_expert
        router = F.linear(h2, mlp.gate.weight)
        probs = torch.softmax(router, dim=-1, dtype=torch.float)
        top_w, top_i = torch.topk(probs, fast.top_k, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(router.dtype)
        uniq = len(set(top_i.flatten().tolist()))
        ex = mlp.experts
        per_expert = (ex.gate_up_proj[0].numel() + ex.down_proj[0].numel()) * 2

        def nb(t):
            return t.numel() * t.element_size()

        def topk_chain():
            r = F.linear(h2, mlp.gate.weight)
            p = torch.softmax(r, dim=-1, dtype=torch.float)
            w, i = torch.topk(p, fast.top_k, dim=-1)
            return (w / w.sum(dim=-1, keepdim=True)).to(r.dtype), i

        sg_out, su_out = F.linear(h2, sh.gate_proj.weight), F.linear(h2, sh.up_proj.weight)
        shared_act = F.silu(sg_out) * su_out
        rec_bytes = rec_state[0].numel() * 4 * 2 * B
        ops = [
            ("add_rms", lambda: kernels.add_rms(x, h1, layer.post_attention_layernorm.weight, eps), 0),
            ("in_proj_qkv", lambda: F.linear(h1, m.in_proj_qkv.weight), nb(m.in_proj_qkv.weight)),
            ("in_proj_z", lambda: F.linear(h1, m.in_proj_z.weight), nb(m.in_proj_z.weight)),
            ("in_proj_b", lambda: F.linear(h1, m.in_proj_b.weight), nb(m.in_proj_b.weight)),
            ("in_proj_a", lambda: F.linear(h1, m.in_proj_a.weight), nb(m.in_proj_a.weight)),
            ("gdn_conv", lambda: kernels.gdn_conv_(conv_state, slots, qkv, m.conv1d.weight), 0),
            ("gdn_fused", lambda: kernels.gdn_fused_(rec_state, slots, mixed, a, b, z, m), rec_bytes),
            ("out_proj", lambda: F.linear(core, m.out_proj.weight), nb(m.out_proj.weight)),
            ("router+softmax+topk+renorm", topk_chain, nb(mlp.gate.weight)),
            ("shared_gate", lambda: F.linear(h2, sh.gate_proj.weight), nb(sh.gate_proj.weight)),
            ("shared_up", lambda: F.linear(h2, sh.up_proj.weight), nb(sh.up_proj.weight)),
            ("shared_silu_mul", lambda: F.silu(sg_out) * su_out, 0),
            ("shared_down", lambda: F.linear(shared_act, sh.down_proj.weight), nb(sh.down_proj.weight)),
            ("shared_expert_gate+sigmoid", lambda: torch.sigmoid(F.linear(h2, mlp.shared_expert_gate.weight)),
             nb(mlp.shared_expert_gate.weight)),
            ("moe_decode", lambda: kernels.moe_decode(h2, top_i, top_w, ex.gate_up_proj, ex.down_proj,
                                                       use_triton=True), uniq * per_expert),
            ("residual_add", lambda: x + h1, 0),
        ]
        total_us, total_bytes = 0.0, 0
        for name, fn, nbytes in ops:
            us = graph_ms(fn, reps=20) * 1e3
            total_us += us
            total_bytes += nbytes
            gbps = nbytes / (us * 1e-6) / 1e9 if nbytes else 0.0
            print(f"RESULT op B={B} {name} us={us:.1f} MB={nbytes / 1e6:.2f} GBps={gbps:.0f}")
        print(f"RESULT op_sum B={B} us={total_us:.1f} MB={total_bytes / 1e6:.1f} "
              f"GBps={total_bytes / (total_us * 1e-6) / 1e9:.0f} layer_graph_us={t_layer_graph * 1e3:.1f} "
              f"layer_floor_GBps={total_bytes / (t_layer_graph * 1e-3) / 1e9:.0f}")
        verdict["layer_GBps"] = total_bytes / (t_layer_graph * 1e-3) / 1e9

        # torch.profiler: kernels of the eager layer and of graph replays.
        from torch.profiler import ProfilerActivity, profile

        g = graph_of(gdn_layer, reps=10)
        for label, fn, calls in (("eager", gdn_layer, 20), ("graph", g.replay, 2)):
            for _ in range(3):
                fn()
            _sync()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t0 = time.perf_counter()
                for _ in range(calls):
                    fn()
                _sync()
                wall = (time.perf_counter() - t0) * 1e3
            layers_run = calls * (1 if label == "eager" else 10)
            kev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            busy = sum(e.time_range.elapsed_us() for e in kev)
            spans = [(e.time_range.start, e.time_range.end) for e in kev]
            span = (max(s[1] for s in spans) - min(s[0] for s in spans)) if spans else 0
            per_name: dict[str, list[float]] = {}
            for e in kev:
                per_name.setdefault(e.name, []).append(e.time_range.elapsed_us())
            print(f"RESULT profile mode={label} layers={layers_run} kernels_per_layer="
                  f"{len(kev) / layers_run:.1f} busy_us_per_layer={busy / layers_run:.1f} "
                  f"device_span_us_per_layer={span / layers_run:.1f} "
                  f"busy_frac_of_span={busy / max(span, 1e-9):.2f} host_wall_ms={wall:.2f}")
            top = sorted(per_name.items(), key=lambda kv: -sum(kv[1]))[:25]
            for name, ts in top:
                print(f"  kernel[{label}] n/layer={len(ts) / layers_run:.1f} "
                      f"us/layer={sum(ts) / layers_run:.1f} avg_us={sum(ts) / len(ts):.1f} {name[:110]}")
            if label == "graph" and kev:
                verdict["busy_frac"] = busy / max(span, 1e-9)
                verdict["busy_us_per_layer"] = busy / layers_run
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("RESULT part_c error")

    # ---------------------------------------------------------------- verdict
    print("RESULT verdict_inputs", {k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in verdict.items()})
    kills = []
    if verdict.get("barrier_us") is None or verdict["barrier_us"] > 15:
        kills.append("barrier > 15 us or failed")
    if verdict.get("proto_speedup_b2", 0) < 1.8:
        kills.append("prototype < 1.8x at B=2 (or failed parity)")
    narrow = (verdict.get("busy_frac", 0) >= 0.7 and verdict.get("layer_GBps", 0) >= 700)
    print(f"RESULT verdict kill={kills or 'none'} narrow_to_fusion_only={narrow}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
