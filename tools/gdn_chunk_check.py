"""Plan 40 part B: the fused GDN chunk chain (``engine.fast.gdn_chunk``) against fla's, on one GCD.

    python tools/gdn_chunk_check.py [--sizes 64,576,1792] [--counts 61,561,1650] [--reps 50]

Per TP2 rank of Qwen3.6-35B-A3B (H = 8 key heads, HV = 16 value heads, K = V = 128), with the
checkpoint's A_log and dt_bias of one GDN layer when the weights load (else synthetic), from a
convolved projection ``mixed`` [size, 4096] (silu of normal draws) and the raw ``a``, ``b``
columns, with ``count`` real rows (the rest padding, as a captured bucket) and a random fp32
initial state:

- accuracy: core output (real rows) and final state of the fused chain and of fla's chain as
  ``FastModel._pg_gdn`` runs it, each against the token-by-token fp32 rule; and fused vs fla
  directly (max abs, max abs / max |ref|);
- repeatability: the fused chain twice on the same inputs, bitwise;
- time per layer (CUDA events, median of ``--reps``) of each chain from ``mixed`` to the core
  output and the new state, as the graph runs them; and the fused chain's launch options swept
  (ENGINE_GDN_CHUNK_BV x ENGINE_GDN_CHUNK_WARPS).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine.fast import gdn_chunk, prefill_kernels  # noqa: E402

H, HV, KD = 8, 16, 128


def result(kind, **kw):
    print("RESULT " + json.dumps({"kind": kind, **kw}), flush=True)


def gates():
    """(A_log, dt_bias) of a GDN layer of the checkpoint, rank 0's half; synthetic if absent."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        from engine.generator import CHECKPOINT

        index = json.load(open(hf_hub_download(CHECKPOINT["model"], "model.safetensors.index.json",
                                               revision=CHECKPOINT["revision"])))
        wm = index["weight_map"]
        key = next(k for k in wm if k.endswith("linear_attn.A_log"))
        dkey = key.replace("A_log", "dt_bias")
        out = []
        for k in (key, dkey):
            with safe_open(hf_hub_download(CHECKPOINT["model"], wm[k], revision=CHECKPOINT["revision"]), "pt") as f:
                out.append(f.get_tensor(k)[:HV].float())
        result("gates", source=key)
        return out
    except Exception as exc:  # noqa: BLE001
        result("gates", source=f"synthetic ({type(exc).__name__}: {exc})")
        g = torch.Generator().manual_seed(0)
        return torch.randn(HV, generator=g) * 0.5 + 1.0, torch.randn(HV, generator=g)


def fla_chain(mixed, a, b, a_log, dt_bias, rec, grow, keep):
    """``FastModel._pg_gdn`` from the convolved projection to the core output and new state."""
    from engine import gdn

    tokens = mixed.shape[0]
    q, k, v = torch.split(mixed, [H * KD, H * KD, HV * KD], dim=-1)
    q = q.reshape(tokens, -1, KD)
    k = k.reshape(tokens, -1, KD)
    v = v.reshape(tokens, -1, KD)
    q = q.repeat_interleave(HV // H, dim=1)
    k = k.repeat_interleave(HV // H, dim=1)
    g, beta = prefill_kernels.gdn_gates(a, b, a_log, dt_bias)
    k = torch.where(keep[:, None, None], k, 0.0)
    v = torch.where(keep[:, None, None], v, 0.0)
    g = torch.where(keep[:, None], g, 0.0)
    beta = torch.where(keep[:, None], beta, 0.0)
    core, final = gdn.fla_chunk(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g=g.unsqueeze(0), beta=beta.unsqueeze(0),
        initial_state=rec.index_select(0, grow), output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    rec.index_copy_(0, grow, final.to(rec.dtype))
    return core[0]


def err(got, ref):
    got, ref = got.float(), ref.float()
    d = float((got - ref).abs().max())
    return {"max_abs": d, "rel": d / max(1e-12, float(ref.abs().max()))}


def time_ms(fn, reps):
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="64,576,1792,2048")
    p.add_argument("--counts", default="61,561,1650,2048")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--sweep", default="16:4,4;32:4,4;32:4,8;64:4,4;64:4,8;32:2,4;16:4,2")
    args = p.parse_args()
    from engine import gdn

    fla = gdn.fla_kernels()
    assert fla is not None, gdn.fla_error()
    from engine.fast import fla_pin

    result("fla_pins", pins=fla_pin.apply())
    dev = torch.device("cuda")
    a_log, dt_bias = (t.to(dev) for t in gates())
    sizes = [int(x) for x in args.sizes.split(",")]
    counts = [int(x) for x in args.counts.split(",")]
    for size, count in zip(sizes, counts):
        g = torch.Generator(device=dev).manual_seed(size)
        width = 2 * H * KD + HV * KD
        proj = torch.randn(size, width + 64 + 2 * HV, generator=g, device=dev).bfloat16()  # merged-GEMM-like rows
        mixed = torch.nn.functional.silu(torch.randn(size, width, generator=g, device=dev) * 1.5).bfloat16()
        a = proj[:, width + 64 : width + 64 + HV]
        b = proj[:, width + 64 + HV :]
        rec0 = torch.randn(4, HV, KD, KD, generator=g, device=dev) * 0.05
        grow = torch.tensor([2], device=dev)
        meta = torch.tensor([count, 0], device=dev)
        keep = torch.arange(size, device=dev) < count
        # fp32 token-by-token reference
        ref_o, ref_s = gdn_chunk.reference(mixed.cpu(), a.cpu(), b.cpu(), a_log.cpu(), dt_bias.cpu(), rec0[2].cpu(),
                                           H, count=count)
        rf = rec0.clone()
        fo = fla_chain(mixed, a, b, a_log, dt_bias, rf, grow, keep)
        rm = rec0.clone()
        mo = gdn_chunk.chunk(mixed, a, b, a_log, dt_bias, rm, H, row=grow, meta=meta)
        rm2 = rec0.clone()
        mo2 = gdn_chunk.chunk(mixed, a, b, a_log, dt_bias, rm2, H, row=grow, meta=meta)
        torch.cuda.synchronize()
        others_untouched = bool(torch.equal(rm[[0, 1, 3]], rec0[[0, 1, 3]]))
        result(
            "accuracy", size=size, count=count,
            fused_vs_ref={"core": err(mo[:count].cpu(), ref_o[:count]), "state": err(rm[2].cpu(), ref_s)},
            fla_vs_ref={"core": err(fo[:count].cpu(), ref_o[:count]), "state": err(rf[2].cpu(), ref_s)},
            fused_vs_fla={"core": err(mo[:count], fo[:count]), "state": err(rm[2], rf[2])},
            fused_repeat_bitwise=bool(torch.equal(mo, mo2) and torch.equal(rm, rm2)),
            padding_rows_zero=bool((mo[count:] == 0).all()),
            other_state_rows_untouched=others_untouched,
            finite=bool(torch.isfinite(mo).all() and torch.isfinite(rm).all()),
        )
        # time: each chain from ``mixed`` to core + state, as the graph runs it
        rf = rec0.clone()
        rm = rec0.clone()
        t_fla = time_ms(lambda: fla_chain(mixed, a, b, a_log, dt_bias, rf, grow, keep), args.reps)
        t_fused = time_ms(lambda: gdn_chunk.chunk(mixed, a, b, a_log, dt_bias, rm, H, row=grow, meta=meta), args.reps)
        sweep = {}
        for item in args.sweep.split(";"):
            bv, warps = item.split(":")
            os.environ["ENGINE_GDN_CHUNK_BV"], os.environ["ENGINE_GDN_CHUNK_WARPS"] = bv, warps
            sweep[item] = round(time_ms(lambda: gdn_chunk.chunk(mixed, a, b, a_log, dt_bias, rm, H, row=grow,
                                                                meta=meta), args.reps), 4)
        os.environ.pop("ENGINE_GDN_CHUNK_BV")
        os.environ.pop("ENGINE_GDN_CHUNK_WARPS")
        # per kernel: the fused chain's two launches (profiler)
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                gdn_chunk.chunk(mixed, a, b, a_log, dt_bias, rm, H, row=grow, meta=meta)
                fla_chain(mixed, a, b, a_log, dt_bias, rf, grow, keep)
            torch.cuda.synchronize()
        per = {}
        for e in prof.key_averages():
            if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0):
                us = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                if us:
                    per[e.key[:60]] = round(us / 5 / 1e3, 4)
        result("time", size=size, count=count, fla_chain_ms=round(t_fla, 4), fused_ms=round(t_fused, 4),
               ratio=round(t_fused / t_fla, 3), sweep_ms=sweep, per_kernel_ms_per_call=per)
    return 0


if __name__ == "__main__":
    sys.exit(main())
