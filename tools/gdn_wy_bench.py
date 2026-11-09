"""Plan 54: the forms of the GDN prefill WY stage (``engine.fast.gdn_chunk.wy``) on one GCD.

    python tools/gdn_wy_bench.py [--sizes 64,576,1664] [--counts 61,561,1650] [--reps 20]

Per TP2 rank of Qwen3.6-35B-A3B (H = 8 key heads, HV = 16 value heads, K = V = 128), with a
checkpoint GDN layer's A_log, dt_bias and conv weight when they load (else synthetic), on a
convolved projection of ``size`` rows of which ``count`` are real (the rest padding, as a
captured prefill bucket), for each form:

- time per call: one CUDA graph of 30 back-to-back WY calls (a layer stack's worth, as the
  captured prefill pass replays them), replayed ``--reps`` times, median / 30;
- outputs (qn, kn, w, u, gc, aqk) against the fused kernel at 8 warps (today's default):
  bitwise per output, else the max abs difference over max |ref|; and the whole chain's core
  output and state through ``gdn_chunk.chunk`` the same way.

Forms: ``mono:W`` (the fused kernel at W warps), ``stack:W`` (fused, four diagonal
substitutions in one stacked chain), ``split:S:U:BW`` / ``split1:S:U:BW`` (solve kernel at S
warps, stacked / chained substitution, then the dot kernel at U warps over BW-column
programs). Prints one RESULT line per (size, form), then a ``best`` line: the fastest form at
the size whose count is 561, with the per-GCD saving over 30 layers, and the best ``stack``.
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
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from engine.fast import gdn_chunk, kernels  # noqa: E402
from engine.fast.prefill_kernels import gdn_conv_  # noqa: E402

H, HV, KD = 8, 16, 128
LAYERS = 30
BASE = "mono:8"


def result(kind, **kw):
    print("RESULT " + json.dumps({"kind": kind, **kw}), flush=True)


def parse(form: str):
    p = form.split(":")
    if p[0] in ("mono", "stack"):
        return (p[0], 0, 0, 0), int(p[1])
    return (p[0], int(p[1]), int(p[2]), int(p[3])), None


def forms() -> list[str]:
    out = [f"mono:{w}" for w in (8, 4, 2, 1)] + [f"stack:{w}" for w in (8, 4, 2, 1)]
    for s in (1, 2, 4):
        for u in (2, 4, 8):
            for bw in (64, 128):
                out.append(f"split:{s}:{u}:{bw}")
    out += [f"split1:{s}:4:64" for s in (1, 2, 4)]
    return out


def env_of(form: str) -> str:
    """The ENGINE_GDN_WY value that selects ``form`` in the engine."""
    return form


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="64,576,1664")
    p.add_argument("--counts", default="61,561,1650")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--forms", default=None, help="comma-separated (default: the sweep)")
    p.add_argument("--best-out", default=None, help="write the best form's ENGINE_GDN_WY value here")
    args = p.parse_args()
    from gdn_chunk_check import gates

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")  # cpu: TRITON_INTERPRET=1 smoke run, no timing
    a_log, dt_bias, cweight = (t.to(dev) for t in gates())
    sizes = [int(x) for x in args.sizes.split(",")]
    counts = [int(x) for x in args.counts.split(",")]
    todo = args.forms.split(",") if args.forms else forms()
    times: dict[int, dict[str, float]] = {}
    for size, count in zip(sizes, counts):
        g = torch.Generator(device=dev).manual_seed(size)
        sync = torch.cuda.synchronize if cuda else (lambda: None)
        width = 2 * H * KD + HV * KD
        proj = (torch.randn(size, width + 64 + 2 * HV, generator=g, device=dev) * 1.5).bfloat16()
        qkv = proj[:, :width]
        a = proj[:, width + 64 : width + 64 + HV]
        b = proj[:, width + 64 + HV :]
        conv0 = (torch.randn(width, 4, generator=g, device=dev) * 1.5).bfloat16()
        rec0 = torch.randn(4, HV, KD, KD, generator=g, device=dev) * 0.05
        meta = torch.tensor([count, 0], device=dev)
        row = torch.tensor([2], device=dev)
        mixed = gdn_conv_(qkv, conv0.clone(), cweight, meta=meta)

        def bufs():
            return (
                torch.zeros((size, H, KD), dtype=torch.bfloat16, device=dev),
                torch.zeros((size, H, KD), dtype=torch.bfloat16, device=dev),
                torch.zeros((size, HV, KD), dtype=torch.bfloat16, device=dev),
                torch.zeros((size, HV, KD), dtype=torch.bfloat16, device=dev),
                torch.zeros((size, HV), dtype=torch.float32, device=dev),
                torch.zeros((size, HV, gdn_chunk.BT), dtype=torch.bfloat16, device=dev),
            )

        def run_wy(form, out):
            variant, warps = parse(form)
            with kernels.on_device(mixed):
                gdn_chunk.wy(mixed, a, b, a_log, dt_bias, meta, H, *out, warps=warps, variant=variant)

        def run_chain(form):
            st = rec0.clone()
            cv = conv0.clone()
            os.environ["ENGINE_GDN_WY"] = env_of(form)
            try:
                o = gdn_chunk.chunk(qkv, cv, cweight, a, b, a_log, dt_bias, st, H, row=row, meta=meta)
            finally:
                os.environ.pop("ENGINE_GDN_WY", None)
            sync()
            return o, st

        ref = bufs()
        run_wy(BASE, ref)
        ref_o, ref_s = run_chain(BASE)
        sync()
        times[count] = {}
        for form in todo:
            try:
                out = bufs()
                run_wy(form, out)
                out2 = bufs()
                run_wy(form, out2)
                sync()
                cmp = {}
                for name, x, y in zip(("qn", "kn", "w", "u", "gc", "aqk"), out, ref):
                    # rows past T never written; all rows here are < T
                    if torch.equal(x, y):
                        cmp[name] = "bitwise"
                    else:
                        d = float((x.float() - y.float()).abs().max())
                        cmp[name] = round(d / max(1e-12, float(y.float().abs().max())), 6)
                repeat = all(torch.equal(x, y) for x, y in zip(out, out2))
                o, s = run_chain(form)
                chain = {
                    "core": "bitwise" if torch.equal(o, ref_o) else round(
                        float((o[:count].float() - ref_o[:count].float()).abs().max())
                        / max(1e-12, float(ref_o[:count].float().abs().max())), 6),
                    "state": "bitwise" if torch.equal(s, ref_s) else round(
                        float((s - ref_s).abs().max()) / max(1e-12, float(ref_s.abs().max())), 6),
                }
                # time: a graph of LAYERS back-to-back calls
                if not cuda:
                    times[count][form] = 1.0
                    result("form", size=size, count=count, form=form, vs_base=cmp, chain_vs_base=chain,
                           repeat_bitwise=repeat)
                    continue
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        run_wy(form, out)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(LAYERS):
                        run_wy(form, out)
                graph.replay()
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.reps):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graph.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1))
                del graph
                us = statistics.median(ts) / LAYERS * 1e3
                times[count][form] = us
                result("form", size=size, count=count, form=form, us_per_call=round(us, 2),
                       per_gcd_ms_30_layers=round(us * LAYERS / 1e3, 3), vs_base=cmp, chain_vs_base=chain,
                       repeat_bitwise=repeat)
            except Exception as exc:  # noqa: BLE001
                result("form_error", size=size, count=count, form=form, error=f"{type(exc).__name__}: {exc}"[:600])
    key = 561 if 561 in times else counts[len(counts) // 2]
    t = times[key]
    base = t.get(BASE)
    best = min(t, key=t.get)
    stacks = {f: v for f, v in t.items() if f.startswith("stack")}
    best_stack = min(stacks, key=stacks.get) if stacks else None
    summary = {}
    for c, tc in times.items():
        if BASE in tc and best in tc:
            summary[c] = {"base_ms_per_gcd": round(tc[BASE] * LAYERS / 1e3, 3),
                          "best_ms_per_gcd": round(tc[best] * LAYERS / 1e3, 3),
                          "saving_ms_per_gcd": round((tc[BASE] - tc[best]) * LAYERS / 1e3, 3)}
            if best_stack in tc:
                summary[c]["stack_saving_ms_per_gcd"] = round((tc[BASE] - tc[best_stack]) * LAYERS / 1e3, 3)
    result("best", at_count=key, best=best, engine_env=env_of(best), best_stack=best_stack, by_count=summary,
           ranking=sorted(((round(v, 1), f) for f, v in t.items()))[:10])
    if args.best_out:
        Path(args.best_out).write_text(env_of(best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
