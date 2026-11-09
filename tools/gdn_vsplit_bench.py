"""Plan 48 step 0: the chained verify's GDN recurrence split over value columns, vs today.

    python tools/gdn_vsplit_bench.py [--configs 2x4,1x4,4x4,2x2] [--replays 50]

One GCD, TP2 rank-0 shard dims of Qwen3.6-35B-A3B (16 v-heads, 8 k-heads, DK = DV = 128,
conv_dim 4096, projection row 6176), the real per-layer A_log, dt_bias, norm and conv1d
weights of all 30 GDN layers (rank-0 slices, read from the pinned checkpoint; random with real
scales if it cannot be read, said in the output). Ring of 4 states per slot, 8 slots + dummy,
as ``FastModel`` allocates it; state row of slot s, ring entry j: ``s + j * 9``.

1. Bitwise: 3 layers, states warmed by 20 steps of today's kernels from zero, then
   ``--replays`` steps with fresh random projections, each variant on its own copy of the
   states (carried across steps) vs today's kernels: ring states, conv rings, core output
   (and out_proj output for N2) compared every step.
2. Timing: per (slots n, rows per slot G), every variant's 30 GDN layers (own states and
   inputs per layer) in ONE captured graph, ms per 30 layers (median of 7 x 20 replays):
   A0  today's two kernels (conv chain, chain step)            [the claim's baseline]
   A   A0 + out_proj GEMV (``mfma.gemv``, 2048 x 2048 shard)
   B   vsplit BV=128 (hoisted inputs, fused conv, norm in kernel) + q/k ring kernel
   C   vsplit BV in {16, 32, 64} x warps + N1 (norm + q/k ring kernel)
   C*+out   best C + out_proj GEMV;  N2  best C's split + out_proj GEMV with the norm in its
   x load (+ q/k ring programs);  ceiling: state streaming at the split shape and at today's
   shape (read the slice, write G ring rows, no math);  empty: 30 empty launches.
Prints ``RESULT`` lines and the plan's kill rule at the end. A hint for the plan, not the
task's metric.

Step 0 (j229, MI210 gfx90a 104 CUs, real weights of 30 layers), ms per 30 layers at 2 slots x
4 rows: today 1.324; best v-split + N1 1.111 (C64w4; C32w4 with a plain-layout norm 1.024);
hoist-only BV=128 1.217; state-streaming ceiling 0.245 (split) / 0.317 (today's shape); 30
empty launches 0.053. 1 slot x 4 rows: 1.286 -> 0.877 (0.800 plain-layout norm). Norm folded
into out_proj (N2) saved <= 0.08 ms over today's kernels + out_proj. States not bitwise
(max rel 3e-7, reduction order over DK), conv rings bitwise. Kill (< 0.4 ms at 2 x 4).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine.fast import gdn_vsplit as gv  # noqa: E402
from engine.fast import mfma  # noqa: E402

RANKS = 2
SLOTS = 8
RING = 4


def log(msg: str) -> None:
    print(msg, flush=True)


def checkpoint_weights(dev):
    """Rank-0 slices of every GDN layer's A_log, dt_bias, norm.weight, conv1d.weight."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    ck = json.loads((ROOT / "engine" / "checkpoint.json").read_text())
    repo, rev = ck["model"], ck["revision"]
    cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=rev, local_files_only=True)).read_text())
    tc = cfg.get("text_config", cfg)
    index = Path(hf_hub_download(repo, "model.safetensors.index.json", revision=rev, local_files_only=True))
    wmap = json.loads(index.read_text())["weight_map"]
    pat = re.compile(r"^(.*layers\.(\d+)\.linear_attn\.)A_log$")
    prefixes = sorted(
        ((int(m.group(2)), m.group(1)) for k in wmap if "mtp" not in k and (m := pat.match(k))),
    )
    nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    key_dim, value_dim = nk * dk, nv * dv
    kd, vd = key_dim // RANKS, value_dim // RANKS
    channels = torch.cat([
        torch.arange(0, kd), torch.arange(key_dim, key_dim + kd),
        torch.arange(2 * key_dim, 2 * key_dim + vd),
    ])
    files: dict[str, object] = {}
    layers = []
    for _, pre in prefixes:
        def get(name):
            f = wmap[pre + name]
            if f not in files:
                files[f] = safe_open(hf_hub_download(repo, f, revision=rev, local_files_only=True), "pt")
            return files[f].get_tensor(pre + name)
        layers.append(dict(
            A_log=get("A_log")[: nv // RANKS].contiguous().to(dev),
            dt_bias=get("dt_bias")[: nv // RANKS].contiguous().to(dev),
            norm_w=get("norm.weight").contiguous().to(dev),
            conv_w=get("conv1d.weight").index_select(0, channels).contiguous().to(dev),
        ))
    dims = dict(num_k_heads=nk // RANKS, num_v_heads=nv // RANKS, head_k_dim=dk, head_v_dim=dv,
                key_dim=kd, value_dim=vd, eps=float(tc.get("rms_norm_eps", 1e-6)))
    return layers, dims, f"checkpoint {repo}@{rev[:8]} ({len(layers)} GDN layers)"


def random_weights(dev, count=30):
    dims = dict(num_k_heads=8, num_v_heads=16, head_k_dim=128, head_v_dim=128, key_dim=1024,
                value_dim=2048, eps=1e-6)
    g = torch.Generator().manual_seed(7)
    layers = []
    for _ in range(count):
        layers.append(dict(
            A_log=torch.log(torch.empty(16).uniform_(1, 16, generator=g)).to(dev),
            dt_bias=(torch.randn(16, generator=g) * 0.5 + 1).to(dev),
            norm_w=(1 + 0.1 * torch.randn(128, generator=g)).to(torch.bfloat16).to(dev),
            conv_w=(torch.randn(4096, 1, 4, generator=g) * 0.3).to(torch.bfloat16).to(dev),
        ))
    return layers, dims, "random (real scales)"


def make_layer(w, dims, dev, out_w):
    L = NS(**dims, **w)
    conv_dim = 2 * L.key_dim + L.value_dim
    L.conv = torch.zeros((RING * (SLOTS + 1), conv_dim, 4), dtype=torch.bfloat16, device=dev)
    L.rec = torch.zeros((RING * (SLOTS + 1), L.num_v_heads, L.head_k_dim, L.head_v_dim),
                        dtype=torch.float32, device=dev)
    L.out_w = out_w
    L.width = conv_dim + L.value_dim + 2 * L.num_v_heads
    return L


def ring_states(n, groups, dev):
    return torch.tensor([s + j * (SLOTS + 1) for j in range(groups) for s in range(n)],
                        dtype=torch.long, device=dev)


def rand_proj(rows, width, gen, dev):
    return (torch.randn((rows, width), generator=gen) * 1.0).to(torch.bfloat16).to(dev)


# ------------------------------------------------------------------ variants (one layer)
def run_variant(name, L, proj, states, n, G, bufs):
    """Run variant ``name`` on layer L; returns (core or None, out or None)."""
    rows = n * G
    core, obuf = bufs["core"][:rows], bufs["obuf"][:rows]
    if name == "A0":
        gv.today(L, proj, states, core, n, G)
        return core, None
    if name == "A":
        gv.today(L, proj, states, core, n, G)
        return core, mfma.gemv(core, L.out_w)
    if name == "B":
        gv.vsplit(L, proj, states, core, n, G, 128, 8, True)
        gv.norm_qk(L, proj, states, obuf, core, n, G, do_norm=False)
        return core, None
    m = re.fullmatch(r"C(\d+)w(\d+)(x?)(\+out)?", name)
    if m:
        bv, warps = int(m.group(1)), int(m.group(2))
        gv.vsplit(L, proj, states, obuf, n, G, bv, warps, False)
        gv.norm_qk(L, proj, states, obuf, core, n, G, do_norm=True, exact=not m.group(3))
        return core, (mfma.gemv(core, L.out_w) if m.group(4) else None)
    m = re.fullmatch(r"W(\d+)w(\d+)(\+out)?", name)
    if m:  # chunked (WY) form; BV = DV keeps the norm in the kernel
        bv, warps = int(m.group(1)), int(m.group(2))
        norm_in = bv == L.head_v_dim
        gv.wy(L, proj, states, core if norm_in else obuf, n, G, bv, warps, norm_in)
        gv.norm_qk(L, proj, states, obuf, core, n, G, do_norm=not norm_in, exact=False)
        return core, (mfma.gemv(core, L.out_w) if m.group(3) else None)
    m = re.fullmatch(r"N2_(\d+)w(\d+)", name)
    if m:
        gv.vsplit(L, proj, states, obuf, n, G, int(m.group(1)), int(m.group(2)), False)
        return None, gv.out_gnorm(L, obuf, proj, states, L.out_w, n, G)
    m = re.fullmatch(r"stream(\d+)w(\d+)", name)
    if m:
        gv.stream(L, states, n, G, int(m.group(1)), int(m.group(2)))
        return None, None
    if name == "empty":
        gv.empty(states)
        return None, None
    raise ValueError(name)


def bufs_for(dev, dims):
    hv = dims["num_v_heads"] * dims["head_v_dim"]
    return {"core": torch.empty((16, hv), dtype=torch.bfloat16, device=dev),
            "obuf": torch.empty((16, hv), dtype=torch.float32, device=dev)}


# ------------------------------------------------------------------ bitwise
def ulps(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """(elements differing, max bf16 ulp distance) of two bf16 tensors."""
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -32768 - ai, ai)  # monotone integer order of the bf16 values
    bi = torch.where(bi < 0, -32768 - bi, bi)
    d = (ai - bi).abs()
    return int((d > 0).sum()), int(d.max())


def ulps32(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max fp32 ulp distance of two fp32 tensors."""
    ai = a.contiguous().view(torch.int32).to(torch.int64)
    bi = b.contiguous().view(torch.int32).to(torch.int64)
    ai = torch.where(ai < 0, -2147483648 - ai, ai)
    bi = torch.where(bi < 0, -2147483648 - bi, bi)
    return int((ai - bi).abs().max())


def bitwise(names, layers_w, dims, dev, n, G, replays, out_ws):
    gen = torch.Generator().manual_seed(1000 + 10 * n + G)
    states = ring_states(n, G, dev)
    base = [make_layer(w, dims, dev, ow) for w, ow in zip(layers_w, out_ws, strict=True)]
    bufs = bufs_for(dev, dims)
    for _ in range(20):  # warm the states with today's kernels
        for L in base:
            gv.today(L, rand_proj(n * G, L.width, gen, dev), states, bufs["core"][: n * G], n, G)
    copies = {}
    for name in ["A"] + names:
        copies[name] = []
        for L in base:
            c = NS(**vars(L))
            c.conv, c.rec = L.conv.clone(), L.rec.clone()
            copies[name].append(c)
    vb = {name: bufs_for(dev, dims) for name in ["A"] + names}
    rows_idx = states
    stats = {name: dict(state_bw=0, conv_bw=0, core_bw=0, out_bw=0, steps=0, state_max=0.0,
                        state_rel=0.0, core_diff=0, core_ulp=0, out_diff=0, out_ulp=0, finite=True,
                        state_ulp=0, state_ulp_big=0)
             for name in names}
    for _ in range(replays):
        for li in range(len(base)):
            raw = rand_proj(n * G, base[li].width, gen, dev)
            ca, oa = run_variant("A", copies["A"][li], raw.clone(), states, n, G, vb["A"])
            ca = ca.clone()
            for name in names:
                L = copies[name][li]
                cv, ov = run_variant(name, L, raw.clone(), states, n, G, vb[name])
                st = stats[name]
                st["steps"] += 1
                ra, rv = copies["A"][li].rec[rows_idx], L.rec[rows_idx]
                st["finite"] &= bool(torch.isfinite(rv).all())
                st["state_bw"] += int(torch.equal(ra, rv))
                dmax = float((ra - rv).abs().max())
                st["state_max"] = max(st["state_max"], dmax)
                st["state_rel"] = max(st["state_rel"], dmax / max(float(ra.abs().max()), 1e-30))
                st["state_ulp"] = max(st["state_ulp"], ulps32(ra, rv))
                big = ra.abs() >= 1e-3 * float(ra.abs().max())  # ulps away from near-zero entries
                if bool(big.any()):
                    st["state_ulp_big"] = max(st["state_ulp_big"], ulps32(ra[big], rv[big]))
                st["conv_bw"] += int(torch.equal(copies["A"][li].conv[rows_idx], L.conv[rows_idx]))
                if cv is not None:
                    st["core_bw"] += int(torch.equal(ca, cv))
                    nd, mu = ulps(ca, cv)
                    st["core_diff"] = max(st["core_diff"], nd)
                    st["core_ulp"] = max(st["core_ulp"], mu)
                if ov is not None:
                    st["out_bw"] += int(torch.equal(oa, ov))
                    nd, mu = ulps(oa, ov)
                    st["out_diff"] = max(st["out_diff"], nd)
                    st["out_ulp"] = max(st["out_ulp"], mu)
    torch.cuda.synchronize(dev)
    for name, st in stats.items():
        log(f"RESULT bitwise n={n} G={G} {name} steps={st['steps']} state_bitwise={st['state_bw']} "
            f"conv_bitwise={st['conv_bw']} core_bitwise={st['core_bw']} out_bitwise={st['out_bw']} "
            f"state_maxabs={st['state_max']:.3e} state_maxrel={st['state_rel']:.3e} "
            f"state_max_ulp={st['state_ulp']} state_max_ulp_above_1e-3max={st['state_ulp_big']} "
            f"core_max_diff_elems={st['core_diff']} core_max_ulp={st['core_ulp']} "
            f"out_max_diff_elems={st['out_diff']} out_max_ulp={st['out_ulp']} finite={st['finite']}")
    return stats


# ------------------------------------------------------------------ timing
def capture(dev, fn):
    stream = torch.cuda.Stream(dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream), torch.inference_mode():
        fn()
        fn()
    torch.cuda.current_stream(dev).wait_stream(stream)
    torch.cuda.synchronize(dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
        fn()
    torch.cuda.synchronize(dev)
    return graph


def time_graph(dev, graph, reps=20, trials=7) -> float:
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize(dev)
    out = []
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            graph.replay()
        e1.record()
        e1.synchronize()
        out.append(e0.elapsed_time(e1) / reps)
    return statistics.median(out)


def timing(names, layers, dev, n, G, dims):
    gen = torch.Generator().manual_seed(5 + n * 10 + G)
    states = ring_states(n, G, dev)
    projs = [rand_proj(n * G, L.width, gen, dev) for L in layers]
    bufs = [bufs_for(dev, dims) for _ in layers]
    res = {}
    for name in names:
        try:
            def fn(name=name):
                for L, p, b in zip(layers, projs, bufs, strict=True):
                    run_variant(name, L, p, states, n, G, b)
            graph = capture(dev, fn)
            res[name] = time_graph(dev, graph)
            del graph
            log(f"RESULT time n={n} G={G} {name} ms_per_30={res[name]:.4f}")
        except Exception:  # noqa: BLE001
            log(f"RESULT time n={n} G={G} {name} error:\n{traceback.format_exc()}")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slots", default="2,1,4")
    p.add_argument("--configs", default="2x4,1x4,4x4,2x2")
    p.add_argument("--groups", default="4,2")
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--skip-bitwise", action="store_true")
    args = p.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    log(f"RESULT device={props.name} arch={getattr(props, 'gcnArchName', '?')} "
        f"cus={props.multi_processor_count} torch={torch.__version__} "
        f"visible={os.environ.get('HIP_VISIBLE_DEVICES')}")
    try:
        ws, dims, src = checkpoint_weights(dev)
    except Exception:  # noqa: BLE001
        log(f"RESULT weights from checkpoint failed:\n{traceback.format_exc()}")
        ws, dims, src = random_weights(dev)
    while len(ws) < 30:
        ws = ws + ws
    ws = ws[:30]
    log(f"RESULT weights={src} dims={dims}")
    gout = torch.Generator().manual_seed(3)
    hv = dims["num_v_heads"] * dims["head_v_dim"]
    out_ws = [(torch.randn((2048, hv), generator=gout) * hv ** -0.5).to(torch.bfloat16).to(dev)
              for _ in range(30)]

    slots = [int(v) for v in args.slots.split(",")]
    groups = [int(v) for v in args.groups.split(",")]
    cvars = ["C32w4", "C64w4"]
    wvars = ["W128w8", "W128w4", "W64w4", "W64w8", "W32w4", "W32w2"]

    started = time.perf_counter()
    if not args.skip_bitwise:
        for n, G in ((2, 4), (1, 4), (4, 4), (2, 2)):
            try:
                bitwise(["C32w4x"] + wvars, ws[:3], dims, dev, n, G,
                        args.replays, out_ws[:3])
            except Exception:  # noqa: BLE001
                log(f"RESULT bitwise n={n} G={G} error:\n{traceback.format_exc()}")
        log(f"RESULT bitwise_done s={time.perf_counter() - started:.0f}")

    layers = [make_layer(w, dims, dev, ow) for w, ow in zip(ws, out_ws, strict=True)]
    summary = {}
    wsum = {}
    for n, G in ((int(c.split("x")[0]), int(c.split("x")[1])) for c in args.configs.split(",")):
        if True:
            for L in layers:  # warm states (realistic magnitudes) with today's kernels
                L.conv.zero_()
                L.rec.zero_()
            gen = torch.Generator().manual_seed(11)
            st = ring_states(n, G, dev)
            b = bufs_for(dev, dims)
            for _ in range(10):
                for L in layers:
                    gv.today(L, rand_proj(n * G, L.width, gen, dev), st, b["core"][: n * G], n, G)
            res = timing(["A0", "A"] + cvars + ["C32w4x"] + wvars + ["empty", "stream128w8", "stream64w4",
                                                                  "stream32w4"],
                         layers, dev, n, G, dims)
            ws_ = {k: v for k, v in res.items() if k in wvars}
            if ws_:
                wbest = min(ws_, key=ws_.get)
                res.update(timing([wbest + "+out"], layers, dev, n, G, dims))
                wsum[(n, G)] = (res, wbest)
            cs = {k: v for k, v in res.items() if k in cvars}
            if cs:
                best = min(cs, key=cs.get)
                bv, warps = re.fullmatch(r"C(\d+)w(\d+)", best).groups()
                more = timing([best + "+out", f"N2_{bv}w{warps}"], layers, dev, n, G, dims)
                res.update(more)
                summary[(n, G)] = (res, best, f"N2_{bv}w{warps}", f"stream{bv}w{warps}")
            torch.cuda.synchronize(dev)
    log(f"RESULT timing_done s={time.perf_counter() - started:.0f}")

    try:
        report_wy(wsum)
    except Exception:  # noqa: BLE001
        log(f"RESULT wy summary error:\n{traceback.format_exc()}")
    try:
        report(summary)
    except Exception:  # noqa: BLE001
        log(f"RESULT summary error:\n{traceback.format_exc()}")


def report_wy(wsum) -> None:
    for (n, G), (res, wbest) in wsum.items():
        a0, w = res["A0"], res[wbest]
        line = (f"RESULT wy n={n} G={G} A0_today_ms={a0:.4f} best_W={wbest} W_ms={w:.4f} "
                f"save_ms={a0 - w:.4f} ratio={w / a0:.3f} "
                f"ceiling_stream_today_shape_ms={res.get('stream128w8', float('nan')):.4f} "
                f"ceiling_stream64_ms={res.get('stream64w4', float('nan')):.4f}")
        if "A" in res and wbest + "+out" in res:
            line += f" A_with_out_ms={res['A']:.4f} W_out_ms={res[wbest + '+out']:.4f}"
        log(line)
    if (2, 4) in wsum:
        res, wbest = wsum[(2, 4)]
        w = res[wbest]
        log(f"RESULT wy_gate n=2 G=4 best={wbest} ms_per_30={w:.4f} (gate <= 0.6) -> "
            f"{'PASS' if w <= 0.6 else 'MISS'}")


def report(summary) -> None:
    for (n, G), (res, best, n2, ceil) in summary.items():
        a0, a = res.get("A0"), res.get("A")
        c = res.get(best)
        cfull = res.get(best + "+out")
        n2t = res.get(n2)
        line = (f"RESULT summary n={n} G={G} A0_today_ms={a0:.4f} best_C={best} C_ms={c:.4f} "
                f"save_vs_A0_ms={a0 - c:.4f} ratio={c / a0:.3f} B_ms={res.get('B', float('nan')):.4f}")
        if a is not None and cfull is not None:
            line += f" A_with_out_ms={a:.4f} C_out_ms={cfull:.4f} save_N1_full_ms={a - cfull:.4f}"
        if a is not None and n2t is not None:
            line += f" N2_ms={n2t:.4f} save_N2_full_ms={a - n2t:.4f}"
        line += (f" ceiling_{ceil}_ms={res.get(ceil, float('nan')):.4f}"
                 f" ceiling_today_shape_ms={res.get('stream128w8', float('nan')):.4f}"
                 f" empty30_ms={res.get('empty', float('nan')):.4f}")
        log(line)
    if (2, 4) in summary:
        res, best, n2, _ = summary[(2, 4)]
        a0 = res["A0"]
        saves = [a0 - res[best]]
        if "A" in res and n2 in res:
            saves.append(res["A"] - res[n2])
        save = max(saves)
        if a0 < 0.9:
            log(f"RESULT NOTE today's two kernels take {a0:.3f} ms < 0.9 ms per step at 2 slots x 4 rows:"
                " the room halves")
        eff = min(res[best], res[n2] - (res["A"] - a0) if "A" in res and n2 in res else 9e9)
        verdict = "PASS" if save >= 0.6 and eff <= 0.45 * a0 else (
            "KILL" if save < 0.4 else "GO (between kill and pass)")
        log(f"RESULT kill_rule n=2 G=4 best_save_ms={save:.4f} (kill < 0.4, pass >= 0.6 and <= 0.45x)"
            f" -> {verdict}")


if __name__ == "__main__":
    main()
