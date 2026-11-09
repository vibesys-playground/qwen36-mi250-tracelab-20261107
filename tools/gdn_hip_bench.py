"""Plan 50 stage 0: the decode / chained-verify GDN step as a wavefront-resident HIP kernel
(``engine/fast/gdn_hip.py``, ``engine/fast/hip/gdn_chain.hip``) vs today's two Triton kernels.

    python tools/gdn_hip_bench.py [--configs 2x4,1x4,4x4,2x2,1x2] [--replays 30]

Plan 48's bench (``tools/gdn_vsplit_bench.py``) supplies the setup, so the numbers compare with
j229 / j231: one GCD, TP2 rank-0 dims of Qwen3.6-35B-A3B, the real A_log, dt_bias, norm and
conv1d weights of all 30 GDN layers (rank-0 slices of the pinned checkpoint), ring of 4 states
per slot, 8 slots + dummy; ring entry 0 is the slot's own state row (in place, as served).

1. compile: hipRTC (child process) per layout; VGPRs, spills, occupancy, LDS per kernel.
2. error vs float64: 3 layers, states warmed by 20 steps of today's kernels; ``--replays``
   steps of fresh random projections; each variant on its own state copy (carried across
   steps); every step each variant's output is compared with a float64 reference of the same
   step from that variant's own input state (same bf16 rounding points, fp64 everywhere else):
   state max |err| / max |ref| and rms, core max |err| / max |ref| and the share of core
   elements != bf16(ref). Conv ring rows bitwise vs today's.
3. graph equality: eager vs captured replay of 3 layers x 3 steps, bitwise (states, rings, core).
4. timing: each variant's 30 layers in ONE captured graph, ms per 30 layers (median of 7 x 20
   replays); ceilings ``stream128w8`` (today's launch shape) / ``stream32w4`` and 30 empty
   launches from plan 48's kernels.
5. per-phase time (s_memtime cycles; s_memrealtime ticks) of each layout / mode's chain kernel
   at n=2, G=4: prologue (state loads, conv, gates, norms), chain, epilogue; median and max over
   blocks (j244: ~67 cycles per realtime tick, i.e. a 25 MHz realtime clock at ~1.7 GHz).
Prints ``RESULT`` lines and the plan's pass / kill rule. A hint for the plan, not the metric.

j244 (563d158, MI210, real weights): best 0.760 ms per 30 layers at 2 x 4 (L4x4 sep) vs today
1.323; 1 x 4 0.712 vs 1.283. Chain ~5.7k cycles per row at 32 blocks; fused-conv prologue slow.
j247 (c666a1f, hoisted loads + column splits): 2 x 4 0.560 (L8x4_v2) vs 1.320 (pass bar 0.55);
1 x 4 0.435 (L16x4_v4) vs 1.284; 2 x 2 0.446 vs 0.768; 1 x 2 0.360 vs 0.748; 4 x 4 0.886 (L4x4
sep) vs 1.516; streaming ceiling 0.31 / 0.16-0.26 ms. Error vs fp64 0.95-1.02x today's (state and
core), conv rings bitwise, eager == graph replay bitwise for all 26 variants, no hang. Chain
1.5-2.4k cycles per row with splits; the prologue (~12-13k cycles, ~7.5 us) is now the largest
phase. Stage 0: GO (save 0.76 ms at 2 x 4; 2 x 4 misses 0.55 by 0.01 ms, 1 x 4 passes).
"""

from __future__ import annotations

import argparse
import faulthandler
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

import gdn_vsplit_bench as vb  # noqa: E402
from engine.fast import gdn_hip, hipmega_rtc  # noqa: E402
from engine.fast import gdn_vsplit as gv  # noqa: E402

log = vb.log
HIPS: dict[str, gdn_hip.GdnHip] = {}


def variants() -> list[str]:
    return [f"{lay}_{mode}" for lay in HIPS for mode in gdn_hip.modes(lay)]


def run(name, L, proj, states, core, n, G):
    if name == "A0":
        gv.today(L, proj, states, core, n, G)
        return
    if name == "empty":
        gv.empty(states)
        return
    if name.startswith("stream"):
        bv, w = name[6:].split("w")
        gv.stream(L, states, n, G, int(bv), int(w))
        return
    lay, mode = name.split("_")
    HIPS[lay].step(L, proj, states, core, n, G, mode)


# ------------------------------------------------------------------ float64 reference
def bf(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).double()


def ref_step(L, raw, conv0, rec0, n, G):
    """float64 step from conv state rows ``conv0`` [n, C, 4] and states ``rec0`` [n, H, DK, DV]:
    (states [G, n, H, DK, DV], core [G*n, H*DV] before its final bf16 rounding)."""
    C = 2 * L.key_dim + L.value_dim
    H, DK, DV, NK = L.num_v_heads, L.head_k_dim, L.head_v_dim, L.num_k_heads
    x = raw[:, :C].double().view(G, n, C)
    win = conv0[:, :, 1:].double()
    w = L.conv_w.reshape(C, 4).double()
    ys = []
    for j in range(G):
        full = torch.cat([win, x[j][..., None]], -1)
        y = bf((full * w).sum(-1))
        ys.append(bf(y * torch.sigmoid(y)))
        win = full[..., 1:]
    y = torch.stack(ys)
    q = y[..., : L.key_dim].reshape(G, n, NK, DK)
    k = y[..., L.key_dim: 2 * L.key_dim].reshape(G, n, NK, DK)
    v = y[..., 2 * L.key_dim:].reshape(G, n, H, DV)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(H // NK, dim=2)
    k = k.repeat_interleave(H // NK, dim=2)
    z_off = C
    b_off = z_off + L.value_dim
    a_off = b_off + H
    b = raw[:, b_off: b_off + H].double().view(G, n, H)
    a = raw[:, a_off: a_off + H].double().view(G, n, H) + L.dt_bias.double()
    beta = bf(torch.sigmoid(b))
    sp = torch.where(a > 20, a, torch.log1p(torch.exp(a)))
    dec = torch.exp(-torch.exp(L.A_log.double()) * sp)
    z = raw[:, z_off: z_off + L.value_dim].double().view(G, n, H, DV)
    nw = L.norm_w.double()
    S = rec0.double()
    states, outs = [], []
    for j in range(G):
        S = S * dec[j][..., None, None]
        kv = torch.einsum("nhk,nhkv->nhv", k[j], S)
        S = S + k[j][..., None] * ((v[j] - kv) * beta[j][..., None])[..., None, :]
        o = bf(torch.einsum("nhk,nhkv->nhv", q[j], S))
        nrm = bf(o / torch.sqrt((o * o).mean(-1, keepdim=True) + L.eps))
        yv = bf(nw * nrm)
        outs.append(yv * z[j] * torch.sigmoid(z[j]))
        states.append(S)
    return torch.stack(states), torch.stack(outs).reshape(G * n, H * DV)


def errors(names, layers_w, dims, dev, n, G, replays):
    gen = torch.Generator().manual_seed(2000 + 10 * n + G)
    states = vb.ring_states(n, G, dev)
    first = states[:n]
    base = [vb.make_layer(w, dims, dev, None) for w in layers_w]
    for L in base:
        gdn_hip.prepack(L)
    core0 = torch.empty((n * G, dims["num_v_heads"] * dims["head_v_dim"]), dtype=torch.bfloat16, device=dev)
    for _ in range(20):
        for L in base:
            gv.today(L, vb.rand_proj(n * G, L.width, gen, dev), states, core0, n, G)
    copies = {}
    for name in names:
        copies[name] = []
        for L in base:
            c = NS(**vars(L))
            c.conv, c.rec = L.conv.clone(), L.rec.clone()
            copies[name].append(c)
    st = {name: dict(se=0.0, sr=0.0, se2=0.0, sr2=0.0, ce=0.0, cr=0.0, cmis=0, celems=0, ring_bw=0,
                     core_bw_a0=0, steps=0, finite=True) for name in names}
    for _ in range(replays):
        for li in range(len(base)):
            raw = vb.rand_proj(n * G, base[li].width, gen, dev)
            ring_a0 = core_a0 = None
            for name in names:
                L = copies[name][li]
                conv0, rec0 = L.conv[first].clone(), L.rec[first].clone()
                core = torch.empty_like(core0)
                run(name, L, raw.clone(), states, core, n, G)
                ref_s, ref_o = ref_step(L, raw, conv0, rec0, n, G)
                ks = L.rec[states].double().view_as(ref_s)
                s = st[name]
                s["steps"] += 1
                s["finite"] &= bool(torch.isfinite(ks).all()) and bool(torch.isfinite(core.float()).all())
                e = (ks - ref_s).abs()
                s["se"] = max(s["se"], float(e.max()))
                s["sr"] = max(s["sr"], float(ref_s.abs().max()))
                s["se2"] += float((e * e).sum())
                s["sr2"] += float((ref_s * ref_s).sum())
                ce = (core.double() - ref_o).abs()
                s["ce"] = max(s["ce"], float(ce.max()))
                s["cr"] = max(s["cr"], float(ref_o.abs().max()))
                s["cmis"] += int((core != ref_o.to(torch.bfloat16)).sum())
                s["celems"] += core.numel()
                ring = L.conv[states].clone()
                if name == "A0":
                    ring_a0, core_a0 = ring, core.clone()
                else:
                    s["ring_bw"] += int(torch.equal(ring, ring_a0))
                    s["core_bw_a0"] += int(torch.equal(core, core_a0))
    torch.cuda.synchronize(dev)
    out = {}
    for name, s in st.items():
        out[name] = dict(state_maxrel=s["se"] / s["sr"], state_rmsrel=(s["se2"] / s["sr2"]) ** 0.5,
                         core_maxrel=s["ce"] / s["cr"], core_mis=s["cmis"] / s["celems"])
        log(f"RESULT error n={n} G={G} {name} steps={s['steps']} state_maxrel={out[name]['state_maxrel']:.3e} "
            f"state_rmsrel={out[name]['state_rmsrel']:.3e} core_maxrel={out[name]['core_maxrel']:.3e} "
            f"core_ne_bf16ref={out[name]['core_mis']:.4%} conv_ring_bitwise_vs_A0={s['ring_bw'] if name != 'A0' else '-'}"
            f"/{s['steps']} core_bitwise_vs_A0={s['core_bw_a0'] if name != 'A0' else '-'} finite={s['finite']}")
    a0 = out["A0"]
    for name in names:
        if name == "A0":
            continue
        o = out[name]
        r = {k: o[k] / max(a0[k], 1e-30) for k in o}
        ok = max(r["state_maxrel"], r["state_rmsrel"]) <= 2.0 and o["core_mis"] <= 2 * a0["core_mis"] + 1e-4 \
            and st[name]["ring_bw"] == st[name]["steps"] and st[name]["finite"]
        log(f"RESULT error_ratio n={n} G={G} {name} state_max={r['state_maxrel']:.2f}x "
            f"state_rms={r['state_rmsrel']:.2f}x core_max={r['core_maxrel']:.2f}x core_ne={r['core_mis']:.2f}x "
            f"-> {'OK' if ok else 'OVER_2X'}")
        out[name]["ok"] = ok
    return out


# ------------------------------------------------------------------ graph equality
def graph_equal(names, layers_w, dims, dev, n, G):
    gen = torch.Generator().manual_seed(77)
    states = vb.ring_states(n, G, dev)
    base = [vb.make_layer(w, dims, dev, None) for w in layers_w]
    for L in base:
        gdn_hip.prepack(L)
    core0 = torch.empty((n * G, dims["num_v_heads"] * dims["head_v_dim"]), dtype=torch.bfloat16, device=dev)
    for _ in range(10):
        for L in base:
            gv.today(L, vb.rand_proj(n * G, L.width, gen, dev), states, core0, n, G)
    raws = [vb.rand_proj(n * G, L.width, gen, dev) for L in base]
    res = {}
    for name in names:
        sets = []
        for _ in range(2):
            ls = []
            for L in base:
                c = NS(**vars(L))
                c.conv, c.rec = L.conv.clone(), L.rec.clone()
                ls.append(c)
            sets.append((ls, [r.clone() for r in raws], [torch.empty_like(core0) for _ in base]))

        def fn(s):
            for L, p, c in zip(*s, strict=True):
                run(name, L, p, states, c, n, G)
        eq = True
        try:
            # eager set 0; graph set 1 (the capture's warm-up runs eagerly on a scratch copy)
            scratch = []
            for L in sets[1][0]:
                c = NS(**vars(L))
                c.conv, c.rec = L.conv.clone(), L.rec.clone()
                scratch.append(c)
            sc = (scratch, [r.clone() for r in raws], [torch.empty_like(core0) for _ in base])
            stream = torch.cuda.Stream(dev)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                fn(sc)
            torch.cuda.current_stream(dev).wait_stream(stream)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fn(sets[1])
            for step in range(3):
                fn(sets[0])
                graph.replay()
                torch.cuda.synchronize(dev)
                for a, b in zip(sets[0][0], sets[1][0], strict=True):
                    eq &= torch.equal(a.rec, b.rec) and torch.equal(a.conv, b.conv)
                for a, b in zip(sets[0][2], sets[1][2], strict=True):
                    eq &= torch.equal(a, b)
            del graph
            res[name] = eq
            log(f"RESULT graph_equal n={n} G={G} {name} eager_vs_replay_bitwise={eq} (3 layers x 3 steps)")
        except Exception:  # noqa: BLE001
            res[name] = False
            log(f"RESULT graph_equal n={n} G={G} {name} error:\n{traceback.format_exc()}")
    return res


# ------------------------------------------------------------------ timing
def timing(names, layers, dev, n, G, dims):
    gen = torch.Generator().manual_seed(5 + n * 10 + G)
    states = vb.ring_states(n, G, dev)
    projs = [vb.rand_proj(n * G, L.width, gen, dev) for L in layers]
    hv = dims["num_v_heads"] * dims["head_v_dim"]
    cores = [torch.empty((n * G, hv), dtype=torch.bfloat16, device=dev) for _ in layers]
    res = {}
    for name in names:
        try:
            def fn(name=name):
                for L, p, c in zip(layers, projs, cores, strict=True):
                    run(name, L, p, states, c, n, G)
            graph = vb.capture(dev, fn)
            res[name] = vb.time_graph(dev, graph)
            del graph
            log(f"RESULT time n={n} G={G} {name} ms_per_30={res[name]:.4f}")
        except Exception:  # noqa: BLE001
            log(f"RESULT time n={n} G={G} {name} error:\n{traceback.format_exc()}")
    return res


def phases(layers, dev, dims, n=2, G=4):
    gen = torch.Generator().manual_seed(9)
    states = vb.ring_states(n, G, dev)
    L = layers[0]
    hv = dims["num_v_heads"] * dims["head_v_dim"]
    core = torch.empty((n * G, hv), dtype=torch.bfloat16, device=dev)
    for lay, h in HIPS.items():
        for mode in gdn_hip.modes(lay):
            try:
                proj = vb.rand_proj(n * G, L.width, gen, dev)
                _, hpw, vs = gdn_hip.MODES[mode]
                blocks = n * dims["num_v_heads"] // hpw * vs
                vals = []
                for _ in range(5):
                    h.prof.zero_()
                    h.step(L, proj.clone(), states, core, n, G, mode, prof=1)
                    torch.cuda.synchronize(dev)
                    vals.append(h.prof[: blocks * 8].view(blocks, 8).cpu())
                p = torch.stack(vals[1:]).view(-1, 8).double()
                med = p.median(0).values
                mx = p.max(0).values
                log(f"RESULT phase n={n} G={G} {lay}_{mode} blocks={blocks} cycles(s_memtime) median: "
                    f"prologue={med[4]:.0f} chain={med[5]:.0f} chain_per_row={med[5] / G:.0f} "
                    f"epilogue={med[6]:.0f} total={med[7]:.0f} (max over blocks {mx[7]:.0f}); "
                    f"s_memrealtime ticks: prologue={med[0]:.0f} chain={med[1]:.0f} epilogue={med[2]:.0f} "
                    f"total={med[3]:.0f} (max {mx[3]:.0f}) cycles_per_tick={med[7] / max(med[3], 1):.1f}")
            except Exception:  # noqa: BLE001
                log(f"RESULT phase {lay}_{mode} error:\n{traceback.format_exc()}")


def main() -> None:
    faulthandler.dump_traceback_later(2300, exit=True)
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="2x4,1x4,4x4,2x2,1x2")
    p.add_argument("--layouts", default=",".join(gdn_hip.LAYOUTS))
    p.add_argument("--replays", type=int, default=30)
    args = p.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    arch = getattr(props, "gcnArchName", "gfx90a").split(":")[0]
    log(f"RESULT device={props.name} arch={arch} cus={props.multi_processor_count} torch={torch.__version__}")
    try:
        ws, dims, src = vb.checkpoint_weights(dev)
    except Exception:  # noqa: BLE001
        log(f"RESULT weights from checkpoint failed:\n{traceback.format_exc()}")
        ws, dims, src = vb.random_weights(dev)
    while len(ws) < 30:
        ws = ws + ws
    ws = ws[:30]
    log(f"RESULT weights={src} dims={dims}")
    started = time.perf_counter()

    for lay in args.layouts.split(","):
        t0 = time.perf_counter()
        try:
            code, clog = gdn_hip.compile_layout(lay, dims)
            HIPS[lay] = gdn_hip.GdnHip(lay, code, dev)
            for k, v in sorted(hipmega_rtc.resources(clog).items()):
                if "_p1" in k:
                    continue
                m = k.split("_")
                occ = ""
                if k.startswith("gdn_L"):
                    _, hpw, vs = gdn_hip.MODES[m[2]]
                    nt = gdn_hip.threads(m[1], hpw, vs, dims["head_v_dim"])
                    occ = f" threads={nt} blocks_per_CU={HIPS[lay].mod.fn(k).occupancy(nt)}"
                log(f"RESULT compile {k} vgprs={v.get('VGPRs')} agprs={v.get('AGPRs')} "
                    f"spill={v.get('VGPRs Spill')} sgprs={v.get('SGPRs')} "
                    f"occ_waves_per_simd={v.get('Occupancy [waves/SIMD]')} lds={v.get('LDS Size [bytes/block]')}{occ}")
            log(f"RESULT compiled {lay} s={time.perf_counter() - t0:.1f}")
        except Exception:  # noqa: BLE001
            log(f"RESULT compile {lay} error:\n{traceback.format_exc()[-6000:]}")
    if not HIPS:
        log("RESULT no layout compiled")
        return
    names = variants()

    err = {}
    for n, G in ((2, 4), (1, 4), (2, 2)):
        try:
            err[(n, G)] = errors(["A0"] + names, ws[:3], dims, dev, n, G, args.replays)
        except Exception:  # noqa: BLE001
            log(f"RESULT error n={n} G={G} failed:\n{traceback.format_exc()}")
    log(f"RESULT errors_done s={time.perf_counter() - started:.0f}")
    try:
        geq = graph_equal(names, ws[:3], dims, dev, 2, 4)
    except Exception:  # noqa: BLE001
        geq = {}
        log(f"RESULT graph_equal failed:\n{traceback.format_exc()}")

    layers = [vb.make_layer(w, dims, dev, None) for w in ws]
    for L in layers:
        gdn_hip.prepack(L)
    tim = {}
    for cfg in args.configs.split(","):
        n, G = (int(v) for v in cfg.split("x"))
        for L in layers:
            L.conv.zero_()
            L.rec.zero_()
        gen = torch.Generator().manual_seed(11)
        st = vb.ring_states(n, G, dev)
        core = torch.empty((n * G, dims["num_v_heads"] * dims["head_v_dim"]), dtype=torch.bfloat16, device=dev)
        for _ in range(10):
            for L in layers:
                gv.today(L, vb.rand_proj(n * G, L.width, gen, dev), st, core, n, G)
        tim[(n, G)] = timing(["A0"] + names + ["stream128w8", "stream32w4", "empty"], layers, dev, n, G, dims)
    log(f"RESULT timing_done s={time.perf_counter() - started:.0f}")
    try:
        phases(layers, dev, dims)
    except Exception:  # noqa: BLE001
        log(f"RESULT phases failed:\n{traceback.format_exc()}")

    # ---- summary and the plan's rule
    for (n, G), res in tim.items():
        hs = {k: v for k, v in res.items() if k in names}
        if not hs or "A0" not in res:
            continue
        best = min(hs, key=hs.get)
        a0 = res["A0"]
        log(f"RESULT summary n={n} G={G} A0_today_ms={a0:.4f} best={best} best_ms={hs[best]:.4f} "
            f"save_ms={a0 - hs[best]:.4f} ratio={hs[best] / a0:.3f} "
            f"ceiling_today_shape_ms={res.get('stream128w8', float('nan')):.4f} "
            f"ceiling_stream32w4_ms={res.get('stream32w4', float('nan')):.4f} empty30_ms={res.get('empty', float('nan')):.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in sorted(hs.items(), key=lambda kv: kv[1])))
    if (2, 4) in tim and (1, 4) in tim:
        def ok(name):
            e = [err.get(c, {}).get(name, {}).get("ok", False) for c in err]
            return all(e) and bool(e) and geq.get(name, False)
        good = [k for k in names if ok(k) and k in tim[(2, 4)] and k in tim[(1, 4)]]
        log(f"RESULT numerics_ok_variants={good}")
        if good:
            b = min(good, key=lambda k: tim[(2, 4)][k])
            t24, t14 = tim[(2, 4)][b], tim[(1, 4)][b]
            save = tim[(2, 4)]["A0"] - t24
            verdict = "PASS" if t24 <= 0.55 and t14 <= 0.5 else ("KILL" if save < 0.45 else "GO (between kill and pass)")
            log(f"RESULT rule best={b} n2G4_ms={t24:.4f} (<=0.55) n1G4_ms={t14:.4f} (<=0.5) save_2x4_ms={save:.4f} "
                f"(kill <0.45) -> {verdict}")
        else:
            log("RESULT rule no variant within 2x error / graph-equal -> KILL")
    log(f"RESULT done s={time.perf_counter() - started:.0f}")


if __name__ == "__main__":
    main()
