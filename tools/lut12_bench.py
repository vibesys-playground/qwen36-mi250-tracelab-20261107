"""Plan 44 step 0: LUT-12 weights decoded in HIP MoE kernels vs today's Triton MoE decode kernels.

    python tools/lut12_bench.py [--devices 6,7] [--layers 4] [--samples 6]

1. Compile ``engine/fast/hip/lut12_moe.hip`` with hipRTC from torch's wheel (a child process;
   plan 39's loader, ``engine/fast/hipmega_rtc.py``) for MAXP = 4 and 8 rows and two tile
   configs, and print each kernel's VGPRs / spills / occupancy.
2. Load one 2-GCD replica as ``tools/moe_decode_bench.py`` does (``ENGINE_TP=1``, whole layers
   per GCD), prefill ``--sessions`` replay-like prompts (Request Factory-style synthetic coding
   lines, not the benchmark's sessions) and greedy-decode them eagerly, recording the MoE
   input ``h`` of every row at ``--layers`` layers held by cuda:0.
3. Pack every routed expert's TP2 shard (both ranks; gate_up [E, 2*I/2, H], down
   [E, H, I/2]), the shared expert's down shard and lm_head in LUT-12 on the GPU. Per class:
   tiles, fallback-tile rate, streamed ratio; round trip of every weight, bit for bit, both
   through the torch reference decoder and through the HIP kernels' own decoder.
4. Routing sets shaped like the served chained verify: R=4 = 2 slots x 2 rows (k=1), R=8 =
   2 slots x 4 rows (k=3), consecutive tokens of 2 sessions; the router GEMV and top-k from
   today's kernels on rank 0's TP2 shard of the real layer. Methods, all on the same sets:
   ``triton``: ``fused._moe_up_kernel`` + ``fused._moe_down_kernel`` exactly as
   ``FusedDecode._moe`` launches them (no residual, as under TP2); ``hip-raw`` and ``hip-pk``
   (BF16 vs LUT-12 weights, same kernels otherwise). Timed as CUDA graphs of back-to-back
   launches cycling over the sets (``moe_decode_bench.time_us``). Bitwise: hip-pk == hip-raw
   on every set (act and out); hip-raw vs triton: max relative difference.
5. A captured graph mixing Triton and hipRTC launches (top-k -> HIP up -> Triton down),
   replayed, against the same chain run eagerly.
6. The kill rule of plan 44 step 0 (lead brief): at 8 rows hip-pk up+down <= 0.85x of triton,
   round trip exact, size ratio <= 0.80.

Prints ``RESULT`` lines. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from engine.fast import hipmega_rtc as rtc  # noqa: E402
from engine.fast import lut12  # noqa: E402

CONFIGS = {"a": dict(NT=16, ITD=1, UN=2), "b": dict(NT=32, ITD=2, UN=2), "c": dict(NT=16, ITD=2, UN=4)}


def out(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================ compile


def compile_all(shape: dict) -> dict:
    """(maxp, cfg) -> (code, resources), compiled in a child process."""
    src = lut12.hip_source()
    path = Path(tempfile.gettempdir()) / "lut12_moe.hip"
    path.write_text(src)
    res = {}
    for maxp in (4, 8):
        for name, cfg in CONFIGS.items():
            defs = dict(shape, MAXP=maxp, **cfg)
            opts = tuple(f"-D{k}={v}" for k, v in defs.items())
            t0 = time.perf_counter()
            code, log = rtc.compile_hip_subprocess(str(path), options=opts)
            r = rtc.resources(log)
            res[(maxp, name)] = code
            for k, v in r.items():
                out(f"RESULT compile maxp={maxp} cfg={name}{cfg} kernel={k} vgprs={v.get('VGPRs')} "
                    f"spill={v.get('VGPRs Spill')} occ={v.get('Occupancy [waves/SIMD]')} "
                    f"lds={v.get('LDS Size [bytes/block]')} s={time.perf_counter() - t0:.1f}")
    return res


class Hip:
    """The kernels of one compiled variant on one device."""

    def __init__(self, code: bytes, dev: torch.device, maxp: int, cfg: dict, shape: dict) -> None:
        self.mod = rtc.Module(code, dev)
        self.dev, self.maxp, self.cfg, self.shape = dev, maxp, cfg, shape
        self.fn = {n: self.mod.fn(n) for n in ("moe_up_raw", "moe_up_pk", "moe_down_raw", "moe_down_pk",
                                               "lut12_decode")}

    def up(self, pk: bool, h, idx, w, side, act, R: int) -> None:
        S = self.shape
        grid = R * S["TOPK"] * (S["I"] // self.cfg["NT"])
        self.fn["moe_up_pk" if pk else "moe_up_raw"].launch(
            grid, 256, [h, idx, w, side, act, ctypes.c_int(R)])

    def down(self, pk: bool, act, r, idx, wt, w, side, sd, sside, res, outt, R: int) -> None:
        S = self.shape
        grid = R * (S["H"] // (8 * self.cfg["ITD"]))
        has_res = res is not None
        self.fn["moe_down_pk" if pk else "moe_down_raw"].launch(
            grid, 256, [act, r, idx, wt, w, side, sd, sside, res if has_res else outt, outt,
                        ctypes.c_int(R), ctypes.c_int(int(has_res))])

    def decode(self, p: lut12.Packed) -> torch.Tensor:
        nt = p.tiles
        o = torch.empty(nt * 128, dtype=torch.int16, device=p.planes.device)
        self.fn["lut12_decode"].launch((nt + 15) // 16, 256, [p.planes, p.side, o, ctypes.c_int64(nt)])
        return o


# ============================================================================ packing stats


class Stats:
    def __init__(self) -> None:
        self.c: dict[str, list] = {}

    def add(self, cls: str, p: lut12.Packed, exact: bool) -> None:
        s = self.c.setdefault(cls, [0, 0, 0, 0, True, 0])
        s[0] += p.tiles
        s[1] += p.fallback
        s[2] += p.nbytes
        s[3] += p.tiles * 256
        s[4] &= exact
        s[5] += 1

    def report(self) -> tuple[float, bool, float]:
        tot = [0, 0, 0, 0, True]
        for cls, (t, f, b, raw, ok, n) in sorted(self.c.items()):
            out(f"RESULT pack class={cls} tensors={n} tiles={t} fallback_tiles={f} fallback_rate={f / t:.2e} "
                f"raw_GB={raw / 1e9:.3f} packed_GB={b / 1e9:.3f} ratio={b / raw:.4f} roundtrip_bitexact={ok}")
            for i, v in enumerate((t, f, b, raw)):
                tot[i] += v
            tot[4] &= ok
        out(f"RESULT pack total tiles={tot[0]} fallback_tiles={tot[1]} fallback_rate={tot[1] / tot[0]:.2e} "
            f"raw_GB={tot[3] / 1e9:.2f} packed_GB={tot[2] / 1e9:.2f} ratio={tot[2] / tot[3]:.4f} "
            f"roundtrip_all_bitexact={tot[4]}")
        return tot[2] / tot[3], tot[4], tot[1] / tot[0]


def pack_checked(w: torch.Tensor, hip: Hip | None) -> tuple[lut12.Packed, bool]:
    p = lut12.pack(w)
    ref = w.contiguous().view(torch.int16).flatten()
    ok = torch.equal(lut12.unpack(p).view(torch.int16).flatten(), ref)
    if hip is not None:
        ok &= torch.equal(hip.decode(p), ref)
    return p, ok


def shard(mlp, rank: int, ranks: int = 2):
    """Rank ``rank``'s TP shard of a layer's routed and shared down weights (tp._moe_shard)."""
    gu, dn = mlp.experts.gate_up_proj, mlp.experts.down_proj
    inter = gu.shape[1] // 2
    i2 = inter // ranks
    lo, hi = rank * i2, (rank + 1) * i2
    sd = mlp.shared_expert.down_proj.weight
    s2 = sd.shape[1] // ranks
    return (torch.cat([gu[:, lo:hi], gu[:, inter + lo : inter + hi]], dim=1).contiguous(),
            dn[:, :, lo:hi].contiguous(), sd[:, rank * s2 : (rank + 1) * s2].contiguous())


# ============================================================================ main


def real(args) -> None:
    os.environ.setdefault("ENGINE_MTP", "1")
    os.environ.setdefault("ENGINE_TP", "1")
    import moe_decode_bench as mdb
    from engine.fast import fused
    from engine.generator import CHECKPOINT, Generator

    triton = fused.triton
    shape = dict(H=2048, I=256, IS=256, E=256, N3=256 + 2 * 256 + 1, TOPK=8)
    codes = compile_all(shape)

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast, tok = g.fast, g.tokenizer
    if fast.fused is None:
        raise SystemExit(f"needs the fused layers: {getattr(fast, 'fused_error', None)}")
    dev0 = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev0)
    cus = props.multi_processor_count
    out(f"RESULT env load_s={time.perf_counter() - started:.0f} gcn={getattr(props, 'gcnArchName', '?')} "
        f"cus={cus} torch={torch.__version__} triton={triton.__version__} "
        f"visible={os.environ.get('HIP_VISIBLE_DEVICES')} moe_order={fused.MOE_ORDER} "
        f"moe_grid={fused.MOE_GRID} moe_tile_rows={fused.MOE_TILE_ROWS} tiles={fused.TILES}")
    mlp0 = fast.layers[0].mlp
    E, two_i, H = mlp0.experts.gate_up_proj.shape
    assert (E, two_i // 2 // 2, H, mlp0.shared_expert.down_proj.weight.shape[1] // 2) == (
        shape["E"], shape["I"], shape["H"], shape["IS"]), "unexpected model shape"
    devs = sorted({fast.layers[i].mlp.experts.gate_up_proj.device for i in range(len(fast.layers))},
                  key=lambda d: d.index)
    hips = {(maxp, name, d): Hip(codes[(maxp, name)], d, maxp, CONFIGS[name], shape)
            for maxp in (4, 8) for name in CONFIGS for d in devs}

    # ---- routing: record h at the picked layers in an eager decode
    on0 = [i for i in range(len(fast.layers)) if fast.layers[i].mlp.experts.gate_up_proj.device == dev0]
    pick = list(dict.fromkeys(on0[round(j * (len(on0) - 1) / max(1, args.layers - 1))]
                              for j in range(args.layers)))
    out(f"RESULT layers on_gcd0={len(on0)} picked={pick}")

    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    S = args.sessions
    corpus = pool(9973 * S + args.prompt_len + 10)
    prompts = [corpus[9973 * k : 9973 * k + args.prompt_len] for k in range(S)]

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(prompt) - 1
        position = 0
        while position < body:
            end = min((position // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[position:end], position, prompt[end])
            position = end
        return fast.prefill(slot, prompt[body:], body)

    rec: dict[int, list] = {i: [] for i in pick}
    state = {"on": False}
    moe0 = fast.fused._moe

    def moe_rec(i, x, h):
        if state["on"] and i in rec:
            rec[i].append(h.detach().clone())
        return moe0(i, x, h)

    fast.fused._moe = moe_rec
    t0 = time.perf_counter()
    with torch.inference_mode():
        slots = list(range(S))
        lasts = [int(torch.argmax(prefill(s, p))) for s, p in zip(slots, prompts, strict=True)]
        fast.use_graphs = False
        for step in range(args.steps):
            state["on"] = True
            logits = fast.decode(lasts, slots, [args.prompt_len + step] * S)
            state["on"] = False
            lasts = logits.argmax(-1).tolist()
        for s in slots:
            fast.reset(s)
    fast.fused._moe = moe0
    out(f"RESULT decode sessions={S} steps={args.steps} s={time.perf_counter() - t0:.1f}")

    with torch.inference_mode():
        # ---- (a) pack every routed/shared-down TP2 shard and lm_head; round trip every weight
        t0 = time.perf_counter()
        st = Stats()
        packed: dict[int, tuple] = {}
        for i, layer in enumerate(fast.layers):
            mlp = layer.mlp
            d = mlp.experts.gate_up_proj.device
            hp = hips[(8, "a", d)]
            for rank in (0, 1):
                gu, dn, sd = shard(mlp, rank)
                pg, okg = pack_checked(gu, hp)
                pd, okd = pack_checked(dn, hp)
                ps, oks = pack_checked(sd, hp)
                st.add("routed_gate_up_tp2", pg, okg)
                st.add("routed_down_tp2", pd, okd)
                st.add("shared_down_tp2", ps, oks)
                if rank == 0 and i in pick:
                    packed[i] = (gu, dn, sd, pg, pd, ps)
                else:
                    del gu, dn, sd, pg, pd, ps
            torch.cuda.empty_cache()
        lm = getattr(fast, "lm_head", None)
        lmw = getattr(lm, "weight", lm) if lm is not None else None
        if isinstance(lmw, torch.Tensor) and lmw.dtype == torch.bfloat16 and lmw.shape[-1] % 128 == 0:
            hp = hips[(8, "a", lmw.device)] if (8, "a", lmw.device) in hips else None
            p, ok = pack_checked(lmw, hp)
            st.add("lm_head", p, ok)
            del p
        else:
            out(f"RESULT pack lm_head skipped (fast.lm_head is {type(lm).__name__})")
        ratio, exact, fb_rate = st.report()
        out(f"RESULT pack s={time.perf_counter() - t0:.0f}")

        # ---- (b) routing sets and the three methods
        # The HIP decoder of the last packed tensor left cuda:1 current; Triton launches on the
        # current device, so make cuda:0 (where the picked layers live) current first (j216).
        torch.cuda.set_device(dev0)
        fd = fast.fused
        layers = {}
        for i in pick:
            mlp = fast.layers[i].mlp
            gu, dn, sd, pg, pd, ps = packed[i]
            s2 = sd.shape[1]
            sh = mlp.shared_expert
            w3 = torch.cat([mlp.gate.weight, sh.gate_proj.weight[:s2], sh.up_proj.weight[:s2],
                            mlp.shared_expert_gate.weight]).contiguous()
            L = mdb.Layer(f"L{i}", gu, dn, sd, w3=w3)
            L.pg, L.pd, L.ps = pg, pd, ps
            layers[i] = L
        T = args.steps
        sets_by_r: dict[int, list] = {}
        for R, per in ((4, 2), (8, 4)):
            sets = []
            for k in range(args.samples):
                t = (k * 5 + 3) % (T - per)
                sess = [(2 * k) % S, (2 * k + 1) % S]
                spec = [(t + j, s) for s in sess for j in range(per)]
                for i in pick:
                    hs = rec[i]
                    h = torch.stack([hs[tt][s] for tt, s in spec]).contiguous()
                    x = torch.zeros_like(h)
                    r = fd._gemv(h, layers[i].w3)
                    sets.append(mdb.Set(layers[i], h, x, r, fast.top_k))
            sets_by_r[R] = sets
        torch.cuda.synchronize()
        torch.cuda.set_device(dev0)

        bn = fused.TILES["moe_bn"]

        def tri_up(s):
            L = s.layer
            order = 1 if s.R >= fused.MOE_TILE_ROWS else 0
            items = s.R * s.topk * triton.cdiv(L.I, bn)
            fused._moe_up_kernel[(items if order else min(items, fused.MOE_GRID * cus),)](
                s.h, s.idx, L.gu, s.act, s.R, L.gu.stride(0), s.r, s.wt, L.N3,
                H=L.H, I=L.I, TOPK=s.topk, BN=bn, BK=min(1024, mdb.p2(L.H)), ORDER=order, E=L.E,
                TOPKP=mdb.p2(s.topk), FUSE_TOPK=False, num_warps=4)

        def tri_dn(s, act=None, outt=None):
            L = s.layer
            order = 1 if s.R >= fused.MOE_TILE_ROWS else 0
            items = s.R * triton.cdiv(L.H, bn)
            o = s.out if outt is None else outt
            fused._moe_down_kernel[(min(items, fused.MOE_GRID * cus),)](
                s.act if act is None else act, s.r, s.idx, s.wt, L.dn, L.sd, o, o, s.R,
                L.dn.stride(0), L.N3, H=L.H, I=L.I, IS=L.IS, E=L.E, TOPK=s.topk, BN=bn,
                BKI=min(1024, mdb.p2(L.I)), BKS=min(1024, mdb.p2(L.IS)), ORDER=order,
                HAS_RES=False, num_warps=4)

        def hip_up(s, pk, cfg):
            L = s.layer
            hp = hips[(s.R, cfg, dev0)]
            if pk:
                hp.up(True, s.h, s.idx, L.pg.planes, L.pg.side, s.hact, s.R)
            else:
                hp.up(False, s.h, s.idx, L.gu, L.pg.side, s.hact, s.R)

        def hip_dn(s, pk, cfg, act=None, outt=None):
            L = s.layer
            hp = hips[(s.R, cfg, dev0)]
            a = s.hact if act is None else act
            o = s.hout if outt is None else outt
            if pk:
                hp.down(True, a, s.r, s.idx, s.wt, L.pd.planes, L.pd.side, L.ps.planes, L.ps.side, None, o, s.R)
            else:
                hp.down(False, a, s.r, s.idx, s.wt, L.dn, L.pd.side, L.sd, L.ps.side, None, o, s.R)

        # correctness
        for R, sets in sets_by_r.items():
            nus = []
            for s in sets:
                s.hact = torch.empty_like(s.act)
                s.hout = torch.empty_like(s.out)
                tri_up(s)
                tri_dn(s)
                s.ref_act, s.ref_out = s.act.clone(), s.out.clone()
                s.nu = len(set(s.idx.flatten().tolist()))
                nus.append(s.nu)
            out(f"RESULT routing R={R} sets={len(sets)} pairs={R * 8} distinct_mean={statistics.mean(nus):.2f} "
                f"min={min(nus)} max={max(nus)}")
            for cfg in CONFIGS:
                same_act = same_out = same_dn = True
                rel_act = rel_out = rel_dn = 0.0
                for s in sets:
                    res = {}
                    for pk in (False, True):
                        s.hact.fill_(0)
                        s.hout.fill_(0)
                        hip_up(s, pk, cfg)
                        hip_dn(s, pk, cfg)
                        a1, o1 = s.hact.clone(), s.hout.clone()
                        s.hout.fill_(0)
                        hip_dn(s, pk, cfg, act=s.ref_act)
                        res[pk] = (a1, o1, s.hout.clone())
                    torch.cuda.synchronize()
                    same_act &= torch.equal(res[False][0], res[True][0])
                    same_out &= torch.equal(res[False][1], res[True][1])
                    same_dn &= torch.equal(res[False][2], res[True][2])

                    def rel(a, b):
                        a, b = a.float(), b.float()
                        return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))

                    rel_act = max(rel_act, rel(res[False][0], s.ref_act))
                    rel_out = max(rel_out, rel(res[False][1], s.ref_out))
                    rel_dn = max(rel_dn, rel(res[False][2], s.ref_out))
                out(f"RESULT bitwise R={R} cfg={cfg} pk_eq_raw_act={same_act} pk_eq_raw_out={same_out} "
                    f"pk_eq_raw_down_on_triton_act={same_dn} raw_vs_triton_maxrel_act={rel_act:.2e} "
                    f"out={rel_out:.2e} down_only={rel_dn:.2e}")

        # timing
        I_, H_, IS_ = shape["I"], shape["H"], shape["IS"]
        up_b, dn_b, sd_b = 2 * I_ * H_ * 2, H_ * I_ * 2, H_ * IS_ * 2
        L0 = layers[pick[0]]
        pk_up_b = L0.pg.nbytes / E
        pk_dn_b = L0.pd.nbytes / E
        pk_sd_b = L0.ps.nbytes
        summary = {}
        for R, sets in sets_by_r.items():
            nu = statistics.mean(s.nu for s in sets)
            raw_bytes = nu * (up_b + dn_b) + sd_b
            pk_bytes = nu * (pk_up_b + pk_dn_b) + pk_sd_b
            t_up = mdb.time_us([lambda s=s: tri_up(s) for s in sets])
            t_dn = mdb.time_us([lambda s=s: tri_dn(s) for s in sets])
            t_ud = mdb.time_us([lambda s=s: (tri_up(s), tri_dn(s)) for s in sets])
            out(f"RESULT time R={R} method=triton up_us={t_up:.1f} dn_us={t_dn:.1f} updn_us={t_ud:.1f} "
                f"GBps_unique={raw_bytes / t_ud / 1e3:.0f} unique_experts={nu:.1f}")
            summary[(R, "triton")] = t_ud
            for cfg in CONFIGS:
                for pk in (False, True):
                    m = "hip-pk" if pk else "hip-raw"
                    try:
                        u = mdb.time_us([lambda s=s: hip_up(s, pk, cfg) for s in sets])
                        d = mdb.time_us([lambda s=s: hip_dn(s, pk, cfg) for s in sets])
                        ud = mdb.time_us([lambda s=s: (hip_up(s, pk, cfg), hip_dn(s, pk, cfg)) for s in sets])
                    except Exception as exc:  # noqa: BLE001
                        out(f"RESULT time R={R} method={m} cfg={cfg} failed: {exc!r}")
                        summary[(R, m, cfg)] = float("inf")
                        continue
                    b = pk_bytes if pk else raw_bytes
                    out(f"RESULT time R={R} method={m} cfg={cfg}{CONFIGS[cfg]} up_us={u:.1f} dn_us={d:.1f} "
                        f"updn_us={ud:.1f} x_triton={ud / t_ud:.3f} up_x={u / t_up:.3f} dn_x={d / t_dn:.3f} "
                        f"GBps_streamed={b / ud / 1e3:.0f} GBps_raw_equiv={raw_bytes / ud / 1e3:.0f}")
                    summary[(R, m, cfg)] = ud

        # ---- (c) graph capture: Triton top-k -> hipRTC up -> Triton down, replayed vs eager
        s = sets_by_r[8][0]
        L = s.layer
        idx2, wt2 = torch.empty_like(s.idx), torch.empty_like(s.wt)
        act2, out2 = torch.empty_like(s.act), torch.empty_like(s.out)
        hp = hips[(8, "a", dev0)]

        def chain():
            fused._topk_kernel[(s.R,)](s.r, idx2, wt2, L.N3, E=L.E, TOPK=s.topk, TOPKP=mdb.p2(s.topk),
                                       num_warps=1)
            hp.up(True, s.h, idx2, L.pg.planes, L.pg.side, act2, s.R)
            tri_dn(s, act=act2, outt=out2)

        try:
            chain()
            torch.cuda.synchronize()
            eager = out2.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                chain()
            for buf in (idx2, act2, out2):
                buf.zero_()
            wt2.zero_()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            cap_ok = torch.equal(out2, eager)
            nodes = None
            try:
                nodes = graph.raw_cuda_graph() if hasattr(graph, "raw_cuda_graph") else None
            except Exception:  # noqa: BLE001
                nodes = None
            out(f"RESULT graph_capture mixed(triton topk -> hipRTC up -> triton down) replay_equals_eager={cap_ok} "
                f"out_nonzero={bool(out2.abs().sum() > 0)} raw_graph={nodes is not None}")
        except Exception as exc:  # noqa: BLE001
            out(f"RESULT graph_capture failed: {exc!r}")

        # ---- verdict
        best = {}
        for R in (4, 8):
            for m in ("hip-raw", "hip-pk"):
                best[(R, m)] = min((summary[(R, m, c)], c) for c in CONFIGS)
            tri = summary[(R, "triton")]
            out(f"RESULT summary R={R} triton_updn_us={tri:.1f} hip_raw_best_us={best[(R, 'hip-raw')][0]:.1f} "
                f"(cfg {best[(R, 'hip-raw')][1]}) raw_x_triton={best[(R, 'hip-raw')][0] / tri:.3f} "
                f"hip_pk_best_us={best[(R, 'hip-pk')][0]:.1f} (cfg {best[(R, 'hip-pk')][1]}) "
                f"pk_x_triton={best[(R, 'hip-pk')][0] / tri:.3f} pk_x_raw={best[(R, 'hip-pk')][0] / best[(R, 'hip-raw')][0]:.3f}")
        pk8 = best[(8, "hip-pk")][0] / summary[(8, "triton")]
        raw8 = best[(8, "hip-raw")][0] / summary[(8, "triton")]
        passed = pk8 <= 0.85 and exact and ratio <= 0.80
        out(f"RESULT verdict {'PASS' if passed else 'KILL'} pk_x_triton_R8={pk8:.3f} (<=0.85) "
            f"roundtrip_exact={exact} ratio={ratio:.4f} (<=0.80) fallback_rate={fb_rate:.2e} "
            f"raw_x_triton_R8={raw8:.3f} raw_hip_slower_than_1.15x={raw8 > 1.15}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="6,7")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--prompt-len", type=int, default=1800)
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    try:
        real(args)
    except Exception:  # noqa: BLE001
        out("RESULT fatal " + traceback.format_exc().replace("\n", " | "))
        raise


if __name__ == "__main__":
    main()
