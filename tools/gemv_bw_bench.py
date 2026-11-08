"""Plan 19 step 0: weight bandwidth of the dense decode GEMVs on one gfx90a GCD, and the
kernel split of one B=2 verify step.

    python tools/gemv_bw_bench.py [--devices 0,1] [--no-step] [--quick]

Part 1 (one GCD, cuda:0, random BF16 weights of the real shapes): for 1, 2, 4 and 8 rows,
- ``cur-*``: the fused GEMV of ``engine.fast.fused`` (``_gemv_kernel``) with the tiles each
  ``ENGINE_FUSED_ROW_TILE`` gives (pairs = default, fixed, rows), at its default grid (2
  programs per CU) and at 4 and 8 per CU; ``pairs`` is also the lm_head's default
  (``ENGINE_FUSED_HEAD_TILE=4,2048``, grid 2);
- ``blas-linear`` / ``blas-matmul``: ``F.linear(x, w)`` and ``torch.matmul(x, w.t())``
  (hipBLASLt or rocBLAS, whichever torch picks);
- ``mr``: a prototype row-invariant multi-row GEMV (below), swept over outputs per tile BN,
  K chunk BK, warps, accumulation mode, split-K, pipelining stages and grid.
Every timing is a CUDA graph of back-to-back launches over rotating weight copies (>= 64 MB,
so the 8 MB L2 holds none of them), replayed and timed with events; us per launch and the
weight bytes / time as GB/s.

The prototype ``_mr_kernel``: a tile is BN outputs x one K part (K / KSPLIT). Each K chunk
of the weight [BN, BK] is loaded once and used for every row; each row has its own
accumulator: MODE 0 a per-lane fp32 [BN, BK] (one cross-lane reduction per tile), MODE 1 a
[BN] vector reduced per chunk. Rows are a constexpr count NR, unrolled with one accumulator
each of the same shape, so a row's arithmetic never depends on NR. KSPLIT > 1 writes fp32
partials and a second kernel sums them in a fixed order. Bitwise check: for each config, the
outputs of x[:R] for R = 2, 4, 8 against each row run alone (R = 1).

Part 2 (``--devices``, one 2-GCD replica as the server loads it): one B=2 verify step
(4 rows, captured graphs) under torch.profiler, the per-kernel table per step; then the
same step eager with the GEMV calls labeled by shape, to attribute GEMV time to weights.

Prints ``RESULT`` lines. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (name, N, K, uses per verify B=2 step, kernel family in the engine at c7e3502)
SHAPES = [
    ("in_proj", 12352, 2048, 30, "fused"),   # GDN in_proj_qkv|z|b|a
    ("out_proj", 2048, 4096, 30, "fused"),   # GDN out_proj
    ("qkv", 9216, 2048, 11, "fused"),        # attention q(+gate)|k|v, 10 layers + MTP layer
    ("o_proj", 2048, 4096, 11, "fused"),     # attention o_proj, 10 + MTP
    ("router_shared", 1281, 2048, 41, "fused"),  # router|shared gate|up|shared_expert_gate
    ("mtp_fc", 2048, 4096, 1, "blas"),       # MTP fc (F.linear)
    ("lm_head", 248320, 2048, 1, "fused"),   # base lm_head (fused head)
    ("mtp_lm_head", 248320, 2048, 1, "blas"),  # MTP head's lm_head read (F.linear)
]
ROWS = (1, 2, 4, 8)


# ============================================================================ prototype kernel
# Module level so Triton resolves the helpers as globals. torch import does not touch the GPUs,
# so HIP_VISIBLE_DEVICES set in main() still applies.
import torch  # noqa: E402

from engine.fast import kernels as _kernels  # noqa: E402

triton, tl = _kernels.triton, _kernels.tl

if triton is not None:
    @triton.jit
    def _zero(BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
        if MODE == 0:
            return tl.zeros([BN, BK], dtype=tl.float32)
        else:
            return tl.zeros([BN], dtype=tl.float32)

    @triton.jit
    def _row(a, w, x_ptr, r: tl.constexpr, K: tl.constexpr, cols, MODE: tl.constexpr):
        xr = tl.load(x_ptr + r * K + cols).to(tl.float32)
        if MODE == 0:
            return a + w * xr[None, :]
        else:
            return a + tl.sum(w * xr[None, :], axis=1)

    @triton.jit
    def _fin(a, MODE: tl.constexpr):
        if MODE == 0:
            return tl.sum(a, axis=1)
        else:
            return a

    @triton.jit
    def _put(a, r: tl.constexpr, n, nok, out_ptr, part_ptr, kp, N, NR: tl.constexpr,
             KSPLIT: tl.constexpr, MODE: tl.constexpr):
        s = _fin(a, MODE)
        if KSPLIT == 1:
            tl.store(out_ptr + r * N + n, s.to(tl.bfloat16), mask=nok)
        else:
            tl.store(part_ptr + (kp * NR + r) * N + n, s, mask=nok)

    @triton.jit
    def _mr_kernel(x_ptr, w_ptr, out_ptr, part_ptr, N, ntiles,
                   K: tl.constexpr, NR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   KSPLIT: tl.constexpr, MODE: tl.constexpr):
        KP: tl.constexpr = K // KSPLIT
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        for t in range(pid, ntiles, nprog):
            nb = t // KSPLIT
            kp = t % KSPLIT
            n = nb * BN + tl.arange(0, BN)
            nok = n < N
            wbase = w_ptr + n[:, None].to(tl.int64) * K
            a0 = _zero(BN, BK, MODE)
            a1 = _zero(BN, BK, MODE)
            a2 = _zero(BN, BK, MODE)
            a3 = _zero(BN, BK, MODE)
            a4 = _zero(BN, BK, MODE)
            a5 = _zero(BN, BK, MODE)
            a6 = _zero(BN, BK, MODE)
            a7 = _zero(BN, BK, MODE)
            for k0 in range(kp * KP, kp * KP + KP, BK):
                cols = k0 + tl.arange(0, BK)
                w = tl.load(wbase + cols[None, :], mask=nok[:, None], other=0.0).to(tl.float32)
                a0 = _row(a0, w, x_ptr, 0, K, cols, MODE)
                if NR > 1:
                    a1 = _row(a1, w, x_ptr, 1, K, cols, MODE)
                if NR > 2:
                    a2 = _row(a2, w, x_ptr, 2, K, cols, MODE)
                if NR > 3:
                    a3 = _row(a3, w, x_ptr, 3, K, cols, MODE)
                if NR > 4:
                    a4 = _row(a4, w, x_ptr, 4, K, cols, MODE)
                if NR > 5:
                    a5 = _row(a5, w, x_ptr, 5, K, cols, MODE)
                if NR > 6:
                    a6 = _row(a6, w, x_ptr, 6, K, cols, MODE)
                if NR > 7:
                    a7 = _row(a7, w, x_ptr, 7, K, cols, MODE)
            _put(a0, 0, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)
            if NR > 1:
                _put(a1, 1, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)
            if NR > 2:
                _put(a2, 2, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)
            if NR > 3:
                _put(a3, 3, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)
            if NR > 4:
                _put(a4, 4, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)
            if NR > 5:
                _put(a5, 5, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)
            if NR > 6:
                _put(a6, 6, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)
            if NR > 7:
                _put(a7, 7, n, nok, out_ptr, part_ptr, kp, N, NR, KSPLIT, MODE)

    @triton.jit
    def _ksum_kernel(part_ptr, out_ptr, M, KSPLIT: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < M
        s = tl.load(part_ptr + i, mask=m, other=0.0)
        for p in tl.static_range(1, KSPLIT):
            s += tl.load(part_ptr + p * M + i, mask=m, other=0.0)
        tl.store(out_ptr + i, s.to(tl.bfloat16), mask=m)


def out(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--no-step", action="store_true")
    parser.add_argument("--no-gemv", action="store_true")
    parser.add_argument("--quick", action="store_true", help="a small sweep (smoke test)")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")
    import torch

    if not args.no_gemv:
        try:
            gemv_part(torch, args)
        except Exception:  # noqa: BLE001
            out("RESULT gemv_part_error " + traceback.format_exc().replace("\n", " | "))
        torch.cuda.empty_cache()
    if not args.no_step:
        step_part(torch, args)


# ============================================================================ part 1


def gemv_part(torch, args) -> None:
    import torch.nn.functional as F

    from engine.fast import fused

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    cus = props.multi_processor_count
    try:
        blas = str(torch.backends.cuda.preferred_blas_library())
    except Exception:  # noqa: BLE001
        blas = "?"
    out(f"RESULT env device={props.name} gcn={getattr(props, 'gcnArchName', '?')} cus={cus} "
        f"torch={torch.__version__} triton={triton.__version__} hip={torch.version.hip} blas={blas} "
        f"visible={os.environ.get('HIP_VISIBLE_DEVICES')}")

    # ------------------------------------------------------------ launchers

    def cur_launch(x, w, o, bp, bn, bk, grid_per_cu):
        rows, k = x.shape
        n = w.shape[0]
        work = triton.cdiv(n, bn) * triton.cdiv(rows, bp)
        grid = min(work, grid_per_cu * cus)
        fused._gemv_kernel[(grid,)](
            x, w, o, rows, n, o.stride(0), o, o, o, 0, 0,
            K=k, BP=bp, BN=bn, BK=bk, CONV=False, KC=2, num_warps=4,
        )

    def cur_tiles(variant, rows, k):
        p2 = triton.next_power_of_2
        if variant == "pairs":
            bp = 2
        elif variant == "fixed":
            bp = max(2, p2(fused.MAX_ROWS))
        else:
            bp = max(2, p2(rows))
        bk = min(fused.TILES["bk4"] if bp <= 4 else fused.TILES["bk8"], p2(k))
        return bp, fused.TILES["bn"], bk

    def mr_launch(x, w, o, part, cfg, grid_per_cu):
        bn, bk, warps, mode, ksplit, stages = cfg
        rows, k = x.shape
        n = w.shape[0]
        ntiles = triton.cdiv(n, bn) * ksplit
        grid = ntiles if grid_per_cu == 0 else min(ntiles, grid_per_cu * cus)
        _mr_kernel[(grid,)](x, w, o, part, n, ntiles, K=k, NR=rows, BN=bn, BK=bk,
                            KSPLIT=ksplit, MODE=mode, num_warps=warps, num_stages=stages)
        if ksplit > 1:
            m = rows * n
            _ksum_kernel[(triton.cdiv(m, 1024),)](part, o, m, KSPLIT=ksplit, BLOCK=1024,
                                                  num_warps=4)

    def valid(cfg, k):
        bn, bk, warps, mode, ksplit, stages = cfg
        return k % ksplit == 0 and (k // ksplit) % bk == 0 and (k // ksplit) >= bk

    # ------------------------------------------------------------ timing

    weights: dict = {}

    def get_weights(n, k):
        key = (n, k)
        if key not in weights:
            weights.clear()
            torch.cuda.empty_cache()
            nbytes = n * k * 2
            copies = max(1, min(16, math.ceil(64e6 / nbytes)))
            g = torch.Generator(device=dev).manual_seed(n * 7 + k)
            weights[key] = [
                (torch.randn((n, k), device=dev, generator=g, dtype=torch.float32) * (k ** -0.5))
                .to(torch.bfloat16)
                for _ in range(copies)
            ]
        return weights[key]

    def time_us(fn, ws, nbytes) -> float:
        """us per launch of fn(w) in a CUDA graph cycling over the weight copies."""
        launches = max(len(ws), 4, min(64, math.ceil(400e6 / nbytes)))
        for w in ws[:2]:
            fn(w)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(ws[0])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(launches):
                fn(ws[i % len(ws)])
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(5):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) * 1e3 / launches)
        del graph
        return statistics.median(times)

    results: dict = defaultdict(dict)  # (shape, R) -> {label: us}

    def record(name, n, k, r, label, us):
        nbytes = n * k * 2
        gbps = nbytes / (us * 1e-6) / 1e9
        results[(name, r)][label] = us
        out(f"RESULT gemv shape={name} N={n} K={k} R={r} kernel={label} us={us:.1f} GBps={gbps:.0f}")

    xs = {}

    def get_x(r, k):
        if (r, k) not in xs:
            g = torch.Generator(device=dev).manual_seed(1000 + k)
            base = torch.randn((8, k), device=dev, generator=g).to(torch.bfloat16)
            xs[(8, k)] = base
            for rr in ROWS:
                xs[(rr, k)] = base[:rr].contiguous()
        return xs[(r, k)]

    # Reference: a large device copy (read + write bytes) for the GCD's practical peak.
    big = torch.empty(int(512e6), dtype=torch.uint8, device=dev)
    big2 = torch.empty_like(big)
    us = time_us(lambda w: big2.copy_(w), [big], 1e9)
    out(f"RESULT copy_ref bytes_rw=1.024e9 us={us:.1f} GBps={1.024e9 / (us * 1e-6) / 1e9:.0f}")
    del big, big2

    distinct = []
    for name, n, k, uses, fam in SHAPES:
        if (n, k) not in [(d[1], d[2]) for d in distinct]:
            distinct.append((name, n, k))

    # Stage A grid of prototype configs (at R=4, grid 4 per CU, stages 2).
    if args.quick:
        bnbk = [(8, 512), (16, 256), (4, 2048)]
        warps_l, modes, ksplits = (4,), (0, 1), (1, 2)
    else:
        bnbk = [(4, 2048), (8, 1024), (8, 512), (16, 512), (16, 256), (32, 256), (32, 128),
                (64, 128), (64, 64)]
        warps_l, modes, ksplits = (4, 8), (0, 1), (1, 2, 4)

    best_cfgs: dict = {}
    for name, n, k in distinct:
        ws = get_weights(n, k)
        nbytes = n * k * 2
        # current kernel variants and BLAS
        for r in ROWS:
            x = get_x(r, k)
            o = torch.empty((r, n), dtype=torch.bfloat16, device=dev)
            for variant in ("pairs", "fixed", "rows"):
                bp, bn, bk = cur_tiles(variant, r, k)
                grids = (2, 4, 8) if variant == "pairs" else (2,)
                for gpc in grids:
                    label = f"cur-{variant}-g{gpc}"
                    try:
                        us = time_us(lambda w: cur_launch(x, w, o, bp, bn, bk, gpc), ws, nbytes)
                        record(name, n, k, r, label, us)
                    except Exception as exc:  # noqa: BLE001
                        out(f"RESULT gemv shape={name} R={r} kernel={label} ERROR {type(exc).__name__}: {str(exc)[:200]}")
            try:
                us = time_us(lambda w: F.linear(x, w), ws, nbytes)
                record(name, n, k, r, "blas-linear", us)
                us = time_us(lambda w: torch.matmul(x, w.t()), ws, nbytes)
                record(name, n, k, r, "blas-matmul", us)
            except Exception as exc:  # noqa: BLE001
                out(f"RESULT gemv shape={name} R={r} kernel=blas ERROR {type(exc).__name__}: {str(exc)[:200]}")

        # prototype stage A at R=4
        r = 4
        x = get_x(r, k)
        o = torch.empty((r, n), dtype=torch.bfloat16, device=dev)
        part = torch.empty((4 * r * n,), dtype=torch.float32, device=dev)
        stage_a = []
        for bn, bk in bnbk:
            for warps in warps_l:
                for mode in modes:
                    for ks in ksplits:
                        cfg = (bn, bk, warps, mode, ks, 2)
                        if not valid(cfg, k):
                            continue
                        label = "mr-" + "-".join(map(str, cfg)) + "-g4"
                        try:
                            us = time_us(lambda w: mr_launch(x, w, o, part, cfg, 4), ws, nbytes)
                            record(name, n, k, r, label, us)
                            stage_a.append((us, cfg))
                        except Exception as exc:  # noqa: BLE001
                            out(f"RESULT gemv shape={name} R={r} kernel={label} ERROR {type(exc).__name__}: {str(exc)[:200]}")
        stage_a.sort()
        top = [c for _, c in stage_a[:3]]
        # stage B: top configs x stages x rows x grid, plus the bitwise row-invariance check
        cand = []
        for cfg in top:
            for stages in (1, 2):
                cand.append(cfg[:5] + (stages,))
        cand = list(dict.fromkeys(cand))
        agg = {}
        for cfg in cand:
            per_r = {}
            for r in ROWS:
                x = get_x(r, k)
                o = torch.empty((r, n), dtype=torch.bfloat16, device=dev)
                part = torch.empty((max(1, cfg[4]) * r * n,), dtype=torch.float32, device=dev)
                best = None
                for gpc in (1, 2, 4, 8, 0):
                    label = "mr-" + "-".join(map(str, cfg)) + f"-g{gpc}"
                    try:
                        us = time_us(lambda w: mr_launch(x, w, o, part, cfg, gpc), ws, nbytes)
                        record(name, n, k, r, label, us)
                        if best is None or us < best[0]:
                            best = (us, gpc)
                    except Exception as exc:  # noqa: BLE001
                        out(f"RESULT gemv shape={name} R={r} kernel={label} ERROR {type(exc).__name__}: {str(exc)[:200]}")
                if best:
                    per_r[r] = best
            ok = check_invariance(torch, ws[0], k, n, cfg, mr_launch, get_x, name)
            if len(per_r) == len(ROWS):
                agg[cfg] = (sum(v[0] for v in per_r.values()), per_r, ok)
        # recommended: the invariant config with the least summed time over R (grid may vary)
        inv = [(v[0], c, v[1]) for c, v in agg.items() if v[2]]
        inv.sort()
        if inv:
            _, cfg, per_r = inv[0]
            best_cfgs[(n, k)] = (cfg, per_r)
            desc = " ".join(
                f"R{r}={per_r[r][0]:.1f}us/{nbytes / (per_r[r][0] * 1e-6) / 1e9:.0f}GBps(g{per_r[r][1]})"
                for r in ROWS
            )
            out(f"RESULT mr_best shape={name} N={n} K={k} cfg(BN,BK,warps,mode,ksplit,stages)={cfg} {desc}")
        else:
            out(f"RESULT mr_best shape={name} none (no invariant config completed)")
        # the current pair kernel's invariance (reference)
        check_invariance_cur(torch, ws[0], k, n, cur_launch, cur_tiles, get_x, name)

    # ------------------------------------------------------------ summary tables
    out("RESULT ---- bandwidth table (GB/s of weight bytes; us) ----")
    labels_fixed = ["cur-pairs-g2", "cur-fixed-g2", "cur-rows-g2", "cur-pairs-g4", "cur-pairs-g8",
                    "blas-linear", "blas-matmul"]
    for name, n, k in distinct:
        nbytes = n * k * 2
        for r in ROWS:
            row = results[(name, r)]
            cells = []
            for lab in labels_fixed:
                if lab in row:
                    cells.append(f"{lab}={nbytes / (row[lab] * 1e-6) / 1e9:.0f}/{row[lab]:.1f}")
            if (n, k) in best_cfgs:
                us = best_cfgs[(n, k)][1][r][0]
                cells.append(f"mr-best={nbytes / (us * 1e-6) / 1e9:.0f}/{us:.1f}")
            mrs = [v for lab, v in row.items() if lab.startswith("mr-")]
            if mrs:
                us = min(mrs)
                cells.append(f"mr-any={nbytes / (us * 1e-6) / 1e9:.0f}/{us:.1f}")
            out(f"RESULT table shape={name} R={r} " + " ".join(cells))

    # Step-level: verify B=2 = 4 rows through every dense GEMV as the engine runs them.
    shape_of = {(d[1], d[2]): d[0] for d in distinct}
    for r in (4, 2, 8):
        tot_bytes = cur_us = mr_us = blas_us = 0.0
        for name, n, k, uses, fam in SHAPES:
            key = shape_of[(n, k)]
            row = results[(key, r)]
            eng = row.get("cur-pairs-g2") if fam == "fused" else row.get("blas-linear")
            if eng is None or (n, k) not in best_cfgs:
                continue
            tot_bytes += uses * n * k * 2
            cur_us += uses * eng
            mr_us += uses * best_cfgs[(n, k)][1][r][0]
            blas_us += uses * row.get("blas-linear", float("nan"))
            out(f"RESULT step_gemv R={r} {name} x{uses} engine={eng:.1f}us mr={best_cfgs[(n, k)][1][r][0]:.1f}us "
                f"blas={row.get('blas-linear', float('nan')):.1f}us")
        if cur_us:
            out(f"RESULT step_dense R={r} bytes={tot_bytes / 1e9:.2f}GB engine={cur_us / 1e3:.2f}ms "
                f"({tot_bytes / (cur_us * 1e-6) / 1e12:.2f}TB/s) mr={mr_us / 1e3:.2f}ms "
                f"({tot_bytes / (mr_us * 1e-6) / 1e12:.2f}TB/s) blas={blas_us / 1e3:.2f}ms "
                f"({tot_bytes / (blas_us * 1e-6) / 1e12:.2f}TB/s) saving={(cur_us - mr_us) / 1e3:.2f}ms")

    # Kill rule.
    r = 4
    tot = sum(u * n * k * 2 for _, n, k, u, _ in SHAPES)
    cur = sum(u * (results[(shape_of[(n, k)], r)].get("cur-pairs-g2" if f == "fused" else "blas-linear") or 0)
              for _, n, k, u, f in SHAPES)
    cur_tbs = tot / (cur * 1e-6) / 1e12 if cur else float("nan")

    def mr_tbs(n, k):
        if (n, k) not in best_cfgs:
            return 0.0
        return n * k * 2 / (best_cfgs[(n, k)][1][r][0] * 1e-6) / 1e12

    ip, lm = mr_tbs(12352, 2048), mr_tbs(248320, 2048)
    a = cur_tbs >= 0.9
    b = ip < 0.9 or lm < 0.9
    out(f"RESULT kill current_R4={cur_tbs:.2f}TB/s(>=0.9 kills: {a}) mr_R4 in_proj={ip:.2f} "
        f"lm_head={lm:.2f}TB/s(<0.9 on either kills: {b}) -> {'KILL' if (a or b) else 'GO'}")


def check_invariance(torch, w, k, n, cfg, mr_launch, get_x, name) -> bool:
    dev = w.device
    x8 = get_x(8, k)
    singles = []
    for r in range(8):
        o = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
        part = torch.empty((cfg[4] * n,), dtype=torch.float32, device=dev)
        mr_launch(x8[r : r + 1].contiguous(), w, o, part, cfg, 4)
        singles.append(o)
    ref = (x8.float() @ w.float().t())
    ok = True
    worst = 0.0
    for rows in (2, 4, 8):
        for gpc in (1, 4, 0):
            o = torch.empty((rows, n), dtype=torch.bfloat16, device=dev)
            part = torch.empty((cfg[4] * rows * n,), dtype=torch.float32, device=dev)
            mr_launch(x8[:rows].contiguous(), w, o, part, cfg, gpc)
            for r in range(rows):
                if not torch.equal(o[r], singles[r][0]):
                    ok = False
            worst = max(worst, float((o.float() - ref[:rows]).abs().max()))
    torch.cuda.synchronize()
    out(f"RESULT invariance shape={name} kernel=mr cfg={cfg} bitwise_R1_vs_R2/4/8_all_grids={ok} "
        f"max_abs_err_vs_fp32={worst:.4g}")
    return ok


def check_invariance_cur(torch, w, k, n, cur_launch, cur_tiles, get_x, name) -> None:
    dev = w.device
    x8 = get_x(8, k)
    res = {}
    for variant in ("pairs", "fixed", "rows"):
        singles = []
        for r in range(8):
            o = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
            bp, bn, bk = cur_tiles(variant, 1, k)
            cur_launch(x8[r : r + 1].contiguous(), w, o, bp, bn, bk, 2)
            singles.append(o)
        ok = True
        for rows in (2, 4, 8):
            o = torch.empty((rows, n), dtype=torch.bfloat16, device=dev)
            bp, bn, bk = cur_tiles(variant, rows, k)
            cur_launch(x8[:rows].contiguous(), w, o, bp, bn, bk, 2)
            ok = ok and all(torch.equal(o[r], singles[r][0]) for r in range(rows))
        res[variant] = ok
    import torch.nn.functional as F

    singles = [F.linear(x8[r : r + 1], w) for r in range(8)]
    ok = all(torch.equal(F.linear(x8[:rows], w)[r], singles[r][0]) for rows in (2, 4, 8) for r in range(rows))
    out(f"RESULT invariance shape={name} cur_pairs={res['pairs']} cur_fixed={res['fixed']} "
        f"cur_rows={res['rows']} blas_linear={ok}")


# ============================================================================ part 2


def step_part(torch, args) -> None:
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile, record_function

    from engine.fast import fused as fused_mod
    from engine.generator import CHECKPOINT, Generator

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    started = time.perf_counter()
    g = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
        prefix_cache_tokens=0, prefix_cache_entries=0,
    )
    fast = g.fast
    out(f"RESULT step_load_s={time.perf_counter() - started:.0f} gpus={torch.cuda.device_count()} "
        f"mtp={fast.mtp is not None} graphs={fast.use_graphs} fused={fast.fused is not None if hasattr(fast, 'fused') else '?'}")
    fz = fast.fused
    # actual weight shapes (to confirm part 1's)
    i_gdn = next(i for i in range(len(fast.layers)) if fz._is_gdn(i))
    i_att = next(i for i in range(len(fast.layers)) if not fz._is_gdn(i))
    out(f"RESULT shapes in_proj={tuple(fz.w_in[i_gdn].shape)} "
        f"out_proj={tuple(fast.layers[i_gdn].linear_attn.out_proj.weight.shape)} "
        f"qkv={tuple(fz.w_in[i_att].shape)} o_proj={tuple(fast.layers[i_att].self_attn.o_proj.weight.shape)} "
        f"router_shared={tuple(fz.w_moe[i_gdn].shape)} lm_head={tuple(fast.lm_head.shape)} "
        f"mtp_fc={tuple(fast.mtp.fc.weight.shape)} devices={sorted({str(s.device) for s in fast.vsegments})}")

    tok = g.tokenizer
    ids: list[int] = []
    i = 0
    while len(ids) < 4000:
        for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i}"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1

    def prefill(slot, prompt):
        fast.reset(slot)
        body = len(prompt) - 1
        pos = 0
        while pos < body:
            end = min((pos // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[pos:end], pos, prompt[end])
            pos = end
        return fast.prefill(slot, prompt[body:], body)

    plen = 1800
    slots = [0, 1]
    lasts = [int(torch.argmax(prefill(s, ids[s * 100 : s * 100 + plen]))) for s in slots]

    def step():
        fast.verify(lasts, [0, 0], slots, [plen, plen])

    for _ in range(5):
        step()
    sync()
    times = []
    for _ in range(args.iters):
        sync()
        t0 = time.perf_counter()
        step()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    out(f"RESULT step verify_B2_ms={statistics.median(times):.2f} (median of {args.iters}, graphs={fast.use_graphs}, error={fast.graph_error})")

    def table(prof, nsteps, tag):
        agg = defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            dt = getattr(e, "device_type", None)
            if dt is None or "CUDA" not in str(dt) and "HIP" not in str(dt):
                continue
            t = e.time_range.elapsed_us()
            dev_i = getattr(e, "device_index", -1)
            agg[(dev_i, e.name)][0] += t
            agg[(dev_i, e.name)][1] += 1
        total = sum(v[0] for v in agg.values()) / nsteps
        out(f"RESULT kernels {tag} total_gpu_us_per_step={total:.0f} kinds={len(agg)}")
        for (dev_i, name), (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:40]:
            out(f"RESULT kernels {tag} dev={dev_i} us_per_step={t / nsteps:8.1f} "
                f"share={t / nsteps / max(total, 1e-9) * 100:5.1f}% calls_per_step={c / nsteps:6.1f} name={name[:110]}")
        # grouped
        groups = defaultdict(float)
        for (dev_i, name), (t, c) in agg.items():
            low = name.lower()
            if "_gemv_kernel" in name:
                key = "fused GEMV (_gemv_kernel)"
            elif "cijk" in low or "gemm" in low or "gemv" in low or "hipblaslt" in low:
                key = "BLAS GEMM/GEMV"
            elif "_moe_" in name:
                key = "MoE (_moe_up/_moe_down)"
            elif "gdn" in low or "conv" in low:
                key = "GDN step/conv"
            elif "attn" in low:
                key = "attention"
            elif "copy" in low or "index" in low or "memcpy" in low or "elementwise" in low or "fill" in low:
                key = "copies/elementwise"
            else:
                key = "other"
            groups[key] += t / nsteps
        for key, t in sorted(groups.items(), key=lambda kv: -kv[1]):
            out(f"RESULT kernels {tag} group={key} us_per_step={t:.0f} share={t / max(total, 1e-9) * 100:.1f}%")
        return total

    nsteps = 5
    try:
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(nsteps):
                step()
            sync()
        table(prof, nsteps, "graph")
    except Exception:  # noqa: BLE001
        out("RESULT kernels graph error " + traceback.format_exc().replace("\n", " | "))

    # Eager, GEMVs labeled by weight shape and rows.
    orig_gemv = fused_mod.FusedDecode._gemv
    orig_linear = F.linear

    order: list[tuple[int, str]] = []

    def gemv(self, x, w, *a, **kw):
        order.append((x.device.index, f"{w.shape[0]}x{w.shape[1]} R{x.shape[0]}"))
        with record_function(f"GEMV fused {w.shape[0]}x{w.shape[1]} R{x.shape[0]}"):
            return orig_gemv(self, x, w, *a, **kw)

    def linear(x, w, *a, **kw):
        with record_function(f"GEMV blas {w.shape[0]}x{w.shape[1]} R{x.reshape(-1, x.shape[-1]).shape[0]}"):
            return orig_linear(x, w, *a, **kw)

    use_graphs = fast.use_graphs
    try:
        fused_mod.FusedDecode._gemv = gemv
        F.linear = linear
        fast.use_graphs = False
        for _ in range(3):
            step()
        sync()
        order.clear()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(nsteps):
                step()
            sync()
        total = table(prof, nsteps, "eager")
        # _gemv_kernel launches in issue order per device, matched to the wrapper's labels.
        evs = defaultdict(list)
        for e in prof.events():
            if "_gemv_kernel" in e.name and "CUDA" in str(getattr(e, "device_type", "")):
                evs[e.device_index].append((e.time_range.start, e.time_range.elapsed_us()))
        labs = defaultdict(list)
        for d, lab in order:
            labs[d].append(lab)
        by = defaultdict(lambda: [0.0, 0])
        for d in labs:
            ev = sorted(evs.get(d, []))
            out(f"RESULT gemv_order dev={d} labels={len(labs[d])} kernels={len(ev)}")
            if len(ev) == len(labs[d]):
                for lab, (_, us) in zip(labs[d], ev):
                    by[lab][0] += us
                    by[lab][1] += 1
        for lab, (us, c) in sorted(by.items(), key=lambda kv: -kv[1][0]):
            n, k = (int(v) for v in lab.split()[0].split("x"))
            per, calls = us / nsteps, c / nsteps
            gb = n * k * 2 * calls / 1e9
            out(f"RESULT gemv_split ordered fused {lab} calls_per_step={calls:.0f} us_per_step={per:.0f} "
                f"avg_us={us / max(c, 1):.1f} weightGB_per_step={gb:.3f} GBps={gb / max(per * 1e-6, 1e-12):.0f}")
        labeled = defaultdict(lambda: [0.0, 0])
        for e in prof.key_averages():
            if e.key.startswith("GEMV "):
                t = getattr(e, "device_time_total", None)
                if t is None:
                    t = getattr(e, "cuda_time_total", 0)
                labeled[e.key][0] += t
                labeled[e.key][1] += e.count
        gsum = 0.0
        for key, (t, c) in sorted(labeled.items(), key=lambda kv: -kv[1][0]):
            n, k = (int(v) for v in key.split()[2].split("x"))
            per = t / nsteps
            gsum += per
            calls = c / nsteps
            gb = n * k * 2 * calls / 1e9
            out(f"RESULT gemv_split eager {key} calls_per_step={calls:.0f} us_per_step={per:.0f} "
                f"weightGB_per_step={gb:.3f} GBps={gb / max(per * 1e-6, 1e-12):.0f}")
        out(f"RESULT gemv_split eager total_gemv_us_per_step={gsum:.0f} of total_gpu_us={total:.0f}")
    except Exception:  # noqa: BLE001
        out("RESULT kernels eager error " + traceback.format_exc().replace("\n", " | "))
    finally:
        fused_mod.FusedDecode._gemv = orig_gemv
        F.linear = orig_linear
        fast.use_graphs = use_graphs


if __name__ == "__main__":
    main()
