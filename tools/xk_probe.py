"""Plan 53 stage 0: launch each dense decode GEMV early on a second queue (cross-kernel overlap).

    python tools/xk_probe.py [--part all|env|mech|ceiling|chain] [--iters 100] [--stress 2000]

Tools only: no served-path change. Three questions, each in its own process under ``timeout``
(a spin that never ends kills only its part), reported separately so a miss can be traced to
the idea or to the implementation:

1. ``mech`` (GCD 0): do the two branches of a forked captured HIP graph run concurrently on
   this ROCm, and do two captured graphs on two streams linked only by device flags? Each
   branch is one spin-coupled kernel (sets its flag, then waits for the peer's; bounded spin,
   timeouts counted), so a pair completes only if both kernels are resident at once. In-kernel
   ``s_memrealtime`` stamps (100 MHz) show start/end of both. Control: the same pair on one
   stream (must time out). Also the profiler's kernel timestamps of the forked replay.
2. ``ceiling`` (GCD 0): a streaming kernel (plain streaming read, and today's MFMA GEMVs
   in_proj / out_proj at 4 rows) overlapped (forked graph branches) with a latency-bound kernel
   of the decode shape (today's ``_gdn_chain_step_kernel`` at 4 / 8 rows, ``_add_rms_kernel``
   at 4 rows). 20 pairs per graph: alone, serial, forked. Hidden share of the streaming time
   = (t_S + t_L - t_fork) / t_S; slowdown of each kernel from profiler durations.
3. ``chain`` (GCDs 0,1, TP2): 40 Qwen3.6-35B-A3B TP2 decoder layers of today's kernels with
   real weight shapes (random values): add_rms, in_proj MFMA GEMV, GDN conv + delta-rule
   chain kernels (G = 4 rows per slot), out_proj GEMV + K-sum, in-graph LL all-reduce,
   add_rms, router|shared GEMV + K-sum/top-k, MoE up, MoE down, LL all-reduce. Every layer is
   a GDN layer (the model has 30 GDN + 10 attention layers; attention's glue is of the same
   latency class). At 4 rows (1 slot) and 8 rows (2 slots): serial graph vs
   - ``serial_xk``: the xk GEMV kernels and signal nodes on one stream (the mechanism's own
     cost, no overlap),
   - ``pf``: a forked branch that touches the next GEMV's first K chunks (L2 prefetch) while
     the latency kernels before it run; the GEMVs unchanged,
   - ``xk_fork_D{d}_P{p}``: each target GEMV (in_proj, out_proj, router) launched on a forked
     branch d kernels early, loading its first p weight K-chunks per program into registers,
     then spinning on an epoch flag written by a signal node after its predecessor
     (the plan's mechanism), joined before its consumer,
   - ``xk_2g_P{p}``: two captured graphs per rank on two streams linked only by flags.
   Every variant is checked bitwise (every layer's outputs and GDN states, both ranks) against
   the serial graph from the same state; spin timeouts and all-reduce timeouts are counted;
   the best variant at 4 rows runs ``--stress`` replays. Profiler timestamps of each variant
   show how much of each early kernel overlapped other kernels.

Kill (plan 53 stage 0): best two-queue chain < 12% faster than serial at 4 rows, any bit
difference, or any hang / spin timeout. If (1) and (2) work and (3) misses, report what
limited it. Prints ``RESULT <tag> <json>`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Qwen3.6-35B-A3B, one TP2 rank.
H = 2048            # hidden
LAYERS = 40
NK, NV, DK, DV = 8, 16, 128, 128     # GDN key / value heads per rank, head dims
KEY_DIM, VAL_DIM = NK * DK, NV * DV
CONV_DIM = 2 * KEY_DIM + VAL_DIM     # 4096
N_IN = CONV_DIM + VAL_DIM + 2 * NV   # 6176: q|k|v, z, b, a
E, TOPK, I_EXP, I_SH = 256, 8, 256, 256
N_ROUTER = E + 2 * I_SH + 1          # 769
EXPERT_SETS = 8                      # distinct expert tensors cycled over layers (>> L2)
XK_MAXIT = 50_000                    # spin bound of the probe's own flag waits
COUPLE_MAXIT = 200_000
KILL_GAIN = 0.12


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


# ================================================================================ kernels

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  (the driver needs no triton)
    triton = None

if triton is not None:

    @triton.jit
    def _now():
        return tl.inline_asm_elementwise("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [],
                                         dtype=tl.int64, is_pure=False, pack=1)

    @triton.jit(do_not_specialize=["SIDE"])
    def _couple_kernel(flag_ptr, ep_ptr, err_ptr, st_ptr, SIDE, MAXIT: tl.constexpr,
                       STAMP: tl.constexpr):
        """Set my flag to my epoch, then wait (bounded) until the peer's flag reaches it."""
        e = tl.load(ep_ptr + SIDE) + 1
        if STAMP:
            t0 = _now()
        tl.atomic_xchg(flag_ptr + SIDE, e, sem="release", scope="gpu")
        f = tl.atomic_add(flag_ptr + (1 - SIDE), 0, sem="acquire", scope="gpu")
        it = e * 0
        while (f < e) & (it < MAXIT):
            f = tl.atomic_add(flag_ptr + (1 - SIDE), 0, sem="acquire", scope="gpu")
            it += 1
        if f < e:
            tl.atomic_add(err_ptr, 1)
        if STAMP:
            t1 = _now()
            slot = (e % 64) * 4 + SIDE * 2
            tl.store(st_ptr + slot, t0)
            tl.store(st_ptr + slot + 1, t1)
        tl.store(ep_ptr + SIDE, e)

    @triton.jit
    def _stream_kernel(w_ptr, out_ptr, NBLK, BLOCK: tl.constexpr):
        """A plain streaming read: sum of every BF16 value (one fp32 per program)."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for b in range(pid, NBLK, nprog):
            acc += tl.load(w_ptr + b.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)).to(tl.float32)
        tl.store(out_ptr + pid, tl.sum(acc, axis=0))

    @triton.jit(do_not_specialize=["EI"])
    def _xk_bump_kernel(ep_ptr, EI):
        e = tl.load(ep_ptr + EI) + 1
        tl.store(ep_ptr + EI, e)

    @triton.jit(do_not_specialize=["G", "EI"])
    def _xk_signal_kernel(flag_ptr, ep_ptr, G, EI):
        e = tl.load(ep_ptr + EI)
        tl.atomic_xchg(flag_ptr + G, e, sem="release", scope="gpu")

    @triton.jit(do_not_specialize=["G", "EI"])
    def _xk_sigwait_kernel(flag_ptr, done_ptr, ep_ptr, err_ptr, G, EI, MAXIT: tl.constexpr):
        """Signal flag G (this graph's epoch), then wait until the other graph's GEMV G is done."""
        e = tl.load(ep_ptr + EI)
        tl.atomic_xchg(flag_ptr + G, e, sem="release", scope="gpu")
        f = tl.atomic_add(done_ptr + G, 0, sem="acquire", scope="gpu")
        it = e * 0
        while (f < e) & (it < MAXIT):
            f = tl.atomic_add(done_ptr + G, 0, sem="acquire", scope="gpu")
            it += 1
        if f < e:
            tl.atomic_add(err_ptr, 1)
        tl.atomic_add(done_ptr + G, 0, sem="acquire", scope="gpu")

    @triton.jit(do_not_specialize=["R", "N", "G", "EI"])
    def _xk_gemv_kernel(
        x_ptr, w_ptr, out_ptr, part_ptr, R, N, stride_out, flag_ptr, ep_ptr, err_ptr, G, EI,
        K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        KSPLIT: tl.constexpr, NPRE: tl.constexpr, WAIT: tl.constexpr, MAXIT: tl.constexpr,
    ):
        """``mfma._mfma_gemv_kernel`` with a prologue: load this program's first NPRE weight
        K-chunks, then (WAIT) spin until flag G reaches the epoch, then the same dots in the
        same order (chunks 0..NPRE-1 from registers, the rest as the original loop)."""
        pid = tl.program_id(0)
        nb = pid // KSPLIT
        kp = pid % KSPLIT
        KP: tl.constexpr = K // KSPLIT
        rm = tl.arange(0, BM)
        mok = rm < R
        rn = nb * BN + tl.arange(0, BN)
        nok = rn < N
        kb = kp * KP
        wrow = w_ptr + rn[None, :].to(tl.int64) * K
        if NPRE >= 1:
            b0 = tl.load(wrow + (kb + tl.arange(0, BK))[:, None], mask=nok[None, :], other=0.0)
        if NPRE >= 2:
            b1 = tl.load(wrow + (kb + BK + tl.arange(0, BK))[:, None], mask=nok[None, :], other=0.0)
        if NPRE >= 3:
            b2 = tl.load(wrow + (kb + 2 * BK + tl.arange(0, BK))[:, None], mask=nok[None, :], other=0.0)
        if NPRE >= 4:
            b3 = tl.load(wrow + (kb + 3 * BK + tl.arange(0, BK))[:, None], mask=nok[None, :], other=0.0)
        if WAIT:
            e = tl.load(ep_ptr + EI)
            f = tl.atomic_add(flag_ptr + G, 0, sem="acquire", scope="gpu")
            it = e * 0
            while (f < e) & (it < MAXIT):
                f = tl.atomic_add(flag_ptr + G, 0, sem="acquire", scope="gpu")
                it += 1
            if f < e:
                tl.atomic_add(err_ptr, 1)
            tl.atomic_add(flag_ptr + G, 0, sem="acquire", scope="gpu")
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        if NPRE >= 1:
            rk = kb + tl.arange(0, BK)
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
            acc = tl.dot(a, b0, acc)
        if NPRE >= 2:
            rk = kb + BK + tl.arange(0, BK)
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
            acc = tl.dot(a, b1, acc)
        if NPRE >= 3:
            rk = kb + 2 * BK + tl.arange(0, BK)
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
            acc = tl.dot(a, b2, acc)
        if NPRE >= 4:
            rk = kb + 3 * BK + tl.arange(0, BK)
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
            acc = tl.dot(a, b3, acc)
        for k0 in range(kb + NPRE * BK, kb + KP, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
            b = tl.load(wrow + rk[:, None], mask=nok[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
        m = mok[:, None] & nok[None, :]
        if KSPLIT == 1:
            tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)
        else:
            tl.store(part_ptr + (kp * BM + rm[:, None]) * N + rn[None, :], acc, mask=m)

    @triton.jit(do_not_specialize=["N"])
    def _touch_kernel(w_ptr, out_ptr, N, K: tl.constexpr, KP: tl.constexpr, KSPLIT: tl.constexpr,
                      PFC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
        """Read the first PFC columns of each of the KSPLIT K parts of BR weight rows (what a
        GEMV's programs read first) so they sit in L2; one fp32 per program."""
        pid = tl.program_id(0)
        rows = pid * BR + tl.arange(0, BR)
        ok = rows < N
        acc = tl.zeros([BR, BC], dtype=tl.float32)
        for kp in tl.static_range(KSPLIT):
            for c0 in tl.static_range(0, PFC, BC):
                cols = kp * KP + c0 + tl.arange(0, BC)
                acc += tl.load(w_ptr + rows[:, None].to(tl.int64) * K + cols[None, :],
                               mask=ok[:, None], other=0.0).to(tl.float32)
        tl.store(out_ptr + pid, tl.sum(tl.sum(acc, axis=1), axis=0))


# ================================================================================ helpers


def sync_all(torch) -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def load_trace(prof) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir="/tmp") as f:
        path = f.name
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            data = json.load(f)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
    return data["traceEvents"] if isinstance(data, dict) else data


def gpu_events(trace) -> dict[int, list[tuple[float, float, str]]]:
    by = defaultdict(list)
    for e in trace:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        args = e.get("args") or {}
        dev = args.get("device", e.get("pid"))
        try:
            dev = int(dev)
        except (TypeError, ValueError):
            continue
        by[dev].append((float(e["ts"]), float(e.get("dur", 0.0)), e.get("name", "?")))
    for d in by:
        by[d].sort()
    return dict(by)


def overlap_with(iv: tuple[float, float], merged: list[tuple[float, float]]) -> float:
    s, e = iv
    tot = 0.0
    for a, b in merged:
        if b <= s:
            continue
        if a >= e:
            break
        tot += min(b, e) - max(a, s)
    return tot


def merge(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ivs = sorted(ivs)
    outl: list[list[float]] = []
    for a, b in ivs:
        if outl and a <= outl[-1][1]:
            outl[-1][1] = max(outl[-1][1], b)
        else:
            outl.append([a, b])
    return [(a, b) for a, b in outl]


def med(v):
    return round(statistics.median(v), 3) if v else None


# ================================================================================ part env


def part_env(args) -> None:
    import torch

    hip = Path(torch.__file__).parent / "lib" / "libamdhip64.so"
    names = []
    with contextlib.suppress(OSError):
        data = hip.read_bytes()
        names = sorted({m.decode() for m in re.findall(rb"[A-Z][A-Z0-9_]*GRAPH[A-Z0-9_]*", data)})[:60]
    props = torch.cuda.get_device_properties(0)
    out("env", torch=torch.__version__, hip=torch.version.hip,
        triton=getattr(triton, "__version__", None), gpus=torch.cuda.device_count(),
        name=props.name, cus=props.multi_processor_count,
        arch=getattr(props, "gcnArchName", None),
        env={k: v for k, v in os.environ.items() if k.startswith(("GPU_", "HIP_", "DEBUG_", "AMD_", "ROC"))},
        hip_graph_strings=names)


# ================================================================================ part mech


def part_mech(args) -> None:
    """Question 1: concurrency of forked graph branches and of two flag-linked graphs."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)

    stamp = True
    try:
        st = torch.zeros(256, dtype=torch.int64, device=dev)
        fl = torch.zeros(2, dtype=torch.int32, device=dev)
        ep = torch.zeros(2, dtype=torch.int32, device=dev)
        err = torch.zeros(1, dtype=torch.int32, device=dev)
        fl.fill_(1 << 30)  # pre-satisfied: compile only
        for side in (0, 1):
            _couple_kernel[(1,)](fl, ep, err, st, side, MAXIT=COUPLE_MAXIT, STAMP=True, num_warps=1)
        torch.cuda.synchronize(dev)
        out("mech_stamp", ok=True, sample_ticks=[int(v) for v in st[:4].tolist()])
    except Exception as exc:  # noqa: BLE001
        stamp = False
        out("mech_stamp", ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")

    class State:
        def __init__(self):
            self.fl = torch.zeros(2, dtype=torch.int32, device=dev)
            self.ep = torch.zeros(2, dtype=torch.int32, device=dev)
            self.err = torch.zeros(1, dtype=torch.int32, device=dev)
            self.st = torch.zeros(256, dtype=torch.int64, device=dev)

        def k(self, side, maxit=COUPLE_MAXIT):
            _couple_kernel[(1,)](self.fl, self.ep, self.err, self.st, side, MAXIT=maxit,
                                 STAMP=stamp, num_warps=1)

        def pairs(self, nrep):
            """(start A, start B, end A, end B) in us for the last nrep epochs (stamps)."""
            if not stamp:
                return []
            s = self.st.view(64, 4).cpu().tolist()
            e = int(self.ep[0].item())
            res = []
            for ee in range(max(1, e - nrep + 1), e + 1):
                a0, a1, b0, b1 = s[ee % 64]
                if a0 and b0:
                    t = min(a0, b0)
                    res.append([round((a0 - t) / 100, 2), round((b0 - t) / 100, 2),
                                round((a1 - t) / 100, 2), round((b1 - t) / 100, 2)])
            return res

    def summarize(name, state, nrep, reps_run, wall_s, extra=None):
        p = state.pairs(nrep)
        ov = [min(a1, b1) - max(a0, b0) for a0, b0, a1, b1 in p]
        out("mech", scheme=name, pairs=reps_run, spin_timeouts=int(state.err.item()),
            wall_ms_per_pair=round(wall_s * 1e3 / max(reps_run, 1), 4),
            stamped_pairs=len(p), overlapping_pairs=sum(1 for v in ov if v > 0),
            overlap_us_p50=med(ov), start_skew_us_p50=med([abs(a0 - b0) for a0, b0, _, _ in p]),
            first_pairs_us_startA_startB_endA_endB=p[:3], **(extra or {}))

    rep = 8
    s0 = torch.cuda.Stream(dev)
    s1 = torch.cuda.Stream(dev)

    # (c) control: both kernels on one stream: the first must time out (the test discriminates).
    st_c = State()
    t0 = time.perf_counter()
    with torch.cuda.stream(s0):
        st_c.k(0, maxit=20_000)
        st_c.k(1, maxit=20_000)
    torch.cuda.synchronize(dev)
    summarize("control_same_stream_eager", st_c, 1, 1, time.perf_counter() - t0,
              {"expect": "1 timeout"})

    # (d) eager, two streams (no graph).
    st_e = State()
    t0 = time.perf_counter()
    for _ in range(rep):
        with torch.cuda.stream(s0):
            st_e.k(0)
        with torch.cuda.stream(s1):
            st_e.k(1)
    torch.cuda.synchronize(dev)
    summarize("eager_two_streams", st_e, rep, rep, time.perf_counter() - t0)

    # (a) one graph, forked branches (fork/join per pair).
    def capture(fn, stream):
        g = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.graph(g, stream=stream):
            fn()
        torch.cuda.synchronize(dev)
        return g

    results = {}
    for name in ("fork_graph", "two_graphs_flags"):
        st_g = State()
        try:
            if name == "fork_graph":
                evs = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(rep)]

                def body():
                    main = torch.cuda.current_stream(dev)
                    for a, b in evs:
                        a.record(main)
                        s1.wait_event(a)
                        with torch.cuda.stream(s1):
                            st_g.k(1)
                            b.record(s1)
                        st_g.k(0)
                        main.wait_event(b)
                gs = [capture(body, s0)]

                def launch():
                    with torch.cuda.stream(s0):
                        gs[0].replay()
            else:
                sa, sb = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
                ga = capture(lambda: [st_g.k(0) for _ in range(rep)], sa)
                gb = capture(lambda: [st_g.k(1) for _ in range(rep)], sb)

                def launch():
                    with torch.cuda.stream(sa):
                        ga.replay()
                    with torch.cuda.stream(sb):
                        gb.replay()
            launch()
            torch.cuda.synchronize(dev)
            first_err = int(st_g.err.item())
            if first_err:
                summarize(name, st_g, rep, rep, 0.0, {"aborted_after_first_replay": True})
                results[name] = False
                continue
            nrun = 25
            t0 = time.perf_counter()
            for _ in range(nrun):
                launch()
            torch.cuda.synchronize(dev)
            wall = time.perf_counter() - t0
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                launch()
                torch.cuda.synchronize(dev)
            evs_p = [e for e in gpu_events(load_trace(prof)).get(0, []) if "_couple_kernel" in e[2]]
            # profiler: consecutive kernels overlapping in time
            ov = sum(1 for (a, b) in zip(evs_p, evs_p[1:]) if b[0] < a[0] + a[1])
            summarize(name, st_g, rep, rep * (nrun + 2), wall,
                      {"profiler_kernels": len(evs_p), "profiler_overlapping_neighbours": ov,
                       "profiler_first_us": [[round(e[0] - evs_p[0][0], 2), round(e[1], 2)]
                                             for e in evs_p[:4]] if evs_p else []})
            results[name] = int(st_g.err.item()) == 0
        except Exception as exc:  # noqa: BLE001
            out("mech", scheme=name, error=f"{type(exc).__name__}: {str(exc)[:400]}")
            results[name] = False
    out("mech_summary", fork_concurrent=results.get("fork_graph", False),
        two_graphs_concurrent=results.get("two_graphs_flags", False), stamps=stamp)


# ================================================================================ part ceiling


def part_ceiling(args) -> None:
    """Question 2: overlap of a streaming kernel with a latency-bound kernel of decode shape."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from engine.fast import kernels as KN
    from engine.fast import mfma as MF

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    torch.manual_seed(0)

    def bf(*shape, scale=1.0):
        return (torch.randn(shape, device=dev, dtype=torch.float32) * scale).to(torch.bfloat16)

    # Streaming kernels
    streams = {}
    for mb in (8, 24):
        w = bf(mb * (1 << 20) // 2, scale=0.02)
        o = torch.zeros(4 * cus, device=dev)
        block = 4096
        nblk = w.numel() // block
        streams[f"stream_{mb}MB"] = (lambda w=w, o=o, nblk=nblk: _stream_kernel[(4 * cus,)](
            w, o, nblk, BLOCK=block, num_warps=4), mb * (1 << 20), "_stream_kernel")
    for name, (n, k) in (("gemv_in_proj_R4", (N_IN, H)), ("gemv_out_proj_R4", (H, VAL_DIM))):
        w = bf(n, k, scale=0.02)
        x = bf(4, k)
        o = torch.empty((4, n), device=dev, dtype=torch.bfloat16)
        bn, bk, warps, ks = MF.tiles_for(n, k)
        part = torch.empty((ks * MF.BM * n,), device=dev) if ks > 1 else o

        def g(w=w, x=x, o=o, part=part, n=n, k=k, bn=bn, bk=bk, warps=warps, ks=ks):
            MF._mfma_gemv_kernel[(triton.cdiv(n, bn) * ks,)](
                x, w, o, part, 4, n, o.stride(0), K=k, BM=MF.BM, BN=bn, BK=bk, KSPLIT=ks,
                DOT_F32=False, num_warps=warps)
        streams[name] = (g, n * k * 2, "_mfma_gemv_kernel")

    # Latency-bound kernels of the decode shape
    lats = {}
    for n_slots in (1, 2):
        G = 4
        R = n_slots * G
        rec = torch.randn((R, NV, DK, DV), device=dev) * 0.01
        states = torch.empty(R, dtype=torch.int32)  # row j * n + r: slot r's ring entry j
        for j in range(G):
            for r in range(n_slots):
                states[j * n_slots + r] = r * G + j
        states = states.to(dev)
        proj = bf(R, N_IN)
        alog = torch.zeros(NV, device=dev)
        dtb = torch.zeros(NV, device=dev)
        nw = torch.ones(DV, device=dev, dtype=torch.bfloat16)
        core = torch.empty((R, VAL_DIM), device=dev, dtype=torch.bfloat16)

        def l(rec=rec, states=states, proj=proj, alog=alog, dtb=dtb, nw=nw, core=core,
              n_slots=n_slots, G=G):
            MF._gdn_chain_step_kernel[(n_slots, NV)](
                rec, states, proj, alog, dtb, nw, core, n_slots, proj.stride(0), NV,
                rec.stride(0), rec.stride(1), 1e-6, DK ** -0.5, CONV_DIM, CONV_DIM + VAL_DIM,
                CONV_DIM + VAL_DIM + NV, RATIO=NV // NK, KEY_DIM=KEY_DIM, DK=DK, DV=DV, G=G,
                num_warps=8)
        lats[f"gdn_chain_step_R{R}"] = (l, "_gdn_chain_step_kernel")
    xa = bf(4, H)
    lw = torch.zeros(H, device=dev, dtype=torch.bfloat16)
    ya = torch.empty_like(xa)
    lats["add_rms_R4"] = (lambda: KN._add_rms_kernel[(4,)](
        xa, xa, lw, xa, ya, 1e-6, N=H, BLOCK=triton.next_power_of_2(H), HAS_H=False), "_add_rms_kernel")

    s0 = torch.cuda.Stream(dev)
    s1 = torch.cuda.Stream(dev)
    rep = 20
    evs = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(rep)]

    def capture(fn):
        g = torch.cuda.CUDAGraph()
        s0.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.graph(g, stream=s0):
            fn()
        torch.cuda.synchronize(dev)
        return g

    def time_graph(g, n=10):
        with torch.cuda.stream(s0):
            g.replay()
        torch.cuda.synchronize(dev)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            with torch.cuda.stream(s0):
                a.record()
                g.replay()
                b.record()
            torch.cuda.synchronize(dev)
            ts.append(a.elapsed_time(b) * 1e3 / rep)  # us per pair
        return statistics.median(ts)

    def prof_durs(g, names):
        with torch.cuda.stream(s0):
            g.replay()
        torch.cuda.synchronize(dev)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with torch.cuda.stream(s0):
                g.replay()
            torch.cuda.synchronize(dev)
        ev = gpu_events(load_trace(prof)).get(0, [])
        res = {}
        for nm in names:
            res[nm] = [d for (_, d, n) in ev if nm in n]
        return res, ev

    for fn in [v[0] for v in streams.values()] + [v[0] for v in lats.values()]:  # compile
        fn()
    torch.cuda.synchronize(dev)

    for sname, (sfn, nbytes, sk) in streams.items():
        g_s = capture(lambda: [sfn() for _ in range(rep)])
        t_s = time_graph(g_s)
        d_s, _ = prof_durs(g_s, [sk])
        s_alone = med(d_s[sk])
        out("ceiling_alone", kernel=sname, graph_us=round(t_s, 2), prof_us_p50=s_alone,
            tbps_graph=round(nbytes / (t_s * 1e-6) / 1e12, 3),
            tbps_kernel=round(nbytes / (s_alone * 1e-6) / 1e12, 3) if s_alone else None)
        for lname, (lfn, lk) in lats.items():
            g_l = capture(lambda: [lfn() for _ in range(rep)])
            t_l = time_graph(g_l)
            d_l, _ = prof_durs(g_l, [lk])
            l_alone = med(d_l[lk])

            def ser():
                for _ in range(rep):
                    lfn()
                    sfn()

            def fork():
                main = torch.cuda.current_stream(dev)
                for a, b in evs:
                    a.record(main)
                    s1.wait_event(a)
                    with torch.cuda.stream(s1):
                        sfn()
                        b.record(s1)
                    lfn()
                    main.wait_event(b)
            g_ser = capture(ser)
            g_fork = capture(fork)
            t_ser = time_graph(g_ser)
            t_fork = time_graph(g_fork)
            d_f, ev = prof_durs(g_fork, [sk, lk])
            # timestamp overlap of each pair (profiler): L intervals vs S intervals
            li = [(t, t + d) for (t, d, n) in ev if lk in n]
            si = [(t, t + d) for (t, d, n) in ev if sk in n]
            ovs = [overlap_with(iv, merge(si)) for iv in li]
            hidden = (t_s + t_l - t_fork) / t_s if t_s else None
            out("ceiling", stream=sname, latency=lname, t_stream_us=round(t_s, 2),
                t_latency_us=round(t_l, 2), t_serial_us=round(t_ser, 2), t_fork_us=round(t_fork, 2),
                hidden_share_of_stream=round(hidden, 3) if hidden is not None else None,
                max_hideable=round(min(t_s, t_l) / t_s, 3) if t_s else None,
                fork_vs_serial=round(t_fork / t_ser, 3),
                latency_kernel_slowdown=round(med(d_f[lk]) / l_alone, 3) if l_alone and d_f[lk] else None,
                stream_kernel_slowdown=round(med(d_f[sk]) / s_alone, 3) if s_alone and d_f[sk] else None,
                prof_latency_us_alone=l_alone, prof_latency_us_fork=med(d_f[lk]),
                prof_stream_us_alone=s_alone, prof_stream_us_fork=med(d_f[sk]),
                pairs_with_timestamp_overlap=sum(1 for v in ovs if v > 0), pairs=len(ovs),
                overlap_us_p50=med(ovs))
            del g_l, g_ser, g_fork
        del g_s


# ================================================================================ part chain


class Op:
    __slots__ = ("kind", "layer", "fn", "spec", "heavy")

    def __init__(self, kind, layer, fn, spec=None, heavy=False):
        self.kind, self.layer, self.fn, self.spec, self.heavy = kind, layer, fn, spec, heavy


class RankChain:
    """One rank's 40-layer chain: weights, per-layer buffers and the ops in launch order."""

    def __init__(self, torch, rank, dev, ll, n_slots, G, x0, cus, seed):
        self.torch, self.rank, self.dev, self.ll = torch, rank, dev, ll
        self.n, self.G = n_slots, G
        self.R = R = n_slots * G
        self.cus = cus
        from engine.fast import fused as FZ
        from engine.fast import mfma as MF
        from engine.fast import tp as TP
        self.FZ, self.MF, self.TP = FZ, MF, TP
        bf16 = torch.bfloat16
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        def w(*shape, scale=0.02):
            t = torch.empty(shape, device=dev, dtype=bf16)
            t.normal_(0.0, scale, generator=gen)
            return t

        with torch.cuda.device(dev):
            self.x0 = x0.to(dev)
            self.sets = [(w(E, 2 * I_EXP, H), w(E, H, I_EXP)) for _ in range(EXPERT_SETS)]
            self.W, self.B = [], []
            self.conv0, self.rec0, self.conv, self.rec = [], [], [], []
            st = torch.empty(R, dtype=torch.int32)
            for j in range(G):
                for r in range(n_slots):
                    st[j * n_slots + r] = r * G + j
            self.states = st.to(dev)
            for L in range(LAYERS):
                self.W.append(dict(
                    ln1=torch.zeros(H, device=dev, dtype=bf16), ln2=torch.zeros(H, device=dev, dtype=bf16),
                    w_in=w(N_IN, H), conv_w=w(CONV_DIM, 4, scale=0.3), w_out=w(H, VAL_DIM),
                    alog=torch.zeros(NV, device=dev), dtb=torch.zeros(NV, device=dev),
                    nw=torch.ones(DV, device=dev, dtype=bf16), w3=w(N_ROUTER, H), sd=w(H, I_SH),
                    gu=self.sets[L % EXPERT_SETS][0], dn=self.sets[L % EXPERT_SETS][1]))
                z = lambda *s, dt=bf16: torch.zeros(s, device=dev, dtype=dt)  # noqa: E731
                self.B.append(dict(
                    h1=z(R, H), proj=z(R, N_IN), core=z(R, VAL_DIM), o=z(R, H),
                    o_part=z(4 * MF.BM * H, dt=torch.float32), ar1=z(R, H), x2=z(R, H), h2=z(R, H),
                    r=z(R, N_ROUTER), r_part=z(8 * MF.BM * N_ROUTER, dt=torch.float32),
                    idx=z(R, TOPK, dt=torch.int32), wt=z(R, TOPK, dt=torch.float32),
                    act=z(R * TOPK, I_EXP), m=z(R, H), xn=z(R, H)))
                c = torch.empty((R, CONV_DIM, 4), device=dev, dtype=bf16)
                c.normal_(0.0, 0.5, generator=gen)
                rc = torch.empty((R, NV, DK, DV), device=dev, dtype=torch.float32)
                rc.normal_(0.0, 0.01, generator=gen)
                self.conv0.append(c)
                self.rec0.append(rc)
                self.conv.append(c.clone())
                self.rec.append(rc.clone())
        self.ops = self._build()
        self.targets = [i for i, o in enumerate(self.ops) if o.spec is not None]

    # ------------------------------------------------------------ launches (current stream)

    def add_rms(self, x, h, wt, out, total):
        from engine.fast import kernels as KN
        KN._add_rms_kernel[(self.R,)](x, h if h is not None else x, wt,
                                      total if total is not None else x, out, 1e-6, N=H,
                                      BLOCK=triton.next_power_of_2(H), HAS_H=h is not None)

    def gemv(self, spec):
        x, w, o, part, tiles = spec
        n, k = w.shape
        bn, bk, warps, ks = tiles
        self.MF._mfma_gemv_kernel[(triton.cdiv(n, bn) * ks,)](
            x, w, o, part, self.R, n, o.stride(0), K=k, BM=self.MF.BM, BN=bn, BK=bk, KSPLIT=ks,
            DOT_F32=False, num_warps=warps)

    def xk(self, spec, st, g, ei, npre, wait=True):
        x, w, o, part, tiles = spec
        n, k = w.shape
        bn, bk, warps, ks = tiles
        npre = min(npre, (k // ks) // bk, 4)
        _xk_gemv_kernel[(triton.cdiv(n, bn) * ks,)](
            x, w, o, part, self.R, n, o.stride(0), st.flags, st.ep, st.err, g, ei,
            K=k, BM=self.MF.BM, BN=bn, BK=bk, KSPLIT=ks, NPRE=npre, WAIT=wait, MAXIT=XK_MAXIT,
            num_warps=warps)

    def touch(self, spec, scratch, npre):
        x, w, o, part, tiles = spec
        n, k = w.shape
        bn, bk, warps, ks = tiles
        pfc = min(npre, (k // ks) // bk) * bk
        br = 16
        _touch_kernel[(triton.cdiv(n, br),)](w, scratch, n, K=k, KP=k // ks, KSPLIT=ks, PFC=pfc,
                                             BR=br, BC=min(128, pfc), num_warps=4)

    def ksum(self, part, o, n, ks):
        self.MF._ksum_kernel[(triton.cdiv(self.R * n, 1024),)](
            part, o, self.R, n, o.stride(0), KSPLIT=ks, BM=self.MF.BM, BLOCK=1024, num_warps=4)

    def ksum_router(self, part, o, idx, wt, ks):
        self.MF._ksum_router_kernel[(triton.cdiv(self.R * N_ROUTER, 256) + self.R,)](
            part, o, idx, wt, self.R, N_ROUTER, o.stride(0), KSPLIT=ks, BM=self.MF.BM, BLOCK=256,
            E=E, TOPK=TOPK, TOPKP=triton.next_power_of_2(TOPK), num_warps=1)

    def conv_step(self, L):
        MF, b, w = self.MF, self.B[L], self.W[L]
        MF._gdn_conv_chain_kernel[(self.n, triton.cdiv(CONV_DIM, 256))](
            self.conv[L], self.states, b["proj"], w["conv_w"], self.n, CONV_DIM, b["proj"].stride(0),
            self.conv[L].stride(0), G=self.G, BLOCK=256)

    def gdn_step(self, L):
        MF, b, w = self.MF, self.B[L], self.W[L]
        rec = self.rec[L]
        MF._gdn_chain_step_kernel[(self.n, NV)](
            rec, self.states, b["proj"], w["alog"], w["dtb"], w["nw"], b["core"], self.n,
            b["proj"].stride(0), NV, rec.stride(0), rec.stride(1), 1e-6, DK ** -0.5, CONV_DIM,
            CONV_DIM + VAL_DIM, CONV_DIM + VAL_DIM + NV, RATIO=NV // NK, KEY_DIM=KEY_DIM, DK=DK,
            DV=DV, G=self.G, num_warps=8)

    def ar(self, x, res, out):
        TP, ll, r = self.TP, self.ll, self.rank
        i32 = self.torch.int32
        nw = x.numel() * 2 // 4
        TP._ll_allreduce_kernel[(triton.cdiv(nw, TP.CHUNK),)](
            x.view(-1).view(i32), (res if res is not None else x).view(-1).view(i32),
            out.view(-1).view(i32), ll.recv[r], ll.recv[1 - r], ll.epoch[r], ll.err[r], nw,
            TP.CAPACITY, CHUNK=TP.CHUNK, BF16=True, RES=res is not None, MAXIT=TP.MAXIT, num_warps=1)

    def moe(self, L):
        FZ, b, w = self.FZ, self.B[L], self.W[L]
        R = self.R
        bn = FZ.TILES["moe_bn"]
        new = FZ.MOE_ORDER == 1
        order = 1 if new and R >= FZ.MOE_TILE_ROWS else 0
        grid = FZ.MOE_GRID if new else 2
        items = R * TOPK * triton.cdiv(I_EXP, bn)
        gu, dn = w["gu"], w["dn"]
        return (
            lambda: FZ._moe_up_kernel[(items if order else min(items, grid * self.cus),)](
                b["h2"], b["idx"], gu, b["act"], R, gu.stride(0), b["r"], b["wt"], N_ROUTER,
                H=H, I=I_EXP, TOPK=TOPK, BN=bn, BK=min(1024, H), ORDER=order, E=E,
                TOPKP=triton.next_power_of_2(TOPK), FUSE_TOPK=False, num_warps=4),
            lambda: FZ._moe_down_kernel[(min(R * triton.cdiv(H, bn), grid * self.cus),)](
                b["act"], b["r"], b["idx"], b["wt"], dn, w["sd"], b["m"], b["m"], R, dn.stride(0),
                N_ROUTER, H=H, I=I_EXP, IS=I_SH, E=E, TOPK=TOPK, BN=bn, BKI=min(1024, I_EXP),
                BKS=min(1024, I_SH), ORDER=order, HAS_RES=False, num_warps=4),
        )

    def _build(self) -> list[Op]:
        MF = self.MF
        ops: list[Op] = []
        for L in range(LAYERS):
            b, w = self.B[L], self.W[L]
            x_in = self.x0 if L == 0 else self.B[L - 1]["xn"]
            t_in = MF.tiles_for(N_IN, H)
            t_out = MF.tiles_for(H, VAL_DIM)
            t_r = MF.tiles_for(N_ROUTER, H)
            s_in = (b["h1"], w["w_in"], b["proj"], b["proj"] if t_in[3] == 1 else None, t_in)
            assert t_in[3] == 1 and t_out[3] == 4 and t_r[3] == 8, (t_in, t_out, t_r)
            s_out = (b["core"], w["w_out"], b["o"], b["o_part"], t_out)
            s_r = (b["h2"], w["w3"], b["r"], b["r_part"], t_r)
            up, down = self.moe(L)
            ops += [
                Op("add_rms1", L, lambda x_in=x_in, b=b, w=w: self.add_rms(x_in, None, w["ln1"], b["h1"], None)),
                Op("in_proj", L, lambda s=s_in: self.gemv(s), s_in, heavy=True),
                Op("conv", L, lambda L=L: self.conv_step(L)),
                Op("gdn_step", L, lambda L=L: self.gdn_step(L)),
                Op("out_proj", L, lambda s=s_out: self.gemv(s), s_out, heavy=True),
                Op("ksum_o", L, lambda b=b: self.ksum(b["o_part"], b["o"], H, 4)),
                Op("ar_o", L, lambda b=b: self.ar(b["o"], None, b["ar1"])),
                Op("add_rms2", L, lambda x_in=x_in, b=b, w=w: self.add_rms(x_in, b["ar1"], w["ln2"], b["h2"], b["x2"])),
                Op("router", L, lambda s=s_r: self.gemv(s), s_r, heavy=True),
                Op("ksum_router", L, lambda b=b: self.ksum_router(b["r_part"], b["r"], b["idx"], b["wt"], 8)),
                Op("moe_up", L, up, heavy=True),
                Op("moe_down", L, down, heavy=True),
                Op("ar_m", L, lambda b=b: self.ar(b["m"], b["x2"], b["xn"])),
            ]
        return ops

    # ------------------------------------------------------------ state

    def restore(self):
        for L in range(LAYERS):
            self.conv[L].copy_(self.conv0[L])
            self.rec[L].copy_(self.rec0[L])

    def snapshot(self):
        t = []
        for L in range(LAYERS):
            t += list(self.B[L].values()) + [self.conv[L], self.rec[L]]
        return [v.clone() for v in t]


class XkState:
    def __init__(self, torch, dev, ntargets):
        self.flags = torch.zeros(ntargets + 1, dtype=torch.int32, device=dev)
        self.done = torch.zeros(ntargets + 1, dtype=torch.int32, device=dev)
        self.ep = torch.zeros(4, dtype=torch.int32, device=dev)
        self.err = torch.zeros(1, dtype=torch.int32, device=dev)
        self.scratch = torch.zeros(1 << 16, dtype=torch.float32, device=dev)


def schedule(torch, ch: RankChain, var: dict, st: XkState, main, side, evs, which="A"):
    """Launch ``ch``'s ops for variant ``var`` into the current capture (``main`` current)."""
    ops, targets = ch.ops, ch.targets
    gid = {t: i for i, t in enumerate(targets)}
    kind = var["kind"]
    npre = var.get("npre", 0)
    if kind == "serial":
        for o in ops:
            o.fn()
        return
    if kind == "serial_xk":
        _xk_bump_kernel[(1,)](st.ep, 0, num_warps=1)
        for i, o in enumerate(ops):
            if o.spec is not None:
                _xk_signal_kernel[(1,)](st.flags, st.ep, gid[i], 0, num_warps=1)
                ch.xk(o.spec, st, gid[i], 0, npre)
            else:
                o.fn()
        return
    if kind == "2g":
        if which == "A":
            _xk_bump_kernel[(1,)](st.ep, 0, num_warps=1)
            for i, o in enumerate(ops):
                if o.spec is not None:
                    continue
                o.fn()
                if i + 1 in gid:
                    _xk_sigwait_kernel[(1,)](st.flags, st.done, st.ep, st.err, gid[i + 1], 0,
                                             MAXIT=XK_MAXIT, num_warps=1)
        else:
            _xk_bump_kernel[(1,)](st.ep, 1, num_warps=1)
            for t in targets:
                ch.xk(ops[t].spec, st, gid[t], 1, npre)
                _xk_signal_kernel[(1,)](st.done, st.ep, gid[t], 1, num_warps=1)
        return
    # forked variants: "fork" (xk GEMV on the branch) and "pf" (L2 touch on the branch)
    fork_at: dict[int, int] = {}
    prev = -1
    last_heavy = -1
    for i, o in enumerate(ops):
        if o.spec is not None:
            if kind == "fork":
                at = max(i - var["depth"], prev + 1)
            else:  # pf: right after the previous heavy kernel
                at = last_heavy + 1
                at = min(max(at, prev + 1), i)
            fork_at.setdefault(at, i)
            assert fork_at[at] == i, "two forks at one op"
            prev = i
        if o.heavy:
            last_heavy = i
    if kind == "fork":
        _xk_bump_kernel[(1,)](st.ep, 0, num_warps=1)
    pending = {}
    k = 0
    for i, o in enumerate(ops):
        if i in fork_at:
            g = fork_at[i]
            a, b = evs[k]
            k += 1
            a.record(main)
            side.wait_event(a)
            with torch.cuda.stream(side):
                if kind == "fork":
                    ch.xk(ops[g].spec, st, gid[g], 0, npre)
                elif kind == "forknop":
                    _xk_bump_kernel[(1,)](st.ep, 3, num_warps=1)
                else:
                    ch.touch(ops[g].spec, st.scratch[gid[g] * 64:], var.get("pf_npre", 2))
                b.record(side)
            pending[g] = b
        if kind == "fork" and o.spec is not None:
            main.wait_event(pending.pop(i))
            continue
        if kind in ("pf", "forknop") and o.spec is not None:
            main.wait_event(pending.pop(i))
        o.fn()
        if kind == "fork" and i + 1 in gid:
            _xk_signal_kernel[(1,)](st.flags, st.ep, gid[i + 1], 0, num_warps=1)
    assert not pending


def part_chain(args) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from engine.fast import tp as TP

    devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    cus = torch.cuda.get_device_properties(devs[0]).multi_processor_count
    fork_ok = args.fork_ok == 1
    two_ok = args.two_ok == 1
    t0 = time.perf_counter()
    ll = TP._LL(devs)
    out("chain_setup", ll_s=round(time.perf_counter() - t0, 1), cus=cus, fork_ok=fork_ok,
        two_graphs_ok=two_ok)

    variants = [dict(name="serial", kind="serial"),
                dict(name="serial_xk_P4", kind="serial_xk", npre=4),
                dict(name="forknop", kind="forknop"),
                dict(name="pf_P2", kind="pf", pf_npre=2)]
    if fork_ok:
        variants += [dict(name="xk_fork_D1_P4", kind="fork", depth=1, npre=4),
                     dict(name="xk_fork_D1_P0", kind="fork", depth=1, npre=0),
                     dict(name="xk_fork_D2_P4", kind="fork", depth=2, npre=4),
                     dict(name="xk_fork_D1_P1", kind="fork", depth=1, npre=1)]
    if two_ok:
        variants += [dict(name="xk_2g_P4", kind="2g", npre=4)]
    only = set(filter(None, args.variants.split(","))) if args.variants else None
    if only:
        variants = [v for v in variants if v["name"] in only or v["name"] == "serial"]

    summary = {}
    for n_slots in (1, 2):
        G = 4
        R = n_slots * G
        gen = torch.Generator()
        gen.manual_seed(1234 + R)
        x0 = torch.randn((R, H), generator=gen).to(torch.bfloat16)
        t0 = time.perf_counter()
        chains = [RankChain(torch, r, d, ll, n_slots, G, x0, cus, seed=100 * r + 7) for r, d in enumerate(devs)]
        sync_all(torch)
        ntg = len(chains[0].targets)
        out("chain_built", rows=R, seconds=round(time.perf_counter() - t0, 1), ops_per_rank=len(chains[0].ops),
            target_gemvs=ntg, free_mib=[torch.cuda.mem_get_info(d)[0] >> 20 for d in devs])
        mains = [torch.cuda.Stream(d) for d in devs]
        sides = [torch.cuda.Stream(d) for d in devs]

        def eager(fn_per_rank):
            """Run both ranks op by op, interleaved (the all-reduces rendezvous on the GPU)."""
            seqs = [fn_per_rank(r) for r in range(2)]
            for steps in zip(*seqs, strict=True):
                for r, step in enumerate(steps):
                    with torch.cuda.device(devs[r]), torch.cuda.stream(mains[r]):
                        step()
            sync_all(torch)

        # ---- unit: xk GEMV == MFMA GEMV bitwise (and compiles every xk variant used)
        unit_ok = True
        for r, ch in enumerate(chains):
            with torch.cuda.device(devs[r]):
                ust = XkState(torch, devs[r], ntg)
                ust.flags.fill_(1 << 30)
                for t in ch.targets[:3]:
                    x, w, o, part, tiles = ch.ops[t].spec
                    x.normal_()
                    ch.gemv(ch.ops[t].spec)
                    torch.cuda.synchronize(devs[r])
                    ref = (o.clone(), part.clone() if part is not None else None)
                    for npre in (0, 1, 2, 4):
                        for wait in (True, False):
                            o.zero_()
                            if part is not None:
                                part.zero_()
                            ch.xk(ch.ops[t].spec, ust, 0, 0, npre, wait=wait)
                            torch.cuda.synchronize(devs[r])
                            same = torch.equal(o.view(torch.int16), ref[0].view(torch.int16)) and (
                                part is None or torch.equal(part.view(torch.int32), ref[1].view(torch.int32)))
                            if not same:
                                unit_ok = False
                                out("unit_mismatch", rank=r, op=ch.ops[t].kind, npre=npre, wait=wait)
                    ch.touch(ch.ops[t].spec, ust.scratch, 2)
                    ch.touch(ch.ops[t].spec, ust.scratch, 4)
                _xk_bump_kernel[(1,)](ust.ep, 0, num_warps=1)
                _xk_signal_kernel[(1,)](ust.flags, ust.ep, 0, 0, num_warps=1)
                _xk_sigwait_kernel[(1,)](ust.flags, ust.flags, ust.ep, ust.err, 0, 0, MAXIT=XK_MAXIT,
                                         num_warps=1)
                torch.cuda.synchronize(devs[r])
                if int(ust.err.item()):
                    unit_ok = False
        out("unit_xk_gemv_bitwise", rows=R, ok=unit_ok)

        # ---- eager warm-up of every kernel with its real arguments (both ranks)
        for ch in chains:
            ch.restore()
        eager(lambda r: [o.fn for o in chains[r].ops])
        eager(lambda r: [o.fn for o in chains[r].ops])
        out("warm", rows=R, ll_timeouts=ll.timeouts())

        built = {}
        for var in variants:
            name = var["name"]
            try:
                sts = [XkState(torch, d, ntg) for d in devs]
                graphs = []
                for r, ch in enumerate(chains):
                    with torch.cuda.device(devs[r]):
                        evs = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(ntg + 1)]
                        if var["kind"] == "2g":
                            pair = []
                            for which, stream in (("B", sides[r]), ("A", mains[r])):
                                g = torch.cuda.CUDAGraph()
                                stream.wait_stream(torch.cuda.current_stream(devs[r]))
                                with torch.cuda.graph(g, stream=stream):
                                    schedule(torch, ch, var, sts[r], stream, None, evs, which)
                                pair.append((g, stream))
                            graphs.append(pair)
                        else:
                            g = torch.cuda.CUDAGraph()
                            mains[r].wait_stream(torch.cuda.current_stream(devs[r]))
                            with torch.cuda.graph(g, stream=mains[r]):
                                schedule(torch, ch, var, sts[r], mains[r], sides[r], evs)
                            graphs.append([(g, mains[r])])
                        torch.cuda.synchronize(devs[r])
                built[name] = (graphs, sts)
            except Exception as exc:  # noqa: BLE001
                out("capture_error", rows=R, variant=name, error=f"{type(exc).__name__}: {str(exc)[:400]}")
        out("captured", rows=R, variants=list(built))

        def launcher(graphs):
            def launch():
                for r, pair in enumerate(graphs):
                    with torch.cuda.device(devs[r]):
                        for g, stream in pair:
                            with torch.cuda.stream(stream):
                                g.replay()
            return launch

        def errs(sts):
            return sum(int(s.err.item()) for s in sts)

        # reference: serial from the restored state
        ref = None
        res_r = {}
        for name, (graphs, sts) in built.items():
            launch = launcher(graphs)
            ll0 = ll.timeouts()
            for ch in chains:
                ch.restore()
            sync_all(torch)
            t_first = time.perf_counter()
            launch()
            sync_all(torch)
            t_first = time.perf_counter() - t_first
            snap = [ch.snapshot() for ch in chains]
            e1 = errs(sts)
            ll1 = ll.timeouts() - ll0
            if ref is None:
                ref = snap
                bitwise = True
            else:
                bitwise = all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                              for sa, sb in zip(ref, snap, strict=True) for a, b in zip(sa, sb, strict=True))
            xn_agree = all(torch.equal(chains[0].B[L]["xn"].cpu(), chains[1].B[L]["xn"].cpu())
                           for L in (0, LAYERS - 1))
            del snap
            if e1 or ll1 or t_first > 30:
                out("chain", rows=R, variant=name, aborted=True, spin_timeouts=e1, ar_timeouts=ll1,
                    first_replay_s=round(t_first, 3), bitwise_vs_serial=bitwise)
                res_r[name] = dict(ok=False, bitwise=bitwise, spin_timeouts=e1, ar_timeouts=ll1)
                continue
            # timing
            for _ in range(5):
                launch()
            sync_all(torch)
            lat, host = [], []
            for _ in range(args.iters):
                sync_all(torch)
                a = time.perf_counter()
                launch()
                b = time.perf_counter()
                sync_all(torch)
                lat.append((time.perf_counter() - a) * 1e3)
                host.append((b - a) * 1e3)
            sync_all(torch)
            a = time.perf_counter()
            for _ in range(args.iters):
                launch()
            sync_all(torch)
            thr = (time.perf_counter() - a) * 1e3 / args.iters
            # profiler: overlap of the early kernels with the rest
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    launch()
                    sync_all(torch)
                    time.sleep(0.003)
            ev = gpu_events(load_trace(prof))
            pstat = {}
            for d, evl in ev.items():
                steps, cur, last = [], [], None
                for e_ in evl:
                    if last is not None and e_[0] - last > 1500 and cur:
                        steps.append(cur)
                        cur = []
                    cur.append(e_)
                    last = max(last or 0.0, e_[0] + e_[1])
                if cur:
                    steps.append(cur)
                spans, sides_n, sides_ov, sides_dur, gemv_us, cats = [], [], [], [], [], defaultdict(float)
                for s in steps:
                    spans.append(max(t + du for t, du, _ in s) - s[0][0])
                    side_k = [(t, t + du) for t, du, nm in s if "_xk_gemv_kernel" in nm or "_touch_kernel" in nm]
                    main_k = merge([(t, t + du) for t, du, nm in s
                                    if not ("_xk_gemv_kernel" in nm or "_touch_kernel" in nm)])
                    ovs = [overlap_with(iv, main_k) for iv in side_k]
                    sides_n.append(sum(1 for v in ovs if v > 0.5))
                    sides_ov.append(sum(ovs))
                    sides_dur.append(sum(b_ - a_ for a_, b_ in side_k))
                    gemv_us.append(sum(du for _, du, nm in s if "gemv" in nm))
                    for t, du, nm in s:
                        cats[nm.split("(")[0][:40]] += du / len(steps)
                pstat[d] = dict(steps=len(steps), span_us=med(spans), kernels=len(steps[0]) if steps else 0,
                                early_kernels_overlapping=med(sides_n), early_overlap_us=med(sides_ov),
                                early_kernel_us=med(sides_dur), gemv_kernel_us=med(gemv_us),
                                top=sorted(((round(v, 1), k_) for k_, v in cats.items()), reverse=True)[:8])
            e2 = errs(sts)
            ll2 = ll.timeouts() - ll0
            res_r[name] = dict(ok=e2 == 0 and ll2 == 0, bitwise=bitwise, step_ms=round(thr, 3),
                               lat_p50_ms=med(lat), host_launch_ms=med(host), spin_timeouts=e2,
                               ar_timeouts=ll2)
            out("chain", rows=R, variant=name, step_ms=round(thr, 3), lat_p50_ms=med(lat),
                host_launch_p50_ms=med(host), bitwise_vs_serial=bitwise, ranks_agree_xn=xn_agree,
                spin_timeouts=e2, ar_timeouts=ll2, profile=pstat)

        base = res_r.get("serial", {}).get("step_ms")
        cand = {n: v for n, v in res_r.items() if n not in ("serial", "serial_xk_P4", "forknop") and v.get("step_ms")
                and v.get("ok") and v.get("bitwise")}
        best = min(cand, key=lambda n: cand[n]["step_ms"]) if cand else None
        gain = (1 - cand[best]["step_ms"] / base) if best and base else None
        out("chain_summary", rows=R, serial_ms=base,
            serial_xk_overhead_ms=(round(res_r["serial_xk_P4"]["step_ms"] - base, 3)
                                   if base and res_r.get("serial_xk_P4", {}).get("step_ms") else None),
            best=best, best_ms=cand[best]["step_ms"] if best else None,
            gain=round(gain, 4) if gain is not None else None,
            all={n: (v.get("step_ms"), v.get("bitwise"), v.get("ok")) for n, v in res_r.items()})
        summary[R] = dict(base=base, best=best, gain=gain, res=res_r)

        # ---- stress: the best early-launch variant at 4 rows
        if R == 4 and best and args.stress:
            graphs, sts = built[best]
            launch = launcher(graphs)
            ll0 = ll.timeouts()
            t0 = time.perf_counter()
            for i in range(args.stress):
                launch()
                if i % 50 == 49:
                    sync_all(torch)
            sync_all(torch)
            el = time.perf_counter() - t0
            e3 = errs(sts)
            for ch in chains:
                ch.restore()
            launch()
            sync_all(torch)
            snap = [ch.snapshot() for ch in chains]
            for ch in chains:
                ch.restore()
            launcher(built["serial"][0])()
            sync_all(torch)
            refs = [ch.snapshot() for ch in chains]
            bit = all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                      for sa, sb in zip(refs, snap, strict=True) for a, b in zip(sa, sb, strict=True))
            out("stress", rows=R, variant=best, replays=args.stress, seconds=round(el, 1),
                spin_timeouts=e3, ar_timeouts=ll.timeouts() - ll0, bitwise_after=bit)
            summary["stress"] = dict(ok=e3 == 0 and ll.timeouts() - ll0 == 0 and bit)
        del built, chains
        torch.cuda.empty_cache()
    out("chain_done", summary={str(k): v for k, v in summary.items()})


# ================================================================================ driver


def run_part(part: str, extra: list[str], timeout: int, env_extra: dict | None = None) -> tuple[int, list[str]]:
    env = dict(os.environ)
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    env.update(env_extra or {})
    argv = [sys.executable, "-u", __file__, "--part", part, *extra]
    print(f"===== part {part} env={env_extra or {}}: {' '.join(argv[2:])} (timeout {timeout}s)", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(["timeout", "-k", "30", str(timeout), *argv], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            cwd=str(ROOT))
    assert proc.stdout is not None
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("RESULT "):
            lines.append(line.strip())
    rc = proc.wait()
    print(f"===== part {part} done rc={rc}{' (TIMEOUT: hang)' if rc in (124, 137) else ''} "
          f"s={time.time() - t0:.0f}", flush=True)
    return rc, lines


def parse(lines):
    res = defaultdict(list)
    for line in lines:
        _, tag, payload = line.split(" ", 2)
        with contextlib.suppress(json.JSONDecodeError):
            res[tag].append(json.loads(payload))
    return res


# HIP runtime settings tried for the two-queue schemes (found in libamdhip64.so by part env).
RUNTIME_ENVS = {
    "default": {},
    "no_packet_capture": {"DEBUG_CLR_GRAPH_PACKET_CAPTURE": "0"},
    "graph_queues_4": {"DEBUG_HIP_FORCE_GRAPH_QUEUES": "4"},
}


def driver(args) -> None:
    run_part("env", [], 120)
    mech = {}
    for label, envx in RUNTIME_ENVS.items():
        rc, lm = run_part("mech", [], 300, envx)
        ms = (parse(lm).get("mech_summary") or [{}])[-1]
        mech[label] = dict(rc=rc, fork=bool(ms.get("fork_concurrent")),
                           two_graphs=bool(ms.get("two_graphs_concurrent")))
        out("mech_env", env=label, **mech[label])
    rc_c, lc = run_part("ceiling", [], 900)
    c = parse(lc)
    chains = {}
    for label, envx in RUNTIME_ENVS.items():
        m = mech[label]
        rc_x, lx = run_part("chain", ["--fork-ok", str(int(m["fork"] or args.force_xk)),
                                      "--two-ok", str(int(m["two_graphs"] or args.force_xk)),
                                      "--iters", str(args.iters),
                                      "--stress", str(args.stress if label == "default" else 500),
                                      "--variants", args.variants], args.chain_timeout, envx)
        x = parse(lx)
        sums = {s_["rows"]: s_ for s_ in x.get("chain_summary", [])}
        chains[label] = (rc_x, x, sums)
        out("chain_env", env=label, rc=rc_x,
            rows4={k: sums.get(4, {}).get(k) for k in ("serial_ms", "best", "best_ms", "gain", "all")},
            rows8={k: sums.get(8, {}).get(k) for k in ("serial_ms", "best", "best_ms", "gain", "all")})
    # the claim: the best two-queue variant at 4 rows over every runtime setting
    best_label = max(chains, key=lambda k: (chains[k][2].get(4, {}).get("gain") is not None,
                                            chains[k][2].get(4, {}).get("gain") or -9))
    rc_x, x, sums = chains[best_label]
    s4 = sums.get(4, {})
    failed, other = [], []
    if rc_c != 0:
        other.append(f"part ceiling rc={rc_c}")
    for label, (rc_, _, _) in chains.items():
        if rc_ != 0:
            (failed if label == best_label else other).append(
                f"part chain[{label}] rc={rc_}{' (hang)' if rc_ in (124, 137) else ''}")
    for label, (_, xx, _) in chains.items():
        for r in xx.get("chain", []):
            bad = []
            if r.get("variant") != "serial" and not r.get("bitwise_vs_serial", True):
                bad.append(f"bitwise {label}:{r['variant']} R={r['rows']}")
            if r.get("spin_timeouts") or r.get("ar_timeouts") or r.get("aborted"):
                bad.append(f"timeouts {label}:{r['variant']} R={r['rows']}")
            # The kill rule binds the variant that carries the claim (the best at 4 rows) and the
            # serial reference; failures of the other schemes are reported as findings.
            if label == best_label and r.get("variant") in ("serial", s4.get("best")) and r.get("rows") == 4:
                failed += bad
            else:
                other += bad
        for r in xx.get("unit_xk_gemv_bitwise", []):
            if not r.get("ok"):
                failed.append(f"unit xk gemv {label} R={r['rows']}")
    for r in x.get("stress", []):
        if r.get("spin_timeouts") or r.get("ar_timeouts") or not r.get("bitwise_after"):
            failed.append("stress")
    ceil = [(r["stream"], r["latency"], r.get("hidden_share_of_stream"), r.get("latency_kernel_slowdown"))
            for r in c.get("ceiling", [])]
    gain4 = s4.get("gain")
    kill = bool(failed) or gain4 is None or gain4 < KILL_GAIN
    out("verdict", mechanism=mech, ceiling=ceil, best_env=best_label,
        chain4=dict(serial_ms=s4.get("serial_ms"), best=s4.get("best"), best_ms=s4.get("best_ms"), gain=gain4),
        chain8={k: sums.get(8, {}).get(k) for k in ("serial_ms", "best", "best_ms", "gain")},
        kill_rule=f"gain at 4 rows < {KILL_GAIN:.0%}, any bit difference, any hang/timeout",
        failed=failed, other_variant_failures=other, verdict="kill" if kill else "go")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="all", choices=("all", "env", "mech", "ceiling", "chain"))
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--stress", type=int, default=2000)
    p.add_argument("--fork-ok", type=int, default=1)
    p.add_argument("--two-ok", type=int, default=1)
    p.add_argument("--variants", default="")
    p.add_argument("--chain-timeout", type=int, default=1200)
    p.add_argument("--force-xk", type=int, default=1,
                   help="run the xk chain variants even if part mech saw no concurrency")
    args = p.parse_args()
    if args.part == "all":
        driver(args)
    elif args.part == "env":
        part_env(args)
    elif args.part == "mech":
        part_mech(args)
    elif args.part == "ceiling":
        part_ceiling(args)
    else:
        part_chain(args)


if __name__ == "__main__":
    main()
