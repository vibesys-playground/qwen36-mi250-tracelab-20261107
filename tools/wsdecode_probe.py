"""Plan 65 stage 0: wave-specialized decode GEMV (producer waves DMA weights HBM -> LDS with
``buffer_load_dword ... lds``), feasibility on one gfx90a GCD. Tools only, no engine change.

    python tools/wsdecode_probe.py --part a|b|c [--device N] [--quick]

Kernel: engine/fast/hip/wsdecode.hip (``ws_gemv``, see its header), compiled with hipRTC through
engine/fast/hipmega_rtc.py in a child process. Shapes are one TP2 rank's shards, random BF16
weights: in_proj 6176x2048, out_proj 2048x2048, lm_head 124160x2048, and the routed experts of
one layer, gate_up [256, 512, 2048] and down [256, 2048, 256], of which a routing of R rows x 8
reads the distinct experts (39 at R = 8, as the plan's B=2 k=3 count).

(a) compile: every variant's VGPRs / spills / occupancy / LDS, and the disassembly lines that
    show the HBM -> LDS DMA (``buffer_load_dword ... lds``) and the producer's vmcnt waits.
(b) streaming: us per launch and weight TB/s of ws_gemv (variants: producer waves P, tiles in
    flight per producer PD, 256-VGPR budget or not) at 4 and 8 rows, against today's kernels
    on the same shapes and bytes (``mfma.gemv`` for the dense shards, ``fused._moe_up_kernel`` +
    ``_moe_down_kernel`` for the experts). Each timing is a CUDA graph of back-to-back launches
    over rotating weight copies (>= 64 MB per rotation, so L2 holds none), median of 5 replays.
    Checks per config: error count of the capped spins (0 = no protocol timeout), max error vs
    an fp32 reference, rows 0-3 bitwise equal between R = 4 and R = 8, and 100 launches bitwise
    equal to the first.
(c) chain: in_proj GEMV -> 20 us latency-bound phase -> out_proj GEMV whose input is the first
    2048 outputs of in_proj. Graph: ``mfma.gemv`` -> spin kernel (every CU spins 20 us, then
    copies the next input) -> ``mfma.gemv``. WS: one ws_gemv launch, phase 0 in_proj, grid
    barrier + 20 us spin by the consumers, phase 1 out_proj, the producers prefetching phase 1's
    weights during the barrier and the spin (and the same with prefetch off). Same checks.

Prints ``RESULT`` lines. A hint for the plan, never the task's metric.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "engine" / "fast" / "hip" / "wsdecode.hip"

# variant name -> compile defines
VARIANTS = {
    "p1d3": dict(P=1, C=4, PD=3, NS=12, GR=3, HIGHREG=1),
    "p2d3": dict(P=2, C=4, PD=3, NS=12, GR=3, HIGHREG=1),
    "p2d2": dict(P=2, C=4, PD=2, NS=12, GR=3, HIGHREG=1),
    "p4d2": dict(P=4, C=4, PD=2, NS=12, GR=3, HIGHREG=1),
    "p2d3-lowreg": dict(P=2, C=4, PD=3, NS=12, GR=3, HIGHREG=0),
}
H = 2048
TP2 = {"in_proj": (6176, 2048), "out_proj": (2048, 2048), "lm_head": (124160, 2048)}
E, I2, TOPK = 256, 256, 8  # experts, TP2 expert intermediate shard, top-k
N3 = E + 2 * I2 + 1        # router | shared gate | shared up | shared gate scalar (TP2: 769)


def out(msg: str) -> None:
    print(msg, flush=True)


def opts_of(defs: dict) -> tuple[str, ...]:
    return tuple(f"-D{k}={v}" for k, v in defs.items())


# ------------------------------------------------------------------------------- disassembly

def comgr_disasm(code: bytes) -> str:
    """Disassemble a code object with torch's libamd_comgr (action 10 is
    DISASSEMBLE_EXECUTABLE_TO_SOURCE in ROCm 7.0's enum; found by trying offline)."""
    import torch

    lib = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libamd_comgr.so"))

    class Hd(ctypes.Structure):
        _fields_ = [("handle", ctypes.c_uint64)]

    def ck(rc, what):
        if rc:
            raise RuntimeError(f"comgr {what} rc={rc}")

    d = Hd()
    ck(lib.amd_comgr_create_data(8, ctypes.byref(d)), "create_data")
    ck(lib.amd_comgr_set_data(d, ctypes.c_size_t(len(code)), code), "set_data")
    ck(lib.amd_comgr_set_data_name(d, b"k.co"), "set_data_name")
    ins, outs, info = Hd(), Hd(), Hd()
    ck(lib.amd_comgr_create_data_set(ctypes.byref(ins)), "create_data_set")
    ck(lib.amd_comgr_data_set_add(ins, d), "data_set_add")
    ck(lib.amd_comgr_create_data_set(ctypes.byref(outs)), "create_data_set")
    ck(lib.amd_comgr_create_action_info(ctypes.byref(info)), "create_action_info")
    ck(lib.amd_comgr_action_info_set_isa_name(info, b"amdgcn-amd-amdhsa--gfx90a"), "isa")
    ck(lib.amd_comgr_do_action(10, info, ins, outs), "do_action")
    o = Hd()
    ck(lib.amd_comgr_action_data_get_data(outs, 1, 0, ctypes.byref(o)), "get_data")
    n = ctypes.c_size_t()
    ck(lib.amd_comgr_get_data(o, ctypes.byref(n), None), "size")
    buf = ctypes.create_string_buffer(n.value)
    ck(lib.amd_comgr_get_data(o, ctypes.byref(n), buf), "get")
    return buf.raw.decode(errors="replace")


def compile_variant(name: str) -> tuple[bytes, str]:
    from engine.fast import hipmega_rtc as rtc

    return rtc.compile_hip_subprocess(str(SRC), "gfx90a", opts_of(VARIANTS[name]))


def part_a(args) -> None:
    from engine.fast import hipmega_rtc as rtc

    import torch

    out(f"RESULT a stack torch={torch.__version__} hip={torch.version.hip}")
    for name in VARIANTS:
        t0 = time.time()
        try:
            co, log = compile_variant(name)
        except Exception as e:  # noqa: BLE001
            out(f"RESULT a compile {name} FAIL {str(e)[-1500:]}")
            continue
        res = rtc.resources(log)
        for fn, r in res.items():
            keep = {k: r.get(k) for k in ("VGPRs", "AGPRs", "VGPRs Spill", "ScratchSize [bytes/lane]",
                                          "Occupancy [waves/SIMD]", "LDS Size [bytes/block]")}
            out(f"RESULT a compile {name} {fn} {keep} {time.time() - t0:.1f}s")
        if name == "p2d3":
            # disassemble in a child too (comgr's LLVM stays out of this process)
            code = ("import sys; sys.path.insert(0, %r); from wsdecode_probe import comgr_disasm; "
                    "sys.stdout.write(comgr_disasm(open(%r, 'rb').read()))")
            tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "wsdecode_p2d3.co"
            tmp.write_bytes(co)
            try:
                asm = subprocess.run([sys.executable, "-c", code % (str(ROOT / "tools"), str(tmp))], check=True,
                                     timeout=300, capture_output=True, text=True).stdout
            except Exception as e:  # noqa: BLE001
                out(f"RESULT a disasm FAIL {e}")
                continue
            lines = [ln.split("//")[0].rstrip() for ln in asm.splitlines()]
            dma = [ln for ln in lines if "buffer_load" in ln and ln.rstrip().endswith("lds")]
            vm = [ln.strip() for ln in lines if "s_waitcnt vmcnt" in ln]
            out(f"RESULT a disasm buffer_load_lds={len(dma)} mfma={sum('v_mfma' in ln for ln in lines)} "
                f"vmcnt_waits={sorted(set(vm))}")
            # the producer's hot block: the 16 DMAs of one tile with their M0 updates
            first = next((i for i, ln in enumerate(lines) if "buffer_load" in ln and ln.endswith("lds")), None)
            if first is not None:
                for ln in lines[max(0, first - 3): first + 12]:
                    out(f"ASM {ln.strip()}")
                last = max(i for i, ln in enumerate(lines) if "buffer_load" in ln and ln.endswith("lds"))
                for ln in lines[last: last + 10]:
                    out(f"ASM {ln.strip()}")


# ------------------------------------------------------------------------------- GPU helpers

def pick_device(arg: int | None) -> int:
    import torch

    n = torch.cuda.device_count()
    if arg is not None:
        return arg
    free = []
    for d in range(n):
        try:
            f, t = torch.cuda.mem_get_info(d)
        except Exception:  # noqa: BLE001
            f, t = 0, 1
        free.append((f, d))
    out(f"RESULT devices n={n} free_GB={[round(f / 1e9, 1) for f, _ in sorted(free, key=lambda x: x[1])]}")
    return max(free)[1]


def time_graph(fns, min_launches: int = 32, reps: int = 5) -> float:
    """us per call of launch closures, captured back to back as one CUDA graph."""
    import torch

    seq = list(fns) * max(1, math.ceil(min_launches / len(fns)))
    for f in fns:
        f()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fns[0]()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for f in seq:
            f()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(seq))
    del graph
    return statistics.median(times)


class WS:
    """One compiled ws_gemv variant on a device and its launch tables."""

    def __init__(self, name: str, dev):
        import torch

        from engine.fast import hipmega_rtc as rtc

        self.name, self.dev = name, dev
        self.defs = VARIANTS[name]
        co, _ = compile_variant(name)
        self.mod = rtc.Module(co, dev)
        self.fn = self.mod.fn("ws_gemv")
        self.spin = self.mod.fn("spin_copy")
        self.cus = torch.cuda.get_device_properties(dev).multi_processor_count
        self.block = (self.defs["P"] + self.defs["C"]) * 64
        occ = self.fn.occupancy(self.block)
        if occ < 1:
            raise RuntimeError(f"{name}: occupancy {occ} blocks/CU")
        self.gbar = torch.zeros(4, dtype=torch.int32, device=dev)
        self.err = torch.zeros(4, dtype=torch.int32, device=dev)

    def tables(self, segs: list[dict], phases: list[list[int]]):
        """Device tables for a launch: ``segs`` dicts (w, x, out tensors, xs, os), ``phases`` a
        list of segment-index lists, one per phase. Each phase's groups split contiguously over
        the blocks; every block gets >= 1 group of every phase."""
        import torch

        C = self.defs["C"]
        nb = self.cus
        rows = []
        for s in segs:
            w = s["w"]
            n, k = w.shape
            assert n % 16 == 0 and k % 128 == 0 and k <= 2048, (n, k)
            kc = k // 128
            assert kc % C == 0 or C % kc == 0, kc
            rows.append([w.data_ptr(), s["x"].data_ptr(), s["out"].data_ptr(), n, k, s["xs"], s["os"],
                         w.numel() * 2])
        per_block: list[list[tuple[int, int, int]]] = [[] for _ in range(nb)]
        for ph, seg_ids in enumerate(phases):
            gl = [(si, n0, ph) for si in seg_ids for n0 in range(0, rows[si][3], 16)]
            assert len(gl) >= nb, f"phase {ph}: {len(gl)} groups < {nb} blocks"
            for b in range(nb):
                per_block[b].extend(gl[b * len(gl) // nb:(b + 1) * len(gl) // nb])
        flat, gstart = [], [0]
        for b in range(nb):
            t = 0
            for (si, n0, ph) in per_block[b]:
                kc = rows[si][4] // 128
                if kc < C:
                    assert t % kc == 0
                t += kc
                flat.append([si, n0, ph, 0])
            gstart.append(len(flat))
        segt = torch.tensor(rows, dtype=torch.int64, device=self.dev)
        grp = torch.tensor(flat, dtype=torch.int32, device=self.dev)
        gs = torch.tensor(gstart, dtype=torch.int32, device=self.dev)
        return (segt, grp, gs)

    def launch(self, tab, R: int, wait_ticks: int = 0, prefetch: int = 1) -> None:
        segt, grp, gs = tab
        self.fn.launch(self.cus, self.block,
                       [segt, grp, gs, self.gbar, ctypes.c_int(wait_ticks), ctypes.c_int(R),
                        ctypes.c_int(prefetch), self.err])

    def errors(self) -> int:
        import torch

        torch.cuda.synchronize()
        return int(self.err[0].item())


def bitwise_repeat(run, outs, n: int = 100) -> tuple[int, int]:
    """(launches, mismatching launches) over n launches of ``run`` against the first's outputs."""
    import torch

    run()
    torch.cuda.synchronize()
    ref = [o.clone() for o in outs]
    bad = 0
    for _ in range(n):
        run()
        torch.cuda.synchronize()
        if not all(torch.equal(a, b) for a, b in zip(outs, ref)):
            bad += 1
    return n, bad


def rel_err(y, ref) -> float:
    return float((y.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))


# ------------------------------------------------------------------------------- part b

def experts_routing(R: int, gen) -> list[list[int]]:
    """R x 8 routing over 39 distinct experts at R = 8 (row r: window 5r .. 5r+7 of a shuffled
    pool of 39, so the 8 rows cover all 39); rows 0..R-1 of it for smaller R."""
    import torch

    pool = torch.randperm(E, generator=gen)[:39].tolist()
    rows = [[pool[(5 * r + j) % 39] for j in range(TOPK)] for r in range(8)]
    return rows[:R]


def part_b(args, dev) -> None:
    import torch

    from engine.fast import fused, mfma

    gen = torch.Generator().manual_seed(65)
    torch.manual_seed(65)
    bf = torch.bfloat16
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    names = list(VARIANTS) if not args.quick else ["p2d3"]

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=gen) * scale).to(bf).to(dev)

    # dense shards with rotating copies (>= 64 MB per rotation)
    dense = {}
    for nm, (n, k) in TP2.items():
        copies = max(1, math.ceil(64e6 / (n * k * 2)))
        dense[nm] = [rnd(n, k) for _ in range(copies)]
    gu = rnd(E, 2 * I2, H)          # [256, 512, 2048]
    dn = rnd(E, H, I2)              # [256, 2048, 256]
    sd = rnd(H, I2)                 # shared expert down shard [2048, 256]
    out(f"RESULT b setup device={dev} cus={cus} copies={ {k: len(v) for k, v in dense.items()} }")

    xs = {R: rnd(R, H, scale=1.0) for R in (4, 8)}
    acts = {R: rnd(R, I2, scale=1.0) for R in (4, 8)}
    routes = {8: experts_routing(8, gen)}
    routes[4] = routes[8][:4]

    # ---------------- baselines (today's kernels)
    base = {}
    for R in (4, 8):
        x = xs[R]
        for nm, ws_ in dense.items():
            n, k = ws_[0].shape
            o = torch.empty(R, n, dtype=bf, device=dev)
            fns = [lambda w=w: mfma.gemv(x, w, o) for w in ws_]
            us = time_graph(fns)
            base[(nm, R)] = us
            out(f"RESULT b base {nm} R={R} us={us:.1f} TBps={n * k * 2 / us / 1e6:.3f}")
        # experts: today's up and down kernels on the routing
        idx = torch.tensor(routes[R], dtype=torch.int32, device=dev)
        wt = torch.rand(R, TOPK, generator=gen).to(dev)
        r = rnd(R, N3, scale=1.0)
        act = torch.empty(R * TOPK, I2, dtype=bf, device=dev)
        o = torch.empty(R, H, dtype=bf, device=dev)
        order = 1 if R >= fused.MOE_TILE_ROWS else 0
        grid = fused.MOE_GRID
        bn = fused.TILES["moe_bn"]
        items_up = R * TOPK * math.ceil(I2 / bn)
        items_dn = R * math.ceil(H / bn)

        def up():
            fused._moe_up_kernel[(items_up if order else min(items_up, grid * cus),)](
                x, idx, gu, act, R, gu.stride(0), r, wt, N3, H=H, I=I2, TOPK=TOPK, BN=bn, BK=1024,
                ORDER=order, E=E, TOPKP=8, FUSE_TOPK=False, num_warps=4)

        def down():
            fused._moe_down_kernel[(min(items_dn, grid * cus),)](
                act, r, idx, wt, dn, sd, o, o, R, dn.stride(0), N3, H=H, I=I2, IS=I2, E=E, TOPK=TOPK,
                BN=bn, BKI=256, BKS=256, ORDER=order, HAS_RES=False, num_warps=4)

        distinct = sorted({e for row in routes[R] for e in row})
        b_up = len(distinct) * 2 * I2 * H * 2
        b_all = b_up + len(distinct) * H * I2 * 2 + H * I2 * 2
        us_up = time_graph([up])
        us_all = time_graph([lambda: (up(), down())])
        base[("experts_up", R)] = us_up
        base[("experts_all", R)] = us_all
        out(f"RESULT b base experts_up R={R} distinct={len(distinct)} us={us_up:.1f} TBps={b_up / us_up / 1e6:.3f}")
        out(f"RESULT b base experts_all R={R} distinct={len(distinct)} us={us_all:.1f} TBps={b_all / us_all / 1e6:.3f}")

    # ---------------- ws_gemv variants
    for vn in names:
        try:
            ws = WS(vn, dev)
        except Exception as e:  # noqa: BLE001
            out(f"RESULT b ws {vn} SETUP FAIL {e}")
            continue
        out(f"RESULT b ws {vn} defs={ws.defs} block={ws.block} occ={ws.fn.occupancy(ws.block)}")
        for R in (4, 8):
            x = xs[R]
            cases = {}
            for nm, ws_ in dense.items():
                n, k = ws_[0].shape
                o = torch.empty(R, n, dtype=bf, device=dev)
                tabs = [ws.tables([dict(w=w, x=x, out=o, xs=k, os=n)], [[0]]) for w in ws_]
                cases[nm] = (tabs, [o], n * k * 2, lambda o=o, w=ws_[0]: [(o, (x.float() @ w.float().t()))])
            distinct = sorted({e for row in routes[R] for e in row})
            o_up = torch.empty(len(distinct), R, 2 * I2, dtype=bf, device=dev)
            segs_up = [dict(w=gu[e], x=x, out=o_up[i], xs=H, os=2 * I2) for i, e in enumerate(distinct)]
            o_dn = torch.empty(len(distinct) + 1, R, H, dtype=bf, device=dev)
            a = acts[R]
            segs_dn = [dict(w=dn[e], x=a, out=o_dn[i], xs=I2, os=H) for i, e in enumerate(distinct)]
            segs_dn.append(dict(w=sd, x=a, out=o_dn[-1], xs=I2, os=H))
            b_up = len(distinct) * 2 * I2 * H * 2
            b_all = b_up + (len(distinct) + 1) * H * I2 * 2
            cases["experts_up"] = ([ws.tables(segs_up, [list(range(len(segs_up)))])], [o_up], b_up,
                                   lambda: [(o_up[i], x.float() @ gu[e].float().t()) for i, e in enumerate(distinct[:4])])
            allsegs = segs_up + segs_dn
            cases["experts_all"] = ([ws.tables(allsegs, [list(range(len(allsegs)))])], [o_up, o_dn], b_all,
                                    lambda: [(o_dn[i], a.float() @ dn[e].float().t()) for i, e in enumerate(distinct[:4])]
                                    + [(o_dn[-1], a.float() @ sd.float().t())])
            for nm, (tabs, outs, nbytes, refs) in cases.items():
                if args.quick and nm == "lm_head":
                    continue
                tag = f"{vn} {nm} R={R}"
                try:
                    ws.err.zero_()
                    ws.gbar.zero_()
                    ws.launch(tabs[0], R)
                    e0 = ws.errors()
                    errs = [rel_err(y, ref) for y, ref in refs()]
                    if e0:
                        out(f"RESULT b ws {tag} ERR spins={e0} relerr={max(errs):.3g} (timing skipped)")
                        continue
                    us = time_graph([lambda t=t: ws.launch(t, R) for t in tabs])
                    n, bad = bitwise_repeat(lambda: ws.launch(tabs[0], R), outs, 100)
                    e1 = ws.errors()
                    rb = base.get((nm, R))
                    out(f"RESULT b ws {tag} us={us:.1f} TBps={nbytes / us / 1e6:.3f} "
                        f"vs_base={'%.2fx' % (rb / us) if rb else '-'} relerr={max(errs):.3g} "
                        f"repeat_bad={bad}/{n} spin_errs={e1}")
                except Exception as e:  # noqa: BLE001
                    out(f"RESULT b ws {tag} FAIL {type(e).__name__}: {str(e)[:400]}")
                    traceback.print_exc()
        # row invariance: rows 0-3 of an R=8 launch == an R=4 launch on the same x rows
        try:
            w = dense["in_proj"][0]
            n, k = w.shape
            o8 = torch.empty(8, n, dtype=bf, device=dev)
            o4 = torch.empty(4, n, dtype=bf, device=dev)
            ws.launch(ws.tables([dict(w=w, x=xs[8], out=o8, xs=k, os=n)], [[0]]), 8)
            ws.launch(ws.tables([dict(w=w, x=xs[8], out=o4, xs=k, os=n)], [[0]]), 4)
            torch.cuda.synchronize()
            out(f"RESULT b ws {vn} row_invariance in_proj R4==R8[:4] {torch.equal(o8[:4], o4)} spin_errs={ws.errors()}")
        except Exception as e:  # noqa: BLE001
            out(f"RESULT b ws {vn} row_invariance FAIL {e}")
        del ws
        torch.cuda.synchronize()


# ------------------------------------------------------------------------------- part c

def part_c(args, dev) -> None:
    import torch

    from engine.fast import mfma

    gen = torch.Generator().manual_seed(66)
    bf = torch.bfloat16
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    (na, ka), (nbo, kb) = TP2["in_proj"], TP2["out_proj"]
    copies = 3  # 3 x (25.3 + 8.4) MB = 101 MB per rotation
    wa = [(torch.randn(na, ka, generator=gen) * 0.02).to(bf).to(dev) for _ in range(copies)]
    wb = [(torch.randn(nbo, kb, generator=gen) * 0.02).to(bf).to(dev) for _ in range(copies)]
    nbytes = na * ka * 2 + nbo * kb * 2
    wait_us = 20.0
    ticks = int(wait_us * 100)  # s_memrealtime is 100 MHz on gfx90a
    names = ["p2d3", "p4d2", "p1d3"] if not args.quick else ["p2d3"]
    for R in (4, 8):
        x = (torch.randn(R, ka, generator=gen)).to(bf).to(dev)
        y0 = torch.empty(R, na, dtype=bf, device=dev)
        x1 = torch.empty(R, kb, dtype=bf, device=dev)
        y1 = torch.empty(R, nbo, dtype=bf, device=dev)
        ws0 = WS("p2d3", dev)  # its spin_copy kernel serves the graph baseline

        def spin(t=ticks):
            ws0.spin.launch(cus, 64, [ctypes.c_int(t), y0, ctypes.c_int64(na), x1, ctypes.c_int(R), ctypes.c_int(kb)])

        a_us = time_graph([lambda w=w: mfma.gemv(x, w, y0) for w in wa])
        b_us = time_graph([lambda w=w: mfma.gemv(x1, w, y1) for w in wb])
        s_us = time_graph([spin])
        s0_us = time_graph([lambda: spin(0)])
        chain = [lambda i=i: (mfma.gemv(x, wa[i], y0), spin(), mfma.gemv(x1, wb[i], y1)) for i in range(copies)]
        g_us = time_graph(chain)
        chain0 = [lambda i=i: (mfma.gemv(x, wa[i], y0), spin(0), mfma.gemv(x1, wb[i], y1)) for i in range(copies)]
        g0_us = time_graph(chain0)
        chain[0]()
        torch.cuda.synchronize()
        gref = y1.clone()
        out(f"RESULT c graph R={R} in_proj={a_us:.1f} out_proj={b_us:.1f} spin20={s_us:.1f} spin0={s0_us:.1f} "
            f"chain20={g_us:.1f} chain0={g0_us:.1f} boundary20={g_us - a_us - b_us - s_us:.1f} "
            f"boundary0={g0_us - a_us - b_us - s0_us:.1f}")
        for vn in names:
            try:
                ws = ws0 if vn == "p2d3" else WS(vn, dev)
                tabs = [ws.tables([dict(w=wa[i], x=x, out=y0, xs=ka, os=na),
                                   dict(w=wb[i], x=y0, out=y1, xs=na, os=nbo)], [[0], [1]]) for i in range(copies)]
                single_a = [ws.tables([dict(w=wa[i], x=x, out=y0, xs=ka, os=na)], [[0]]) for i in range(copies)]
                single_b = [ws.tables([dict(w=wb[i], x=x1, out=y1, xs=kb, os=nbo)], [[0]]) for i in range(copies)]
                ws.err.zero_()
                ws.gbar.zero_()
                ws.launch(tabs[0], R, ticks, 1)
                e0 = ws.errors()
                y0c = y0[:, :kb].float()
                rr = rel_err(y1, y0c @ wb[0].float().t())
                vs_graph = rel_err(y1, gref.float())
                if e0:
                    out(f"RESULT c ws {vn} R={R} ERR spins={e0} relerr={rr:.3g} (timing skipped)")
                    continue
                res = {}
                for label, tl, tk, pf in (("chain20", tabs, ticks, 1), ("chain20_noprefetch", tabs, ticks, 0),
                                          ("chain0", tabs, 0, 1), ("in_proj", single_a, 0, 1),
                                          ("out_proj", single_b, 0, 1)):
                    res[label] = time_graph([lambda t=t, tk=tk, pf=pf: ws.launch(t, R, tk, pf) for t in tl])
                n, bad = bitwise_repeat(lambda: ws.launch(tabs[0], R, ticks, 1), [y0, y1], 100)
                e1 = ws.errors()
                out(f"RESULT c ws {vn} R={R} " + " ".join(f"{k}={v:.1f}" for k, v in res.items())
                    + f" ratio_vs_graph20={res['chain20'] / g_us:.3f} ratio_vs_graph0={res['chain0'] / g0_us:.3f}"
                    f" stream_TBps_chain20={nbytes / (res['chain20'] - wait_us) / 1e6:.3f}"
                    f" relerr={rr:.3g} vs_graph_relerr={vs_graph:.3g} repeat_bad={bad}/{n} spin_errs={e1}")
            except Exception as e:  # noqa: BLE001
                out(f"RESULT c ws {vn} R={R} FAIL {type(e).__name__}: {str(e)[:400]}")
                traceback.print_exc()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "c"], required=True)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.part == "a":
        part_a(args)
        return
    import torch

    d = pick_device(args.device)
    dev = torch.device("cuda", d)
    torch.cuda.set_device(dev)
    out(f"RESULT device {d} {torch.cuda.get_device_name(dev)} cus={torch.cuda.get_device_properties(dev).multi_processor_count}")
    {"b": part_b, "c": part_c}[args.part](args, dev)


if __name__ == "__main__":
    main()
