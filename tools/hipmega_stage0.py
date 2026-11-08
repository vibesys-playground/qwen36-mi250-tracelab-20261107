"""Plan 39 step 0: can a hand-written HIP persistent verify kernel per GCD beat today's graph?

    python tools/hipmega_stage0.py env      # (a) compile path, trivial bandwidth, barrier, P2P
    python tools/hipmega_stage0.py slice    # (b) per-phase bandwidth, (c) the 3-layer slice

Prints ``RESULT`` lines as it goes (flushed). No engine change: the kernels live in
engine/fast/hipmega/*.hip and are compiled at run time with hipRTC from torch's wheel
(engine/fast/hipmega_rtc.py).

(a) ``env``: what the image offers to compile HIP (hipcc, hipRTC in torch/lib, torch's own
    ``cpp_extension``/``_compile_kernel``), then engine/fast/hipmega/probe.hip: its compiler
    resource report, HBM read bandwidth of a kernel held to <= 128 VGPRs with 1-16 16-byte
    loads in flight per thread, the cost of a software grid barrier, and the one-way latency of
    an in-kernel flag between the two GCDs of one card.
(b)/(c) ``slice``: see ``slice_main``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIPDIR = ROOT / "engine" / "fast" / "hipmega"


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def events_ms(fn, iters: int, dev) -> list[float]:
    import torch

    times = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(torch.cuda.current_stream(dev))
        fn()
        b.record(torch.cuda.current_stream(dev))
        b.synchronize()
        times.append(a.elapsed_time(b))
    return times


# ============================================================================== (a) env


def env_main(args) -> None:
    import torch

    from engine.fast import hipmega_rtc as rtc

    lib = rtc.LIB
    out("env", torch=torch.__version__, hip=torch.version.hip,
        hipcc_on_path=shutil.which("hipcc"), opt_rocm_hipcc=os.path.exists("/opt/rocm/bin/hipcc"),
        opt_rocm_clang=os.path.exists("/opt/rocm/llvm/bin/clang"),
        torch_lib_hiprtc=(lib / "libhiprtc.so").exists(), torch_lib_comgr=(lib / "libamd_comgr.so").exists(),
        torch_lib_amdhip=(lib / "libamdhip64.so").exists(),
        compile_kernel=hasattr(torch.cuda, "_compile_kernel"), gpus=torch.cuda.device_count(),
        arch=torch.cuda.get_device_properties(0).gcnArchName,
        cus=torch.cuda.get_device_properties(0).multi_processor_count)
    # torch's own run-time compile helper (NVRTC API, hipRTC on ROCm), if this build has it.
    if hasattr(torch.cuda, "_compile_kernel"):
        try:
            k = torch.cuda._compile_kernel(
                'extern "C" __global__ void add1(float* x, int n) { int i = blockIdx.x * 256 + threadIdx.x; if (i < n) x[i] += 1.0f; }',
                "add1")
            x = torch.zeros(1000, device="cuda:0")
            k(grid=(4, 1, 1), block=(256, 1, 1), args=[x, 1000])
            torch.cuda.synchronize()
            out("torch_compile_kernel", ok=bool((x == 1).all()))
        except Exception as exc:  # noqa: BLE001
            out("torch_compile_kernel", ok=False, error=f"{type(exc).__name__}: {exc}"[:400])
    t0 = time.perf_counter()
    src = (HIPDIR / "probe.hip").read_text()
    code, log = rtc.compile_hip(src)
    out("hiprtc_compile", seconds=round(time.perf_counter() - t0, 2), code_bytes=len(code),
        resources=rtc.resources(log))

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    mod = rtc.Module(code, dev)
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    nbytes = 1 << 30
    buf = torch.empty(nbytes // 4, dtype=torch.int32, device=dev).random_()
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    n16 = nbytes // 16
    # torch's own copy as a yardstick: read + write of 512 MiB each.
    half = buf[: buf.numel() // 2]
    other = torch.empty_like(half)
    t = events_ms(lambda: other.copy_(half), 10, dev)
    out("bw_torch_copy", gbps_rw=round(2 * half.numel() * 4 / statistics.median(t) / 1e6, 1))
    for name in ("bw1", "bw4", "bw8", "bw16"):
        f = mod.fn(name)
        for block in (256, 512):
            occ = f.occupancy(block)
            for per_cu in (1, 2, 4):
                if per_cu > occ:
                    continue
                grid = per_cu * cus

                def run(f=f, grid=grid, block=block):
                    f.launch(grid, block, [buf, sink, n16])

                run()
                t = events_ms(run, 10, dev)
                out("bw", kernel=name, block=block, blocks_per_cu=per_cu, occupancy=occ,
                    regs=f.attr(4), gbps=round(nbytes / statistics.median(t) / 1e6, 1),
                    ms=round(statistics.median(t), 3))
    # grid barrier
    f = mod.fn("grid_barrier")
    bar = torch.zeros(1, dtype=torch.int32, device=dev)
    for block, per_cu in ((512, 1), (256, 2)):
        grid = per_cu * cus
        base = [0]

        def run_bar(iters, grid=grid, block=block):
            f.launch(grid, block, [bar, ctypes.c_uint(base[0] & 0xFFFFFFFF), ctypes.c_int(iters)])
            base[0] += iters * grid

        run_bar(10)
        t0_ = statistics.median(events_ms(lambda: run_bar(1), 10, dev))
        t1_ = statistics.median(events_ms(lambda: run_bar(1001), 5, dev))
        out("grid_barrier", grid=grid, block=block, us_per_barrier=round((t1_ - t0_) * 1e3 / 1000, 3))

    # P2P ping-pong between GCD 0 and 1
    if torch.cuda.device_count() >= 2:
        try:
            p2p(rtc, code)
        except Exception as exc:  # noqa: BLE001
            out("p2p", ok=False, error=traceback.format_exc()[-1200:])


def fine_alloc(rtc, dev_index: int, nbytes: int) -> int:
    h = rtc.hip()
    rtc.check(h.hipSetDevice(dev_index), "hipSetDevice")
    ptr = ctypes.c_void_p()
    rtc.check(h.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1)),
              "hipExtMallocWithFlags")
    rtc.check(h.hipMemset(ptr, 0, ctypes.c_size_t(nbytes)), "hipMemset")
    rtc.check(h.hipDeviceSynchronize(), "hipDeviceSynchronize")
    return ptr.value


def enable_peer(rtc, a: int, b: int) -> None:
    h = rtc.hip()
    for x, y in ((a, b), (b, a)):
        rtc.check(h.hipSetDevice(x), "hipSetDevice")
        rc = h.hipDeviceEnablePeerAccess(y, 0)
        if rc not in (0, 704):
            raise RuntimeError(f"hipDeviceEnablePeerAccess({x}->{y}) failed: {rc}")


def p2p(rtc, code) -> None:
    import torch

    devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    enable_peer(rtc, 0, 1)
    words = [fine_alloc(rtc, d.index, 4096) for d in devs]
    mods = [rtc.Module(code, d) for d in devs]
    errs = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    base = [0]

    def run(iters):
        for r in (1, 0):  # the waiter first
            with torch.cuda.device(devs[r]):
                mods[r].fn("p2p_pingpong").launch(1, 64, [
                    ctypes.c_void_p(words[r]), ctypes.c_void_p(words[1 - r]), ctypes.c_int(r),
                    iters, base[0], errs[r]])
        base[0] += 2 * iters + 2
        for d in devs:
            torch.cuda.synchronize(d)

    run(10)
    t0 = time.perf_counter()
    run(1)
    t1 = time.perf_counter()
    run(10001)
    t2 = time.perf_counter()
    rt_us = ((t2 - t1) - (t1 - t0)) / 10000 * 1e6
    out("p2p", ok=int(errs[0].item()) == 0 and int(errs[1].item()) == 0,
        round_trip_us=round(rt_us, 3), one_way_us=round(rt_us / 2, 3))


# ============================================================================== main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("env", "slice"))
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--layers", default="0,1,2")
    args = parser.parse_args()
    try:
        if args.part == "env":
            env_main(args)
        else:
            from tools import hipmega_slice  # noqa: F401  (step (b)/(c), next file)

            hipmega_slice.main(args)
    except Exception:  # noqa: BLE001
        out("error", part=args.part, trace=traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
