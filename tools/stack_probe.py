"""Plan 59 step 0: what a ROCm/torch/Triton stack change does on gfx90a, class by class.

    python tools/stack_probe.py env            # versions, arch list, devices, driver, smoke kernels
    python tools/stack_probe.py bench [--device 0]

Run the same file under two stacks (the image's torch, and another torch put first on
PYTHONPATH) to compare them in one job; every line is a ``RESULT`` with the stack's versions.

``env``: torch / HIP / Triton versions, ``torch.cuda.get_arch_list()``, every visible device's
name and gcnArchName, the amdgpu driver version (/sys/module/amdgpu/version), rocminfo's gfx90a
agents; then on every device a BF16 matmul against an fp32 reference, a trivial Triton kernel
(vector add, checked) and, on device 0, a hipRTC compile for gfx90a through
``engine.fast.hipmega_rtc`` (the path of the engine's HIP GDN kernels) launched and checked.

``bench`` (one GCD, random BF16 weights of the TP2 shard shapes):
(a) streaming bandwidth: ``copy_`` of 1 GiB, a plain Triton read-reduce over 1 GiB, the
    engine's MFMA GEMV (``mfma.gemv``, the chained-verify path) on the in_proj and lm_head
    shard shapes at 4 and 8 rows, and the engine's MoE decode exactly as ``FusedDecode._moe``
    runs it (router|shared GEMV with the top-k epilogue, routed up, routed down + shared +
    residual) at 4 and 8 rows over 4 layers of random expert weights; TB/s over the unique
    bytes each launch reads (MoE: the distinct experts of the rows, from the same router
    logits in torch, plus the shared down and router|shared weights).
    Timed as CUDA graphs of back-to-back calls cycling over buffers larger than L2.
(b) small-kernel latency and launch: eager launches of an empty Triton kernel (host-bound
    us per launch), a 1-program kernel and a 104-program kernel replayed as a 576-node graph
    (us per node: the replay cost of the decode step's node count), and torch's small add.
(c) prefill GEMMs: ``F.linear`` (hipBLASLt/rocBLAS as torch picks) at M = 61, 561, 1650 on
    the TP2 projection shapes, TFLOP/s, and a 4096^3 BF16 GEMM.

A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

STACK = {}


def out(kind: str, **kw) -> None:
    print("RESULT " + json.dumps({"kind": kind, **STACK, **kw}), flush=True)


def versions() -> dict:
    import triton

    return {"torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__}


# ---------------------------------------------------------------------------------- env
def env() -> int:
    import triton
    import triton.language as tl

    ok = True
    drv = Path("/sys/module/amdgpu/version")
    out("versions", arch_list=torch.cuda.get_arch_list(), devices=torch.cuda.device_count(),
        driver=drv.read_text().strip() if drv.exists() else None,
        torch_file=torch.__file__, triton_file=triton.__file__)
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=120)
        names = [ln.split(":", 1)[1].strip() for ln in r.stdout.splitlines() if ln.strip().startswith("Name:")]
        rv = Path("/opt/rocm/.info/version")
        out("rocminfo", rc=r.returncode, gfx90a_agents=sum(n == "gfx90a" for n in names),
            image_rocm=rv.read_text().strip() if rv.exists() else None)
    except Exception as exc:  # noqa: BLE001
        out("rocminfo", error=repr(exc)[:300])

    @triton.jit
    def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    for d in range(torch.cuda.device_count()):
        dev = torch.device("cuda", d)
        p = torch.cuda.get_device_properties(dev)
        rec = {"device": d, "name": p.name, "gcn": getattr(p, "gcnArchName", None),
               "cus": p.multi_processor_count, "mem_gib": round(p.total_memory / 2**30, 1)}
        try:
            torch.manual_seed(d)
            a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
            b = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
            ref = a.float() @ b.float()
            rec["matmul_rel_err"] = float(((a @ b).float() - ref).abs().max() / ref.abs().max())
            x = torch.randn(10_000, device=dev)
            y = torch.randn(10_000, device=dev)
            o = torch.empty_like(x)
            with torch.cuda.device(dev):
                _add[(triton.cdiv(10_000, 1024),)](x, y, o, 10_000, BLOCK=1024)
            torch.cuda.synchronize(dev)
            rec["triton_ok"] = bool(torch.equal(o, x + y))
            ok &= rec["triton_ok"] and rec["matmul_rel_err"] < 0.02
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)[:400]
            ok = False
        out("device", **rec)

    try:
        from engine.fast import hipmega_rtc as rtc

        src = r'''
#include <hip/hip_runtime.h>
extern "C" __global__ void axpy(float* o, const float* x, const float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a * x[i] + y[i];
}
'''
        t0 = time.time()
        rtc.CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / f"stack-probe-rtc-{os.getpid()}"
        code, log = rtc.compile_hip(src, "gfx90a")
        dt = time.time() - t0
        dev = torch.device("cuda", 0)
        mod = rtc.Module(code, dev)
        n = 4096
        x = torch.randn(n, device=dev)
        y = torch.randn(n, device=dev)
        o = torch.empty_like(x)
        mod.fn("axpy").launch(n // 256, 256, [o, x, y, 2.0, ctypes.c_int(n)])
        torch.cuda.synchronize(dev)
        good = bool(torch.allclose(o, 2.0 * x + y))
        out("hiprtc", ok=good, compile_s=round(dt, 2), code_bytes=len(code))
        ok &= good
    except Exception as exc:  # noqa: BLE001
        out("hiprtc", ok=False, error=repr(exc)[:600])
        ok = False
    out("env_verdict", ok=ok)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------- timing
def graph_time(fn, calls: int, reps: int = 5) -> float:
    """us per call: ``calls`` back-to-back ``fn(j)`` captured in one graph, median of reps."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for j in range(3):
            fn(j)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for j in range(calls):
            fn(j)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / calls)
    ts.sort()
    return ts[len(ts) // 2]


def event_time(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def bench(dev_index: int) -> int:
    import triton
    import triton.language as tl

    from engine.fast import fused, mfma

    dev = torch.device("cuda", dev_index)
    torch.cuda.set_device(dev)
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    torch.manual_seed(0)

    # ---------------------------------------------------------------- (a) streaming
    big = torch.empty(2**29, dtype=torch.bfloat16, device=dev).normal_()  # 1 GiB
    dst = torch.empty_like(big)
    us = event_time(lambda: dst.copy_(big))
    out("a_copy", gib=1, us=round(us, 1), tbs_rw=round(2 * big.numel() * 2 / us / 1e6, 3))

    @triton.jit
    def _read(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(pid * BLOCK, n, tl.num_programs(0) * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
        tl.store(o_ptr + pid, tl.sum(acc, axis=0))

    for per_cu, block, warps in ((4, 2048, 4), (8, 4096, 8), (16, 2048, 4)):
        grid = per_cu * cus
        o = torch.empty(grid, device=dev)
        us = event_time(lambda: _read[(grid,)](big, o, big.numel(), BLOCK=block, num_warps=warps))
        out("a_triton_read", grid=grid, block=block, warps=warps, us=round(us, 1),
            tbs=round(big.numel() * 2 / us / 1e6, 3))
    del dst

    for name, n, k in (("in_proj", 6176, 2048), ("lm_head", 124160, 2048), ("o_proj", 2048, 2048)):
        copies = max(2, -(-(256 << 20) // (n * k * 2)))
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        for rows in (4, 8):
            x = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
            o = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
            us = graph_time(lambda j: mfma.gemv(x, ws[j % copies], o), 64)
            out("a_mfma_gemv", shape=name, n=n, k=k, rows=rows, tiles=list(mfma.tiles_for(n, k)),
                us=round(us, 2), tbs=round(n * k * 2 / us / 1e6, 3))
        del ws

    # The MoE decode exactly as ``FusedDecode._moe`` runs it (TP2 rank shard).
    E, H, I2, S2, TOPK, L = 256, 2048, 256, 256, 8, 4
    layers, w_moe = [], {}
    for i in range(L):
        gu = torch.randn(E, 2 * I2, H, device=dev, dtype=torch.bfloat16) * 0.02
        dn = torch.randn(E, H, I2, device=dev, dtype=torch.bfloat16) * 0.02
        sd = torch.randn(H, S2, device=dev, dtype=torch.bfloat16) * 0.02
        layers.append(NS(mlp=NS(experts=NS(gate_up_proj=gu, down_proj=dn),
                                shared_expert=NS(down_proj=NS(weight=sd)))))
        w_moe[i] = torch.randn(E + 2 * S2 + 1, H, device=dev, dtype=torch.bfloat16) * 0.02
    fd = fused.FusedDecode.__new__(fused.FusedDecode)
    fd.fast = NS(top_k=TOPK, layers=layers, mtp=None)
    fd.w_moe, fd.cus = w_moe, {}
    out("a_moe_config", order=fused.MOE_ORDER, grid=fused.MOE_GRID, tile_rows=fused.MOE_TILE_ROWS,
        topk_epi=mfma.TOPK_EPI, mfma_gemv=mfma.GEMV, glue_topk=fused.GLUE_TOPK, bn=fused.TILES["moe_bn"])
    per_expert = (2 * I2 * H + H * I2) * 2
    fixed = (H * S2 + w_moe[0].numel()) * 2
    for rows in (4, 8):
        sets = 4
        hs = [torch.randn(rows, H, device=dev, dtype=torch.bfloat16) for _ in range(sets)]
        xs = [torch.randn(rows, H, device=dev, dtype=torch.bfloat16) for _ in range(sets)]
        uniq = []
        for i in range(L):
            for s in range(sets):
                logits = (hs[s] @ w_moe[i][:E].t()).float()
                uniq.append(int(torch.unique(logits.topk(TOPK, dim=-1).indices).numel()))
        nb = sum(u * per_expert + fixed for u in uniq) / len(uniq)
        calls = L * sets * 4

        def call(j, hs=hs, xs=xs):
            fd._moe(j % L, xs[(j // L) % sets], hs[(j // L) % sets])

        us = graph_time(call, calls)
        out("a_moe_decode", rows=rows, unique_experts_mean=round(sum(uniq) / len(uniq), 1),
            mb_per_call=round(nb / 1e6, 2), us=round(us, 2), tbs=round(nb / us / 1e6, 3))
    del layers, w_moe, fd, big
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- (b) small kernels
    @triton.jit
    def _tiny(o_ptr, v):
        tl.store(o_ptr + tl.program_id(0), v)

    o = torch.zeros(1024, device=dev)
    for _ in range(10):
        _tiny[(1,)](o, 1.0)
    torch.cuda.synchronize()
    n = 3000
    t0 = time.perf_counter()
    for _ in range(n):
        _tiny[(1,)](o, 1.0)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    out("b_eager_triton_launch", launches=n, host_us_per_launch=round((t1 - t0) * 1e6 / n, 2),
        wall_us_per_launch=round((t2 - t0) * 1e6 / n, 2))
    a = torch.zeros(64, device=dev)
    for _ in range(10):
        a.add_(1.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        a.add_(1.0)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    out("b_eager_torch_add", launches=n, host_us_per_launch=round((t1 - t0) * 1e6 / n, 2),
        wall_us_per_launch=round((t2 - t0) * 1e6 / n, 2))
    for progs in (1, cus):
        us = graph_time(lambda j: _tiny[(progs,)](o, float(j)), 576)
        out("b_graph_node", programs=progs, nodes=576, us_per_node=round(us, 3),
            step_us=round(us * 576, 1))
    us = graph_time(lambda j: a.add_(1.0), 576)
    out("b_graph_node_torch_add", nodes=576, us_per_node=round(us, 3))

    # ---------------------------------------------------------------- (c) prefill GEMMs
    import torch.nn.functional as F

    shapes = (("in_proj", 6176, 2048), ("out_proj", 2048, 2048), ("qkv", 4608, 2048),
              ("router_shared", 769, 2048), ("mtp_fc", 2048, 4096))
    for m in (61, 561, 1650):
        tot_us, tot_fl = 0.0, 0.0
        for name, n, k in shapes:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
            us = graph_time(lambda j: F.linear(x, w), 32)
            fl = 2 * m * n * k
            tot_us += us
            tot_fl += fl
            out("c_gemm", m=m, shape=name, n=n, k=k, us=round(us, 2), tflops=round(fl / us / 1e6, 1))
        out("c_gemm_sum", m=m, us=round(tot_us, 1), tflops=round(tot_fl / tot_us / 1e6, 1))
    x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    w = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    us = graph_time(lambda j: x @ w, 16)
    out("c_gemm_4096", us=round(us, 1), tflops=round(2 * 4096**3 / us / 1e6, 1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("env", "bench"))
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    STACK.update(versions())
    return env() if a.what == "env" else bench(a.device)


if __name__ == "__main__":
    sys.exit(main())
