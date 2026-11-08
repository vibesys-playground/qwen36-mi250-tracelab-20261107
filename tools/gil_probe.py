"""Plan 30 step 0: does sharing one GIL between the two TP2 ranks set the prefill pass floor,
and can the in-graph all-reduce's fine-grained buffer be shared between processes?

    python tools/gil_probe.py all [--devices 0,1] [--reps 12]
    python tools/gil_probe.py ipc        # two processes on the visible GPUs 0 and 1
    python tools/gil_probe.py passes     # one process, ENGINE_TP=2, the visible GPUs

``passes`` loads the replica as the server does (``Generator.load``: warmup, graphs), then
resumes a 3,600-token context (prefilled with both ranks) and times one pass of N new tokens
(N = 61, 561, 1,650; ``TPModel.prefill_packed``, the server's pass), synchronized wall,
median of ``--reps``, in four modes, interleaved rep by rep:

- ``both``: today's path, ``TPModel._both`` (rank 0 here, rank 1 on its thread, the eager
  all-reduce through the host mailbox).
- ``solo0``: rank 0 alone on this thread with ``Group.solo`` (the all-reduces return the
  local partial: the same kernels, wrong numbers). Stand-in for a rank in its own process.
- ``solo1``: rank 1 alone on this thread, likewise.
- ``solo_threads``: both ranks on their threads with ``Group.solo``: the GIL is shared, the
  rendezvous is gone.

One profiled pass per mode at 561 tokens gives the per-GCD kernel time and launch count.
Then the in-process baselines: today's eager all-reduce (``Group.allreduce`` from both
threads) at 8 KB and at 561 x 2048 BF16, and the in-graph LL all-reduce (``_LL``) at 8 KB
in a graph of 100 calls per GCD.

``ipc`` spawns two processes (rank r on GPU r, both GPUs visible): each allocates the LL
receive buffers with ``hipExtMallocWithFlags`` (fine-grained), exports them with
``hipIpcGetMemHandle``, opens the peer's with ``hipIpcOpenMemHandle``, and runs the unchanged
``_ll_allreduce_kernel`` across the processes: 1,000 eager calls at 8 KB checked bit for bit
against ``bf16(fp32(x0) + fp32(x1))``, then 1,000 calls in graphs (10 replays of 100) for the
time; then one prefill-size exchange (561 x 2048 BF16) with the same kernel on a larger
buffer, checked and timed. If the fine-grained export fails, the same on ``hipMalloc``
memory. Then RCCL (torch.distributed) across the two processes: 8 KB eager (checked) and in
a graph, and 561 x 2048 in BF16 and fp32.

``all`` runs ``ipc`` then ``passes`` in child processes and prints the kill rules of plan 30:
park if solo0 / both at 561 tokens >= 0.8, or if IPC fails and RCCL adds > 0.3 ms per
verify step (RCCL graph time - LL graph time) x all-reduces per step.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CONTEXT = 3600
NEW = (61, 561, 1650)
MODES = ("both", "solo0", "solo1", "solo_threads")
HIDDEN = 2048
PREFILL_ROWS = 561
KILL_RATIO = 0.8
KILL_VERIFY_MS = 0.3


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


# =============================================================================== ipc


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]  # hipIpcMemHandle_t (HIP_IPC_HANDLE_SIZE)


def _alloc(hip, dev: int, nbytes: int, fine: bool) -> int:
    ptr = ctypes.c_void_p()
    if hip.hipSetDevice(dev) != 0:
        raise RuntimeError("hipSetDevice failed")
    if fine:
        rc = hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1))
    else:
        rc = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if rc != 0:
        raise RuntimeError(f"{'hipExtMallocWithFlags' if fine else 'hipMalloc'} failed: {rc}")
    if hip.hipMemset(ptr, 0, ctypes.c_size_t(nbytes)) != 0 or hip.hipDeviceSynchronize() != 0:
        raise RuntimeError("hipMemset failed")
    return ptr.value


def _export(hip, ptr: int) -> bytes:
    h = _Handle()
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_Handle), ctypes.c_void_p]
    hip.hipIpcGetMemHandle.restype = ctypes.c_int
    rc = hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr))
    if rc != 0:
        raise RuntimeError(f"hipIpcGetMemHandle failed: {rc}")
    return ctypes.string_at(ctypes.addressof(h), 64)  # never h.reserved: it stops at a NUL


def _open(hip, handle: bytes, info: dict) -> int:
    """Open a peer's handle: the prototype declared (the 64-byte handle is passed by value),
    with hipIpcMemLazyEnablePeerAccess (1), then with no flags; every failure is recorded."""
    fn = hip.hipIpcOpenMemHandle
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), _Handle, ctypes.c_uint]
    fn.restype = ctypes.c_int
    h = _Handle()
    ctypes.memmove(ctypes.addressof(h), handle, 64)
    tried = []
    for flags in (1, 0):
        ptr = ctypes.c_void_p()
        rc = fn(ctypes.byref(ptr), h, flags)
        if rc == 0:
            info.setdefault("open_flags", []).append(flags)
            return ptr.value
        hip.hipGetErrorName.restype = ctypes.c_char_p
        name = (hip.hipGetErrorName(rc) or b"").decode()
        tried.append(f"flags={flags}: rc={rc} {name}")
        hip.hipGetLastError()
    raise RuntimeError("hipIpcOpenMemHandle failed: " + "; ".join(tried))


def _ipc_worker(rank: int, conn, barrier, port: int) -> None:
    rows: list[dict] = []

    def emit(kind: str, **fields) -> None:
        row = {"kind": kind, "rank": rank, **fields}
        rows.append(row)
        print("RESULT " + json.dumps(row), flush=True)

    try:
        _ipc_body(rank, conn, barrier, port, emit)
    except Exception as exc:  # noqa: BLE001 - report, let the peer time out
        emit("ipc_error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass
    # Leave without tearing down RCCL or the IPC mappings (j139: the workers did not exit).
    sys.stdout.flush()
    os._exit(0)


def _ipc_body(rank: int, conn, barrier, port: int, emit) -> None:
    import torch
    import torch.multiprocessing  # noqa: F401 - registers the CUDA tensor reductions for the pipe

    from engine.fast import tp as tpm

    peer_rank = 1 - rank
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    hip = tpm._hip()
    rc = hip.hipDeviceEnablePeerAccess(peer_rank, 0)
    if hip.hipSetDevice(rank) != 0:
        raise RuntimeError("hipSetDevice failed")
    emit("ipc_peer_access", rc=rc)
    chunk = tpm.CHUNK
    cap_small = tpm.CAPACITY
    cap_big = -(-(PREFILL_ROWS * HIDDEN // 2) // chunk) * chunk

    def sync() -> None:
        torch.cuda.synchronize(dev)

    def wait(tag: str) -> None:
        barrier.wait(timeout=300)

    # ---------------------------------------------------------------- share the buffers
    # fine: hipExtMallocWithFlags(fine-grained) + hipIpc via ctypes; coarse: hipMalloc + hipIpc
    # via ctypes; torch: torch's allocator, shared by torch.multiprocessing (its own hipIpc).
    shared = None
    keep = []
    for name in ("fine", "coarse", "torch"):
        info = {"mem": name}
        hs = hb = None
        try:
            if name == "torch":
                small_t = torch.zeros(2 * cap_small, dtype=torch.int64, device=dev)
                big_t = torch.zeros(2 * cap_big, dtype=torch.int64, device=dev)
                keep += [small_t, big_t]
                small, big = small_t.data_ptr(), big_t.data_ptr()
                hs, hb = small_t, big_t
            else:
                small = _alloc(hip, rank, 2 * cap_small * 8, name == "fine")
                big = _alloc(hip, rank, 2 * cap_big * 8, name == "fine")
                t0 = time.perf_counter()
                hs, hb = _export(hip, small), _export(hip, big)
                info["export_us"] = round((time.perf_counter() - t0) * 1e6, 1)
            ok = True
        except Exception as exc:  # noqa: BLE001
            info["export_error"] = f"{type(exc).__name__}: {exc}"
            hs = hb = None
            ok = False
        torch.cuda.synchronize(dev)
        try:
            conn.send((ok, hs, hb))
            peer_ok, phs, phb = conn.recv()
        except Exception as exc:  # noqa: BLE001
            info["send_error"] = f"{type(exc).__name__}: {exc}"
            conn.send(False)
            conn.recv()
            info["ok"] = False
            emit("ipc_share", **info)
            continue
        if ok and peer_ok:
            try:
                t0 = time.perf_counter()
                if name == "torch":
                    keep += [phs, phb]
                    info["peer_tensor_device"] = str(phs.device)
                    psmall, pbig = phs.data_ptr(), phb.data_ptr()
                else:
                    psmall, pbig = _open(hip, phs, info), _open(hip, phb, info)
                info["open_us"] = round((time.perf_counter() - t0) * 1e6, 1)
            except Exception as exc:  # noqa: BLE001
                info["open_error"] = f"{type(exc).__name__}: {exc}"
                ok = False
        conn.send(ok)
        peer_open = conn.recv()
        info["ok"] = bool(ok and peer_ok and peer_open)
        emit("ipc_share", **info)
        hip.hipSetDevice(rank)
        torch.cuda.set_device(rank)
        if info["ok"] and name == "torch":
            # j146: torch maps the peer's memory for the peer's GPU only; the LL kernel on this
            # GPU faulted writing to it. Record that it opens, and do not run the kernel on it.
            emit("ipc_torch_mapping_not_used", why="mapped for the owner GPU only (j146 fault)")
            break
        if info["ok"]:
            shared = (name, small, big, psmall, pbig)
            break
    torch.cuda.set_device(rank)
    if shared is not None:
        _ll_tests(rank, dev, tpm, shared, cap_small, cap_big, chunk, sync, wait, emit)
    else:
        emit("ipc_ll_skipped", why="no buffer kind could be shared")
    # ---------------------------------------------------------------- RCCL across processes
    try:
        _rccl_tests(rank, dev, port, sync, emit)
    except Exception as exc:  # noqa: BLE001
        emit("rccl_error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1200:])


def _pairs(shape, count: int, dtype, dev):
    """Deterministic inputs of both ranks and their expected sum ``bf16(fp32 a + fp32 b)``."""
    import torch

    g = torch.Generator().manual_seed(1234)
    xs = [[], []]
    exp = []
    for _ in range(count):
        a = torch.randn(shape, generator=g).to(dtype)
        b = torch.randn(shape, generator=g).to(dtype)
        xs[0].append(a.to(dev))
        xs[1].append(b.to(dev))
        exp.append((a.float() + b.float()).to(dtype).to(dev))
    return xs, exp


def _ll_tests(rank, dev, tpm, shared, cap_small, cap_big, chunk, sync, wait, emit) -> None:
    import torch

    mem, small, big, psmall, pbig = shared
    triton = tpm.triton
    kernel = tpm._ll_allreduce_kernel
    err = torch.zeros(1, dtype=torch.int32, device=dev)

    def make(cap: int, local: int, peer: int):
        epoch = torch.zeros(cap // chunk, dtype=torch.int32, device=dev)
        lraw, praw = tpm._Raw(local, torch.int64), tpm._Raw(peer, torch.int64)

        def launch(x, out):
            nw = x.numel() * x.element_size() // 4
            kernel[(triton.cdiv(nw, chunk),)](
                x.view(-1).view(torch.int32), x.view(-1).view(torch.int32), out.view(-1).view(torch.int32),
                lraw, praw, epoch, err, nw, cap, CHUNK=chunk, BF16=x.dtype == torch.bfloat16, RES=False,
                MAXIT=tpm.MAXIT, num_warps=1,
            )

        return launch

    # ---- 8 KB, eager, checked
    ll = make(cap_small, small, psmall)
    xs, exp = _pairs((2, HIDDEN), 8, torch.bfloat16, dev)
    out = torch.empty_like(xs[0][0])
    bad = torch.zeros((), dtype=torch.int64, device=dev)
    ll(xs[rank][0], out)  # compile (the first call may time out waiting for the peer's compile)
    sync()
    wait("ll8k")
    err.zero_()
    iters = 1000
    t0 = time.perf_counter()
    for i in range(iters):
        k = i % 8
        ll(xs[rank][k], out)
        bad += (out.view(torch.int16) != exp[k].view(torch.int16)).sum()
    sync()
    eager_us = (time.perf_counter() - t0) / iters * 1e6
    emit("ipc_ll_8k_eager", mem=mem, calls=iters, mismatched_elems=int(bad.item()),
         timeouts=int(err.item()), us_per_call_incl_check=round(eager_us, 2))
    # ---- 8 KB, in graphs: 10 replays of 100 calls, inputs cycling, last output checked
    stream = torch.cuda.Stream(dev)
    graph = torch.cuda.CUDAGraph()
    gout = [torch.empty_like(xs[0][0]) for _ in range(8)]
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for i in range(100):
                ll(xs[rank][i % 8], gout[i % 8])
    sync()
    wait("ll8k-graph-warm")
    graph.replay()
    sync()
    wait("ll8k-graph")
    t0 = time.perf_counter()
    for _ in range(10):
        graph.replay()
    sync()
    graph_us = (time.perf_counter() - t0) / 1000 * 1e6
    gbad = sum(int((gout[k].view(torch.int16) != exp[k].view(torch.int16)).sum()) for k in range(8))
    emit("ipc_ll_8k_graph", mem=mem, calls=1100, us_per_call=round(graph_us, 2), mismatched_elems_last=gbad,
         timeouts=int(err.item()))
    # ---- 561 x 2048 BF16 (one prefill all-reduce), eager, checked and timed
    llb = make(cap_big, big, pbig)
    xb, eb = _pairs((PREFILL_ROWS, HIDDEN), 2, torch.bfloat16, dev)
    ob = torch.empty_like(xb[0][0])
    llb(xb[rank][0], ob)
    sync()
    wait("llbig")
    err.zero_()
    bad.zero_()
    lat = []
    for i in range(20):
        wait("llbig-one")
        t0 = time.perf_counter()
        llb(xb[rank][i % 2], ob)
        sync()
        lat.append((time.perf_counter() - t0) * 1e6)
        bad += (ob.view(torch.int16) != eb[i % 2].view(torch.int16)).sum()
    sync()
    wait("llbig-loop")
    t0 = time.perf_counter()
    for i in range(50):
        llb(xb[rank][i % 2], ob)
    sync()
    loop_us = (time.perf_counter() - t0) / 50 * 1e6
    emit("ipc_ll_561x2048", mem=mem, mismatched_elems=int(bad.item()), timeouts=int(err.item()),
         one_shot_us_median=round(statistics.median(lat), 1), back_to_back_us=round(loop_us, 1))


def _rccl_tests(rank, dev, port, sync, emit) -> None:
    import datetime

    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            device_id=dev, timeout=datetime.timedelta(seconds=180))
    xs, exp = _pairs((2, HIDDEN), 8, torch.bfloat16, dev)
    buf = torch.empty_like(xs[0][0])
    for _ in range(20):
        buf.copy_(xs[rank][0])
        dist.all_reduce(buf)
    sync()
    dist.barrier()
    bad = torch.zeros((), dtype=torch.int64, device=dev)
    iters = 1000
    t0 = time.perf_counter()
    for i in range(iters):
        k = i % 8
        buf.copy_(xs[rank][k])
        dist.all_reduce(buf)
        bad += (buf.view(torch.int16) != exp[k].view(torch.int16)).sum()
    sync()
    eager_us = (time.perf_counter() - t0) / iters * 1e6
    emit("rccl_8k_eager", calls=iters, mismatched_elems_vs_fp32_sum=int(bad.item()),
         us_per_call_incl_copy_check=round(eager_us, 2))
    try:
        s = torch.cuda.Stream(dev)
        g = torch.cuda.CUDAGraph()
        x = xs[rank][0].clone()
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                for _ in range(100):
                    dist.all_reduce(x)
        sync()
        dist.barrier()
        g.replay()
        sync()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            g.replay()
        sync()
        emit("rccl_8k_graph", calls=1000, us_per_call=round((time.perf_counter() - t0) / 1000 * 1e6, 2))
    except Exception as exc:  # noqa: BLE001
        emit("rccl_8k_graph", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn((PREFILL_ROWS, HIDDEN), device=dev).to(dtype)
        for _ in range(5):
            dist.all_reduce(x)
        sync()
        lat = []
        for _ in range(20):
            dist.barrier()
            sync()
            t0 = time.perf_counter()
            dist.all_reduce(x)
            sync()
            lat.append((time.perf_counter() - t0) * 1e6)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(50):
            dist.all_reduce(x)
        sync()
        emit("rccl_561x2048", dtype=str(dtype).split(".")[-1], one_shot_us_median=round(statistics.median(lat), 1),
             back_to_back_us=round((time.perf_counter() - t0) / 50 * 1e6, 1))
    dist.barrier()


def ipc(args) -> int:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    barrier = ctx.Barrier(2)
    port = 29600 + os.getpid() % 1000
    procs = [ctx.Process(target=_ipc_worker, args=(r, c, barrier, port)) for r, c in ((0, a), (1, b))]
    for p in procs:
        p.start()
    deadline = time.time() + args.timeout
    for p in procs:
        p.join(max(1.0, deadline - time.time()))
    for p in procs:
        if p.is_alive():
            print(f"ipc: worker {p.pid} still running after {args.timeout} s: killing it", flush=True)
            p.kill()
            result("ipc_hang", pid=p.pid)
    print(f"ipc: exit codes {[p.exitcode for p in procs]}", flush=True)
    return 0


# =============================================================================== passes


def _kernel_stats(path: str) -> dict:
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    per_dev: dict[str, list] = defaultdict(list)
    ksum: Counter = Counter()
    kcount: Counter = Counter()
    launches = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            a, d = float(e["ts"]), float(e.get("dur", 0))
            dev = str((e.get("args") or {}).get("device", e.get("pid")))
            per_dev[dev].append((a, a + d))
            if cat == "kernel":
                ksum[dev] += d
                kcount[dev] += 1
        elif cat in ("cuda_runtime", "cuda_driver") and "launch" in e.get("name", "").lower():
            launches += 1
    busy = {}
    for dev, iv in per_dev.items():
        total, end = 0.0, None
        for a, b in sorted(iv):
            if end is None or a > end:
                total += b - a
                end = b
            elif b > end:
                total += b - end
                end = b
        busy[dev] = round(total / 1e3, 2)
    return {"kernel_ms": {d: round(v / 1e3, 2) for d, v in ksum.items()}, "kernels": dict(kcount),
            "busy_ms": busy, "launch_calls": launches}


def passes(args) -> int:
    import tempfile

    import torch
    from torch.profiler import ProfilerActivity, profile

    from engine.fast.tp import _on
    from engine.generator import CHECKPOINT, Generator
    from serve_probe import token_pool

    print(f"passes: ENGINE_TP={os.environ.get('ENGINE_TP')} visible={os.environ.get('HIP_VISIBLE_DEVICES')} "
          f"torch {torch.__version__} hip {torch.version.hip} gpus {torch.cuda.device_count()} "
          f"python {sys.version.split()[0]}", flush=True)
    t0 = time.perf_counter()
    gen = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
                         prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30)
    fast = gen.fast
    if fast is None or not hasattr(fast, "ranks"):
        raise RuntimeError("the TP2 fast path did not load")
    result("load", seconds=round(time.perf_counter() - t0, 1), graphs=fast.use_graphs,
           mtp=fast.mtp is not None, switch_interval_s=float(os.environ.get("ENGINE_TP_SWITCH_S", "0.0002")))
    group = fast.group
    r0, r1 = fast.ranks
    slot = gen._free_slots[-1]
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 40_000, "synthetic")
    devices = list(range(torch.cuda.device_count()))
    serial = [0]

    def sync() -> None:
        for d in devices:
            torch.cuda.synchronize(d)

    def context(new: int) -> list[int]:
        serial[0] += 1
        off = (serial[0] * 7919) % (len(pool) - CONTEXT - new - 1)
        ids = pool[off : off + CONTEXT + new]
        fast.reset(slot)
        fast.prefill_packed([(slot, ids[:2048], 0)])
        fast.prefill_packed([(slot, ids[2048:CONTEXT], 2048)])
        return ids[CONTEXT:]

    def run(mode: str, new_ids: list[int]) -> None:
        segs = [(slot, new_ids, CONTEXT)]
        if mode == "both":
            fast.prefill_packed(segs)
            return
        r0._ensure_pages(slot, CONTEXT + len(new_ids))
        group.solo = True
        try:
            with torch.inference_mode():
                if mode == "solo_threads":
                    fast._both(lambda m: m.prefill_packed(segs))
                else:
                    model = r0 if mode == "solo0" else r1
                    with _on(model):
                        model.prefill_packed(segs)
        finally:
            group.solo = False

    def timed(mode: str, new: int, wrap=None) -> float:
        new_ids = context(new)
        sync()
        a = time.perf_counter()
        if wrap is None:
            run(mode, new_ids)
        else:
            wrap(lambda: run(mode, new_ids))
        sync()
        return (time.perf_counter() - a) * 1e3

    for _ in range(2):  # warm every shape and mode
        for new in NEW:
            for mode in MODES:
                timed(mode, new)
    # The number of all-reduces in one pass (per rank).
    calls = Counter()
    orig = group.allreduce

    def counting(rank, x, residual=None):
        calls[rank] += 1
        return orig(rank, x, residual)

    def counted(fn):  # only the measured pass (the context's passes run before the wrap)
        group.allreduce = counting
        try:
            fn()
        finally:
            group.allreduce = orig

    timed("both", 561, counted)
    result("allreduce_calls_per_pass", per_rank=dict(calls))
    walls: dict[tuple[str, int], list[float]] = defaultdict(list)
    for _ in range(args.reps):
        for new in NEW:
            for mode in MODES:
                walls[(mode, new)].append(timed(mode, new))
    table = {}
    for (mode, new), v in sorted(walls.items(), key=lambda kv: (kv[0][1], MODES.index(kv[0][0]))):
        med = statistics.median(v)
        table[(mode, new)] = med
        result("pass", mode=mode, new_tokens=new, context=CONTEXT, reps=len(v), wall_ms_median=round(med, 2),
               wall_ms_min=round(min(v), 2), wall_ms_max=round(max(v), 2), wall_ms_all=[round(x, 1) for x in v])
    # One profiled pass per mode at 561 tokens: per-GCD kernel time and launches.
    tmp = tempfile.mkdtemp(prefix="gil-trace-")
    for mode in MODES:
        path = os.path.join(tmp, f"{mode}.json")

        def profiled(fn):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                fn()
                sync()
            prof.export_chrome_trace(path)

        try:
            ms = timed(mode, 561, profiled)
            result("profiled_pass", mode=mode, new_tokens=561, profiled_wall_ms=round(ms, 1), **_kernel_stats(path))
        except Exception as exc:  # noqa: BLE001
            result("profiled_pass", mode=mode, error=f"{type(exc).__name__}: {exc}")
    fast.reset(slot)
    # Today's eager all-reduce (two threads, host mailbox) at 8 KB and at one prefill's size.
    for rows, iters in ((2, 1000), (PREFILL_ROWS, 50)):
        xs = [torch.randn((rows, HIDDEN), device=m.embed.device).to(torch.bfloat16) for m in fast.ranks]
        fast._both(lambda m: [m.tp.allreduce(m.rank, xs[m.rank]) for _ in range(5)])
        sync()
        a = time.perf_counter()
        fast._both(lambda m: [m.tp.allreduce(m.rank, xs[m.rank]) for _ in range(iters)])
        sync()
        result("thread_eager_allreduce", rows=rows, hidden=HIDDEN, dtype="bfloat16", calls=iters,
               us_per_call=round((time.perf_counter() - a) / iters * 1e6, 1))
    # Today's in-graph LL all-reduce (one process) at 8 KB: a graph of 100 calls per GCD.
    try:
        ll = group.ll
        graphs, outs = [], []
        for rank, m in enumerate(fast.ranks):
            dev = m.embed.device
            x = torch.randn((2, HIDDEN), device=dev).to(torch.bfloat16)
            s = torch.cuda.Stream(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.device(dev):
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
                    for _ in range(100):
                        outs.append(ll._launch(rank, x, None))
            graphs.append((dev, g))
        sync()

        def replay(n):
            for _ in range(n):
                for dev, g in graphs:
                    with torch.cuda.device(dev):
                        g.replay()
            sync()

        replay(1)
        a = time.perf_counter()
        replay(10)
        result("thread_ll_8k_graph", calls=1000, us_per_call=round((time.perf_counter() - a) / 1000 * 1e6, 2),
               timeouts=group.timeouts())
    except Exception as exc:  # noqa: BLE001
        result("thread_ll_8k_graph", error=f"{type(exc).__name__}: {exc}")
    return 0


# =============================================================================== all


def _child(cmd: list[str], env: dict, timeout: int) -> list[dict]:
    print(f"all: {' '.join(cmd)}", flush=True)
    rows = []
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    start = time.time()
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith("RESULT "):
            try:
                rows.append(json.loads(line[7:]))
            except json.JSONDecodeError:
                pass
        if time.time() - start > timeout:
            proc.kill()
            print("all: child timed out", flush=True)
            break
    proc.wait()
    print(f"all: exited {proc.returncode}", flush=True)
    return rows


def run_all(args) -> int:
    started = time.time()
    subprocess.run(["sh", "-c", "rocm-smi --showuse --showmemuse 2>&1 | head -30; hostname"], check=False)
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = args.devices
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["ENGINE_TP"] = "2"
    me = str(Path(__file__).resolve())
    rows = []
    if "ipc" in args.parts:
        rows += _child([sys.executable, me, "ipc", "--timeout", "900"], env, 1000)
    if "passes" in args.parts:
        rows += _child([sys.executable, me, "passes", "--reps", str(args.reps)], env, 5400)
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    print("\n=== prefill pass wall (ms, median), 3,600-token context ===", flush=True)
    walls = {(r["mode"], r["new_tokens"]): r["wall_ms_median"] for r in by["pass"]}
    print(f"{'new':>5} " + " ".join(f"{m:>13}" for m in MODES) + f" {'solo0/both':>10}")
    for new in NEW:
        both = walls.get(("both", new))
        s0 = walls.get(("solo0", new))
        ratio = round(s0 / both, 3) if both and s0 else None
        print(f"{new:>5} " + " ".join(f"{walls.get((m, new), float('nan')):>13.1f}" for m in MODES) + f" {ratio!s:>10}")
    both561, solo561 = walls.get(("both", 561)), walls.get(("solo0", 561))
    ratio = solo561 / both561 if both561 and solo561 else None
    result("kill_ratio", threshold=KILL_RATIO, solo0_over_both_at_561=None if ratio is None else round(ratio, 3),
           verdict=None if ratio is None else ("KILL" if ratio >= KILL_RATIO else "PASS"))
    shared = [r for r in by["ipc_share"] if r.get("ok")]
    ll8 = [r for r in by["ipc_ll_8k_eager"]]
    llg = [r for r in by["ipc_ll_8k_graph"] if "us_per_call" in r]
    ipc_ok = (len({r["rank"] for r in shared}) == 2 and len(ll8) == 2
              and all(r["mismatched_elems"] == 0 and r["timeouts"] == 0 for r in ll8)
              and len(llg) == 2 and all(r["mismatched_elems_last"] == 0 for r in llg))
    mem = shared[0]["mem"] if shared else None
    calls = max((by["allreduce_calls_per_pass"][0]["per_rank"].values() if by["allreduce_calls_per_pass"] else [84]),
                default=84)
    rg = [r["us_per_call"] for r in by["rccl_8k_graph"] if "us_per_call" in r]
    ll_graph = max(r["us_per_call"] for r in llg) if llg else None
    tl = [r["us_per_call"] for r in by["thread_ll_8k_graph"] if "us_per_call" in r]
    base = tl[0] if tl else ll_graph
    added = (max(rg) - base) * calls / 1e3 if rg and base is not None else None
    result("kill_ipc", ipc_ok=ipc_ok, mem=mem, ll_graph_us_cross_process=ll_graph,
           ll_graph_us_in_process=tl[0] if tl else None, rccl_graph_us=max(rg) if rg else None,
           allreduces_per_step=calls, rccl_added_ms_per_verify_step=None if added is None else round(added, 3),
           threshold_ms=KILL_VERIFY_MS,
           verdict=("PASS (IPC works)" if ipc_ok else
                    "KILL" if added is None or added > KILL_VERIFY_MS else "PASS (RCCL fallback)"))
    print(f"all: done in {time.time() - started:.0f} s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_all = sub.add_parser("all")
    p_all.add_argument("--devices", default="0,1")
    p_all.add_argument("--reps", type=int, default=12)
    p_all.add_argument("--parts", default="ipc,passes")
    p_ipc = sub.add_parser("ipc")
    p_ipc.add_argument("--timeout", type=int, default=900)
    p_pass = sub.add_parser("passes")
    p_pass.add_argument("--reps", type=int, default=12)
    args = parser.parse_args()
    if args.cmd == "all":
        return run_all(args)
    if args.cmd == "ipc":
        return ipc(args)
    return passes(args)


if __name__ == "__main__":
    sys.exit(main())
