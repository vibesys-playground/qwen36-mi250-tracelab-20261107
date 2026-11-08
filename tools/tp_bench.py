"""Plan 13 step 0: is tensor parallelism over the two GCDs of a replica (TP2) worth building?

    python tools/tp_bench.py [--part all|topo|ar|rccl|layers] [--step-ms 11.55]

``all`` (default) runs every part in its own subprocess (a GPU fault or hang in one part
cannot take the others down) with ``HIP_VISIBLE_DEVICES=0,1`` (replica 0 of
``engine.router.resolve_layout``'s default 4 x 2 layout), then prints the kill-rule verdicts
and the estimated TP2 decode step. All results are ``RESULT`` lines.

(a) ``topo``: ``rocm-smi --showtopo`` (and the KFD sysfs io_links as a fallback), the PCI bus
    of each visible device, peer access, and the dev0 -> dev1 copy bandwidth.
(b) ``ar``: a one-shot peer-to-peer all-reduce between GCD0 and GCD1 written in Triton. Each
    rank writes its partial into the peer's receive buffer (double-buffered by epoch parity),
    and reads the peer's partial out of its own; out = x0 + x1 in fp32, identical on both
    ranks. The epoch is a per-CTA device counter the kernel bumps itself, so the kernel is
    graph-capturable with no host involvement. Two protocols:
      ``ll``   - every 8-byte word carries 4 data bytes and the 4-byte epoch (NCCL's LL
                 protocol): the reader spins until every word of its chunk has this epoch;
                 no fences needed, payload doubles.
      ``flag`` - data stores, then a system-scope release exchange of the peer's per-CTA flag;
                 the reader spins on its own flag with system-scope acquire atomics.
    Buffers (receive and flag) from torch's allocator (coarse-grained), or
    ``hipExtMallocWithFlags`` fine-grained or uncached. Correctness: a fill kernel writes
    epoch-dependent exact values before each all-reduce and a check kernel compares after it,
    eagerly and inside HIP graphs (many thousand iterations), so a stale read shows up as a
    mismatch. Time: a HIP graph per GCD of 80 back-to-back all-reduces (one decode step's
    worth), both replayed from the host back to back.
    ``rccl``: torch.distributed (RCCL) all_reduce between two processes, eager and in a graph.
(c) ``layers``: real weights of layers 0 (Gated DeltaNet) and 3 (attention) from the HF cache,
    full width against TP2's half (GDN 8 of 16 key heads / 16 of 32 value heads; attention 8
    of 16 query heads and 1 of 2 KV heads; every routed and the shared expert's intermediate
    256 of 512; router replicated), through ``engine.fast.fused.FusedDecode``'s kernels, at
    B=1/2/4, in a HIP graph of 10 calls on 10 different inputs. Also the lm_head GEMV full
    vs half vocab.

Kill rule (plan 13): park if the all-reduce takes > 12 us at 8 KB, or a half-width layer
takes > 0.65x the full-width time at B=2.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
STEPS_PER_AR = 80  # 2 all-reduces per layer x 40 layers


def _hip():
    import torch

    path = Path(torch.__file__).parent / "lib" / "libamdhip64.so"
    return ctypes.CDLL(str(path) if path.exists() else "libamdhip64.so")


# =============================================================================== (a) topology


def part_topo() -> int:
    for cmd in (["rocm-smi", "--showbus"], ["rocm-smi", "--showtopo"],
                ["/opt/rocm/bin/rocm-smi", "--showtopo"], ["amd-smi", "topology"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(f"RESULT topo_cmd {' '.join(cmd)} rc={out.returncode}")
            print(out.stdout[-6000:])
            if out.returncode == 0 and cmd[-1] == "--showtopo":
                break
        except Exception as exc:  # noqa: BLE001
            print(f"RESULT topo_cmd {' '.join(cmd)} unavailable: {exc}")
    # KFD sysfs: one node per CPU/GPU agent; io_links carry type (11 = XGMI), weight, bandwidth.
    base = Path("/sys/class/kfd/kfd/topology/nodes")
    try:
        gpu_nodes = {}
        for node in sorted(base.iterdir(), key=lambda p: int(p.name)):
            props = dict(line.split(None, 1) for line in (node / "properties").read_text().splitlines()
                         if " " in line)
            if int(props.get("simd_count", "0")) == 0:
                continue
            gpu_nodes[node.name] = props
            loc = int(props.get("location_id", "0"))
            print(f"RESULT kfd_node {node.name} location_id={loc:#x} bus={loc >> 8:#04x} "
                  f"unique_id={props.get('unique_id', '?').strip()} simd={props.get('simd_count')}")
        for name in gpu_nodes:
            links = base / name / "io_links"
            for link in sorted(links.iterdir(), key=lambda p: int(p.name)):
                p = dict(line.split(None, 1) for line in (link / "properties").read_text().splitlines()
                         if " " in line)
                to = p.get("node_to", "?").strip()
                if to not in gpu_nodes:
                    continue
                print(f"RESULT kfd_link {name}->{to} type={p.get('type','?').strip()} "
                      f"weight={p.get('weight','?').strip()} min_bw={p.get('min_bandwidth','?').strip()} "
                      f"max_bw={p.get('max_bandwidth','?').strip()} (type 11 = XGMI, MB/s)")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT kfd_topology unavailable: {exc}")
    import torch

    n = torch.cuda.device_count()
    print(f"RESULT visible HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} count={n}")
    for d in range(n):
        p = torch.cuda.get_device_properties(d)
        print(f"RESULT device {d} name={p.name} cus={p.multi_processor_count} "
              f"pci_bus={getattr(p, 'pci_bus_id', '?')} uuid={getattr(p, 'uuid', '?')}")
    if n >= 2:
        print(f"RESULT peer can_access 0->1={torch.cuda.can_device_access_peer(0, 1)} "
              f"1->0={torch.cuda.can_device_access_peer(1, 0)}")
        for nbytes in (8192, 1 << 20, 256 << 20):
            a = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
            b = torch.empty(nbytes, dtype=torch.uint8, device="cuda:1")
            for _ in range(3):
                b.copy_(a)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            reps = 20 if nbytes > (1 << 20) else 200
            t0 = time.perf_counter()
            for _ in range(reps):
                b.copy_(a)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            dt = (time.perf_counter() - t0) / reps
            print(f"RESULT p2p_copy 0->1 bytes={nbytes} us={dt * 1e6:.1f} GB/s={nbytes / dt / 1e9:.1f}")
    return 0


# =============================================================================== (b) all-reduce


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only machines
    triton = None

if triton is not None:

    @triton.jit
    def ar_ll_kernel(x_ptr, out_ptr, recv_local, recv_peer, epoch_ptr, err_ptr, NW,
                     CHUNK: tl.constexpr, BF16: tl.constexpr, MAXIT: tl.constexpr,
                     LOAD_CV: tl.constexpr, STORE_WT: tl.constexpr):
        p = tl.program_id(0)
        e = tl.load(epoch_ptr + p) + 1
        buf = (e % 2).to(tl.int64)
        offs = p * CHUNK + tl.arange(0, CHUNK)
        m = offs < NW
        xi = tl.load(x_ptr + offs, mask=m, other=0)
        word = (e.to(tl.int64) << 32) | (xi.to(tl.int64) & 0xFFFFFFFF)
        dst = recv_peer + buf * NW + offs
        if STORE_WT:
            tl.store(dst, word, mask=m, cache_modifier=".wt")
        else:
            tl.store(dst, word, mask=m)
        src = recv_local + buf * NW + offs
        if LOAD_CV:
            w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
        else:
            w = tl.load(src, mask=m, other=0, volatile=True)
        pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
        it = pending * 0
        while (pending > 0) & (it < MAXIT):
            if LOAD_CV:
                w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
            else:
                w = tl.load(src, mask=m, other=0, volatile=True)
            pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
            it += 1
        if pending > 0:
            tl.atomic_add(err_ptr, 1)
        yi = (w & 0xFFFFFFFF).to(tl.int32)
        if BF16:
            a_lo = ((xi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
            a_hi = (xi & -65536).to(tl.float32, bitcast=True)
            b_lo = ((yi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
            b_hi = (yi & -65536).to(tl.float32, bitcast=True)
            s_lo = (a_lo + b_lo).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            s_hi = (a_hi + b_hi).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
            res = s_lo | s_hi
        else:
            res = (xi.to(tl.float32, bitcast=True) + yi.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, res, mask=m)
        tl.store(epoch_ptr + p, e)

    @triton.jit
    def ar_flag_kernel(x_ptr, out_ptr, recv_local, recv_peer, flag_local, flag_peer, epoch_ptr,
                       err_ptr, NW, CHUNK: tl.constexpr, BF16: tl.constexpr, MAXIT: tl.constexpr,
                       LOAD_CV: tl.constexpr, STORE_WT: tl.constexpr):
        p = tl.program_id(0)
        e = tl.load(epoch_ptr + p) + 1
        buf = (e % 2).to(tl.int64)
        offs = p * CHUNK + tl.arange(0, CHUNK)
        m = offs < NW
        xi = tl.load(x_ptr + offs, mask=m, other=0)
        dst = recv_peer + buf * NW + offs
        if STORE_WT:
            tl.store(dst, xi, mask=m, cache_modifier=".wt")
        else:
            tl.store(dst, xi, mask=m)
        tl.debug_barrier()
        tl.atomic_xchg(flag_peer + p, e, sem="release", scope="sys")
        v = tl.atomic_add(flag_local + p, 0, sem="acquire", scope="sys")
        it = v * 0
        while (v < e) & (it < MAXIT):
            v = tl.atomic_add(flag_local + p, 0, sem="acquire", scope="sys")
            it += 1
        if v < e:
            tl.atomic_add(err_ptr, 1)
        src = recv_local + buf * NW + offs
        if LOAD_CV:
            yi = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
        else:
            yi = tl.load(src, mask=m, other=0, volatile=True)
        if BF16:
            a_lo = ((xi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
            a_hi = (xi & -65536).to(tl.float32, bitcast=True)
            b_lo = ((yi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
            b_hi = (yi & -65536).to(tl.float32, bitcast=True)
            s_lo = (a_lo + b_lo).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            s_hi = (a_hi + b_hi).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
            res = s_lo | s_hi
        else:
            res = (xi.to(tl.float32, bitcast=True) + yi.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, res, mask=m)
        tl.store(epoch_ptr + p, e)

    @triton.jit
    def fill_kernel(x_ptr, epoch_ptr, N, RANK: tl.constexpr, BLOCK: tl.constexpr):
        e = tl.load(epoch_ptr) + 1
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        v = ((e * 7 + offs * 3 + RANK * 5) % 61 - 30).to(tl.float32)
        tl.store(x_ptr + offs, v.to(x_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def check_kernel(out_ptr, epoch_ptr, err_ptr, N, BLOCK: tl.constexpr):
        e = tl.load(epoch_ptr)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        exp = ((e * 7 + offs * 3) % 61 - 30).to(tl.float32) + ((e * 7 + offs * 3 + 5) % 61 - 30).to(tl.float32)
        got = tl.load(out_ptr + offs, mask=m, other=0).to(tl.float32)
        bad = tl.sum(((got != exp) & m).to(tl.int32), axis=0)
        if bad > 0:
            tl.atomic_add(err_ptr + 1, bad)


def _ar_kernels():
    return NS(ll=ar_ll_kernel, flag=ar_flag_kernel, fill=fill_kernel, check=check_kernel)


class Raw:
    """A device pointer Triton accepts as a tensor argument (``data_ptr`` and ``dtype``)."""

    def __init__(self, ptr: int, dtype) -> None:
        self.ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.ptr


class AllReduce:
    """One-shot P2P all-reduce state for one (size, dtype, protocol, memory) configuration."""

    MAX_CTAS = 4096

    def __init__(self, hip, K, n: int, dtype, proto: str, mem: str, chunk: int, flags: tuple) -> None:
        import torch

        self.K, self.hip, self.proto, self.chunk = K, hip, proto, chunk
        self.load_cv, self.store_wt = flags
        self.n = n
        self.dtype = dtype
        self.nw = n * torch.empty((), dtype=dtype).element_size() // 4  # 32-bit words
        self.grid = (self.nw + chunk - 1) // chunk
        assert self.grid <= self.MAX_CTAS
        word = torch.int64 if proto == "ll" else torch.int32
        self.x, self.out, self.epoch, self.err, self.recv, self.flag = [], [], [], [], [], []
        self._keep = []
        for r in (0, 1):
            dev = torch.device("cuda", r)
            self.x.append(torch.zeros(n, dtype=dtype, device=dev))
            self.out.append(torch.zeros(n, dtype=dtype, device=dev))
            self.epoch.append(torch.zeros(self.MAX_CTAS, dtype=torch.int32, device=dev))
            self.err.append(torch.zeros(2, dtype=torch.int32, device=dev))
            self.recv.append(self._alloc(r, 2 * self.nw, word, mem))
            self.flag.append(self._alloc(r, self.MAX_CTAS, torch.int32, mem))
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

    def _alloc(self, rank: int, count: int, dtype, mem: str):
        import torch

        if mem == "torch":
            t = torch.zeros(count, dtype=dtype, device=torch.device("cuda", rank))
            self._keep.append(t)
            return t
        flag = {"fine": 1, "uncached": 3}[mem]
        size = count * torch.empty((), dtype=dtype).element_size()
        ptr = ctypes.c_void_p()
        hip = self.hip
        assert hip.hipSetDevice(rank) == 0
        rc = hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(flag))
        if rc != 0:
            raise RuntimeError(f"hipExtMallocWithFlags({mem}) rc={rc}")
        assert hip.hipMemset(ptr, 0, ctypes.c_size_t(size)) == 0
        assert hip.hipDeviceSynchronize() == 0
        torch.cuda.set_device(0)
        return Raw(ptr.value, dtype)

    def call(self, rank: int, fill: bool = False, check: bool = False) -> None:
        import torch

        K = self.K
        peer = 1 - rank
        with torch.cuda.device(rank):
            if fill:
                K.fill[(triton_cdiv(self.n, 1024),)](self.x[rank], self.epoch[rank], self.n,
                                                     RANK=rank, BLOCK=1024)
            xi = self.x[rank].view(torch.int32)
            oi = self.out[rank].view(torch.int32)
            common = dict(CHUNK=self.chunk, BF16=self.dtype == torch.bfloat16, MAXIT=100000,
                          LOAD_CV=self.load_cv, STORE_WT=self.store_wt, num_warps=1)
            if self.proto == "ll":
                K.ll[(self.grid,)](xi, oi, self.recv[rank], self.recv[peer], self.epoch[rank],
                                   self.err[rank], self.nw, **common)
            else:
                K.flag[(self.grid,)](xi, oi, self.recv[rank], self.recv[peer], self.flag[rank],
                                     self.flag[peer], self.epoch[rank], self.err[rank], self.nw,
                                     **common)
            if check:
                K.check[(triton_cdiv(self.n, 1024),)](self.out[rank], self.epoch[rank],
                                                      self.err[rank], self.n, BLOCK=1024)

    def both(self, **kw) -> None:
        self.call(0, **kw)
        self.call(1, **kw)

    def errors(self) -> tuple[int, int]:
        """(spin timeouts, mismatched elements) over both ranks."""
        import torch

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        e = [t.cpu().tolist() for t in self.err]
        return e[0][0] + e[1][0], e[0][1] + e[1][1]

    def graphs(self, reps: int, **kw):
        import torch

        out = []
        for r in (0, 1):
            with torch.cuda.device(r):
                s = torch.cuda.Stream()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.stream(s):
                    with torch.cuda.graph(g, stream=s):
                        for _ in range(reps):
                            self.call(r, **kw)
                torch.cuda.synchronize(r)
            out.append(g)
        return out


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def enable_peer(hip) -> None:
    import torch

    for a, b in ((0, 1), (1, 0)):
        assert hip.hipSetDevice(a) == 0
        rc = hip.hipDeviceEnablePeerAccess(b, 0)
        print(f"RESULT enable_peer {a}->{b} rc={rc} (0 ok, 704 already enabled)")
    torch.cuda.set_device(0)


def part_ar(mem: str, protos: list[str], sizes: list[int]) -> int:
    import torch

    hip = _hip()
    torch.cuda.init()
    torch.zeros(1, device="cuda:0")
    torch.zeros(1, device="cuda:1")
    enable_peer(hip)
    K = _ar_kernels()
    variants = [(0, 0), (1, 0), (1, 1)]  # (load .cv, store .wt)
    broken: set = set()
    for proto in protos:
        for flags in variants:
            for nbytes in sizes:
                if (proto, flags) in broken:
                    continue
                for dtype in (torch.bfloat16, torch.float32):
                    n = nbytes // (2 if dtype == torch.bfloat16 else 4)
                    nw = nbytes // 4
                    chunks = sorted({min(c, triton_next_pow2(nw)) for c in (64, 256, 1024)})
                    if nbytes > 65536:
                        chunks = [1024]
                    for chunk in chunks:
                        if (proto, flags) in broken:
                            break
                        tag = (f"mem={mem} proto={proto} cv={flags[0]} wt={flags[1]} bytes={nbytes} "
                               f"dtype={str(dtype).split('.')[-1]} chunk={chunk}")
                        try:
                            ar = AllReduce(hip, K, n, dtype, proto, mem, chunk, flags)
                            # Eager: correctness over many iterations (fill/AR/check per call).
                            iters = 300 if nbytes <= 65536 else 30
                            for _ in range(5):
                                ar.both(fill=True, check=True)
                            to, bad = ar.errors()
                            if to:
                                broken.add((proto, flags))
                                print(f"RESULT ar {tag} eager_timeouts={to} mismatches={bad} -> variant broken, skipped")
                                continue
                            t0 = time.perf_counter()
                            for _ in range(iters):
                                ar.both(fill=True, check=True)
                            to, bad = ar.errors()
                            eager_us = (time.perf_counter() - t0) / iters * 1e6
                            if to:
                                broken.add((proto, flags))
                                print(f"RESULT ar {tag} eager_timeouts={to} mismatches={bad} -> variant broken, skipped")
                                continue
                            # Graph with fill/check: correctness in the replay regime.
                            gc = ar.graphs(STEPS_PER_AR, fill=True, check=True)
                            for _ in range(20):
                                gc[0].replay()
                                gc[1].replay()
                            to2, bad2 = ar.errors()
                            # Graph of 80 bare all-reduces: time per all-reduce.
                            gt = ar.graphs(STEPS_PER_AR)
                            for _ in range(5):
                                gt[0].replay()
                                gt[1].replay()
                            torch.cuda.synchronize(0)
                            torch.cuda.synchronize(1)
                            reps = 30 if nbytes <= 65536 else 5
                            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            t0 = time.perf_counter()
                            with torch.cuda.device(0):
                                ev[0].record()
                            for _ in range(reps):
                                gt[0].replay()
                                gt[1].replay()
                            with torch.cuda.device(0):
                                ev[1].record()
                            torch.cuda.synchronize(0)
                            torch.cuda.synchronize(1)
                            wall = (time.perf_counter() - t0) / (reps * STEPS_PER_AR) * 1e6
                            dev0 = ev[0].elapsed_time(ev[1]) * 1e3 / (reps * STEPS_PER_AR)
                            to3, bad3 = ar.errors()
                            print(f"RESULT ar {tag} graph_us={wall:.2f} dev0_us={dev0:.2f} "
                                  f"eager_us={eager_us:.1f} checked={iters + 20 * STEPS_PER_AR} "
                                  f"timeouts={to + to2 + to3} mismatches={bad + bad2} "
                                  f"ok={int(to + to2 + to3 + bad + bad2 == 0)}")
                            del gc, gt
                        except Exception as exc:  # noqa: BLE001
                            broken.add((proto, flags))
                            print(f"RESULT ar {tag} error={type(exc).__name__}: {str(exc)[:300]} -> variant skipped")
                            traceback.print_exc()
    # Per-node cost of a graph of small local kernels (what any in-graph op pays anyway).
    try:
        x = torch.zeros(4096, dtype=torch.bfloat16, device="cuda:0")
        ep = torch.zeros(4, dtype=torch.int32, device="cuda:0")
        with torch.cuda.device(0):
            s = torch.cuda.Stream()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, stream=s):
                    for _ in range(STEPS_PER_AR):
                        K.fill[(4,)](x, ep, 4096, RANK=0, BLOCK=1024)
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize(0)
            t0 = time.perf_counter()
            for _ in range(30):
                g.replay()
            torch.cuda.synchronize(0)
        print(f"RESULT graph_node_us local_fill_8KB={(time.perf_counter() - t0) / (30 * STEPS_PER_AR) * 1e6:.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT graph_node_us error={exc}")
    return 0


def triton_next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _rccl_worker(rank: int, port: int, sizes: list[int]) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, device_id=dev)
    for nbytes in sizes:
        for dtype in (torch.bfloat16, torch.float32):
            n = nbytes // (2 if dtype == torch.bfloat16 else 4)
            x = torch.ones(n, dtype=dtype, device=dev)
            tag = f"bytes={nbytes} dtype={str(dtype).split('.')[-1]}"
            try:
                for _ in range(20):
                    dist.all_reduce(x)
                torch.cuda.synchronize()
                dist.barrier()
                iters = 200
                t0 = time.perf_counter()
                for _ in range(iters):
                    dist.all_reduce(x)
                torch.cuda.synchronize()
                eager = (time.perf_counter() - t0) / iters * 1e6
                graph = float("nan")
                try:
                    s = torch.cuda.Stream()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(s):
                        with torch.cuda.graph(g, stream=s):
                            for _ in range(STEPS_PER_AR):
                                dist.all_reduce(x)
                    torch.cuda.synchronize()
                    dist.barrier()
                    for _ in range(3):
                        g.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        g.replay()
                    torch.cuda.synchronize()
                    graph = (time.perf_counter() - t0) / (10 * STEPS_PER_AR) * 1e6
                except Exception as exc:  # noqa: BLE001
                    if rank == 0:
                        print(f"RESULT rccl {tag} graph_error={type(exc).__name__}: {str(exc)[:200]}")
                if rank == 0:
                    print(f"RESULT rccl {tag} eager_us={eager:.2f} graph_us={graph:.2f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                if rank == 0:
                    print(f"RESULT rccl {tag} error={type(exc).__name__}: {str(exc)[:200]}")
    dist.destroy_process_group()


def part_rccl(sizes: list[int]) -> int:
    import torch.multiprocessing as mp

    port = 29500 + os.getpid() % 1000
    mp.spawn(_rccl_worker, args=(port, sizes), nprocs=2, join=True)
    return 0


# =============================================================================== (c) layers


def _clone(t, dev):
    return t.detach().to(dev).clone().contiguous()


def _gdn_shim(m, dev, half: bool):
    nk, nv, dk, dv = m.num_k_heads, m.num_v_heads, m.head_k_dim, m.head_v_dim
    key_dim, value_dim = nk * dk, nv * dv
    qkv = m.in_proj_qkv.weight
    z, b, a = m.in_proj_z.weight, m.in_proj_b.weight, m.in_proj_a.weight
    cw = m.conv1d.weight
    if half:
        nk2, nv2 = nk // 2, nv // 2
        kd2, vd2 = nk2 * dk, nv2 * dv
        import torch

        ch = torch.cat([torch.arange(0, kd2), torch.arange(key_dim, key_dim + kd2),
                        torch.arange(2 * key_dim, 2 * key_dim + vd2)]).to(qkv.device)
        qkv, cw = qkv[ch], cw[ch]
        z, b, a = z[:vd2], b[:nv2], a[:nv2]
        out = m.out_proj.weight[:, :vd2]
        alog, dtb = m.A_log[:nv2], m.dt_bias[:nv2]
        nk, nv, key_dim, value_dim = nk2, nv2, kd2, vd2
    else:
        out, alog, dtb = m.out_proj.weight, m.A_log, m.dt_bias
    return NS(
        in_proj_qkv=NS(weight=_clone(qkv, dev)), in_proj_z=NS(weight=_clone(z, dev)),
        in_proj_b=NS(weight=_clone(b, dev)), in_proj_a=NS(weight=_clone(a, dev)),
        conv1d=NS(weight=_clone(cw, dev)), out_proj=NS(weight=_clone(out, dev)),
        A_log=_clone(alog, dev), dt_bias=_clone(dtb, dev),
        norm=NS(weight=_clone(m.norm.weight, dev), variance_epsilon=m.norm.variance_epsilon),
        num_k_heads=nk, num_v_heads=nv, head_k_dim=dk, head_v_dim=dv, key_dim=key_dim,
        value_dim=value_dim, conv_dim=2 * key_dim + value_dim, conv_kernel_size=cw.shape[-1],
    )


def _attn_shim(m, dev, half: bool, dim: int, q_heads: int, kv_heads: int):
    q, k, v, o = m.q_proj.weight, m.k_proj.weight, m.v_proj.weight, m.o_proj.weight
    if half:
        q = q[: (q_heads // 2) * 2 * dim]
        k = k[: (kv_heads // 2) * dim]
        v = v[: (kv_heads // 2) * dim]
        o = o[:, : (q_heads // 2) * dim]
    return NS(
        q_proj=NS(weight=_clone(q, dev)), k_proj=NS(weight=_clone(k, dev)),
        v_proj=NS(weight=_clone(v, dev)), o_proj=NS(weight=_clone(o, dev)),
        q_norm=NS(weight=_clone(m.q_norm.weight, dev), eps=m.q_norm.eps),
        k_norm=NS(weight=_clone(m.k_norm.weight, dev), eps=m.k_norm.eps), scaling=m.scaling,
    )


def _moe_shim(mlp, dev, half: bool):
    import torch

    ex, sh = mlp.experts, mlp.shared_expert
    gu, dn = ex.gate_up_proj, ex.down_proj
    inter = gu.shape[1] // 2
    sg, su, sd = sh.gate_proj.weight, sh.up_proj.weight, sh.down_proj.weight
    if half:
        i2 = inter // 2
        gu = torch.cat([gu[:, :i2], gu[:, inter:inter + i2]], dim=1)
        dn = dn[:, :, :i2]
        is2 = sg.shape[0] // 2
        sg, su, sd = sg[:is2], su[:is2], sd[:, :is2]
    return NS(
        gate=NS(weight=_clone(mlp.gate.weight, dev)),
        shared_expert=NS(gate_proj=NS(weight=_clone(sg, dev)), up_proj=NS(weight=_clone(su, dev)),
                         down_proj=NS(weight=_clone(sd, dev))),
        shared_expert_gate=NS(weight=_clone(mlp.shared_expert_gate.weight, dev)),
        experts=NS(gate_up_proj=_clone(gu, dev), down_proj=_clone(dn, dev)),
    )


def part_layers(rows_list: list[int], step_ms: float) -> int:
    import torch
    from transformers import AutoConfig

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    os.environ["ENGINE_FAST"] = "0"  # only the Transformers modules' weights are needed
    import mega_stage0 as ms
    from engine.fast import fused, kernels
    from engine.fast.model import PAGE
    from engine.generator import CHECKPOINT, Generator

    model, revision = CHECKPOINT["model"], CHECKPOINT["revision"]
    config = AutoConfig.from_pretrained(model, revision=revision)
    for cfg in {id(c): c for c in (config, config.get_text_config())}.values():
        if hasattr(cfg, "layer_types"):
            cfg.layer_types = list(cfg.layer_types)[:4]
            cfg.num_hidden_layers = 4
        if hasattr(cfg, "mtp_num_hidden_layers"):
            cfg.mtp_num_hidden_layers = 0
    t0 = time.perf_counter()
    gen = Generator.load(model, revision, config=config, max_model_len=8192, chunk_tokens=2048,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    lm = gen.model
    text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
    cfg = text.config
    types = list(cfg.layer_types)
    print(f"RESULT layers_load_s={time.perf_counter() - t0:.1f} types={types}")
    gi = types.index("linear_attention")
    ai = types.index("full_attention")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    torch.set_grad_enabled(False)
    am = text.layers[ai].self_attn
    dim = am.head_dim
    q_heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    gm = text.layers[gi].linear_attn
    print(f"RESULT geometry hidden={cfg.hidden_size} gdn k_heads={gm.num_k_heads} v_heads={gm.num_v_heads} "
          f"dk={gm.head_k_dim} dv={gm.head_v_dim} attn q_heads={q_heads} kv_heads={kv_heads} dim={dim} "
          f"experts={tuple(text.layers[gi].mlp.experts.gate_up_proj.shape)} "
          f"shared={tuple(text.layers[gi].mlp.shared_expert.gate_proj.weight.shape)}")

    def norm(t):
        return NS(weight=_clone(t.weight, dev))

    shim_layers, shim_types = [], []
    for kind, src in (("linear_attention", gi), ("full_attention", ai)):
        layer = text.layers[src]
        for half in (False, True):
            mixer = (_gdn_shim(layer.linear_attn, dev, half) if kind == "linear_attention"
                     else _attn_shim(layer.self_attn, dev, half, dim, q_heads, kv_heads))
            ns = NS(input_layernorm=norm(layer.input_layernorm),
                    post_attention_layernorm=norm(layer.post_attention_layernorm),
                    mlp=_moe_shim(layer.mlp, dev, half))
            setattr(ns, "linear_attn" if kind == "linear_attention" else "self_attn", mixer)
            shim_layers.append(ns)
            shim_types.append(kind)
    names = ["gdn_full", "gdn_half", "attn_full", "attn_half"]
    rmax = max(rows_list)
    slots = rmax + 1
    pos = 2048
    pages_per_slot = pos // PAGE + 1
    pages = slots * pages_per_slot + 1
    max_len = 8192
    gdn_state, kv = {}, {}
    for i, (layer, kind) in enumerate(zip(shim_layers, shim_types)):
        if kind == "linear_attention":
            m = layer.linear_attn
            gdn_state[i] = (
                (torch.randn((slots, m.conv_dim, m.conv_kernel_size), device=dev) * 0.1).to(torch.bfloat16),
                torch.randn((slots, m.num_v_heads, m.head_k_dim, m.head_v_dim), device=dev) * 0.01,
            )
        else:
            kvh = layer.self_attn.k_proj.weight.shape[0] // dim
            shape = (pages, kvh, PAGE, dim)
            kv[i] = ((torch.randn(shape, device=dev) * 0.5).to(torch.bfloat16),
                     (torch.randn(shape, device=dev) * 0.5).to(torch.bfloat16))
    table = torch.zeros((slots, max_len // PAGE), dtype=torch.int32, device=dev)
    for s in range(slots):
        table[s, :pages_per_slot] = torch.arange(s * pages_per_slot, (s + 1) * pages_per_slot)
    meta = NS(page_table=table)
    shim = NS(layers=shim_layers, layer_types=shim_types, inv_freq={dev: text.rotary_emb.inv_freq.float().to(dev)},
              max_len=max_len, embed=text.embed_tokens.weight, gdn_state=gdn_state, kv=kv,
              top_k=cfg.num_experts_per_tok, eps=cfg.rms_norm_eps, mtp=None)
    fd = fused.FusedDecode(shim)
    hidden = cfg.hidden_size
    times: dict[tuple[str, str, int], float] = {}
    for R in rows_list:
        st = torch.arange(R, device=dev)
        pt = torch.full((R,), pos, device=dev)
        xs = [(torch.randn(R, hidden, device=dev) * 0.3).to(torch.bfloat16) for _ in range(10)]
        hs = [kernels.add_rms(x, None, shim_layers[0].input_layernorm.weight, cfg.rms_norm_eps)[1] for x in xs]
        for i, name in enumerate(names):
            gdn = shim_types[i] == "linear_attention"

            def layer_fn(i=i):
                for x in xs:
                    fd.layer(i, x, st, pt, meta)

            def mixer_fn(i=i, gdn=gdn):
                for h in hs:
                    if gdn:
                        fd._gdn(i, h, st)
                    else:
                        fd._attn(i, h, st, pt, meta)

            def moe_fn(i=i):
                for x, h in zip(xs, hs):
                    fd._moe(i, x, h)

            for part, fn in (("layer", layer_fn), ("mixer", mixer_fn), ("moe", moe_fn)):
                try:
                    with torch.inference_mode():
                        t = ms.graph_ms(fn, reps=1, n=30) / len(xs)
                    times[(name, part, R)] = t
                    print(f"RESULT layer_ms {name} part={part} R={R} ms={t:.4f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"RESULT layer_ms {name} part={part} R={R} error={type(exc).__name__}: {str(exc)[:300]}")
                    traceback.print_exc()
        for kind in ("gdn", "attn"):
            for part in ("layer", "mixer", "moe"):
                f, h = times.get((f"{kind}_full", part, R)), times.get((f"{kind}_half", part, R))
                if f and h:
                    print(f"RESULT half_ratio {kind} part={part} R={R} full_ms={f:.4f} half_ms={h:.4f} "
                          f"ratio={h / f:.3f}")
    # lm_head GEMV: full vocab vs half (TP2 splits the vocabulary).
    lm_head = lm.lm_head.weight
    w_full = _clone(lm_head, dev)
    w_half = w_full[: w_full.shape[0] // 2].contiguous()
    fd.cus[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    for R in rows_list:
        x = (torch.randn(R, hidden, device=dev) * 0.3).to(torch.bfloat16)
        for name, w in (("full", w_full), ("half", w_half)):
            for impl in ("fused_gemv", "torch_linear"):
                try:
                    if impl == "fused_gemv":
                        fn = lambda w=w: fd._gemv(x, w)  # noqa: E731
                    else:
                        fn = lambda w=w: torch.nn.functional.linear(x, w)  # noqa: E731
                    with torch.inference_mode():
                        t = ms.graph_ms(fn, reps=4, n=20)
                    times[("lm_" + name, impl, R)] = t
                    print(f"RESULT lm_head_ms {name} impl={impl} R={R} rows={w.shape[0]} ms={t:.4f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"RESULT lm_head_ms {name} impl={impl} R={R} error={exc}")
    # Layer sums for the step estimate at B=2.
    R = 2
    try:
        full = 30 * times[("gdn_full", "layer", R)] + 10 * times[("attn_full", "layer", R)]
        half = 30 * times[("gdn_half", "layer", R)] + 10 * times[("attn_half", "layer", R)]
        lm_cut = min(times[("lm_full", i, R)] for i in ("fused_gemv", "torch_linear")) - \
            min(times[("lm_half", i, R)] for i in ("fused_gemv", "torch_linear"))
        print(f"RESULT layer_sum R=2 full_40_layers_ms={full:.3f} half_40_layers_ms={half:.3f} "
              f"saved_ms={full - half:.3f} lm_head_saved_ms={lm_cut:.3f} "
              f"step_ms_ref={step_ms} step_minus_savings_ms={step_ms - (full - half) - lm_cut:.3f}")
    except KeyError as exc:
        print(f"RESULT layer_sum missing {exc}")
    return 0


# =============================================================================== driver


def run_part(argv: list[str], timeout: int) -> list[str]:
    env = dict(os.environ)
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    cmd = [sys.executable, str(Path(__file__).resolve())] + argv
    print(f"===== part: {' '.join(argv)} (timeout {timeout}s)", flush=True)
    lines: list[str] = []
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(ROOT))
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            if time.perf_counter() - t0 > timeout:
                raise subprocess.TimeoutExpired(cmd, timeout)
        proc.wait(timeout=max(1, timeout - (time.perf_counter() - t0)))
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"RESULT part_timeout {' '.join(argv)}", flush=True)
    print(f"===== part done rc={proc.returncode} s={time.perf_counter() - t0:.0f}", flush=True)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="all")
    ap.add_argument("--mem", default="torch")
    ap.add_argument("--protos", default="ll,flag")
    ap.add_argument("--sizes", default="2048,4096,8192,16384,65536")
    ap.add_argument("--rows", default="1,2,4")
    ap.add_argument("--step-ms", type=float, default=11.55)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    if args.part == "topo":
        return part_topo()
    if args.part == "ar":
        return part_ar(args.mem, args.protos.split(","), sizes)
    if args.part == "rccl":
        return part_rccl(sizes)
    if args.part == "layers":
        return part_layers([int(r) for r in args.rows.split(",")], args.step_ms)
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    out: list[str] = []
    out += run_part(["--part", "topo"], 300)
    for mem in ("torch", "fine", "uncached"):
        out += run_part(["--part", "ar", "--mem", mem, "--sizes", args.sizes + ",2867200"], 600)
    out += run_part(["--part", "rccl", "--sizes", args.sizes], 400)
    out += run_part(["--part", "layers", "--rows", args.rows, "--step-ms", str(args.step_ms)], 1500)
    # ------------------------------------------------------------------ verdicts
    best8 = None
    for line in out:
        m = re.search(r"RESULT ar (.*bytes=8192 dtype=bfloat16.*) graph_us=([\d.]+).*ok=1", line)
        if m and (best8 is None or float(m.group(2)) < best8[0]):
            best8 = (float(m.group(2)), m.group(1))
    rccl8 = None
    for line in out:
        m = re.search(r"RESULT rccl bytes=8192 dtype=bfloat16 eager_us=([\d.]+) graph_us=([\d.na]+)", line)
        if m:
            rccl8 = (m.group(1), m.group(2))
    ratio2 = {}
    for line in out:
        m = re.search(r"RESULT half_ratio (\w+) part=layer R=2 .* ratio=([\d.]+)", line)
        if m:
            ratio2[m.group(1)] = float(m.group(2))
    sub = None
    for line in out:
        m = re.search(r"step_minus_savings_ms=([\d.-]+)", line)
        if m:
            sub = float(m.group(1))
    print(f"RESULT verdict allreduce_8KB_bf16_best_graph_us={best8[0] if best8 else None} "
          f"config=[{best8[1] if best8 else ''}] rccl_8KB_bf16(eager,graph)={rccl8} "
          f"kill_allreduce={'none-correct' if best8 is None else best8[0] > 12.0}")
    print(f"RESULT verdict half_over_full_layer_R2={ratio2} "
          f"kill_layer={any(r > 0.65 for r in ratio2.values()) if ratio2 else 'missing'}")
    if best8 is not None and sub is not None:
        print(f"RESULT estimate tp2_step_R2_ms={sub + STEPS_PER_AR * best8[0] / 1000:.2f} "
              f"(= {sub:.3f} step after half-width savings + {STEPS_PER_AR} x {best8[0]:.2f} us) "
              f"vs measured {args.step_ms} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
