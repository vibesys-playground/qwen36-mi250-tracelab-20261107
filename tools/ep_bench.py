"""Plan 16 step 0: could one node-wide engine (TP8 dense + EP8 experts) beat four 2-GCD replicas?

    python tools/ep_bench.py [--part all|topo|ar|rccl|routes|parts] [--tp 8|4] [--ref-ms 28.7]

``all`` (default) runs every part in its own subprocess (a fault or hang in one part cannot
take the others down), then prints the projected step, the kill-rule verdicts and the
comparison with the replica layout. Every result is a ``RESULT`` line. No engine change: the
dense layers run through ``engine.fast.fused.FusedDecode``'s kernels on 1/8-width shims, the
local-expert MoE and the all-reduce are new kernels in this file.

``topo``    KFD io_links of all 8 GCDs and the HIP-device -> KFD-node map (by PCI bus).
``ar``      In-graph one-shot LL all-reduce on fine-grained buffers (plan 13's protocol:
            ``hipExtMallocWithFlags(..., 1)``, ``.cv`` loads, ``.wt`` stores, epoch
            double-buffering, chunk 64 / num_warps 1), generalized to a group of G ranks:
            every rank writes its LL words into slot ``me`` of every member's receive buffer
            and sums the G slots in rank order, so every member gets the same bits.
              flat8  - one group of all 8 GCDs
              hier   - pair (2 GCDs of a card), then the 4 same-side GCDs across cards
              tp4    - two groups of 4 GCDs (2 cards each) at once (the 2 x 4 fallback)
            at 4, 16 and 64 KB (1, 4 and 16 rows of hidden 2048 in BF16). Correctness: a fill
            kernel writes epoch-dependent exact integers on every rank before each all-reduce
            and a check kernel compares the sum after it, eagerly and inside HIP graphs; spin
            timeouts are counted. Time: per GCD a HIP graph of 100 all-reduces, replayed from one
            host thread per GCD; per replay the slowest GCD's time / 100; p50 / p99 over replays.
``rccl``    torch.distributed (RCCL) all_reduce over 8 processes, eager and in a graph, same sizes.
``routes``  Real router outputs: the checkpoint on GCDs 0,1 (``Generator.load``), prefill of 8
            replay-like prompts (Request Factory's synthetic coding lines, as the scored
            workload) and 8 chunks of real source code, recording each token's top-8 experts in
            every layer. A 16-row verify batch = 8 sessions x 2 consecutive tokens. Expert
            counts per GCD for EP8 (experts 32g..32g+31 on GCD g) and the busiest GCD.
``parts``   All 8 GCDs at once, 16 rows (TP8) or 8 rows per group (TP4), random BF16 weights of
            the real shapes for all 40 layers + the MTP layer (distinct per layer, so nothing
            hits in L2), routing from ``routes``:
              gdn   - add_rms + 1/8 Gated DeltaNet verify layer (2 of 16 k heads, 4 of 32 v heads)
              attn  - add_rms + 1/8 attention (2 of 16 q heads, 1 of the 2 KV heads)
              moe   - add_rms + router/shared-slice GEMV + top-k + local-expert MoE (32 experts
                      per GCD, whole) + shared-expert slice (64 of 512) combine
              lm    - lm_head vocabulary slice (248320 / 8 rows) at 16 rows
              step  - the composed step: 41 layers x (mixer + all-reduce + MoE + all-reduce),
                      final norm, lm_head slice twice (verify + MTP draft) + 2 small
                      all-reduces, one graph per GCD; with flat8 / hier / no all-reduce.
            The composed step must leave the replicated hidden state bit-identical on all GCDs.

Kill rules (plan 16): park if the projected 16-row step > 12 ms, the all-reduce p50 at 64 KB
> 20 us, or any mismatch or timeout. If only the all-reduce fails, the 2 x 4 GCD layout
(``--tp 4``) is the fallback.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
ROUTES = Path(os.environ.get("EP_BENCH_ROUTES", "/tmp/ep_bench_routes.pt"))
AR_REPS = 100  # all-reduces per timing graph
MAXIT = 1_000_000  # spin iterations before an all-reduce counts a timeout


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":")), flush=True)


def pct(xs, q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def _hip():
    import torch

    path = Path(torch.__file__).parent / "lib" / "libamdhip64.so"
    return ctypes.CDLL(str(path) if path.exists() else "libamdhip64.so")


# =============================================================================== topology


def kfd_topology() -> tuple[dict, list]:
    """({kfd node: bus}, [(from, to, max_bw MB/s, weight)]) for GPU nodes."""
    base = Path("/sys/class/kfd/kfd/topology/nodes")
    nodes, links = {}, []
    for node in sorted(base.iterdir(), key=lambda p: int(p.name)):
        props = dict(line.split(None, 1) for line in (node / "properties").read_text().splitlines()
                     if " " in line)
        if int(props.get("simd_count", "0")) == 0:
            continue
        nodes[int(node.name)] = int(props.get("location_id", "0")) >> 8
    for n in nodes:
        for link in sorted((base / str(n) / "io_links").iterdir(), key=lambda p: int(p.name)):
            p = dict(line.split(None, 1) for line in (link / "properties").read_text().splitlines()
                     if " " in line)
            to = int(p.get("node_to", "-1").strip())
            if to in nodes:
                links.append((n, to, int(p.get("max_bandwidth", "0").strip()),
                              int(p.get("weight", "0").strip())))
    return nodes, links


def part_topo() -> int:
    import torch

    n = torch.cuda.device_count()
    buses = {d: getattr(torch.cuda.get_device_properties(d), "pci_bus_id", None) for d in range(n)}
    out("devices", count=n, hip_visible=os.environ.get("HIP_VISIBLE_DEVICES"), pci_bus=buses)
    for d in range(n):
        free, total = torch.cuda.mem_get_info(d)
        out("mem", device=d, free_gb=round(free / 2**30, 1), total_gb=round(total / 2**30, 1))
    try:
        nodes, links = kfd_topology()
        node_of = {}
        for d, bus in buses.items():
            for k, b in nodes.items():
                if b == bus:
                    node_of[d] = k
        dev_of = {k: d for d, k in node_of.items()}
        out("kfd_map", hip_to_kfd=node_of)
        mat = [["." for _ in range(n)] for _ in range(n)]
        for a, b, bw, w in links:
            if a in dev_of and b in dev_of:
                mat[dev_of[a]][dev_of[b]] = (f"{bw // 1000}G" if bw else f"hop(w{w})")
        for d in range(n):
            out("link_row", device=d, to=mat[d])
    except Exception as exc:  # noqa: BLE001
        out("kfd_topology_error", error=str(exc)[:300])
    return 0


# =============================================================================== all-reduce

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only machines
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def ar_ll_kernel(x_ptr, out_ptr, recv_local, p0, p1, p2, p3, p4, p5, p6, p7,
                     epoch_ptr, err_ptr, NW, ME, G: tl.constexpr, CHUNK: tl.constexpr,
                     BF16: tl.constexpr, MAXIT_C: tl.constexpr):
        """One-shot LL all-reduce over G ranks. p0..p{G-1} are the members' receive buffers
        ([2 parities][G slots][NW] int64); this rank writes slot ME of each, then sums the G
        slots of its own buffer in slot order (identical bits on every member)."""
        p = tl.program_id(0)
        e = tl.load(epoch_ptr + p) + 1
        buf = (e % 2).to(tl.int64)
        offs = p * CHUNK + tl.arange(0, CHUNK)
        m = offs < NW
        xi = tl.load(x_ptr + offs, mask=m, other=0)
        word = (e.to(tl.int64) << 32) | (xi.to(tl.int64) & 0xFFFFFFFF)
        dst = (buf * G + ME) * NW + offs
        tl.store(p0 + dst, word, mask=m, cache_modifier=".wt")
        if G > 1:
            tl.store(p1 + dst, word, mask=m, cache_modifier=".wt")
        if G > 2:
            tl.store(p2 + dst, word, mask=m, cache_modifier=".wt")
        if G > 3:
            tl.store(p3 + dst, word, mask=m, cache_modifier=".wt")
        if G > 4:
            tl.store(p4 + dst, word, mask=m, cache_modifier=".wt")
        if G > 5:
            tl.store(p5 + dst, word, mask=m, cache_modifier=".wt")
        if G > 6:
            tl.store(p6 + dst, word, mask=m, cache_modifier=".wt")
        if G > 7:
            tl.store(p7 + dst, word, mask=m, cache_modifier=".wt")
        acc_lo = tl.zeros([CHUNK], dtype=tl.float32)
        acc_hi = tl.zeros([CHUNK], dtype=tl.float32)
        for j in tl.static_range(G):
            src = recv_local + (buf * G + j) * NW + offs
            w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
            pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
            it = pending * 0
            while (pending > 0) & (it < MAXIT_C):
                w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
                pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
                it += 1
            if pending > 0:
                tl.atomic_add(err_ptr, 1)
            yi = (w & 0xFFFFFFFF).to(tl.int32)
            if BF16:
                acc_lo += ((yi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
                acc_hi += (yi & -65536).to(tl.float32, bitcast=True)
            else:
                acc_lo += yi.to(tl.float32, bitcast=True)
        if BF16:
            s_lo = acc_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            s_hi = acc_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
            res = s_lo | s_hi
        else:
            res = acc_lo.to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, res, mask=m)
        tl.store(epoch_ptr + p, e)

    @triton.jit
    def fill_kernel(x_ptr, tick_ptr, N, RANK: tl.constexpr, BLOCK: tl.constexpr):
        e = tl.load(tick_ptr) + 1
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        v = ((e * 7 + offs * 3 + RANK * 5) % 61 - 30).to(tl.float32)
        tl.store(x_ptr + offs, v.to(x_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def check_kernel(out_ptr, tick_ptr, err_ptr, N, MASK: tl.constexpr, BLOCK: tl.constexpr):
        """Compare with the sum of the fill values of the ranks in bitmask MASK; bumps the tick."""
        e = tl.load(tick_ptr) + 1
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        exp = tl.zeros([BLOCK], dtype=tl.float32)
        for r in tl.static_range(8):
            if (MASK >> r) & 1:
                exp += ((e * 7 + offs * 3 + r * 5) % 61 - 30).to(tl.float32)
        got = tl.load(out_ptr + offs, mask=m, other=0).to(tl.float32)
        bad = tl.sum(((got != exp) & m).to(tl.int32), axis=0)
        if bad > 0:
            tl.atomic_add(err_ptr + 1, bad)

    @triton.jit
    def tick_kernel(tick_ptr):
        tl.store(tick_ptr, tl.load(tick_ptr) + 1)


class Raw:
    """A device pointer Triton accepts as a tensor argument (``data_ptr`` and ``dtype``)."""

    def __init__(self, ptr: int, dtype) -> None:
        self.ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.ptr


def fine_alloc(hip, dev: int, nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    assert hip.hipSetDevice(dev) == 0
    rc = hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1))
    if rc != 0:
        raise RuntimeError(f"hipExtMallocWithFlags(fine) rc={rc}")
    assert hip.hipMemset(ptr, 0, ctypes.c_size_t(nbytes)) == 0
    assert hip.hipDeviceSynchronize() == 0
    return ptr.value


def enable_all_peers(hip, n: int) -> None:
    bad = []
    for a in range(n):
        assert hip.hipSetDevice(a) == 0
        for b in range(n):
            if a != b:
                rc = hip.hipDeviceEnablePeerAccess(b, 0)
                if rc not in (0, 704):
                    bad.append((a, b, rc))
    out("enable_peer", pairs=n * (n - 1), failures=bad)


class LLGroup:
    """Receive buffers, epochs and error counters of one all-reduce group (member devices in
    rank order). ``max_nw``: the largest message in 32-bit words."""

    MAX_CTAS = 4096

    def __init__(self, hip, members: list[int], max_nw: int, chunk: int = 64) -> None:
        import torch

        self.members = list(members)
        self.G = len(members)
        self.chunk = chunk
        self.max_nw = max_nw
        self.recv, self.epoch, self.err = {}, {}, {}
        for d in members:
            self.recv[d] = fine_alloc(hip, d, 2 * self.G * max_nw * 8)
            self.epoch[d] = torch.zeros(self.MAX_CTAS, dtype=torch.int32, device=torch.device("cuda", d))
            self.err[d] = torch.zeros(2, dtype=torch.int32, device=torch.device("cuda", d))
        self.ptrs = [Raw(self.recv[d], torch.int64) for d in members]
        while len(self.ptrs) < 8:
            self.ptrs.append(self.ptrs[0])

    def __call__(self, dev: int, x, out=None):
        """All-reduce ``x`` (bf16 or fp32, on ``dev``) over the group into ``out``."""
        import torch

        if out is None:
            out = torch.empty_like(x)
        nw = x.numel() * x.element_size() // 4
        assert nw <= self.max_nw, (nw, self.max_nw)
        grid = (nw + self.chunk - 1) // self.chunk
        assert grid <= self.MAX_CTAS
        me = self.members.index(dev)
        with torch.cuda.device(dev):
            ar_ll_kernel[(grid,)](
                x.view(torch.int32), out.view(torch.int32), Raw(self.recv[dev], torch.int64),
                *self.ptrs, self.epoch[dev], self.err[dev], nw, me,
                G=self.G, CHUNK=self.chunk, BF16=x.dtype == torch.bfloat16, MAXIT_C=MAXIT,
                num_warps=max(1, self.chunk // 64),
            )
        return out

    def errors(self) -> tuple[int, int]:
        import torch

        to = bad = 0
        for d in self.members:
            torch.cuda.synchronize(d)
            e = self.err[d].cpu().tolist()
            to += e[0]
            bad += e[1]
        return to, bad


class AllReduce:
    """One all-reduce layout over all 8 GCDs: ``flat8``, ``hier`` (pair, then cross-card) or
    ``tp4`` (two independent groups of 4). ``run(dev, x)`` is the all-reduce of ``dev``'s
    group; ``mask(dev)`` the bitmask of global ranks its result sums."""

    def __init__(self, hip, kind: str, max_nw: int, chunk: int = 64) -> None:
        self.kind = kind
        self.stages: dict[int, list[LLGroup]] = {d: [] for d in range(8)}
        if kind == "flat8":
            g = LLGroup(hip, list(range(8)), max_nw, chunk)
            for d in range(8):
                self.stages[d] = [g]
        elif kind == "hier":
            pairs = [LLGroup(hip, [2 * c, 2 * c + 1], max_nw, chunk) for c in range(4)]
            sides = [LLGroup(hip, [0, 2, 4, 6], max_nw, chunk), LLGroup(hip, [1, 3, 5, 7], max_nw, chunk)]
            for d in range(8):
                self.stages[d] = [pairs[d // 2], sides[d % 2]]
        elif kind == "tp4":
            groups = [LLGroup(hip, [0, 1, 2, 3], max_nw, chunk), LLGroup(hip, [4, 5, 6, 7], max_nw, chunk)]
            for d in range(8):
                self.stages[d] = [groups[d // 4]]
        else:
            raise ValueError(kind)
        self.groups = list({id(g): g for st in self.stages.values() for g in st}.values())

    def mask(self, dev: int) -> int:
        if self.kind == "tp4":
            return 0x0F if dev < 4 else 0xF0
        return 0xFF

    def run(self, dev: int, x, out=None):
        y = x
        for i, g in enumerate(self.stages[dev]):
            y = g(dev, y, out if i == len(self.stages[dev]) - 1 else None)
        return y

    def errors(self) -> tuple[int, int]:
        to = bad = 0
        for g in self.groups:
            a, b = g.errors()
            to += a
            bad += b
        return to, bad


def concurrent_replay(graphs: dict, rounds: int, pre=None) -> dict:
    """Replay ``graphs[dev]`` ``rounds`` times on every device at once, one host thread per
    device (graph replay releases the GIL). ``pre(dev, k)`` runs on the device's stream before
    replay k. Returns {dev: (device ms per replay list, host launch ms list)}."""
    import torch

    devs = sorted(graphs)
    barrier = threading.Barrier(len(devs))
    res, errs = {}, []

    def worker(d):
        try:
            torch.cuda.set_device(d)
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                   for _ in range(rounds)]
            host = []
            barrier.wait()
            for k in range(rounds):
                if pre is not None:
                    pre(d, k)
                evs[k][0].record()
                t0 = time.perf_counter()
                graphs[d].replay()
                host.append((time.perf_counter() - t0) * 1e3)
                evs[k][1].record()
            torch.cuda.synchronize(d)
            res[d] = ([a.elapsed_time(b) for a, b in evs], host)
        except Exception as exc:  # noqa: BLE001
            errs.append((d, repr(exc)))
            try:
                barrier.abort()
            except Exception:  # noqa: BLE001
                pass

    threads = [threading.Thread(target=worker, args=(d,)) for d in devs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise RuntimeError(f"replay failed: {errs}")
    return res


def capture_all(fn_of_dev, devs=range(8)) -> dict:
    """{dev: graph of fn_of_dev(dev)}. Every device runs fn once first (compiles every kernel
    outside the capture), all launched before any host wait: a cross-GCD all-reduce inside fn
    would otherwise spin on peers whose work the host has not launched yet."""
    import torch

    devs = list(devs)
    streams = {}
    for d in devs:
        with torch.cuda.device(d):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            streams[d] = s
            with torch.cuda.stream(s):
                fn_of_dev(d)
    for d in devs:
        torch.cuda.synchronize(d)
    graphs = {}
    for d in devs:
        with torch.cuda.device(d):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(streams[d]):
                with torch.cuda.graph(g, stream=streams[d]):
                    fn_of_dev(d)
            graphs[d] = g
    for d in devs:
        torch.cuda.synchronize(d)
    return graphs


def part_ar(sizes: list[int], kinds: list[str], chunks: list[int]) -> int:
    import torch

    hip = _hip()
    torch.cuda.init()
    n = torch.cuda.device_count()
    assert n == 8, f"needs 8 GCDs, sees {n}"
    for d in range(n):
        torch.zeros(1, device=f"cuda:{d}")
    enable_all_peers(hip, n)
    for kind in kinds:
        for chunk in chunks:
            for nbytes in sizes:
                tag = dict(kind=kind, bytes=nbytes, chunk=chunk, dtype="bf16")
                try:
                    elems = nbytes // 2
                    ar = AllReduce(hip, kind, nbytes // 4, chunk)
                    xs = {d: torch.zeros(elems, dtype=torch.bfloat16, device=f"cuda:{d}") for d in range(8)}
                    ys = {d: torch.zeros(elems, dtype=torch.bfloat16, device=f"cuda:{d}") for d in range(8)}
                    ticks = {d: torch.zeros(1, dtype=torch.int32, device=f"cuda:{d}") for d in range(8)}
                    errc = {d: torch.zeros(2, dtype=torch.int32, device=f"cuda:{d}") for d in range(8)}

                    def checked(d):
                        with torch.cuda.device(d):
                            fill_kernel[(triton.cdiv(elems, 1024),)](xs[d], ticks[d], elems, RANK=d, BLOCK=1024)
                            ar.run(d, xs[d], ys[d])
                            check_kernel[(triton.cdiv(elems, 1024),)](ys[d], ticks[d], errc[d], elems,
                                                                       MASK=ar.mask(d), BLOCK=1024)
                            tick_kernel[(1,)](ticks[d])

                    # Eager: every rank's launch from one thread, many iterations.
                    for _ in range(50):
                        for d in range(8):
                            checked(d)
                    to0, bad0 = ar.errors()
                    bad0 += sum(int(errc[d][1].item()) for d in range(8))
                    if to0:
                        out("ar", **tag, eager_timeouts=to0, mismatches=bad0, ok=0, note="variant broken")
                        continue
                    # In a graph, fill/AR/check x AR_REPS, replayed concurrently.
                    gc = capture_all(lambda d: [checked(d) for _ in range(AR_REPS)])
                    concurrent_replay(gc, 20)
                    to1, bad1 = ar.errors()
                    bad1 += sum(int(errc[d][1].item()) for d in range(8)) - bad0
                    # Timing: AR_REPS bare all-reduces per graph.
                    gt = capture_all(lambda d: [ar.run(d, xs[d], ys[d]) for _ in range(AR_REPS)])
                    concurrent_replay(gt, 5)
                    res = concurrent_replay(gt, 200)
                    per = [max(res[d][0][k] for d in range(8)) * 1e3 / AR_REPS for k in range(200)]
                    to2, _ = ar.errors()
                    # The eager pass checked 50 x 8, the graph pass 20 x AR_REPS per rank.
                    checked_n = 50 + 20 * AR_REPS
                    total_to = to0 + to1 + to2
                    total_bad = bad0 + bad1
                    out("ar", **tag, p50_us=round(pct(per, 50), 2), p99_us=round(pct(per, 99), 2),
                        mean_us=round(statistics.mean(per), 2), checked_per_rank=checked_n,
                        timeouts=total_to, mismatches=total_bad, ok=int(total_to + total_bad == 0),
                        host_launch_ms_p50=round(pct(res[0][1], 50), 3))
                    del gc, gt
                except Exception as exc:  # noqa: BLE001
                    out("ar", **tag, error=f"{type(exc).__name__}: {str(exc)[:300]}", ok=0)
                    traceback.print_exc()
    return 0


def _rccl_worker(rank: int, port: int, sizes: list[int], world: int) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, device_id=dev)
    for nbytes in sizes:
        n = nbytes // 2
        tag = dict(world=world, bytes=nbytes, dtype="bf16")
        try:
            x = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device=dev)
            dist.all_reduce(x)
            torch.cuda.synchronize()
            want = world * (world + 1) / 2
            bad = int((x.float() != want).sum().item())
            for _ in range(20):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(200):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            eager = (time.perf_counter() - t0) / 200 * 1e6
            s = torch.cuda.Stream()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, stream=s):
                    for _ in range(AR_REPS):
                        dist.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(50)]
            for a, b in evs:
                a.record()
                g.replay()
                b.record()
            torch.cuda.synchronize()
            per = torch.tensor([a.elapsed_time(b) * 1e3 / AR_REPS for a, b in evs], device=dev)
            dist.all_reduce(per, op=dist.ReduceOp.MAX)
            per = per.tolist()
            if rank == 0:
                out("rccl", **tag, eager_us=round(eager, 2), graph_p50_us=round(pct(per, 50), 2),
                    graph_p99_us=round(pct(per, 99), 2), mismatches_first_call=bad)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                out("rccl", **tag, error=f"{type(exc).__name__}: {str(exc)[:300]}")
    dist.destroy_process_group()


def part_rccl(sizes: list[int]) -> int:
    import torch.multiprocessing as mp

    port = 29500 + os.getpid() % 1000
    mp.spawn(_rccl_worker, args=(port, sizes, 8), nprocs=8, join=True)
    return 0


# =============================================================================== routes


def part_routes(sessions: int, length: int, out_path: Path) -> int:
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("ENGINE_MTP", "0")
    from engine.generator import CHECKPOINT, Generator

    t0 = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=8192,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast, tok = g.fast, g.tokenizer
    out("routes_load", seconds=round(time.perf_counter() - t0), gpus=torch.cuda.device_count(),
        layers=len(fast.layers), top_k=fast.top_k)
    index = {id(layer.mlp): i for i, layer in enumerate(fast.layers)}
    record = {"on": False, "idx": {}, "wt": {}}
    orig = fast._moe

    def moe_rec(mlp, x, decode=False):
        i = index.get(id(mlp))
        if record["on"] and i is not None:
            probs = torch.softmax(F.linear(x, mlp.gate.weight), dim=-1, dtype=torch.float)
            w, e = torch.topk(probs, fast.top_k, dim=-1)
            record["idx"][i] = e.to("cpu", torch.int16)
            record["wt"][i] = (w / w.sum(-1, keepdim=True)).to("cpu", torch.float32)
        return orig(mlp, x, decode)

    fast._moe = moe_rec
    # Replay-like: Request Factory's synthetic coding lines (the scored workload's token pool).
    pool: list[int] = []
    k = 0
    stride = length + 600
    while len(pool) < sessions * stride + length:
        for line in (f"def case_{k}(items):", f"    return sorted(set(items))  # {k} 0f3a9c"):
            pool.extend(tok.encode(line, add_special_tokens=False))
        k += 1
    replay = [pool[s * stride: s * stride + length] for s in range(sessions)]
    # Real source code: this repository's Python files, one chunk per session.
    code_ids: list[int] = []
    for path in sorted((ROOT / "engine").rglob("*.py")):
        code_ids.extend(tok.encode(path.read_text(), add_special_tokens=False))
        if len(code_ids) > sessions * length:
            break
    code = [code_ids[s * length: (s + 1) * length] for s in range(sessions)
            if len(code_ids) >= (s + 1) * length]
    data = {}
    for name, prompts in (("replay", replay), ("code", code)):
        idx_s, wt_s = [], []
        for s, ids in enumerate(prompts):
            fast.reset(s)
            record["on"], record["idx"], record["wt"] = True, {}, {}
            with torch.inference_mode():
                fast.prefill(s, ids, 0, None)
            record["on"] = False
            idx_s.append(torch.stack([record["idx"][i] for i in range(len(fast.layers))], dim=1))
            wt_s.append(torch.stack([record["wt"][i] for i in range(len(fast.layers))], dim=1))
        if idx_s:
            data[name] = {"idx": torch.stack(idx_s), "wt": torch.stack(wt_s)}  # [S, T, L, K]
            out("routes_set", set=name, shape=list(data[name]["idx"].shape))
    torch.save(data, out_path)
    route_stats(data)
    return 0


def batches(idx, slots: int, count: int, start: int = 64, gap: int = 0):
    """16-row verify batches: ``slots`` sessions x 2 consecutive tokens; rows [first tokens of
    every slot, then their next tokens] as the engine's verify orders them. -> [count, 2*slots, L, K]."""
    import torch

    S, T = idx.shape[0], idx.shape[1]
    gap = gap or max(2, (T - start - 2) // count)
    out_b = []
    for c in range(count):
        p = start + c * gap
        if p + 1 >= T:
            break
        rows = [idx[s % S, p] for s in range(slots)] + [idx[s % S, p + 1] for s in range(slots)]
        out_b.append(torch.stack(rows))
    return torch.stack(out_b)


def route_stats(data) -> None:
    import torch

    for name, d in data.items():
        idx = d["idx"].long()
        L = idx.shape[2]
        for tp, slots in ((8, 8), (4, 4)):
            per = 256 // tp
            b = batches(idx, slots, 24)  # [N, R, L, K]
            uniq, busiest, mean_g, sum_max, max_sum = [], [], [], [], []
            for n in range(b.shape[0]):
                hit = torch.zeros((L, 256), dtype=torch.bool)
                for r in range(b.shape[1]):
                    hit[torch.arange(L)[:, None], b[n, r]] = True
                cnt = hit.view(L, tp, per).sum(-1)  # [L, tp]
                uniq.append(hit.sum(-1).float().mean().item())
                busiest.append(cnt.max(-1).values.float().mean().item())
                mean_g.append(cnt.float().mean().item())
                sum_max.append(cnt.max(-1).values.sum().item())
                max_sum.append(cnt.sum(0).max().item())
            # A static placement that spreads the hottest experts (by frequency in this set).
            freq = torch.zeros((L, 256))
            for r in range(idx.shape[1]):
                freq.scatter_add_(1, idx[:, r].reshape(-1, L, idx.shape[-1]).permute(1, 0, 2).reshape(L, -1),
                                  torch.ones(L, idx.shape[0] * idx.shape[-1]))
            order = freq.argsort(-1, descending=True)
            gpu_of = torch.zeros((L, 256), dtype=torch.long)
            for rank in range(256):
                rnd, pos = divmod(rank, tp)
                g = pos if rnd % 2 == 0 else tp - 1 - pos
                gpu_of[torch.arange(L), order[:, rank]] = g
            spread = []
            for n in range(b.shape[0]):
                hit = torch.zeros((L, 256), dtype=torch.bool)
                for r in range(b.shape[1]):
                    hit[torch.arange(L)[:, None], b[n, r]] = True
                cnt = torch.zeros((L, tp))
                cnt.scatter_add_(1, gpu_of, hit.float())
                spread.append(cnt.max(-1).values.mean().item())
            out("route_stats", set=name, tp=tp, rows=2 * slots, batches=b.shape[0],
                unique_per_layer=round(statistics.mean(uniq), 2),
                per_gcd_mean=round(statistics.mean(mean_g), 2),
                busiest_gcd_per_layer=round(statistics.mean(busiest), 2),
                busiest_over_mean=round(statistics.mean(busiest) / statistics.mean(mean_g), 3),
                sum_layers_busiest=round(statistics.mean(sum_max), 1),
                busiest_gcd_sum_layers=round(statistics.mean(max_sum), 1),
                hot_spread_busiest_per_layer=round(statistics.mean(spread), 2),
                uniform_unique_expect=round(256 * (1 - (1 - 8 / 256) ** (2 * slots)), 1))


# =============================================================================== parts

if triton is not None:

    @triton.jit
    def ep_up_kernel(h_ptr, idx_ptr, w_ptr, act_ptr, R, E0, stride_we,
                     H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr,
                     EL: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """act[e, r, n] = bf16(silu(bf16(Wg_e[n] . h_r)) * bf16(Wu_e[n] . h_r)) for local expert
        e (global E0 + e) and every row r that picked it; each expert's weights read once."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        rr = tl.arange(0, BP)
        rok = rr < R
        kk = tl.arange(0, TOPKP)
        ids = tl.load(idx_ptr + rr[:, None] * TOPK + kk[None, :],
                      mask=rok[:, None] & (kk[None, :] < TOPK), other=-1)
        ntile = I // BN
        for t in range(pid, EL * ntile, nprog):
            e = t // ntile
            tile = t % ntile
            hit = tl.sum((ids == E0 + e).to(tl.int32), axis=1) > 0
            nhit = tl.sum(hit.to(tl.int32), axis=0)
            if nhit > 0:
                n = tile * BN + tl.arange(0, BN)
                wbase = w_ptr + e.to(tl.int64) * stride_we
                acc_g = tl.zeros([BP, BN], dtype=tl.float32)
                acc_u = tl.zeros([BP, BN], dtype=tl.float32)
                for k0 in range(0, H, BK):
                    c = k0 + tl.arange(0, BK)
                    x = tl.load(h_ptr + rr[:, None] * H + c[None, :], mask=hit[:, None], other=0.0)
                    wg = tl.load(wbase + n[None, :] * H + c[:, None])
                    wu = tl.load(wbase + (n[None, :] + I) * H + c[:, None])
                    acc_g += tl.dot(x, wg)
                    acc_u += tl.dot(x, wu)
                g = acc_g.to(tl.bfloat16).to(tl.float32)
                u = acc_u.to(tl.bfloat16).to(tl.float32)
                a = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
                tl.store(act_ptr + (e * BP + rr[:, None]) * I + n[None, :], (a * u).to(tl.bfloat16),
                         mask=hit[:, None])

    @triton.jit
    def ep_down_kernel(act_ptr, idx_ptr, wt_ptr, w_ptr, part_ptr, R, E0, stride_we,
                       H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr,
                       EL: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """part[r, j, :] = bf16(bf16(Wd_e . act[e, r]) * w_rj) for every (row, slot j) whose
        expert e is local; each expert's down weights read once."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        rr = tl.arange(0, BP)
        rok = rr < R
        kk = tl.arange(0, TOPKP)
        km = rok[:, None] & (kk[None, :] < TOPK)
        ids = tl.load(idx_ptr + rr[:, None] * TOPK + kk[None, :], mask=km, other=-1)
        wts = tl.load(wt_ptr + rr[:, None] * TOPK + kk[None, :], mask=km, other=0.0)
        ntile = H // BN
        for t in range(pid, EL * ntile, nprog):
            e = t // ntile
            tile = t % ntile
            sel = ids == E0 + e
            hit = tl.sum(sel.to(tl.int32), axis=1) > 0
            nhit = tl.sum(hit.to(tl.int32), axis=0)
            if nhit > 0:
                wrow = tl.sum(tl.where(sel, wts, 0.0), axis=1)
                jrow = tl.sum(tl.where(sel, kk[None, :], 0), axis=1)
                n = tile * BN + tl.arange(0, BN)
                wbase = w_ptr + e.to(tl.int64) * stride_we
                acc = tl.zeros([BP, BN], dtype=tl.float32)
                for k0 in range(0, I, BK):
                    c = k0 + tl.arange(0, BK)
                    a = tl.load(act_ptr + (e * BP + rr[:, None]) * I + c[None, :], mask=hit[:, None], other=0.0)
                    w = tl.load(wbase + n[None, :] * I + c[:, None])
                    acc += tl.dot(a, w)
                p = (acc.to(tl.bfloat16).to(tl.float32) * wrow[:, None]).to(tl.bfloat16)
                tl.store(part_ptr + (rr[:, None] * TOPK + jrow[:, None]) * H + n[None, :], p,
                         mask=hit[:, None])

    @triton.jit
    def ep_combine_kernel(part_ptr, idx_ptr, r_ptr, sd_ptr, out_ptr, E0, N3,
                          H: tl.constexpr, TOPK: tl.constexpr, EL: tl.constexpr, E: tl.constexpr,
                          IS: tl.constexpr, BN: tl.constexpr):
        """This GCD's MoE partial of row r: bf16(sum of its local routed slots, in slot order)
        + bf16(bf16(sigmoid(shared gate)) * its shared-expert slice's down projection)."""
        row = tl.program_id(0)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        nm = n < H
        acc = tl.zeros([BN], dtype=tl.float32)
        for j in tl.static_range(TOPK):
            ex = tl.load(idx_ptr + row * TOPK + j)
            loc = (ex >= E0) & (ex < E0 + EL)
            acc += tl.load(part_ptr + (row * TOPK + j) * H + n, mask=nm & loc, other=0.0).to(tl.float32)
        routed = acc.to(tl.bfloat16).to(tl.float32)
        sc = tl.arange(0, IS)
        rbase = r_ptr + row * N3
        g = tl.load(rbase + E + sc).to(tl.float32)
        u = tl.load(rbase + E + IS + sc).to(tl.float32)
        sa = ((g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16).to(tl.float32)
        sw = tl.load(sd_ptr + n[:, None] * IS + sc[None, :], mask=nm[:, None], other=0.0).to(tl.float32)
        shared = tl.sum(sw * sa[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
        sg = tl.sigmoid(tl.load(rbase + E + 2 * IS).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        y = (routed + (sg * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + row * H + n, y, mask=nm)


def geometry() -> NS:
    """The text model's shapes from the pinned config (offline HF cache), else j86's values."""
    geo = NS(hidden=2048, nk=16, nv=32, dk=128, dv=128, conv=4, qh=16, kvh=2, dim=256,
             experts=256, inter=512, shared=512, top_k=8, vocab=248320, layers=40,
             types=(["linear_attention"] * 3 + ["full_attention"]) * 10, rot=64, eps=1e-6,
             theta=10_000_000.0, source="j86 defaults")
    try:
        sys.path.insert(0, str(ROOT))
        from transformers import AutoConfig

        from engine.generator import CHECKPOINT

        cfg = AutoConfig.from_pretrained(CHECKPOINT["model"], revision=CHECKPOINT["revision"])
        c = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        rope = getattr(c, "rope_parameters", None) or {}
        geo = NS(hidden=c.hidden_size, nk=c.linear_num_key_heads, nv=c.linear_num_value_heads,
                 dk=c.linear_key_head_dim, dv=c.linear_value_head_dim, conv=c.linear_conv_kernel_dim,
                 qh=c.num_attention_heads, kvh=c.num_key_value_heads, dim=c.head_dim,
                 experts=c.num_experts, inter=c.moe_intermediate_size,
                 shared=c.shared_expert_intermediate_size, top_k=c.num_experts_per_tok,
                 vocab=getattr(cfg, "vocab_size", None) or c.vocab_size, layers=c.num_hidden_layers,
                 types=list(c.layer_types),
                 rot=int(c.head_dim * rope.get("partial_rotary_factor",
                                               getattr(c, "partial_rotary_factor", 0.25))),
                 eps=c.rms_norm_eps, theta=float(rope.get("rope_theta", getattr(c, "rope_theta", 1e7))),
                 source="config")
    except Exception as exc:  # noqa: BLE001
        out("geometry_fallback", error=str(exc)[:200])
    return geo


def build_shard(dev: int, geo: NS, tp: int, rows: int, ctx: int, seed: int, rep_seed: int):
    """One GCD's shim of the model at 1/tp width: 40 layers + the MTP layer (attention),
    random BF16 weights of the real shard shapes; replicated tensors (norms, router, shared
    gate) from ``rep_seed`` so every GCD holds the same ones."""
    import torch

    from engine.fast.model import PAGE

    d = torch.device("cuda", dev)
    gen = torch.Generator(device=d)
    gen.manual_seed(seed)
    rgen = torch.Generator(device=d)
    rgen.manual_seed(rep_seed)

    def rnd(*shape, g=gen, scale=0.02):
        return (torch.randn(*shape, generator=g, device=d, dtype=torch.float32) * scale).to(torch.bfloat16)

    H = geo.hidden
    nk, nv = geo.nk // tp, geo.nv // tp
    kd, vd = nk * geo.dk, nv * geo.dv
    qh = geo.qh // tp
    kvh = max(1, geo.kvh * qh // geo.qh)
    el = geo.experts // tp
    ish = geo.shared // tp
    types = list(geo.types) + ["full_attention"]  # + the MTP head's layer
    slots = rows // 2
    rows_total = 2 * slots + 1
    pages_per_slot = ctx // PAGE + 2
    pages = slots * pages_per_slot + 1
    max_len = ((ctx + 64) // PAGE + 1) * PAGE
    layers, gdn_state, kv = [], {}, {}
    for i, kind in enumerate(types):
        ln1 = NS(weight=rnd(H, g=rgen, scale=0.1))
        ln2 = NS(weight=rnd(H, g=rgen, scale=0.1))
        mlp = NS(
            gate=NS(weight=rnd(geo.experts, H, g=rgen)),
            shared_expert=NS(gate_proj=NS(weight=rnd(ish, H)), up_proj=NS(weight=rnd(ish, H)),
                             down_proj=NS(weight=rnd(H, ish))),
            shared_expert_gate=NS(weight=rnd(1, H, g=rgen)),
            experts=NS(gate_up_proj=rnd(el, 2 * geo.inter, H), down_proj=rnd(el, H, geo.inter)),
        )
        ns = NS(input_layernorm=ln1, post_attention_layernorm=ln2, mlp=mlp)
        if kind == "linear_attention":
            ns.linear_attn = NS(
                in_proj_qkv=NS(weight=rnd(2 * kd + vd, H)), in_proj_z=NS(weight=rnd(vd, H)),
                in_proj_b=NS(weight=rnd(nv, H)), in_proj_a=NS(weight=rnd(nv, H)),
                conv1d=NS(weight=rnd(2 * kd + vd, 1, geo.conv, scale=0.3)),
                out_proj=NS(weight=rnd(H, vd)),
                A_log=torch.zeros(nv, device=d), dt_bias=torch.zeros(nv, device=d),
                norm=NS(weight=torch.ones(geo.dv, device=d, dtype=torch.bfloat16), variance_epsilon=geo.eps),
                num_k_heads=nk, num_v_heads=nv, head_k_dim=geo.dk, head_v_dim=geo.dv, key_dim=kd,
                value_dim=vd, conv_dim=2 * kd + vd, conv_kernel_size=geo.conv,
            )
            gdn_state[i] = ((torch.randn((rows_total, 2 * kd + vd, geo.conv), device=d) * 0.1).to(torch.bfloat16),
                            torch.randn((rows_total, nv, geo.dk, geo.dv), device=d) * 0.01)
        else:
            ns.self_attn = NS(
                q_proj=NS(weight=rnd(qh * 2 * geo.dim, H)), k_proj=NS(weight=rnd(kvh * geo.dim, H)),
                v_proj=NS(weight=rnd(kvh * geo.dim, H)), o_proj=NS(weight=rnd(H, qh * geo.dim)),
                q_norm=NS(weight=torch.zeros(geo.dim, device=d, dtype=torch.bfloat16), eps=geo.eps),
                k_norm=NS(weight=torch.zeros(geo.dim, device=d, dtype=torch.bfloat16), eps=geo.eps),
                scaling=geo.dim ** -0.5,
            )
            shape = (pages, kvh, PAGE, geo.dim)
            kv[i] = ((torch.randn(shape, device=d) * 0.5).to(torch.bfloat16),
                     (torch.randn(shape, device=d) * 0.5).to(torch.bfloat16))
        layers.append(ns)
    table = torch.zeros((slots + 1, max_len // PAGE), dtype=torch.int32, device=d)
    for s in range(slots):
        table[s, :pages_per_slot] = torch.arange(s * pages_per_slot, (s + 1) * pages_per_slot)
    inv = 1.0 / (geo.theta ** (torch.arange(0, geo.rot, 2, dtype=torch.float32) / geo.rot))
    shim = NS(layers=layers, layer_types=types, inv_freq={d: inv.to(d)}, max_len=max_len,
              embed=torch.zeros(1, H, dtype=torch.bfloat16, device=d), gdn_state=gdn_state, kv=kv,
              top_k=geo.top_k, eps=geo.eps, mtp=None)
    lm_head = rnd(geo.vocab // tp, H)
    final_norm = rnd(H, g=rgen, scale=0.1)
    meta = NS(page_table=table)
    slots_t = torch.cat([torch.arange(slots), torch.arange(slots)]).to(d)
    states_t = torch.cat([torch.arange(slots), torch.arange(slots) + slots]).to(d)
    pos_t = torch.cat([torch.full((slots,), ctx), torch.full((slots,), ctx + 1)]).to(d)
    return NS(shim=shim, lm_head=lm_head, final_norm=final_norm, meta=meta, slots=slots_t,
              states=states_t, positions=pos_t, el=el, e0=(dev % tp) * el, types=types, ish=ish)


class Shard:
    """One GCD's fused layer pieces (FusedDecode on the shim) plus the local-expert MoE."""

    def __init__(self, dev: int, geo: NS, tp: int, rows: int, ctx: int, routes, cfg: dict) -> None:
        import torch

        from engine.fast import fused

        self.dev, self.geo, self.tp, self.rows, self.cfg = dev, geo, tp, rows, cfg
        with torch.cuda.device(dev):
            self.s = build_shard(dev, geo, tp, rows, ctx, seed=1000 + dev, rep_seed=7 + dev // tp)
            self.fd = fused.FusedDecode(self.s.shim)
            d = torch.device("cuda", dev)
            n_layers = len(self.s.types)
            # Real routing, per layer, for this GCD's rows: routes [N, rows, L, K] (L = 40; the
            # MTP layer reuses layer 39's routes).
            self.routes_idx = routes["idx"].to(d, torch.int32)
            self.routes_wt = routes["wt"].to(d, torch.float32)
            self.idx = torch.zeros((n_layers, rows, geo.top_k), dtype=torch.int32, device=d)
            self.wt = torch.zeros((n_layers, rows, geo.top_k), dtype=torch.float32, device=d)
            self.load_routes(0)
            self.x0 = (torch.randn(rows, geo.hidden, generator=torch.Generator(device=d).manual_seed(3),
                                   device=d) * 0.3).to(torch.bfloat16)
            self.cus = torch.cuda.get_device_properties(d).multi_processor_count

    def load_routes(self, k: int) -> None:
        n = self.routes_idx.shape[0]
        L = self.routes_idx.shape[2]
        src_i = self.routes_idx[k % n]  # [rows, L, K]
        src_w = self.routes_wt[k % n]
        nl = self.idx.shape[0]
        lay = [min(i, L - 1) for i in range(nl)]
        self.idx.copy_(src_i[:, lay].permute(1, 0, 2))
        self.wt.copy_(src_w[:, lay].permute(1, 0, 2))

    # ------------------------------------------------------------------ pieces

    def mixer(self, i: int, x, pending):
        """add_rms(x + pending) and the 1/tp mixer; (x, partial mixer output)."""
        from engine.fast import kernels

        lay = self.s.shim.layers[i]
        x, h = kernels.add_rms(x, pending, lay.input_layernorm.weight, self.geo.eps)
        if self.s.types[i] == "linear_attention":
            m = self.fd._gdn_verify(i, h, self.s.states)
        else:
            m = self.fd._attn(i, h, self.s.slots, self.s.positions, self.s.meta)
        return x, m

    def moe(self, i: int, x, mixed):
        """add_rms(x + mixed) and this GCD's MoE partial (local experts + shared slice)."""
        import torch

        from engine.fast import fused, kernels

        geo, cfg = self.geo, self.cfg
        lay = self.s.shim.layers[i]
        x, h = kernels.add_rms(x, mixed, lay.post_attention_layernorm.weight, geo.eps)
        w3 = self.fd.w_moe[i]
        r = self.fd._gemv(h, w3)
        rows = h.shape[0]
        # The router's own top-k (its cost is real); the experts below use the recorded routes.
        sidx = torch.empty((rows, geo.top_k), dtype=torch.int32, device=h.device)
        swt = torch.empty((rows, geo.top_k), dtype=torch.float32, device=h.device)
        fused._topk_kernel[(rows,)](r, sidx, swt, w3.shape[0], E=geo.experts, TOPK=geo.top_k,
                                    TOPKP=triton.next_power_of_2(geo.top_k), num_warps=1)
        part = self.local_experts(i, h)
        y = torch.empty_like(h)
        sd = lay.mlp.shared_expert.down_proj.weight
        ep_combine_kernel[(rows, triton.cdiv(geo.hidden, 256))](
            part, self.idx[i], r, sd, y, self.s.e0, w3.shape[0], H=geo.hidden, TOPK=geo.top_k,
            EL=self.s.el, E=geo.experts, IS=self.s.ish, BN=256, num_warps=4)
        return x, y

    def local_experts(self, i: int, h):
        import torch

        geo, cfg = self.geo, self.cfg
        ex = self.s.shim.layers[i].mlp.experts
        gu, dn = ex.gate_up_proj, ex.down_proj
        rows = h.shape[0]
        bp = max(16, triton.next_power_of_2(rows))
        act = torch.empty((self.s.el, bp, geo.inter), dtype=h.dtype, device=h.device)
        part = torch.empty((rows, geo.top_k, geo.hidden), dtype=h.dtype, device=h.device)
        grid = min(self.s.el * (geo.inter // cfg["up_bn"]), cfg["grid_mult"] * self.cus)
        ep_up_kernel[(grid,)](h, self.idx[i], gu, act, rows, self.s.e0, gu.stride(0),
                              H=geo.hidden, I=geo.inter, TOPK=geo.top_k,
                              TOPKP=triton.next_power_of_2(geo.top_k), EL=self.s.el, BP=bp,
                              BN=cfg["up_bn"], BK=cfg["up_bk"], num_warps=4)
        grid = min(self.s.el * (geo.hidden // cfg["dn_bn"]), cfg["grid_mult"] * self.cus)
        ep_down_kernel[(grid,)](act, self.idx[i], self.wt[i], dn, part, rows, self.s.e0, dn.stride(0),
                                H=geo.hidden, I=geo.inter, TOPK=geo.top_k,
                                TOPKP=triton.next_power_of_2(geo.top_k), EL=self.s.el, BP=bp,
                                BN=cfg["dn_bn"], BK=cfg["dn_bk"], num_warps=4)
        return part

    def moe_reference(self, i: int, h):
        """PyTorch reference of local_experts + combine (no shared part): [rows, H] fp32."""
        import torch
        import torch.nn.functional as F

        ex = self.s.shim.layers[i].mlp.experts
        idx, wt = self.idx[i].long(), self.wt[i]
        y = torch.zeros((h.shape[0], self.geo.hidden), dtype=torch.float32, device=h.device)
        for r in range(h.shape[0]):
            for j in range(self.geo.top_k):
                e = int(idx[r, j]) - self.s.e0
                if 0 <= e < self.s.el:
                    gu = F.linear(h[r:r + 1].float(), ex.gate_up_proj[e].float())
                    g, u = gu[:, :self.geo.inter], gu[:, self.geo.inter:]
                    a = F.silu(g) * u
                    y[r] += (F.linear(a, ex.down_proj[e].float()) * wt[r, j])[0]
        return y

    def lm(self, x):
        from engine.fast import kernels

        _, h = kernels.add_rms(x, None, self.s.final_norm, self.geo.eps)
        logits = self.fd._gemv(h, self.s.lm_head)
        return logits.float().max(-1)

    # ------------------------------------------------------------------ the composed step

    def step(self, ar, n_layers: int | None = None):
        """The whole verify step on this GCD; ``ar(dev, t)`` the all-reduce (None: skipped)."""
        import torch

        x = self.x0
        pending = None
        n = n_layers or len(self.s.types)
        red = (lambda t: ar(self.dev, t)) if ar is not None else (lambda t: t)
        for i in range(n):
            x, m = self.mixer(i, x, pending)
            x, y = self.moe(i, x, red(m))
            pending = red(y)
        from engine.fast import kernels

        x, _ = kernels.add_rms(x, pending, self.s.final_norm, self.geo.eps)
        picks = []
        for _ in range(2):  # verify logits, then the MTP head's draft logits
            v, ix = self.lm(x)
            pair = torch.stack([v, ix.float()], dim=-1).contiguous()  # local max + index
            picks.append(red(pair))
        return x, picks


def part_parts(tp: int, ctx: int, rounds: int, routes_set: str, cfg: dict) -> int:
    import torch

    os.environ["ENGINE_FUSED_MAX_ROWS"] = "16"
    sys.path.insert(0, str(ROOT))
    from engine.fast import fused, kernels  # noqa: F401  (after MAX_ROWS)

    torch.set_grad_enabled(False)
    n = torch.cuda.device_count()
    assert n == 8, f"needs 8 GCDs, sees {n}"
    hip = _hip()
    for d in range(n):
        torch.zeros(1, device=f"cuda:{d}")
    enable_all_peers(hip, n)
    geo = geometry()
    out("geometry", tp=tp, **{k: (v if not isinstance(v, list) else len(v)) for k, v in vars(geo).items()})
    rows = 16 if tp == 8 else 8
    slots = rows // 2
    # Routing: recorded router outputs, 16-row (or 8-row) verify batches.
    try:
        data = torch.load(ROUTES)
        src = data.get(routes_set) or next(iter(data.values()))
        groups = 8 // tp
        rb = {}
        for grp in range(groups):
            sub = {k: v[grp * slots:(grp + 1) * slots] if v.shape[0] >= (grp + 1) * slots else v
                   for k, v in src.items()}
            rb[grp] = {"idx": batches(sub["idx"].long(), slots, 24), "wt": batches(sub["wt"], slots, 24)}
        route_note = f"recorded ({routes_set})"
    except Exception as exc:  # noqa: BLE001
        out("routes_missing", error=str(exc)[:200], fallback="uniform random top-8")
        g = torch.Generator().manual_seed(0)
        rb = {}
        for grp in range(8 // tp):
            idx = torch.stack([torch.stack([torch.randperm(geo.experts, generator=g)[:geo.top_k]
                                            for _ in range(geo.layers)]) for _ in range(24 * rows)])
            idx = idx.view(24, rows, geo.layers, geo.top_k)
            rb[grp] = {"idx": idx, "wt": torch.full(idx.shape, 1.0 / geo.top_k)}
        route_note = "uniform random"
    t0 = time.perf_counter()
    shards = {d: Shard(d, geo, tp, rows, ctx, rb[d // tp], cfg) for d in range(8)}
    torch.cuda.synchronize()
    out("parts_setup", tp=tp, rows=rows, ctx=ctx, routes=route_note, seconds=round(time.perf_counter() - t0, 1),
        mem_alloc_gb=[round(torch.cuda.memory_allocated(d) / 2**30, 1) for d in range(8)],
        busiest_counts=None)
    # Expert counts per GCD in these batches (the experts each GCD must read per layer).
    cnt = []
    for d in range(8):
        sh = shards[d]
        ri = sh.routes_idx  # [N, rows, L, K]
        loc = (ri >= sh.s.e0) & (ri < sh.s.e0 + sh.s.el)
        per = []
        for k in range(ri.shape[0]):
            per.append(sum(len(torch.unique(ri[k, :, l][loc[k, :, l]])) for l in range(ri.shape[2])))
        cnt.append(per)
    busiest = [max(cnt[d][k] for d in range(8)) for k in range(len(cnt[0]))]
    out("expert_reads", tp=tp, rows=rows, layers=geo.layers,
        per_gcd_mean_per_layer=[round(statistics.mean(c) / geo.layers, 2) for c in cnt],
        busiest_gcd_sum_per_step=round(statistics.mean(busiest), 1),
        busiest_gcd_per_layer=round(statistics.mean(busiest) / geo.layers, 2))

    # Correctness of the local-expert kernels against PyTorch (one layer, every GCD).
    for d in range(8):
        sh = shards[d]
        with torch.cuda.device(d):
            from engine.fast import kernels as K

            _, h = K.add_rms(sh.x0, None, sh.s.shim.layers[0].post_attention_layernorm.weight, geo.eps)
            part = sh.local_experts(0, h)
            got = torch.zeros((rows, geo.hidden), dtype=torch.float32, device=h.device)
            idx = sh.idx[0].long()
            for j in range(geo.top_k):
                loc = ((idx[:, j] >= sh.s.e0) & (idx[:, j] < sh.s.e0 + sh.s.el)).float()[:, None]
                got += torch.nan_to_num(part[:, j].float()) * loc
            ref = sh.moe_reference(0, h)
            err = (got - ref).abs().max().item()
            scale = ref.abs().max().item()
            if d in (0, 7):
                out("ep_moe_check", device=d, max_abs_err=round(err, 5), ref_max=round(scale, 4),
                    rel=round(err / max(scale, 1e-9), 4), ok=int(err <= 0.02 * max(scale, 1e-6) + 1e-3))
            elif err > 0.02 * max(scale, 1e-6) + 1e-3:
                out("ep_moe_check", device=d, max_abs_err=err, ref_max=scale, ok=0)

    def pre(d, k):
        shards[d].load_routes(k)

    def timed(name: str, fn_of_dev, per_unit: int, count: int = rounds) -> dict:
        try:
            graphs = capture_all(fn_of_dev)
            concurrent_replay(graphs, 3, pre)
            res = concurrent_replay(graphs, count, pre)
            per_dev = {d: statistics.median(res[d][0]) / per_unit for d in range(8)}
            per_round = [max(res[d][0][k] for d in range(8)) / per_unit for k in range(count)]
            r = dict(p50_ms=round(pct(per_round, 50), 4), p99_ms=round(pct(per_round, 99), 4),
                     busiest_gcd_median_ms=round(max(per_dev.values()), 4),
                     gcd_median_ms=[round(per_dev[d], 4) for d in range(8)],
                     host_launch_ms=round(pct(res[0][1], 50), 3))
            out("part", name=name, tp=tp, rows=rows, per=per_unit, **r)
            del graphs
            return r
        except Exception as exc:  # noqa: BLE001
            out("part", name=name, tp=tp, error=f"{type(exc).__name__}: {str(exc)[:300]}")
            traceback.print_exc()
            return {}

    gdn_layers = [i for i, t in enumerate(shards[0].s.types) if t == "linear_attention"]
    attn_layers = [i for i, t in enumerate(shards[0].s.types) if t != "linear_attention"]
    nl = len(shards[0].s.types)
    res = {}

    def run_mixers(d, layers):
        sh = shards[d]
        with torch.cuda.device(d):
            for i in layers:
                sh.mixer(i, sh.x0, None)

    def run_moes(d):
        sh = shards[d]
        with torch.cuda.device(d):
            for i in range(nl):
                sh.moe(i, sh.x0, None)

    def run_lm(d, impl):
        sh = shards[d]
        with torch.cuda.device(d):
            for _ in range(4):
                if impl == "fused":
                    sh.fd._gemv(sh.x0, sh.s.lm_head)
                else:
                    torch.nn.functional.linear(sh.x0, sh.s.lm_head)

    res["gdn"] = timed("gdn_layer", lambda d: run_mixers(d, gdn_layers), len(gdn_layers))
    res["attn"] = timed("attn_layer", lambda d: run_mixers(d, attn_layers), len(attn_layers))
    res["moe"] = timed("moe_layer", run_moes, nl)
    res["lm_fused"] = timed("lm_head_fused", lambda d: run_lm(d, "fused"), 4)
    res["lm_torch"] = timed("lm_head_torch", lambda d: run_lm(d, "torch"), 4)
    # Single GEMV shapes at this row count: fused GEMV vs torch (hipBLASLt).
    for name, i, attr in (("gdn_in", gdn_layers[0], "w_in"), ("attn_in", attn_layers[0], "w_in"),
                          ("moe_router", 0, "w_moe")):
        w = getattr(shards[0].fd, attr)[i]
        for impl in ("fused", "torch"):
            def f(d, w_of=lambda d: getattr(shards[d].fd, attr)[i], impl=impl):
                sh = shards[d]
                with torch.cuda.device(d):
                    for _ in range(10):
                        if impl == "fused":
                            sh.fd._gemv(sh.x0, w_of(d))
                        else:
                            torch.nn.functional.linear(sh.x0, w_of(d))
            timed(f"gemv_{name}_{impl}_{w.shape[0]}x{w.shape[1]}", f, 10, count=20)

    # The composed step: no all-reduce (compute only), then each all-reduce layout.
    ar_kinds = ["none"] + (["flat8", "hier"] if tp == 8 else ["tp4"])
    steps = {}
    for kind in ar_kinds:
        try:
            ar = None
            if kind != "none":
                layout = AllReduce(hip, kind, rows * geo.hidden // 2, 64)
                ar = layout.run
            outs = {}

            def stepfn(d, ar=ar):
                with torch.cuda.device(d):
                    x, picks = shards[d].step(ar)
                    outs[d] = (x, picks)

            r = timed(f"step_{kind}", stepfn, 1)
            steps[kind] = r
            if kind != "none" and r:
                to, bad = layout.errors()
                # Replicated hidden state: every member of a TP group must hold the same bits.
                same = []
                for d in range(8):
                    ref = outs[(d // tp) * tp][0]
                    same.append(bool(torch.equal(outs[d][0].cpu().view(torch.int16),
                                                 ref.cpu().view(torch.int16))))
                out("step_check", kind=kind, tp=tp, ar_timeouts=to, replicated_hidden_identical=same,
                    ok=int(to == 0 and all(same)))
        except Exception as exc:  # noqa: BLE001
            out("part", name=f"step_{kind}", tp=tp, error=f"{type(exc).__name__}: {str(exc)[:300]}")
            traceback.print_exc()
    out("parts_done", tp=tp)
    return 0


# =============================================================================== driver


def run_part(argv: list[str], timeout: int, env_extra: dict | None = None) -> list[str]:
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [sys.executable, str(Path(__file__).resolve())] + argv
    print(f"===== part: {' '.join(argv)} env={env_extra or {}} (timeout {timeout}s)", flush=True)
    lines: list[str] = []
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(ROOT))
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
    finally:
        killer.cancel()
    if time.perf_counter() - t0 >= timeout:
        print(f"RESULT part_timeout {json.dumps(argv)}", flush=True)
    print(f"===== part done rc={proc.returncode} s={time.perf_counter() - t0:.0f}", flush=True)
    return lines


def results(lines: list[str], tag: str) -> list[dict]:
    got = []
    for line in lines:
        m = re.match(rf"RESULT {tag} (\{{.*\}})\s*$", line)
        if m:
            try:
                got.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass
    return got


def verdict(lines: list[str], ref_ms: float) -> None:
    ars = results(lines, "ar")
    best = {}
    for r in ars:
        if r.get("ok") == 1 and "p50_us" in r:
            key = (r["kind"], r["bytes"])
            if key not in best or r["p50_us"] < best[key]["p50_us"]:
                best[key] = r
    bad_ar = [r for r in ars if r.get("ok") == 0]
    for (kind, nbytes), r in sorted(best.items()):
        out("summary_ar", kind=kind, bytes=nbytes, chunk=r["chunk"], p50_us=r["p50_us"], p99_us=r["p99_us"])
    for r in results(lines, "rccl"):
        out("summary_rccl", **r)
    b64 = [best[k]["p50_us"] for k in (("flat8", 65536), ("hier", 65536)) if k in best]
    ar64 = min(b64) if b64 else None
    ar4 = min([best[k]["p50_us"] for k in (("flat8", 4096), ("hier", 4096)) if k in best] or [float("nan")])
    parts = {(r.get("name"), r.get("tp")): r for r in results(lines, "part")}

    def p(name, tp=8, key="busiest_gcd_median_ms"):
        return parts.get((name, tp), {}).get(key)

    gdn, attn, moe = p("gdn_layer"), p("attn_layer"), p("moe_layer")
    lm = min([v for v in (p("lm_head_fused"), p("lm_head_torch")) if v is not None] or [float("nan")])
    proj = None
    if None not in (gdn, attn, moe, ar64):
        # 30 GDN + 10 attention layers + the MTP layer (attention); 2 all-reduces per layer
        # (64 KB at 16 rows); final norm folded; lm_head slice twice (verify + MTP draft) and a
        # small all-reduce for each argmax.
        proj = 30 * gdn + 11 * attn + 41 * moe + 82 * ar64 / 1000 + 2 * lm + 2 * ar4 / 1000
        out("projection", formula="30*gdn + 11*attn + 41*moe(busiest GCD, real routes) + 82*AR64 + 2*lm + 2*AR4",
            gdn_ms=gdn, attn_ms=attn, moe_ms=moe, ar64_us=ar64, lm_ms=lm, ar4_us=ar4,
            step_ms=round(proj, 3), ref_stack_f_B2_verify_ms=ref_ms,
            node_decode_rate_ratio=round(ref_ms / proj, 2))
    for kind in ("none", "flat8", "hier"):
        v = p(f"step_{kind}", key="p50_ms")
        if v is not None:
            out("composed_step", kind=kind, tp=8, p50_ms=v, p99_ms=p(f"step_{kind}", key="p99_ms"),
                node_decode_rate_ratio_vs_ref=round(ref_ms / v, 2))
    for kind in ("none", "tp4"):
        v = p(f"step_{kind}", tp=4, key="p50_ms")
        if v is not None:
            # Two TP4 engines of 8 rows each: 16 rows per step on the node.
            out("composed_step", kind=kind, tp=4, p50_ms=v, node_decode_rate_ratio_vs_ref=round(ref_ms / v, 2))
    checks = results(lines, "step_check") + results(lines, "ep_moe_check")
    failed_checks = [c for c in checks if c.get("ok") == 0]
    composed = [p(f"step_{k}", key="p50_ms") for k in ("flat8", "hier")]
    composed = [c for c in composed if c is not None]
    step_for_rule = min(composed) if composed else proj
    kill_step = step_for_rule is None or step_for_rule > 12.0
    kill_ar = ar64 is None or ar64 > 20.0
    kill_err = bool(bad_ar) or bool(failed_checks)
    out("verdict", step_ms_for_rule=step_for_rule, projected_ms=None if proj is None else round(proj, 3),
        kill_step_gt_12ms=kill_step, ar64_best_p50_us=ar64, kill_ar_gt_20us=kill_ar,
        mismatch_or_timeout=kill_err, failed=[*(f"ar {r.get('kind')} {r.get('bytes')}" for r in bad_ar),
                                             *(json.dumps(c) for c in failed_checks)],
        recommend="park" if (kill_step or kill_ar or kill_err) else "fund")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all")
    ap.add_argument("--sizes", default="4096,16384,65536")
    ap.add_argument("--kinds", default="flat8,hier,tp4")
    ap.add_argument("--chunks", default="64,256")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--routes-set", default="replay")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--length", type=int, default=1800)
    ap.add_argument("--ref-ms", type=float, default=28.7)
    ap.add_argument("--up-bn", type=int, default=16)
    ap.add_argument("--up-bk", type=int, default=128)
    ap.add_argument("--dn-bn", type=int, default=32)
    ap.add_argument("--dn-bk", type=int, default=128)
    ap.add_argument("--grid-mult", type=int, default=4)
    ap.add_argument("--skip", default="", help="comma list of parts to skip in --part all")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    cfg = dict(up_bn=args.up_bn, up_bk=args.up_bk, dn_bn=args.dn_bn, dn_bk=args.dn_bk,
               grid_mult=args.grid_mult)
    if args.part == "topo":
        return part_topo()
    if args.part == "ar":
        return part_ar(sizes, args.kinds.split(","), [int(c) for c in args.chunks.split(",")])
    if args.part == "rccl":
        return part_rccl(sizes)
    if args.part == "routes":
        return part_routes(args.sessions, args.length, ROUTES)
    if args.part == "parts":
        return part_parts(args.tp, args.ctx, args.rounds, args.routes_set, cfg)
    skip = set(filter(None, args.skip.split(",")))
    lines: list[str] = []
    common = ["--ctx", str(args.ctx), "--rounds", str(args.rounds), "--routes-set", args.routes_set,
              "--up-bn", str(args.up_bn), "--up-bk", str(args.up_bk), "--dn-bn", str(args.dn_bn),
              "--dn-bk", str(args.dn_bk), "--grid-mult", str(args.grid_mult)]
    if "topo" not in skip:
        lines += run_part(["--part", "topo"], 300)
    if "ar" not in skip:
        lines += run_part(["--part", "ar", "--sizes", args.sizes, "--kinds", args.kinds,
                           "--chunks", args.chunks], 1500)
    if "rccl" not in skip:
        lines += run_part(["--part", "rccl", "--sizes", args.sizes], 600)
    if "routes" not in skip:
        lines += run_part(["--part", "routes", "--sessions", str(args.sessions), "--length",
                           str(args.length)], 1500, {"HIP_VISIBLE_DEVICES": "0,1"})
    if "parts" not in skip:
        lines += run_part(["--part", "parts", "--tp", "8"] + common, 2400)
    if "parts4" not in skip:
        lines += run_part(["--part", "parts", "--tp", "4"] + common, 2400)
    verdict(lines, args.ref_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
