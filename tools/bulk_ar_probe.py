"""Plan 52 step 0: is a bulk ("simple" protocol) in-graph TP2 all-reduce worth building?

    python tools/bulk_ar_probe.py [--devices auto|a,b] [--replays 2000] [--reps 20]

Two GCDs of one card (``--devices auto`` picks the card pair with the most free memory, then
re-runs itself with ``HIP_VISIBLE_DEVICES`` set to it). No engine change: the production LL
all-reduce is ``engine.fast.tp._LL._launch``; the bulk kernel is a draft defined here.

Bulk protocol (``_bulk_ar_kernel``): one CTA per chunk of BW 32-bit words. Phase 1: load the
chunk of x, store it to the peer's receive buffer (parity ``e & 1``) as plain data with
write-through stores, a CTA barrier, then one system-scope release exchange of the peer's
flag of this CTA to the CTA's epoch e. Phase 2: spin with system-scope acquire on the CTA's own
flag until it reaches e (wrapping compare, ``MAXIT`` bound, timeouts counted), a barrier, read
the peer's chunk with uncached loads and sum with the LL kernel's exact formula
(``bf16(fp32 a + fp32 b)``, then ``bf16(res + that)``). Epochs live in device memory per CTA,
as LL's, so the graphs need no host input.

Peer copy variant (``pc``): a captured ``hipMemcpyAsync`` pushes x into the peer's receive
buffer (parity fixed per captured call, alternating), then one kernel: CTA 0 releases the
peer's flag, every CTA acquires its own and sums.

Parts (all ``RESULT`` lines):
(a) per call time, rows 64/576/1024/1792/2048 x 2048 BF16, with and without residual, LL vs
    bulk (BW, num_warps) variants, each a HIP graph of 84 back-to-back calls per rank (a
    prefill pass's count), median of ``--reps`` replays of both ranks;
(b) peer write bandwidth at 1792 rows: store-only kernels (bulk layouts) and the copy engine
    (captured hipMemcpyAsync), both directions at once; the peer copy all-reduce variant;
(c) correctness: per bulk variant ``--replays`` graph replays of fresh random data (graph-safe
    RNG, the same seed on both GPUs so each rank also knows the peer's data), calls of
    576/1792/2048 rows with and without residual, each compared bitwise with the LL kernel's
    output and with the torch formula; epochs and flags start just below 2^31 so they wrap
    during the run; parity flips at every call; timeouts must be 0. Also the loopback
    (self-peer) mode against ``bf16(x + x) [+ res]``.

Kill (plan 52): park if the best bulk variant is > 100 us per call at 1792 rows, or if any
replay is not bitwise equal or times out.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALLS = 84  # all-reduces per prefill pass (2 per layer x 40 layers + MTP + head)
HID = 2048
ROWS = (64, 576, 1024, 1792, 2048)
VARIANTS = ((1024, 1), (2048, 1), (4096, 1), (4096, 4), (8192, 4), (8192, 8), (16384, 8))
BULK_CAP = 2048 * HID // 2  # words per parity buffer (2,048 rows of BF16)
MAXIT = 2_000_000
START = (1 << 31) - 3000  # epochs and flags start here in (c): they wrap during the run


def pick_pair() -> str:
    import torch

    free = []
    for d in range(torch.cuda.device_count()):
        f, t = torch.cuda.mem_get_info(d)
        free.append(f)
    pairs = [(a, a + 1) for a in range(0, len(free) - 1, 2)]
    best = max(pairs, key=lambda p: min(free[p[0]], free[p[1]]))
    print("RESULT free_gb " + " ".join(f"{d}:{f / 2**30:.1f}" for d, f in enumerate(free)))
    return f"{best[0]},{best[1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--replays", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()
    if not args.child:
        devs = args.devices
        if devs == "auto":
            out = subprocess.run([sys.executable, "-c",
                                  "import sys; sys.path.insert(0, %r); import bulk_ar_probe as b; "
                                  "print(b.pick_pair())" % str(ROOT / "tools")],
                                 capture_output=True, text=True, timeout=300)
            print(out.stdout, end="")
            devs = out.stdout.strip().splitlines()[-1]
        env = dict(os.environ, HIP_VISIBLE_DEVICES=devs)
        print(f"RESULT devices HIP_VISIBLE_DEVICES={devs}", flush=True)
        cmd = [sys.executable, "-u", "-X", "faulthandler", __file__, "--child",
               "--replays", str(args.replays), "--reps", str(args.reps)]
        try:
            return subprocess.run(cmd, env=env, timeout=2400).returncode
        except subprocess.TimeoutExpired:
            print("RESULT child timed out (hang guard)")
            return 3
    return child(args)


# ============================================================================== kernels

import torch  # noqa: E402

from engine.fast import tp  # noqa: E402

triton, tl = tp.triton, tp.tl


@triton.jit
def _sum_words(xi, yi, res_ptr, offs, m, RES: tl.constexpr):
    """The LL kernel's BF16 formula on 32-bit words of BF16 pairs."""
    s_lo = (((xi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
            + ((yi & 0xFFFF) << 16).to(tl.float32, bitcast=True))
    s_hi = ((xi & -65536).to(tl.float32, bitcast=True)
            + (yi & -65536).to(tl.float32, bitcast=True))
    s_lo = s_lo.to(tl.bfloat16).to(tl.float32)
    s_hi = s_hi.to(tl.bfloat16).to(tl.float32)
    if RES:
        ri = tl.load(res_ptr + offs, mask=m, other=0)
        s_lo = (((ri & 0xFFFF) << 16).to(tl.float32, bitcast=True) + s_lo).to(tl.bfloat16).to(tl.float32)
        s_hi = ((ri & -65536).to(tl.float32, bitcast=True) + s_hi).to(tl.bfloat16).to(tl.float32)
    lo = s_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    hi = s_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
    return lo | hi


@triton.jit
def _bulk_ar_kernel(x_ptr, res_ptr, out_ptr, recv_local, recv_peer, flag_local, flag_peer,
                    epoch_ptr, err_ptr, NW, CAP, BW: tl.constexpr, RES: tl.constexpr,
                    MAXIT: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(epoch_ptr + p) + 1
    buf = (e & 1).to(tl.int64) * CAP
    offs = p.to(tl.int64) * BW + tl.arange(0, BW)
    m = offs < NW
    xi = tl.load(x_ptr + offs, mask=m, other=0)
    tl.store(recv_peer + buf + offs, xi, mask=m, cache_modifier=".wt")
    tl.debug_barrier()
    tl.atomic_xchg(flag_peer + p, e, sem="release", scope="sys")
    v = tl.atomic_add(flag_local + p, 0, sem="acquire", scope="sys")
    it = v * 0
    while ((v - e) < 0) & (it < MAXIT):
        v = tl.atomic_add(flag_local + p, 0, sem="acquire", scope="sys")
        it += 1
    if (v - e) < 0:
        tl.atomic_add(err_ptr, 1)
    tl.debug_barrier()
    yi = tl.load(recv_local + buf + offs, mask=m, other=0, volatile=True, cache_modifier=".cv")
    tl.store(out_ptr + offs, _sum_words(xi, yi, res_ptr, offs, m, RES), mask=m)
    tl.store(epoch_ptr + p, e)


@triton.jit
def _store_only_kernel(x_ptr, recv_peer, NW, BW: tl.constexpr):
    p = tl.program_id(0)
    offs = p.to(tl.int64) * BW + tl.arange(0, BW)
    m = offs < NW
    xi = tl.load(x_ptr + offs, mask=m, other=0)
    tl.store(recv_peer + offs, xi, mask=m, cache_modifier=".wt")


@triton.jit
def _pc_sum_kernel(x_ptr, res_ptr, out_ptr, recv_buf, flag_local, flag_peer, epoch_ptr, err_ptr,
                   NW, BW: tl.constexpr, RES: tl.constexpr, MAXIT: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(epoch_ptr + p) + 1
    if p == 0:
        tl.atomic_xchg(flag_peer, e, sem="release", scope="sys")
    v = tl.atomic_add(flag_local, 0, sem="acquire", scope="sys")
    it = v * 0
    while ((v - e) < 0) & (it < MAXIT):
        v = tl.atomic_add(flag_local, 0, sem="acquire", scope="sys")
        it += 1
    if (v - e) < 0:
        tl.atomic_add(err_ptr, 1)
    tl.debug_barrier()
    offs = p.to(tl.int64) * BW + tl.arange(0, BW)
    m = offs < NW
    xi = tl.load(x_ptr + offs, mask=m, other=0)
    yi = tl.load(recv_buf + offs, mask=m, other=0, volatile=True, cache_modifier=".cv")
    tl.store(out_ptr + offs, _sum_words(xi, yi, res_ptr, offs, m, RES), mask=m)
    tl.store(epoch_ptr + p, e)


# ============================================================================== state


class Bulk:
    """Receive buffers (shared by variants) and per-variant flags/epochs for both ranks."""

    def __init__(self, ll: "tp._LL") -> None:
        self.ll = ll
        self.devs = ll.devices
        self.recv = [self._fine(r, 2 * BULK_CAP * 4) for r in (0, 1)]
        self.loop = [self._fine(r, 2 * BULK_CAP * 4) for r in (0, 1)]
        self.err = [torch.zeros(1, dtype=torch.int32, device=d) for d in self.devs]
        self.state: dict = {}

    def _fine(self, rank: int, nbytes: int):
        raw = self.ll._fine(self.devs[rank], nbytes)
        return tp._Raw(raw.ptr, torch.int32)

    def var(self, bw: int):
        if bw not in self.state:
            n = BULK_CAP // bw
            flags = [self._fine(r, n * 4) for r in (0, 1)]
            lflags = [self._fine(r, n * 4) for r in (0, 1)]
            epoch = [torch.zeros(n, dtype=torch.int32, device=d) for d in self.devs]
            lepoch = [torch.zeros(n, dtype=torch.int32, device=d) for d in self.devs]
            self.state[bw] = (flags, lflags, epoch, lepoch)
        return self.state[bw]

    def set_epochs(self, bw: int, value: int) -> None:
        flags, _, epoch, _ = self.var(bw)
        n = BULK_CAP // bw
        hip = self.ll.hip
        for r in (0, 1):
            epoch[r].fill_(value)
            host = (ctypes.c_int32 * n)(*([value] * n))
            hip.hipSetDevice(self.devs[r].index)
            rc = hip.hipMemcpy(ctypes.c_void_p(flags[r].ptr), host, ctypes.c_size_t(n * 4), 1)
            if rc != 0:
                raise RuntimeError(f"hipMemcpy flags rc={rc}")
            torch.cuda.synchronize(self.devs[r])

    def launch(self, rank: int, x, residual, bw: int, warps: int, loop: bool = False):
        x = x.contiguous()
        nw = x.numel() // 2
        assert x.dtype == torch.bfloat16 and nw <= BULK_CAP
        out = torch.empty_like(x)
        res = residual if residual is not None else x
        flags, lflags, epoch, lepoch = self.var(bw)
        if loop:
            rl = rp = self.loop[rank]
            fl = fp = lflags[rank]
            ep = lepoch[rank]
        else:
            rl, rp = self.recv[rank], self.recv[1 - rank]
            fl, fp = flags[rank], flags[1 - rank]
            ep = epoch[rank]
        with torch.cuda.device(x.device):
            k = _bulk_ar_kernel[(triton.cdiv(nw, bw),)](
                x.view(-1).view(torch.int32), res.view(-1).view(torch.int32),
                out.view(-1).view(torch.int32), rl, rp, fl, fp, ep, self.err[rank], nw, BULK_CAP,
                BW=bw, RES=residual is not None, MAXIT=MAXIT, num_warps=warps)
        return out, k

    def timeouts(self) -> int:
        return sum(int(e.item()) for e in self.err)


class PeerCopy:
    """The in-graph peer copy variant: captured hipMemcpyAsync push + flag/sum kernel."""

    def __init__(self, bulk: Bulk, bw: int = 4096) -> None:
        self.b = bulk
        self.bw = bw
        n = BULK_CAP // bw
        self.flag = [bulk._fine(r, 4) for r in (0, 1)]
        self.epoch = [torch.zeros(n, dtype=torch.int32, device=d) for d in bulk.devs]
        self.calls = [0, 0]

    def launch(self, rank: int, x, residual):
        b = self.b
        x = x.contiguous()
        nw = x.numel() // 2
        par = self.calls[rank] % 2
        self.calls[rank] += 1
        dst = b.recv[1 - rank].ptr + par * BULK_CAP * 4
        stream = torch.cuda.current_stream(x.device).cuda_stream
        rc = b.ll.hip.hipMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(x.data_ptr()),
                                     ctypes.c_size_t(nw * 4), 3, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"hipMemcpyAsync rc={rc}")
        out = torch.empty_like(x)
        res = residual if residual is not None else x
        local = tp._Raw(b.recv[rank].ptr + par * BULK_CAP * 4, torch.int32)
        with torch.cuda.device(x.device):
            _pc_sum_kernel[(triton.cdiv(nw, self.bw),)](
                x.view(-1).view(torch.int32), res.view(-1).view(torch.int32),
                out.view(-1).view(torch.int32), local, self.flag[rank], self.flag[1 - rank],
                self.epoch[rank], b.err[rank], nw, BW=self.bw, RES=residual is not None,
                MAXIT=MAXIT, num_warps=8)
        return out


# ============================================================================== helpers


def capture(devs, body):
    """One graph per rank: ``body(rank)`` captured on a side stream of that rank's GPU."""
    graphs = []
    for r in (0, 1):
        with torch.cuda.device(devs[r]):
            s = torch.cuda.Stream(devs[r])
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, stream=s):
                    body(r)
            torch.cuda.synchronize(devs[r])
        graphs.append(g)
    return graphs


def replay_time(devs, graphs, reps: int) -> float:
    """Median over ``reps`` of the slower rank's GPU time of one replay of both graphs (ms)."""
    times = []
    for i in range(reps + 2):
        evs = []
        for r in (0, 1):
            with torch.cuda.device(devs[r]):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[r].replay()
                b.record()
                evs.append((a, b))
        for r in (0, 1):
            torch.cuda.synchronize(devs[r])
        if i >= 2:
            times.append(max(a.elapsed_time(b) for a, b in evs))
    return statistics.median(times)


def isa_excerpt(k, label: str) -> None:
    try:
        asm = k.asm["amdgcn"]
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT isa {label} unavailable: {exc}")
        return
    lines = [ln.strip() for ln in asm.splitlines()]
    idx = [i for i, ln in enumerate(lines) if "atomic_swap" in ln]
    print(f"RESULT isa {label} vgprs={next((ln for ln in lines if '.vgpr_count' in ln), '?')} "
          f"spill={next((ln for ln in lines if 'vgpr_spill' in ln), '?')}")
    for i in idx[:1]:
        lo = max(0, i - 25)
        body = [ln for ln in lines[lo:i + 3] if ln and not ln.startswith(";") and not ln.startswith(".")]
        print(f"RESULT isa {label} before the flag swap: " + " | ".join(body))
    if idx:
        first = next((i for i, ln in enumerate(lines) if ln.startswith("global_load")), 0)
        seg = [ln for ln in lines[first:idx[0]] if ln.startswith(("global_", "buffer_", "s_waitcnt", "s_barrier"))]
        print(f"RESULT isa {label} memory ops from the first load to the flag swap: " + " | ".join(seg)[:3000])
    stores = sum(1 for ln in lines if ln.startswith("global_store_dwordx4") or ln.startswith("buffer_store_dwordx4"))
    sstore = sum(1 for ln in lines if ln.startswith("global_store_dword ") or ln.startswith("buffer_store_dword "))
    print(f"RESULT isa {label} dwordx4 stores={stores} dword stores={sstore}")


def child(args) -> int:
    devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    for d in devs:
        torch.zeros(1, device=d)
    print(f"RESULT env torch={torch.__version__} triton={triton.__version__} "
          f"dev={torch.cuda.get_device_properties(0).name} cus={torch.cuda.get_device_properties(0).multi_processor_count}",
          flush=True)
    ll = tp._LL(devs)
    bulk = Bulk(ll)
    t_start = time.time()

    # ------------------------------------------------ (c0) loopback mode, compile + ISA
    for bw, warps in VARIANTS:
        for residual in (False, True):
            r = 0
            x = torch.randn(576, HID, device=devs[r]).bfloat16()
            res = torch.randn(576, HID, device=devs[r]).bfloat16() if residual else None
            out, k = bulk.launch(r, x, res, bw, warps, loop=True)
            torch.cuda.synchronize(devs[r])
            ref = (x.float() + x.float()).bfloat16()
            if residual:
                ref = (res.float() + ref.float()).bfloat16()
            ok = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            print(f"RESULT loopback bw={bw} warps={warps} res={int(residual)} bitwise={ok}", flush=True)
            if not ok:
                print("RESULT KILL loopback mismatch")
        isa_excerpt(k, f"bw={bw} warps={warps}")
    for r in (0, 1):
        for bw, warps in VARIANTS:
            for residual in (False, True):
                x = torch.zeros(64, HID, device=devs[r], dtype=torch.bfloat16)
                bulk.launch(r, x, x if residual else None, bw, warps, loop=True)
        torch.cuda.synchronize(devs[r])
    # compile the store-only and peer-copy kernels eagerly (never inside a capture): each rank
    # against its own loopback buffer and a scratch flag of its own
    for r in (0, 1):
        x = torch.zeros(64, HID, device=devs[r], dtype=torch.bfloat16)
        nw = x.numel() // 2
        for bw, warps in ((4096, 1), (8192, 8), (2048, 1)):
            with torch.cuda.device(devs[r]):
                _store_only_kernel[(triton.cdiv(nw, bw),)](x.view(-1).view(torch.int32), bulk.loop[r], nw,
                                                           BW=bw, num_warps=warps)
        flag = bulk._fine(r, 4)
        ep = torch.zeros(BULK_CAP // 4096, dtype=torch.int32, device=devs[r])
        for residual in (False, True):
            out = torch.empty_like(x)
            with torch.cuda.device(devs[r]):
                _pc_sum_kernel[(triton.cdiv(nw, 4096),)](
                    x.view(-1).view(torch.int32), x.view(-1).view(torch.int32), out.view(-1).view(torch.int32),
                    bulk.loop[r], flag, flag, ep, bulk.err[r], nw, BW=4096, RES=residual, MAXIT=MAXIT,
                    num_warps=8)
        torch.cuda.synchronize(devs[r])
    print(f"RESULT loopback timeouts={bulk.timeouts()} s={time.time() - t_start:.0f}", flush=True)

    # ------------------------------------------------ (a) per-call time
    best1792 = None
    table = {}
    for rows in ROWS:
        for residual in (False, True):
            xs = [torch.randn(rows, HID, device=d).bfloat16() for d in devs]
            rs = [torch.randn(rows, HID, device=d).bfloat16() for d in devs] if residual else [None, None]
            outs = {}
            # LL
            last = {}

            def body_ll(r):
                for _ in range(CALLS):
                    last[r] = ll._launch(r, xs[r], rs[r])

            g = capture(devs, body_ll)
            ms = replay_time(devs, g, args.reps)
            outs["ll"] = [last[0].clone(), last[1].clone()]
            table[(rows, residual, "ll")] = ms
            print(f"RESULT time rows={rows} res={int(residual)} ll us_per_call={ms * 1000 / CALLS:.1f}", flush=True)
            del g
            for bw, warps in VARIANTS:
                lastb = {}

                def body_b(r, bw=bw, warps=warps):
                    for _ in range(CALLS):
                        lastb[r] = bulk.launch(r, xs[r], rs[r], bw, warps)[0]

                g = capture(devs, body_b)
                ms = replay_time(devs, g, args.reps)
                eq = all(torch.equal(lastb[r].view(torch.int16), outs["ll"][r].view(torch.int16)) for r in (0, 1))
                us = ms * 1000 / CALLS
                table[(rows, residual, (bw, warps))] = ms
                print(f"RESULT time rows={rows} res={int(residual)} bulk bw={bw} warps={warps} "
                      f"us_per_call={us:.1f} GB/s_payload={rows * HID * 2 / (us * 1e3):.1f} "
                      f"bitwise_vs_ll={eq}", flush=True)
                if not eq:
                    print("RESULT KILL bulk != ll")
                if rows == 1792 and (best1792 is None or us < best1792[0]):
                    best1792 = (us, bw, warps)
                del g
            ref0 = (xs[0].float() + xs[1].to(devs[0]).float()).bfloat16()
            ref = [ref0, ref0.to(devs[1])]
            if residual:
                ref = [(rs[r].float() + ref[r].float()).bfloat16() for r in (0, 1)]
            okref = all(torch.equal(outs["ll"][r].view(torch.int16), ref[r].view(torch.int16)) for r in (0, 1))
            print(f"RESULT ll_vs_torch rows={rows} res={int(residual)} bitwise={okref}", flush=True)
            torch.cuda.empty_cache()
    print(f"RESULT timeouts after (a) ll={ll.timeouts()} bulk={bulk.timeouts()}", flush=True)

    # ------------------------------------------------ (b) peer write bandwidth, peer copy
    rows = 1792
    nbytes = rows * HID * 2
    xs = [torch.randn(rows, HID, device=d).bfloat16() for d in devs]
    for bw, warps in ((4096, 1), (8192, 8), (2048, 1)):
        def body_s(r, bw=bw, warps=warps):
            nw = rows * HID // 2
            with torch.cuda.device(devs[r]):
                for i in range(CALLS):
                    dst = tp._Raw(bulk.recv[1 - r].ptr + (i % 2) * BULK_CAP * 4, torch.int32)
                    _store_only_kernel[(triton.cdiv(nw, bw),)](xs[r].view(-1).view(torch.int32), dst, nw,
                                                               BW=bw, num_warps=warps)

        g = capture(devs, body_s)
        ms = replay_time(devs, g, args.reps)
        us = ms * 1000 / CALLS
        print(f"RESULT peer_write store_kernel bw={bw} warps={warps} rows={rows} us={us:.1f} "
              f"GB/s_each_direction={nbytes / (us * 1e3):.1f}", flush=True)
        del g

    def body_c(r):
        stream = torch.cuda.current_stream(devs[r]).cuda_stream
        for i in range(CALLS):
            dst = bulk.recv[1 - r].ptr + (i % 2) * BULK_CAP * 4
            rc = ll.hip.hipMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(xs[r].data_ptr()),
                                       ctypes.c_size_t(nbytes), 3, ctypes.c_void_p(stream))
            assert rc == 0, rc

    g = capture(devs, body_c)
    ms = replay_time(devs, g, args.reps)
    us = ms * 1000 / CALLS
    print(f"RESULT peer_write copy_engine rows={rows} us={us:.1f} GB/s_each_direction={nbytes / (us * 1e3):.1f}",
          flush=True)
    del g
    pc = PeerCopy(bulk)
    for residual in (False, True):
        rs = [torch.randn(rows, HID, device=d).bfloat16() for d in devs] if residual else [None, None]
        lastp, lastl = {}, {}

        def body_p(r):
            for _ in range(CALLS):
                lastp[r] = pc.launch(r, xs[r], rs[r])
            lastl[r] = ll._launch(r, xs[r], rs[r])

        pc.calls = [0, 0]
        g = capture(devs, body_p)
        ms = replay_time(devs, g, args.reps)
        eq = all(torch.equal(lastp[r].view(torch.int16), lastl[r].view(torch.int16)) for r in (0, 1))
        ll_ms = table[(rows, residual, "ll")]
        us = (ms - ll_ms / CALLS) * 1000 / CALLS  # one LL call per replay: subtract it
        print(f"RESULT time rows={rows} res={int(residual)} peer_copy us_per_call={us:.1f} bitwise_vs_ll={eq}",
              flush=True)
        if not eq:
            print("RESULT KILL peer_copy != ll")
        del g
    print(f"RESULT timeouts after (b) ll={ll.timeouts()} bulk={bulk.timeouts()}", flush=True)

    # ------------------------------------------------ (c) 2,000-replay bitwise test
    sizes = ((576, False), (1792, True), (2048, False), (1792, False), (576, True), (1024, True))
    total_bad = 0
    for bw, warps in VARIANTS:
        bulk.set_epochs(bw, START)
        torch.cuda.synchronize(devs[0])
        bad = [torch.zeros(4, dtype=torch.int64, device=d) for d in devs]
        for r in (0, 1):
            torch.cuda.manual_seed_all(1234)
        bufs = {}
        for r in (0, 1):
            bufs[r] = {n: (torch.empty(n, HID, dtype=torch.bfloat16, device=devs[r]),
                           torch.empty(n, HID, dtype=torch.bfloat16, device=devs[r]),
                           torch.empty(n, HID, dtype=torch.bfloat16, device=devs[r])) for n in (576, 1024, 1792, 2048)}

        def body_t(r, bw=bw, warps=warps):
            for n, residual in sizes:
                a, b, c = bufs[r][n]
                a.normal_()  # rank 0's partial on both GPUs (same seed, same sequence)
                b.normal_()  # rank 1's
                c.normal_()  # residual
                x = a if r == 0 else b
                res = c if residual else None
                ob = bulk.launch(r, x, res, bw, warps)[0]
                ol = ll._launch(r, x, res)
                ref = (a.float() + b.float()).bfloat16()
                if residual:
                    ref = (c.float() + ref.float()).bfloat16()
                bad[r][0] += (ob.view(torch.int16) != ol.view(torch.int16)).sum()
                bad[r][1] += (ob.view(torch.int16) != ref.view(torch.int16)).sum()
                bad[r][2] += (ol.view(torch.int16) != ref.view(torch.int16)).sum()

        g = capture(devs, body_t)
        e0 = bulk.timeouts()
        t0 = time.time()
        for i in range(args.replays):
            for r in (0, 1):
                g[r].replay()
            if i % 100 == 99:
                for r in (0, 1):
                    torch.cuda.synchronize(devs[r])
        for r in (0, 1):
            torch.cuda.synchronize(devs[r])
        b = [t.cpu().tolist() for t in bad]
        flags, _, epoch, _ = bulk.var(bw)
        ep = int(epoch[0][0].item())
        to = bulk.timeouts() - e0
        print(f"RESULT bitwise bw={bw} warps={warps} replays={args.replays} calls={args.replays * len(sizes)} "
              f"bulk_vs_ll_mismatch={b[0][0] + b[1][0]} bulk_vs_torch={b[0][1] + b[1][1]} "
              f"ll_vs_torch={b[0][2] + b[1][2]} timeouts={to} epoch_cta0_end={ep} (start {START}, "
              f"wrapped={ep < 0}) s={time.time() - t0:.1f}", flush=True)
        total_bad += b[0][0] + b[1][0] + to
        del g
        torch.cuda.empty_cache()

    # ------------------------------------------------ verdict
    print(f"RESULT ll timeouts total={ll.timeouts()}")
    if best1792:
        us, bw, warps = best1792
        llus = table[(1792, False, "ll")] * 1000 / CALLS
        print(f"RESULT best_1792 bulk bw={bw} warps={warps} us_per_call={us:.1f} (ll {llus:.1f}) "
              f"saving_per_pass_ms={(llus - us) * CALLS / 1000:.2f}")
        verdict = "PASS" if us <= 100 and total_bad == 0 else "KILL"
        print(f"RESULT verdict {verdict} (kill if > 100 us at 1792 rows or any mismatch/timeout; "
              f"mismatches+timeouts={total_bad}) total_s={time.time() - t_start:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
