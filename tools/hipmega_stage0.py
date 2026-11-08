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


# ============================================================================== (b, c) slice

NIN, H, NMOE, MAXL = 6176, 2048, 769, 4


def _structs():
    class LayerW(ctypes.Structure):
        _fields_ = [(n, ctypes.c_void_p) for n in (
            "ln_in", "ln_post", "w_in", "conv_w", "w_out", "w_moe", "gate_up", "down", "sh_down",
            "conv_state", "rec_state", "A_log", "dt_bias", "norm_w")] + [
            ("conv_stride", ctypes.c_long), ("rec_stride", ctypes.c_long),
            ("eps", ctypes.c_float), ("gn_eps", ctypes.c_float)]

    class Args(ctypes.Structure):
        _fields_ = [("L", LayerW * MAXL)] + [(n, ctypes.c_void_p) for n in (
            "x_in", "x_out", "states", "proj", "core", "rbuf", "gu", "topk_idx", "topk_w",
            "part_out", "recv", "peer", "bar", "err")] + [("rank", ctypes.c_int), ("nl", ctypes.c_int)]

    return LayerW, Args


class RankSlice:
    """One rank's HIP slice: weights of ``layers`` (views of the loaded shard), buffers, args."""

    def __init__(self, rtc, code, m, rank: int, layers, states, x_in, recv_ptrs, nl: int) -> None:
        import torch

        LayerW, Args = _structs()
        dev = m.embed.device
        self.dev, self.m, self.rank = dev, m, rank
        self.mod = rtc.Module(code, dev)
        self.keep = []
        fd = m.fused
        a = Args()
        for slot, i in enumerate(layers):
            layer = m.layers[i]
            lin = layer.linear_attn
            mlp = layer.mlp
            conv, rec = m.gdn_state[i]
            assert m.layer_types[i] == "linear_attention"
            assert tuple(fd.w_in[i].shape) == (NIN, H) and tuple(fd.w_moe[i].shape) == (NMOE, H)
            assert conv.stride(1) == 4 and conv.stride(2) == 1 and rec.dtype == torch.float32
            assert tuple(mlp.experts.gate_up_proj.shape) == (256, 512, H), mlp.experts.gate_up_proj.shape
            assert tuple(mlp.experts.down_proj.shape) == (256, H, 256), mlp.experts.down_proj.shape
            ts = dict(
                ln_in=layer.input_layernorm.weight.contiguous(),
                ln_post=layer.post_attention_layernorm.weight.contiguous(),
                w_in=fd.w_in[i], conv_w=lin.conv1d.weight.contiguous(),
                w_out=lin.out_proj.weight.contiguous(), w_moe=fd.w_moe[i],
                gate_up=mlp.experts.gate_up_proj.contiguous(), down=mlp.experts.down_proj.contiguous(),
                sh_down=mlp.shared_expert.down_proj.weight.contiguous(),
                conv_state=conv, rec_state=rec,
                A_log=lin.A_log.detach().float().contiguous(), dt_bias=lin.dt_bias.detach().float().contiguous(),
                norm_w=lin.norm.weight.detach().float().contiguous(),
            )
            for k, t in ts.items():
                assert t.device == dev, (k, t.device, dev)
                if k not in ("conv_state", "rec_state", "A_log", "dt_bias", "norm_w"):
                    assert t.dtype == torch.bfloat16, (k, t.dtype)
                setattr(a.L[slot], k, t.data_ptr())
                self.keep.append(t)
            a.L[slot].conv_stride = conv.stride(0)
            a.L[slot].rec_stride = rec.stride(0)
            a.L[slot].eps = float(m.eps)
            a.L[slot].gn_eps = float(getattr(lin.norm, "variance_epsilon", getattr(lin.norm, "eps", 1e-6)))
        bf = torch.bfloat16
        z = lambda *shape, dtype=bf: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731
        self.x_in = x_in
        self.x_out = z(8, H)
        self.states = states
        self.proj, self.core = z(8, NIN), z(8, H)
        self.rbuf, self.gu = z(MAXL, 8, NMOE), z(64, 512)
        self.topk_idx = z(MAXL, 8, 8, dtype=torch.int32)
        self.topk_w = z(MAXL, 8, 8, dtype=torch.float32)
        self.part_out = z(8, H)
        self.bar = z(4, dtype=torch.int32)
        self.err = z(4, dtype=torch.int32)
        for k in ("x_in", "x_out", "states", "proj", "core", "rbuf", "gu", "topk_idx", "topk_w",
                  "part_out", "bar", "err"):
            setattr(a, k, getattr(self, k).data_ptr())
        a.recv, a.peer = recv_ptrs[rank], recv_ptrs[1 - rank]
        a.rank, a.nl = rank, nl
        self.args_host = a
        raw = bytes(a)
        self.args = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev)
        self.slice_fn = self.mod.fn("hm_slice8")
        self.cus = torch.cuda.get_device_properties(dev).multi_processor_count
        occ = self.slice_fn.occupancy(512)
        if occ < 1:
            raise RuntimeError(f"hm_slice8 cannot be resident at 512 threads (occupancy {occ})")
        self.grid = self.cus  # one block per CU, all resident
        self.bar_base = 0

    def set_nl(self, nl: int) -> None:
        import torch

        self.args_host.nl = nl
        self.args.copy_(torch.frombuffer(bytearray(bytes(self.args_host)), dtype=torch.uint8))
        torch.cuda.synchronize(self.dev)

    def launch(self, ep_base: int, nbar: int) -> None:
        import torch

        with torch.cuda.device(self.dev):
            self.slice_fn.launch(self.grid, 512, [self.args, ctypes.c_uint(self.bar_base & 0xFFFFFFFF),
                                                  ctypes.c_uint(ep_base & 0xFFFFFFFF)])
        self.bar_base += nbar * self.grid

    def probe(self, which: int, rows: int, reps: int, nl: int) -> None:
        import torch

        f = self.mod.fn(f"hm_probe{rows}")
        with torch.cuda.device(self.dev):
            f.launch(self.grid, 512, [self.args, ctypes.c_int(which), ctypes.c_int(reps),
                                      ctypes.c_uint(self.bar_base & 0xFFFFFFFF)])
        self.bar_base += reps * nl * self.grid


def slice_main(args) -> None:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")
    import torch

    from engine.fast import hipmega_rtc as rtc
    from engine.fast import mfma
    from engine.generator import CHECKPOINT, Generator

    layers = [int(v) for v in args.layers.split(",")]
    nl = len(layers)

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    # ---------------------------------------------------------------- compile first (cheap)
    t0 = time.perf_counter()
    code, log = rtc.compile_hip_subprocess(str(HIPDIR / "slice.hip"))
    out("slice_compile", seconds=round(time.perf_counter() - t0, 2), resources=rtc.resources(log))

    # ---------------------------------------------------------------- load as the server does
    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    tok = g.tokenizer
    assert hasattr(fast, "ranks"), "needs the TP2 engine"
    ranks = fast.ranks
    out("load", seconds=round(time.perf_counter() - started), chain=mfma.CHAIN, kmax=mfma.KMAX,
        layer_types=[ranks[0].layer_types[i] for i in layers])

    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    corpus = pool(9973 * 2 + 4000)

    def prefill(slot: int, prompt: list[int]):
        fast.reset(slot)
        logits = None
        for begin, end, _ in g._chunks(tuple(prompt), 0):
            follow = prompt[end] if end < len(prompt) else None
            logits = fast.prefill(slot, prompt[begin:end], begin, follow)
        return logits

    slots = [0, 1]
    plen = 1800
    prompts = [corpus[9973 * k : 9973 * k + plen] for k in range(2)]
    lasts = [int(torch.argmax(prefill(s, p))) for s, p in zip(slots, prompts, strict=True)]
    # real drafts: the next prompt tokens are a plausible chain; any token ids work for timing
    drafts = [[int(t) for t in corpus[9973 * b + plen : 9973 * b + plen + 3]] for b in range(2)]

    # ---------------------------------------------------------------- today's full step (context)
    with torch.inference_mode():
        fn = lambda: fast.verify_chain(lasts, drafts, slots, [plen] * 2)  # noqa: E731
        for _ in range(3):
            fn()
        times = []
        for _ in range(args.iters):
            sync()
            t1 = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - t1) * 1e3)
        step_ms = statistics.median(times)
        out("today_step", k=3, slots=2, rows=8, verify_chain_ms=round(step_ms, 3),
            per_layer_share_ms=round(step_ms * nl / 40, 3))

    # ---------------------------------------------------------------- the step's inputs, as the engine has them
    groups = 4
    with torch.inference_mode():
        host = fast._chain_host(lasts, drafts, slots, [plen] * 2, groups, 2)
        fast._load_inputs(host)
        sync()
        metas = [m.devices[m.embed.device] for m in ranks]
        tokens = [mt.tokens[:8].clone() for mt in metas]
        x_in = [torch.nn.functional.embedding(tokens[r], ranks[r].embed).contiguous() for r in range(2)]
        states = [mt.gdn[:8].clone() for mt in metas]
        out("inputs", tokens=tokens[0].tolist(), state_rows=states[0].tolist(),
            slots=metas[0].slots[:8].tolist(), positions=metas[0].positions[:8].tolist())

    # state snapshot (the rows the step reads and writes)
    def snap():
        res = []
        for r, m in enumerate(ranks):
            rows = states[r]
            res.append({i: (m.gdn_state[i][0].index_select(0, rows).clone(),
                            m.gdn_state[i][1].index_select(0, rows).clone()) for i in layers})
        sync()
        return res

    def restore(sn):
        for r, m in enumerate(ranks):
            rows = states[r]
            for i in layers:
                m.gdn_state[i][0].index_copy_(0, rows, sn[r][i][0])
                m.gdn_state[i][1].index_copy_(0, rows, sn[r][i][1])
        sync()

    base = snap()

    # ---------------------------------------------------------------- reference: today's fused kernels, captured
    def ref_layers(m, x):
        meta = m.devices[m.embed.device]
        for i in layers:
            x = m._fused_layer(i, x, meta.slots[:8], meta.positions[:8], meta, meta.gdn[:8], groups)
        return x

    routing_ref = {}

    with torch.inference_mode():
        # eager (both ranks in threads, host all-reduce) with the routing recorded per layer
        for r, m in enumerate(ranks):
            fd = m.fused
            orig = fd._moe

            def rec_moe(i, x, h, fd=fd, orig=orig, r=r):
                if r == 0:
                    rl = mfma.gemv(h.contiguous(), fd.w_moe[i])[:, :256].float()
                    routing_ref[i] = torch.topk(torch.softmax(rl, dim=-1), 8, dim=-1).indices.sort(-1).values.cpu()
                return orig(i, x, h)

            fd._moe = rec_moe
        restore(base)
        eager = fast._both(lambda m: ref_layers(m, x_in[m.rank]).clone())
        for m in ranks:
            del m.fused._moe
        sync()
        eager = [e.clone() for e in eager]
        restore(base)
        # capture one graph per rank (rank 1 on the worker thread)
        graphs, gouts = [None, None], [None, None]
        static_in = [x.clone() for x in x_in]

        def cap(m):
            dev = m.embed.device
            s = torch.cuda.Stream(dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            torch.cuda.synchronize(dev)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s, capture_error_mode=fast.capture_mode or "thread_local"):
                o = ref_layers(m, static_in[m.rank])
            torch.cuda.synchronize(dev)
            return gr, o

        with torch.cuda.device(ranks[0].embed.device):
            graphs[0], gouts[0] = cap(ranks[0])
        graphs[1], gouts[1] = fast._pool.submit(fast._on_rank1, lambda: cap(ranks[1])).result()
        sync()

        def ref_replay():
            for r in (0, 1):
                with torch.cuda.device(ranks[r].embed.device):
                    graphs[r].replay()

        restore(base)
        ref_replay()
        sync()
        ref_out = [o.clone() for o in gouts]
        ref_state = snap()
        out("ref_check", graph_eq_eager=[bool(torch.equal(ref_out[r], eager[r])) for r in (0, 1)],
            ranks_equal=bool(torch.equal(ref_out[0].cpu(), ref_out[1].cpu())),
            timeouts=fast.timeouts())

    # ---------------------------------------------------------------- timing: 3 layers, graph vs HIP
    def timed(fn, iters):
        restore(base)
        for _ in range(3):
            fn()
        sync()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        with torch.cuda.device(ranks[0].embed.device):
            ev[0].record()
        t1 = time.perf_counter()
        for _ in range(iters):
            fn()
        with torch.cuda.device(ranks[0].embed.device):
            ev[1].record()
        sync()
        wall = (time.perf_counter() - t1) * 1e3 / iters
        return ev[0].elapsed_time(ev[1]) / iters, wall

    with torch.inference_mode():
        ref_ms, ref_wall = timed(ref_replay, args.iters)
    out("ref_time", layers=layers, rows=8, ref_graph_ms=round(ref_ms, 4), ref_wall_ms=round(ref_wall, 4))

    # ---------------------------------------------------------------- the HIP slice
    out("step", at="enable_peer")
    enable_peer(rtc, ranks[0].embed.device.index, ranks[1].embed.device.index)
    recv_ptrs = [fine_alloc(rtc, m.embed.device.index, 2 * 2 * 8192 * 8) for m in ranks]
    torch.cuda.set_device(ranks[0].embed.device)
    out("step", at="rank_slices", recv=[hex(p) for p in recv_ptrs])
    rs = [RankSlice(rtc, code, ranks[r], r, layers, states[r], x_in[r], recv_ptrs, nl) for r in (0, 1)]
    out("slice_setup", grid=rs[0].grid, occupancy=[r.slice_fn.occupancy(512) for r in rs],
        regs=rs[0].slice_fn.attr(4))
    torch.cuda.set_device(ranks[0].embed.device)
    torch.cuda.set_device(ranks[0].embed.device)
    nbar = 6 * nl
    cur_nl = [nl]
    ep_next = [1]  # all-reduce call epochs only grow (two calls per layer), never reused

    def hip_run():
        for r in (1, 0):
            rs[r].launch(ep_next[0], nbar)
        ep_next[0] += 2 * cur_nl[0]

    restore(base)
    out("step", at="first_launch")
    hip_run()
    sync()
    errs = [rs[r].err.tolist() for r in (0, 1)]
    hip_out = [rs[r].x_out.clone() for r in (0, 1)]
    hip_state = snap()
    out("slice_run", errors=errs, barriers_expected=nbar)

    def cmp(a, b):
        a, b = a.float(), b.float()
        d = (a - b).abs()
        return dict(max_abs=float(d.max()), rel_l2=float(d.norm() / (b.norm() + 1e-30)),
                    equal_frac=float((a == b).float().mean()), ref_absmax=float(b.abs().max()))

    out("parity_x", ranks_bitwise_equal=bool(torch.equal(hip_out[0].cpu(), hip_out[1].cpu())),
        vs_ref=cmp(hip_out[0], ref_out[0]))
    for i_pos, i in enumerate(layers):
        conv_c = cmp(hip_state[0][i][0], ref_state[0][i][0])
        rec_c = cmp(hip_state[0][i][1], ref_state[0][i][1])
        hk = rs[0].topk_idx[i_pos].sort(-1).values.cpu()
        rk = routing_ref.get(i)
        same = None if rk is None else int((hk == rk).all(-1).sum())
        nu = len(set(rs[0].topk_idx[i_pos].flatten().tolist()))
        out("parity_layer", layer=i, conv_state=conv_c, rec_state=rec_c, routing_rows_identical=same,
            rows=8, unique_experts=nu)

    hip_ms, hip_wall = timed(hip_run, args.iters)
    out("slice_time", layers=layers, rows=8, ref_graph_ms=round(ref_ms, 4), ref_wall_ms=round(ref_wall, 4),
        hip_ms=round(hip_ms, 4), hip_wall_ms=round(hip_wall, 4), ratio=round(hip_ms / ref_ms, 3),
        share_of_step_ms=round(step_ms * nl / 40, 4), ratio_vs_share=round(hip_ms / (step_ms * nl / 40), 3),
        errors=[rs[r].err.tolist() for r in (0, 1)], ll_timeouts_engine=fast.timeouts())
    # per-layer marginal: the same kernel over 1 layer
    for r in (0, 1):
        rs[r].set_nl(1)
    nbar, cur_nl[0] = 6, 1
    hip1_ms, _ = timed(hip_run, args.iters)
    for r in (0, 1):
        rs[r].set_nl(nl)
    nbar, cur_nl[0] = 6 * nl, nl
    out("slice_time_1layer", hip_ms=round(hip1_ms, 4), per_extra_layer_ms=round((hip_ms - hip1_ms) / max(1, nl - 1), 4))
    restore(base)
    hip_run()
    sync()

    # ---------------------------------------------------------------- (b) per-phase bandwidth, rank 0 alone
    r0 = rs[0]
    topk = r0.topk_idx[:nl].cpu()
    names = {0: "in_proj", 1: "out_proj", 2: "router_shared", 3: "moe_up", 4: "moe_down"}
    for rows in (4, 8):
        for which in range(5):
            if which == 0:
                per_layer = [NIN * H * 2] * nl
            elif which == 1:
                per_layer = [H * H * 2] * nl
            elif which == 2:
                per_layer = [NMOE * H * 2] * nl
            else:
                nus = [len(set(topk[l, :rows].flatten().tolist())) for l in range(nl)]
                per_layer = [n * 512 * H * 2 for n in nus] if which == 3 else [n * H * 256 * 2 + H * 256 * 2 for n in nus]
            reps = 10
            dev = r0.dev
            r0.probe(which, rows, 1, nl)
            torch.cuda.synchronize(dev)
            with torch.cuda.device(dev):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                r0.probe(which, rows, reps, nl)
                e1.record()
            torch.cuda.synchronize(dev)
            ms = e0.elapsed_time(e1)
            nbytes = sum(per_layer) * reps
            out("phase_bw", phase=names[which], rows=rows, us_per_layer=round(ms * 1e3 / (reps * nl), 2),
                mb_per_layer=round(sum(per_layer) / nl / 1e6, 2), gbps=round(nbytes / (ms * 1e-3) / 1e9, 1),
                errors=r0.err.tolist())
    restore(base)


# ============================================================================== main


def main() -> None:
    import faulthandler

    faulthandler.enable()
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
            slice_main(args)
    except Exception:  # noqa: BLE001
        out("error", part=args.part, trace=traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
