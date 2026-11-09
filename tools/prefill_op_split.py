"""Plan 26 step 0: per-GCD prefill GPU time by call site (which eager work could be fused).

A captured (plan 33 prefill graph) pass runs no Python per op, so its kernels are attributed
by kernel name only; ``all --eager`` (ENGINE_PREFILL_GRAPH=0) gives the per-call-site split.
Both print the GDN chunk + prep total that plan 40 part B's kill rule reads.

    python tools/prefill_op_split.py all [--devices 4,5] [--news 61,561,1650] [--reps 5]
    python tools/prefill_op_split.py passes [--news ...] [--reps 5]   # one process, visible GPUs

Drives prefill through the server's own path (``PackedPrefill.add`` then ``.step``, the
loaded ``Generator`` with warmup and graphs) as ``tools/prefill_host_probe.py`` does: a
3,600-token context of synthetic ids is prefilled, retired as a session tail, then resumed
with N new tokens; the resumed pass is the measured one. ENGINE_TP is left at the head's
default unless set.

Attribution. While a pass is profiled, the engine's prefill call sites are wrapped (nothing
else changes; the wrappers only log host timestamps and open ``record_function`` ranges):
``FastModel`` methods (``prefill``, ``_gdn_prefill``, ``_gdn_project``, ``_gdn_heads``,
``_gdn_out``, ``_gdn_gate``, ``_attn_prefill``, ``_qkv``, ``_moe``, ``_reduce``,
``_mtp_prefill``, ``_mtp_in``, ``_mtp_layer``, ``_mtp_logits``), the module helpers
``_rms``, ``_rope``, ``prefill_conv``, ``F.scaled_dot_product_attention``, the chunked
delta rule (``hf.torch_chunk_gated_delta_rule``), ``moe_prefill.moe_prefill`` and
``tp.Group._exchange``. Each wrapper logs (thread, rank, label path, start, end) with
``time.perf_counter_ns``. The torch profiler only records CPU ops of the calling thread,
while rank 1 runs on a worker thread, so the log, not the profiler's CPU ops, is the source
of ranges: its clock is aligned to the trace's by the main thread's ``record_function``
twins of the same wrappers (median offset; the spread is printed). Every GPU kernel/copy
is tied to its launching runtime call by ``correlation`` id, and charged to the innermost
logged range of the launching thread (by trace tid when it matches a logged thread, else by
the kernel's device -> rank) that contains the launch time. Within each range the kernels
are classed by name (weight GEMM, MoE expert kernel, fla chunk kernel, conv, attention
fallback matmul, other eager), giving the per-op table in ms per GCD.

Also per case: synchronized wall (median of ``--reps`` unprofiled passes), per-GCD kernel
sum and busy time, idle, launches, and an unprofiled "host log" pass with the wrappers on:
host ms per range per rank thread, time blocked in the all-reduce rendezvous, which says
what sets the floor at small N.

Kill rule (plan 26): the parts the plan would fuse (attention fallback, norms/residual,
RoPE/gates/qkv elementwise, GDN prep, router, MoE tail) < 35 ms per GCD at 561 tokens ->
park.
"""

from __future__ import annotations

import argparse
import bisect
import functools
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from prefill_host_probe import analyze_trace, result  # noqa: E402

CONTEXT = 3600
NEWS = (61, 561, 1650)
KILL_MS = 35.0


# ============================================================================ kernel classes


def kclass(name: str) -> str:
    n = name.lower()
    if "_gate_up_kernel" in n or "_down_kernel" in n:
        return "moe_kernel"
    if "ll_allreduce" in n:
        return "allreduce_kernel"
    if any(s in n for s in ("miopen", "conv", "naive_conv")):
        return "conv"
    if any(
        s in n
        for s in (
            "chunk_", "gated_delta", "fused_recurrent", "l2norm", "solve_tril", "wy_", "kkt",
            "cumsum_", "recompute_w_u", "merge_16x16", "fwd_o", "fwd_h", "prepare_",
        )
    ):
        return "fla"
    if any(s in n for s in ("cijk", "gemm", "gemv", "hipblaslt", "rocblas", "tensile")):
        return "gemm"
    if "softmax" in n:
        return "softmax"
    if "topk" in n or "radix" in n or "bitonic" in n:
        return "topk"
    if "reduce_kernel" in n:
        return "reduce"
    if "cat" in n and "batched" in n:
        return "cat"
    if "copy" in n or "memcpy" in n:
        return "copy"
    if "index" in n or "gather" in n or "scatter" in n:
        return "index"
    if "sort" in n or "scan" in n:
        return "sort/scan"
    return "elementwise"


# ============================================================================ wrappers


class Log:
    """Host ranges of the wrapped call sites, per thread, while ``active``."""

    def __init__(self) -> None:
        self.active = False
        self.rf = False  # open record_function twins (profiled passes)
        self.local = threading.local()
        self.entries: list[tuple] = []  # (native tid, rank, path, start_ns, end_ns)
        self.lock = threading.Lock()
        self.restore: list[tuple[object, str, object]] = []

    def stack(self) -> list[str]:
        s = getattr(self.local, "stack", None)
        if s is None:
            s = self.local.stack = []
        return s

    def wrap(self, owner, attr: str, label: str, rank_from_self: bool = False) -> None:
        orig = getattr(owner, attr)
        log = self

        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            if not log.active:
                return orig(*args, **kwargs)
            from torch.profiler import record_function

            st = log.stack()
            if rank_from_self and args:
                r = getattr(args[0], "rank", None)
                if r is not None:
                    log.local.rank = r
            path = (st[-1] + ">" if st else "") + label
            st.append(path)
            a = time.perf_counter_ns()
            try:
                if log.rf:
                    with record_function("ops:" + path):
                        return orig(*args, **kwargs)
                return orig(*args, **kwargs)
            finally:
                b = time.perf_counter_ns()
                st.pop()
                with log.lock:
                    log.entries.append((threading.get_native_id(), getattr(log.local, "rank", -1), path, a, b))

        self.restore.append((owner, attr, orig))
        setattr(owner, attr, wrapper)

    def install(self) -> None:
        import torch.nn.functional as F
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

        from engine import gdn
        from engine.fast import model, moe_prefill, tp

        FM = model.FastModel
        self.wrap(FM, "prefill", "prefill", rank_from_self=True)
        self.wrap(FM, "prefill_packed", "prefill_packed", rank_from_self=True)
        for attr, label in (
            ("_gdn_prefill", "gdn"), ("_gdn_project", "proj"), ("_gdn_heads", "heads"),
            ("_gdn_out", "out"), ("_gdn_gate", "gate_norm"), ("_attn_prefill", "attn"),
            ("_qkv", "qkv"), ("_moe", "moe"), ("_reduce", "reduce"), ("_mtp_prefill", "mtp"),
            ("_mtp_in", "in"), ("_mtp_layer", "layer"), ("_mtp_logits", "logits"),
        ):
            self.wrap(FM, attr, label)
        self.wrap(model, "_rms", "rms")
        self.wrap(model, "_rope", "rope")
        self.wrap(model, "prefill_conv", "conv")
        self.wrap(F, "scaled_dot_product_attention", "sdpa")
        chunk = hf.torch_chunk_gated_delta_rule
        self.wrap(hf, "torch_chunk_gated_delta_rule", "chunk")
        if getattr(gdn, "fla_chunk", None) is chunk:  # keep the packed path's identity test true
            self.restore.append((gdn, "fla_chunk", chunk))
            gdn.fla_chunk = hf.torch_chunk_gated_delta_rule
        self.wrap(moe_prefill, "moe_prefill", "routed")
        try:  # plan 40 part B's fused chunk chain (ENGINE_GDN_CHUNK), where the head has it
            from engine.fast import gdn_chunk
        except ImportError:
            gdn_chunk = None
        if gdn_chunk is not None and hasattr(gdn_chunk, "chunk"):
            self.wrap(gdn_chunk, "chunk", "chunk")
        self.wrap(tp.Group, "_exchange", "exchange")
        # The fused prefill path (plan 26, ENGINE_PREFILL_FUSED), where the head has it.
        for attr, label in (("_prefill_fused", "fused"), ("_reduce_rms", "reduce"), ("_gdn_prefill_fused", "gdn")):
            if hasattr(FM, attr):
                self.wrap(FM, attr, label)
        try:
            from engine.fast import prefill_kernels as pk
        except ImportError:
            pk = None
        if pk is not None:
            for attr, label in (("prefill_attention", "sdpa"), ("route", "route"), ("gdn_conv_", "conv"),
                                ("gdn_gates", "heads"), ("gated_rms", "gate_norm")):
                if hasattr(pk, attr):
                    self.wrap(pk, attr, label)

    def uninstall(self) -> None:
        for owner, attr, orig in reversed(self.restore):
            setattr(owner, attr, orig)
        self.restore.clear()


# ============================================================================ attribution


def bucket(path: str, cls: str) -> str:
    """The plan-26 bucket of a kernel of class ``cls`` launched in range ``path``."""
    parts = path.split(">") if path else []
    leaf = parts[-1] if parts else ""
    in_mtp = "mtp" in parts
    pre = "mtp: " if in_mtp else ""
    if not parts:
        return "outside model prefill (sample/gather/H2D)"
    if leaf == "exchange":
        return "all-reduce exchange/wait (copy)"
    if leaf == "reduce":
        return pre + "residual add + all-reduce sum (eager)"
    if leaf == "rms":
        if "qkv" in parts:
            return pre + "q/k norm"
        if "in" in parts and in_mtp:
            return "mtp: pre-fc norms"
        if "logits" in parts:
            return "mtp: final norm"
        return pre + "norm (layer/final)"
    if leaf == "rope":
        return pre + "rope"
    if leaf == "qkv":
        return pre + ("qkv proj GEMM" if cls == "gemm" else "qkv elementwise (cos/sin, split)")
    if leaf == "sdpa":
        return pre + ("attention fallback matmul (fp32)" if cls == "gemm" else "attention fallback softmax/mask/expand")
    if leaf == "attn":
        return pre + ("o_proj GEMM" if cls == "gemm" else "attn KV write/gather/mask/cat/gate")
    if leaf == "proj" and "gdn" in parts:
        return "gdn in-proj GEMMs" if cls == "gemm" else "gdn proj other"
    if leaf == "conv":
        return "gdn conv (pad + MIOpen)"
    if leaf == "heads":
        return "gdn heads prep (split, beta, g, repeat_interleave)"
    if leaf == "chunk":
        return "gdn chunk (fla)" if cls == "fla" else "gdn chunk wrapper other"
    if leaf == "gate_norm":
        return "gdn gated norm"
    if leaf == "out":
        return "gdn out_proj GEMM" if cls == "gemm" else "gdn out other"
    if leaf == "gdn":
        return "gdn prep (cat window, state copy, silu, transpose)"
    if leaf == "routed":
        if cls == "moe_kernel":
            return pre + "MoE expert kernels"
        if cls in ("reduce",) or cls == "elementwise":
            return pre + "MoE tail (pair sum, cast)"
        return pre + "MoE routing index prep (argsort/scatter/cumsum)"
    if leaf == "route":
        return pre + "router softmax/topk"
    if leaf == "fused":
        return "lm_head GEMM" if cls == "gemm" else "prefill other (embed, to, arange)"
    if leaf == "moe":
        if cls == "moe_kernel":
            return pre + "MoE expert kernels"
        if cls == "gemm":
            return pre + "MoE shared+router+shared-gate GEMMs"
        if cls in ("softmax", "topk"):
            return pre + "router softmax/topk"
        return pre + "router renorm + shared act + shared-gate tail"
    if leaf == "in" and in_mtp:
        return "mtp: fc GEMM" if cls == "gemm" else "mtp: embed/cat"
    if leaf == "logits":
        return "mtp: lm_head GEMM" if cls == "gemm" else "mtp: logits other"
    if leaf == "layer":
        return "mtp: layer other"
    if leaf == "mtp":
        return "mtp: other (argmax, index)" if cls != "gemm" else "mtp: GEMM other"
    if leaf in ("prefill", "prefill_packed"):
        return "lm_head GEMM" if cls == "gemm" else "prefill other (embed, to, arange)"
    return f"{leaf}:{cls}"


FUSE = {  # buckets plan 26 would fuse (mtp: variants count too)
    "attention fallback matmul (fp32)": "attention",
    "attention fallback softmax/mask/expand": "attention",
    "attn KV write/gather/mask/cat/gate": "attention",
    "norm (layer/final)": "norms/residual",
    "residual add + all-reduce sum (eager)": "norms/residual",
    "q/k norm": "rope/gates/qk",
    "rope": "rope/gates/qk",
    "qkv elementwise (cos/sin, split)": "rope/gates/qk",
    "gdn conv (pad + MIOpen)": "gdn prep",
    "gdn heads prep (split, beta, g, repeat_interleave)": "gdn prep",
    "gdn prep (cat window, state copy, silu, transpose)": "gdn prep",
    "gdn gated norm": "gdn prep",
    "router softmax/topk": "router",
    "router renorm + shared act + shared-gate tail": "MoE tail",
    "MoE tail (pair sum, cast)": "MoE tail",
    "MoE routing index prep (argsort/scatter/cumsum)": "router",
    "mtp: pre-fc norms": "norms/residual",
    "mtp: final norm": "norms/residual",
}


def fuse_group(b: str) -> str | None:
    if b in FUSE:
        return FUSE[b]
    if b.startswith("mtp: "):
        return FUSE.get(b[5:])
    return None


def attribute(trace_path: str, entries: list[tuple], main_tid: int, ranks_by_dev: dict | None = None) -> dict:
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    xs = [e for e in events if e.get("ph") == "X"]
    # Clock alignment: main-thread record_function twins of the logged ranges (the trace
    # thread whose annotation labels follow the main thread's logged labels).
    ann_by_tid: dict = defaultdict(list)
    for e in xs:
        if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("ops:"):
            ann_by_tid[e.get("tid")].append(e)
    mine = sorted((x for x in entries if x[0] == main_tid), key=lambda x: x[3])
    labels = [m[2] for m in mine]
    diffs: list[float] = []
    for tid, ann in ann_by_tid.items():
        ann.sort(key=lambda e: float(e["ts"]))
        seq = [e["name"][4:] for e in ann]
        if seq == labels:
            diffs = [float(e["ts"]) - m[3] / 1e3 for e, m in zip(ann, mine)]
            break
    if not diffs:
        for tid, ann in ann_by_tid.items():  # best effort: per-label order
            by: dict[str, list] = defaultdict(list)
            for m in mine:
                by[m[2]].append(m)
            seen: Counter = Counter()
            cand = []
            for e in ann:
                lab = e["name"][4:]
                k = seen[lab]
                seen[lab] += 1
                if k < len(by[lab]):
                    cand.append(float(e["ts"]) - by[lab][k][3] / 1e3)
            if len(cand) > len(diffs):
                diffs = cand
    if not diffs:
        # A captured pass (plan 33's prefill graphs) runs no Python per op, so no wrapper
        # fires: attribute by kernel name only.
        return attribute_by_name(xs, note=f"no clock pairs (annotation threads {len(ann_by_tid)}, "
                                           f"main entries {len(mine)}): kernel names only")
    offset = statistics.median(diffs)
    spread = max(abs(d - offset) for d in diffs)
    # Ranges per tid and per rank, in trace microseconds.
    by_tid: dict[int, list] = defaultdict(list)
    by_rank: dict[int, list] = defaultdict(list)
    for tid, rank, path, a, b in entries:
        iv = (a / 1e3 + offset, b / 1e3 + offset, path)
        by_tid[tid].append(iv)
        by_rank[rank].append(iv)

    def finder(ivs):
        ivs = sorted(ivs, key=lambda v: v[0])
        starts = [v[0] for v in ivs]

        def find(t):
            # Ranges of one thread nest: the latest-starting range that contains t is the innermost.
            i = bisect.bisect_right(starts, t) - 1
            while i >= 0:
                a, b, p = ivs[i]
                if a <= t <= b:
                    return p
                i -= 1
            return None

        return find

    find_tid = {t: finder(v) for t, v in by_tid.items()}
    find_rank = {r: finder(v) for r, v in by_rank.items()}
    runtime = {}
    for e in xs:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                runtime[c] = e
    gpu = [e for e in xs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    devs = sorted({str((e.get("args") or {}).get("device", e.get("pid"))) for e in gpu}, key=lambda d: (len(d), d))
    if ranks_by_dev is None:
        ranks_by_dev = {d: i for i, d in enumerate(devs)}
    table: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])  # (dev, bucket) -> ms, n
    path_table: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    names: dict[str, Counter] = defaultdict(Counter)
    stats = Counter()
    for e in gpu:
        dev = str((e.get("args") or {}).get("device", e.get("pid")))
        dur = float(e.get("dur", 0)) / 1e3
        c = (e.get("args") or {}).get("correlation")
        rt = runtime.get(c)
        cls = kclass(e["name"]) if e.get("cat") == "kernel" else "copy"
        path = None
        if rt is None:
            stats["no_runtime_match"] += 1
            path = "?unmatched"
        else:
            t = float(rt["ts"])
            tid = rt.get("tid")
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                pass
            if tid in find_tid:
                path = find_tid[tid](t)
                stats["by_tid"] += 1
            else:
                r = ranks_by_dev.get(dev)
                path = find_rank[r](t) if r in find_rank else None
                stats["by_device"] += 1
        b = bucket(path or "", cls) if path != "?unmatched" else "unmatched"
        cell = table[(dev, b)]
        cell[0] += dur
        cell[1] += 1
        pc = path_table[(dev, path or "-", cls)]
        pc[0] += dur
        pc[1] += 1
        names[b][e["name"][:70]] += dur
    out_rows = defaultdict(dict)
    for (dev, b), (ms, n) in table.items():
        out_rows[b][dev] = (round(ms, 3), n)
    return {
        "devices": devs,
        "clock_offset_us": round(offset, 1),
        "clock_spread_us": round(spread, 1),
        "clock_pairs": len(diffs),
        "attribution": dict(stats),
        "buckets": out_rows,
        "paths": {f"{dev}|{p}|{c}": (round(v[0], 3), v[1]) for (dev, p, c), v in path_table.items()},
        "bucket_top_names": {b: [(n, round(t, 2)) for n, t in c.most_common(4)] for b, c in names.items()},
    }


def name_bucket(name: str) -> str:
    """The bucket of a kernel by its name alone (the graph-replayed pass)."""
    n = name.lower()
    cls = kclass(name)
    if "gdn_chunk_" in n:
        return "gdn chunk (fused, ENGINE_GDN_CHUNK)"
    if cls == "fla":
        return "gdn chunk (fla)"
    if "_gdn_conv" in n:
        return "gdn conv (pad + MIOpen)"
    if "_gdn_gates" in n:
        return "gdn heads prep (split, beta, g, repeat_interleave)"
    if "_gated_rms" in n:
        return "gdn gated norm"
    if cls == "conv":
        return "gdn conv (pad + MIOpen)"
    if "repeat_interleave" in n:
        return "gdn prep (cat window, state copy, silu, transpose)"
    return f"by name: {cls}"


def attribute_by_name(xs: list, note: str) -> dict:
    gpu = [e for e in xs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    devs = sorted({str((e.get("args") or {}).get("device", e.get("pid"))) for e in gpu}, key=lambda d: (len(d), d))
    table: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    names: dict[str, Counter] = defaultdict(Counter)
    for e in gpu:
        dev = str((e.get("args") or {}).get("device", e.get("pid")))
        b = name_bucket(e["name"]) if e.get("cat") == "kernel" else "by name: copy"
        cell = table[(dev, b)]
        cell[0] += float(e.get("dur", 0)) / 1e3
        cell[1] += 1
        names[b][e["name"][:70]] += float(e.get("dur", 0)) / 1e3
    out_rows = defaultdict(dict)
    for (dev, b), (ms, n) in table.items():
        out_rows[b][dev] = (round(ms, 3), n)
    return {
        "devices": devs, "mode": "names", "note": note, "buckets": out_rows,
        "bucket_top_names": {b: [(n, round(t, 3)) for n, t in c.most_common(12)] for b, c in names.items()},
    }


GDN_CHUNK_PREP = (  # the kill rule's "chunk + prep" (plan 40 part B): conv through the core output
    "gdn conv (pad + MIOpen)",
    "gdn heads prep (split, beta, g, repeat_interleave)",
    "gdn prep (cat window, state copy, silu, transpose)",
    "gdn chunk (fla)",
    "gdn chunk wrapper other",
    "gdn chunk (fused, ENGINE_GDN_CHUNK)",
)


# ============================================================================ passes


def passes(args) -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from engine.generator import CHECKPOINT, Generator, PackedPrefill
    from engine.protocol import Sampling
    from serve_probe import token_pool

    tp = os.environ.get("ENGINE_TP", "(default)")
    print(f"passes: ENGINE_TP={tp} visible={os.environ.get('HIP_VISIBLE_DEVICES')} torch {torch.__version__} "
          f"gpus {torch.cuda.device_count()}", flush=True)
    t0 = time.perf_counter()
    gen = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
        prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30,
    )
    assert gen.fast is not None and gen.packing, "the fast path with packing did not load"
    result("load", tp=tp, kind_fast=type(gen.fast).__name__, seconds=round(time.perf_counter() - t0, 1),
           mtp=gen.fast.mtp is not None)
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 80_000, "synthetic")
    packer = PackedPrefill(gen)
    one = Sampling(max_tokens=1, temperature=0.0, ignore_eos=True)
    devices = list(range(torch.cuda.device_count()))
    serial = [0]

    def sync() -> None:
        for d in devices:
            torch.cuda.synchronize(d)

    def run_one(ctx: int, new: int, wrap=None) -> float:
        serial[0] += 1
        offset = (serial[0] * 7919) % (len(pool) - ctx - new - 1)
        ids = pool[offset : offset + ctx + new]
        packer.add(tuple(ids[:ctx]), one)
        done = packer.step()
        g = done[0][1]
        assert not isinstance(g, Exception), g
        g.next_step()
        g.retire()
        sync()
        packer.add(tuple(ids), one)
        sync()
        b = time.perf_counter()
        done = packer.step() if wrap is None else wrap(packer.step)
        sync()
        c = time.perf_counter()
        g = done[0][1]
        assert not isinstance(g, Exception), g
        assert g.cached_tokens == ctx, (g.cached_tokens, ctx)
        g.next_step()
        gen._reclaim()
        return (c - b) * 1e3

    news = [int(n) for n in args.news.split(",")]
    for _ in range(2):
        for n in news:
            run_one(CONTEXT, n)
    sync()
    log = Log()
    log.install()
    tmp = tempfile.mkdtemp(prefix="op-split-")
    main_tid = threading.get_native_id()
    for n in news:
        walls = [run_one(CONTEXT, n) for _ in range(args.reps)]
        wall = statistics.median(walls)
        # Host log pass: wrappers on, no profiler.
        log.entries.clear()
        log.active, log.rf = True, False
        host_wall = run_one(CONTEXT, n)
        log.active = False
        host = defaultdict(float)
        for tid, rank, path, a, b in log.entries:
            host[(rank, path)] += (b - a) / 1e6
        tops = {}
        for r in sorted({k[0] for k in host}):
            tops[r] = {p: round(v, 2) for (rr, p), v in sorted(host.items(), key=lambda kv: -kv[1]) if rr == r}
        exch = {r: round(sum(v for (rr, p), v in host.items() if rr == r and p.endswith("exchange")), 2) for r in tops}
        top_level = {r: round(sum(v for (rr, p), v in host.items() if rr == r and p in ("prefill", "prefill_packed")), 2)
                     for r in tops}
        result("host_log", new_tokens=n, wall_ms_median=round(wall, 2), wall_all=[round(w, 1) for w in walls],
               host_log_pass_ms=round(host_wall, 1), rank_prefill_host_ms=top_level, exchange_host_ms=exch,
               host_ms_by_range=tops)
        # Profiled passes.
        for k in range(args.prof):
            trace = os.path.join(tmp, f"split-{n}-{k}.json")
            log.entries.clear()

            def profiled(step):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function("probe_pass"):
                        log.active, log.rf = True, True
                        try:
                            out = step()
                        finally:
                            log.active, log.rf = False, False
                    sync()
                prof.export_chrome_trace(trace)
                return out

            pms = run_one(CONTEXT, n, profiled)
            try:
                stats = analyze_trace(trace, main_tid)
            except Exception as exc:  # noqa: BLE001
                stats = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                split = attribute(trace, list(log.entries), main_tid)
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                split = {"error": f"{type(exc).__name__}: {exc}"}
            keep = {k2: stats.get(k2) for k2 in ("prof_wall_ms", "kernels", "kernels_total", "kernel_sum_ms",
                                                  "copy_sum_ms", "busy_ms", "busy_any_ms", "idle_any_ms",
                                                  "gpu_ms_by_category", "runtime_launch_calls", "runtime_sync_calls")}
            result("split", new_tokens=n, prof_rep=k, wall_ms_median=round(wall, 2), profiled_pass_ms=round(pms, 1),
                   **keep, **split)
            os.remove(trace)
    log.uninstall()
    return 0


# ============================================================================ all + report


def report(splits: list[dict], hosts: list[dict]) -> None:
    by_n: dict[int, dict] = {}
    for s in splits:
        if "buckets" in s:
            by_n[s["new_tokens"]] = s  # the last profiled pass per size
    if not by_n:
        print("no split results", flush=True)
        return
    ns = sorted(by_n)
    ndev = max(len(by_n[n]["devices"]) for n in ns)
    rows = set()
    for n in ns:
        rows |= set(by_n[n]["buckets"])

    def per_gcd(n, b):
        cell = by_n[n]["buckets"].get(b, {})
        devs = by_n[n]["devices"]
        ms = sum(cell.get(d, (0, 0))[0] for d in devs) / max(1, len(devs))
        cnt = sum(cell.get(d, (0, 0))[1] for d in devs) / max(1, len(devs))
        return ms, cnt

    order = sorted(rows, key=lambda b: -per_gcd(ns[len(ns) // 2], b)[0])
    print(f"\n=== per-GCD prefill GPU time by call site (ms per GCD = sum over {ndev} GCDs / {ndev}; launches per GCD) ===")
    print(f"{'bucket':<58} {'fuse':<15}" + "".join(f" {f'{n} ms':>9} {'#':>5}" for n in ns))
    totals = defaultdict(float)
    fuse_tot = defaultdict(lambda: defaultdict(float))
    for b in order:
        g = fuse_group(b) or ""
        line = f"{b[:58]:<58} {g:<15}"
        for n in ns:
            ms, cnt = per_gcd(n, b)
            totals[n] += ms
            if g:
                fuse_tot[n][g] += ms
            line += f" {ms:>9.2f} {cnt:>5.0f}"
        print(line)
    print(f"{'TOTAL kernel+copy':<74}" + "".join(f" {totals[n]:>9.2f} {'':>5}" for n in ns))
    print("\n=== GDN chunk + prep (plan 40 part B kill rule: conv through the core output), ms per GCD ===")
    for n in ns:
        mode = by_n[n].get("mode", "call sites")
        parts = {b: round(per_gcd(n, b)[0], 3) for b in GDN_CHUNK_PREP if b in by_n[n]["buckets"]}
        tot = round(sum(parts.values()), 3)
        launches = sum(per_gcd(n, b)[1] for b in GDN_CHUNK_PREP if b in by_n[n]["buckets"])
        print(f"N={n} ({mode}): {tot} ms, {launches:.0f} launches per GCD; {parts}")
        result("gdn_chunk_prep", new_tokens=n, mode=mode, ms_per_gcd=tot, launches_per_gcd=launches, parts=parts,
               gated_norm_ms=round(per_gcd(n, "gdn gated norm")[0], 3))
    print("\n=== fusible groups (plan 26), ms per GCD ===")
    groups = sorted({g for n in ns for g in fuse_tot[n]})
    for g in groups:
        print(f"{g:<20}" + "".join(f" {n:>5}: {fuse_tot[n][g]:>7.2f}" for n in ns))
    print(f"{'ALL FUSIBLE':<20}" + "".join(f" {n:>5}: {sum(fuse_tot[n].values()):>7.2f}" for n in ns))
    for n in ns:
        s = by_n[n]
        print(f"\nN={n}: wall median {s['wall_ms_median']} ms; kernel_sum_ms {s.get('kernel_sum_ms')}; busy_ms {s.get('busy_ms')}; "
              f"busy_any {s.get('busy_any_ms')}; idle_any {s.get('idle_any_ms')}; kernels {s.get('kernels')}; "
              f"clock spread {s.get('clock_spread_us')} us over {s.get('clock_pairs')} pairs; attribution {s.get('attribution') or s.get('note')}")
    for h in hosts:
        print(f"host N={h['new_tokens']}: wall {h['wall_ms_median']} ms; per-rank prefill host ms {h['rank_prefill_host_ms']}; "
              f"rendezvous {h['exchange_host_ms']}")
    at = 561 if 561 in fuse_tot else ns[len(ns) // 2]
    total = sum(fuse_tot[at].values())
    result("kill", at_tokens=at, fusible_ms_per_gcd=round(total, 2), threshold=KILL_MS,
           groups={g: round(v, 2) for g, v in sorted(fuse_tot[at].items(), key=lambda kv: -kv[1])},
           verdict="PARK (fusible < 35 ms/GCD)" if total < KILL_MS else "CONTINUE (fusible >= 35 ms/GCD)")


def run_all(args) -> int:
    started = time.time()
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = args.devices
    if args.eager:  # per-op call-site attribution needs Python per op: no captured prefill
        env["ENGINE_PREFILL_GRAPH"] = "0"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = [sys.executable, str(Path(__file__).resolve()), "passes", "--reps", str(args.reps), "--news", args.news,
           "--prof", str(args.prof)]
    print(f"all: {' '.join(cmd)} (HIP_VISIBLE_DEVICES={args.devices}, ENGINE_TP={env.get('ENGINE_TP', 'default')}, "
          f"ENGINE_PREFILL_GRAPH={env.get('ENGINE_PREFILL_GRAPH', 'default')})",
          flush=True)
    splits, hosts = [], []
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith("RESULT "):
            body = json.loads(line[7:])
            if body.get("kind") == "split":
                splits.append(body)
            elif body.get("kind") == "host_log":
                hosts.append(body)
    proc.wait()
    print(f"all: passes exited {proc.returncode}", flush=True)
    report(splits, hosts)
    print(f"all: done in {time.time() - started:.0f} s", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("all", "passes"):
        p = sub.add_parser(name)
        p.add_argument("--news", default=",".join(str(n) for n in NEWS))
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--prof", type=int, default=2)
        if name == "all":
            p.add_argument("--devices", default="4,5")
            p.add_argument("--eager", action="store_true",
                           help="ENGINE_PREFILL_GRAPH=0: eager passes, attributed per call site (the "
                                "default captured passes are attributed by kernel name only)")
    args = parser.parse_args()
    return run_all(args) if args.cmd == "all" else passes(args)


if __name__ == "__main__":
    sys.exit(main())
