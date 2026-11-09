"""Plan 29 step 0: how much of the TP2 captured decode/verify step is fusible glue?

    python tools/glue_split.py [--devices 0,1] [--prompt-len 1800] [--steps 5] [--iters 30]

Loads one TP2 replica as the server does (``Generator.load``, ``ENGINE_TP=2``, MTP on) on
``--devices``, prefills 4 slots with replay-like prompts, then for each captured graph
(decode B=2/4, verify B=1/2):

1. Wall time: the median of the full ``fast.decode`` / ``fast.verify`` call (host I/O included,
   as tools/spec_step_bench.py) and of the bare graph replay of both ranks.
2. A torch.profiler trace of ``--steps`` bare graph replays of both ranks (one step at a time,
   a 3 ms pause between steps so they separate on the GPU timeline). Per GCD per step: kernel
   count, kernel time, span (first start to last end) and the idle gaps between kernels.
3. One eager step of the same shape under the profiler, with the engine's glue call sites
   wrapped in ``record_function`` labels (rank 0's thread): ``Group.allreduce``,
   ``Group.argmax``, ``torch.zeros_like``, ``FastModel._logits`` (the fp32 cast of the head),
   ``FastModel._mtp_in`` / ``_mtp_logits`` (torch RMSNorm, embedding, cat vs the fc / lm_head
   GEMMs), ``torch.stack``. Each eager launch gets the labels enclosing it; an eager
   all-reduce (exchange, copy, add) collapses into one ``_ll_allreduce_kernel`` entry, as it
   is one kernel in the graph. The graph's kernel sequence of each GCD is aligned to rank 0's
   eager sequence by name (difflib), so each graph kernel inherits its call site.

Categories (the plan's five items, then the rest):
  1_add_rms     _add_rms_kernel (residual add + RMSNorm, the launches an AR+RMS kernel removes)
  2_topk        _topk_kernel (router top-k, to move into the MoE-up prologue)
  3_zeros_like  the zero residual fill for the MoE-down kernel
  4_argmax      Group.argmax: argmax, gather, zeros, index stores, its all-reduce, where, long
  4b_logits_f32 the bf16 -> fp32 cast of the head's logits in FastModel._logits
  5_mtp_glue    torch RMSNorms, embedding and cat of _mtp_in, torch.stack of the verify picks
  5b_mtp_lg_rms the torch RMSNorm of _mtp_logits (plan 28's code: not in this plan's scope)
  allreduce     the layers' LL all-reduce kernels (the host of item 1, not removed)
  gemv / moe / gdn / attn / blas / other_torch / other
Each kernel's cost is its time plus the idle gap before it on its GCD (the graph node's
launch overhead: removing the node removes both). Kill (plan 29): items 1-5 (incl. 4b, not
5b) < 800 us per GCD per verify B=2 step.

Plan 37: ``--configs`` also takes ``chain:B:k`` (the chained verify, ``verify_chain`` with k
drafts per slot, B (k + 1) rows, ``ENGINE_SPEC_CHAIN=1``), and ``--parity`` adds chain B=2
k=1, 3 hashes; the flags printed include ``ENGINE_TOPK_EPI``. The router top-k fused into the
K-sum kernel (``_ksum_router_kernel``) is listed under its own name in ``2_topk``.

Prints ``RESULT`` lines. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GLUE = ("1_add_rms", "2_topk", "3_zeros_like", "4_argmax", "4b_logits_f32", "5_mtp_glue")
ORDER = GLUE + ("5b_mtp_lg_rms", "allreduce", "allreduce_rms", "gemv", "moe", "gdn", "attn", "blas",
                "other_torch", "other")
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
LABEL = "glue:"


def out(msg: str) -> None:
    print(msg, flush=True)


def norm_name(name: str) -> str:
    if "_ll_ar_rms_kernel" in name:
        return "_ll_ar_rms_kernel"
    return "_ll_allreduce_kernel" if "_ll_allreduce_kernel" in name else name


def family(name: str) -> str:
    n = name
    if "_ll_ar_rms_kernel" in n:
        return "allreduce_rms"
    if "_ll_allreduce_kernel" in n:
        return "allreduce"
    if "_add_rms_kernel" in n:
        return "1_add_rms"
    if "_topk_kernel" in n or "_ksum_router_kernel" in n:
        return "2_topk"
    if "_gemv_kernel" in n:
        return "gemv"
    if "_moe_" in n:
        return "moe"
    if "_gdn" in n or n.startswith("gdn_") or "chunk_" in n or "fused_recurrent" in n:
        return "gdn"
    if "_paged_attn" in n or "_attn_" in n:
        return "attn"
    if any(k in n for k in ("Cijk", "rocblas", "hipblas", "gemm", "Gemm")):
        return "blas"
    if "at::native" in n or "Memcpy" in n or "Memset" in n or "memcpy" in n.lower():
        return "other_torch"
    return "other"


def category(name: str, labels: tuple[str, ...] | None) -> str:
    """The plan item of one kernel from its name and (eager) call-site labels."""
    fam = family(name)
    if fam in ("1_add_rms", "2_topk"):
        return fam
    if labels is None:
        return fam
    if "argmax" in labels:
        return "4_argmax"
    if fam == "allreduce" or "AR" in labels:
        return "allreduce"
    if "zeros_like" in labels:
        return "3_zeros_like"
    inner = labels[-1] if labels else ""
    if "mtp_in" in labels:
        return fam if inner == "linear" else "5_mtp_glue"
    if "mtp_logits" in labels:
        return fam if inner == "linear" else "5b_mtp_lg_rms"
    if "stack" in labels:
        return "5_mtp_glue"
    if "logits" in labels and fam not in ("gemv", "blas"):
        return "4b_logits_f32"
    return fam


# ------------------------------------------------------------------------- trace parsing


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


def gpu_events(trace: list[dict]) -> dict[int, list[tuple[float, float, str, int | None]]]:
    """device -> [(start us, dur us, name, correlation)] sorted by start."""
    by = defaultdict(list)
    for e in trace:
        if e.get("ph") != "X" or e.get("cat") not in GPU_CATS:
            continue
        args = e.get("args") or {}
        dev = args.get("device", e.get("pid"))
        try:
            dev = int(dev)
        except (TypeError, ValueError):
            continue
        by[dev].append((float(e["ts"]), float(e.get("dur", 0.0)), e.get("name", "?"),
                        args.get("correlation")))
    for d in by:
        by[d].sort()
    return dict(by)


def split_steps(evs, nsteps: int, pause_us: float = 1500.0):
    """Cut one device's events into steps at idle gaps longer than ``pause_us``."""
    steps, cur, last_end = [], [], None
    for ev in evs:
        if last_end is not None and ev[0] - last_end > pause_us and cur:
            steps.append(cur)
            cur = []
        cur.append(ev)
        last_end = max(last_end or 0.0, ev[0] + ev[1])
    if cur:
        steps.append(cur)
    return steps


def eager_sequence(trace: list[dict]):
    """Rank 0's eager launches in issue order: [(name, labels)] and its GPU device."""
    anns = [e for e in trace if e.get("ph") == "X" and e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith(LABEL)]
    root = [a for a in anns if a["name"] == LABEL + "probe_eager"]
    if not root:
        raise RuntimeError("no probe_eager annotation in the eager trace")
    root = root[0]
    tid, pid = root["tid"], root["pid"]
    r0, r1 = float(root["ts"]), float(root["ts"]) + float(root["dur"])
    mine = [(float(a["ts"]), float(a["ts"]) + float(a["dur"]), a["name"][len(LABEL):], id(a))
            for a in anns if a["tid"] == tid and a["pid"] == pid and a is not root]
    kernels = {}
    for e in trace:
        if e.get("ph") == "X" and e.get("cat") in GPU_CATS:
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                kernels.setdefault(c, []).append(e)
    launches = []
    for e in trace:
        if e.get("ph") != "X" or e.get("cat") in GPU_CATS or e.get("cat") in ("user_annotation",
                                                                               "gpu_user_annotation"):
            continue
        if e.get("tid") != tid or e.get("pid") != pid:
            continue
        c = (e.get("args") or {}).get("correlation")
        ts = float(e["ts"])
        if c is None or c not in kernels or not (r0 <= ts <= r1):
            continue
        for k in kernels[c]:
            launches.append((ts, k))
    seen = set()
    launches = [t for t in sorted(launches, key=lambda t: t[0])
                if not (id(t[1]) in seen or seen.add(id(t[1])))]
    devs = defaultdict(int)
    seq = []
    ar_seen = set()
    for ts, k in launches:
        enclosing = sorted((a for a in mine if a[0] <= ts <= a[1]), key=lambda a: (a[0], -a[1]))
        labels = tuple(a[2] for a in enclosing)
        ar = [a for a in enclosing if a[2] == "AR"]
        if ar:
            key = ar[0][3]
            if key in ar_seen:
                continue
            ar_seen.add(key)
            outer = tuple(a[2] for a in enclosing if a[0] < ar[0][0] or a[3] == key)
            seq.append(("_ll_allreduce_kernel", outer))
            continue
        try:
            devs[int((k.get("args") or {}).get("device", k.get("pid")))] += 1
        except (TypeError, ValueError):
            pass
        seq.append((norm_name(k.get("name", "?")), labels))
    dev = max(devs, key=devs.get) if devs else None
    return seq, dev


# ------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--configs", default="verify:2,verify:1,decode:2,decode:4")
    parser.add_argument("--parity", action="store_true",
                        help="plan 29: hashes of captured logits and picks under the ENGINE_GLUE_* flags")
    parser.add_argument("--parity-steps", type=int, default=24)
    args = parser.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = args.devices
    os.environ.setdefault("ENGINE_MTP", "1")
    os.environ.setdefault("ENGINE_TP", "2")

    import torch
    from torch.nn import functional as F
    from torch.profiler import ProfilerActivity, profile, record_function

    from engine.fast import kernels as KN
    from engine.fast import model as MD
    from engine.fast import tp as TP
    from engine.generator import CHECKPOINT, Generator

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def median_ms(fn, iters: int) -> float:
        for _ in range(3):
            fn()
        times = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(times)

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    ranks = getattr(fast, "ranks", None)
    out(f"RESULT load_s={time.perf_counter() - started:.0f} gpus={torch.cuda.device_count()} "
        f"tp={os.environ.get('ENGINE_TP')} ranks={len(ranks) if ranks else 1} "
        f"mtp={fast.mtp is not None} graphs={fast.use_graphs} graph_error={fast.graph_error} "
        f"fused={fast.fused is not None}")
    if not ranks:
        raise SystemExit("not a TP2 replica")
    tok = g.tokenizer

    ids: list[int] = []
    i = 0
    while len(ids) < 4 * 9973 + args.prompt_len:
        for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i}"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(prompt) - 1
        pos = 0
        while pos < body:
            end = min((pos // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[pos:end], pos, prompt[end])
            pos = end
        return fast.prefill(slot, prompt[body:], body)

    plen = args.prompt_len
    slots = [0, 1, 2, 3]
    lasts = [int(torch.argmax(prefill(s, ids[9973 * s: 9973 * s + plen]))) for s in slots]

    def call(kind: str, b: int, k: int = 0):
        if kind == "chain":
            return lambda: fast.verify_chain(lasts[:b], [[0] * k] * b, slots[:b], [plen] * b)
        if kind == "decode":
            return lambda: fast.decode(lasts[:b], slots[:b], [plen] * b)
        return lambda: fast.verify(lasts[:b], [0] * b, slots[:b], [plen] * b)

    def replay(kind: str, size: int, k: int = 0):
        if kind == "chain":
            segs = [m.chain_segments(k + 1)[0] for m in ranks]
        else:
            segs = [(m.vsegments if kind == "verify" else m.segments)[0] for m in ranks]

        def run():
            for seg in segs:
                with torch.cuda.device(seg.device):
                    seg.graphs[size].replay()
        return run

    # Labels for the eager step.
    patches = []

    def wrap(obj, attr, label):
        orig = getattr(obj, attr)

        def wrapped(*a, **kw):
            with record_function(LABEL + label):
                return orig(*a, **kw)
        patches.append((obj, attr, orig, wrapped))

    wrap(TP.Group, "allreduce", "AR")
    wrap(TP.Group, "argmax", "argmax")
    wrap(torch, "zeros_like", "zeros_like")
    wrap(torch, "stack", "stack")
    wrap(torch, "cat", "cat")
    wrap(F, "linear", "linear")
    wrap(F, "embedding", "embedding")
    wrap(MD, "_rms", "torch_rms")
    wrap(MD.FastModel, "_logits", "logits")
    wrap(MD.FastModel, "_mtp_in", "mtp_in")
    wrap(MD.FastModel, "_mtp_logits", "mtp_logits")
    wrap(KN, "add_rms", "add_rms")

    summary = {}
    for spec in args.configs.split(","):
        kind, b, *kk = spec.split(":")
        b = int(b)
        k = int(kk[0]) if kk else 0
        size = fast.bucket(b)
        if kind == "chain":
            kind = f"chain{k}"
        base = "chain" if kind.startswith("chain") else kind
        try:
            full = median_ms(call(base, b, k), args.iters)  # also captures the graph
            if fast.graph_error is not None:
                out(f"RESULT {kind} B={b} graph_error={fast.graph_error}")
                continue
            bare = median_ms(replay(base, size, k), args.iters)
            out(f"RESULT wall {kind} B={b} bucket={size} call_ms={full:.3f} graph_replay_ms={bare:.3f}")

            # 2. graph replays under the profiler
            run = replay(base, size, k)
            run()
            sync()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(args.steps):
                    run()
                    sync()
                    time.sleep(0.003)
            gtrace = load_trace(prof)
            gev = gpu_events(gtrace)

            # 3. one eager step with labels
            saved = [m.use_graphs for m in ranks]
            for m in ranks:
                m.use_graphs = False
            try:
                call(base, b, k)()
                sync()
                for obj, attr, _, wrapped in patches:
                    setattr(obj, attr, wrapped)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function(LABEL + "probe_eager"):
                        call(base, b, k)()
                        sync()
            finally:
                for obj, attr, orig, _ in patches:
                    setattr(obj, attr, orig)
                for m, u in zip(ranks, saved):
                    m.use_graphs = u
            eseq, edev = eager_sequence(load_trace(prof))
            enames = [n for n, _ in eseq]
            out(f"RESULT eager {kind} B={b} rank0_launches={len(eseq)} rank0_profiler_device={edev}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            out(f"RESULT {kind} B={b} error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        devs = sorted(d for d, evs in gev.items() if any("_gemv_kernel" in e[2] for e in evs))
        per_gcd = {}
        for gcd, d in enumerate(devs):
            steps = split_steps(gev[d], args.steps)
            counts = [len(s) for s in steps]
            if len(steps) != args.steps or len(set(counts)) != 1:
                out(f"RESULT warn {kind} B={b} gcd={gcd} steps={len(steps)} kernel_counts={counts}")
            n = statistics.median(counts)
            good = [s for s in steps if len(s) == n]
            # align one step's names to rank 0's eager sequence
            gnames = [norm_name(e[2]) for e in good[0]]
            sm = difflib.SequenceMatcher(a=gnames, b=enames, autojunk=False)
            labels: list[tuple[str, ...] | None] = [None] * len(gnames)
            for blk in sm.get_matching_blocks():
                for k in range(blk.size):
                    labels[blk.a + k] = eseq[blk.b + k][1]
            matched = sum(1 for x in labels if x is not None)
            cats = [category(nm, lb) for nm, lb in zip(gnames, labels)]
            st = defaultdict(lambda: [0.0, 0.0, 0.0])  # cat -> [calls, kernel us, gap us]
            spans, busys, gaps = [], [], []
            names_t = defaultdict(float)
            for s in good:
                t0 = s[0][0]
                end = t0
                busy = gap_tot = 0.0
                for k, (ts, dur, name, _) in enumerate(s):
                    gap = max(0.0, ts - end) if k else 0.0
                    c = cats[k]
                    st[c][0] += 1
                    st[c][1] += dur
                    st[c][2] += gap
                    busy += dur
                    gap_tot += gap
                    end = max(end, ts + dur)
                    if c in ("other", "other_torch") or c in GLUE or c == "5b_mtp_lg_rms":
                        names_t[(c, name[:100])] += dur + gap
                spans.append(end - t0)
                busys.append(busy)
                gaps.append(gap_tot)
            ns = len(good)
            span, busy, gap = (statistics.median(spans), statistics.median(busys),
                               statistics.median(gaps))
            out(f"RESULT step {kind} B={b} gcd={gcd} kernels={n:.0f} span_us={span:.0f} "
                f"kernel_us={busy:.0f} gap_us={gap:.0f} mean_gap_us={gap / max(n - 1, 1):.2f} "
                f"aligned={matched}/{len(gnames)} steps={ns}")
            glue = glue_n = 0.0
            for c in ORDER:
                if c not in st:
                    continue
                calls, kus, gus = (v / ns for v in st[c])
                out(f"RESULT split {kind} B={b} gcd={gcd} item={c:14s} calls={calls:6.1f} "
                    f"kernel_us={kus:8.1f} gap_us={gus:7.1f} total_us={kus + gus:8.1f} "
                    f"share={(kus + gus) / max(span, 1e-9) * 100:5.1f}%")
                if c in GLUE:
                    glue += kus + gus
                    glue_n += calls
            per_gcd[gcd] = (glue, glue_n, span)
            out(f"RESULT glue {kind} B={b} gcd={gcd} fusible_us={glue:.0f} nodes={glue_n:.0f} "
                f"of_span={glue / max(span, 1e-9) * 100:.1f}% "
                f"(alt: kernel_us + 1.55us x nodes = "
                f"{sum(st[c][1] for c in GLUE if c in st) / ns + 1.55 * glue_n:.0f})")
            for (c, name), us in sorted(names_t.items(), key=lambda kv: -kv[1])[:18]:
                out(f"RESULT top {kind} B={b} gcd={gcd} {c:14s} us={us / ns:7.1f} name={name}")
        summary[(kind, b)] = per_gcd

    v2 = summary.get(("verify", 2))
    if v2:
        worst = min(v[0] for v in v2.values())
        mean = statistics.mean(v[0] for v in v2.values())
        out(f"RESULT kill_test verify B=2 fusible_us_per_gcd mean={mean:.0f} min={worst:.0f} "
            f"threshold=800 -> {'GO' if mean >= 800 else 'KILL'}")
    if args.parity:
        glue_parity(fast, ranks, prefill, ids, plen, args.parity_steps, torch, sync)
    if hasattr(fast, "timeouts"):
        out(f"RESULT tp_allreduce_timeouts={fast.timeouts()}")


def glue_parity(fast, ranks, prefill, ids, plen, steps, torch, sync) -> None:
    """Captured decode (B=2, 4: ``steps`` greedy steps) and verify (B=1, 2, 4: one step from a
    fresh prefill) under this process's ENGINE_GLUE_* flags: a SHA-256 of the fp32 logits and
    the picks of each, to compare bitwise with a run under other flags (another process)."""
    import hashlib

    from engine.fast import fused as FZ
    from engine.fast import tp as TP

    def digest(t) -> str:
        return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

    from engine.fast import mfma as MF

    flags = (f"topk={int(FZ.GLUE_TOPK)} nozero={int(FZ.GLUE_NOZERO)} arrms={int(TP.GLUE_ARRMS)} "
             f"topk_epi={MF.TOPK_EPI}")
    if MF.CHAIN:
        for k in (1, 3):
            if k > MF.KMAX:
                continue
            slots = [0, 1]
            cur = [int(torch.argmax(prefill(s, ids[9973 * s: 9973 * s + plen]))) for s in slots]
            drafts = [[(c + 17 * (j + 1)) % 1000 for j in range(k)] for c in cur]
            logits, picks, acc, nxt = fast.verify_chain(cur, drafts, slots, [plen] * 2)
            out(f"RESULT parity_hash chain k={k} B=2 {flags} graphs={fast.graph_error is None} "
                f"logits={digest(logits.float().cpu())} picks={picks} acc={acc} next={nxt}")
    for b in (2, 4):
        slots = list(range(b))
        cur = [int(torch.argmax(prefill(s, ids[9973 * s: 9973 * s + plen]))) for s in slots]
        logs, toks = [], []
        for step in range(steps):
            o = fast.decode(cur, slots, [plen + step] * b).float().cpu()
            logs.append(o)
            cur = o.argmax(-1).tolist()
            toks.append(cur)
        out(f"RESULT parity_hash decode B={b} steps={steps} {flags} graphs={fast.graph_error is None} "
            f"logits={digest(torch.stack(logs))} tokens={digest(torch.tensor(toks))}")
    for b in (1, 2, 4):
        slots = list(range(b))
        cur = [int(torch.argmax(prefill(s, ids[9973 * s: 9973 * s + plen]))) for s in slots]
        drafts = [(c + 17) % 1000 for c in cur]
        logits, g1, g2, d1, d2 = fast.verify(cur, drafts, slots, [plen] * b)
        out(f"RESULT parity_hash verify B={b} {flags} graphs={fast.graph_error is None} "
            f"logits={digest(logits.float().cpu())} picks={(g1, g2, d1, d2)}")
    for s in range(4):
        fast.reset(s)

if __name__ == "__main__":
    main()
