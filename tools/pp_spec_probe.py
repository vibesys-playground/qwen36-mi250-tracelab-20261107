"""Plan 68 stage 0: is a speculative two-chunk pipeline over a layer-split pair worth building?

    python -u tools/pp_spec_probe.py stage --tp 1|2 [--devices 0,1] [--configs 1:3,2:3,...]
    python -u tools/pp_spec_probe.py deep  [--devices 2,3] [--gen 320] [--links 15]
    python -u tools/pp_spec_probe.py mem   [--devices 4,5] [--split 22]
    python -u tools/pp_spec_probe.py project LOG [LOG ...]

Tools only: no engine code changes. Every subcommand prints ``RESULT`` lines, and the ones
``project`` needs as ``RESULT json <kind> {...}``. A hint for the plan, not the task's metric.

stage (a). Loads one replica as the server does (``Generator.load``) under ``ENGINE_TP``:
  tp=2 today's tensor-parallel replica, tp=1 the layer split (``device_map="auto"`` over the
  pair: GCD0 the embedding and the first layers, GCD1 the rest, the final norm, lm_head, MTP
  head). Two slots prefilled with replay-like prompts; for each (B, k), the chained verify
  (``verify_chain``, k corpus drafts per slot, its graphs captured at load):
  - call_ms: the whole ``verify_chain`` call (host included, no run-ahead);
  - tp2: replay_ms, both ranks' graphs replayed together (the GPU step);
  - tp1: seg0_ms / seg1_ms each segment's graph replayed alone (stage 1 / stage 2 as split
    by device_map), serial_ms both in turn with the activation copy, overlap_ms both replayed
    at once with no dependency (one pipeline tick, any interference between the two GCDs
    included), handoff_ms the [rows, 2048] BF16 copy GCD0 -> GCD1;
  - per-op graphs (tp1): every op of the step (embedding, each layer, final norm, chain head)
    captured as its own graph and timed alone, scaled per device to the segment's measured
    time, to place the split anywhere (the layers' weights stay where device_map put them,
    the timing per layer does not depend on which GCD runs it);
  - link_ms (tp1): one extra MTP chain link on B rows (the head's layer, final norm, IVF
    draft) as its own graph: the pipeline's stage 2 needs 2k + 1 drafts, today's head makes k;
  - the kernel class split of each segment (tp1) or each GCD (tp2) from a profiled replay,
    with tools/glue_split.py's families.
deep (b). On the layer split (the pipeline's own head: full-width MTP and IVF on one GCD),
  replay-like and natural prompts: (1) a greedy continuation of ``--gen`` tokens through the
  served chained verify (graphs, IVF on, k = 3), its acceptance; (2) the same text teacher-
  forced one position at a time (eager, the same kernels): at every anchor the head's chain
  of ``--links`` drafts as ``_chain_head_op`` builds it (link 1 from the anchor row's head
  output, link j the head on its own previous output); ``M`` = the leading drafts that equal
  the greedy text. Per-depth acceptance, and M per anchor (hex, capped at 15) for ``project``.
mem (c). Loads the TP2 replica with, per GCD, the other rank's shard of the layers its
  pipeline stage owns (GCD0 layers < split, GCD1 layers >= split and the MTP layer) and a
  second set of GDN state rings for them, allocated before the KV pool is sized; reports free
  HBM per GCD after the weights of both layouts, the KV pages the pool gets, and free HBM
  after load and capture.
project (d). Exact simulation along each text's M per anchor: TP2 serial spec decode (a
  step from bonus position m commits min(M, k) + 1 tokens) and the two-chunk pipeline (one
  tick = max(stage1, stage2 + (k + 1) links) + handoff; a chunk's rows are drafted from the
  latest verified anchor at depths k + 1 .. 2k + 1 while the chunk ahead of it is in stage 2;
  a miss flushes the chunk behind it and costs one stage-2 bubble), per (B, k) and corpus,
  against TP2 at the same B; the kill rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

NATURAL = [
    "Write a Python function that parses an ISO-8601 timestamp into a timezone-aware datetime, "
    "with validation, a docstring and three doctests.",
    "Implement a thread-safe LRU cache class in Python and explain the design.",
    "Explain the difference between TCP and UDP to a new backend engineer.",
    "Write a short story (about 200 words) about a lighthouse keeper.",
    "Write a Go HTTP handler that returns the median of a JSON list of integers, with tests.",
    "Explain Rust's ownership and borrowing rules with three small code examples.",
    "Summarize the causes and consequences of the French Revolution in five paragraphs.",
    "Write a SQL query that finds the top three customers by revenue per region, and explain it.",
]
HEX = "0123456789abcdef"


def out(msg: str) -> None:
    print(msg, flush=True)


def jout(kind: str, obj) -> None:
    out(f"RESULT json {kind} " + json.dumps(obj, separators=(",", ":")))


def log(msg: str) -> None:
    out(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ============================================================================ GPU helpers


def _load(devices: str, tp: int):
    os.environ["HIP_VISIBLE_DEVICES"] = devices
    os.environ["ENGINE_TP"] = str(tp)
    os.environ.setdefault("ENGINE_MTP", "1")
    import torch

    from engine.generator import CHECKPOINT, Generator

    t0 = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    free = [round(torch.cuda.mem_get_info(d)[0] / 2**30, 2) for d in range(torch.cuda.device_count())]
    log(f"loaded tp={tp} in {time.perf_counter() - t0:.0f}s graphs={fast.use_graphs} "
        f"graph_error={fast.graph_error} ivf={(fast.ranks[0] if hasattr(fast, 'ranks') else fast).ivf is not None} "
        f"kv_pages={fast.pool.pages} free_gb={free}")
    return g, fast, torch


def _pool_ids(tok, need: int) -> list[int]:
    """The replay's synthetic coding lines (tools/serve_probe.py --corpus synthetic)."""
    ids: list[int] = []
    i = 0
    while len(ids) < need:
        for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} 0f3a9c"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1
    return ids


def _prefill(fast, slot: int, prompt: list[int]):
    fast.reset(slot)
    body = len(prompt) - 1
    pos = 0
    while pos < body:
        end = min((pos // 2048 + 1) * 2048, body)
        fast.prefill(slot, prompt[pos:end], pos, prompt[end])
        pos = end
    return fast.prefill(slot, prompt[body:], body)


def _sync(torch) -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def _median_ms(torch, fn, iters: int = 30, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    times = []
    for _ in range(iters):
        _sync(torch)
        t0 = time.perf_counter()
        fn()
        _sync(torch)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _per_launch_ms(torch, fn, n: int = 40, reps: int = 7) -> float:
    """Median over ``reps`` of (n back-to-back launches) / n: GPU time when the host keeps up."""
    fn()
    _sync(torch)
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        _sync(torch)
        vals.append((time.perf_counter() - t0) * 1e3 / n)
    return statistics.median(vals)


def _class_split(torch, run, steps: int = 5) -> dict:
    """Per device: median span (us) and per-family time (kernel + the idle gap before it)."""
    import glue_split as GS
    from torch.profiler import ProfilerActivity, profile

    run()
    _sync(torch)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            run()
            _sync(torch)
            time.sleep(0.003)
    gev = GS.gpu_events(GS.load_trace(prof))
    res = {}
    for d, evs in gev.items():
        st = GS.split_steps(evs, steps)
        if not st:
            continue
        n = statistics.median(len(s) for s in st)
        good = [s for s in st if len(s) == n] or st
        fam: dict[str, float] = {}
        spans = []
        for s in good:
            end = s[0][0]
            for j, (ts, dur, name, _) in enumerate(s):
                gap = max(0.0, ts - end) if j else 0.0
                f = GS.family(name)
                fam[f] = fam.get(f, 0.0) + dur + gap
                end = max(end, ts + dur)
            spans.append(end - s[0][0])
        ns = len(good)
        res[str(d)] = {"kernels": int(n), "span_us": round(statistics.median(spans)),
                       "steps": ns, "family_us": {k: round(v / ns, 1) for k, v in sorted(fam.items())}}
    return res


def _capture(torch, fast, dev, fn, static_in):
    """One graph of ``fn(static_in)`` on ``dev`` (as ``FastModel._capture`` does a segment)."""
    streams = fast.__dict__.setdefault("_capture_streams", {})
    if dev not in streams:
        streams[dev] = torch.cuda.Stream(dev)
    s = streams[dev]
    with torch.cuda.device(dev):
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(static_in)
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=fast._pool(dev), stream=s,
                              capture_error_mode=fast.capture_mode or "global"):
            o = fn(static_in)
    return graph, o


def _draft(fast, h, first: bool = False):
    if fast.ivf is not None:
        return fast.ivf.draft(h, fast, first=first)
    return fast._argmax(fast.fused.gemv(h, fast.lm_head))


# ============================================================================ (a) stage


def cmd_stage(args) -> None:
    g, fast, torch = _load(args.devices, args.tp)
    tok = g.tokenizer
    plen = args.prompt_len
    ids = _pool_ids(tok, 4 * 9973 + plen + 64)
    slots = [0, 1]
    lasts = []
    with torch.inference_mode():
        for s in slots:
            lasts.append(int(torch.argmax(_prefill(fast, s, ids[9973 * s: 9973 * s + plen]))))
    ranks = getattr(fast, "ranks", None)
    if args.tp == 1:
        segs = fast.chain_segments(4)
        layer_dev = [str(l.input_layernorm.weight.device) for l in fast.layers]
        split = sum(1 for d in layer_dev if d == layer_dev[0])
        log(f"layer split: segments={len(segs)} devices={[str(s.device) for s in segs]} "
            f"ops={[len(s.ops) for s in segs]} layers_on_first={split} lm_head={fast.lm_head.device} "
            f"mtp={fast.mtp.fc.weight.device}")
        jout("tp1_layout", {"segments": len(segs), "ops": [len(s.ops) for s in segs], "split": split,
                            "layer_types": fast.layer_types})
        if len(segs) != 2:
            log("WARNING: the layer split is not two segments; per-segment numbers follow the segments")
    for spec in args.configs.split(","):
        b, k = (int(x) for x in spec.split(":"))
        groups = k + 1
        size = fast.bucket(b)
        rows = groups * size
        drafts = [ids[9973 * s + plen + 1: 9973 * s + plen + 1 + k] for s in slots[:b]]
        rec = {"tp": args.tp, "B": b, "k": k, "size": size, "rows": rows}
        try:
            with torch.inference_mode():
                call = lambda: fast.verify_chain(lasts[:b], drafts, slots[:b], [plen] * b)  # noqa: E731
                rec["call_ms"] = round(_median_ms(torch, call, args.iters), 3)
                if fast.graph_error is not None:
                    raise RuntimeError(f"graph_error {fast.graph_error}")
                if args.tp == 2:
                    gs = [m.chain_segments(groups)[0] for m in ranks]

                    def both():
                        for sg in gs:
                            with torch.cuda.device(sg.device):
                                sg.graphs[size].replay()
                    rec["replay_ms"] = round(_median_ms(torch, both, args.iters), 3)
                    rec["replay_b2b_ms"] = round(_per_launch_ms(torch, both), 3)
                    if not args.quick:
                        rec["classes"] = _class_split(torch, both)
                else:
                    s0, s1 = fast.chain_segments(groups)[:2]

                    def r0():
                        with torch.cuda.device(s0.device):
                            s0.graphs[size].replay()

                    def r1():
                        with torch.cuda.device(s1.device):
                            s1.graphs[size].replay()

                    def serial():
                        r0()
                        with torch.cuda.device(s1.device):
                            s1.inputs[size].copy_(s0.outputs[size])
                        r1()

                    def overlap():
                        r0()
                        r1()

                    rec["seg0_ms"] = round(_median_ms(torch, r0, args.iters), 3)
                    rec["seg1_ms"] = round(_median_ms(torch, r1, args.iters), 3)
                    rec["seg0_b2b_ms"] = round(_per_launch_ms(torch, r0), 3)
                    rec["seg1_b2b_ms"] = round(_per_launch_ms(torch, r1), 3)
                    rec["serial_ms"] = round(_median_ms(torch, serial, args.iters), 3)
                    rec["overlap_ms"] = round(_median_ms(torch, overlap, args.iters), 3)
                    rec["overlap_b2b_ms"] = round(_per_launch_ms(torch, overlap), 3)
                    x = s0.outputs[size]

                    def handoff():
                        with torch.cuda.device(s1.device):
                            s1.inputs[size].copy_(x)
                    rec["handoff_ms"] = round(_median_ms(torch, handoff, args.iters), 4)
                    if not args.quick:
                        rec["classes_seg0"] = _class_split(torch, r0)
                        rec["classes_seg1"] = _class_split(torch, r1)
                        rec.update(_per_op(torch, fast, groups, size, b))
            log(f"stage {json.dumps({k_: v for k_, v in rec.items() if not k_.startswith(('classes', 'op_'))})}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rec["error"] = f"{type(exc).__name__}: {exc}"
            log(f"stage B={b} k={k} error {rec['error']}")
        jout("stage", rec)
    if hasattr(fast, "timeouts"):
        out(f"RESULT tp_allreduce_timeouts={fast.timeouts()}")


def _per_op(torch, fast, groups: int, size: int, b: int) -> dict:
    """Every op of the chained step as its own graph, timed alone; one extra chain link."""
    rows = groups * size
    segs = fast.chain_segments(groups)
    ops = [(sg.device, op) for sg in segs for op in sg.ops]
    # the step's real inputs (slots at their positions), restored after capture as _run does
    saved = [(meta, meta.packed.clone()) for meta in fast.devices.values()]
    for meta in fast.devices.values():
        fast._fill_dummy(meta, size, rows, False, groups)
    x = fast.devices[fast.embed.device].tokens[:rows]
    graphs = []
    keep = []  # every graph's static input and output stay alive (one shared memory pool)
    try:
        for dev, op in ops:
            meta = fast.devices[dev]
            static_in = x if x.device == dev else x.to(dev).clone()
            gr, o = _capture(torch, fast, dev, lambda t, op=op, meta=meta: op(t, meta), static_in)
            graphs.append((dev, gr))
            keep.append((static_in, o))
            x = o
        for meta, packed in saved:
            meta.packed.copy_(packed)
        times = []
        for dev, gr in graphs:
            def run(dev=dev, gr=gr):
                with torch.cuda.device(dev):
                    gr.replay()
            times.append(round(_per_launch_ms(torch, run, n=30, reps=5), 4))
        # one more chain link on B rows (size rows: the bucket), as the chain head's loop runs it
        hdev = fast.lm_head.device
        meta = fast.devices[hdev]
        with torch.cuda.device(hdev):
            h0 = torch.zeros((size, fast.config.hidden_size), dtype=fast.embed.dtype, device=hdev)
            h0.normal_()
            d0 = torch.zeros((size,), dtype=torch.long, device=hdev)
            pos = meta.positions[:size].clone() + groups  # the chain's next link position

        def link(h):
            z = fast._fused_layer(fast.mtp_index, fast._mtp_in(h, d0, True), meta.slots[:size], pos, meta)
            y = fast._mtp_norm(z)
            return _draft(fast, y)

        lg, _ = _capture(torch, fast, hdev, link, h0)

        def rl():
            with torch.cuda.device(hdev):
                lg.replay()
        link_ms = _per_launch_ms(torch, rl, n=30, reps=5)
        del lg
    finally:
        for meta, packed in saved:
            meta.packed.copy_(packed)
        graphs.clear()
        keep.clear()
        torch.cuda.empty_cache()
    devs = [str(d) for d, _ in ops]
    return {"op_ms": times, "op_dev": devs, "op_n": len(ops), "link_ms": round(link_ms, 4)}


# ============================================================================ (b) deep


def cmd_deep(args) -> None:
    g, fast, torch = _load(args.devices, 1)
    tok = g.tokenizer
    if fast.graph_error is not None or not fast.chain:
        raise SystemExit(f"no chained verify graphs on the layer split: {fast.graph_error}")
    pool = _pool_ids(tok, 9973 * 8 + args.replay_len + 64)
    corpora = {
        "replay": [pool[9973 * k + 50: 9973 * k + 50 + args.replay_len] for k in range(args.texts)],
        "natural": [],
    }
    for k in range(args.texts):
        msg = [{"role": "user", "content": NATURAL[k % len(NATURAL)]}]
        s = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        corpora["natural"].append(tok.encode(s, add_special_tokens=False))
    L = args.links
    for name, prompts in corpora.items():
        B = len(prompts)
        slots = list(range(B))
        t0 = time.perf_counter()
        # (1) the greedy text through the served chained verify (k = 3, graphs, IVF on)
        with torch.inference_mode():
            first = [int(torch.argmax(_prefill(fast, s, p))) for s, p in zip(slots, prompts)]
        gen = [[f] for f in first]
        toks = list(first)
        pos = [len(p) for p in prompts]
        drafts = [[f] * args.k for f in first]
        steps = acc_sum = 0
        hist = [0] * (args.k + 1)
        with torch.inference_mode():
            while min(len(x) for x in gen) < args.gen + L + 2:
                _, picks, acc, nxt = fast.verify_chain(toks, drafts, slots, pos)
                for b in range(B):
                    a = acc[b]
                    gen[b].extend(picks[b][: a + 1])
                    fast.advance(slots[b], a)
                    pos[b] += a + 1
                    toks[b] = picks[b][a]
                    drafts[b] = nxt[b]
                    if steps > 0:
                        hist[a] += 1
                        acc_sum += a
                steps += 1
        n = sum(hist)
        log(f"deep {name}: greedy {args.gen}+ tokens x {B} in {time.perf_counter() - t0:.0f}s, served k={args.k} "
            f"steps={steps} mean_accepted={acc_sum / max(n, 1):.3f} hist={hist}")
        jout("served_acc", {"corpus": name, "k": args.k, "hist": hist})
        # (2) teacher-forced: the head's chain of L drafts at every anchor
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s, p in zip(slots, prompts):
                _prefill(fast, s, p)
        segs = fast.chain_segments(2)
        ops = [(sg.device, op) for sg in segs for op in sg.ops]
        hdev = fast.lm_head.device
        meta = fast.devices[hdev]
        M = [[] for _ in slots]
        pick_bad = 0
        depth_ok = [[0, 0] for _ in range(L)]  # [checked, accepted] per depth (conditional)
        with torch.inference_mode():
            for i in range(args.gen):
                base = [len(p) + i for p in prompts]
                rows_tok = [gen[b][i] for b in slots]
                nxt_tok = [[gen[b][i + 1]] for b in slots]
                for s, q in zip(slots, base):
                    fast._ensure_pages(s, q + L + 4)
                host = fast._chain_host(rows_tok, nxt_tok, slots, base, 2, B)
                for m in fast.devices.values():
                    m.packed.copy_(host)
                x = fast.devices[fast.embed.device].tokens[: 2 * B]
                for dev, op in ops[:-1]:
                    with torch.cuda.device(dev):
                        x = op(x.to(dev), fast.devices[dev])
                with torch.cuda.device(hdev):
                    x = x.to(hdev)
                    logits = fast._logits(x, verify=True)
                    picks = fast._argmax(logits)
                    rows = 2 * B
                    z = fast._fused_layer(fast.mtp_index, fast._mtp_in(x, picks, True), meta.slots[:rows],
                                          meta.positions[:rows], meta)
                    y = fast._mtp_norm(z)
                    h = y[:B].contiguous()
                    d = _draft(fast, h, first=True)
                    chain = [d]
                    for j in range(1, L):
                        z = fast._fused_layer(fast.mtp_index, fast._mtp_in(h, d, True), meta.slots[:B],
                                              meta.positions[:B] + j, meta)
                        h = fast._mtp_norm(z)
                        d = _draft(fast, h)
                        chain.append(d)
                    ch = torch.stack(chain).cpu().tolist()  # [L, B]
                    pk = picks[:B].cpu().tolist()
                for b in slots:
                    if pk[b] != gen[b][i + 1]:
                        pick_bad += 1
                    # anchor m = i + 1 (the bonus); link j predicts gen[m + j]
                    mm = 0
                    for j in range(L):
                        depth_ok[j][0] += 1
                        if ch[j][b] == gen[b][i + 1 + j + 1]:
                            depth_ok[j][1] += 1
                            mm += 1
                        else:
                            break
                    M[b].append(mm)
        n = sum(len(x) for x in M)
        cum = [sum(1 for x in M for v in x if v >= j) / n for j in range(1, L + 1)]
        cond = [round(a / c, 4) if c else None for c, a in depth_ok]
        log(f"deep {name}: teacher-forced {args.gen} anchors x {B} in {time.perf_counter() - t0:.0f}s, "
            f"pick mismatches vs greedy {pick_bad}/{n}")
        out(f"RESULT deep {name} cond_acc_by_depth={cond}")
        out(f"RESULT deep {name} P(M>=j) j=1..{L}: {[round(c, 4) for c in cum]}")
        for k in (1, 3, 5, 7):
            if 2 * k + 1 <= L:
                full = sum(1 for x in M for v in x if v >= k) / n
                spec_valid = sum(1 for x in M for v in x if v >= k + 1) / n
                spec_full = sum(1 for x in M for v in x if v >= 2 * k + 1) / n
                out(f"RESULT deep {name} k={k} P(chunk full, M>=k)={full:.4f} "
                    f"P(next chunk valid, M>=k+1)={spec_valid:.4f} P(next chunk full, M>=2k+1)={spec_full:.4f} "
                    f"P(depths k+1..2k+1 | M>=k)={spec_full / max(full, 1e-9):.4f}")
        jout("deep", {"corpus": name, "links": L, "pick_bad": pick_bad, "cond": cond,
                      "M": ["".join(HEX[min(v, 15)] for v in x) for x in M]})
        with torch.inference_mode():
            for s in slots:
                fast.reset(s)


# ============================================================================ (c) mem


def cmd_mem(args) -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = args.devices
    os.environ["ENGINE_TP"] = "2"
    os.environ.setdefault("ENGINE_MTP", "1")
    import torch

    from engine.fast import mfma
    from engine.fast import tp as TP
    from engine.fast.model import PAGE

    keep: list = []
    info: dict = {}
    orig = TP.shard_model

    def tensors(obj, seen):
        if isinstance(obj, torch.Tensor):
            if obj.numel() and id(obj) not in seen:
                seen.add(id(obj))
                yield obj
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from tensors(v, seen)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                yield from tensors(v, seen)

    def hooked(lm, mtp, devices):
        shards = orig(lm, mtp, devices)
        torch.cuda.empty_cache()
        info["free_tp2_weights_gb"] = [round(torch.cuda.mem_get_info(d)[0] / 2**30, 2) for d in devices]
        layers = [s[0].model.language_model.layers for s in shards]
        per_layer = [sum(t.numel() * t.element_size() for t in tensors(layers[1][i], set()))
                     for i in range(len(layers[0]))]
        info["half_layer_gb"] = [round(x / 2**30, 3) for x in per_layer]
        split = args.split
        extra = [0, 0]
        for i in range(len(layers[0])):
            dst, src = (0, 1) if i < split else (1, 0)
            for t in tensors(layers[src][i], set()):
                keep.append(t.to(devices[dst], copy=True))
                extra[dst] += t.numel() * t.element_size()
        if shards[0][1] is not None:  # the MTP layer in full on GCD1
            for t in tensors(shards[0][1].layers, set()):
                keep.append(t.to(devices[1], copy=True))
                extra[1] += t.numel() * t.element_size()
        # a second set of GDN state rings (full heads of the stage's GDN layers)
        cfg = shards[0][0].model.language_model.config
        rows = int(os.environ.get("ENGINE_SLOTS", "14")) + 1
        gdn = [0, 0]
        for i, kind in enumerate(cfg.layer_types):
            if kind != "linear_attention":
                continue
            m = layers[0][i].linear_attn
            per = mfma.RING * rows * (2 * m.conv_dim * m.conv_kernel_size * 2
                                      + 2 * m.num_v_heads * m.head_k_dim * m.head_v_dim * 4)
            dst = 0 if i < split else 1
            keep.append(torch.empty(per, dtype=torch.uint8, device=devices[dst]))
            gdn[dst] += per
        torch.cuda.empty_cache()
        info.update(split=split, extra_weights_gb=[round(x / 2**30, 2) for x in extra],
                    dup_gdn_state_gb=[round(x / 2**30, 2) for x in gdn],
                    free_both_layouts_gb=[round(torch.cuda.mem_get_info(d)[0] / 2**30, 2) for d in devices])
        log(f"mem after both layouts' weights: {json.dumps(info)}")
        return shards

    TP.shard_model = hooked
    from engine.generator import CHECKPOINT, Generator

    t0 = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    r0 = fast.ranks[0]
    kv_page_layer = None
    for k_, v_ in r0.kv.values():
        kv_page_layer = (k_[0].numel() * k_.element_size()) * 2
        break
    info.update(
        load_s=round(time.perf_counter() - t0), graph_error=fast.graph_error, kv_pages=r0.pool.pages,
        kv_tokens=r0.pool.pages * PAGE, need_pages_262k=r0.max_pages + 1, attn_layers_per_gcd=len(r0.kv),
        kv_bytes_per_page_layer_gcd=kv_page_layer,
        kv_gb=[round(sum(k.numel() * k.element_size() * 2 for k, _ in m.kv.values()) / 2**30, 2) for m in fast.ranks],
        gdn_state_gb=[round(sum(c.numel() * c.element_size() + r.numel() * r.element_size()
                                for c, r in m.gdn_state.values()) / 2**30, 2) for m in fast.ranks],
        free_after_load_gb=[round(torch.cuda.mem_get_info(d)[0] / 2**30, 2) for d in range(torch.cuda.device_count())],
        reserved_gb=[round(torch.cuda.memory_reserved(d) / 2**30, 2) for d in range(torch.cuda.device_count())],
        allocated_gb=[round(torch.cuda.memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())],
        env={k: os.environ.get(k) for k in ("ENGINE_SLOTS", "ENGINE_KV_FRACTION", "ENGINE_RESERVE_GB",
                                            "ENGINE_KV_SEQS", "ENGINE_SPEC_KMAX")},
    )
    jout("mem", info)


# ============================================================================ (d) project


def _sim_tp2(M: list[int], k: int) -> tuple[int, int]:
    """Serial spec decode along one text: (tokens, steps)."""
    m, tokens, steps = 0, 0, 0
    n = len(M)
    while m + 2 * k + 2 < n:
        a = min(M[m], k)
        tokens += a + 1
        m += a + 1
        steps += 1
    return tokens, steps


def _sim_pp(M: list[int], k: int) -> tuple[int, int, int, int]:
    """The two-chunk pipeline along one text: (tokens, ticks, flushes, chunks verified).
    M[m]: leading correct drafts of the chain drafted at bonus position m (link j -> m + j).
    A chunk is (p, anchor): rows at p .. p + k, row q's token the bonus if q == anchor, else
    the anchor's link q - anchor."""
    n = len(M)
    m = 0
    s1 = None  # in stage 1 this tick
    s2 = None  # in stage 2 this tick
    tokens = ticks = flushes = chunks = 0
    while m + 2 * k + 2 < n:
        ticks += 1
        full = False
        if s2 is not None:
            p, anc = s2
            off = p - anc
            valid = p == m and (off == 0 or M[anc] >= off)
            if valid:
                a = min(k, M[anc] - off)
                tokens += a + 1
                m = p + a + 1
                full = a == k
                chunks += 1
        # the chunk in stage 1 goes on to stage 2 only if it starts at the new bonus and its
        # first row's token is that bonus
        nxt2 = None
        if s1 is not None:
            p, anc = s1
            off = p - anc
            if (s2 is None or full) and p == m and (off == 0 or M[anc] >= off):
                nxt2 = s1
            else:
                flushes += 1
        if nxt2 is not None:
            p = nxt2[0] + k + 1
            if p - m + k > 15:
                nxt1 = None
            else:
                nxt1 = (p, m)
        else:
            nxt1 = (m, m)
        s2, s1 = nxt2, nxt1
    return tokens, ticks, flushes, chunks


def cmd_project(args) -> None:
    recs: dict[str, list] = {}
    for path in args.logs:
        for line in Path(path).read_text(errors="replace").splitlines():
            mt = re.search(r"RESULT json (\w+) (\{.*\})\s*$", line)
            if mt:
                try:
                    recs.setdefault(mt.group(1), []).append(json.loads(mt.group(2)))
                except json.JSONDecodeError:
                    pass
    stage = {(r["tp"], r["B"], r["k"]): r for r in recs.get("stage", []) if "error" not in r}
    deep = {r["corpus"]: [[HEX.index(c) for c in s] for s in r["M"]] for r in recs.get("deep", [])}
    layout = (recs.get("tp1_layout") or [None])[0]
    out(f"RESULT project inputs: stage configs={sorted(stage)} corpora={sorted(deep)}")
    rows_out = []
    kill_stage = {}
    best = {}
    for (tp, b, k), r1 in sorted(stage.items()):
        if tp != 1:
            continue
        r2 = stage.get((2, b, k))
        if r2 is None:
            continue
        tp2 = r2.get("replay_b2b_ms") or r2["replay_ms"]
        s1 = r1.get("seg0_b2b_ms") or r1["seg0_ms"]
        s2 = r1.get("seg1_b2b_ms") or r1["seg1_ms"]
        link = r1.get("link_ms", 0.0)
        hand = r1.get("handoff_ms", 0.0)
        s2pp = s2 + (k + 1) * link
        # balanced split from the per-op times, scaled to the measured segment times
        bal = None
        if "op_ms" in r1 and layout:
            ops, devs = r1["op_ms"], r1["op_dev"]
            d0 = devs[0]
            sc0 = s1 / max(sum(t for t, d in zip(ops, devs) if d == d0), 1e-9)
            sc1 = s2 / max(sum(t for t, d in zip(ops, devs) if d != d0), 1e-9)
            # ops: embed, layers..., norm, head
            emb, lay, tail = ops[0], ops[1:-2], ops[-2:]
            ldev = devs[1:-2]
            # per-layer cost as the GCD it ran on measured it (scale of that GCD)
            lay_s = [t * (sc0 if d == d0 else sc1) for t, d in zip(lay, ldev)]
            emb_s = emb * sc0
            tail_s = sum(tail) * sc1
            best_s = None
            for sp in range(1, len(lay)):
                st1 = emb_s + sum(lay_s[:sp])
                st2 = sum(lay_s[sp:]) + tail_s + (k + 1) * link
                if best_s is None or max(st1, st2) < max(best_s[1], best_s[2]):
                    best_s = (sp, st1, st2)
            bal = {"split": best_s[0], "stage1": round(best_s[1], 3), "stage2": round(best_s[2], 3),
                   "scale": [round(sc0, 3), round(sc1, 3)]}
        tick_now = max(s1, s2pp) + hand
        tick_bal = (max(bal["stage1"], bal["stage2"]) + hand) if bal else tick_now
        big = max(s1, s2pp) if not bal else max(bal["stage1"], bal["stage2"])
        kill_stage[(b, k)] = big / tp2
        out(f"RESULT project stage B={b} k={k} tp2_ms={tp2:.3f} stage1={s1:.3f} stage2={s2:.3f} "
            f"link={link:.3f} stage2+links={s2pp:.3f} handoff={hand:.4f} tick_as_split={tick_now:.3f} "
            f"balanced={bal} tick_bal={tick_bal:.3f} larger_stage/tp2={big / tp2:.3f} "
            f"(without extra links {max(s1, s2) / tp2:.3f}) overlap_measured={r1.get('overlap_b2b_ms')}")
        for corpus, Ms in sorted(deep.items()):
            t2 = st2 = tp_ = tk = fl = ch = 0
            for M in Ms:
                a, s = _sim_tp2(M, k)
                t2 += a
                st2 += s
                a, t, f, c = _sim_pp(M, k)
                tp_ += a
                tk += t
                fl += f
                ch += c
            tpms2 = t2 / st2 / tp2 * b
            tpms_now = tp_ / tk / tick_now * b
            tpms_bal = tp_ / tk / tick_bal * b
            rec = {"B": b, "k": k, "corpus": corpus, "tp2_tok_per_step": round(t2 / st2, 3),
                   "pp_tok_per_tick": round(tp_ / tk, 3), "flush_per_chunk": round(fl / max(ch, 1), 4),
                   "tp2_tok_ms": round(tpms2, 4), "pp_tok_ms_as_split": round(tpms_now, 4),
                   "pp_tok_ms_balanced": round(tpms_bal, 4), "ratio_as_split": round(tpms_now / tpms2, 3),
                   "ratio_balanced": round(tpms_bal / tpms2, 3)}
            rows_out.append(rec)
            out("RESULT project " + " ".join(f"{k_}={v}" for k_, v in rec.items()))
            bk = (b, corpus)
            best.setdefault(bk, {"tp2": (0, None), "pp": (0, None)})
            if tpms2 > best[bk]["tp2"][0]:
                best[bk]["tp2"] = (tpms2, k)
            if tpms_bal > best[bk]["pp"][0]:
                best[bk]["pp"] = (tpms_bal, k)
    for (b, corpus), v in sorted(best.items()):
        r = v["pp"][0] / max(v["tp2"][0], 1e-12)
        out(f"RESULT project best B={b} corpus={corpus} tp2 best k={v['tp2'][1]} {v['tp2'][0]:.4f} tok/ms, "
            f"pipeline best k={v['pp'][1]} {v['pp'][0]:.4f} tok/ms (balanced split) -> {r:.3f}x")
    # kill rules
    gain_ok = any(v["pp"][0] / max(v["tp2"][0], 1e-12) >= 1.15 for (b, c), v in best.items() if c == "replay")
    stage_ok = {bk: x <= 0.85 for bk, x in kill_stage.items()}
    out(f"RESULT kill gain>=1.15x at 1 or 2 slots on replay-like: {'PASS' if gain_ok else 'KILL'}")
    out(f"RESULT kill larger stage <= 0.85 x tp2 step by (B,k): "
        f"{ {f'{b}:{k}': round(x, 3) for (b, k), x in kill_stage.items()} } -> "
        f"{'PASS' if any(stage_ok.values()) else 'KILL'} (any) / {'PASS' if stage_ok and all(stage_ok.values()) else 'KILL'} (all)")
    for r in recs.get("mem", []):
        fb = r.get("free_both_layouts_gb")
        out(f"RESULT kill dual layout free per GCD (weights of both + dup GDN rings, before KV) {fb} GB "
            f"-> {'PASS' if fb and min(fb) >= 8 else 'KILL'}; kv_pages={r.get('kv_pages')} "
            f"need_262k={r.get('need_pages_262k')} free_after_load={r.get('free_after_load_gb')}")


# ============================================================================ main


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage")
    a.add_argument("--tp", type=int, required=True)
    a.add_argument("--devices", default="0,1")
    a.add_argument("--configs", default="1:1,1:3,1:5,1:7,2:1,2:3,2:5,2:7")
    a.add_argument("--prompt-len", type=int, default=1800)
    a.add_argument("--iters", type=int, default=40)
    a.add_argument("--quick", action="store_true")
    d = sub.add_parser("deep")
    d.add_argument("--devices", default="2,3")
    d.add_argument("--texts", type=int, default=8)
    d.add_argument("--replay-len", type=int, default=1800)
    d.add_argument("--gen", type=int, default=320)
    d.add_argument("--links", type=int, default=15)
    d.add_argument("--k", type=int, default=3)
    m = sub.add_parser("mem")
    m.add_argument("--devices", default="4,5")
    m.add_argument("--split", type=int, default=22)
    p = sub.add_parser("project")
    p.add_argument("logs", nargs="+")
    args = ap.parse_args()
    {"stage": cmd_stage, "deep": cmd_deep, "mem": cmd_mem, "project": cmd_project}[args.cmd](args)


if __name__ == "__main__":
    main()
