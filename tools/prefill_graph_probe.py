"""Plan 33 step 0: can the fused TP2 prefill pass be captured as a HIP graph, and how much of
its wall does replay remove?

    HIP_VISIBLE_DEVICES=0,1 python tools/prefill_graph_probe.py [--news 61,561,1650]
        [--start 3600] [--reps 12] [--ll-rows 64,561,1650,2600]

Loads the checkpoint as a replica does (``Generator.load``, ENGINE_TP=2 over the two visible
GCDs, warmup and decode/verify graphs included) with the in-graph all-reduce buffer
(``tp.CAPACITY``) raised before the group is built, so one LL all-reduce can carry a whole
prefill pass's hidden state (``--capacity-log2``, default 2^22 32-bit words = 4,096 rows of
2,048 BF16). Then one slot gets a ``--start``-token context of synthetic ids (two prefill
chunks) and, per N in ``--news``, the resumed pass of N new tokens at ``--start`` is run as:

- A, eager (the head's path): ``TPModel.prefill``, both ranks in their host threads, every
  all-reduce a host rendezvous (``Group._exchange``) plus a peer copy;
- B, eager with the LL all-reduce: the same pass with every all-reduce (``_reduce_rms``'s sum,
  ``_reduce``, the greedy pick's) through the in-graph kernel (``_LL._launch``) launched
  eagerly, so the ranks meet on the device and never on the host. ``_reduce_rms`` then adds
  the LL sum as its ``h`` (no peer): the same fp32 sum rounded to BF16 then the residual add;
- C, graph replay: per rank, the pass B captured once in a CUDA/HIP graph (rank 0 on this
  thread, rank 1 on the TP worker thread, one after the other, as ``TPModel.capture``), then
  both graphs replayed from this thread.

The probe captures a FIXED (N, start): no bucketing, the ids, page table and start baked in.
That is valid for a probe that replays the same inputs. The pass builds small index tensors
on the host (``torch.tensor(..., device=)``, ``_PackPlan.on``'s ``.to(dev)``); a capture
cannot hold a pageable host copy, so they are staged: during B's first run every host->device
copy is recorded (keyed by device, dtype, shape and bytes), and during capture the same copy
returns the recorded device tensor instead (the graph reads it from memory). Nothing else in
the engine changes. Any copy missing from the record fails the capture loudly.

Between runs the slot's Gated DeltaNet rows (conv and recurrent, both ranks) are restored from
a copy taken after the context; the pass rewrites the same KV positions with the same values.

Per N: wall (median of ``--reps`` synchronized runs) of A, B, C, and C's host launch time;
replay/eager ratio; outputs (gathered fp32 logits, MTP draft, every GDN conv/rec row, the KV
written at start..start+N of every attention layer incl. the MTP layer, both ranks) of B vs
A, C vs B, C vs A, C vs C (replay twice), A vs A: bitwise, else max |diff| and top-1.

Then, for step 1's padding design (rank 0's weights, eager, on GCD 0), real rows of N vs the
same rows padded to the bucket (64 / 640 / 1,792): every 2-D weight GEMM of one GDN layer, one
attention layer, the shared expert, router, lm_head and MTP fc; the fused MoE block; fla's
chunked delta rule with pad rows of zero q, k, v, g, beta (output rows and final state); the
GDN conv (output rows, state); prefill attention (query rows).

Last, the LL all-reduce alone, in a graph of 20 back-to-back calls per rank at
``--ll-rows`` rows x 2,048 BF16, against the eager host rendezvous at the same size.

Kill (plan 33): replay/eager at 561 > 0.70, or capture needs a change to what the model
computes, or one LL all-reduce at 561 rows > 150 us.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

KILL_RATIO = 0.70
KILL_AR_US = 150.0
BUCKET = {61: 64, 561: 640, 1650: 1792}


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


# ============================================================================ H2D staging


class Staging:
    """Record host->device copies during a warm pass; hand back the same device tensors during
    capture (a pageable copy cannot be captured)."""

    def __init__(self, torch) -> None:
        self.torch = torch
        self.mode: str | None = None  # None, "record", "capture"
        self.cache: dict[tuple, object] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses: list[str] = []
        self._orig_tensor = torch.tensor
        self._orig_to = torch.Tensor.to

    def _dev(self, dev):
        torch = self.torch
        if isinstance(dev, int):
            return torch.device("cuda", dev)
        dev = torch.device(dev)
        if dev.type == "cuda" and dev.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return dev

    def h2d(self, cpu, dev, extra: str, args, kwargs):
        dev = self._dev(dev)
        if dev.type != "cuda":
            return self._orig_to(cpu, *args, **kwargs)
        flat = cpu.detach().contiguous().reshape(-1)
        raw = flat.view(self.torch.uint8).numpy().tobytes() if flat.numel() else b""
        key = (str(dev), str(cpu.dtype), tuple(cpu.shape), extra, hashlib.sha1(raw).hexdigest())
        if self.mode == "record":
            out = self._orig_to(cpu, *args, **kwargs)
            with self.lock:
                self.cache.setdefault(key, out.clone())
            return out
        hit = self.cache.get(key)
        if hit is None:
            self.misses.append(f"{key[:4]} at {traceback.format_stack(limit=4)[0].strip()}")
            raise RuntimeError(f"unstaged host->device copy during capture: {key[:4]}")
        self.hits += 1
        return hit

    def install(self) -> None:
        torch, st = self.torch, self
        orig_tensor, orig_to = self._orig_tensor, self._orig_to

        def tensor(data, *args, **kwargs):
            dev = kwargs.get("device")
            if st.mode is None or dev is None:
                return orig_tensor(data, *args, **kwargs)
            kw = {k: v for k, v in kwargs.items() if k != "device"}
            cpu = orig_tensor(data, *args, **kw)
            return st.h2d(cpu, dev, "tensor" + repr(args) + repr(sorted(kw.items())), (st._dev(dev),), {})

        def to(self_, *args, **kwargs):
            if st.mode is None or self_.device.type != "cpu":
                return orig_to(self_, *args, **kwargs)
            dev = kwargs.get("device")
            rest = args
            if dev is None and args and (isinstance(args[0], (torch.device, str)) or type(args[0]) is int):
                dev, rest = args[0], args[1:]
            if dev is None:
                return orig_to(self_, *args, **kwargs)
            extra = "to" + repr(rest) + repr(sorted((k, v) for k, v in kwargs.items() if k != "device"))
            return st.h2d(self_, dev, extra, args, kwargs)

        torch.tensor = tensor
        torch.Tensor.to = to


# ============================================================================ main probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="61,561,1650")
    ap.add_argument("--start", type=int, default=3600)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--capacity-log2", type=int, default=22)
    ap.add_argument("--ll-rows", default="64,561,1650,2600")
    ap.add_argument("--ll-calls", type=int, default=20)
    args = ap.parse_args()
    news = [int(n) for n in args.news.split(",")]
    start = args.start

    os.environ.setdefault("ENGINE_TP", "2")
    import torch
    import torch.nn.functional as F

    from engine.fast import model as fm
    from engine.fast import prefill_kernels as pk
    from engine.fast import tp as tpmod

    tpmod.CAPACITY = 1 << args.capacity_log2  # before any Group (and its LL buffers) exists
    from engine.generator import CHECKPOINT, Generator
    from serve_probe import token_pool

    print(f"probe: visible={os.environ.get('HIP_VISIBLE_DEVICES')} torch {torch.__version__} "
          f"hip {torch.version.hip} gpus {torch.cuda.device_count()} CAPACITY={tpmod.CAPACITY} words "
          f"PREFILL_FUSED={os.environ.get('ENGINE_PREFILL_FUSED', '1(default)')}", flush=True)
    t0 = time.perf_counter()
    gen = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
        prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30,
    )
    fast = gen.fast
    assert isinstance(fast, tpmod.TPModel), f"not a TPModel: {type(fast)}"
    r0, r1 = fast.ranks
    assert r0.triton_kernels["prefill_fused"], "fused prefill (plan 26) is off"
    result("load", seconds=round(time.perf_counter() - t0, 1), mtp=fast.mtp is not None,
           graphs=fast.use_graphs, graph_error=fast.graph_error, fla_pins=r0.fla_pins,
           ll=fast.group.ll is not None)
    devs = list(fast.devices_tp)

    def sync() -> None:
        for d in devs:
            torch.cuda.synchronize(d)

    # ------------------------------------------------------------ patches (probe only)
    FORCE = [False]
    group = fast.group
    orig_allreduce = tpmod.Group.allreduce
    orig_reduce_rms = fm.FastModel._reduce_rms

    def allreduce(self, rank, x, residual=None):
        if FORCE[0] and not self.solo and x.is_cuda and not torch.cuda.is_current_stream_capturing():
            return self.ll._launch(rank, x.contiguous(), residual)
        return orig_allreduce(self, rank, x, residual)

    def reduce_rms(self, h, residual, weight):
        if (self.tp is not None and not self.tp.solo
                and (FORCE[0] or torch.cuda.is_current_stream_capturing())):
            s = self.tp.ll._launch(self.rank, h.contiguous(), None)
            return pk.reduce_rms(residual, s, None, weight, self.eps)
        return orig_reduce_rms(self, h, residual, weight)

    tpmod.Group.allreduce = allreduce
    fm.FastModel._reduce_rms = reduce_rms
    stage = Staging(torch)
    stage.install()

    # ------------------------------------------------------------ context
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], start + max(news) + 4000, "synthetic")
    slot = 0
    fast.reset(slot)
    ctx = pool[:start]
    with torch.inference_mode():
        for lo in range(0, start, 2048):
            fast.prefill(slot, ctx[lo: min(start, lo + 2048)], lo)
        # Pages for the largest pass and its padded bucket, allocated once: the page table
        # every capture bakes in stays valid.
        r0._ensure_pages(slot, start + max(max(news), max(BUCKET.values())) + 64)
    sync()
    saved = fast._gdn_copies(slot)
    PAGE = fm.PAGE

    def restore() -> None:
        fast._gdn_load(slot, saved)

    def dump(n: int, logits, drafts) -> dict[str, torch.Tensor]:
        out = {"logits": logits.float().cpu(), "drafts": drafts.cpu()}
        for r, m in enumerate(fast.ranks):
            row = m._g(slot)
            for i, (conv, rec) in m.gdn_state.items():
                out[f"r{r}.gdn{i}.conv"] = conv[row].cpu()
                out[f"r{r}.gdn{i}.rec"] = rec[row].cpu()
            table = torch.tensor(m.slot_pages[slot], dtype=torch.long)
            pos = torch.arange(start, start + n)
            page, off = table[pos // PAGE], pos % PAGE
            for i, (k_pool, v_pool) in m.kv.items():
                out[f"r{r}.kv{i}.k"] = k_pool[page.to(k_pool.device), :, off.to(k_pool.device)].cpu()
                out[f"r{r}.kv{i}.v"] = v_pool[page.to(v_pool.device), :, off.to(v_pool.device)].cpu()
        return out

    def compare(a: dict, b: dict) -> dict:
        diff_keys, maxd = [], 0.0
        for k in a:
            if not torch.equal(a[k], b[k]):
                diff_keys.append(k)
                if a[k].is_floating_point():
                    maxd = max(maxd, float((a[k].float() - b[k].float()).abs().max()))
        la, lb = a["logits"], b["logits"]
        return {
            "bitwise": not diff_keys,
            "tensors": len(a),
            "differing": len(diff_keys),
            "differing_kinds": sorted({k.split(".")[-1] if "." in k else k for k in diff_keys}),
            "first_differing": diff_keys[:6],
            "max_abs_diff": maxd,
            "logits_max_abs_diff": float((la - lb).abs().max()),
            "top1_equal": int(la.argmax()) == int(lb.argmax()),
            "drafts_equal": torch.equal(a["drafts"], b["drafts"]),
        }

    def eager(ids, ll: bool):
        FORCE[0] = ll
        try:
            logits = fast.prefill(slot, ids, start)
            return logits, fast.last_drafts.clone()
        finally:
            FORCE[0] = False

    streams = {d: torch.cuda.Stream(d) for d in devs}

    def on_side(fn):
        def run(m):
            dev = m.embed.device
            s = streams[dev]
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                out = fn(m)
            torch.cuda.current_stream(dev).wait_stream(s)
            return out
        return run

    summary = {}
    for n in news:
        ids = pool[start + 7 * n: start + 8 * n]  # new tokens (not a repeat of the context)
        ids = pool[start: start + n] if len(ids) < n else ids
        rec: dict = {"new": n, "start": start}
        try:
            # Warm both eager paths (and record the pass's host->device copies on B's run).
            with torch.inference_mode():
                for ll in (False, True, False, True):
                    restore()
                    eager(ids, ll)
                sync()
                restore()
                stage.mode = "record"
                FORCE[0] = True
                fast._both(on_side(lambda m: m.prefill(slot, ids, start)))
                FORCE[0] = False
                stage.mode = None
                sync()
                restore()
            # Capture, rank 0 here then rank 1 on the TP worker thread, as TPModel.capture.
            graphs, outs, drafts = {}, {}, {}
            mem0 = [torch.cuda.memory_reserved(d) for d in devs]
            cap_err = None
            cap_t = time.perf_counter()
            for mode in ("global", "thread_local", "relaxed"):
                stage.mode = "capture"
                try:
                    def cap(m, mode=mode):
                        dev = m.embed.device
                        s = streams[dev]
                        s.wait_stream(torch.cuda.current_stream(dev))
                        torch.cuda.synchronize(dev)
                        g = torch.cuda.CUDAGraph()
                        with torch.inference_mode(), torch.cuda.device(dev):
                            with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(), stream=s,
                                                  capture_error_mode=mode):
                                out = m.prefill(slot, ids, start, None)
                                dr = m.last_drafts
                        return g, out, dr
                    g0 = cap(r0)
                    g1 = fast._pool.submit(fast._on_rank1, lambda: cap(r1)).result()
                    for r, (g, o, d) in enumerate((g0, g1)):
                        graphs[r], outs[r], drafts[r] = g, o, d
                    rec["capture_mode"] = mode
                    cap_err = None
                    break
                except Exception as exc:  # noqa: BLE001
                    where = [f"{Path(f.filename).name}:{f.lineno} {f.name}"
                             for f in traceback.extract_tb(exc.__traceback__)]
                    cap_err = f"{mode}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''} at {where[-3:]}"
                    print(f"capture N={n}: {cap_err}", flush=True)
                    try:
                        sync()
                    except Exception:  # noqa: BLE001
                        pass
                finally:
                    stage.mode = None
            rec["capture_s"] = round(time.perf_counter() - cap_t, 2)
            rec["capture_error"] = cap_err
            rec["staged_hits"] = stage.hits
            if cap_err is not None:
                result("case", **rec)
                summary[n] = rec
                continue
            sync()
            rec["graph_mem_mb"] = [round((torch.cuda.memory_reserved(d) - m0) / 2**20) for d, m0 in zip(devs, mem0)]

            def replay():
                for r, d in enumerate(devs):
                    with torch.cuda.device(d):
                        graphs[r].replay()
                return fast._gather([outs[0], outs[1]])

            # --- exactness (one run each, from the same restored state)
            with torch.inference_mode():
                to0 = fast.timeouts()
                restore(); la, da = eager(ids, False); sync(); A = dump(n, la, da)
                restore(); lb, db = eager(ids, True); sync(); B = dump(n, lb, db)
                restore(); lc = replay(); sync(); C = dump(n, lc, drafts[0])
                restore(); lc2 = replay(); sync(); C2 = dump(n, lc2, drafts[0])
                restore(); la2, da2 = eager(ids, False); sync(); A2 = dump(n, la2, da2)
                rec["exact"] = {
                    "A_vs_A": compare(A, A2), "B_vs_A": compare(B, A), "C_vs_B": compare(C, B),
                    "C_vs_A": compare(C, A), "C_vs_C": compare(C, C2),
                }
                del A, B, C, C2, A2
                # --- timing
                walls = {"A_eager": [], "B_eager_ll": [], "C_replay": []}
                launch = []
                for _ in range(args.reps):
                    for key in walls:
                        restore()
                        sync()
                        a = time.perf_counter()
                        if key == "C_replay":
                            replay()
                            launch.append((time.perf_counter() - a) * 1e3)
                        else:
                            eager(ids, key == "B_eager_ll")
                        sync()
                        walls[key].append((time.perf_counter() - a) * 1e3)
                rec["ll_timeouts"] = fast.timeouts() - to0
            med = {k: round(statistics.median(v), 2) for k, v in walls.items()}
            rec["wall_ms_median"] = med
            rec["wall_ms_min_max"] = {k: [round(min(v), 2), round(max(v), 2)] for k, v in walls.items()}
            rec["replay_launch_ms_median"] = round(statistics.median(launch), 3)
            rec["reps"] = args.reps
            rec["replay_over_eager"] = round(med["C_replay"] / med["A_eager"], 3)
            rec["eager_ll_over_eager"] = round(med["B_eager_ll"] / med["A_eager"], 3)
            # Free the graphs before the next size (memory), keep the record.
            graphs.clear(); outs.clear(); drafts.clear()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        result("case", **rec)
        summary[n] = rec

    # ------------------------------------------------------------ padding (rank 0, eager)
    pad = {}
    try:
        pad = padding_tests(torch, F, fm, pk, r0, slot, start, news, fast.config)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        pad = {"error": f"{type(exc).__name__}: {exc}"}
    result("padding", **pad)

    # ------------------------------------------------------------ LL all-reduce alone
    ll = {}
    try:
        ll = ll_bench(torch, fast, [int(x) for x in args.ll_rows.split(",")], args.ll_calls)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        ll = {"error": f"{type(exc).__name__}: {exc}"}
    result("allreduce", **ll)

    # ------------------------------------------------------------ summary
    print("\n=== plan 33 step 0 summary (TP2, GCDs " + str(os.environ.get("HIP_VISIBLE_DEVICES")) + ")")
    print(f"{'N':>6} {'A eager ms':>11} {'B eager+LL':>11} {'C replay':>9} {'launch':>7} {'C/A':>6} {'B/A':>6}  exact C vs B | B vs A | C vs A | C vs C | A vs A")
    for n, rec in summary.items():
        if "wall_ms_median" not in rec:
            print(f"{n:>6} capture/run failed: {rec.get('capture_error') or rec.get('error')}")
            continue
        m, e = rec["wall_ms_median"], rec["exact"]
        f = lambda c: "bitwise" if c["bitwise"] else f"diff({c['differing']}/{c['tensors']}, max {c['max_abs_diff']:.3g}, top1 {c['top1_equal']}, drafts {c['drafts_equal']})"  # noqa: E731
        print(f"{n:>6} {m['A_eager']:>11} {m['B_eager_ll']:>11} {m['C_replay']:>9} {rec['replay_launch_ms_median']:>7} "
              f"{rec['replay_over_eager']:>6} {rec['eager_ll_over_eager']:>6}  {f(e['C_vs_B'])} | {f(e['B_vs_A'])} | "
              f"{f(e['C_vs_A'])} | {f(e['C_vs_C'])} | {f(e['A_vs_A'])}")
    if isinstance(ll, dict) and "rows" in ll:
        print("LL all-reduce (in graph, per call) vs eager rendezvous:")
        for row in ll["rows"]:
            print("  " + json.dumps(row))
    r561 = summary.get(561, {})
    ratio = r561.get("replay_over_eager")
    ar = next((r["graph_us"] for r in ll.get("rows", []) if r["rows"] == 561), None) if isinstance(ll, dict) else None
    verdict = []
    if ratio is None:
        verdict.append("561 not measured / capture failed")
    elif ratio > KILL_RATIO:
        verdict.append(f"KILL: replay/eager at 561 = {ratio} > {KILL_RATIO}")
    if ar is not None and ar > KILL_AR_US:
        verdict.append(f"KILL: LL all-reduce at 561 rows = {ar} us > {KILL_AR_US}")
    if any(r.get("capture_error") for r in summary.values()):
        verdict.append("capture failed for some size (see capture_error)")
    result("kill", ratio_561=ratio, allreduce_561_us=ar, verdict=verdict or ["survives"])
    return 0


# ============================================================================ padding tests


def _eq(torch, a, b) -> dict:
    same = torch.equal(a, b)
    return {"bitwise": same, "max_abs_diff": 0.0 if same else float((a.float() - b.float()).abs().max())}


def padding_tests(torch, F, fm, pk, m, slot, start, news, cfg) -> dict:
    """Real rows of an N-row op vs the same rows of the op over N padded to its bucket."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    dev = m.embed.device
    gen = torch.Generator(device=dev).manual_seed(33)
    out: dict = {"gemm": {}, "moe": {}, "gdn_chunk": {}, "gdn_conv": {}, "attention": {}}
    gdn_i = next(i for i, t in enumerate(m.layer_types) if t == "linear_attention")
    attn_i = next(i for i, t in enumerate(m.layer_types) if t != "linear_attention")
    weights = {}
    for name, p in m.layers[gdn_i].linear_attn.named_parameters():
        if p.dim() == 2:
            weights[f"gdn.{name}"] = p
    for name, p in m.layers[attn_i].self_attn.named_parameters():
        if p.dim() == 2:
            weights[f"attn.{name}"] = p
    mlp = m.layers[0].mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        weights[f"shared.{name}"] = getattr(mlp.shared_expert, name).weight
    weights["router"] = mlp.gate.weight
    weights["shared_gate"] = mlp.shared_expert_gate.weight
    weights["lm_head_shard"] = m.lm_head
    if m.mtp is not None:
        weights["mtp.fc"] = m.mtp.fc.weight
    with torch.inference_mode():
        for n in news:
            b = BUCKET.get(n, ((n + 63) // 64) * 64)
            # GEMMs: pad rows random (a padded pass's pad rows carry real-looking activations).
            res = {}
            for name, w in weights.items():
                x = torch.randn((b, w.shape[1]), generator=gen, device=dev, dtype=torch.bfloat16)
                res[name] = _eq(torch, F.linear(x[:n], w), F.linear(x, w)[:n])
            out["gemm"][n] = res
            # Fused MoE block (layer 0), pad rows zero and random.
            hid = mlp.gate.weight.shape[1]
            x = torch.randn((b, hid), generator=gen, device=dev, dtype=torch.bfloat16) * 0.5
            ref = pk.moe(x[:n].contiguous(), mlp, m.top_k)
            xz = x.clone(); xz[n:] = 0
            out["moe"][n] = {"pad_random": _eq(torch, ref, pk.moe(x, mlp, m.top_k)[:n]),
                             "pad_zero": _eq(torch, ref, pk.moe(xz, mlp, m.top_k)[:n])}
            # fla chunked delta rule, pad rows zero q, k, v, g, beta.
            conv, rec = m.gdn_state[gdn_i]
            row = m._g(slot)
            hv, dk, dv = rec.shape[1:]
            q = torch.randn((1, b, hv, dk), generator=gen, device=dev, dtype=torch.bfloat16)
            k = torch.randn((1, b, hv, dk), generator=gen, device=dev, dtype=torch.bfloat16)
            v = torch.randn((1, b, hv, dv), generator=gen, device=dev, dtype=torch.bfloat16)
            g = -torch.rand((1, b, hv), generator=gen, device=dev, dtype=torch.float32) * 0.1
            beta = torch.rand((1, b, hv), generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16)
            for t in (q, k, v, g, beta):
                t[:, n:] = 0
            init = rec[row: row + 1].clone()
            core_n, fin_n = hf.torch_chunk_gated_delta_rule(
                q[:, :n].contiguous(), k[:, :n].contiguous(), v[:, :n].contiguous(), g=g[:, :n].contiguous(),
                beta=beta[:, :n].contiguous(), initial_state=init.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            core_b, fin_b = hf.torch_chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=init.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            out["gdn_chunk"][n] = {"rows": _eq(torch, core_n, core_b[:, :n]), "final_state": _eq(torch, fin_n, fin_b)}
            # GDN conv: output rows and the state it leaves.
            la = m.layers[gdn_i].linear_attn
            chans = conv.shape[1]
            qkv = torch.randn((b, chans), generator=gen, device=dev, dtype=torch.bfloat16)
            st_n, st_b = conv[row].clone(), conv[row].clone()
            o_n = pk.gdn_conv_(qkv[:n].contiguous(), st_n, la.conv1d.weight)
            o_b = pk.gdn_conv_(qkv, st_b, la.conv1d.weight)
            out["gdn_conv"][n] = {"rows": _eq(torch, o_n, o_b[:n]), "state": _eq(torch, st_n, st_b)}
            # Prefill attention: query rows of N vs the same rows of a padded query block.
            k_pool, v_pool = m.kv[attn_i]
            am = m.layers[attn_i].self_attn
            heads = k_pool.shape[1]
            qh = heads * (cfg.num_attention_heads // cfg.num_key_value_heads)
            qa = torch.randn((b, qh, am.head_dim), generator=gen, device=dev, dtype=torch.bfloat16)
            gate = torch.randn((b, qh * am.head_dim), generator=gen, device=dev, dtype=torch.bfloat16)
            table = torch.tensor(m.slot_pages[slot], dtype=torch.long, device=dev)
            a_n = pk.prefill_attention(qa[:n].contiguous(), k_pool, v_pool, table, start, gate[:n].contiguous(), am.scaling)
            a_b = pk.prefill_attention(qa, k_pool, v_pool, table, start, gate, am.scaling)
            out["attention"][n] = dict(_eq(torch, a_n, a_b[:n]), q_heads=qh, kv_heads=heads)
    torch.cuda.synchronize(dev)
    return out


# ============================================================================ LL all-reduce


def ll_bench(torch, fast, rows_list, calls) -> dict:
    ll = fast.group.ll
    devs = fast.devices_tp
    out = {"rows": [], "calls_per_graph": calls}
    for rows in rows_list:
        xs = [torch.randn((rows, 2048), device=d, dtype=torch.bfloat16) for d in devs]
        nbytes = rows * 2048 * 2
        # Exactness of one eager LL call against the eager rendezvous sum.
        with torch.inference_mode():
            ref = [(xs[r].float() + xs[1 - r].to(devs[r]).float()).to(torch.bfloat16) for r in (0, 1)]
            got = [ll._launch(r, xs[r], None) for r in (0, 1)]
            for d in devs:
                torch.cuda.synchronize(d)
            exact = all(torch.equal(got[r], ref[r]) for r in (0, 1))
            for _ in range(3):
                [ll._launch(r, xs[r], None) for r in (0, 1)]
            for d in devs:
                torch.cuda.synchronize(d)
        graphs = []
        streams = [torch.cuda.Stream(d) for d in devs]
        for r, d in enumerate(devs):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.device(d):
                with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(), stream=streams[r]):
                    y = xs[r]
                    for _ in range(calls):
                        y = ll._launch(r, xs[r], None)
            graphs.append(g)
        for d in devs:
            torch.cuda.synchronize(d)

        def rep():
            for r, d in enumerate(devs):
                with torch.cuda.device(d):
                    graphs[r].replay()
            for d in devs:
                torch.cuda.synchronize(d)

        rep()
        ts = []
        for _ in range(10):
            a = time.perf_counter()
            rep()
            ts.append((time.perf_counter() - a) * 1e6 / calls)
        g_us = statistics.median(ts)
        # Eager host rendezvous all-reduce (Group._exchange + add), both rank threads.
        with torch.inference_mode():
            fast._both(lambda m: [m.tp.allreduce(m.rank, xs[m.rank]) for _ in range(3)])
            for d in devs:
                torch.cuda.synchronize(d)
            te = []
            for _ in range(5):
                a = time.perf_counter()
                fast._both(lambda m: [m.tp.allreduce(m.rank, xs[m.rank]) for _ in range(calls)])
                for d in devs:
                    torch.cuda.synchronize(d)
                te.append((time.perf_counter() - a) * 1e6 / calls)
        e_us = statistics.median(te)
        out["rows"].append({
            "rows": rows, "MB": round(nbytes / 1e6, 2), "graph_us": round(g_us, 1),
            "graph_GBps": round(nbytes / (g_us * 1e-6) / 1e9, 1), "eager_rendezvous_us": round(e_us, 1),
            "ll_eq_rendezvous_sum": exact, "timeouts": fast.timeouts(),
        })
        del graphs
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    sys.exit(main())
