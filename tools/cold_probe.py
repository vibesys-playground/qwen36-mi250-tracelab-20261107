"""Plan 21, step 1: do prefill/decode kernels still compile, autotune or load on first use
after the startup warmup, i.e. inside a measured replay?

    HIP_VISIBLE_DEVICES=0,1 python tools/cold_probe.py [--attribute] [--cache-root DIR]

Loads one replica in-process exactly as ``engine.server`` does (``Generator.load``: the
fast path, its captured decode graphs and the normal ``gdn.ready`` warmup), then drives
the engine twice through the same synthetic replay-shaped schedule, the way the server's
worker loop does (``PackedPrefill`` passes, then decode steps of the running requests,
``retire`` into session tails): 8 synthetic sessions whose contexts grow to about 8,200
tokens, rounds with 1 to about 2,600 uncached tokens, released in groups of 1-4 so passes
pack 1-4 segments. Token ids are synthetic (seeded random ids, a different seed per run
so run 2 never hits run 1's prefix cache); nothing comes from the benchmark.

For every prefill pass and decode step it records wall ms in run 1 and run 2 (devices
synchronized at pass boundaries only), and logs what fired in run 1:

- Triton JIT compiles (``jit_cache_hook``: kernel name and specialization) and autotunes
  (``Autotuner.run`` growing its cache: kernel, key, seconds), plus the Triton cache
  directory's file count before and after;
- the allocator growing (``memory_reserved``), i.e. hipMalloc on first use;
- with ``--attribute``: device-synchronized exclusive time per op family (F.linear GEMMs,
  SDPA, the bucketed MIOpen conv, fla's chunk rule, the grouped MoE GEMM, and the
  remainder of the attention, GDN, MoE and MTP blocks), and which GEMM (M,N,K), SDPA
  (q, keys), conv (length) and fla shapes a pass ran that warmup never ran. Excess
  (run 1 - run 2) per family attributes a cold pass to its cause. Synchronizing every op
  inflates pass time equally in both runs; use the run without ``--attribute`` for the
  headline excess.

``--cache-root`` points TRITON_CACHE_DIR and MIOpen's user db / cache at fresh (or, to
model a later sample with a warm /tmp, reused) directories under it.

Prints RESULT lines, a JSON summary line (``COLD_PROBE_JSON``) and the kill-rule verdict:
KILL ("warmup already complete") when the summed excess is under 1.5 s and fewer than 3%
of passes are more than 150 ms slower on the first run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path


def _setup_cache_env(root: str | None) -> dict[str, str]:
    """Point the Triton and MIOpen caches under ``root`` (before torch/triton import)."""
    if not root:
        return {}
    root_path = Path(root)
    env = {
        "TRITON_CACHE_DIR": str(root_path / "triton"),
        "MIOPEN_USER_DB_PATH": str(root_path / "miopen-db"),
        "MIOPEN_CUSTOM_CACHE_DIR": str(root_path / "miopen-cache"),
    }
    for path in env.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    return env


def _count_files(path: str | None) -> int:
    if not path or not os.path.isdir(path):
        return -1
    return sum(len(files) for _, _, files in os.walk(path))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attribute", action="store_true", help="per-op synchronized timing and shape novelty")
    p.add_argument("--cache-root", default=None, help="directory for Triton/MIOpen caches")
    p.add_argument("--label", default="probe")
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--max-context", type=int, default=8200)
    p.add_argument("--max-new", type=int, default=2600)
    p.add_argument("--max-rounds", type=int, default=14)
    p.add_argument("--output", type=int, default=16, help="output tokens per round")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--max-len", type=int, default=262144)
    p.add_argument("--slow-ms", type=float, default=150.0)
    return p.parse_args()


ARGS = _parse_args()
CACHE_ENV = _setup_cache_env(ARGS.cache_root)

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- event capture

PHASE = {"name": "load"}  # what the engine is doing now: load, r1:p12, r2:d40, ...
EVENTS: list[dict] = []


def _event(kind: str, **fields) -> None:
    EVENTS.append({"t": time.perf_counter(), "phase": PHASE["name"], "kind": kind, **fields})


def _install_triton_hooks() -> str:
    try:
        import triton
    except Exception as exc:  # noqa: BLE001
        return f"no triton ({exc})"
    starts: dict = {}

    def _name(kwargs) -> str:
        fn = kwargs.get("fn")
        name = getattr(fn, "name", None) or getattr(getattr(fn, "fn", None), "__name__", None)
        return str(name or fn)

    def cache_hook(*args, **kwargs):
        starts[kwargs.get("key")] = time.perf_counter()
        _event("triton_jit", name=_name(kwargs), spec=str(kwargs.get("repr", ""))[:300])
        return False

    def post_hook(*args, **kwargs):
        t0 = starts.pop(kwargs.get("key"), None)
        if t0 is not None:
            _event("triton_jit_done", name=_name(kwargs), s=time.perf_counter() - t0)
        return False

    how = []
    knobs = getattr(triton, "knobs", None)
    if knobs is not None and hasattr(knobs, "runtime") and hasattr(knobs.runtime, "jit_cache_hook"):
        knobs.runtime.jit_cache_hook = cache_hook
        how.append("knobs.runtime.jit_cache_hook")
        if hasattr(knobs.runtime, "jit_post_compile_hook"):
            knobs.runtime.jit_post_compile_hook = post_hook
            how.append("jit_post_compile_hook")
    else:
        from triton.runtime.jit import JITFunction

        JITFunction.cache_hook = cache_hook
        how.append("JITFunction.cache_hook")
    try:
        from triton.runtime.autotuner import Autotuner

        original = Autotuner.run

        def run(self, *args, **kwargs):
            before = len(self.cache)
            t0 = time.perf_counter()
            out = original(self, *args, **kwargs)
            if len(self.cache) > before:
                name = getattr(getattr(self, "base_fn", None), "__name__", None) or getattr(
                    getattr(self, "fn", None), "__name__", "?"
                )
                key = list(self.cache)[-1]
                _event("autotune", name=str(name), key=str(key)[:200], configs=len(self.configs),
                       s=time.perf_counter() - t0)
            return out

        Autotuner.run = run
        how.append("Autotuner.run")
    except Exception as exc:  # noqa: BLE001
        how.append(f"no autotuner hook ({exc})")
    return f"triton {triton.__version__}: " + ",".join(how)


TRITON_HOOKS = _install_triton_hooks()


def _sync() -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


# ---------------------------------------------------------------- attribution (optional)


class Attribution:
    """Exclusive, device-synchronized time per op family while ``active``; shape novelty."""

    def __init__(self) -> None:
        self.active = False
        self.recording = True  # record shapes (no sync) at load/warmup too
        self.stack: list[float] = []
        self.acc: dict[str, float] = defaultdict(float)
        self.warm_shapes: set = set()  # shapes run before the measured runs (warmup, capture)
        self.seen: set = set()
        self.new_shapes: list = []  # shapes first seen in this pass
        self.measuring = False

    def shape(self, key) -> None:
        if not self.recording:
            return
        if not self.measuring:
            self.warm_shapes.add(key)
        elif key not in self.seen and key not in self.warm_shapes:
            self.seen.add(key)
            self.new_shapes.append(key)

    def wrap(self, name: str, fn, key_of=None):
        def wrapped(*args, **kwargs):
            if key_of is not None:
                try:
                    self.shape((name,) + key_of(*args, **kwargs))
                except Exception:  # noqa: BLE001
                    pass
            if not self.active:
                return fn(*args, **kwargs)
            _sync()
            t0 = time.perf_counter()
            self.stack.append(0.0)
            try:
                return fn(*args, **kwargs)
            finally:
                _sync()
                dt = time.perf_counter() - t0
                child = self.stack.pop()
                self.acc[name] += dt - child
                if self.stack:
                    self.stack[-1] += dt

        wrapped.__wrapped__ = fn
        return wrapped

    def take(self) -> tuple[dict[str, float], list]:
        acc, self.acc = dict(self.acc), defaultdict(float)
        new, self.new_shapes = self.new_shapes, []
        return acc, new


ATTR = Attribution()


def _install_attribution_pre_load() -> None:
    """Module-level patches that must be in place before ``Generator.load`` (warmup shapes)."""
    from torch.nn import functional as F

    import engine.fast.model as fm

    def lin_key(x, w, *a, **k):
        return (int(math.prod(x.shape[:-1])), int(w.shape[0]), int(w.shape[1]), str(x.device))

    def sdpa_key(q, k, *a, **kw):
        return (int(q.shape[-2]), int(k.shape[-2]), int(q.shape[1]), str(q.device))

    def conv_key(window, weight, count):
        length = window.shape[-1]
        return (-(-length // fm.CONV_BUCKET) * fm.CONV_BUCKET, int(window.shape[0]), str(window.device))

    F.linear = ATTR.wrap("gemm", F.linear, lin_key)
    F.scaled_dot_product_attention = ATTR.wrap("sdpa", F.scaled_dot_product_attention, sdpa_key)
    fm.prefill_conv = ATTR.wrap("conv", fm.prefill_conv, conv_key)

    from engine.fast import moe_prefill as mp

    def moe_key(x, top_i, *a, **k):
        tokens = int(x.shape[0])
        return (mp.config(tokens * int(top_i.shape[-1]), int(a[1].shape[0]) if len(a) > 1 else -1), str(x.device))

    mp.moe_prefill = ATTR.wrap("moe_grouped", mp.moe_prefill, moe_key)


def _install_attribution_post_load(fast) -> None:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    from engine import gdn

    def fla_key(q, k, v, *a, cu_seqlens=None, initial_state=None, **kw):
        segs = 1 if cu_seqlens is None else int(cu_seqlens.numel()) - 1
        return (int(q.shape[1]), int(q.shape[1]) % 16 == 0, segs, initial_state is not None, str(q.device))

    if hf.torch_chunk_gated_delta_rule is gdn.fla_chunk:
        wrapped = ATTR.wrap("fla", gdn.fla_chunk, fla_key)
        gdn.fla_chunk = wrapped
        hf.torch_chunk_gated_delta_rule = wrapped
    else:
        hf.torch_chunk_gated_delta_rule = ATTR.wrap("gdn_rule", hf.torch_chunk_gated_delta_rule, fla_key)
    for name, family in (
        ("_attn_prefill", "attn_rest"),
        ("_attn_prefill_packed", "attn_rest"),
        ("_gdn_prefill", "gdn_rest"),
        ("_gdn_prefill_packed", "gdn_rest"),
        ("_moe", "moe_rest"),
        ("_mtp_prefill", "mtp_rest"),
    ):
        setattr(fast, name, ATTR.wrap(family, getattr(fast, name)))


if ARGS.attribute:
    _install_attribution_pre_load()
else:
    ATTR.recording = False

from engine.generator import CHECKPOINT, Generator, PackedPrefill  # noqa: E402
from engine.protocol import Sampling  # noqa: E402

# ---------------------------------------------------------------- the synthetic schedule


def schedule(seed: int) -> list[list[tuple[int, int]]]:
    """Groups of (session, round) released together; per session its rounds' new-token
    counts come from ``rounds``. Deterministic in ``seed``; independent of token ids."""
    rng = random.Random(seed)
    firsts = [2600, 2350, 2000, 1700, 1400, 1000, 600, 250, 1900, 800, 2450, 120]
    rounds: list[list[int]] = []
    for s in range(ARGS.sessions):
        ctx = 0
        news = []
        new = min(ARGS.max_new, firsts[s % len(firsts)])
        while len(news) < ARGS.max_rounds and ctx + new + ARGS.output <= ARGS.max_context:
            news.append(new)
            ctx += new + ARGS.output
            if rng.random() < 0.06:
                new = 0  # an exact continuation: only the last output token is uncached
            else:
                new = int(round(math.exp(rng.uniform(math.log(12), math.log(ARGS.max_new)))))
            room = ARGS.max_context - ctx - ARGS.output
            if room < 12:
                break
            new = min(new, room)  # end on a context near the maximum
        rounds.append(news)
    next_round = [0] * len(rounds)
    groups = []
    sizes = [1, 2, 3, 4, 2, 3, 4, 1, 3, 2, 4]
    k = 0
    while any(next_round[s] < len(rounds[s]) for s in range(len(rounds))):
        live = [s for s in range(len(rounds)) if next_round[s] < len(rounds[s])]
        live.sort(key=lambda s: (next_round[s], rng.random()))
        size = min(sizes[k % len(sizes)], len(live))
        k += 1
        group = []
        for s in live[:size]:
            group.append((s, rounds[s][next_round[s]]))
            next_round[s] += 1
        groups.append(group)
    return groups


def run(generator: Generator, groups, run_index: int) -> dict:
    """Drive the groups like the server's worker loop; per-pass and per-decode-step records."""
    rng = random.Random(1000 * run_index + ARGS.seed)
    vocab = int(generator.model.config.get_text_config().vocab_size)
    hi = min(vocab - 1, 150000)
    prompts: dict[int, tuple[int, ...]] = {}
    packer = PackedPrefill(generator)
    sampling = Sampling(max_tokens=ARGS.output, temperature=0.0, ignore_eos=True)
    passes, decodes = [], []
    tag = f"r{run_index}"
    owner: dict[int, int] = {}  # ticket -> session
    t_run = time.perf_counter()
    for gi, group in enumerate(groups):
        tickets = []
        for s, new in group:
            prev = prompts.get(s, ())
            ids = tuple(rng.randrange(10, hi) for _ in range(new))
            prompt = prev + ids
            if not prompt:
                prompt = (rng.randrange(10, hi),)
            prompts[s] = prompt
            tickets.append(packer.add(prompt, sampling))
            owner[tickets[-1]] = s
        failed = packer.failed()
        if failed:
            raise RuntimeError(f"requests refused: {failed}")
        gens = {}
        while packer.busy:
            pid = len(passes)
            PHASE["name"] = f"{tag}:p{pid}"
            reserved = [torch.cuda.memory_reserved(d) for d in range(torch.cuda.device_count())]
            n_events = len(EVENTS)
            ATTR.active = ARGS.attribute
            _sync()
            t0 = time.perf_counter()
            done = packer.step()
            _sync()
            ms = (time.perf_counter() - t0) * 1e3
            ATTR.active = False
            acc, new_shapes = ATTR.take()
            grown = sum(torch.cuda.memory_reserved(d) - r for d, r in enumerate(reserved))
            for ticket, result in done:
                if isinstance(result, Exception):
                    raise result
                gens[ticket] = result
            passes.append({
                "group": gi,
                "segments": [],  # filled below from the pass shape log
                "ms": ms,
                "acc": {k: v * 1e3 for k, v in acc.items()},
                "new_shapes": [list(map(str, s)) for s in new_shapes],
                "events": EVENTS[n_events:],
                "grown_mb": grown / 2**20,
            })
        # The segments of this group's passes: from the generator's pass log (below).
        # Decode until every request of the group ends, as ``Worker._loop`` / ``_run`` do.
        active = [gens[t] for t in tickets]
        while active:
            waiting = []
            for gen in active:
                while True:
                    step = gen.next_step()
                    if step.finish is not None:
                        gen.retire()
                        break
                    if not gen.has_queued():
                        waiting.append(gen)
                        break
            if not waiting:
                break
            did = len(decodes)
            PHASE["name"] = f"{tag}:d{did}"
            n_events = len(EVENTS)
            _sync()
            t0 = time.perf_counter()
            generator.decode(waiting)
            _sync()
            decodes.append({"batch": len(waiting), "ms": (time.perf_counter() - t0) * 1e3,
                            "events": EVENTS[n_events:]})
            active = waiting
        for t in tickets:  # the next round's prompt continues after this round's output
            prompts[owner[t]] = prompts[owner[t]] + tuple(gens[t]._picked)
    PHASE["name"] = f"{tag}:idle"
    return {"passes": passes, "decodes": decodes, "s": time.perf_counter() - t_run}


def _wrap_pass_log(fast) -> list:
    """Record each prefill pass's segments (new tokens, start) as the engine runs it."""
    log: list = []
    original_packed = fast.prefill_packed

    def prefill_packed(segments, follows=None):
        log.append([(len(ids), int(start)) for _, ids, start in segments])
        return original_packed(segments, follows)

    fast.prefill_packed = prefill_packed
    return log


def main() -> int:
    default_triton = os.path.expanduser("~/.triton/cache")
    print(f"RESULT label={ARGS.label} attribute={ARGS.attribute} hooks={TRITON_HOOKS!r}", flush=True)
    print(f"RESULT cache_env={CACHE_ENV} HOME={os.environ.get('HOME')} "
          f"default_triton_files={_count_files(default_triton)} "
          f"tmp_config_miopen={os.path.isdir('/tmp/.config/miopen')} "
          f"tmp_cache_miopen_files={_count_files('/tmp/.cache/miopen')}", flush=True)
    triton_dir = os.environ.get("TRITON_CACHE_DIR") or default_triton
    miopen_dirs = [os.environ.get("MIOPEN_USER_DB_PATH"), os.environ.get("MIOPEN_CUSTOM_CACHE_DIR")]
    print(f"RESULT before_load triton_files={_count_files(triton_dir)} "
          f"miopen_files={[_count_files(d) for d in miopen_dirs]}", flush=True)
    print("torch", torch.__version__, "hip", torch.version.hip, "gpus", torch.cuda.device_count(), flush=True)
    t0 = time.perf_counter()
    generator = Generator.load(
        CHECKPOINT["model"],
        CHECKPOINT["revision"],
        max_model_len=ARGS.max_len,
        chunk_tokens=ARGS.chunk,
        prefix_cache_tokens=4194304,
        prefix_cache_entries=128,
        prefix_cache_bytes=int(16 * 2**30),
    )
    fast = generator.fast
    assert fast is not None, "the fast path did not load"
    load_s = time.perf_counter() - t0
    load_events = len(EVENTS)
    jit_load = sum(1 for e in EVENTS if e["kind"] == "triton_jit")
    tune_load = sum(1 for e in EVENTS if e["kind"] == "autotune")
    print(f"RESULT load_s={load_s:.1f} mtp={fast.mtp is not None} load_triton_jit={jit_load} "
          f"load_autotune={tune_load} triton_files_after_load={_count_files(triton_dir)}", flush=True)
    if ARGS.attribute:
        _install_attribution_post_load(fast)
    pass_log = _wrap_pass_log(fast)
    ATTR.measuring = True

    groups = schedule(ARGS.seed)
    n_rounds = sum(len(g) for g in groups)
    news = [n for g in groups for _, n in g]
    print(f"RESULT schedule groups={len(groups)} rounds={n_rounds} new_tokens_total={sum(news)} "
          f"max_new={max(news)} pack_sizes={sorted({len(g) for g in groups})}", flush=True)

    runs = []
    files_before = _count_files(triton_dir)
    for r in (1, 2):
        mark = len(pass_log)
        out = run(generator, groups, r)
        shapes = pass_log[mark:]
        if len(shapes) != len(out["passes"]):
            raise RuntimeError(f"pass log {len(shapes)} != passes {len(out['passes'])}")
        for p, segs in zip(out["passes"], shapes, strict=True):
            p["segments"] = segs
        runs.append(out)
        print(f"RESULT run{r} s={out['s']:.1f} passes={len(out['passes'])} decodes={len(out['decodes'])} "
              f"prefill_ms={sum(p['ms'] for p in out['passes']):.0f} "
              f"decode_ms={sum(d['ms'] for d in out['decodes']):.0f} "
              f"triton_files={_count_files(triton_dir)}", flush=True)
    print(f"RESULT triton_files load_end={files_before} run1_end/run2_end above; "
          f"miopen_files={[_count_files(d) for d in miopen_dirs]}", flush=True)

    r1, r2 = runs
    if [p["segments"] for p in r1["passes"]] != [p["segments"] for p in r2["passes"]]:
        print("WARNING: run 1 and run 2 pass shapes differ; pairing by index anyway", flush=True)
    if [d["batch"] for d in r1["decodes"]] != [d["batch"] for d in r2["decodes"]]:
        print("WARNING: decode batch sequence differs between runs (MTP acceptance); "
              "decode excess compared as totals per batch size", flush=True)

    # ---------------------------------------------------------------- per-pass excess
    print("PASS idx segs new_tokens max_total r1_ms r2_ms excess_ms events/new-shapes")
    rows = []
    for i, (a, b) in enumerate(zip(r1["passes"], r2["passes"], strict=False)):
        excess = a["ms"] - b["ms"]
        segs = a["segments"]
        new_tokens = sum(n for n, _ in segs)
        max_total = max(n + s for n, s in segs)
        jit = [e for e in a["events"] if e["kind"] == "triton_jit"]
        tunes = [e for e in a["events"] if e["kind"] == "autotune"]
        fam = {k: a["acc"].get(k, 0.0) - b["acc"].get(k, 0.0) for k in set(a["acc"]) | set(b["acc"])}
        row = {
            "idx": i, "segments": segs, "new_tokens": new_tokens, "max_total": max_total,
            "r1_ms": a["ms"], "r2_ms": b["ms"], "excess_ms": excess,
            "jit": [e["name"] for e in jit], "autotune": [(e["name"], e["key"], round(e["s"], 3)) for e in tunes],
            "jit_s": sum(e["s"] for e in a["events"] if e["kind"] == "triton_jit_done"),
            "grown_mb": a["grown_mb"], "family_excess_ms": fam, "new_shapes": a["new_shapes"],
        }
        rows.append(row)
        info = []
        if jit:
            info.append(f"jit={len(jit)}:{sorted(set(row['jit']))}")
        if tunes:
            info.append(f"autotune={row['autotune']}")
        if a["grown_mb"] > 0:
            info.append(f"alloc+{a['grown_mb']:.0f}MB")
        if fam:
            top = sorted(fam.items(), key=lambda kv: -kv[1])[:3]
            info.append("fam=" + ",".join(f"{k}:{v:+.0f}" for k, v in top))
        if a["new_shapes"]:
            kinds = defaultdict(int)
            for s in a["new_shapes"]:
                kinds[s[0]] += 1
            info.append("new=" + ",".join(f"{k}:{v}" for k, v in sorted(kinds.items())))
        print(f"PASS {i} {len(segs)} {new_tokens} {max_total} {a['ms']:.1f} {b['ms']:.1f} {excess:+.1f} "
              f"{segs} {' '.join(info)}", flush=True)

    # Decode: totals per batch size.
    dec = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for d in r1["decodes"]:
        dec[d["batch"]][0] += d["ms"]
        dec[d["batch"]][2] += 1
    for d in r2["decodes"]:
        dec[d["batch"]][1] += d["ms"]
        dec[d["batch"]][3] += 1
    decode_excess = 0.0
    for batch, (m1, m2, n1, n2) in sorted(dec.items()):
        per1, per2 = m1 / max(n1, 1), m2 / max(n2, 1)
        decode_excess += (per1 - per2) * n1
        print(f"DECODE batch={batch} r1_steps={n1} r1_mean_ms={per1:.2f} r2_steps={n2} r2_mean_ms={per2:.2f} "
              f"excess_ms={(per1 - per2) * n1:+.1f}", flush=True)
    dec_events = [e for d in r1["decodes"] for e in d["events"]]
    slow_decodes = [(i, d["batch"], round(d["ms"], 1)) for i, d in enumerate(r1["decodes"]) if d["ms"] > 100]
    print(f"RESULT decode_events_run1={len(dec_events)} slow_decodes_run1(>100ms)={slow_decodes[:20]}", flush=True)

    # ---------------------------------------------------------------- summary
    excess_pos = sum(max(0.0, r["excess_ms"]) for r in rows)
    excess_net = sum(r["excess_ms"] for r in rows)
    slow = [r for r in rows if r["excess_ms"] > ARGS.slow_ms]
    frac = len(slow) / max(1, len(rows))
    total_s = (excess_net + max(0.0, decode_excess)) / 1e3
    fam_tot = defaultdict(float)
    for r in rows:
        for k, v in r["family_excess_ms"].items():
            fam_tot[k] += v
    run1_jit = [e for e in EVENTS[load_events:] if e["kind"] == "triton_jit" and e["phase"].startswith("r1")]
    run2_jit = [e for e in EVENTS[load_events:] if e["kind"] == "triton_jit" and e["phase"].startswith("r2")]
    run1_tune = [e for e in EVENTS[load_events:] if e["kind"] == "autotune" and e["phase"].startswith("r1")]
    jit_names = defaultdict(int)
    for e in run1_jit:
        jit_names[e["name"]] += 1
    print(f"RESULT triton run1_jit={len(run1_jit)} run2_jit={len(run2_jit)} run1_autotune={len(run1_tune)} "
          f"run1_jit_by_kernel={dict(jit_names)}", flush=True)
    for e in run1_tune:
        print(f"AUTOTUNE {e['phase']} {e['name']} configs={e['configs']} s={e['s']:.2f} key={e['key']}", flush=True)
    for e in run1_jit:
        print(f"JIT {e['phase']} {e['name']} {e['spec'][:200]}", flush=True)
    if ARGS.attribute:
        print("RESULT family_excess_ms " + " ".join(f"{k}={v:+.0f}" for k, v in sorted(fam_tot.items())), flush=True)
        novel = defaultdict(set)
        for r in rows:
            for s in r["new_shapes"]:
                novel[s[0]].add(tuple(s[1:]))
        for kind, shapes in sorted(novel.items()):
            print(f"NEWSHAPES {kind} n={len(shapes)} {sorted(shapes)[:60]}", flush=True)
    print(f"RESULT passes={len(rows)} excess_pos_ms={excess_pos:.0f} excess_net_ms={excess_net:.0f} "
          f"decode_excess_ms={decode_excess:+.0f} total_cold_s={total_s:.2f} "
          f"slow_passes(>{ARGS.slow_ms:.0f}ms)={len(slow)} slow_frac={frac:.3f}", flush=True)
    kill = total_s < 1.5 and frac < 0.03
    print(f"VERDICT {'KILL warmup already complete' if kill else 'PASS cold work remains'} "
          f"(total_cold_s={total_s:.2f} < 1.5 and slow_frac={frac:.3f} < 0.03 -> kill)", flush=True)
    summary = {
        "label": ARGS.label, "attribute": ARGS.attribute, "load_s": load_s,
        "excess_pos_ms": excess_pos, "excess_net_ms": excess_net, "decode_excess_ms": decode_excess,
        "total_cold_s": total_s, "slow_passes": len(slow), "passes": len(rows), "kill": kill,
        "family_excess_ms": dict(fam_tot), "run1_jit": len(run1_jit), "run1_autotune": len(run1_tune),
        "slow": [{k: r[k] for k in ("idx", "segments", "r1_ms", "r2_ms", "excess_ms", "jit", "autotune",
                                     "family_excess_ms", "new_shapes")} for r in slow],
    }
    print("COLD_PROBE_JSON " + json.dumps(summary, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
