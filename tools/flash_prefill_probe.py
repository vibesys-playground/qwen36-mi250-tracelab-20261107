"""Plan 35 step 0: can an MFMA-tiled flash prefill attention kernel beat ``_prefill_attn_kernel``?

    python tools/flash_prefill_probe.py [--reps 20] [--shapes 61:3540,561:3040,...]

One GCD, synthetic BF16 q/gate and K/V pools with the engine's TP2 shapes (8 query heads,
1 KV head, head dim 256, the [pages, heads, 64, 256] paged layout, a shuffled page table
padded with zeros as ``FastModel._push_table``). For each (new, cached) shape it times:

- ``prefill_kernels.prefill_attention`` (the head's kernel);
- ``flash_prefill.flash_prefill_attention`` (the candidate, engine/fast/flash_prefill.py,
  not wired) over a sweep of tiles (BLOCK_R 8/16 -> 64/128 query rows per program, BLOCK_N
  32/64/128 keys per step), warps 4/8, num_stages 1/2, then for the best few: K loaded
  [N, D] and transposed, waves_per_eu, MFMA 16x16 vs 32x32, and key splits;
- torch SDPA over gathered contiguous K/V with a lower-right causal mask, per backend.

Each kernel time is the median of ``--reps`` launches timed with CUDA events, the L2 flushed
(a 128 MB write) before each launch. Every variant is checked against the fp32 reference
(``prefill_attention_reference`` on fp32 copies of the inputs): max abs error, also for the
head's kernel; for bitwise repeatability over 3 runs; and for CUDA-graph capture (capture,
replay, bitwise equal to the eager output: no host sync in the call).

Candidate variants run in child processes (a memory fault kills only the child; the parent
records it and continues with the next variant). The result is printed as tables and a
verdict: KILL (exit 3) if the best correct, repeatable, capturable variant takes more than
500 us per layer at (561, 3040), else GO (exit 0).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QH, HEADS, D, PAGE = 8, 1, 256, 64
SCALE = D ** -0.5
ATTN_LAYERS = 10  # full-attention layers of the 40 (every 4th); the MTP head adds 1 more
KILL_US = 500.0
KILL_SHAPE = (561, 3040)
SHAPES = [(61, 3540), (561, 3040), (1650, 1950), (2300, 5800)]
CUS = 104


# ---------------------------------------------------------------------------- data & timing


def make_inputs(count: int, start: int, device):
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(1000 * count + start)
    total = start + count
    used = -(-total // PAGE)
    pool_pages = used + 37
    k_pool = torch.randn((pool_pages, HEADS, PAGE, D), generator=g, device=device).to(torch.bfloat16)
    v_pool = torch.randn((pool_pages, HEADS, PAGE, D), generator=g, device=device).to(torch.bfloat16)
    # x2.5: scores of std ~2.5, a peaked softmax (outputs of order 0.1-1, not a flat mean)
    q = (2.5 * torch.randn((count, QH, D), generator=g, device=device)).to(torch.bfloat16)
    gate = torch.randn((count, QH * D), generator=g, device=device).to(torch.bfloat16)
    cpu = torch.Generator()
    cpu.manual_seed(count + 7 * start)
    perm = (torch.randperm(pool_pages - 1, generator=cpu)[:used] + 1).to(device)
    table = torch.zeros(used + 64, dtype=torch.int32, device=device)
    table[:used] = perm.to(torch.int32)
    return q, k_pool, v_pool, table, gate


def reference(q, k_pool, v_pool, table, start, gate):
    import torch
    from engine.fast import prefill_kernels

    from torch.nn.attention import SDPBackend, sdpa_kernel

    f = torch.float32
    with sdpa_kernel([SDPBackend.MATH]):
        return prefill_kernels.prefill_attention_reference(
            q.to(f), k_pool.to(f), v_pool.to(f), table, start, gate.to(f), SCALE
        )


_FLUSH = None


def time_us(fn, reps: int) -> float:
    import torch

    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device="cuda")
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        _FLUSH.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(times)


def checks(fn, ref, base=None) -> dict:
    """Max abs error vs the fp32 reference (and vs ``base``), 3-run bitwise repeat, graph."""
    import torch

    outs = [fn().clone() for _ in range(3)]
    torch.cuda.synchronize()
    res = {
        "err": (outs[0].float() - ref).abs().max().item(),
        "repeat": all(torch.equal(outs[0], o) for o in outs[1:]),
        "finite": bool(torch.isfinite(outs[0]).all().item()),
    }
    if base is not None:
        res["vs_head"] = (outs[0].float() - base.float()).abs().max().item()
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        graph.replay()
        torch.cuda.synchronize()
        res["graph"] = bool(torch.equal(captured, outs[0]))
    except Exception as exc:  # noqa: BLE001
        res["graph"] = False
        res["graph_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return res


# ------------------------------------------------------------------------------- children


def cfg_name(c: dict) -> str:
    if c.get("head"):
        return "head _prefill_attn_kernel"
    if not c:
        return "auto (flash_prefill.choose)"
    parts = [f"R{c['block_r']}xN{c['block_n']}", f"w{c['num_warps']}", f"s{c['num_stages']}"]
    if not c.get("kt", True):
        parts.append("kNxD")
    if "waves_per_eu" in c:
        parts.append(f"wpe{c['waves_per_eu']}")
    if "matrix_instr_nonkdim" in c:
        parts.append(f"mfma{c['matrix_instr_nonkdim']}")
    if c.get("split", 1) != 1:
        parts.append(f"split{c['split'] or 'auto'}")
    return "/".join(parts)


def child(args) -> None:
    import torch
    from engine.fast import flash_prefill, prefill_kernels

    jobs = json.loads(Path(args.child).read_text())  # [[count, start, cfg], ...]
    cache = {}
    for i, (count, start, cfg) in enumerate(jobs):
        print(f"@@START {i}", flush=True)
        key = (count, start)
        if key not in cache:
            cache.clear()
            torch.cuda.empty_cache()
            inp = make_inputs(count, start, "cuda")
            ref = reference(*inp[:4], start, inp[4])
            base = prefill_kernels.prefill_attention_rows(*inp[:4], start, inp[4], SCALE)
            cache[key] = (inp, ref, base)
        (q, k, v, table, gate), ref, base = cache[key]
        kw = dict(cfg)
        if kw.get("head"):
            fn = lambda: prefill_kernels.prefill_attention_rows(q, k, v, table, start, gate, SCALE)  # noqa: E731
        else:
            fn = lambda: flash_prefill.flash_prefill_attention(  # noqa: E731
                q, k, v, table, start, gate, SCALE, **kw
            )
        rec = {"i": i, "count": count, "start": start, "cfg": cfg}
        try:
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            rec["first_s"] = round(time.time() - t0, 1)
            rec["us"] = time_us(fn, args.reps)
            rec.update(checks(fn, ref, base))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        print("@@RESULT " + json.dumps(rec), flush=True)


def run_children(jobs: list, reps: int, scratch: Path) -> list[dict]:
    """Run ``jobs`` in child processes; a crash records the job and restarts after it."""
    results: list[dict] = []
    todo = list(range(len(jobs)))
    while todo:
        spec = scratch / f"jobs_{todo[0]}.json"
        spec.write_text(json.dumps([jobs[i] for i in todo]))
        proc = subprocess.Popen(
            [sys.executable, __file__, "--child", str(spec), "--reps", str(reps)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT,
        )
        current = None
        seen = 0
        tail: list[str] = []
        for line in proc.stdout:
            if line.startswith("@@START "):
                current = int(line.split()[1])
            elif line.startswith("@@RESULT "):
                rec = json.loads(line[9:])
                rec["i"] = todo[rec["i"]]
                results.append(rec)
                seen += 1
                current = None
            else:
                tail = (tail + [line.rstrip()])[-6:]
        code = proc.wait()
        if seen == len(todo) or (current is None and code == 0):
            break
        # crashed at ``current`` (or before the next START)
        current = current if current is not None else seen
        idx = todo[current]
        count, start, cfg = jobs[idx]
        results.append({"i": idx, "count": count, "start": start, "cfg": cfg,
                        "error": f"child exit {code}: " + " | ".join(tail)[-300:]})
        print(f"child crashed (exit {code}) on {cfg_name(cfg)} at {count}:{start}: {' | '.join(tail)[-300:]}", flush=True)
        todo = todo[current + 1:]
    return results


# ------------------------------------------------------------------------------------ SDPA


def sdpa_variants(q, k_pool, v_pool, table, start, gate):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right

    count = q.shape[0]
    total = start + count
    used = -(-total // PAGE)
    idx = table[:used].long()

    def gather(expand):
        keys = k_pool[idx].permute(1, 0, 2, 3).reshape(HEADS, -1, D)[None, :, :total]
        vals = v_pool[idx].permute(1, 0, 2, 3).reshape(HEADS, -1, D)[None, :, :total]
        if expand:
            keys = keys.expand(1, QH, total, D) if HEADS == 1 else keys.repeat_interleave(QH // HEADS, 1)
            vals = vals.expand(1, QH, total, D) if HEADS == 1 else vals.repeat_interleave(QH // HEADS, 1)
        return keys, vals

    def make(backend, mask_kind, expand):
        def fn():
            keys, vals = gather(expand)
            qq = q.transpose(0, 1).unsqueeze(0)
            if mask_kind == "lower_right":
                mask = causal_lower_right(count, total)
            else:
                pos = torch.arange(start, total, device=q.device)
                mask = (torch.arange(total, device=q.device)[None, :] <= pos[:, None])[None, None]
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(
                    qq, keys, vals, attn_mask=mask, scale=SCALE, enable_gqa=not expand and QH != HEADS
                )
            out = out[0].transpose(0, 1).reshape(count, -1)
            return out * torch.sigmoid(gate)
        return fn

    B = SDPBackend
    return {
        "sdpa flash lower_right gqa": make(B.FLASH_ATTENTION, "lower_right", False),
        "sdpa flash lower_right expandKV": make(B.FLASH_ATTENTION, "lower_right", True),
        "sdpa efficient lower_right gqa": make(B.EFFICIENT_ATTENTION, "lower_right", False),
        "sdpa efficient lower_right expandKV": make(B.EFFICIENT_ATTENTION, "lower_right", True),
        "sdpa efficient boolmask expandKV": make(B.EFFICIENT_ATTENTION, "bool", True),
    }


# ------------------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--shapes", default=",".join(f"{a}:{b}" for a, b in SHAPES))
    ap.add_argument("--child", default=None)
    ap.add_argument("--tune", action="store_true", help="the tile/split crossover sweep over T")
    args = ap.parse_args()
    if args.child:
        child(args)
        return 0
    if args.tune:
        return tune(args)

    import torch
    import triton

    from engine.fast import flash_prefill, prefill_kernels

    shapes = [tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} triton {triton.__version__} hip {torch.version.hip} "
          f"device {props.name} {getattr(props, 'gcnArchName', '')} CUs {props.multi_processor_count}")
    print(f"shapes (new, cached): {shapes}; QH={QH} KV heads={HEADS} D={D} PAGE={PAGE}; reps={args.reps}")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_fpp")
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"fpp_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)

    head: dict = {}
    sdpa: dict = {}
    for count, start in shapes:
        inp = make_inputs(count, start, "cuda")
        q, k, v, table, gate = inp
        ref = reference(q, k, v, table, start, gate)
        fn = lambda: prefill_kernels.prefill_attention_rows(q, k, v, table, start, gate, SCALE)  # noqa: E731
        rec = {"us": time_us(fn, args.reps)}
        rec.update(checks(fn, ref))
        head[(count, start)] = rec
        print(f"HEAD {count}:{start} {json.dumps(rec)}", flush=True)
        base = fn()
        for name, sfn in sdpa_variants(q, k, v, table, start, gate).items():
            try:
                r = {"us": time_us(sfn, args.reps)}
                r.update(checks(sfn, ref, base))
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0][:200]}"}
            sdpa[(count, start, name)] = r
            print(f"SDPA {count}:{start} {name} {json.dumps(r)}", flush=True)
        del inp, q, k, v, table, gate, ref, base
        torch.cuda.empty_cache()

    # Phase 1: the tile sweep, no split, every shape.
    base_cfgs = [
        {"block_r": r, "block_n": n, "num_warps": w, "num_stages": s, "split": 1}
        for r, n, w, s in itertools.product([8, 16], [32, 64, 128], [4, 8], [1, 2])
    ]
    jobs = [[c, s, cfg] for (c, s) in shapes for cfg in base_cfgs]
    t0 = time.time()
    res = run_children(jobs, args.reps, scratch)
    print(f"phase 1: {len(jobs)} runs in {time.time() - t0:.0f} s", flush=True)

    def ok(r):
        """Correct (error within 2x the head kernel's, or 1e-2), repeatable, capturable."""
        tol = max(2 * head[(r["count"], r["start"])]["err"], 1e-2)
        return "us" in r and r.get("repeat") and r.get("finite") and r.get("graph") and r["err"] <= tol

    # Phase 2: the 3 best tiles per shape (by phase 1) x extra knobs, and key splits.
    jobs2 = []
    for c, s in shapes:
        good = sorted((r for r in res if (r["count"], r["start"]) == (c, s) and ok(r)), key=lambda r: r["us"])[:3]
        for r in good:
            cfg = r["cfg"]
            extras = [{"kt": False}, {"waves_per_eu": 1}, {"waves_per_eu": 2},
                      {"matrix_instr_nonkdim": 16}, {"matrix_instr_nonkdim": 32}]
            if -(-c // cfg["block_r"]) * HEADS < CUS:
                extras += [{"split": n} for n in (2, 4, 8, 16)] + [{"split": None}]
            for e in extras:
                jobs2.append([c, s, {**cfg, **e}])
    t0 = time.time()
    res += run_children(jobs2, args.reps, scratch)
    print(f"phase 2: {len(jobs2)} runs in {time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------------------ report
    print("\n=== all candidate runs (us per layer; err = max abs vs fp32 reference) ===")
    for r in sorted(res, key=lambda r: (r["count"], r["start"], r.get("us", 1e18))):
        if "us" in r:
            print(f"{r['count']:>5}:{r['start']:<5} {cfg_name(r['cfg']):<40} {r['us']:9.1f} us  err {r['err']:.3e}  "
                  f"vs_head {r.get('vs_head', float('nan')):.3e}  repeat {r['repeat']}  graph {r['graph']}  "
                  f"finite {r['finite']}  compile {r.get('first_s')} s")
        else:
            print(f"{r['count']:>5}:{r['start']:<5} {cfg_name(r['cfg']):<40} FAILED {r['error']}")

    print("\n=== summary per shape (per layer, and per pass = x10 attention layers; +MTP = x11) ===")
    print(f"{'new':>5} {'cached':>6} | {'head us':>9} {'head err':>9} rep | {'best flash':<36} {'us':>8} {'err':>9} rep graph | "
          f"{'speedup':>7} | {'head ms/pass':>12} {'best ms/pass':>12} {'(+MTP)':>8}")
    best_at_kill = None
    for c, s in shapes:
        h = head[(c, s)]
        good = sorted((r for r in res if (r["count"], r["start"]) == (c, s) and ok(r)), key=lambda r: r["us"])
        if not good:
            print(f"{c:>5} {s:>6} | {h['us']:9.1f} {h['err']:9.2e} {h['repeat']} | no valid flash variant")
            continue
        b = good[0]
        if (c, s) == KILL_SHAPE:
            best_at_kill = b
        print(f"{c:>5} {s:>6} | {h['us']:9.1f} {h['err']:9.2e} {str(h['repeat'])[0]} | {cfg_name(b['cfg']):<36} "
              f"{b['us']:8.1f} {b['err']:9.2e} {str(b['repeat'])[0]}   {str(b['graph'])[0]}   | {h['us'] / b['us']:6.2f}x | "
              f"{h['us'] * ATTN_LAYERS / 1000:12.2f} {b['us'] * ATTN_LAYERS / 1000:12.2f} {b['us'] * (ATTN_LAYERS + 1) / 1000:8.2f}")
    print("\nhead kernel graph-capturable:", {f"{c}:{s}": head[(c, s)].get("graph") for c, s in shapes})

    print("\n=== SDPA over gathered K/V (gather + SDPA + gate, us per layer) ===")
    for (c, s, name), r in sdpa.items():
        if "us" in r:
            print(f"{c:>5}:{s:<5} {name:<38} {r['us']:9.1f} us  err {r['err']:.3e}  repeat {r['repeat']}  graph {r['graph']}"
                  + (f"  ({r['graph_error']})" if "graph_error" in r else ""))
        else:
            print(f"{c:>5}:{s:<5} {name:<38} UNSUPPORTED/FAILED: {r['error']}")

    # Same config across all shapes: the single best one-size config (no split) and per-shape auto split.
    print("\n=== one config for all shapes (phase 1 tiles, sum of us over the shapes) ===")
    by_cfg: dict = {}
    for r in res:
        if ok(r) and r["cfg"] in base_cfgs:
            by_cfg.setdefault(cfg_name(r["cfg"]), {})[(r["count"], r["start"])] = r["us"]
    full = [(sum(v.values()), k, v) for k, v in by_cfg.items() if len(v) == len(shapes)]
    for tot, k, v in sorted(full)[:6]:
        print(f"{k:<36} total {tot:9.1f} us  " + "  ".join(f"{c}:{s}={v[(c, s)]:.1f}" for c, s in shapes))

    if best_at_kill is None:
        print(f"\nVERDICT: KILL (no valid flash variant at {KILL_SHAPE})")
        return 3
    us = best_at_kill["us"]
    verdict = "GO" if us <= KILL_US else "KILL"
    print(f"\nVERDICT: {verdict}: best at {KILL_SHAPE} = {us:.1f} us/layer ({cfg_name(best_at_kill['cfg'])}), "
          f"threshold {KILL_US:.0f} us; head {head[KILL_SHAPE]['us']:.1f} us")
    return 0 if verdict == "GO" else 3


TUNE_T = [61, 128, 256, 400, 561, 700, 850, 1000, 1200, 1400, 1650]


def tune(args) -> int:
    """Per T (on a 3,600-token context, and 2,300 after 5,800): head, the auto choice and the
    tiles/splits it chooses among; prints the best per T and auto's ratio to it."""
    import torch

    from engine.fast import flash_prefill

    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_fpp")
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"fpt_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    shapes = [(t, 3600 - t) for t in TUNE_T] + [(2300, 5800)]
    short = {"block_r": 8, "block_n": 64, "num_warps": 4, "num_stages": 1}
    long = {"block_r": 16, "block_n": 128, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1}
    cfgs = [{"head": True}, {}]
    cfgs += [{**short, "split": n} for n in (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)]
    cfgs += [{**long, "split": n} for n in (1, 2, 3, 4)]
    cfgs += [{"block_r": 16, "block_n": 64, "num_warps": 4, "num_stages": 1, "split": n} for n in (1, 2)]
    jobs = [[c, s, cfg] for c, s in shapes for cfg in cfgs]
    t0 = time.time()
    res = run_children(jobs, args.reps, scratch)
    print(f"tune: {len(jobs)} runs in {time.time() - t0:.0f} s", flush=True)
    for r in sorted(res, key=lambda r: (r["count"], r.get("us", 1e18))):
        if "us" in r:
            print(f"TUNE {r['count']:>5}:{r['start']:<5} {cfg_name(r['cfg']):<40} {r['us']:9.1f} us  err {r['err']:.3e}  "
                  f"repeat {r['repeat']}  graph {r['graph']}")
        else:
            print(f"TUNE {r['count']:>5}:{r['start']:<5} {cfg_name(r['cfg']):<40} FAILED {r['error']}")
    print(f"\n{'new':>5} {'cached':>6} | {'head us':>9} | {'auto us':>8} {'auto choice':<34} | {'best us':>8} {'best':<34} | auto/best head/auto")
    for c, s in shapes:
        rows = [r for r in res if (r["count"], r["start"]) == (c, s) and "us" in r and r.get("repeat") and r.get("graph")]
        head = next((r["us"] for r in rows if r["cfg"].get("head")), float("nan"))
        auto = next((r["us"] for r in rows if not r["cfg"]), float("nan"))
        flash = sorted((r for r in rows if r["cfg"] and not r["cfg"].get("head")), key=lambda r: r["us"])
        ch = flash_prefill.choose(c, s, HEADS, QH // HEADS, CUS)
        if flash:
            b = flash[0]
            print(f"{c:>5} {s:>6} | {head:9.1f} | {auto:8.1f} {cfg_name(ch):<34} | {b['us']:8.1f} {cfg_name(b['cfg']):<34} | "
                  f"{auto / b['us']:8.2f} {head / auto:8.2f}")
    del torch
    return 0


if __name__ == "__main__":
    sys.exit(main())
