"""Plan 66 step 0 (a-c): single-segment prefill passes of first-round lengths as graphs at
64-row sizes, at the head's coarser buckets, and eager; a two-segment eager pack against two
single replays; capture time and memory.

    ENGINE_PREFILL_GRAPH_BUCKETS=... HIP_VISIBLE_DEVICES=4,5 python tools/plan66_probe.py
        [--news 1700,1900,2050,2300,2600] [--start 0] [--reps 7] [--packs 482+300,1200+482]

Loads a TP2 replica as the server does (``Generator.load``: decode/verify graphs, the prefill
graphs of every ``ENGINE_PREFILL_GRAPH_BUCKETS`` size, warmup), then per N a fresh slot (or one
holding a ``--start``-token context) runs the pass of N new tokens through the server's call
(``prefill_packed([(slot, ids, start)])``) with its Gated DeltaNet rows restored between runs:

- ``own``: the smallest captured size >= N (what this bucket list replays);
- ``head``: the smallest size >= N of the head's list (``HEAD``), replayed directly when it is
  captured and differs from ``own``; the head runs N > 2,048 eagerly;
- ``eager``: the same call with the graphs hidden.

Median of ``--reps`` synchronized runs each, interleaved; top-1 and max |logit diff| of own vs
eager and own vs head. Packs: per ``A+B``, two slots each with a 3,600-token context, then the
two segments as one packed (eager) pass against two single-segment passes back to back.
Prints ``RESULT`` lines.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

HEAD = (64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640, 704, 768, 896,
        1024, 1152, 1280, 1408, 1536, 1792, 2048)


def result(kind: str, **fields) -> None:
    print("RESULT " + json.dumps({"kind": kind, **fields}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="1700,1900,2050,2300,2600")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--packs", default="482+300,1200+482")
    ap.add_argument("--pack-start", type=int, default=3600)
    args = ap.parse_args()
    news = [int(n) for n in args.news.split(",") if n]
    packs = [tuple(int(x) for x in p.split("+")) for p in args.packs.split(",") if p]
    os.environ.setdefault("ENGINE_TP", "2")
    import torch

    from engine.fast import tp as tpmod
    from engine.generator import CHECKPOINT, Generator
    from serve_probe import token_pool

    t0 = time.perf_counter()
    gen = Generator.load(
        CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
        prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30,
    )
    fast = gen.fast
    assert isinstance(fast, tpmod.TPModel)
    devs = list(fast.devices_tp)
    gib = 2**30
    result("load", seconds=round(time.perf_counter() - t0, 1), buckets=len(tpmod.PG_BUCKETS),
           sizes=len(fast.pg), largest=max(fast.pg) if fast.pg else None, pg_error=fast.pg_error,
           graph_error=fast.graph_error, kv_pages=fast.ranks[0].pool.pages,
           reserved_gib=[round(torch.cuda.memory_reserved(d) / gib, 2) for d in devs],
           max_reserved_gib=[round(torch.cuda.max_memory_reserved(d) / gib, 2) for d in devs],
           allocated_gib=[round(torch.cuda.memory_allocated(d) / gib, 2) for d in devs],
           free_gib=[round(torch.cuda.mem_get_info(d)[0] / gib, 2) for d in devs])
    graphs = dict(fast.pg)

    def sync() -> None:
        for d in devs:
            torch.cuda.synchronize(d)

    need = max([args.start + 7 * max(news, default=1)] + [args.pack_start * 3 + 4 * sum(p) for p in packs]) + 100
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], need, "synthetic")

    def context(slot: int, start: int, base: int) -> list:
        fast.reset(slot)
        with torch.inference_mode():
            for lo in range(0, start, 2048):
                fast.prefill(slot, pool[base + lo: base + min(start, lo + 2048)], lo)
        sync()
        return fast._gdn_copies(slot)

    def single(slot: int, ids, start: int, mode) -> torch.Tensor:
        """mode: None (eager), 'own' (the bucket list's size) or an int size to replay."""
        if mode is None or mode == "own":
            fast.pg = graphs if mode == "own" else {}
            try:
                return fast.prefill_packed([(slot, ids, start)], [None])[0]
            finally:
                fast.pg = graphs
        fast.ranks[0]._ensure_pages(slot, start + len(ids))
        return fast._pg_run(mode, slot, ids, start, None)

    def head_size(n: int) -> int | None:
        return next((s for s in HEAD if s >= n), None)

    slot = 0
    with torch.inference_mode():
        saved = context(slot, args.start, 0)
        for n in news:
            ids = pool[args.start + 5 * n: args.start + 6 * n]
            own = fast._pg_size(n)
            head = head_size(n)
            modes = {"eager": None}
            if own is not None:
                modes["own"] = "own"
            if head is not None and head != own and head in graphs:
                modes["head"] = head
            logits = {}
            for key, mode in modes.items():
                fast._gdn_load(slot, saved)
                logits[key] = single(slot, ids, args.start, mode).float().cpu()
            walls = {k: [] for k in modes}
            for _ in range(args.reps):
                for key, mode in modes.items():
                    fast._gdn_load(slot, saved)
                    sync()
                    a = time.perf_counter()
                    single(slot, ids, args.start, mode)
                    sync()
                    walls[key].append((time.perf_counter() - a) * 1e3)
            med = {k: round(statistics.median(v), 2) for k, v in walls.items()}
            cmp = {}
            for key in modes:
                if key != "own" and "own" in logits:
                    cmp[f"own_vs_{key}"] = {
                        "top1_equal": int(logits["own"].argmax()) == int(logits[key].argmax()),
                        "max_abs_diff": round(float((logits["own"] - logits[key]).abs().max()), 4)}
            result("pass", new=n, start=args.start, own_size=own, head_size=head if head in graphs else None,
                   head_runs="graph" if head is not None else "eager", wall_ms_median=med,
                   wall_ms_min_max={k: [round(min(v), 2), round(max(v), 2)] for k, v in walls.items()},
                   **cmp)
        fast._gdn_load(slot, saved)

        # Packs: two slots with contexts, one packed eager pass vs two single replays.
        s0, s1 = 1, 2
        for a_n, b_n in packs:
            st = args.pack_start
            sv0 = context(s0, st, 0)
            sv1 = context(s1, st, st + 2 * (a_n + b_n))
            ia = pool[2 * st + 3 * (a_n + b_n): 2 * st + 3 * (a_n + b_n) + a_n]
            ib = pool[2 * st + 4 * (a_n + b_n): 2 * st + 4 * (a_n + b_n) + b_n]
            segs = [(s0, ia, st), (s1, ib, st)]

            def packed():
                return fast.prefill_packed(segs, [None, None])

            def serial():
                x = fast.prefill_packed([segs[0]], [None])
                y = fast.prefill_packed([segs[1]], [None])
                return torch.cat([x, y])

            outs = {}
            for key, fn in (("packed", packed), ("serial", serial)):
                fast._gdn_load(s0, sv0); fast._gdn_load(s1, sv1)
                outs[key] = fn().float().cpu()
            walls = {"packed": [], "serial": []}
            for _ in range(args.reps):
                for key, fn in (("packed", packed), ("serial", serial)):
                    fast._gdn_load(s0, sv0); fast._gdn_load(s1, sv1)
                    sync()
                    t = time.perf_counter()
                    fn()
                    sync()
                    walls[key].append((time.perf_counter() - t) * 1e3)
            result("pack", segments=[a_n, b_n], start=st, sizes=[fast._pg_size(a_n), fast._pg_size(b_n)],
                   wall_ms_median={k: round(statistics.median(v), 2) for k, v in walls.items()},
                   wall_ms_min_max={k: [round(min(v), 2), round(max(v), 2)] for k, v in walls.items()},
                   top1_equal=[int(outs["packed"][i].argmax()) == int(outs["serial"][i].argmax()) for i in range(2)],
                   max_abs_diff=round(float((outs["packed"] - outs["serial"]).abs().max()), 4))
            fast.reset(s0); fast.reset(s1)
    result("timeouts", ll=fast.timeouts(), max_reserved_gib=[round(torch.cuda.max_memory_reserved(d) / gib, 2) for d in devs])
    return 0


if __name__ == "__main__":
    sys.exit(main())
