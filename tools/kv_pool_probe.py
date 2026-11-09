"""Plan 58 stage 0: the KV pool and GDN state memory of one TP2 replica under a slot / ring setting.

    ENGINE_SPEC_KMAX=7 ENGINE_SLOTS=16 python tools/kv_pool_probe.py --devices 0,1

Loads one replica as the server does (``Generator.load``), then prints one ``RESULT pool`` line:
the KV pages per attention layer (``fast.pool.pages``, sized from free memory after the GDN
state rings are allocated), the tokens they hold, the GDN state bytes (conv + recurrent, all
ring entries of every slot row) per GCD, the KV bytes per GCD and each GCD's free memory after
load and graph capture. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")
    os.environ.setdefault("ENGINE_TP", "2")

    import torch

    from engine.fast import mfma
    from engine.fast.model import PAGE
    from engine.generator import CHECKPOINT, Generator

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    ranks = list(getattr(fast, "ranks", [fast]))
    per_gcd = []
    for m in ranks:
        gdn = sum(c.numel() * c.element_size() + r.numel() * r.element_size() for c, r in m.gdn_state.values())
        kv = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in m.kv.values())
        dev = m.embed.device
        free, total = torch.cuda.mem_get_info(dev)
        per_gcd.append({"device": str(dev), "gdn_state_gb": round(gdn / 2**30, 3), "kv_gb": round(kv / 2**30, 3),
                        "free_gb_after_load": round(free / 2**30, 2), "total_gb": round(total / 2**30, 2)})
    r0 = ranks[0]
    line = {
        "kmax": mfma.KMAX, "ring": r0.ring, "slots": r0.slots, "gdn_rows": r0.gdn_rows,
        "kv_pages": r0.pool.pages, "kv_tokens": r0.pool.pages * PAGE, "page": PAGE,
        "max_pages_per_seq": r0.max_pages, "graphs": r0.use_graphs, "graph_error": fast.graph_error,
        "per_gcd": per_gcd, "load_s": round(time.perf_counter() - started),
        "env": {k: os.environ.get(k) for k in ("ENGINE_SPEC_KMAX", "ENGINE_SLOTS", "ENGINE_KV_FRACTION",
                                               "ENGINE_RESERVE_GB", "ENGINE_KV_SEQS")},
    }
    print("RESULT pool " + json.dumps(line, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
