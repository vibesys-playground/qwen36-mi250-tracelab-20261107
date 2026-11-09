"""Two TP2 replicas in one process, so they can split long prefills (``engine.split_prefill``).

The router (``engine.router.start_replicas``) starts this when split prefill is on: one
``engine.server`` process per two device pairs, with ``ENGINE_PAIR_PORTS=<port a>,<port b>``
and its four GPUs visible (pair a on devices 0,1, pair b on 2,3). The process loads the
checkpoint once and shards it onto both pairs, so each pair is exactly the replica a
one-pair process would hold: its own ``TPModel`` (weights, KV pool, slots, captured graphs),
``Generator`` (prefix cache, counters) and worker thread, behind its own HTTP port. The
router still sees four replicas. The pairs share only the host process: the GIL, and the
split-prefill machinery.

Host placement: the process's CPU slice is its two replicas' slices together, and
``ENGINE_RANK_CPUS`` lists four rank cores: pair a's worker and rank-1 threads on the first two,
pair b's on the last two.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from typing import Any

import torch

from engine import cpu_place, experts, gdn, speculative, split_prefill


def ports() -> list[int] | None:
    text = os.environ.get("ENGINE_PAIR_PORTS", "").strip()
    return [int(p) for p in text.split(",") if p.strip()] if text else None


def load(args: Any, revision: str | None) -> list[Any]:
    """Both pairs' ``Generator``s, each serving through a ``SplitFast`` of its ``TPModel``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from engine.fast import tp as tpmod
    from engine.generator import Generator, capture_fast, experts_implementation

    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"a pair host needs 4 visible GPUs, sees {torch.cuda.device_count()}")
    t0 = time.perf_counter()
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    experts.register()
    chosen = args.experts_implementation or experts_implementation(torch.version.hip)
    gib = 2**30
    # The loaded copy goes on pair b's GPUs: pair a is sharded from it first (keeping it),
    # then pair b (freeing it layer by layer), so each pair's KV pool is sized on GPUs that
    # hold only its own shard.
    lm = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: 1 * gib, 1: 1 * gib, 2: 62 * gib, 3: 62 * gib},
        **({"experts_implementation": chosen} if chosen else {}),
    )
    lm.eval()
    kernels = gdn.install(lm)
    placed = {str(p.device) for p in lm.parameters()}
    if any(d in ("cpu", "meta", "disk") for d in placed):
        raise RuntimeError(f"weights did not fit the visible GPUs (placed on {sorted(placed)})")
    try:
        eos = GenerationConfig.from_pretrained(model, revision=revision).eos_token_id
    except OSError:
        eos = None
    if eos is None:
        eos = lm.config.eos_token_id
    eos_ids = frozenset([eos] if isinstance(eos, int) else (eos or []))
    head = speculative.load_head(lm, model, revision)
    opts = {
        "slots": int(os.environ.get("ENGINE_SLOTS", "32")),
        "max_len": args.max_model_len + (1 if head is not None else 0),
        "mtp": head,
    }
    devs = [torch.device("cuda", i) for i in range(4)]
    text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
    kept = [(t, t.data) for t in (text.embed_tokens.weight, text.norm.weight, lm.lm_head.weight)]
    real_free = tpmod._free
    tpmod._free = lambda module: None
    try:
        a = tpmod.TPModel(lm, devices=devs[0:2], **opts)
    finally:
        tpmod._free = real_free
        for t, d in kept:
            t.data = d
    del kept
    b = tpmod.TPModel(lm, devices=devs[2:4], **opts)
    torch.cuda.empty_cache()
    torch.cuda.set_device(devs[0])
    print(f"engine: pair host sharded two pairs in {time.perf_counter() - t0:.1f}s", flush=True)
    tps = [a, b]
    for p, tp in enumerate(tps):
        # Rank 1's thread of pair p on its own core (the pool starts its thread on first use).
        tp._pool.shutdown(wait=False)
        tp._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"tp{p}-rank1",
            initializer=cpu_place.pin_rank, initargs=(2 * p + 1,),
        )
        capture_fast(tp, head)
    gates = [split_prefill.Gate() for _ in tps]
    pairs = [split_prefill.Pair(p, tp, gate) for p, (tp, gate) in enumerate(zip(tps, gates, strict=True))]
    splitter = split_prefill.Splitter(pairs)
    if split_prefill.enabled():
        sizes = {tp._pg_size(n) for tp in tps for n in range(split_prefill.MIN_TOKENS // 2, 2049)}
        for pair in pairs:
            pair.chains.capture(sorted(s for s in sizes if s is not None))
    options = dict(
        max_model_len=args.max_model_len,
        chunk_tokens=args.prefill_chunk_tokens,
        prefix_cache_tokens=args.prefix_cache_tokens,
        prefix_cache_entries=args.prefix_cache_entries,
        prefix_cache_bytes=int(args.prefix_cache_gib * 2**30),
    )
    generators = []
    for pair in pairs:
        fast = split_prefill.SplitFast(splitter, pair)
        g = Generator(lm, tokenizer, eos_ids, fast=fast, **options)
        g._free_slots.remove(pair.slot)  # the pair's helper slot never serves a request
        stats = g.stats
        g.stats = lambda stats=stats, fast=fast: {**stats(), **fast.split_stats()}
        gdn.ready(g, kernels)
        generators.append(g)
    if split_prefill.enabled():
        _warm_splits(splitter, generators)
    return generators, gates


def _warm_splits(splitter: Any, generators: list[Any]) -> None:
    """One split each way before serving (the first cross-pair copies set up peer access),
    into the owner's own reserved slot; both slots are reset after."""
    vocab = int(generators[0].model.config.get_text_config().vocab_size)
    span = max(1, min(vocab - 1, 4096))
    n = max(split_prefill.MIN_TOKENS, 2 * 769)
    ids = [(7919 * i + 11) % span + 1 for i in range(n)]
    t0 = time.perf_counter()
    for owner in splitter.pairs:
        fast = split_prefill.SplitFast(splitter, owner)
        slot = owner.slot
        if not splitter.wants(owner, [(slot, ids, 0)], [None]):
            print(f"engine: split prefill not warmed for pair {owner.index} at {n} tokens", flush=True)
            continue
        owner.tp.ranks[0]._ensure_pages(slot, n)
        with torch.inference_mode():
            out = splitter.try_split(owner, slot, ids, 0, None)
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
        owner.tp.reset(slot)
        if out is None:
            raise RuntimeError("split prefill warmup could not borrow the helper")
        owner.splits_owned = 0
        owner.split_ms = 0.0
    for p in splitter.pairs:
        p.splits_helped = 0
    print(f"engine: split prefill on (min {split_prefill.MIN_TOKENS} uncached tokens), "
          f"warmed in {time.perf_counter() - t0:.1f}s", flush=True)


def serve(args: Any, revision: str | None, port_list: list[int]) -> None:
    """Load both pairs, then serve pair p on ``port_list[p]`` until stopped."""
    from engine import server

    generators, gates = load(args, revision)
    name = args.served_model_name or args.model
    servers = []
    for p, (g, gate) in enumerate(zip(generators, gates, strict=True)):
        servers.append(server._Server((args.host, port_list[p]), g, name, gate=gate, rank=2 * p))
    per_gpu = ", ".join(f"{d}: {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB"
                        for d in range(torch.cuda.device_count()))
    print(f"engine: weights loaded, allocated per GPU {per_gpu}", flush=True)
    for p, s in enumerate(servers):
        print(f"engine: serving {args.model} pair {p} on http://{args.host}:{port_list[p]}", flush=True)
    try:
        from engine.fast import jit_log

        jit_log.mark_serving()
    except ImportError:
        pass
    threads = [threading.Thread(target=s.serve_forever, name=f"http-pair{p}", daemon=True)
               for p, s in enumerate(servers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
