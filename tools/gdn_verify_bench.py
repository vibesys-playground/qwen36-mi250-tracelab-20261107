"""Where a fused verify step's cost over plain decode goes, on one 2-GCD replica (plan 18 step 0).

    python tools/gdn_verify_bench.py [--devices 0,1] [--prompt-len 1800] [--iters 30]

Loads one replica as the server does (``Generator.load`` with the MTP head, sharded over
``--devices``), prefills 8 slots with replay-like prompts, then for B = 1, 2, 4:

1. Whole steps (captured graphs, every device synchronized, median over ``--iters``):
   ``verify`` B slots (2B rows, MTP head inside), plain ``decode`` of 2B slots (same rows),
   and the MTP head alone on 2B rows. ``overhead = verify - decode(2B) - mtp``: what a verify
   step costs beyond the same rows of plain decode plus the head.
2. The GDN mixer of all 30 GDN layers at 2B rows, each device's layers in one captured graph,
   summed over devices: ``FusedDecode._gdn`` (decode: GEMV with conv epilogue, step, GEMV) vs
   ``FusedDecode._gdn_verify`` (the verify path), and the pieces of the verify path:
   the in_proj GEMV with / without the conv epilogue, the conv of both halves (with their
   ``contiguous`` and copy-back), the delta-rule step on both halves vs on all rows at once,
   the bank copies (``index_select`` + ``index_copy_`` of conv and recurrent state), the
   out_proj GEMV, and ``_gdn_verify`` without the bank copies (timing only, wrong math).
3. Kernel counts per layer (torch.profiler, eager, one layer) of ``_gdn`` and ``_gdn_verify``.

Kill rule (plan 18): park if the GDN verify overhead at B=2 summed over 30 layers
(``_gdn_verify - _gdn``) is under 0.8 ms. Prints ``RESULT`` lines; a hint for the plan, not
the task's metric.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    os.environ.setdefault("ENGINE_MTP", "1")

    import torch

    from engine.fast import fused as fused_mod
    from engine.fast import kernels
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
            start = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - start) * 1e3)
        return statistics.median(times)

    started = time.perf_counter()
    g = Generator.load(
        CHECKPOINT["model"],
        CHECKPOINT["revision"],
        max_model_len=262144,
        chunk_tokens=args.chunk,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )
    fast = g.fast
    fused = fast.fused
    print(
        f"RESULT load_s={time.perf_counter() - started:.0f} gpus={torch.cuda.device_count()} "
        f"mtp={fast.mtp is not None} graphs={fast.use_graphs} graph_error={fast.graph_error} "
        f"fused={fused is not None} fused_verify={fast.triton_kernels.get('fused_verify')} "
        f"capture_mode={fast.capture_mode}",
        flush=True,
    )
    if fast.mtp is None or fused is None:
        raise SystemExit("needs the MTP head and the fused layers")
    tok = g.tokenizer
    chunk = args.chunk

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(prompt) - 1
        position = 0
        while position < body:
            end = min((position // chunk + 1) * chunk, body)
            fast.prefill(slot, prompt[position:end], position, prompt[end])
            position = end
        return fast.prefill(slot, prompt[body:], body)

    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    corpus = pool(9973 * 8 + 20000)
    slots = list(range(8))
    length = args.prompt_len
    prompts = [corpus[9973 * k : 9973 * k + length] for k in slots]
    lasts = [int(torch.argmax(prefill(s, p))) for s, p in zip(slots, prompts, strict=True)]
    drafts = [0] * 8
    batches = (1, 2, 4)

    # ---------------------------------------------------------------- 1. whole steps
    steps: dict[int, dict[str, float]] = {}
    for batch in batches:
        try:
            rows = 2 * batch
            ver = median_ms(
                lambda: fast.verify(lasts[:batch], drafts[:batch], slots[:batch], [length] * batch),
                args.iters,
            )
            dec = median_ms(
                lambda: fast.decode(lasts[:rows], slots[:rows], [length] * rows).argmax(-1).cpu(),
                args.iters,
            )
            size = fast.bucket(batch)
            dev = fast.lm_head.device
            meta = fast.devices[dev]
            hidden = torch.randn((2 * size, fast.config.hidden_size), dtype=fast.embed.dtype, device=dev)
            toks = meta.tokens[: 2 * size].clone()
            graph = capture(torch, dev, lambda: fast._mtp_decode(hidden, toks, meta), fast.capture_mode)
            mtp = median_ms(lambda graph=graph: replay(torch, dev, graph), args.iters)
            over = ver - dec - mtp
            steps[batch] = {"verify": ver, "decode": dec, "mtp": mtp, "over": over}
            print(
                f"RESULT step B={batch} verify_ms={ver:.2f} decode_{rows}rows_ms={dec:.2f} "
                f"mtp_{rows}rows_ms={mtp:.2f} verify_overhead_ms={over:.2f} "
                f"(verify - decode(2B) - mtp)",
                flush=True,
            )
        except Exception:  # noqa: BLE001 - report and go on to the per-layer part
            print(f"RESULT step B={batch} error:\n{traceback.format_exc()}", flush=True)
    for s in slots:
        fast.reset(s)

    # ---------------------------------------------------------------- 2. GDN layers in isolation
    gdn_layers = [i for i in range(len(fast.layers)) if fast.layer_types[i] == "linear_attention"]
    by_dev: dict[torch.device, list[int]] = {}
    for i in gdn_layers:
        by_dev.setdefault(fast.layers[i].input_layernorm.weight.device, []).append(i)
    print(
        "RESULT gdn_layers=" + str(len(gdn_layers)) + " "
        + " ".join(f"{d}:{len(v)}" for d, v in by_dev.items()),
        flush=True,
    )
    hidden_size = fast.config.hidden_size
    gdn_rows = fast.gdn_rows

    for batch in batches:
        rows = 2 * batch
        n = batch
        try:
            per_dev = {}
            for dev, layer_ids in by_dev.items():
                with torch.cuda.device(dev):
                    h = (torch.randn((rows, hidden_size), device=dev) * 0.05).to(fast.embed.dtype)
                    dec_states = torch.arange(rows, dtype=torch.long, device=dev)
                    ver_states = torch.cat(
                        (torch.arange(n, device=dev), torch.arange(n, device=dev) + gdn_rows)
                    ).to(torch.long)
                    first, second = ver_states[:n], ver_states[n:]
                    # Per-layer buffers built once, outside the graphs.
                    bufs = {}
                    for i in layer_ids:
                        m = fast.layers[i].linear_attn
                        conv_state, rec_state = fast.gdn_state[i]
                        conv_dim = 2 * m.key_dim + m.value_dim
                        proj = fused._gemv(h, fused.w_in[i])
                        core = torch.empty((rows, m.num_v_heads * m.head_v_dim), dtype=h.dtype, device=dev)
                        bufs[i] = (m, conv_state, rec_state, conv_dim, proj, core)
                    per_dev[dev] = (layer_ids, h, dec_states, ver_states, first, second, bufs)

            def variant(name):
                """fn(dev) running ``name`` over that device's GDN layers."""

                def run(dev):
                    layer_ids, h, dec_states, ver_states, first, second, bufs = per_dev[dev]
                    with kernels.on_device(h):
                        for i in layer_ids:
                            m, conv_state, rec_state, conv_dim, proj, core = bufs[i]
                            if name == "gdn":
                                fused._gdn(i, h, dec_states)
                            elif name == "verify":
                                fused._gdn_verify(i, h, ver_states)
                            elif name == "verify_banks":
                                fused._gdn_verify(i, h, ver_states, bank_fused=False)
                            elif name == "verify_bankfused":
                                fused._gdn_verify(i, h, ver_states, bank_fused=True)
                            elif name == "verify_nocopy":
                                p = fused._gemv(h, fused.w_in[i])
                                c = torch.empty_like(core)
                                for rs, st in ((slice(0, n), first), (slice(n, rows), second)):
                                    part = p[rs]
                                    part[:, :conv_dim] = kernels.gdn_conv_(
                                        conv_state, st, part[:, :conv_dim], m.conv1d.weight
                                    )
                                    fused._gdn_step(m, rec_state, st, part, c[rs])
                                fused._gemv(c, m.out_proj.weight)
                            elif name == "gemv_in_conv":
                                fused._gemv(h, fused.w_in[i],
                                            conv=(conv_state, dec_states, m.conv1d.weight, conv_dim))
                            elif name == "gemv_in":
                                fused._gemv(h, fused.w_in[i])
                            elif name == "conv_halves":
                                for rs, st in ((slice(0, n), first), (slice(n, rows), second)):
                                    part = proj[rs]
                                    part[:, :conv_dim] = kernels.gdn_conv_(
                                        conv_state, st, part[:, :conv_dim], m.conv1d.weight
                                    )
                            elif name == "step_halves":
                                for rs, st in ((slice(0, n), first), (slice(n, rows), second)):
                                    fused._gdn_step(m, rec_state, st, proj[rs], core[rs])
                            elif name == "step_all":
                                fused._gdn_step(m, rec_state, dec_states, proj, core)
                            elif name == "bank_copies":
                                conv_state.index_copy_(0, second, conv_state.index_select(0, first))
                                rec_state.index_copy_(0, second, rec_state.index_select(0, first))
                            elif name == "bank_copy_rec":
                                rec_state.index_copy_(0, second, rec_state.index_select(0, first))
                            elif name == "out_proj":
                                fused._gemv(core, m.out_proj.weight)
                            else:
                                raise ValueError(name)

                return run

            names = ["gdn", "verify", "verify_banks", "verify_bankfused", "verify_nocopy", "gemv_in_conv", "gemv_in", "conv_halves",
                     "step_halves", "step_all", "bank_copies", "bank_copy_rec", "out_proj"]
            totals = {}
            for name in names:
                try:
                    fn = variant(name)
                    total = 0.0
                    for dev in per_dev:
                        graph = capture(torch, dev, lambda dev=dev: fn(dev), fast.capture_mode)
                        total += median_ms(lambda dev=dev, graph=graph: replay(torch, dev, graph), args.iters)
                        del graph
                    totals[name] = total
                    print(
                        f"RESULT layers B={batch} rows={rows} {name}_30layers_ms={total:.3f} "
                        f"per_layer_us={total / len(gdn_layers) * 1e3:.1f}",
                        flush=True,
                    )
                except Exception:  # noqa: BLE001
                    print(f"RESULT layers B={batch} {name} error:\n{traceback.format_exc()}", flush=True)
            if "gdn" in totals and "verify" in totals:
                gap = totals["verify"] - totals["gdn"]
                copies = totals.get("bank_copies", float("nan"))
                nocopy_gap = totals.get("verify_nocopy", float("nan")) - totals["gdn"]
                conv_gap = (totals.get("gemv_in", float("nan")) + totals.get("conv_halves", float("nan"))
                            - totals.get("gemv_in_conv", float("nan")))
                step_gap = totals.get("step_halves", float("nan")) - totals.get("step_all", float("nan"))
                print(
                    f"RESULT gap B={batch} verify-gdn_30layers_ms={gap:.3f} "
                    f"bank_copies_ms={copies:.3f} (rec only {totals.get('bank_copy_rec', float('nan')):.3f}) "
                    f"verify_nocopy-gdn_ms={nocopy_gap:.3f} "
                    f"conv_split_ms={conv_gap:.3f} (gemv_in+conv_halves-gemv_in_conv) "
                    f"step_split_ms={step_gap:.3f} (step_halves-step_all)",
                    flush=True,
                )
                if "verify_banks" in totals and "verify_bankfused" in totals:
                    print(
                        f"RESULT bank_fused B={batch} old(copies)-gdn_ms={totals['verify_banks'] - totals['gdn']:.3f} "
                        f"new(bank_fused)-gdn_ms={totals['verify_bankfused'] - totals['gdn']:.3f} "
                        f"saved_ms={totals['verify_banks'] - totals['verify_bankfused']:.3f} "
                        f"(ENGINE_GDN_BANK_FUSED default={fused_mod.BANK_FUSED})",
                        flush=True,
                    )
                if batch in steps:
                    over = steps[batch]["over"]
                    print(
                        f"RESULT share B={batch} step_overhead_ms={over:.2f} gdn_gap_ms={gap:.3f} "
                        f"elsewhere_ms={over - gap:.3f}",
                        flush=True,
                    )
                if batch == 2:
                    verdict = "PARK" if gap < 0.8 else "GO"
                    print(f"RESULT kill_test B=2 gdn_verify_overhead_30layers_ms={gap:.3f} threshold=0.8 "
                          f"-> {verdict}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"RESULT layers B={batch} error:\n{traceback.format_exc()}", flush=True)

    # ---------------------------------------------------------------- 2b. bitwise old vs new
    # Every GDN layer at B=1..4, either bank current: the bank-fused path and the old path
    # with bank copies from the same random states must give identical outputs and states.
    try:
        for batch in (1, 2, 3, 4):
            for flip in (0, 1):
                bad = []
                for dev, layer_ids in by_dev.items():
                    with torch.cuda.device(dev), kernels.on_device(torch.empty(0, device=dev)), \
                            torch.inference_mode():
                        h = (torch.randn((2 * batch, hidden_size), device=dev) * 0.5).to(fast.embed.dtype)
                        cur = torch.arange(batch, device=dev) + flip * gdn_rows
                        oth = torch.arange(batch, device=dev) + (1 - flip) * gdn_rows
                        st = torch.cat((cur, oth)).to(torch.long)
                        for i in layer_ids:
                            conv_state, rec_state = fast.gdn_state[i]
                            conv_state.copy_((torch.randn(conv_state.shape, device=dev) * 0.5).to(conv_state.dtype))
                            rec_state.copy_(torch.randn(rec_state.shape, device=dev) * 0.1)
                            c0, r0 = conv_state.clone(), rec_state.clone()
                            a = fused._gdn_verify(i, h, st, bank_fused=False).clone()
                            ca, ra = conv_state.clone(), rec_state.clone()
                            conv_state.copy_(c0)
                            rec_state.copy_(r0)
                            b = fused._gdn_verify(i, h, st, bank_fused=True).clone()
                            if not (torch.equal(a, b) and torch.equal(ca, conv_state) and torch.equal(ra, rec_state)):
                                bad.append(i)
                            conv_state.zero_()
                            rec_state.zero_()
                print(f"RESULT bitwise B={batch} current_bank={flip} layers={len(gdn_layers)} "
                      f"mismatching={bad if bad else 'none'}", flush=True)
    except Exception:  # noqa: BLE001
        print(f"RESULT bitwise error:\n{traceback.format_exc()}", flush=True)

    # ---------------------------------------------------------------- 3. kernel counts
    try:
        from torch.profiler import ProfilerActivity, profile

        dev, layer_ids = next(iter(by_dev.items()))
        i = layer_ids[0]
        for batch in batches:
            rows = 2 * batch
            with torch.cuda.device(dev):
                h = (torch.randn((rows, hidden_size), device=dev) * 0.05).to(fast.embed.dtype)
                dec_states = torch.arange(rows, dtype=torch.long, device=dev)
                ver_states = torch.cat(
                    (torch.arange(batch, device=dev), torch.arange(batch, device=dev) + gdn_rows)
                ).to(torch.long)
                counts = {}
                for name, fn in (("gdn", lambda: fused._gdn(i, h, dec_states)),
                                 ("verify", lambda: fused._gdn_verify(i, h, ver_states)),
                                 ("verify_banks", lambda: fused._gdn_verify(i, h, ver_states, bank_fused=False))):
                    with kernels.on_device(h), torch.inference_mode():
                        fn()
                    sync()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        with kernels.on_device(h), torch.inference_mode():
                            fn()
                        sync()
                    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    counts[name] = len(events)
                    if batch == 2:
                        names = sorted({e.name[:60] for e in events})
                        print(f"RESULT kernels B=2 {name}: " + " | ".join(names), flush=True)
                print(
                    f"RESULT kernel_count B={batch} gdn={counts['gdn']} verify={counts['verify']} "
                    f"verify_banks={counts['verify_banks']} "
                    f"extra_per_step_30layers={(counts['verify'] - counts['gdn']) * len(gdn_layers)}",
                    flush=True,
                )
    except Exception:  # noqa: BLE001
        print(f"RESULT kernel_count error:\n{traceback.format_exc()}", flush=True)
    for s in slots:
        fast.reset(s)


def capture(torch, dev, fn, mode):
    """One CUDA graph of ``fn`` on ``dev`` (warmed twice on a side stream first)."""
    with torch.cuda.device(dev):
        stream = torch.cuda.Stream(dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream), torch.inference_mode():
            fn()
            fn()
        torch.cuda.current_stream(dev).wait_stream(stream)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode=mode or "thread_local"), \
                torch.inference_mode():
            fn()
        torch.cuda.synchronize(dev)
    return graph


def replay(torch, dev, graph) -> None:
    with torch.cuda.device(dev):
        graph.replay()


if __name__ == "__main__":
    main()
