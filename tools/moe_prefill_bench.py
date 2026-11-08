"""Grouped prefill experts (``engine.fast.moe_prefill``) against the current path, on GPUs.

    python tools/moe_prefill_bench.py [--layers 8] [--kernel] [--passes] [--parity]

Loads the checkpoint as the server does (``Generator.load``, the fast path,
over the visible GPUs; ``--layers N`` keeps only the first N decoder layers so
a few real layers fit one small GPU), then:

``--kernel``  Per MoE layer, with the hidden states that layer really sees
              while prefilling 4096 tokens of sample text (the source of the
              installed Transformers package, nothing from the benchmark):
              the routed experts of the first T tokens, T in --tokens, with the
              layer's real routing and with skewed routing (the same inputs,
              router logits of --hot experts raised by --skew), through the
              padded batched matmul (``engine.experts``, which falls back to its
              per-expert loop above PADDED_MAX_ROWS), the per-expert loop, and
              the grouped kernels (each tile config in --configs, the default
              first). Prints ms and the max error relative to the loop, and
              whether the grouped call synchronized with the host.
``--passes``  Whole prefill passes (``FastModel.prefill`` / ``prefill_packed``)
              with the grouped path on and off: 1xT for T in --pass-tokens and
              packed 2x700, median of --reps.
``--parity``  Top-1 agreement of the grouped path against the current one:
              last-token logits after prefills of --parity-prompts prefixes of
              random length, and teacher-forced greedy decode steps after
              eight of them (decode itself is unchanged; its state comes from
              the prefill); the per-expert loop against the current path is the
              noise floor (same math, another reduction order).

Every number is a hint for the lead, not the task's metric.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sync() -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def timed(fn, reps: int, prepare=None, warm: int = 2) -> float:
    for _ in range(warm):
        if prepare:
            prepare()
        fn()
    times = []
    for _ in range(reps):
        if prepare:
            prepare()
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def sample_ids(tokenizer, total: int, skip: int = 0) -> list[int]:
    import transformers

    files = sorted(Path(transformers.__file__).parent.glob("models/*/modeling_*.py"))[skip:]
    ids: list[int] = []
    for f in files:
        ids.extend(tokenizer.encode(f.read_text()))
        if len(ids) >= total:
            return ids[:total]
    raise RuntimeError("not enough sample text")


def load(layers: int):
    from engine.generator import CHECKPOINT, Generator

    config = None
    if layers:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(CHECKPOINT["model"], revision=CHECKPOINT["revision"])
        for cfg in {id(c): c for c in (config, config.get_text_config())}.values():
            if hasattr(cfg, "layer_types"):
                cfg.layer_types = list(cfg.layer_types)[:layers]
                cfg.num_hidden_layers = layers
            if hasattr(cfg, "mtp_num_hidden_layers"):
                cfg.mtp_num_hidden_layers = 0
    t0 = time.perf_counter()
    generator = Generator.load(
        CHECKPOINT["model"],
        CHECKPOINT["revision"],
        config=config,
        max_model_len=32768 if layers else 262144,
        chunk_tokens=int(os.environ.get("ENGINE_CHUNK", "2048")),
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )
    fast = generator.fast
    if fast is None:
        raise RuntimeError("the fast path did not load")
    print(
        f"RESULT load_s={time.perf_counter() - t0:.1f} layers={len(fast.layers)} "
        f"triton={fast.triton_kernels} graphs={fast.use_graphs}",
        flush=True,
    )
    return generator, fast


def routing(mlp, x, top_k, hot=None, skew=0.0):
    from torch.nn import functional as F

    logits = F.linear(x, mlp.gate.weight)
    if hot is not None:
        logits = logits.clone()
        logits[:, hot] += skew
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    w, i = torch.topk(probs, top_k, dim=-1)
    return i, (w / w.sum(dim=-1, keepdim=True)).to(logits.dtype)


def kernel_bench(generator, fast, args) -> None:
    from engine import experts as ex
    from engine.fast import moe_prefill as mp

    # The hidden states each MoE layer sees while prefilling sample text (grouped path on).
    captured: dict[int, torch.Tensor] = {}
    index = {id(layer.mlp): i for i, layer in enumerate(fast.layers)}
    original = fast._moe

    def capture(mlp, x, decode=False):
        if not decode and x.shape[0] > 16:
            captured[index[id(mlp)]] = x.detach().clone()
        return original(mlp, x, decode)

    fast._moe = capture
    tokens = max(args.tokens)
    ids = sample_ids(generator.tokenizer, tokens)
    fast.reset(0)
    with torch.inference_mode():
        fast.prefill(0, ids, 0)
    fast.reset(0)
    del fast._moe
    sync()
    layers = sorted(captured)
    if args.max_layers:
        layers = layers[:: max(1, len(layers) // args.max_layers)][: args.max_layers]
    configs = [None] + [c for c in args.configs.split(";") if c]
    best: dict[tuple[int, str], dict[str, float]] = {}
    for li in layers:
        mlp = fast.layers[li].mlp
        experts = mlp.experts
        gu, dn = experts.gate_up_proj, experts.down_proj
        hot = torch.arange(args.hot, device=gu.device) * 7 % gu.shape[0]
        for mode in ("real", "skewed"):
            for T in args.tokens:
                x = captured[li][:T].to(gu.device)
                with torch.inference_mode():
                    if mode == "real":
                        top_i, top_w = routing(mlp, x, fast.top_k)
                    else:
                        top_i, top_w = routing(mlp, x, fast.top_k, hot, args.skew)
                    counts = torch.bincount(top_i.reshape(-1), minlength=gu.shape[0])
                    cap = int(counts.max())
                    used = int((counts > 0).sum())
                    falls_back = cap * gu.shape[0] > ex.PADDED_MAX_ROWS
                    loop = ex._sorted_experts_forward(experts, x, top_i, top_w)
                    padded = ex._padded_experts_forward(experts, x, top_i, top_w)
                    scale = float(loop.float().abs().max())
                    t_pad = timed(lambda: ex._padded_experts_forward(experts, x, top_i, top_w), args.reps)
                    t_loop = timed(lambda: ex._sorted_experts_forward(experts, x, top_i, top_w), args.reps)
                    err_pad = float((padded.float() - loop.float()).abs().max()) / scale
                    results = []
                    for cfg in configs:
                        if cfg:
                            os.environ["ENGINE_MOE_GROUPED_CFG"] = cfg
                        else:
                            os.environ.pop("ENGINE_MOE_GROUPED_CFG", None)
                        try:
                            got = mp.moe_prefill(x, top_i, top_w, gu, dn)
                            sync()
                            err = float((got.float() - loop.float()).abs().max()) / scale
                            same = bool(torch.equal(got, mp.moe_prefill(x, top_i, top_w, gu, dn)))
                            t = timed(lambda: mp.moe_prefill(x, top_i, top_w, gu, dn), args.reps)
                            results.append((cfg or "default", t, err, same))
                        except Exception as exc:  # noqa: BLE001
                            print(f"RESULT kernel cfg={cfg} FAILED {exc!r}", flush=True)
                    os.environ.pop("ENGINE_MOE_GROUPED_CFG", None)
                    # Host synchronizations inside one grouped call.
                    syncs = "n/a"
                    try:
                        torch.cuda.set_sync_debug_mode("warn")
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            mp.moe_prefill(x, top_i, top_w, gu, dn)
                        syncs = str(sum("synchroniz" in str(w.message) for w in caught))
                    finally:
                        torch.cuda.set_sync_debug_mode(0)
                    # Sanity of the sync detector: the padded path's .item() should count.
                    try:
                        torch.cuda.set_sync_debug_mode("warn")
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            ex._padded_experts_forward(experts, x, top_i, top_w)
                        pad_syncs = str(sum("synchroniz" in str(w.message) for w in caught))
                    finally:
                        torch.cuda.set_sync_debug_mode(0)
                default = results[0] if results else ("-", float("nan"), float("nan"), False)
                fastest = min(results, key=lambda r: r[1]) if results else default
                print(
                    f"RESULT kernel layer={li} routing={mode} T={T} pairs={T * fast.top_k} "
                    f"experts_used={used} max_rows={cap} padded_falls_back_to_loop={falls_back} "
                    f"padded_ms={t_pad:.3f} loop_ms={t_loop:.3f} grouped_ms={default[1]:.3f} "
                    f"speedup_vs_padded={t_pad / default[1]:.2f}x speedup_vs_loop={t_loop / default[1]:.2f}x "
                    f"relerr_grouped={default[2]:.4f} relerr_padded={err_pad:.4f} deterministic={default[3]} "
                    f"host_syncs_grouped={syncs} host_syncs_padded={pad_syncs} "
                    f"best_cfg={fastest[0]} best_ms={fastest[1]:.3f}",
                    flush=True,
                )
                for cfg, t, err, _ in results:
                    print(f"RESULT cfg layer={li} routing={mode} T={T} cfg={cfg} ms={t:.3f} relerr={err:.4f}", flush=True)
                    best.setdefault((T, mode), {}).setdefault(cfg, 0.0)
                    best[(T, mode)][cfg] += t
    for (T, mode), totals in sorted(best.items()):
        order = sorted(totals.items(), key=lambda kv: kv[1])
        print(f"RESULT cfg_rank T={T} routing={mode} " + " ".join(f"{c}:{v:.2f}" for c, v in order[:4]), flush=True)


def passes(generator, fast, args) -> None:
    src = [sample_ids(generator.tokenizer, max(args.pass_tokens) + 1, skip=20 + 7 * j) for j in range(2)]

    def reset(slots):
        for s in slots:
            fast.reset(s)

    out = {}
    with torch.inference_mode():
        for flag in (False, True):
            fast.triton_kernels["moe_prefill"] = flag
            name = "grouped" if flag else "current"
            for T in args.pass_tokens:
                ms = timed(lambda T=T: fast.prefill(0, src[0][:T], 0), args.reps, lambda: reset([0]))
                out[(name, 1, T)] = ms
                print(f"RESULT pass path={name} n=1 T={T} ms={ms:.1f}", flush=True)
            segs = [(j, src[j][:700], 0) for j in range(2)]
            ms = timed(lambda: fast.prefill_packed(segs), args.reps, lambda: reset([0, 1]))
            out[(name, 2, 700)] = ms
            print(f"RESULT pass path={name} n=2 T=700 ms={ms:.1f}", flush=True)
            reset([0, 1])
        fast.triton_kernels["moe_prefill"] = True
    for T in args.pass_tokens:
        print(f"RESULT pass_speedup n=1 T={T} {out[('current', 1, T)] / out[('grouped', 1, T)]:.2f}x", flush=True)
    print(
        f"RESULT pass_speedup n=2 T=700 {out[('current', 2, 700)] / out[('grouped', 2, 700)]:.2f}x ; "
        f"grouped packed 2x700 {out[('grouped', 2, 700)]:.1f} ms vs 2 x 1x700 "
        f"{2 * out[('grouped', 1, 700)]:.1f} ms -> packed_cheaper={out[('grouped', 2, 700)] < 2 * out[('grouped', 1, 700)]}",
        flush=True,
    )


def parity(generator, fast, args) -> None:
    """Top-1 agreement with the current path ("current": padded bmm, its loop under skew) of
    the grouped path and, as the noise floor of a mere change of reduction order, of the
    per-expert loop alone (PADDED_MAX_ROWS=0)."""
    from engine import experts as ex

    rng = random.Random(0)
    texts = [sample_ids(generator.tokenizer, 4097, skip=40 + 5 * j) for j in range(8)]
    padded_max = ex.PADDED_MAX_ROWS

    def use(variant: str) -> None:
        fast.triton_kernels["moe_prefill"] = variant == "grouped"
        ex.PADDED_MAX_ROWS = 0 if variant == "loop" else padded_max

    variants = ("current", "grouped", "loop")
    agree = {v: 0 for v in variants[1:]}
    d_agree = {v: 0 for v in variants[1:]}
    max_diff = {v: 0.0 for v in variants[1:]}
    total = d_total = 0
    steps = args.decode_steps
    try:
        with torch.inference_mode():
            for n in range(args.parity_prompts):
                ids = texts[n % 8][: rng.randint(17, 4096)]
                res = {}
                for v in variants:
                    use(v)
                    fast.reset(0)
                    res[v] = fast.prefill(0, ids, 0).float().cpu()
                fast.reset(0)
                total += 1
                for v in variants[1:]:
                    agree[v] += int(res[v].argmax() == res["current"].argmax())
                    max_diff[v] = max(max_diff[v], float((res[v] - res["current"]).abs().max()))
            # Teacher-forced greedy decode after prefills with each path, fed the current
            # path's tokens (decode itself is unchanged; its state comes from the prefill).
            for j in range(8):
                ids = texts[j][: 300 + 450 * j]
                tokens = {}
                for v in variants:
                    use(v)
                    fast.reset(0)
                    seq = [int(fast.prefill(0, ids, 0).float().argmax())]
                    for s in range(steps):
                        feed = seq[s] if v == "current" else tokens["current"][s]
                        out = fast.decode([feed], [0], [len(ids) + s]).float()
                        seq.append(int(out[0].argmax()))
                    tokens[v] = seq
                fast.reset(0)
                d_total += len(tokens["current"])
                for v in variants[1:]:
                    d_agree[v] += sum(a == b for a, b in zip(tokens["current"], tokens[v]))
    finally:
        use("grouped")
    for v in variants[1:]:
        print(
            f"RESULT parity[{v} vs current] prefill_last_token top1={100 * agree[v] / total:.2f}% "
            f"({agree[v]}/{total}) max_logit_diff={max_diff[v]:.3f} "
            f"teacher_forced_decode top1={100 * d_agree[v] / d_total:.2f}% ({d_agree[v]}/{d_total}) "
            f"combined top1={100 * (agree[v] + d_agree[v]) / (total + d_total):.2f}%",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=0, help="first N decoder layers only (0: all)")
    parser.add_argument("--kernel", action="store_true")
    parser.add_argument("--passes", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--tokens", default="64,700,1400,2048,4096")
    parser.add_argument("--pass-tokens", default="64,256,700,2048")
    parser.add_argument("--max-layers", type=int, default=4, help="MoE layers to microbenchmark")
    parser.add_argument("--hot", type=int, default=8)
    parser.add_argument("--skew", type=float, default=4.0)
    parser.add_argument("--configs", default="", help="';'-separated BM,BN1,BK1,BN2,BK2,warps,stages")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--parity-prompts", type=int, default=48)
    parser.add_argument("--decode-steps", type=int, default=24)
    args = parser.parse_args()
    args.tokens = [int(t) for t in args.tokens.split(",")]
    args.pass_tokens = [int(t) for t in args.pass_tokens.split(",")]
    print("torch", torch.__version__, "hip", torch.version.hip, "gpus", torch.cuda.device_count(), flush=True)
    generator, fast = load(args.layers)
    if args.kernel:
        kernel_bench(generator, fast, args)
    if args.passes:
        passes(generator, fast, args)
    if args.parity:
        parity(generator, fast, args)
    print("RESULT done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
