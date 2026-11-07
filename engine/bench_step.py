"""Stage-0 feasibility check of the engine's own forward (``engine.fast``) on the real checkpoint.

    python -m engine.bench_step [--steps 64] [--timing-steps 30]

Loads the checkpoint exactly as the server does (``Generator.load``: BF16,
``device_map="auto"`` over the visible GPUs, the engine's experts
implementation), builds ``FastModel`` over the same weight tensors, then:

1. Parity. Four prompts of about 1k-4k tokens (the text of source files from
   the installed Transformers package, so nothing comes from the benchmark)
   run through the Transformers path: chunked prefill, then greedy decode with
   ``Generator.decode``. The fast path prefills the same chunks into its slots
   and decodes the same tokens (teacher forcing), all four prompts in one
   batch, and its logits are compared step by step: top-1 agreement, maximum
   absolute logit difference, NaNs. Its own free-running greedy tokens are
   compared too.
2. Step time. One decode step at batch 1, 2 and 8 for the Transformers path
   (``Generator.decode``, which merges and splits the per-request caches) and
   for the fast path (graph replay, and eager without graphs), each averaged
   over several steps after warmup, synchronizing every device.

It prints one ``RESULT`` line per number and a final verdict against the
plan's stage-0 criteria.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from engine.generator import CHECKPOINT, Generator
from engine.protocol import Sampling


def _sync() -> None:
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)


def _prompts(generator: Generator, lengths: list[int]) -> list[list[int]]:
    import transformers

    root = Path(transformers.__file__).parent
    files = sorted(root.glob("models/*/modeling_*.py"))
    prompts, index = [], 0
    for length in lengths:
        while True:
            ids = list(generator.tokenizer.encode(files[index].read_text()))
            index += 1
            if len(ids) >= length:
                prompts.append(ids[:length])
                break
    return prompts


def _greedy(generator: Generator, prompt: list[int], steps: int):
    """Transformers path: prefill, then ``steps`` greedy tokens; tokens and fp32 logits."""
    sampling = Sampling(max_tokens=steps, temperature=0.0, ignore_eos=True)
    generation = generator.start(tuple(prompt), sampling)
    tokens, logits = [], [generation.logits.float().cpu()]
    for _ in range(steps):
        step = generation.next_step()
        tokens.append(step.token_id)
        generator.decode([generation])
        logits.append(generation.logits.float().cpu())
    return tokens, logits


def _fast_prefill(fast, slot: int, prompt: list[int], chunk: int) -> torch.Tensor:
    body = len(prompt) - 1
    position = 0
    while position < body:
        end = min((position // chunk + 1) * chunk, body)
        fast.prefill(slot, prompt[position:end], position)
        position = end
    return fast.prefill(slot, prompt[body:], body)


def _time(fn, steps: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    _sync()
    return (time.perf_counter() - start) / steps * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CHECKPOINT["model"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--steps", type=int, default=64, help="greedy tokens per parity prompt")
    parser.add_argument("--timing-steps", type=int, default=30)
    parser.add_argument("--hf-timing-steps", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--max-len", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--prompt-lengths", default="1024,2048,3072,4000")
    parser.add_argument("--timing-length", type=int, default=2048)
    args = parser.parse_args(argv)
    revision = args.revision
    if revision is None and args.model == CHECKPOINT["model"]:
        revision = CHECKPOINT["revision"]

    print("torch", torch.__version__, "hip", torch.version.hip, "gpus", torch.cuda.device_count())
    try:
        import triton

        print("triton", triton.__version__)
    except ImportError:
        print("triton not installed")
    t0 = time.perf_counter()
    generator = Generator.load(
        args.model,
        revision,
        max_model_len=args.max_len,
        chunk_tokens=args.chunk,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )
    print(f"RESULT load_s={time.perf_counter() - t0:.1f}")
    lm = generator.model
    text_cfg = lm.config.get_text_config()
    keys = (
        "hidden_size num_hidden_layers num_experts num_experts_per_tok moe_intermediate_size "
        "shared_expert_intermediate_size num_attention_heads num_key_value_heads head_dim "
        "linear_num_key_heads linear_num_value_heads linear_key_head_dim linear_value_head_dim "
        "linear_conv_kernel_dim vocab_size max_position_embeddings"
    ).split()
    print("config", {k: getattr(text_cfg, k, None) for k in keys})
    print("rope", text_cfg.rope_parameters)
    print("experts_implementation", getattr(text_cfg, "_experts_implementation", None))
    placement: dict[str, int] = {}
    for p in lm.parameters():
        placement[str(p.device)] = placement.get(str(p.device), 0) + p.numel() * p.element_size()
    print("weights_GB_per_device", {k: round(v / 1e9, 2) for k, v in sorted(placement.items())})

    from engine.fast.model import FastModel

    t0 = time.perf_counter()
    fast = FastModel(lm, slots=args.slots, max_len=args.max_len)
    print(
        f"fast: triton={fast.use_triton} graphs={fast.use_graphs} kv_pages={fast.pool.pages} "
        f"(={fast.pool.pages * 64} tokens per attention layer) segments="
        f"{[str(s.device) for s in fast.segments]}"
    )
    fast.capture_all()
    _sync()
    print(f"RESULT capture_s={time.perf_counter() - t0:.1f} graph_error={fast.graph_error}")
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(d)
            print(f"mem cuda:{d} used_GB={(total - free) / 1e9:.1f} of {total / 1e9:.1f}")

    # ---------------------------------------------------------------- parity
    lengths = [int(n) for n in args.prompt_lengths.split(",")]
    prompts = _prompts(generator, lengths)
    reference = []
    for prompt in prompts:
        t0 = time.perf_counter()
        reference.append(_greedy(generator, prompt, args.steps))
        print(f"hf prompt {len(prompt)}: {time.perf_counter() - t0:.1f} s")
    # Top-1 agreement on every token the reference chose: the prefill's logits chose token 0,
    # the logits after feeding token j-1 chose token j (teacher forcing, all prompts in one batch).
    agree, total, nan, max_diff, prefill_diff = 0, 0, False, 0.0, 0.0
    for slot, prompt in enumerate(prompts):
        fast.reset(slot)
        t0 = time.perf_counter()
        logits = _fast_prefill(fast, slot, prompt, args.chunk).float().cpu()
        _sync()
        print(f"fast prefill {len(prompt)}: {time.perf_counter() - t0:.2f} s")
        ref = reference[slot][1][0]
        prefill_diff = max(prefill_diff, float((logits - ref).abs().max()))
        nan |= bool(torch.isnan(logits).any())
        total += 1
        agree += int(int(logits.argmax()) == reference[slot][0][0])
    rows = list(range(len(prompts)))
    for step in range(args.steps - 1):
        tokens = [reference[r][0][step] for r in rows]
        out = fast.decode(tokens, rows, [len(prompts[r]) + step for r in rows]).float().cpu()
        for r in rows:
            ref = reference[r][1][step + 1]
            total += 1
            agree += int(int(out[r].argmax()) == reference[r][0][step + 1])
            nan |= bool(torch.isnan(out[r]).any())
            max_diff = max(max_diff, float((out[r] - ref).abs().max()))
    agree_pct = 100.0 * agree / total
    print(f"RESULT parity_top1_pct={agree_pct:.2f} ({agree}/{total} tokens) nan={nan} "
          f"max_abs_logit_diff_decode={max_diff:.3f} max_abs_logit_diff_prefill={prefill_diff:.3f}")

    # Free-running greedy on the fast path, batch of four: tokens equal to the reference's.
    free_tokens = [[] for _ in prompts]
    current = [None] * len(prompts)
    for slot, prompt in enumerate(prompts):
        fast.reset(slot)
        current[slot] = int(_fast_prefill(fast, slot, prompt, args.chunk).argmax())
    for step in range(args.steps):
        for row in range(len(prompts)):
            free_tokens[row].append(current[row])
        out = fast.decode(current, list(range(len(prompts))), [len(p) + step for p in prompts])
        current = [int(x) for x in out.argmax(dim=-1).tolist()]
    same_prefix = []
    for row in range(len(prompts)):
        n = 0
        while n < args.steps and free_tokens[row][n] == reference[row][0][n]:
            n += 1
        same_prefix.append(n)
    print(f"RESULT free_running_identical_prefix={same_prefix} of {args.steps}")

    # ---------------------------------------------------------------- step time
    timing_prompts = _prompts(generator, [args.timing_length] * 8)
    sampling = Sampling(max_tokens=1024, temperature=0.0, ignore_eos=True)
    hf_gens = []
    for prompt in timing_prompts:
        g = generator.start(tuple(prompt), sampling)
        g.next_step()
        hf_gens.append(g)
    fast_pos = []
    for slot, prompt in enumerate(timing_prompts):
        fast.reset(slot)
        _fast_prefill(fast, slot, prompt, args.chunk)
        fast_pos.append(len(prompt))
    results = {}
    for batch in (1, 2, 8):
        gens = hf_gens[:batch]
        results[f"hf_b{batch}"] = _time(lambda: generator.decode(gens), args.hf_timing_steps, 2)

        def fast_step(batch=batch):
            slots = list(range(batch))
            fast.decode([100] * batch, slots, [fast_pos[s] for s in slots])
            for s in slots:
                fast_pos[s] += 1

        results[f"fast_b{batch}"] = _time(fast_step, args.timing_steps)
        graphs = fast.use_graphs
        fast.use_graphs = False
        results[f"fast_eager_b{batch}"] = _time(fast_step, max(5, args.timing_steps // 3), 2)
        fast.use_graphs = graphs
        print(
            f"RESULT step_ms batch={batch} hf={results[f'hf_b{batch}']:.1f} "
            f"fast_graph={results[f'fast_b{batch}']:.2f} fast_eager={results[f'fast_eager_b{batch}']:.1f} "
            f"speedup_graph={results[f'hf_b{batch}'] / results[f'fast_b{batch}']:.1f}x"
        )

    # ---------------------------------------------------------------- verdict
    graph_ok = fast.use_graphs and fast.graph_error is None
    b8 = results["fast_b8"] if graph_ok else results["fast_eager_b8"]
    speedup = results["hf_b8"] / b8
    need = 5.0 if graph_ok else 3.0
    checks = {
        "parity_top1>=97%": agree_pct >= 97.0,
        "no_nan": not nan,
        f"b8_speedup>={need:.0f}x": speedup >= need,
        "b8_step<40ms": b8 < 40.0,
    }
    print(f"RESULT b8_speedup={speedup:.1f}x graphs={graph_ok} checks={checks}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(f"RESULT stage0={verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
