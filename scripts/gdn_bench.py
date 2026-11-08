"""GPU timing and parity of engine.gdn against Transformers' torch fallback (plan 7).

Run in the candidate image from the project root: ``python scripts/gdn_bench.py``.

1. Environment: torch, Triton, fla, GPU arch.
2. One real Gated DeltaNet layer (weights of layer 0 from the pinned checkpoint) on one GCD:
   forward time per mode (reference = Transformers' torch fallback, torch = fallback
   recurrence + engine conv, fla = engine.gdn with fla) for prefill chunks of 900 and 2048
   tokens on top of a state and decode at batch 1 and 8; output and state parity.
3. Depthwise causal conv on lengths not seen before in this process: first call vs second
   call, F.conv1d (MIOpen) vs engine.gdn.causal_conv1d_fn after its buckets are warmed.
4. The whole model through engine.Generator on every GCD: load + warmup time, prefill of
   a 2048 + 900 token prompt, decode step time at batch 1 and 8, per mode; greedy parity
   (free-running and teacher-forced top-1 agreement, max logit difference) and greedy
   repeatability of the fla path.
Prints one line per number, prefixed with ``RESULT``.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import gdn  # noqa: E402
from engine.generator import CHECKPOINT  # noqa: E402


def result(name: str, value) -> None:
    print(f"RESULT {name} = {value}", flush=True)


def timed(fn, device, reps: int = 20, warm: int = 3) -> tuple[float, float]:
    """Median and min wall ms per call, synchronized (host launch cost included)."""
    for _ in range(warm):
        fn()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times), min(times)


def once(fn, device) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e3


def environment() -> None:
    import transformers
    import triton

    result("torch", torch.__version__)
    result("hip", torch.version.hip)
    result("triton", triton.__version__)
    result("transformers", transformers.__version__)
    result("gpu", f"{torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).gcnArchName} x{torch.cuda.device_count()}")
    kernels = gdn.fla_kernels()
    result("fla", kernels.version if kernels else f"unavailable: {gdn.fla_error()}")


# ---------------------------------------------------------------------------------------------
# One layer


def load_layer(device):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

    repo, rev = CHECKPOINT["model"], CHECKPOINT["revision"]
    config = AutoConfig.from_pretrained(repo, revision=rev).get_text_config()
    layer_idx = config.layer_types.index("linear_attention")
    try:
        index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=rev)).read_text())
    except Exception:  # a single-file checkpoint
        from safetensors import safe_open as _open

        with _open(hf_hub_download(repo, "model.safetensors", revision=rev), framework="pt") as f:
            index = {"weight_map": dict.fromkeys(f.keys(), "model.safetensors")}
    suffix = f"layers.{layer_idx}.linear_attn."
    keys = [k for k in index["weight_map"] if suffix in k and "mtp" not in k and "visual" not in k]
    state = {}
    for file in sorted({index["weight_map"][k] for k in keys}):
        with safe_open(hf_hub_download(repo, file, revision=rev), framework="pt") as f:
            for k in keys:
                if index["weight_map"][k] == file:
                    state[k.split(suffix, 1)[1]] = f.get_tensor(k)
    layer = modeling.Qwen3_5MoeGatedDeltaNet(config, layer_idx).to(torch.bfloat16)
    missing, unexpected = layer.load_state_dict(state, strict=False)
    result("layer_load", f"layer {layer_idx}, {len(state)} tensors, missing {missing}, unexpected {unexpected}")
    if missing:
        raise RuntimeError(f"layer weights missing: {missing}")
    return layer.to(device).eval(), config, layer_idx


def layer_section(device) -> None:
    from transformers import DynamicCache

    layer, config, layer_idx = load_layer(device)
    torch.manual_seed(0)
    hidden = config.hidden_size

    def fresh_cache(batch: int, prime: int = 64):
        cache = DynamicCache(config=config)
        x = torch.randn(batch, prime, hidden, device=device, dtype=torch.bfloat16)
        layer(x, cache_params=cache)
        return cache

    def state(cache):
        lay = cache.layers[layer_idx]
        return lay.conv_states[0], lay.recurrent_states[0]

    modes = ["reference", "torch"] + (["fla"] if gdn.fla_kernels() else [])
    inputs = {
        f"prefill{t}": torch.randn(1, t, hidden, device=device, dtype=torch.bfloat16) for t in (900, 2048)
    }
    inputs["decode1"] = torch.randn(1, 1, hidden, device=device, dtype=torch.bfloat16)
    inputs["decode8"] = torch.randn(8, 1, hidden, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        gdn.install(layer, "reference")  # prime the states with Transformers' torch fallback
        base = {name: fresh_cache(x.shape[0]) for name, x in inputs.items()}
        outputs = {}
        for mode in modes:
            gdn.install(layer, mode)
            for name, x in inputs.items():
              try:
                cache = copy.deepcopy(base[name])
                out = layer(x, cache_params=cache)
                outputs[mode, name] = (out.float(), *[s.float() for s in state(cache)])
                work = copy.deepcopy(base[name])
                med, low = timed(lambda: layer(x, cache_params=work), device)
                result(f"layer_ms {mode} {name}", f"median {med:.3f} min {low:.3f}")
              except Exception as exc:  # one failing mode must not hide the others
                result(f"layer_failed {mode} {name}", f"{type(exc).__name__}: {exc}")
            # The pieces alone at 2048 tokens.
            x = inputs["prefill2048"]
            mixed = layer.in_proj_qkv(x).transpose(1, 2)
            from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

            conv = modeling.causal_conv1d_fn
            w = layer.conv1d.weight.squeeze(1)
            try:
                med, low = timed(lambda: conv(mixed, w, None, activation="silu"), device)
                result(f"conv_ms {mode} 2048", f"median {med:.3f} min {low:.3f}")
            except Exception as exc:
                result(f"conv_failed {mode}", f"{type(exc).__name__}: {exc}")
        gdn.install(layer, "reference")
        for mode in modes[1:]:
            for name in inputs:
                if (mode, name) not in outputs or ("reference", name) not in outputs:
                    continue
                ref, got = outputs["reference", name], outputs[mode, name]
                diffs = [float((a - b).abs().max()) for a, b in zip(got, ref, strict=True)]
                scale = float(ref[0].abs().max())
                result(
                    f"layer_parity {mode} {name}",
                    f"out maxdiff {diffs[0]:.3e} (ref max {scale:.3e}), conv_state maxdiff {diffs[1]:.3e}, "
                    f"recurrent_state maxdiff {diffs[2]:.3e} dtype {outputs[mode, name][2].dtype}",
                )
    del layer
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# Conv cold shapes


def conv_section(device) -> None:
    channels, kernel = 8192, 4
    weight = torch.randn(channels, kernel, device=device, dtype=torch.bfloat16)
    torch.backends.cudnn.benchmark = False
    # What engine.gdn.warm_conv does at startup for the prefill conv: every bucket once.
    start = time.perf_counter()
    for padded in range(gdn.BUCKET, gdn.bucket(2048 + kernel) + 1, gdn.BUCKET):
        gdn.causal_conv1d_fn(torch.zeros(1, channels, padded, device=device, dtype=torch.bfloat16), weight, None, "silu")
    torch.cuda.synchronize(device)
    result("conv_bucket_warmup_ms", f"{(time.perf_counter() - start) * 1e3:.1f}")
    for label, fn, lengths in (
        ("F.conv1d", lambda x: F.conv1d(x, weight.unsqueeze(1), padding=kernel - 1, groups=channels)[..., : x.shape[-1]], (1013, 1237, 1499, 1733, 1877)),
        ("engine", lambda x: gdn.causal_conv1d_fn(x, weight, None, "silu"), (611, 777, 1111, 1999, 1301)),
        ("F.conv1d", lambda x: F.conv1d(x, weight.unsqueeze(1), padding=kernel - 1, groups=channels)[..., : x.shape[-1]], (611, 777, 1111, 1999, 1301)),
        ("engine", lambda x: gdn.causal_conv1d_fn(x, weight, None, "silu"), (1013, 1237, 1499, 1733, 1877)),
    ):
        firsts, seconds = [], []
        for length in lengths:
            x = torch.randn(1, channels, length, device=device, dtype=torch.bfloat16)
            firsts.append(once(lambda: fn(x), device))
            seconds.append(once(lambda: fn(x), device))
        result(
            f"conv_cold {label} lengths {lengths}",
            f"first ms {[round(v, 2) for v in firsts]} second ms {[round(v, 2) for v in seconds]}",
        )


# ---------------------------------------------------------------------------------------------
# Whole model


def prompt_ids(tokenizer, length: int, seed_modules) -> list[int]:
    text = "\n\n".join(inspect.getsource(m) for m in seed_modules)
    ids = tokenizer.encode(text)
    while len(ids) < length:
        ids = ids + ids
    return ids[:length]


def model_section(steps: int) -> None:
    import email.message
    import json.decoder
    import textwrap

    from transformers import DynamicCache

    from engine.generator import Generator

    os.environ.setdefault("ENGINE_GDN", "auto")
    start = time.perf_counter()
    generator = Generator.load(
        CHECKPOINT["model"],
        CHECKPOINT["revision"],
        max_model_len=262144,
        chunk_tokens=2048,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )
    result("model_load_s_including_warmup", f"{time.perf_counter() - start:.1f}")
    model = generator.model
    first_dev = generator._device
    devices = sorted({p.device for p in model.parameters()}, key=str)

    def sync():
        for d in devices:
            torch.cuda.synchronize(d)

    prompts = [
        prompt_ids(generator.tokenizer, 2948, [json.decoder, textwrap]),
        prompt_ids(generator.tokenizer, 1500, [email.message]),
    ]
    modes = ["reference"] + (["fla"] if gdn.fla_kernels() else ["torch"])

    def prefill(ids):
        cache = DynamicCache(config=model.config)
        times = []
        logits = None
        for begin in range(0, len(ids), 2048):
            part = ids[begin : begin + 2048]
            sync()
            t = time.perf_counter()
            logits = generator.forward(part, cache)
            sync()
            times.append((len(part), (time.perf_counter() - t) * 1e3))
        return cache, logits, times

    def greedy(ids, forced=None):
        cache, logits, _ = prefill(ids)
        tokens, all_logits = [], [logits]
        for i in range(steps):
            token = int(logits.argmax())
            tokens.append(token)
            feed = forced[i] if forced is not None else token
            logits = generator.forward([feed], cache)
            all_logits.append(logits)
        return tokens, torch.stack(all_logits)

    reference = {}
    with torch.inference_mode():
        for mode in modes:
            result(f"install {mode}", gdn.install(model, mode))
            for rep in range(3):  # rep 0 sees cold shapes (MIOpen) in reference mode
                _, _, times = prefill(prompts[0])
                result(f"model_prefill_ms {mode} rep{rep}", ", ".join(f"{n} tok {ms:.1f}" for n, ms in times))
            # Decode, batch 1 and batch 8 (stacked caches, as Generator.decode does).
            cache, logits, _ = prefill(prompts[1][:600])
            token = int(logits.argmax())
            per_step = []
            for _ in range(24):
                sync()
                t = time.perf_counter()
                logits = generator.forward([token], cache)
                token = int(logits.argmax())
                per_step.append((time.perf_counter() - t) * 1e3)
            result(f"model_decode_ms {mode} B1", f"median {statistics.median(per_step[4:]):.2f}")
            vocab = model.config.get_text_config().vocab_size
            requests = [gdn._WarmRequest(copy.deepcopy(cache), (token + i) % vocab) for i in range(8)]
            per_step = []
            for _ in range(16):
                sync()
                t = time.perf_counter()
                generator.decode(requests)
                sync()
                per_step.append((time.perf_counter() - t) * 1e3)
                for r in requests:
                    r.last_token = int(r.logits.argmax())
            result(f"model_decode_ms {mode} B8", f"median {statistics.median(per_step[4:]):.2f}")
            del requests, cache

            for p, ids in enumerate(prompts):
                tokens, logits = greedy(ids)
                if mode == "reference":
                    reference[p] = (tokens, logits)
                    continue
                ref_tokens, ref_logits = reference[p]
                same_prefix = next((i for i, (a, b) in enumerate(zip(tokens, ref_tokens)) if a != b), steps)
                result(f"parity free-running prompt{p}", f"identical greedy tokens for {same_prefix}/{steps}")
                forced_tokens, forced_logits = greedy(ids, forced=ref_tokens)
                agree = sum(int(forced_logits[i].argmax()) == ref_tokens[i] for i in range(steps))
                diff = (forced_logits - ref_logits).abs().amax(-1)
                result(
                    f"parity teacher-forced prompt{p}",
                    f"top1 agreement {agree}/{steps}, max logit diff {float(diff.max()):.4f}, "
                    f"prefill logit diff {float(diff[0]):.4f}, ref logit range {float(ref_logits.abs().max()):.2f}",
                )
                again, _ = greedy(ids)
                result(f"repeatable {mode} prompt{p}", again == tokens)
    gdn.install(model, "reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--skip", nargs="*", default=[], choices=["layer", "conv", "model"])
    parser.add_argument("--device", default="cuda:0", help="device of the layer and conv sections")
    args = parser.parse_args()
    environment()
    device = torch.device(args.device)
    for name, section in (
        ("layer", lambda: layer_section(device)),
        ("conv", lambda: conv_section(device)),
        ("model", lambda: model_section(args.steps)),
    ):
        if name in args.skip:
            continue
        start = time.perf_counter()
        try:
            section()
        except Exception as exc:  # report and go on: one failing section must not hide the others
            import traceback

            traceback.print_exc()
            result(f"section_failed {name}", f"{type(exc).__name__}: {exc}")
        result(f"section_s {name}", f"{time.perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
