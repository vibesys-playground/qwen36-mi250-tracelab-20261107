"""Fast paths for the 30 Gated DeltaNet layers, and a warmup run before ``/health``.

Transformers 5.22.1 picks its GDN kernels when ``modeling_qwen3_5_moe`` is
imported: the recurrence from ``fla`` (flash-linear-attention) if it imports,
else its own torch code, and the causal convolution from the ``causal_conv1d``
package, else ``F.conv1d``. Its gated RMSNorm has no fast path without the
``kernels`` hub. The ROCm image has no ``causal_conv1d``, so the prefill conv is
a depthwise ``F.conv1d`` that MIOpen serves, with a new problem (compiled on
first use) for almost every prompt length.

``install`` rebinds the module-level functions the GDN forward calls:

- ``causal_conv1d_fn``: the same ``F.conv1d`` call, on the input right-padded
  to a multiple of ``BUCKET`` positions, so MIOpen sees a few lengths that
  ``warm_conv`` compiles at startup instead of a new one per prompt (30-110 ms
  each on gfx90a, j23). Outputs are those of the unpadded call. The decode
  ``causal_conv1d_update`` stays Transformers' own; its batch sizes are warmed.
- with ``fla`` on a GPU: ``chunk_gated_delta_rule`` (prefill) and
  ``fused_recurrent_gated_delta_rule`` (decode) Triton kernels, and fla's fused
  gated RMSNorm. These are the kernels Transformers itself selects when fla is
  installed; they read bf16 q/k/v, accumulate in fp32 and keep the recurrent
  state in fp32, as the torch fallback does.

``ENGINE_GDN`` chooses: ``auto`` (default: fla when it imports and the model is
on a GPU, else torch), ``fla`` (fail if fla cannot be used), ``torch`` (the
Transformers torch recurrence with this conv) or ``reference`` (Transformers'
own torch fallback for all of it, as served without fla). ``warmup`` compiles the
conv problems and runs a few prefills and decode steps so Triton compiles and
autotunes before the server answers; ``ENGINE_WARMUP=0`` skips it.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from torch.nn import functional as F

MODES = ("auto", "fla", "torch", "reference")
_GLOBALS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)
# Per modeling module: the functions Transformers bound there, saved before the first rebind.
_ORIGINAL: dict[str, dict[str, Any]] = {}
# Per gated-norm class: its own forward.
_ORIGINAL_NORM: dict[type, Any] = {}


# ---------------------------------------------------------------------------------------------
# Causal depthwise convolution on a fixed set of MIOpen problems

# Prefill conv inputs are right-padded to a multiple of this many positions. Zeros after the
# last position never reach a causal output that is kept, so the result is the same; MIOpen
# then sees at most chunk/BUCKET + 1 lengths, all compiled by ``warm_conv`` before serving.
BUCKET = 256
# Decode batch sizes whose conv update ``warm_conv`` compiles.
WARM_BATCH = 16


def bucket(length: int) -> int:
    return -(-length // BUCKET) * BUCKET


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Transformers' ``causal_conv1d_fn`` (x ``[B, D, L]``, weight ``[D, K]``) on a bucketed length.

    The same F.conv1d call as Transformers' torch path, so the same kernel and dtype
    steps: conv in the weight dtype, then SiLU, then the input dtype.
    """
    if kwargs.get("seq_idx") is not None or kwargs.get("cu_seqlens") is not None:
        raise ValueError("packed sequences are not supported by engine.gdn.causal_conv1d_fn")
    channels, length = hidden_states.shape[-2:]
    kernel = weight.shape[-1]
    x = hidden_states.to(weight.dtype)
    padded = bucket(length)
    # Always contiguous, so a bucket is one MIOpen problem whatever the caller's strides.
    x = F.pad(x, (0, padded - length)) if padded != length else x.contiguous()
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=kernel - 1, groups=channels)[..., :length]
    if activation is not None:
        if activation not in ("silu", "swish"):
            raise ValueError(f"unsupported conv activation {activation!r}")
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def warm_conv(model: torch.nn.Module, max_length: int) -> int:
    """Run the prefill conv at every bucket up to ``max_length`` and the decode conv update at
    batch 1..WARM_BATCH, once per device and conv shape, so MIOpen compiles them now.
    Returns the number of problems run."""
    layers = gdn_modules(model)
    if not layers:
        return 0
    module = sys.modules[type(layers[0]).__module__]
    update = module.causal_conv1d_update
    seen, count = set(), 0
    with torch.inference_mode():
        for layer in layers:
            weight = layer.conv1d.weight.squeeze(1)
            key = (weight.device, weight.dtype, tuple(weight.shape))
            if key in seen:
                continue
            seen.add(key)
            channels, kernel = weight.shape
            for padded in range(BUCKET, bucket(max_length) + 1, BUCKET):
                x = torch.zeros(1, channels, padded, dtype=weight.dtype, device=weight.device)
                causal_conv1d_fn(x, weight, layer.conv1d.bias, activation=layer.activation)
                count += 1
            for batch in range(1, WARM_BATCH + 1):
                state = torch.zeros(batch, channels, kernel, dtype=weight.dtype, device=weight.device)
                x = torch.zeros(batch, channels, 1, dtype=weight.dtype, device=weight.device)
                update(x, state, weight, layer.conv1d.bias, layer.activation)
                count += 1
    return count


# ---------------------------------------------------------------------------------------------
# flash-linear-attention kernels


@dataclass(frozen=True)
class _Fla:
    version: str
    chunk: Any
    recurrent: Any
    rms_norm_gated: Any


_FLA: _Fla | None = None
_FLA_ERROR: str | None = None


def fla_kernels() -> _Fla | None:
    """fla's GDN kernels, or None (``fla_error()`` says why)."""
    global _FLA, _FLA_ERROR
    if _FLA is not None or _FLA_ERROR is not None:
        return _FLA
    try:
        import fla
        from fla.modules.fused_norm_gate import rms_norm_gated
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        _FLA = _Fla(
            getattr(fla, "__version__", "?"),
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
            rms_norm_gated,
        )
    except Exception as exc:  # ImportError, or Triton failing to find a backend
        _FLA_ERROR = f"{type(exc).__name__}: {exc}"
    return _FLA


def fla_error() -> str | None:
    return _FLA_ERROR


def _fla_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
):
    kernels = fla_kernels()
    with torch.cuda.device(query.device):
        return kernels.chunk(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )


fla_chunk = _fla_chunk  # what ``install`` binds for the chunked rule under ENGINE_GDN=fla


def _fla_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
):
    kernels = fla_kernels()
    with torch.cuda.device(query.device):
        return kernels.recurrent(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )


def _fla_norm_forward(self: torch.nn.Module, hidden_states: torch.Tensor, gate: torch.Tensor):
    kernels = fla_kernels()
    with torch.cuda.device(hidden_states.device):
        return kernels.rms_norm_gated(
            hidden_states,
            gate,
            self.weight,
            None,
            activation=self.activation,
            eps=self.variance_epsilon,
        )


# ---------------------------------------------------------------------------------------------
# Installing


def gdn_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]


def _unwrapped(fn: Any) -> Any:
    """The torch function under Transformers' fla/causal_conv1d dispatch wrapper."""
    return getattr(fn, "__wrapped__", fn)


def _installed_modules(model: torch.nn.Module) -> list[ModuleType]:
    return [sys.modules[n] for n in {type(m).__module__ for m in gdn_modules(model)}]


def resolve_mode(model: torch.nn.Module, mode: str | None = None) -> str:
    mode = (mode or os.environ.get("ENGINE_GDN", "auto")).strip().lower()
    if mode not in MODES:
        raise ValueError(f"ENGINE_GDN must be one of {MODES}, not {mode!r}")
    on_gpu = any(p.is_cuda for p in model.parameters())
    if mode == "auto":
        return "fla" if on_gpu and fla_kernels() is not None else "torch"
    if mode == "fla" and (not on_gpu or fla_kernels() is None):
        raise RuntimeError(f"ENGINE_GDN=fla but fla is unusable here: {fla_error() or 'model on CPU'}")
    return mode


def install(model: torch.nn.Module, mode: str | None = None) -> dict[str, Any]:
    """Bind the GDN kernels of ``mode`` (see the module docstring); returns what was chosen.

    May be called again with another mode: the original functions are kept.
    """
    layers = gdn_modules(model)
    chosen = resolve_mode(model, mode)
    modules: dict[str, ModuleType] = {type(m).__module__: sys.modules[type(m).__module__] for m in layers}
    norms = {type(m.norm) for m in layers if hasattr(m, "norm")}
    for name, module in modules.items():
        saved = _ORIGINAL.setdefault(name, {g: getattr(module, g) for g in _GLOBALS})
        if chosen == "reference":  # Transformers' torch fallback, even where it found fla
            for g in _GLOBALS:
                setattr(module, g, _unwrapped(saved[g]))
            continue
        module.causal_conv1d_fn = causal_conv1d_fn
        module.causal_conv1d_update = _unwrapped(saved["causal_conv1d_update"])
        if chosen == "fla":
            module.torch_chunk_gated_delta_rule = _fla_chunk
            module.torch_recurrent_gated_delta_rule = _fla_recurrent
        else:
            module.torch_chunk_gated_delta_rule = _unwrapped(saved["torch_chunk_gated_delta_rule"])
            module.torch_recurrent_gated_delta_rule = _unwrapped(saved["torch_recurrent_gated_delta_rule"])
    for cls in norms:
        original = _ORIGINAL_NORM.setdefault(cls, cls.forward)
        cls.forward = _fla_norm_forward if chosen == "fla" else original
    kernels = fla_kernels() if chosen == "fla" else None
    return {
        "mode": chosen,
        "layers": len(layers),
        "fla": kernels.version if kernels else None,
        "fla_error": fla_error(),
    }


# ---------------------------------------------------------------------------------------------
# Warmup


@dataclass
class _WarmRequest:
    """The fields ``Generator.decode`` reads and writes on a request."""

    cache: Any
    last_token: int
    logits: Any = None


def warmup(generator: Any) -> float:
    """Run every kernel variant serving uses once, outside any request; returns seconds taken.

    Prefill a fresh cache over a length that is not a multiple of 16 and the
    chunk size, prefill on top of a state (the initial-state kernels), decode
    one request alone and two together. Uses its own caches, so the prefix
    cache and the decode counters are left as they were.
    """
    from transformers import DynamicCache

    started = time.perf_counter()
    model = generator.model
    vocab = int(model.config.get_text_config().vocab_size)
    span = max(1, min(vocab - 1, 4096))
    # As many ids as the longest prefill pass (``Generator.warmup_tokens``), plus some.
    longest = max(generator._chunk, getattr(generator, "warmup_tokens", 0), 64)
    ids = [(7919 * i + 11) % span + 1 for i in range(longest + 64)]
    counters = (generator.decode_steps, generator.decoded_tokens, generator.largest_batch)
    if getattr(generator, "fast", None) is not None:
        # The engine's own forward: its prefills (the GDN chunk kernel installed here, its
        # own bucketed conv, the prefill experts); its decode graphs were captured at load.
        generator.warmup_fast(ids)
        return time.perf_counter() - started
    if any(module.causal_conv1d_fn is causal_conv1d_fn for module in _installed_modules(model)):
        state_len = int(model.config.get_text_config().linear_conv_kernel_dim)
        warm_conv(model, generator._chunk + state_len)
    with torch.inference_mode():
        first = DynamicCache(config=model.config)
        generator.forward(ids[:37], first)  # fresh prefill, no initial state
        generator.forward(ids[37:101], first)  # on top of a state, length % 16 == 0
        second = DynamicCache(config=model.config)
        generator.forward(ids[: generator._chunk], second)  # a whole chunk
        requests = [_WarmRequest(first, ids[101]), _WarmRequest(second, ids[102])]
        generator.decode(requests[:1])
        generator.decode(requests)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generator.decode_steps, generator.decoded_tokens, generator.largest_batch = counters
    return time.perf_counter() - started


def ready(generator: Any, installed: dict[str, Any]) -> None:
    """Warm up (unless ``ENGINE_WARMUP=0``) and say which GDN kernels serve.

    Under ``ENGINE_GDN=auto``, fla kernels that fail to compile or run during the
    warmup are replaced by the torch path before the server opens.
    """
    seconds = None
    if os.environ.get("ENGINE_WARMUP", "1") != "0":
        try:
            seconds = warmup(generator)
        except Exception as exc:
            explicit = os.environ.get("ENGINE_GDN", "auto").strip().lower() != "auto"
            if installed["mode"] != "fla" or explicit:
                raise
            print(f"engine: fla GDN kernels failed in warmup ({type(exc).__name__}: {exc}); using torch", flush=True)
            installed = install(generator.model, "torch")
            seconds = warmup(generator)
    warm = "skipped" if seconds is None else f"{seconds:.1f}s"
    print(f"engine: gdn {installed}, warmup {warm}", flush=True)
