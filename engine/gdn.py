"""Fast paths for the 30 Gated DeltaNet layers, and a warmup run before ``/health``.

Transformers 5.22.1 picks its GDN kernels when ``modeling_qwen3_5_moe`` is
imported: the recurrence from ``fla`` (flash-linear-attention) if it imports,
else its own torch code, and the causal convolution from the ``causal_conv1d``
package, else ``F.conv1d``. Its gated RMSNorm has no fast path without the
``kernels`` hub. The ROCm image has no ``causal_conv1d``, so the prefill conv is
a depthwise ``F.conv1d`` that MIOpen serves, with a new problem (and possibly a
new solver search) for almost every prompt length.

``install`` rebinds the module-level functions the GDN forward calls:

- ``causal_conv1d_fn`` / ``causal_conv1d_update``: a depthwise causal conv built
  from shifted multiply-adds (fp32 accumulation, rounded to the weight dtype,
  then SiLU: the same steps ``F.conv1d`` + ACT2FN take), so no length is ever
  new to a library and nothing goes to MIOpen.
- with ``fla`` on a GPU: ``chunk_gated_delta_rule`` (prefill) and
  ``fused_recurrent_gated_delta_rule`` (decode) Triton kernels, and fla's fused
  gated RMSNorm. These are the kernels Transformers itself selects when fla is
  installed; they read bf16 q/k/v, accumulate in fp32 and keep the recurrent
  state in fp32, as the torch fallback does.

``ENGINE_GDN`` chooses: ``auto`` (default: fla when it imports and the model is
on a GPU, else torch), ``fla`` (fail if fla cannot be used), ``torch`` (the
Transformers torch recurrence with this conv) or ``reference`` (everything as
Transformers ships it). ``warmup`` runs a few prefills and decode steps so
Triton compiles and autotunes before the server answers; ``ENGINE_WARMUP=0``
skips it.
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
# Causal depthwise convolution without MIOpen


def _taps(padded: torch.Tensor, weight: torch.Tensor, length: int, start: int) -> torch.Tensor:
    """sum_j weight[:, j] * padded[..., start + j : start + j + length], accumulated in fp32."""
    w = weight.float()
    kernel = w.shape[-1]
    acc = padded[..., start : start + length].float() * w[:, 0, None]
    for j in range(1, kernel):
        acc.addcmul_(padded[..., start + j : start + j + length], w[:, j, None])
    return acc


def _finish(acc: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype, activation: str | None):
    if bias is not None:
        acc.add_(bias.float()[:, None])
    out = acc.to(dtype)  # F.conv1d returns the weight dtype; the activation runs on that
    if activation is not None:
        if activation not in ("silu", "swish"):
            raise ValueError(f"unsupported conv activation {activation!r}")
        out = F.silu(out)
    return out


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Transformers' ``causal_conv1d_fn``: x ``[B, D, L]``, weight ``[D, K]``, output ``[B, D, L]``."""
    if kwargs.get("seq_idx") is not None or kwargs.get("cu_seqlens") is not None:
        raise ValueError("packed sequences are not supported by engine.gdn.causal_conv1d_fn")
    length = hidden_states.shape[-1]
    kernel = weight.shape[-1]
    padded = F.pad(hidden_states.to(weight.dtype), (kernel - 1, 0))
    out = _finish(_taps(padded, weight, length, 0), bias, weight.dtype, activation)
    return out.to(hidden_states.dtype)


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Transformers' ``causal_conv1d_update``: shifts ``conv_state`` in place, returns ``[B, D, L]``."""
    length = hidden_states.shape[-1]
    state_len = conv_state.shape[-1]
    kernel = weight.shape[-1]
    full = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(full[:, :, -state_len:])
    # F.conv1d(full, padding=0)[..., -length:]: output i reads full[start + i : start + i + K].
    start = full.shape[-1] - kernel + 1 - length
    if start < 0:
        raise ValueError("conv state shorter than the kernel")
    out = _finish(_taps(full, weight, length, start), bias, weight.dtype, activation)
    return out.to(hidden_states.dtype)


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
        if chosen == "reference":
            for g in _GLOBALS:
                setattr(module, g, saved[g])
            continue
        module.causal_conv1d_fn = causal_conv1d_fn
        module.causal_conv1d_update = causal_conv1d_update
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
    ids = [(7919 * i + 11) % span + 1 for i in range(max(generator._chunk, 64) + 64)]
    counters = (generator.decode_steps, generator.decoded_tokens, generator.largest_batch)
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
    """Warm up (unless ``ENGINE_WARMUP=0``) and say which GDN kernels serve."""
    seconds = None
    if os.environ.get("ENGINE_WARMUP", "1") != "0":
        seconds = warmup(generator)
    warm = "skipped" if seconds is None else f"{seconds:.1f}s"
    print(f"engine: gdn {installed}, warmup {warm}", flush=True)
