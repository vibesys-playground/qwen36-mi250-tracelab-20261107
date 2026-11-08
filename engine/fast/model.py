"""The engine's own Qwen3.6 (``qwen3_5_moe``) text forward over Transformers' loaded weights.

``FastModel`` reads the parameters of a loaded ``Qwen3_5MoeForConditionalGeneration``
in place (no copy, same BF16 tensors, same devices as ``device_map`` chose)
and runs the text model with state it owns:

- Sequence state lives in fixed slots allocated once: per Gated DeltaNet layer a
  BF16 convolution state ``[slots, conv_dim, kernel]`` and an fp32 recurrent
  state ``[slots, v_heads, k_dim, v_dim]`` (the dtypes Transformers' cache
  uses); per attention layer a pool of KV pages ``[pages, kv_heads, page,
  head_dim]`` with a page table per slot. A request owns a slot; a decode batch
  is a tensor of slot indices, so nothing is merged or split between steps.
- ``decode`` runs one token for each of several slots. On GPUs it replays CUDA/HIP
  graphs captured per batch-size bucket, one graph per run of layers on the
  same device, with the hidden state copied between devices in between.
- ``prefill`` runs a chunk of one sequence's tokens into its slot, eagerly.
- ``snapshot``/``restore`` copy a slot's state for the prefix cache; full KV
  pages are shared by reference count, the partial last page is copied.

Every layer computes what Transformers' modules compute (RMSNorm with
``1 + weight``, the gated delta rule with its short convolution, gated
attention with partial rotary embedding, softmax top-k routing with
renormalization over all experts plus the gated shared expert), at the same
precision; only the kernels differ (see ``engine.fast.kernels``).
"""

from __future__ import annotations

import math
import os
import traceback
from pathlib import Path
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

from engine import gdn
from engine.fast import kernels, moe_prefill

PAGE = 64
# Above this many tokens a prefill chunk's experts run through Transformers' experts module
# (the implementation the engine registered); up to it, through the decode kernel.
DECODE_EXPERT_TOKENS = 16
# Bytes of attention scores one prefill attention call may materialize.
SCORE_BUDGET = 1 << 30
# Prefill conv inputs are right-padded to a multiple of this many positions (``prefill_conv``).
CONV_BUCKET = 256


class FastPathUnavailable(RuntimeError):
    """The loaded model is not one this forward implements."""


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5's zero-centered RMSNorm: fp32 normalization times ``1 + weight``."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


def _gated_rms(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Qwen3_5MoeRMSNormGated``: normalize, scale in BF16, multiply by silu(gate) in fp32."""
    dtype = gate.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = weight * xf.to(dtype)
    return (y * F.silu(gate.float())).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Partial rotary embedding on the first ``cos.shape[-1]`` dims; ``x`` is [tokens, heads, dim]."""
    rot = cos.shape[-1]
    cos, sin = cos[:, None, :], sin[:, None, :]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    return torch.cat((x_rot * cos + _rotate_half(x_rot) * sin, x_pass), dim=-1)


def prefill_conv(window: torch.Tensor, weight: torch.Tensor, count: int) -> torch.Tensor:
    """Depthwise conv over ``window`` ``[conv_dim, kernel + count]`` (the slot's conv state,
    then the chunk's inputs): the last ``count`` outputs ``[conv_dim, count]``, before the
    activation, as ``F.conv1d(window)[..., -count:]``.

    The input is right-padded with zeros to a multiple of ``CONV_BUCKET`` positions,
    so MIOpen sees a few problem sizes (compiled by ``FastModel.warmup``) instead of a
    new one per chunk length; an output only reads inputs at or before its last tap,
    so the kept outputs are those of the unpadded call.
    """
    length = window.shape[-1]
    valid = length - weight.shape[-1] + 1  # outputs of the unpadded call
    padded = -(-length // CONV_BUCKET) * CONV_BUCKET
    x = F.pad(window, (0, padded - length)).unsqueeze(0)
    return F.conv1d(x, weight, None, groups=window.shape[0])[0, :, valid - count : valid]


class PagePool:
    """Host-side allocator of KV page ids, with reference counts for shared prefix pages."""

    def __init__(self, pages: int) -> None:
        self.pages = pages
        self._free = list(range(pages - 1, -1, -1))
        self._refs = [0] * pages
        self.on_pressure: Callable[[], bool] | None = None

    @property
    def free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        while not self._free:
            if self.on_pressure is None or not self.on_pressure():
                raise RuntimeError("the KV cache is full")
        page = self._free.pop()
        self._refs[page] = 1
        return page

    def ref(self, page: int) -> None:
        self._refs[page] += 1

    def unref(self, page: int) -> None:
        self._refs[page] -= 1
        if self._refs[page] == 0:
            self._free.append(page)


@dataclass
class Snapshot:
    """A copy of one sequence's model state after ``length`` tokens."""

    length: int
    gdn: list[tuple[torch.Tensor, torch.Tensor]]
    pages: list[int]
    released: bool = False


@dataclass
class _Device:
    """Per-device inputs of a decode step and the page table, kept at fixed addresses."""

    device: torch.device
    packed: torch.Tensor  # [4, longest bucket]: tokens, slots, positions, GDN rows; one copy
    page_table: torch.Tensor

    @property
    def tokens(self) -> torch.Tensor:
        return self.packed[0]

    @property
    def slots(self) -> torch.Tensor:
        return self.packed[1]

    @property
    def positions(self) -> torch.Tensor:
        return self.packed[2]

    @property
    def gdn(self) -> torch.Tensor:
        """Each row's Gated DeltaNet state row (its slot's current bank, see ``FastModel.bank``)."""
        return self.packed[3]


@dataclass
class _Segment:
    """A run of consecutive ops on one device, captured as one graph per batch bucket."""

    device: torch.device
    ops: list[Callable[[torch.Tensor, _Device], torch.Tensor]]
    inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    graphs: dict[int, torch.cuda.CUDAGraph] = field(default_factory=dict)

    def run(self, x: torch.Tensor, meta: _Device) -> torch.Tensor:
        for op in self.ops:
            x = op(x, meta)
        return x


@dataclass
class _PackOnDevice:
    """A packed prefill's index tensors on one device (see ``_PackPlan``)."""

    positions: torch.Tensor  # [tokens] absolute position of each packed row
    pages: torch.Tensor  # [tokens] KV page of each row
    offsets: torch.Tensor  # [tokens] offset in that page
    last: torch.Tensor  # [segments] each segment's last row
    tables: list[torch.Tensor]  # per segment: its used pages, for attention
    spans: list[tuple[int, int, int]]  # per segment: first row, rows, keys after it
    single_rows: torch.Tensor | None  # rows of one-token segments
    single_slots: torch.Tensor | None  # (their Gated DeltaNet state rows, ``FastModel._g``)
    multi_rows: torch.Tensor | None  # rows of longer segments, in packed order
    multi_slots: torch.Tensor | None
    multi_spans: list[tuple[int, int, int]]  # per longer segment: state row, first row in multi_rows, rows
    pad_index: torch.Tensor | None  # multi row j -> its index in [segments x longest]
    cu_seqlens: torch.Tensor | None  # [longer segments + 1] row offsets in the multi rows
    longest: int


class _PackPlan:
    """Where each packed row comes from, built once on the host, copied to each device once."""

    def __init__(self, model: FastModel, segments: Sequence[tuple[int, Sequence[int], int]]) -> None:
        tokens, positions, pages, offsets, last = [], [], [], [], []
        self.spans: list[tuple[int, int, int]] = []
        self.used: list[list[int]] = []
        single_rows, single_slots, multi_rows, multi_slots = [], [], [], []
        multi_spans, pad_index = [], []
        longest = max((len(ids) for _, ids, _ in segments if len(ids) > 1), default=0)
        row = 0
        for slot, ids, start in segments:
            count = len(ids)
            total = start + count
            table = model.slot_pages[slot]
            tokens.extend(ids)
            for pos in range(start, total):
                positions.append(pos)
                pages.append(table[pos // PAGE])
                offsets.append(pos % PAGE)
            self.spans.append((row, count, total))
            self.used.append(table[: math.ceil(total / PAGE)])
            state = model._g(slot)  # the slot's Gated DeltaNet state row
            if count == 1:
                single_rows.append(row)
                single_slots.append(state)
            else:
                index = len(multi_spans)
                multi_spans.append((state, len(multi_rows), count))
                pad_index.extend(range(index * longest, index * longest + count))
                multi_rows.extend(range(row, row + count))
                multi_slots.append(state)
            row += count
            last.append(row - 1)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self._host = {
            "positions": torch.tensor(positions, dtype=torch.long),
            "pages": torch.tensor(pages, dtype=torch.long),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "last": torch.tensor(last, dtype=torch.long),
            "single_rows": torch.tensor(single_rows, dtype=torch.long) if single_rows else None,
            "single_slots": torch.tensor(single_slots, dtype=torch.long) if single_slots else None,
            "multi_rows": torch.tensor(multi_rows, dtype=torch.long) if multi_rows else None,
            "multi_slots": torch.tensor(multi_slots, dtype=torch.long) if multi_slots else None,
            "pad_index": torch.tensor(pad_index, dtype=torch.long) if pad_index else None,
            # Longer segments' boundaries in the multi rows, for a varlen GDN kernel.
            "cu_seqlens": (
                torch.tensor([0] + [lo + n for _, lo, n in multi_spans], dtype=torch.long)
                if multi_spans
                else None
            ),
        }
        self.multi_spans = multi_spans
        self.longest = longest
        self._on: dict[torch.device, _PackOnDevice] = {}

    def on(self, dev: torch.device) -> _PackOnDevice:
        if dev not in self._on:
            moved = {k: (None if v is None else v.to(dev)) for k, v in self._host.items()}
            self._on[dev] = _PackOnDevice(
                tables=[torch.tensor(u, dtype=torch.long, device=dev) for u in self.used],
                spans=self.spans,
                multi_spans=self.multi_spans,
                longest=self.longest,
                **moved,
            )
        return self._on[dev]


class FastModel:
    """Slot-state Qwen3.6 text forward; one thread owns it."""

    def __init__(
        self,
        lm: torch.nn.Module,
        *,
        slots: int,
        max_len: int,
        kv_tokens: int | None = None,
        use_triton: bool | None = None,
        use_graphs: bool | None = None,
        buckets: Sequence[int] = (1, 2, 4, 8, 16, 32),
        mtp: torch.nn.Module | None = None,
    ) -> None:
        text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
        cfg = text.config
        if getattr(cfg, "model_type", "") not in ("qwen3_5_moe_text", "qwen3_5_moe"):
            raise FastPathUnavailable(f"model type {cfg.model_type!r} is not Qwen3.5/3.6 MoE")
        rope = cfg.rope_parameters
        if rope.get("rope_type", "default") != "default" or text.rotary_emb.attention_scaling != 1.0:
            raise FastPathUnavailable(f"rope {rope} is not the default rotary embedding")
        self.config = cfg
        self.text = text
        self.eps = cfg.rms_norm_eps
        self.max_len = max_len
        self.slots = slots
        self.dummy = slots  # the slot padding rows of a decode bucket write to
        self.embed = text.embed_tokens.weight
        self.final_norm = text.norm.weight
        self.lm_head = lm.lm_head.weight
        self.vocab = self.lm_head.shape[0]
        self.layer_types = list(cfg.layer_types)
        self.layers = list(text.layers)
        self.top_k = cfg.num_experts_per_tok
        self.max_pages = math.ceil(max_len / PAGE)
        on_gpu = self.embed.device.type == "cuda"
        self.use_triton = kernels.triton_usable(self.embed.device) if use_triton is None else use_triton
        self.use_graphs = (on_gpu and self.use_triton) if use_graphs is None else use_graphs
        if self.use_graphs and not self.use_triton:
            raise ValueError("graph capture needs the Triton kernels (the references synchronize)")
        # Per kernel, for isolating one kernel's numerics against its reference.
        self.triton_kernels = {
            "moe": self.use_triton,
            "gdn": self.use_triton,
            "attn": self.use_triton,
            # Fused launches: residual add + RMSNorm, and the whole DeltaNet step.
            "fused": self.use_triton and os.environ.get("ENGINE_FUSED", "1") == "1",
            # Whole decode layers in a few launches (``engine.fast.fused``, ENGINE_FUSED_DECODE).
            "fused_decode": False,
            "fused_verify": False,
            # Prefill routed experts as a sync-free grouped GEMM (``moe_prefill``) instead of
            # the registered experts module (``engine.experts``); ENGINE_MOE_GROUPED=0 for that.
            "moe_prefill": self.use_triton and moe_prefill.enabled(),
        }
        self.buckets = sorted(buckets)
        # The checkpoint's MTP head (``engine.mtp.MTPHead``) for self-speculative decoding
        # (``verify``): an extra full-attention layer, index ``mtp_index``, on the lm_head's
        # device, whose KV lives in the same pages as the model's (one more pool).
        self.mtp = mtp
        self.mtp_index = len(self.layers)
        if mtp is not None:
            mdev = mtp.fc.weight.device
            if mdev != self.lm_head.device:
                raise ValueError(f"the MTP head must sit on the lm_head's device {self.lm_head.device}")
            # The head embeds its input token on its own device (a copy of the same table).
            self.mtp_embed = self.embed if self.embed.device == mdev else self.embed.to(mdev)
        # A verify step runs two rows per slot; any batch has at most one (two) rows per slot.
        self._rows = max(max(self.buckets) * (2 if mtp is not None else 1), slots)
        inv_freq = text.rotary_emb.inv_freq.detach().float().cpu()

        # Per-layer state, allocated once on the layer's device.
        rows = slots + 1
        # With the MTP head every slot has two banks of Gated DeltaNet state, rows ``slot`` and
        # ``slot + rows``: a verify step runs its first token on the current bank and its draft
        # on the other (a copy taken after the first token), so a rejected draft leaves the
        # current bank exactly one token on and an accepted one flips ``bank[slot]``.
        self.gdn_rows = rows
        self.bank = [0] * rows
        banks = 2 if mtp is not None else 1
        self.devices: dict[torch.device, _Device] = {}
        self.inv_freq: dict[torch.device, torch.Tensor] = {}
        self.gdn_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        attn_layers: list[int] = []
        for i, layer in enumerate(self.layers):
            dev = layer.input_layernorm.weight.device
            self._device(dev, inv_freq)
            if self.layer_types[i] == "linear_attention":
                m = layer.linear_attn
                conv = torch.zeros(
                    (banks * rows, m.conv_dim, m.conv_kernel_size), dtype=self.embed.dtype, device=dev
                )
                rec = torch.zeros(
                    (banks * rows, m.num_v_heads, m.head_k_dim, m.head_v_dim),
                    dtype=torch.float32,
                    device=dev,
                )
                self.gdn_state[i] = (conv, rec)
            else:
                attn_layers.append(i)
        if mtp is not None:
            attn_layers.append(self.mtp_index)
        for dev in (self.embed.device, self.final_norm.device, self.lm_head.device):
            self._device(dev, inv_freq)

        pages = self._pages_for(attn_layers, kv_tokens)
        self.pool = PagePool(pages)
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i in attn_layers:
            m = self._attn_module(i)
            dev = m.k_proj.weight.device
            shape = (pages, cfg.num_key_value_heads, PAGE, m.head_dim)
            self.kv[i] = (
                torch.zeros(shape, dtype=self.embed.dtype, device=dev),
                torch.zeros(shape, dtype=self.embed.dtype, device=dev),
            )
        # Host copies: each slot's pages and length. The dummy slot owns one page forever.
        self.slot_pages: list[list[int]] = [[] for _ in range(rows)]
        self.slot_pages[self.dummy].append(self.pool.alloc())
        self._push_table(self.dummy)
        self.segments = self._segments()
        self.vsegments = self._segments(verify=True) if mtp is not None else []
        self.last_drafts: torch.Tensor | None = None  # per prefill segment: the MTP head's draft
        self.fused = None
        self.fused_error: str | None = None
        if self.use_triton and (on_gpu or kernels.INTERPRET):
            from engine.fast import fused

            if fused.enabled():
                try:
                    self.fused = fused.FusedDecode(self)
                    self.triton_kernels["fused_decode"] = True
                    # Verify steps (two rows per slot) and the MTP head's layer through the
                    # fused kernels too (ENGINE_FUSED_VERIFY=0: per-op kernels for those).
                    self.triton_kernels["fused_verify"] = fused.verify_enabled()
                except Exception as exc:  # noqa: BLE001 - the per-op decode path still serves
                    self.fused_error = f"{type(exc).__name__}: {exc}"
        self.graph_error: str | None = None
        self.capture_mode: str | None = None

    # ---------------------------------------------------------------- setup

    def _device(self, dev: torch.device, inv_freq: torch.Tensor) -> _Device:
        if dev not in self.devices:
            longest = self._rows
            packed = torch.zeros((4, longest), dtype=torch.long, device=dev)
            packed[1].fill_(self.slots)
            packed[3].fill_(self.slots)
            self.devices[dev] = _Device(
                device=dev,
                packed=packed,
                page_table=torch.zeros(
                    (self.slots + 1, self.max_pages), dtype=torch.int32, device=dev
                ),
            )
            self.inv_freq[dev] = inv_freq.to(dev)
        return self.devices[dev]

    def _pages_for(self, attn_layers: list[int], kv_tokens: int | None) -> int:
        """KV pages per attention layer: ``kv_tokens`` if given, else what the devices hold."""
        cfg = self.config
        need = self.max_pages + 1  # one full-length sequence plus the dummy page
        if kv_tokens is not None:
            return max(need, math.ceil(kv_tokens / PAGE) + 1)
        head_dim = self._attn_module(attn_layers[0]).head_dim if attn_layers else 1
        page_bytes = 2 * cfg.num_key_value_heads * PAGE * head_dim * self.embed.element_size()
        if self.embed.device.type != "cuda":
            return need + 2 * self.slots
        per_device: dict[torch.device, int] = {}
        for i in attn_layers:
            dev = self._attn_module(i).k_proj.weight.device
            per_device[dev] = per_device.get(dev, 0) + 1
        fraction = float(os.environ.get("ENGINE_KV_FRACTION", "0.6"))
        reserve = int(float(os.environ.get("ENGINE_RESERVE_GB", "6")) * (1 << 30))
        pages = None
        for dev, count in per_device.items():
            free, _ = torch.cuda.mem_get_info(dev)
            fit = int(max(0, free - reserve) * fraction) // (count * page_bytes)
            pages = fit if pages is None else min(pages, fit)
        # No more than ENGINE_KV_SEQS full-length sequences at once (8: the replay's sessions).
        seqs = min(self.slots, int(os.environ.get("ENGINE_KV_SEQS", "8")))
        pages = min(pages or 0, (seqs + 1) * self.max_pages + 1)
        if pages < need:
            raise RuntimeError(
                f"the GPUs hold {pages} KV pages per attention layer; one {self.max_len}-token "
                f"sequence needs {need}"
            )
        return pages

    def _attn_module(self, i: int) -> torch.nn.Module:
        """Layer ``i``'s attention module; ``mtp_index`` is the MTP head's."""
        if i == self.mtp_index:
            return self.mtp.layers[0].self_attn
        return self.layers[i].self_attn

    def _segments(self, verify: bool = False) -> list[_Segment]:
        ops: list[tuple[torch.device, Callable]] = [(self.embed.device, self._embed_op)]
        for i in range(len(self.layers)):
            ops.append((self.layers[i].input_layernorm.weight.device, self._layer_op(i, verify)))
        ops.append((self.final_norm.device, self._norm_op_for(verify)))
        ops.append((self.lm_head.device, self._verify_head_op if verify else self._head_op))
        segments: list[_Segment] = []
        for dev, op in ops:
            if not segments or segments[-1].device != dev:
                segments.append(_Segment(device=dev, ops=[]))
            segments[-1].ops.append(op)
        return segments

    # ---------------------------------------------------------------- slots and pages

    def _push_table(self, slot: int) -> None:
        row = torch.zeros(self.max_pages, dtype=torch.int32)
        pages = self.slot_pages[slot]
        if pages:
            row[: len(pages)] = torch.tensor(pages, dtype=torch.int32)
        for meta in self.devices.values():
            meta.page_table[slot].copy_(row)

    def _ensure_pages(self, slot: int, length: int) -> None:
        """Give ``slot`` pages for its first ``length`` tokens."""
        if length > self.max_len:
            raise RuntimeError(f"sequence of {length} tokens exceeds the context {self.max_len}")
        pages = self.slot_pages[slot]
        grew = False
        while len(pages) * PAGE < length:
            pages.append(self.pool.alloc())
            grew = True
        if grew:
            self._push_table(slot)

    def _g(self, slot: int) -> int:
        """The Gated DeltaNet state row that holds ``slot``'s current state."""
        return slot + self.bank[slot] * self.gdn_rows

    def flip(self, slot: int) -> None:
        """Make ``slot``'s other bank current: after ``verify`` accepted its draft, or to undo
        that (the bank left behind still holds the state one token earlier)."""
        if self.mtp is None:
            raise RuntimeError("only a model with the MTP head has two state banks")
        self.bank[slot] ^= 1

    def reset(self, slot: int) -> None:
        """Release ``slot``'s pages and zero its recurrent and convolution state."""
        for page in self.slot_pages[slot]:
            self.pool.unref(page)
        self.slot_pages[slot] = []
        self.bank[slot] = 0
        for conv, rec in self.gdn_state.values():
            conv[slot].zero_()
            rec[slot].zero_()

    def _copy_page(self, src: int) -> int:
        dst = self.pool.alloc()
        for k_pool, v_pool in self.kv.values():
            k_pool[dst].copy_(k_pool[src])
            v_pool[dst].copy_(v_pool[src])
        return dst

    def snapshot(self, slot: int, length: int) -> Snapshot:
        """A copy of ``slot``'s state, which must cover exactly ``length`` tokens."""
        row = self._g(slot)
        gdn = [(conv[row].clone(), rec[row].clone()) for conv, rec in self.gdn_state.values()]
        used = math.ceil(length / PAGE)
        own = self.slot_pages[slot][:used]
        full = length // PAGE
        pages = []
        for index, page in enumerate(own):
            if index < full:
                self.pool.ref(page)
                pages.append(page)
            else:  # the partial last page: the live sequence keeps writing into it
                pages.append(self._copy_page(page))
        return Snapshot(length=length, gdn=gdn, pages=pages)

    def restore(self, snap: Snapshot, slot: int) -> None:
        """Make ``slot`` (reset) hold ``snap``'s state."""
        row = self._g(slot)
        for (conv, rec), (c, r) in zip(self.gdn_state.values(), snap.gdn, strict=True):
            conv[row].copy_(c)
            rec[row].copy_(r)
        full = snap.length // PAGE
        held = list(snap.pages)
        for page in held:  # held first: allocating below may evict ``snap`` itself
            self.pool.ref(page)
        pages = []
        for index, page in enumerate(held):
            if index < full:
                pages.append(page)
            else:
                pages.append(self._copy_page(page))
                self.pool.unref(page)
        self.slot_pages[slot] = pages
        self._push_table(slot)

    def detach(self, slot: int, length: int) -> Snapshot:
        """Move ``slot``'s state, which must cover exactly ``length`` tokens, into a snapshot.

        The snapshot takes the slot's pages as they are, the partial last one
        included (nothing writes the slot again before its reset), and clones
        the Gated DeltaNet states. The slot is left without pages.
        """
        row = self._g(slot)
        gdn = [(conv[row].clone(), rec[row].clone()) for conv, rec in self.gdn_state.values()]
        used = math.ceil(length / PAGE)
        pages = self.slot_pages[slot]
        for page in pages[used:]:
            self.pool.unref(page)
        self.slot_pages[slot] = []
        return Snapshot(length=length, gdn=gdn, pages=pages[:used])

    def adopt(self, snap: Snapshot, slot: int) -> None:
        """Make ``slot`` (reset) hold ``snap``'s state, taking over its pages; ``snap`` is spent.

        Unlike ``restore``, the partial last page is not copied: the slot
        becomes its owner and writes into it. Only for a snapshot nothing else
        will read (a session tail handed over by the prefix cache).
        """
        if snap.released:
            raise RuntimeError("adopting a released snapshot")
        row = self._g(slot)
        for (conv, rec), (c, r) in zip(self.gdn_state.values(), snap.gdn, strict=True):
            conv[row].copy_(c)
            rec[row].copy_(r)
        for page in self.slot_pages[slot]:
            self.pool.unref(page)
        self.slot_pages[slot] = list(snap.pages)
        snap.pages = []
        snap.gdn = []
        snap.released = True
        self._push_table(slot)

    @torch.inference_mode()
    def warmup(
        self,
        slot: int,
        chunk: int,
        token_ids: Sequence[int],
        other: int | None = None,
        pack_tokens: int | None = None,
    ) -> None:
        """Run every prefill variant serving uses once, in ``slot`` (free), then reset it.

        Compiles the bucketed prefill conv problems up to ``chunk`` on every device
        and runs prefills fresh (short, and a whole chunk) and on top of a state,
        and with ``other`` (a second free slot) packed passes of two segments, so
        the GDN chunk kernels (varlen too), the prefill experts and attention
        compile and tune before the server answers. Decode is warmed by
        ``capture_all``.
        """
        if self.triton_kernels["moe_prefill"]:
            self._warm_moe_prefill(chunk)
        seen = set()
        for i, (conv, _) in self.gdn_state.items():
            weight = self.layers[i].linear_attn.conv1d.weight
            key = (weight.device, tuple(weight.shape))
            if key in seen:
                continue
            seen.add(key)
            kernel = conv.shape[-1]
            for length in range(CONV_BUCKET, chunk + kernel + CONV_BUCKET, CONV_BUCKET):
                window = torch.zeros(
                    (conv.shape[1], max(length, kernel + 1)), dtype=conv.dtype, device=conv.device
                )
                prefill_conv(window, weight, 1)
        ids = list(token_ids)
        try:
            self.prefill(slot, ids[:37], 0)  # fresh, no initial state
            self.prefill(slot, ids[37:101], 37)  # on top of a state
            self.reset(slot)
            self.prefill(slot, ids[:chunk], 0)  # a whole chunk
            if other is not None:  # packed passes (``prefill_packed``) in ``slot`` and ``other``
                self.reset(slot)
                self.reset(other)
                if len(ids) < 102:
                    raise ValueError("warmup needs at least 102 token ids")
                self.prefill(slot, ids[:37], 0)
                # Two longer segments, one on top of a state: the varlen GDN kernels, packed
                # attention; then a one-token segment beside a longer one.
                self.prefill_packed([(slot, ids[37:101], 37), (other, ids[:64], 0)])
                end = min(len(ids), max(chunk, 70))
                self.prefill_packed([(slot, ids[101:102], 101), (other, ids[64:end], 64)])
                if pack_tokens:
                    self._warm_pass_lengths(slot, other, min(pack_tokens, len(ids)), ids)
        finally:
            self.reset(slot)
            if other is not None:
                self.reset(other)
        for dev in self.devices:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

    def _warm_pass_lengths(self, slot: int, other: int, longest: int, ids: Sequence[int]) -> None:
        """Packed passes of two fresh segments at every length band up to ``longest`` tokens.

        fla's l2norm (the GDN chunk rule's q/k norm) autotunes once per
        ``NB = cdiv(tokens * heads, 65536)``, a few seconds each, so every band a
        pass can reach is run here rather than first in a request.
        """
        m = self.layers[next(iter(self.gdn_state))].linear_attn
        band = max(1, 65536 // max(m.num_v_heads, m.num_k_heads))
        for total in range(band, longest + band, band):
            total = min(total, longest)
            half = total // 2
            self.reset(slot)
            self.reset(other)
            self.prefill_packed([(slot, ids[:half], 0), (other, ids[half:total], 0)])

    def _warm_moe_prefill(self, chunk: int) -> None:
        """Compile the grouped prefill experts for every tile shape up to ``chunk`` tokens
        (and twice that, a packed pass), on every device that holds experts."""
        seen = set()
        layers = list(self.layers) + (list(self.mtp.layers) if self.mtp is not None else [])
        for layer in layers:  # the MTP head's prefill MoE takes the same path (``_mtp_layer``)
            experts = layer.mlp.experts
            gate_up, down = experts.gate_up_proj, experts.down_proj
            key = (gate_up.device, tuple(gate_up.shape))
            if key in seen:
                continue
            seen.add(key)
            num, _, hidden = gate_up.shape
            shapes = {}
            tokens = DECODE_EXPERT_TOKENS + 1
            while tokens <= 2 * max(chunk, DECODE_EXPERT_TOKENS + 1):
                shapes.setdefault(moe_prefill.config(tokens * self.top_k, num), tokens)
                tokens *= 2
            for tokens in shapes.values():
                x = torch.zeros((tokens, hidden), dtype=gate_up.dtype, device=gate_up.device)
                ids = torch.arange(tokens * self.top_k, device=gate_up.device) % num
                weights = torch.full((tokens, self.top_k), 1.0 / self.top_k, device=gate_up.device)
                if moe_prefill.supported(x, gate_up, down):
                    moe_prefill.moe_prefill(
                        x, ids.view(tokens, self.top_k), weights.to(x.dtype), gate_up, down
                    )

    @staticmethod
    def snapshot_bytes(snap: Snapshot) -> int:
        """Bytes of the snapshot's own Gated DeltaNet state copies (its pages are the pool's)."""
        return sum(c.nbytes + r.nbytes for c, r in snap.gdn)

    def release(self, snap: Snapshot) -> None:
        if not snap.released:
            snap.released = True
            for page in snap.pages:
                self.pool.unref(page)
            snap.gdn = []

    # ---------------------------------------------------------------- the layers

    def _embed_op(self, tokens: torch.Tensor, meta: _Device) -> torch.Tensor:
        return F.embedding(tokens, self.embed)

    def _norm_op(self, x: torch.Tensor, meta: _Device) -> torch.Tensor:
        return _rms(x, self.final_norm, self.eps)

    def _norm_op_for(self, verify: bool) -> Callable[[torch.Tensor, _Device], torch.Tensor]:
        """The final norm of a decode (or verify) step: with the fused layers, the per-row
        Triton RMSNorm, whose result for a row does not depend on how many rows run."""

        def op(x: torch.Tensor, meta: _Device) -> torch.Tensor:
            if self._fused_head(x.shape[0], verify):
                with kernels.on_device(x):
                    return kernels.add_rms(x, None, self.final_norm, self.eps)[1]
            return self._norm_op(x, meta)

        return op

    def _logits(self, x: torch.Tensor, verify: bool = False) -> torch.Tensor:
        """fp32 logits of a decode (or verify) step's normed rows. With the fused layers the
        lm_head runs through the fused GEMV, whose sum order is the same for every row count
        (``engine.fast.fused``), so a slot's logits are the same in a decode step and in a
        verify step; else ``F.linear`` (BF16 out, as the GEMV)."""
        if self._fused_head(x.shape[0], verify):
            with kernels.on_device(x):
                return self.fused.head(x, self.lm_head).float()
        return F.linear(x, self.lm_head).float()

    def _fused_head(self, rows: int, verify: bool) -> bool:
        """Whether a step's final norm and lm_head run through the fused kernels
        (``ENGINE_FUSED_HEAD``, default on, for steps the fused layers serve)."""
        from engine.fast import fused

        return fused.HEAD and self._fused_rows(rows, verify)

    def _head_op(self, x: torch.Tensor, meta: _Device) -> torch.Tensor:
        return self._logits(x)

    def _layer_op(self, i: int, verify: bool = False) -> Callable[[torch.Tensor, _Device], torch.Tensor]:
        def op(x: torch.Tensor, meta: _Device) -> torch.Tensor:
            batch = x.shape[0]
            return self.layer_decode(
                i, x, meta.slots[:batch], meta.positions[:batch], meta, meta.gdn[:batch], verify
            )

        return op

    def layer_decode(
        self,
        i: int,
        x: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
        meta: _Device,
        states: torch.Tensor | None = None,
        verify: bool = False,
    ) -> torch.Tensor:
        """One layer for one token per row; ``states`` are the rows' Gated DeltaNet state rows
        (default ``slots``). With ``verify`` the rows are two halves, each slot's first token
        then its draft (see ``verify``)."""
        states = slots if states is None else states
        if self._fused_rows(x.shape[0], verify):
            return self.fused.layer(i, x, slots, positions, meta, states, verify)
        layer = self.layers[i]
        fused = self.triton_kernels["fused"]
        if fused:
            _, h = kernels.add_rms(x, None, layer.input_layernorm.weight, self.eps)
        else:
            h = _rms(x, layer.input_layernorm.weight, self.eps)
        if self.layer_types[i] == "linear_attention":
            h = self._gdn_verify(i, h, states) if verify else self._gdn_decode(i, h, states)
        else:
            h = self._attn_decode(i, h, slots, positions, meta)
        if fused:
            x, h = kernels.add_rms(x, h, layer.post_attention_layernorm.weight, self.eps)
        else:
            x = x + h
            h = _rms(x, layer.post_attention_layernorm.weight, self.eps)
        return x + self._moe(layer.mlp, h, decode=True)

    def _fused_rows(self, rows: int, verify: bool = False) -> bool:
        """Whether a decode (or verify, or MTP head) step of ``rows`` rows takes the fused
        layers (``engine.fast.fused``)."""
        return (
            self.fused is not None
            and self.triton_kernels["fused_decode"]
            and (not verify or self.triton_kernels.get("fused_verify", False))
            and rows <= self.fused.max_rows
        )

    def _gdn_project(self, m: torch.nn.Module, x: torch.Tensor):
        return (
            F.linear(x, m.in_proj_qkv.weight),
            F.linear(x, m.in_proj_z.weight),
            F.linear(x, m.in_proj_b.weight),
            F.linear(x, m.in_proj_a.weight),
        )

    def _gdn_heads(self, m: torch.nn.Module, mixed: torch.Tensor, b: torch.Tensor, a: torch.Tensor):
        """Split the convolved projection into heads; the decay ``g`` and ``beta`` as Transformers."""
        tokens = mixed.shape[0]
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(tokens, -1, m.head_k_dim)
        k = k.reshape(tokens, -1, m.head_k_dim)
        v = v.reshape(tokens, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
        ratio = m.num_v_heads // m.num_k_heads
        if ratio > 1:
            q = q.repeat_interleave(ratio, dim=1)
            k = k.repeat_interleave(ratio, dim=1)
        return q, k, v, g, beta

    def _gdn_gate(self, m: torch.nn.Module, core: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tokens = core.shape[0]
        out = _gated_rms(
            core.reshape(-1, m.head_v_dim),
            z.reshape(-1, m.head_v_dim),
            m.norm.weight,
            m.norm.variance_epsilon,
        )
        return out.reshape(tokens, -1)

    def _gdn_out(self, m: torch.nn.Module, core: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return F.linear(self._gdn_gate(m, core, z), m.out_proj.weight)

    def _gdn_decode(self, i: int, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        m = self.layers[i].linear_attn
        qkv, z, b, a = self._gdn_project(m, x)
        return F.linear(self._gdn_step(i, qkv, z, b, a, slots), m.out_proj.weight)

    def _gdn_verify(self, i: int, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """A verify step's Gated DeltaNet layer: rows ``[:n]`` are each slot's first token on
        its current state row, rows ``[n:]`` its draft on the other bank's row. The projections
        run over all rows at once; the recurrence runs the first half in place, copies those
        states to the second half's rows, then runs the second half there."""
        m = self.layers[i].linear_attn
        conv_state, rec_state = self.gdn_state[i]
        n = x.shape[0] // 2
        first, second = rows[:n], rows[n:]
        qkv, z, b, a = self._gdn_project(m, x)
        out_a = self._gdn_step(i, qkv[:n], z[:n], b[:n], a[:n], first)
        conv_state.index_copy_(0, second, conv_state.index_select(0, first))
        rec_state.index_copy_(0, second, rec_state.index_select(0, first))
        out_b = self._gdn_step(i, qkv[n:], z[n:], b[n:], a[n:], second)
        return F.linear(torch.cat((out_a, out_b)), m.out_proj.weight)

    def _gdn_step(
        self, i: int, qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """The recurrent step of projected inputs on state rows ``slots`` (in place), before
        ``out_proj``."""
        m = self.layers[i].linear_attn
        conv_state, rec_state = self.gdn_state[i]
        if self.triton_kernels["fused"] and self.triton_kernels["gdn"]:
            mixed = kernels.gdn_conv_(conv_state, slots, qkv, m.conv1d.weight)
            return kernels.gdn_fused_(rec_state, slots, mixed, a, b, z, m)
        # Short causal convolution over the last ``kernel`` inputs (the slot's state holds them).
        window = torch.cat((conv_state.index_select(0, slots)[:, :, 1:], qkv.unsqueeze(-1)), dim=-1)
        conv_state.index_copy_(0, slots, window)
        weight = m.conv1d.weight.squeeze(1)
        mixed = (window.float() * weight.float()).sum(-1).to(qkv.dtype)
        mixed = F.silu(mixed)
        q, k, v, g, beta = self._gdn_heads(m, mixed, b, a)
        qf = hf.l2norm(q.float(), dim=-1, eps=1e-6) * (q.shape[-1] ** -0.5)
        kf = hf.l2norm(k.float(), dim=-1, eps=1e-6)
        core = kernels.gdn_recurrent_(
            rec_state, slots, qf, kf, v.float(), g, beta.float(), use_triton=self.triton_kernels["gdn"]
        ).to(qkv.dtype)
        return self._gdn_gate(m, core, z)

    def _qkv(self, i: int, x: torch.Tensor, positions: torch.Tensor):
        m = self._attn_module(i)
        tokens = x.shape[0]
        qg = F.linear(x, m.q_proj.weight).view(tokens, -1, m.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(tokens, -1)
        q = _rms(q, m.q_norm.weight, m.q_norm.eps)
        k = _rms(F.linear(x, m.k_proj.weight).view(tokens, -1, m.head_dim), m.k_norm.weight, m.k_norm.eps)
        v = F.linear(x, m.v_proj.weight).view(tokens, -1, m.head_dim)
        freqs = positions.float()[:, None] * self.inv_freq[x.device][None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        return m, _rope(q, cos, sin), _rope(k, cos, sin), v, gate

    def _attn_decode(
        self, i: int, x: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor, meta: _Device
    ) -> torch.Tensor:
        m, q, k, v, gate = self._qkv(i, x, positions)
        k_pool, v_pool = self.kv[i]
        page = meta.page_table[slots, positions // PAGE].long()
        offset = positions % PAGE
        k_pool[page, :, offset] = k
        v_pool[page, :, offset] = v
        out = kernels.paged_decode_attention(
            q,
            k_pool,
            v_pool,
            meta.page_table,
            slots,
            (positions + 1).to(torch.int32),
            m.scaling,
            use_triton=self.triton_kernels["attn"],
        )
        out = out.reshape(x.shape[0], -1) * torch.sigmoid(gate)
        return F.linear(out, m.o_proj.weight)

    def _moe(self, mlp: torch.nn.Module, x: torch.Tensor, decode: bool = False) -> torch.Tensor:
        """Shared plus routed experts. Decode steps always use the in-place expert kernel,
        which never synchronizes with the host, so every batch bucket can be captured;
        prefill chunks above ``DECODE_EXPERT_TOKENS`` use the grouped GEMM of
        ``moe_prefill`` (no host synchronization either), or, under
        ``ENGINE_MOE_GROUPED=0`` or without Triton, the registered experts module."""
        tokens = x.shape[0]
        shared = mlp.shared_expert
        shared_out = F.linear(
            F.silu(F.linear(x, shared.gate_proj.weight)) * F.linear(x, shared.up_proj.weight),
            shared.down_proj.weight,
        )
        router = F.linear(x, mlp.gate.weight)
        probs = torch.softmax(router, dim=-1, dtype=torch.float)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(router.dtype)
        experts = mlp.experts
        if decode or tokens <= DECODE_EXPERT_TOKENS:
            routed = kernels.moe_decode(
                x,
                top_i,
                top_w,
                experts.gate_up_proj,
                experts.down_proj,
                use_triton=self.triton_kernels["moe"],
            )
        elif self.triton_kernels["moe_prefill"] and moe_prefill.supported(
            x, experts.gate_up_proj, experts.down_proj
        ):  # prefill: the sync-free grouped GEMM over the real (token, expert) rows
            routed = moe_prefill.moe_prefill(x, top_i, top_w, experts.gate_up_proj, experts.down_proj)
        else:  # prefill: the experts implementation the engine registered with Transformers
            routed = experts(x, top_i, top_w)
        return routed + torch.sigmoid(F.linear(x, mlp.shared_expert_gate.weight)) * shared_out

    # ---------------------------------------------------------------- prefill

    @torch.inference_mode()
    def prefill(
        self, slot: int, ids: Sequence[int], start: int, follow: int | None = None
    ) -> torch.Tensor:
        """Run ``ids`` at positions ``start``... into ``slot``; fp32 logits after the last one.

        With the MTP head, the head also runs over the chunk (``_mtp_prefill``): ``follow``
        is the token after the chunk, None for the greedy pick from the returned logits.
        """
        count = len(ids)
        self._ensure_pages(slot, start + count)
        dev = self.embed.device
        x = F.embedding(torch.tensor(list(ids), dtype=torch.long, device=dev), self.embed)
        for i, layer in enumerate(self.layers):
            ldev = layer.input_layernorm.weight.device
            x = x.to(ldev)
            positions = torch.arange(start, start + count, device=ldev)
            h = _rms(x, layer.input_layernorm.weight, self.eps)
            if self.layer_types[i] == "linear_attention":
                h = self._gdn_prefill(i, h, slot)
            else:
                h = self._attn_prefill(i, h, slot, positions, start)
            x = x + h
            h = _rms(x, layer.post_attention_layernorm.weight, self.eps)
            x = x + self._moe(layer.mlp, h)
        # The logits exactly as without the head (the norm over the last row alone).
        last = _rms(x[-1:].to(self.final_norm.device), self.final_norm, self.eps).to(self.lm_head.device)
        logits = F.linear(last, self.lm_head).float()
        if self.mtp is not None:
            hidden = _rms(x.to(self.final_norm.device), self.final_norm, self.eps).to(self.lm_head.device)
            self._mtp_prefill(
                hidden, [(slot, ids, start)], [follow], logits, _PackPlan(self, [(slot, ids, start)])
            )
        return logits[0]

    def _gdn_prefill(self, i: int, x: torch.Tensor, slot: int) -> torch.Tensor:
        m = self.layers[i].linear_attn
        count = x.shape[0]
        slot = self._g(slot)  # the state row
        if count == 1:  # Transformers' single-token path: the recurrent rule
            slots = torch.tensor([slot], device=x.device)
            return self._gdn_decode(i, x, slots)
        conv_state, rec_state = self.gdn_state[i]
        qkv, z, b, a = self._gdn_project(m, x)
        window = torch.cat((conv_state[slot], qkv.transpose(0, 1)), dim=-1)  # [conv_dim, kernel+T]
        kernel = m.conv_kernel_size
        conv_state[slot].copy_(window[:, -kernel:])
        mixed = prefill_conv(window, m.conv1d.weight, count)
        mixed = F.silu(mixed.to(qkv.dtype)).transpose(0, 1)
        q, k, v, g, beta = self._gdn_heads(m, mixed, b, a)
        core, final = hf.torch_chunk_gated_delta_rule(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=rec_state[slot : slot + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        rec_state[slot].copy_(final[0])
        return self._gdn_out(m, core[0], z)

    def _attn_prefill(
        self, i: int, x: torch.Tensor, slot: int, positions: torch.Tensor, start: int
    ) -> torch.Tensor:
        m, q, k, v, gate = self._qkv(i, x, positions)
        count = x.shape[0]
        k_pool, v_pool = self.kv[i]
        table = torch.tensor(self.slot_pages[slot], dtype=torch.long, device=x.device)
        page = table[positions // PAGE]
        offset = positions % PAGE
        k_pool[page, :, offset] = k
        v_pool[page, :, offset] = v
        total = start + count
        used = table[: math.ceil(total / PAGE)]
        heads = k.shape[1]
        keys = k_pool[used].permute(1, 0, 2, 3).reshape(heads, -1, m.head_dim)[None, :, :total]
        values = v_pool[used].permute(1, 0, 2, 3).reshape(heads, -1, m.head_dim)[None, :, :total]
        query = q.transpose(0, 1).unsqueeze(0)  # [1, q_heads, T, dim]
        q_heads = query.shape[1]
        step = max(1, SCORE_BUDGET // (total * q_heads * 4))
        outs = []
        for lo in range(0, count, step):
            hi = min(count, lo + step)
            key_pos = torch.arange(total, device=x.device)
            mask = key_pos[None, :] <= positions[lo:hi, None]
            outs.append(
                F.scaled_dot_product_attention(
                    query[:, :, lo:hi],
                    keys,
                    values,
                    attn_mask=mask[None, None],
                    scale=m.scaling,
                    enable_gqa=q_heads != heads,
                )
            )
        out = torch.cat(outs, dim=2)[0].transpose(0, 1).reshape(count, -1)
        out = out * torch.sigmoid(gate)
        return F.linear(out, m.o_proj.weight)

    # ---------------------------------------------------------------- packed prefill

    @torch.inference_mode()
    def prefill_packed(
        self,
        segments: Sequence[tuple[int, Sequence[int], int]],
        follows: Sequence[int | None] | None = None,
    ) -> torch.Tensor:
        """Run several sequences' prefill chunks in one pass; fp32 logits [segments, vocab].

        ``follows`` gives, per segment, ``prefill``'s ``follow`` (default: all None).

        Each segment ``(slot, ids, start)`` is what ``prefill(slot, ids, start)``
        would run, and leaves its slot in the same state. The segments' tokens
        are concatenated, so every projection, router, shared expert and routed
        expert runs once over all of them (weights read once, one launch
        sequence per pass instead of one per request). What is per sequence
        stays per sequence: the short convolution and recurrent state of each
        Gated DeltaNet layer (one chunked delta-rule call over the segments,
        right-padded to the longest with zero keys, values, decays and betas,
        which leave a state unchanged), and each segment's paged KV writes and
        causal attention. One-token segments take the decode recurrence, as
        ``prefill`` does.
        """
        slots = [slot for slot, _, _ in segments]
        if len(set(slots)) != len(slots):
            raise ValueError("a packed prefill holds at most one segment per slot")
        if any(len(ids) == 0 for _, ids, _ in segments):
            raise ValueError("an empty prefill segment")
        follows = list(follows) if follows is not None else [None] * len(segments)
        if len(segments) == 1:
            slot, ids, start = segments[0]
            return self.prefill(slot, ids, start, follows[0])[None]
        for slot, ids, start in segments:
            self._ensure_pages(slot, start + len(ids))
        plan = _PackPlan(self, segments)
        x = F.embedding(plan.tokens.to(self.embed.device), self.embed)
        for i, layer in enumerate(self.layers):
            ldev = layer.input_layernorm.weight.device
            x = x.to(ldev)
            h = _rms(x, layer.input_layernorm.weight, self.eps)
            if self.layer_types[i] == "linear_attention":
                h = self._gdn_prefill_packed(i, h, plan.on(ldev))
            else:
                h = self._attn_prefill_packed(i, h, plan.on(ldev))
            x = x + h
            h = _rms(x, layer.post_attention_layernorm.weight, self.eps)
            x = x + self._moe(layer.mlp, h)
        # The logits exactly as without the head (the norm over the segments' last rows alone).
        last = x.index_select(0, plan.on(x.device).last).to(self.final_norm.device)
        last = _rms(last, self.final_norm, self.eps).to(self.lm_head.device)
        logits = F.linear(last, self.lm_head).float()
        if self.mtp is not None:
            hidden = _rms(x.to(self.final_norm.device), self.final_norm, self.eps).to(self.lm_head.device)
            self._mtp_prefill(hidden, segments, follows, logits, plan)
        return logits

    def _gdn_prefill_packed(self, i: int, x: torch.Tensor, p: _PackOnDevice) -> torch.Tensor:
        m = self.layers[i].linear_attn
        out = torch.empty_like(x)
        if p.single_rows is not None:  # one-token segments: the recurrent rule, as ``prefill``
            out.index_copy_(
                0, p.single_rows, self._gdn_decode(i, x.index_select(0, p.single_rows), p.single_slots)
            )
        if p.multi_rows is None:
            return out
        conv_state, rec_state = self.gdn_state[i]
        xm = x.index_select(0, p.multi_rows)
        qkv, z, b, a = self._gdn_project(m, xm)
        kernel = m.conv_kernel_size
        weight = m.conv1d.weight
        mixed_parts = []
        for slot, lo, count in p.multi_spans:  # per-sequence short convolution and its state
            window = torch.cat((conv_state[slot], qkv[lo : lo + count].transpose(0, 1)), dim=-1)
            conv_state[slot].copy_(window[:, -kernel:])
            mixed_parts.append(prefill_conv(window, weight, count))
        mixed = F.silu(torch.cat(mixed_parts, dim=-1).to(qkv.dtype)).transpose(0, 1)
        q, k, v, g, beta = self._gdn_heads(m, mixed, b, a)
        initial = rec_state.index_select(0, p.multi_slots)
        if hf.torch_chunk_gated_delta_rule is gdn.fla_chunk:
            # fla's chunk kernel over the concatenated segments: one sequence per
            # ``cu_seqlens`` span, each from its own slot's recurrent state.
            core, final = gdn.fla_chunk(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=p.cu_seqlens,
            )
            rec_state.index_copy_(0, p.multi_slots, final.to(rec_state.dtype))
            out.index_copy_(0, p.multi_rows, self._gdn_out(m, core[0], z))
            return out
        # Otherwise (torch or reference GDN): the segments right-padded to the longest.
        batch, longest = len(p.multi_spans), p.longest

        def padded(t: torch.Tensor) -> torch.Tensor:
            buf = t.new_zeros((batch * longest, *t.shape[1:]))
            buf.index_copy_(0, p.pad_index, t)
            return buf.view(batch, longest, *t.shape[1:])

        core, final = hf.torch_chunk_gated_delta_rule(
            padded(q),
            padded(k),
            padded(v),
            g=padded(g),
            beta=padded(beta),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        rec_state.index_copy_(0, p.multi_slots, final)
        core = core.reshape(batch * longest, *core.shape[2:]).index_select(0, p.pad_index)
        out.index_copy_(0, p.multi_rows, self._gdn_out(m, core, z))
        return out

    def _attn_prefill_packed(self, i: int, x: torch.Tensor, p: _PackOnDevice) -> torch.Tensor:
        m, q, k, v, gate = self._qkv(i, x, p.positions)
        k_pool, v_pool = self.kv[i]
        k_pool[p.pages, :, p.offsets] = k
        v_pool[p.pages, :, p.offsets] = v
        heads = k.shape[1]
        q_heads = q.shape[1]
        outs = []
        for (lo, count, total), table in zip(p.spans, p.tables, strict=True):
            keys = k_pool[table].permute(1, 0, 2, 3).reshape(heads, -1, m.head_dim)[None, :, :total]
            values = v_pool[table].permute(1, 0, 2, 3).reshape(heads, -1, m.head_dim)[None, :, :total]
            query = q[lo : lo + count].transpose(0, 1).unsqueeze(0)
            positions = p.positions[lo : lo + count]
            step = max(1, SCORE_BUDGET // (total * q_heads * 4))
            key_pos = torch.arange(total, device=x.device)
            for a in range(0, count, step):
                b = min(count, a + step)
                mask = key_pos[None, :] <= positions[a:b, None]
                outs.append(
                    F.scaled_dot_product_attention(
                        query[:, :, a:b],
                        keys,
                        values,
                        attn_mask=mask[None, None],
                        scale=m.scaling,
                        enable_gqa=q_heads != heads,
                    )[0].transpose(0, 1)
                )
        out = torch.cat(outs, dim=0).reshape(x.shape[0], -1)
        out = out * torch.sigmoid(gate)
        return F.linear(out, m.o_proj.weight)

    # ---------------------------------------------------------------- MTP head

    def _mtp_layer(self, z: torch.Tensor, attend: Callable[[torch.Tensor], torch.Tensor], decode: bool):
        """The head's decoder layer (gated full attention by ``attend``, then the MoE)."""
        layer = self.mtp.layers[0]
        h = _rms(z, layer.input_layernorm.weight, self.eps)
        z = z + attend(h)
        h = _rms(z, layer.post_attention_layernorm.weight, self.eps)
        return z + self._moe(layer.mlp, h, decode=decode)

    def _mtp_in(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``fc(cat(norm(embed(x[t+1])), norm(H[t])))``: embedding first (``engine.mtp``)."""
        m = self.mtp
        e = _rms(F.embedding(tokens, self.mtp_embed), m.pre_fc_norm_embedding.weight, self.eps)
        h = _rms(hidden, m.pre_fc_norm_hidden.weight, self.eps)
        return F.linear(torch.cat((e, h), dim=-1), m.fc.weight)

    def _mtp_logits(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(_rms(z, self.mtp.norm.weight, self.eps), self.lm_head)

    def _mtp_prefill(
        self,
        hidden: torch.Tensor,
        segments: Sequence[tuple[int, Sequence[int], int]],
        follows: Sequence[int | None],
        logits: torch.Tensor,
        plan: _PackPlan,
    ) -> None:
        """Run the MTP head over a prefill pass: at each position ``i`` the pair ``(H[i],
        x[i+1])``, writing the head's KV into the slot's pages, so later drafts attend to the
        whole context. A segment's last pair takes its ``follow`` token, or (None) the greedy
        pick from its ``logits`` row. Sets ``last_drafts``: per segment, the head's guess of
        the token after that one."""
        dev = hidden.device
        nxt: list[int] = []
        picked: list[tuple[int, int]] = []  # (row, segment) whose next token is the greedy pick
        for k, ((_, ids, _), follow) in enumerate(zip(segments, follows, strict=True)):
            nxt.extend(ids[1:])
            if follow is None:
                picked.append((len(nxt), k))
            nxt.append(0 if follow is None else follow)
        tokens = torch.tensor(nxt, dtype=torch.long, device=dev)
        if picked:
            rows = torch.tensor([r for r, _ in picked], dtype=torch.long, device=dev)
            segs = torch.tensor([k for _, k in picked], dtype=torch.long, device=dev)
            tokens.index_copy_(0, rows, logits.index_select(0, segs).argmax(-1))
        on = plan.on(dev)
        z = self._mtp_layer(
            self._mtp_in(hidden, tokens),
            lambda h: self._attn_prefill_packed(self.mtp_index, h, on),
            decode=False,
        )
        self.last_drafts = self._mtp_logits(z.index_select(0, on.last)).argmax(-1)

    def _mtp_decode(self, hidden: torch.Tensor, tokens: torch.Tensor, meta: _Device) -> torch.Tensor:
        """The head on one pair per row at the rows' positions; its greedy guess per row."""
        n = hidden.shape[0]
        slots, positions = meta.slots[:n], meta.positions[:n]
        if self._fused_rows(n, verify=True):  # the head's layer through the fused kernels
            z = self.fused.layer(self.mtp_index, self._mtp_in(hidden, tokens), slots, positions, meta)
            return self._mtp_logits(z).argmax(-1)
        z = self._mtp_layer(
            self._mtp_in(hidden, tokens),
            lambda h: self._attn_decode(self.mtp_index, h, slots, positions, meta),
            decode=True,
        )
        return self._mtp_logits(z).argmax(-1)

    def _verify_head_op(self, x: torch.Tensor, meta: _Device) -> torch.Tensor:
        """The verify step's last op: logits from the normed hidden ``x``, each row's greedy
        pick, and the MTP head on (hidden, pick) per row. Leaves ``[2, rows]`` (picks, drafts)
        in ``_verify_out``; returns the logits."""
        logits = self._logits(x, verify=True)
        picks = logits.argmax(-1)
        self._verify_out = torch.stack((picks, self._mtp_decode(x, picks, meta)))
        return logits

    @torch.inference_mode()
    def verify(
        self,
        tokens: Sequence[int],
        drafts: Sequence[int],
        slots: Sequence[int],
        positions: Sequence[int],
    ) -> tuple[torch.Tensor, list[int], list[int], list[int], list[int]]:
        """One self-speculative step: each slot's last token at its position and its draft
        after it, in one forward (two rows per slot), then the MTP head on both.

        Returns (fp32 logits after each first token ``[batch, vocab]``, the greedy pick after
        each first token ``g1``, after each draft ``g2``, the head's next draft if the draft
        is rejected ``d1`` and if accepted ``d2``). The slot's current Gated DeltaNet bank is
        left one token on; the other bank two tokens on: ``flip`` it when ``g1`` equals the
        draft. Attention KV is written at both positions; a rejected draft's entry lies past
        the slot's length and is overwritten by the next step. The logits may be a graph's
        static output: read them before the next call.
        """
        if self.mtp is None:
            raise RuntimeError("verify needs the MTP head")
        batch = len(tokens)
        for slot, position in zip(slots, positions, strict=True):
            self._ensure_pages(slot, position + 2)
        size = self.bucket(batch) if self.use_graphs else batch
        rows = 2 * size
        host = torch.zeros((4, self._rows), dtype=torch.long)
        host[1].fill_(self.dummy)
        host[3].fill_(self.dummy)
        host[3, size:rows] = self.dummy + self.gdn_rows
        cur = [self._g(s) for s in slots]
        other = [s + (1 - self.bank[s]) * self.gdn_rows for s in slots]
        for half, toks, pos, states in (
            (0, tokens, positions, cur),
            (size, drafts, [p + 1 for p in positions], other),
        ):
            host[0, half : half + batch] = torch.tensor(list(toks), dtype=torch.long)
            host[1, half : half + batch] = torch.tensor(list(slots), dtype=torch.long)
            host[2, half : half + batch] = torch.tensor(list(pos), dtype=torch.long)
            host[3, half : half + batch] = torch.tensor(states, dtype=torch.long)
        for meta in self.devices.values():
            meta.packed.copy_(host)
        logits, out = self._run(self.vsegments, size, rows, verify=True)
        picks = out.cpu().tolist()
        g, d = picks[0], picks[1]
        return (
            logits[:batch],
            g[:batch],
            g[size : size + batch],
            d[:batch],
            d[size : size + batch],
        )

    def _run(
        self, segments: list[_Segment], size: int, rows: int, verify: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replay ``segments``' graphs for bucket ``size`` (capturing them first), or run them
        eagerly over ``rows`` rows; the last segment's output (and a verify's picks)."""
        if self.use_graphs and self.graph_error is None:
            if size not in segments[0].graphs:
                self.capture(size, verify)
            if self.graph_error is None:
                x = None
                for seg in segments:
                    with torch.cuda.device(seg.device):
                        if x is not None:
                            seg.inputs[size].copy_(x)
                        seg.graphs[size].replay()
                        x = seg.outputs[size]
                return x, (self._verify_static[size] if verify else None)
        x = self.devices[self.embed.device].tokens[:rows]
        for seg in segments:
            x = seg.run(x.to(seg.device), self.devices[seg.device])
        return x, (self._verify_out if verify else None)

    # ---------------------------------------------------------------- decode

    def bucket(self, batch: int) -> int:
        for b in self.buckets:
            if b >= batch:
                return b
        raise ValueError(f"decode batch {batch} exceeds the largest bucket {self.buckets[-1]}")

    @torch.inference_mode()
    def decode(
        self, tokens: Sequence[int], slots: Sequence[int], positions: Sequence[int]
    ) -> torch.Tensor:
        """One token for each slot at its position; fp32 logits [batch, vocab].

        The returned tensor may be a graph's static output: read it before the next call.
        """
        batch = len(tokens)
        for slot, position in zip(slots, positions, strict=True):
            self._ensure_pages(slot, position + 1)
        size = self.bucket(batch) if self.use_graphs else batch
        host = torch.zeros((4, self._rows), dtype=torch.long)
        host[1].fill_(self.dummy)
        host[3].fill_(self.dummy)
        host[0, :batch] = torch.tensor(list(tokens), dtype=torch.long)
        host[1, :batch] = torch.tensor(list(slots), dtype=torch.long)
        host[2, :batch] = torch.tensor(list(positions), dtype=torch.long)
        host[3, :batch] = torch.tensor([self._g(s) for s in slots], dtype=torch.long)
        for meta in self.devices.values():
            meta.packed.copy_(host)
        x, _ = self._run(self.segments, size, size)
        return x[:batch]

    def capture(self, size: int, verify: bool = False) -> None:
        """Capture every segment's graph for batch ``size``; on failure record why and run eagerly.

        Capture modes are tried in turn (``ENGINE_CAPTURE_MODES``): "global"
        refuses any unsafe HIP call from any thread during capture, while
        "thread_local" and "relaxed" allow the runtime queries some launchers
        make. The first that captures is kept for later buckets.
        """
        modes = getattr(self, "_capture_modes", None)
        if modes is None:
            modes = os.environ.get("ENGINE_CAPTURE_MODES", "global,thread_local,relaxed").split(",")
        errors = []
        for mode in modes:
            error = self._capture(size, mode, verify)
            if error is None:
                self._capture_modes = [mode]
                self.capture_mode = mode
                self.graph_error = None
                return
            errors.append(f"{mode}: {error}")
        self.graph_error = " | ".join(errors)

    def _capture(self, size: int, mode: str, verify: bool = False) -> str | None:
        segments = self.vsegments if verify else self.segments
        rows = 2 * size if verify else size
        try:
            meta0 = self.devices[self.embed.device]
            for meta in self.devices.values():
                meta.slots[:rows].fill_(self.dummy)
                meta.positions[:rows].zero_()
                meta.gdn[:rows].fill_(self.dummy)
                if verify:
                    meta.gdn[size:rows].fill_(self.dummy + self.gdn_rows)
            x = meta0.tokens[:rows]
            for seg in segments:
                with torch.cuda.device(seg.device):
                    meta = self.devices[seg.device]
                    static_in = x if seg is segments[0] else x.to(seg.device).clone()
                    # One side stream per device, also passed to torch.cuda.graph: its default
                    # capture stream is a process-wide singleton made on the first device, so
                    # a segment on another GPU would be captured on a foreign stream.
                    streams = self.__dict__.setdefault("_capture_streams", {})
                    if seg.device not in streams:
                        streams[seg.device] = torch.cuda.Stream(seg.device)
                    stream = streams[seg.device]
                    stream.wait_stream(torch.cuda.current_stream(seg.device))
                    with torch.cuda.stream(stream):
                        for _ in range(2):  # compile the Triton kernels, warm the BLAS handles
                            seg.run(static_in, meta)
                    torch.cuda.current_stream(seg.device).wait_stream(stream)
                    torch.cuda.synchronize(seg.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        graph,
                        pool=self._pool(seg.device),
                        stream=stream,
                        capture_error_mode=mode,
                    ):
                        out = seg.run(static_in, meta)
                    seg.inputs[size], seg.outputs[size], seg.graphs[size] = static_in, out, graph
                    x = out
            if verify:
                self.__dict__.setdefault("_verify_static", {})[size] = self._verify_out
            for dev in self.devices:
                torch.cuda.synchronize(dev)
        except Exception as exc:  # noqa: BLE001 - any capture failure falls back to eager
            for seg in self.segments + self.vsegments:
                seg.graphs.clear()
                seg.inputs.clear()
                seg.outputs.clear()
            # Where in the forward the capture broke: the innermost frame of this package.
            where = [
                f"{Path(f.filename).name}:{f.lineno} {f.name}"
                for f in traceback.extract_tb(exc.__traceback__)
                if "engine" in f.filename
            ]
            first = str(exc).splitlines()[0] if str(exc) else ""
            try:  # clear a sticky error before the next attempt
                for dev in self.devices:
                    torch.cuda.synchronize(dev)
            except Exception:  # noqa: BLE001
                pass
            return f"{type(exc).__name__}: {first} at {where[-1] if where else '?'}"
        return None

    def _pool(self, dev: torch.device):
        pools = self.__dict__.setdefault("_pools", {})
        if dev not in pools:
            pools[dev] = torch.cuda.graph_pool_handle()
        return pools[dev]

    def capture_all(self, decode: bool = True, verify: bool = True) -> None:
        """Capture every bucket, largest first so the graphs share one memory pool per device:
        the decode steps' and (with the MTP head) the verify steps'."""
        if not self.use_graphs:
            return
        kinds = [k for k, on in ((True, verify and self.mtp is not None), (False, decode)) if on]
        for is_verify in kinds:
            segments = self.vsegments if is_verify else self.segments
            for size in sorted(self.buckets, reverse=True):
                if self.graph_error is None and size not in segments[0].graphs:
                    self.capture(size, is_verify)
