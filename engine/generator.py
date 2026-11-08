"""Run Qwen3.6-35B-A3B with Transformers: prefill one request at a time, decode them together.

``Generator.load`` loads the BF16 checkpoint sharded over every visible GPU
(``device_map="auto"``), or onto the CPU when there is none. ``start`` prefills
one prompt in fixed chunks and returns its ``Generation``; ``next_step`` picks
that request's next token, and ``decode`` runs one forward pass that feeds the
last token of every running generation at once. Nothing here is thread-safe:
one thread owns a Generator.

The prefix cache keeps deep copies of the model cache (attention KV plus the
Gated DeltaNet convolution and recurrent states). A recurrent state cannot be
cut back to an earlier position, so a request reuses a snapshot only where one
was taken: at a chunk boundary, or after all but the last token of an earlier
prompt. Because a reused snapshot was computed over the same chunk boundaries a
cold prefill would use, reuse does not change the computation, and identical
greedy requests return identical tokens whether or not they hit the cache.

When a request finishes, its cache covers its prompt and every generated token
but the last (which was picked, never fed). That cache is kept as a *session
tail*: a multi-round client sends the next round as this round's prompt, its
output and new tokens, so the next round resumes from the tail at any length
instead of re-prefilling the history from the last chunk boundary. A tail is
handed over (not copied) to the round that extends it, and a newer tail that
extends an older one replaces it. Entries are budgeted by count, tokens and the
bytes their tensors hold, and evicted least recently used first.

On GPUs the generator runs the engine's own forward instead (``engine.fast``):
the same weights, with each request's state in a fixed slot, decode steps
replayed from captured graphs, and slot-state snapshots for the prefix cache
under the same boundary rules. ``ENGINE_FAST=0`` selects the Transformers path
above, ``ENGINE_FAST=1`` the fast path on any device (the CPU runs its PyTorch
references); the default is the fast path when a GPU is visible. There a
session tail is the finished request's slot state itself: its KV pages are
moved (not copied) into the tail entry and its Gated DeltaNet states cloned,
and the next round adopts those pages into its own slot.
"""

from __future__ import annotations

import copy
import json
import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig

from engine import batching, experts, gdn
from engine.protocol import ChatRequest, FinishReason, RequestError, Sampling

CHECKPOINT = json.loads((Path(__file__).with_name("checkpoint.json")).read_text())


def experts_implementation(hip_version: str | None) -> str | None:
    """The Transformers MoE experts implementation to request; None keeps its default.

    Transformers 5.22 defaults to ``grouped_mm`` (``torch._grouped_mm``). The
    torch 2.9.1+rocm6.4 build raises "grouped gemm is not supported on ROCM"
    when that kernel runs, after Transformers' availability check has passed
    (an MI250 reports compute capability 9.0), so on ROCm every request would
    fail. Transformers' ``eager`` alternative synchronizes with the host once
    per selected expert, so ROCm uses ``engine.experts`` instead.
    """
    return experts.NAME if hip_version else None


def use_fast_path(setting: str | None, gpu: bool) -> bool:
    """``ENGINE_FAST``: "1" forces the engine's own forward, "0" Transformers; unset, GPUs only."""
    if setting in ("0", "1"):
        return setting == "1"
    return gpu


@dataclass(frozen=True)
class Step:
    """One generated token. ``emit`` is false for an end-of-sequence token that ends the output."""

    token_id: int
    emit: bool
    finish: FinishReason | None


def _compact(cache: DynamicCache, *, move: bool) -> tuple[DynamicCache, int]:
    """A copy of ``cache`` whose tensors own compact storage, and the bytes they hold.

    After a batched decode step (``batching.split``), a request's tensors are
    views of the padded batch; keeping such a view (or a plain deep copy of
    it) would keep the whole batch alive. Every tensor is cloned, except that
    with ``move`` a tensor that already owns exactly its storage is taken over
    as is, so ``cache`` must not be used afterwards.
    """
    memo: dict[int, Any] = {}
    nbytes = 0

    def own(tensor: torch.Tensor) -> None:
        nonlocal nbytes
        if id(tensor) in memo:
            return
        owned = tensor._base is None and tensor.untyped_storage().nbytes() == tensor.nbytes
        keep = move and owned
        memo[id(tensor)] = tensor if keep else tensor.clone(memory_format=torch.contiguous_format)
        nbytes += tensor.nbytes

    for layer in cache.layers:
        for value in vars(layer).values():
            values = value.values() if isinstance(value, dict) else (value,)
            for item in values:
                if isinstance(item, torch.Tensor):
                    own(item)
    return copy.deepcopy(cache, memo), nbytes


@dataclass
class _Entry:
    cache: DynamicCache
    nbytes: int
    tail: bool


class PrefixCache:
    """LRU snapshots of the model cache, keyed by the token ids they cover.

    A chunk-boundary or all-but-last-token snapshot is reused only at exactly
    those lengths and is copied on reuse. A session tail (see the module
    docstring) is reused by any longer prompt that starts with it and is handed
    over on reuse.
    """

    def __init__(
        self, chunk_tokens: int, max_tokens: int, max_entries: int, max_bytes: int | None = None
    ) -> None:
        self._chunk = chunk_tokens
        self._max_tokens = max_tokens
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[tuple[int, ...], _Entry] = OrderedDict()
        self._tokens = 0
        self.bytes = 0
        self.tail_hits = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _usable(self, key: tuple[int, ...], entry: _Entry, prompt: tuple[int, ...]) -> bool:
        length = len(key)
        return (
            0 < length < len(prompt)
            and (entry.tail or length % self._chunk == 0 or length == len(prompt) - 1)
            and prompt[:length] == key
        )

    def lookup(self, prompt: tuple[int, ...]) -> tuple[int, DynamicCache | None]:
        """The longest usable snapshot for ``prompt``, and how many tokens it covers.

        The caller owns the returned cache: a snapshot is copied, a tail is removed.
        """
        best = max(
            (k for k, e in self._entries.items() if self._usable(k, e, prompt)),
            key=len,
            default=None,
        )
        if best is None:
            return 0, None
        entry = self._entries[best]
        if entry.tail:
            self._remove(best)
            self.tail_hits += 1
            return len(best), entry.cache
        self._entries.move_to_end(best)
        return len(best), copy.deepcopy(entry.cache)

    def store(self, key: tuple[int, ...], cache: DynamicCache) -> None:
        """Keep a copy of ``cache``, which covers exactly ``key``."""
        if key in self._entries:
            self._entries.move_to_end(key)
            return
        if not self._admits(key):
            return
        copied, nbytes = _compact(cache, move=False)
        self._insert(key, _Entry(copied, nbytes, tail=False))

    def store_tail(self, key: tuple[int, ...], cache: DynamicCache) -> None:
        """Keep ``cache``, which covers exactly ``key``, as a session tail; takes ownership."""
        for older in [k for k, e in self._entries.items() if e.tail and key[: len(k)] == k]:
            self._remove(older)  # superseded: the new tail extends it
        if key in self._entries:
            self._entries[key].tail = True
            self._entries.move_to_end(key)
            return
        if not self._admits(key):
            return
        compact, nbytes = _compact(cache, move=True)
        self._insert(key, _Entry(compact, nbytes, tail=True))

    def _admits(self, key: tuple[int, ...]) -> bool:
        return bool(key) and len(key) <= self._max_tokens and self._max_entries > 0

    def _insert(self, key: tuple[int, ...], entry: _Entry) -> None:
        if self._max_bytes is not None and entry.nbytes > self._max_bytes:
            return
        self._entries[key] = entry
        self._tokens += len(key)
        self.bytes += entry.nbytes
        while (
            self._tokens > self._max_tokens
            or len(self._entries) > self._max_entries
            or (self._max_bytes is not None and self.bytes > self._max_bytes)
        ):
            self._remove(next(iter(self._entries)))
            self.evictions += 1

    def _remove(self, key: tuple[int, ...]) -> None:
        entry = self._entries.pop(key)
        self._tokens -= len(key)
        self.bytes -= entry.nbytes


@dataclass
class _FastEntry:
    snap: Any  # engine.fast.model.Snapshot
    nbytes: int
    tail: bool


class FastPrefixCache:
    """``PrefixCache``'s rules over ``FastModel`` slot snapshots instead of cache copies.

    A snapshot holds cloned Gated DeltaNet states and references to KV pages
    of the fast model's page pool. A chunk-boundary or all-but-last-token
    snapshot is restored by copy; a session tail holds the finished request's
    own pages (moved out of its slot) and is handed over to the round that
    extends it. ``max_bytes`` budgets the cloned Gated DeltaNet states; KV
    pages are bounded by the pool, which evicts the least recently used entry
    when it runs out.
    """

    def __init__(
        self,
        fast: Any,
        chunk_tokens: int,
        max_tokens: int,
        max_entries: int,
        max_bytes: int | None = None,
    ) -> None:
        self._fast = fast
        self._chunk = chunk_tokens
        self._max_tokens = max_tokens
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[tuple[int, ...], _FastEntry] = OrderedDict()
        self._tokens = 0
        self.bytes = 0
        self.tail_hits = 0
        self.evictions = 0
        fast.pool.on_pressure = self.evict_one

    _usable = PrefixCache._usable

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, prompt: tuple[int, ...]) -> tuple[int, Any, bool]:
        """The longest usable snapshot for ``prompt``, how many tokens it covers, and whether
        it is a session tail. A tail is removed and the caller owns it (``FastModel.adopt``
        or ``release`` it); any other snapshot stays cached and is restored by copy."""
        best = max(
            (k for k, e in self._entries.items() if self._usable(k, e, prompt)),
            key=len,
            default=None,
        )
        if best is None:
            return 0, None, False
        entry = self._entries[best]
        if entry.tail:
            self._remove(best)
            self.tail_hits += 1
            return len(best), entry.snap, True
        self._entries.move_to_end(best)
        return len(best), entry.snap, False

    def store(self, key: tuple[int, ...], slot: int) -> None:
        """Snapshot ``slot``, which covers exactly ``key``."""
        if key in self._entries:
            self._entries.move_to_end(key)
            return
        if not self._admits(key):
            return
        snap = self._fast.snapshot(slot, len(key))
        self._insert(key, _FastEntry(snap, self._fast.snapshot_bytes(snap), tail=False))

    def store_tail(self, key: tuple[int, ...], slot: int) -> None:
        """Move ``slot``'s state, which covers exactly ``key``, into a session tail.

        The slot keeps no pages afterwards (it is about to be reset). With an
        entry already at ``key``, that entry becomes the tail and the slot's
        pages are left for the reset to free.
        """
        for older in [k for k, e in self._entries.items() if e.tail and key[: len(k)] == k]:
            self._release(older)  # superseded: the new tail extends it
        if key in self._entries:
            self._entries[key].tail = True
            self._entries.move_to_end(key)
            return
        if not self._admits(key):
            return
        snap = self._fast.detach(slot, len(key))
        self._insert(key, _FastEntry(snap, self._fast.snapshot_bytes(snap), tail=True))

    def _admits(self, key: tuple[int, ...]) -> bool:
        return bool(key) and len(key) <= self._max_tokens and self._max_entries > 0

    def _insert(self, key: tuple[int, ...], entry: _FastEntry) -> None:
        if self._max_bytes is not None and entry.nbytes > self._max_bytes:
            self._fast.release(entry.snap)
            return
        self._entries[key] = entry
        self._tokens += len(key)
        self.bytes += entry.nbytes
        while (
            self._tokens > self._max_tokens
            or len(self._entries) > self._max_entries
            or (self._max_bytes is not None and self.bytes > self._max_bytes)
        ):
            self.evict_one()

    def _remove(self, key: tuple[int, ...]) -> _FastEntry:
        entry = self._entries.pop(key)
        self._tokens -= len(key)
        self.bytes -= entry.nbytes
        return entry

    def _release(self, key: tuple[int, ...]) -> None:
        self._fast.release(self._remove(key).snap)

    def evict_one(self) -> bool:
        """Drop the least recently used snapshot; False when there is none."""
        if not self._entries:
            return False
        self._release(next(iter(self._entries)))
        self.evictions += 1
        return True


class Generation:
    """One request after prefill: its cache, the logits for its next token, and its progress.

    On the fast path ``cache`` is None and the request's state is in ``slot``,
    which holds ``length`` tokens; the slot is released when the request ends
    or is dropped.
    """

    def __init__(
        self,
        generator: Generator,
        sampling: Sampling,
        cached_tokens: int,
        cache: DynamicCache | None,
        logits: torch.Tensor,
        max_tokens: int,
        prompt: tuple[int, ...] = (),
        slot: int | None = None,
        length: int = 0,
    ) -> None:
        self._generator = generator
        self._sampling = sampling
        self._prompt = prompt
        self._picked: list[int] = []
        self.cached_tokens = cached_tokens
        self.cache = cache
        self.logits = logits
        self._max_tokens = max_tokens
        self._count = 0
        self.last_token: int | None = None
        self.done = False
        self.slot = slot
        self._ended_slot: int | None = None  # the slot ``finished`` queued, until ``retire``
        self.length = length
        if slot is not None:
            # A request dropped before it ends (a client gone) gives its slot back when collected.
            self._finalizer = weakref.finalize(self, generator._released.append, slot)
        self._rng = None
        if sampling.seed is not None:
            self._rng = torch.Generator(device=logits.device).manual_seed(sampling.seed)

    def next_step(self) -> Step:
        """Pick the next token from the current logits; ``decode`` must run before the next call."""
        g = self._generator
        token = g.pick(self.logits, self._sampling, self._count, self._rng)
        self._count += 1
        self.last_token = token
        self._picked.append(token)
        if token in g.eos_ids and not self._sampling.ignore_eos:
            self.done = True
            g.finished(self)
            return Step(token, emit=False, finish="stop")
        finish: FinishReason | None = "length" if self._count == self._max_tokens else None
        self.done = finish is not None
        if self.done:
            g.finished(self)
        return Step(token, emit=True, finish=finish)

    def retire(self) -> None:
        """Hand this finished request's cache to the prefix cache as a session tail.

        Call once the request has ended and no further decode step will run for
        it. The cache covers the prompt and every picked token but the last,
        which was never fed to the model.
        """
        g = self._generator
        if g.fast is not None:
            # The slot still holds the state (``finished`` only queues it for the reset).
            slot = self.slot if self.slot is not None else self._ended_slot
            if slot is not None and self._prompt and self._picked:
                key = self._prompt + tuple(self._picked[:-1])
                if len(key) == self.length:
                    g.keep_tail_fast(key, slot)
            self._ended_slot = None
            g.finished(self)
            return
        if self.cache is None or not self._prompt or not self._picked:
            return
        key = self._prompt + tuple(self._picked[:-1])
        cache, self.cache = self.cache, None
        g.keep_tail(key, cache)


class Generator:
    """The model, its tokenizer and the prefix cache, owned by one thread."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        eos_ids: frozenset[int],
        *,
        max_model_len: int,
        chunk_tokens: int,
        prefix_cache_tokens: int,
        prefix_cache_entries: int,
        prefix_cache_bytes: int | None = None,
        fast: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.max_model_len = max_model_len
        self._chunk = chunk_tokens
        self.fast = fast
        self._prefix_cache: PrefixCache | FastPrefixCache
        if fast is None:
            self._prefix_cache = PrefixCache(
                chunk_tokens, prefix_cache_tokens, prefix_cache_entries, prefix_cache_bytes
            )
        else:
            self._prefix_cache = FastPrefixCache(
                fast, chunk_tokens, prefix_cache_tokens, prefix_cache_entries, prefix_cache_bytes
            )
            self._free_slots = list(range(fast.slots - 1, -1, -1))
            # Slots of requests that ended or were dropped (list.append is atomic for finalizers).
            self._released: list[int] = []
        self._device = model.get_input_embeddings().weight.device
        # Prompt counters, shown on /health: tokens run through prefill vs reused from the cache.
        self.prefilled_tokens = 0
        self.cached_tokens = 0
        # Decode counters, shown on /health: tokens fed per step tell whether requests were batched.
        self.decode_steps = 0
        self.decoded_tokens = 0
        self.largest_batch = 0
        think_end = tokenizer.convert_tokens_to_ids("</think>")
        self.think_end_id = (
            think_end
            if isinstance(think_end, int) and think_end != tokenizer.unk_token_id
            else None
        )

    @classmethod
    def load(
        cls,
        model: str,
        revision: str | None,
        *,
        experts_impl: str | None = None,
        config: Any = None,
        **options: int,
    ) -> Generator:
        """Load BF16 weights, sharded over all visible GPUs, else on the CPU.

        ``experts_impl`` names the MoE experts implementation; None chooses
        ``experts_implementation(torch.version.hip)``. ``config`` replaces the
        checkpoint's config (a test harness loads a few layers this way).
        """
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        device_map = "auto" if torch.cuda.is_available() else None
        experts.register()
        chosen = experts_impl or experts_implementation(torch.version.hip)
        lm = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            dtype=torch.bfloat16,
            device_map=device_map,
            **({"experts_implementation": chosen} if chosen else {}),
            **({"config": config} if config is not None else {}),
        )
        lm.eval()
        kernels = gdn.install(lm)
        if device_map is not None:
            placed = {str(p.device) for p in lm.parameters()}
            if any(d in ("cpu", "meta", "disk") for d in placed):
                message = f"weights did not fit the visible GPUs (placed on {sorted(placed)}); refusing to offload"
                raise RuntimeError(message)
        try:
            eos = GenerationConfig.from_pretrained(model, revision=revision).eos_token_id
        except OSError:
            eos = None
        if eos is None:
            eos = lm.config.eos_token_id
        eos_ids = frozenset([eos] if isinstance(eos, int) else (eos or []))
        fast = None
        if use_fast_path(os.environ.get("ENGINE_FAST"), torch.cuda.is_available()):
            from engine.fast.model import FastModel, FastPathUnavailable

            try:
                fast = FastModel(
                    lm,
                    slots=int(os.environ.get("ENGINE_SLOTS", "32")),
                    max_len=options["max_model_len"],
                )
            except FastPathUnavailable as exc:
                print(f"engine: fast path unavailable ({exc}); using Transformers", flush=True)
            else:
                fast.capture_all()
                print(
                    f"engine: fast path, triton={fast.use_triton} graphs={fast.use_graphs} "
                    f"graph_error={fast.graph_error} kv_pages={fast.pool.pages}",
                    flush=True,
                )
        generator = cls(lm, tokenizer, eos_ids, fast=fast, **options)
        gdn.ready(generator, kernels)
        return generator

    def encode_text(self, prompt: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(prompt))

    def encode_chat(self, request: ChatRequest) -> tuple[int, ...]:
        """Render the checkpoint's chat template; thinking is off unless the request enables it."""
        try:
            text = self.tokenizer.apply_chat_template(
                list(request.messages),
                tools=list(request.tools) if request.tools else None,
                add_generation_prompt=True,
                tokenize=False,
                **{"enable_thinking": False, **request.template_kwargs},
            )
        except Exception as exc:  # jinja2 raises TemplateError subclasses and plain errors alike
            raise RequestError(f"chat template rejected the messages: {exc}") from exc
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def start(self, prompt: tuple[int, ...], sampling: Sampling) -> Generation:
        """Validate lengths, reuse the longest usable snapshot, and prefill the rest of the prompt."""
        if not prompt:
            raise RequestError("prompt must contain at least one token")
        if any(t >= self.model.config.vocab_size for t in prompt):
            raise RequestError("prompt contains a token id outside the vocabulary")
        budget = self.max_model_len - len(prompt)
        if budget < 1 or (sampling.max_tokens is not None and sampling.max_tokens > budget):
            raise RequestError(
                f"prompt ({len(prompt)} tokens) plus max_tokens ({sampling.max_tokens}) "
                f"exceeds the context length {self.max_model_len}"
            )
        max_tokens = sampling.max_tokens if sampling.max_tokens is not None else budget
        if self.fast is not None:
            return self._start_fast(prompt, sampling, max_tokens)
        with torch.inference_mode():
            reused, cache = self._prefix_cache.lookup(prompt)
            if cache is None:
                cache = DynamicCache(config=self.model.config)
            logits = self.prefill(prompt, reused, cache)
        self.cached_tokens += reused
        self.prefilled_tokens += len(prompt) - reused
        return Generation(self, sampling, reused, cache, logits, max_tokens, prompt)

    def keep_tail(self, key: tuple[int, ...], cache: DynamicCache) -> None:
        """Keep a finished request's ``cache``, covering exactly ``key``, for the next round."""
        with torch.inference_mode():
            self._prefix_cache.store_tail(key, cache)

    def stats(self) -> dict[str, int]:
        """Counters for /health."""
        pc = self._prefix_cache
        return {
            "decode_steps": self.decode_steps,
            "decoded_tokens": self.decoded_tokens,
            "largest_batch": self.largest_batch,
            "prefilled_tokens": self.prefilled_tokens,
            "cached_tokens": self.cached_tokens,
            "prefix_cache_tail_hits": pc.tail_hits,
            "prefix_cache_entries": len(pc),
            "prefix_cache_bytes": pc.bytes,
            "prefix_cache_evictions": pc.evictions,
        }

    # ---------------------------------------------------------------- fast path

    def warmup_fast(self, token_ids: list[int]) -> None:
        """Run the fast path's prefill variants once in a free slot (see ``FastModel.warmup``)."""
        self._reclaim()
        if self._free_slots:
            self.fast.warmup(self._free_slots[-1], self._chunk, token_ids)

    def _reclaim(self) -> None:
        while self._released:
            slot = self._released.pop()
            if slot not in self._free_slots:
                self.fast.reset(slot)
                self._free_slots.append(slot)

    def finished(self, generation: Generation) -> None:
        """``generation`` produced its last token: its slot is free for the next request."""
        if generation.slot is not None:
            generation._finalizer.detach()
            self._released.append(generation.slot)
            generation._ended_slot = generation.slot
            generation.slot = None

    def keep_tail_fast(self, key: tuple[int, ...], slot: int) -> None:
        """Keep a finished request's slot state, covering exactly ``key``, for the next round.

        ``slot`` must not have been reset yet (it is reset only when reclaimed,
        on this thread, at the next ``start``).
        """
        if slot in self._free_slots:
            return
        with torch.inference_mode():
            self._prefix_cache.store_tail(key, slot)

    def _start_fast(self, prompt: tuple[int, ...], sampling: Sampling, max_tokens: int) -> Generation:
        self._reclaim()
        if not self._free_slots:
            raise RequestError("all sequence slots are busy; retry later", 503)
        slot = self._free_slots.pop()
        snap = None
        try:
            reused, snap, tail = self._prefix_cache.lookup(prompt)
            if snap is not None:
                if tail:  # handed over: the slot takes the tail's pages and states
                    self.fast.adopt(snap, slot)
                else:
                    self.fast.restore(snap, slot)
            logits = self._prefill_fast(prompt, reused, slot)
        except BaseException:
            if snap is not None and tail:
                self.fast.release(snap)
            self.fast.reset(slot)
            self._free_slots.append(slot)
            raise
        self.cached_tokens += reused
        self.prefilled_tokens += len(prompt) - reused
        return Generation(
            self,
            sampling,
            reused,
            None,
            logits,
            max_tokens,
            prompt,
            slot=slot,
            length=len(prompt),
        )

    def _prefill_fast(self, prompt: tuple[int, ...], start: int, slot: int) -> torch.Tensor:
        """``prefill``'s chunks and snapshots, into ``slot``."""
        body = len(prompt) - 1
        last_boundary = (body // self._chunk) * self._chunk
        position = start
        while position < body:
            end = min((position // self._chunk + 1) * self._chunk, body)
            self.fast.prefill(slot, prompt[position:end], position)
            position = end
            if position == last_boundary and position != body:
                self._prefix_cache.store(prompt[:position], slot)
        if body > start:
            self._prefix_cache.store(prompt[:body], slot)
        return self.fast.prefill(slot, prompt[body:], body)

    def _decode_fast(self, generations: list[Generation]) -> None:
        logits = self.fast.decode(
            [g.last_token for g in generations],
            [g.slot for g in generations],
            [g.length for g in generations],
        )
        for g, row in zip(generations, logits, strict=True):
            g.logits = row.clone()
            g.length += 1

    def decode(self, generations: list[Generation]) -> None:
        """Feed each generation's last token in one forward pass and store its next logits.

        One request runs alone on its own cache. Several are stacked into one
        batch: left-padded attention keys and values, a padding mask, and each
        request's own position.
        """
        self.decode_steps += 1
        self.decoded_tokens += len(generations)
        self.largest_batch = max(self.largest_batch, len(generations))
        if self.fast is not None:
            self._decode_fast(generations)
            return
        with torch.inference_mode():
            if len(generations) == 1:
                g = generations[0]
                g.logits = self.forward([g.last_token], g.cache)
                return
            merged, lengths = batching.merge([g.cache for g in generations])
            out = self.model(
                input_ids=torch.tensor([[g.last_token] for g in generations], device=self._device),
                attention_mask=batching.padding_mask(lengths, self._device),
                position_ids=torch.tensor([[length] for length in lengths], device=self._device),
                past_key_values=merged,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1].float()
            for g, cache, row in zip(
                generations, batching.split(merged, lengths), logits, strict=True
            ):
                g.cache = cache
                g.logits = row

    def forward(self, ids: list[int] | tuple[int, ...], cache: DynamicCache) -> torch.Tensor:
        """Run ``ids`` through the model on top of ``cache``; logits of the last position."""
        input_ids = torch.tensor([list(ids)], dtype=torch.long, device=self._device)
        out = self.model(
            input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        return out.logits[0, -1].float()

    def prefill(self, prompt: tuple[int, ...], start: int, cache: DynamicCache) -> torch.Tensor:
        """Fill ``cache`` from ``start`` through the prompt; logits after its last token.

        Chunks end at multiples of the chunk size, so the computation never
        depends on where a reused snapshot began. The last chunk boundary before
        the final token, and the position just before it, are snapshotted.
        """
        body = len(prompt) - 1
        last_boundary = (body // self._chunk) * self._chunk
        position = start
        while position < body:
            end = min((position // self._chunk + 1) * self._chunk, body)
            self.forward(prompt[position:end], cache)
            position = end
            if position == last_boundary and position != body:
                self._prefix_cache.store(prompt[:position], cache)
        if body > start:
            self._prefix_cache.store(prompt[:body], cache)
        return self.forward(prompt[body:], cache)

    def pick(
        self, logits: torch.Tensor, sampling: Sampling, generated: int, rng: torch.Generator | None
    ) -> int:
        """Greedy at temperature 0; otherwise top-k/top-p sampling."""
        if not sampling.ignore_eos and generated < sampling.min_tokens and self.eos_ids:
            logits = logits.clone()
            logits[list(self.eos_ids)] = float("-inf")
        if sampling.temperature == 0.0:
            return int(torch.argmax(logits))
        logits = logits / sampling.temperature
        if sampling.top_k > 0:
            kth = torch.topk(logits, min(sampling.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        if sampling.top_p < 1.0:
            sorted_probs, order = torch.sort(probs, descending=True)
            outside = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > sampling.top_p
            sorted_probs[outside] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
        return int(torch.multinomial(probs, 1, generator=rng))
