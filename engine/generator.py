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

Token selection (``ENGINE_ZERO_SYNC``, on unless "0"): after each decode step
the next token of every request that is greedy at that step (temperature 0 and
no end-of-sequence masking for ``min_tokens``) is picked by one ``argmax`` over
the whole batch on the device and read back with one copy into pinned host
memory, instead of one argmax and one blocking ``int()`` per request on a
cloned logits row. ``torch.argmax`` over the batch returns the same index as
over each row alone (the first maximum), so greedy outputs do not change.
Sampled requests keep their cloned row and ``pick``.
"""

from __future__ import annotations

import copy
import json
import os
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig

from engine import batching, experts, gdn, speculative
from engine.protocol import ChatRequest, FinishReason, RequestError, Sampling

CHECKPOINT = json.loads((Path(__file__).with_name("checkpoint.json")).read_text())


def lookahead_enabled(setting: str | None = None) -> bool:
    """``ENGINE_LOOKAHEAD``: decode steps queued without waiting (``Generator.launch``) and
    the server's one-step lookahead (``engine.server.Worker``); on unless "0"."""
    if setting is None:
        setting = os.environ.get("ENGINE_LOOKAHEAD")
    return (setting or "1").strip() != "0"


def zero_sync_enabled(setting: str | None = None) -> bool:
    """``ENGINE_ZERO_SYNC``: batched on-device greedy picks and off-thread text; on unless "0"."""
    if setting is None:
        setting = os.environ.get("ENGINE_ZERO_SYNC")
    return (setting or "1").strip() != "0"


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
        # The next token when ``Generator.decode`` already picked it on the device (greedy).
        self.next_token: int | None = None
        self.done = False
        self.slot = slot
        self._ended_slot: int | None = None  # the slot ``finished`` queued, until ``retire``
        self.length = length
        # Self-speculative decoding (``engine.speculative``): tokens a verify step picked,
        # the MTP head's draft of the token after ``last_token``, and whether the last step
        # flipped the slot's state bank.
        self.speculates = generator.fast is not None and generator.fast.mtp is not None and (
            speculative.eligible(sampling)
        )
        self.queue: list[int] = []
        self.draft = 0
        self.flipped = False
        # Launched decode steps (``Generator.launch``) that include this request and have not
        # been collected yet; its slot state is then ahead of what the host has seen.
        self.inflight = 0
        if slot is not None:
            # A request dropped before it ends (a client gone) gives its slot back when collected.
            self._finalizer = weakref.finalize(self, generator._released.append, slot)
        self._rng = None
        if sampling.seed is not None:
            self._rng = torch.Generator(device=logits.device).manual_seed(sampling.seed)

    def next_step(self) -> Step:
        """Pick the next token from the current logits; ``decode`` must run before the next call."""
        g = self._generator
        if self.queue:  # picked by a verify step: the argmax ``pick`` would return
            token = self.queue.pop(0)
        elif self.next_token is not None:  # picked on the device by a batched greedy argmax
            token, self.next_token = self.next_token, None
        else:
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

    def feed_token(self) -> int | None:
        """The token the next decode step feeds: the last one picked, taken or not."""
        if self.queue:
            return self.queue[-1]
        if self.next_token is not None:
            return self.next_token
        return self.last_token

    def runs_past_picked(self) -> bool:
        """Whether this request needs another decode step after taking every token already
        picked for it (a verify step's queue, or a batched greedy pick), judged on token ids
        alone, so that step can be launched before those tokens are taken. False when none
        is picked, and for requests with stop strings, whose end depends on their text."""
        picked = list(self.queue)
        if self.next_token is not None:
            picked.append(self.next_token)
        if not picked or self._sampling.stop or self.slot is None:
            return False
        count = self._count
        for token in picked:
            count += 1
            if token in self._generator.eos_ids and not self._sampling.ignore_eos:
                return False
            if count >= self._max_tokens:
                return False
        return True

    def has_queued(self) -> bool:
        """Whether the next token is already picked (the server takes it without a decode)."""
        return bool(self.queue)

    def continues_after(self, token: int) -> bool:
        """Whether, picking ``token`` next, this request still needs another token."""
        if token in self._generator.eos_ids and not self._sampling.ignore_eos:
            return False
        return self._count + 1 < self._max_tokens

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
            speculative.settle(g, self, slot)
            # A launched step still running for this request (never, by the server's rule for
            # launching ahead) would leave the slot past ``key``: keep no tail then.
            if slot is not None and self._prompt and self._picked and not self.inflight:
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
        # Packed prefill counters: segments per pass tell how wide prefill passes were.
        self.prefill_passes = 0
        self.prefill_segments = 0
        # Self-speculative decoding counters: verify steps, drafts checked and accepted.
        self.verify_steps = 0
        self.mtp_proposed = 0
        self.mtp_accepted = 0
        # Launched decode steps and the server's one-step lookahead (``ENGINE_LOOKAHEAD``).
        self.lookahead = fast is not None and lookahead_enabled()
        self.launched_steps = 0
        self.lookahead_steps = 0
        # Batched device-side greedy picks (``ENGINE_ZERO_SYNC``) and their pinned readback buffers.
        self.zero_sync = zero_sync_enabled()
        self._readback: dict[torch.device, tuple[torch.Tensor, Any]] = {}
        # Step timing, shown on /health: the time between consecutive decode steps of a busy
        # loop (wall per step) and, with zero sync, the part spent waiting for the device.
        self.decode_intervals = 0
        self.decode_interval_s = 0.0
        self.decode_wait_s = 0.0
        self._last_decode: float | None = None
        # Connection threads detokenize with their own copy, which only ever decodes.
        self.text_tokenizer = tokenizer
        if self.zero_sync:
            try:
                self.text_tokenizer = copy.deepcopy(tokenizer)
            except Exception:  # noqa: BLE001 - decoding is read-only; sharing also works
                pass
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

            head = speculative.load_head(lm, model, revision)
            try:
                fast = FastModel(
                    lm,
                    slots=int(os.environ.get("ENGINE_SLOTS", "32")),
                    # A verify step writes its draft one position past the last token.
                    max_len=options["max_model_len"] + (1 if head is not None else 0),
                    mtp=head,
                )
            except FastPathUnavailable as exc:
                print(f"engine: fast path unavailable ({exc}); using Transformers", flush=True)
            else:
                # With the MTP head every step is a verify step: capture those, not plain decode.
                fast.capture_all(decode=head is None)
                print(
                    f"engine: fast path, triton={fast.use_triton} graphs={fast.use_graphs} "
                    f"graph_error={fast.graph_error} kv_pages={fast.pool.pages} "
                    f"mtp={head is not None}",
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
        max_tokens = self._validate(prompt, sampling)
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

    def start_many(
        self, requests: list[tuple[tuple[int, ...], Sampling]]
    ) -> list[Generation | Exception]:
        """``start`` for several prompts, with their prefill chunks packed into shared passes.

        Returns, per request, its ``Generation`` or the exception ``start``
        would have raised. Without packing (``packing`` is False) it calls
        ``start`` per request.
        """
        if not self.packing or len(requests) == 1:
            results: list[Generation | Exception] = []
            for prompt, sampling in requests:
                try:
                    results.append(self.start(prompt, sampling))
                except Exception as exc:  # noqa: BLE001 - reported per request
                    results.append(exc)
            return results
        packer = PackedPrefill(self)
        tickets = [packer.add(prompt, sampling) for prompt, sampling in requests]
        done: dict[int, Generation | Exception] = dict(packer.failed())
        while packer.busy:
            done.update(packer.step())
        return [done[t] for t in tickets]

    @property
    def packing(self) -> bool:
        """Whether prefill chunks of several requests share passes (fast path, ENGINE_PREFILL_PACK)."""
        return self.fast is not None and os.environ.get("ENGINE_PREFILL_PACK", "1") != "0"

    def _chunks(self, prompt: tuple[int, ...], start: int) -> list[tuple[int, int, bool]]:
        """``_prefill_fast``'s passes as (begin, end, snapshot after) spans."""
        body = len(prompt) - 1
        last_boundary = (body // self._chunk) * self._chunk
        spans = []
        position = start
        while position < body:
            end = min((position // self._chunk + 1) * self._chunk, body)
            store = (end == last_boundary and end != body) or end == body
            spans.append((position, end, store))
            position = end
        spans.append((body, len(prompt), False))
        return spans

    def _validate(self, prompt: tuple[int, ...], sampling: Sampling) -> int:
        """``start``'s checks; the request's max_tokens."""
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
        return sampling.max_tokens if sampling.max_tokens is not None else budget

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
            "prefill_passes": self.prefill_passes,
            "prefill_segments": self.prefill_segments,
            "mtp_verify_steps": self.verify_steps,
            "mtp_proposed": self.mtp_proposed,
            "mtp_accepted": self.mtp_accepted,
            "prefilled_tokens": self.prefilled_tokens,
            "cached_tokens": self.cached_tokens,
            "prefix_cache_tail_hits": pc.tail_hits,
            "prefix_cache_entries": len(pc),
            "prefix_cache_bytes": pc.bytes,
            "prefix_cache_evictions": pc.evictions,
            "zero_sync": int(self.zero_sync),
            "lookahead": int(self.lookahead),
            "launched_steps": self.launched_steps,
            "lookahead_steps": self.lookahead_steps,
            "decode_intervals": self.decode_intervals,
            "decode_interval_ms": round(self.decode_interval_s * 1e3, 1),
            "decode_wait_ms": round(self.decode_wait_s * 1e3, 1),
        }

    # ---------------------------------------------------------------- fast path

    def warmup_fast(self, token_ids: list[int]) -> None:
        """Run the fast path's prefill variants once in a free slot (see ``FastModel.warmup``)."""
        self._reclaim()
        if self._free_slots:
            other = self._free_slots[-2] if self.packing and len(self._free_slots) > 1 else None
            self.fast.warmup(self._free_slots[-1], self._chunk, token_ids, other)

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
        generation = Generation(
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
        if self.fast.mtp is not None:
            generation.draft = int(self.fast.last_drafts[0])
        return generation

    def _prefill_fast(self, prompt: tuple[int, ...], start: int, slot: int) -> torch.Tensor:
        """``prefill``'s chunks and snapshots, into ``slot``."""
        body = len(prompt) - 1
        last_boundary = (body // self._chunk) * self._chunk
        position = start
        while position < body:
            end = min((position // self._chunk + 1) * self._chunk, body)
            # ``prompt[end]`` follows the chunk (for the MTP head's last pair; ``end < body``).
            self.fast.prefill(slot, prompt[position:end], position, prompt[end])
            position = end
            if position == last_boundary and position != body:
                self._prefix_cache.store(prompt[:position], slot)
        if body > start:
            self._prefix_cache.store(prompt[:body], slot)
        return self.fast.prefill(slot, prompt[body:], body)

    def _decode_fast(self, generations: list[Generation]) -> None:
        if self.fast.mtp is not None:
            speculative.verify_step(self, generations)
            return
        logits = self.fast.decode(
            [g.last_token for g in generations],
            [g.slot for g in generations],
            [g.length for g in generations],
        )
        for g in generations:
            g.length += 1
        self._select(generations, logits, clone=True)

    def _greedy_now(self, generation: Generation) -> bool:
        """Whether ``pick`` would take a plain argmax for this request's next token."""
        s = getattr(generation, "_sampling", None)  # None: a warmup stand-in, never picked
        if s is None or s.temperature != 0.0:
            return False
        return s.ignore_eos or generation._count >= s.min_tokens or not self.eos_ids

    def _select(self, generations: list[Generation], logits: torch.Tensor, *, clone: bool) -> None:
        """Hand each generation its next logits row; with zero sync, also pick greedy tokens.

        ``logits`` is [batch, vocab]; ``clone`` copies the rows a later ``pick``
        reads (the fast path returns a graph's static output, overwritten by the
        next step). Rows of requests picked here are left as views.
        """
        if not self.zero_sync:
            for g, row in zip(generations, logits, strict=True):
                g.logits = row.clone() if clone else row
            return
        greedy = []
        for i, (g, row) in enumerate(zip(generations, logits, strict=True)):
            if self._greedy_now(g):
                greedy.append(i)
                g.logits = row
            else:
                g.logits = row.clone() if clone else row
        if not greedy:
            return
        rows = logits
        if len(greedy) != len(generations):
            rows = logits.index_select(0, torch.tensor(greedy, device=logits.device))
        tokens = self._read_ids(torch.argmax(rows, dim=-1))
        for i, token in zip(greedy, tokens, strict=True):
            generations[i].next_token = token

    def _read_ids(self, ids: torch.Tensor) -> list[int]:
        """``ids.tolist()`` through one non-blocking copy into a reused pinned buffer."""
        dev = ids.device
        if dev.type != "cuda":
            return ids.tolist()
        n = ids.numel()
        buf, event = self._readback.get(dev, (None, None))
        if buf is None or buf.numel() < n:
            buf = torch.empty(max(n, 64), dtype=ids.dtype, pin_memory=True)
            event = torch.cuda.Event()
            self._readback[dev] = (buf, event)
        out = buf[:n]
        with torch.cuda.device(dev):
            out.copy_(ids, non_blocking=True)
            event.record(torch.cuda.current_stream(dev))
        start = time.perf_counter()
        event.synchronize()  # releases the GIL: connection threads render text meanwhile
        self.decode_wait_s += time.perf_counter() - start
        return out.tolist()

    def decode(self, generations: list[Generation]) -> None:
        """Feed each generation's last token in one forward pass and store its next logits.

        One request runs alone on its own cache. Several are stacked into one
        batch: left-padded attention keys and values, a padding mask, and each
        request's own position.
        """
        self._count_step(generations)
        if self.fast is not None:
            self._decode_fast(generations)
            return
        with torch.inference_mode():
            if len(generations) == 1:
                g = generations[0]
                logits = self.forward([g.last_token], g.cache)
                self._select(generations, logits.unsqueeze(0), clone=False)
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
            for g, cache in zip(generations, batching.split(merged, lengths), strict=True):
                g.cache = cache
            self._select(generations, logits, clone=False)

    def _count_step(self, generations: list[Generation]) -> None:
        self.decode_steps += 1
        self.decoded_tokens += len(generations)
        self.largest_batch = max(self.largest_batch, len(generations))
        now = time.perf_counter()
        if self._last_decode is not None and now - self._last_decode < 1.0:
            self.decode_intervals += 1
            self.decode_interval_s += now - self._last_decode
        self._last_decode = now

    # ---------------------------------------------------------------- launched steps

    def launchable(self, generations: list[Generation]) -> bool:
        """Whether one decode step for ``generations`` can be launched without waiting for it
        (``launch``): the fast path with ``ENGINE_LOOKAHEAD`` on, and every request greedy
        (``speculative.eligible``; with the MTP head, every one speculating), so its next
        tokens are the step's argmax picks and its logits need not be kept."""
        if not self.lookahead or not generations:
            return False
        mtp = self.fast.mtp is not None
        return all(
            g.slot is not None
            and speculative.eligible(g._sampling)
            and (g.speculates or not mtp)
            for g in generations
        )

    def launch(self, generations: list[Generation], ahead: bool = False) -> Launched | None:
        """Run one decode step for ``generations``; with ``launchable`` ones, only queue it.

        Each request feeds its last picked token (``Generation.feed_token``):
        normally the one it just took; ``ahead`` (the server's lookahead), the
        last one a collected step picked for it, before the request takes it.
        Returns the queued step, which ``collect`` finishes, or None when the
        step ran here as ``decode``.
        """
        if not self.launchable(generations):
            self.decode(generations)
            return None
        self._count_step(generations)
        self.launched_steps += 1
        self.lookahead_steps += int(ahead)
        fast = self.fast
        tokens = [g.feed_token() for g in generations]
        slots = [g.slot for g in generations]
        lengths = [g.length for g in generations]
        if fast.mtp is not None:
            step = fast.launch_verify(tokens, [g.draft for g in generations], slots, lengths)
        else:
            step = fast.launch_decode(tokens, slots, lengths)
            for g in generations:
                g.length += 1
        for g in generations:
            g.inflight += 1
        return Launched(list(generations), step, fast.mtp is not None)

    def collect(self, launched: Launched) -> None:
        """Wait for a launched step and hand each request its picks, exactly as ``decode``
        would have: a verify step's queue and draft (``speculative.apply``), or the greedy
        token of a plain step."""
        start = time.perf_counter()
        try:
            rows = launched.step.result()
        finally:
            self.decode_wait_s += time.perf_counter() - start
            for g in launched.generations:
                g.inflight -= 1
        if launched.verify:
            g1, d1, g2, d2 = rows
            self.verify_steps += 1
            speculative.apply(self, launched.generations, g1, g2, d1, d2)
        else:
            for g, token in zip(launched.generations, rows[0], strict=True):
                g.next_token = token

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


@dataclass
class Launched:
    """A decode step ``Generator.launch`` queued: its requests, in batch order, and its picks."""

    generations: list[Generation]
    step: Any  # engine.fast.model.LaunchedStep
    verify: bool


@dataclass
class _Pending:
    """One request inside a ``PackedPrefill``: its slot and the chunks still to run."""

    ticket: int
    prompt: tuple[int, ...]
    sampling: Sampling
    max_tokens: int
    slot: int
    reused: int
    spans: list[tuple[int, int, bool]]
    at: int = 0


class PackedPrefill:
    """Prefill of several requests on the fast path, one shared pass per ``step``.

    ``add`` validates a request, takes a slot and restores its longest usable
    prefix snapshot, exactly as ``Generator.start``; its prompt is split into
    the chunks ``start`` would run (``Generator._chunks``), with the same
    snapshots at the same boundaries. Each ``step`` runs the next chunk of every
    pending request that fits ``ENGINE_PREFILL_PACK_TOKENS`` tokens (default
    8192; the oldest request always fits) as one ``FastModel.prefill_packed``
    pass and returns the requests it finished. Requests may be added between
    steps, so a caller can interleave passes with decode steps and admit new
    arrivals into the next pass. A failed pass fails only its own requests.
    """

    def __init__(self, generator: Generator) -> None:
        self._g = generator
        self._budget = int(os.environ.get("ENGINE_PREFILL_PACK_TOKENS", "8192"))
        self._pending: list[_Pending] = []
        self._failed: list[tuple[int, Exception]] = []
        self._tickets = 0

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def add(self, prompt: tuple[int, ...], sampling: Sampling) -> int:
        """Queue a request; its ticket, under which ``step`` or ``failed`` reports it."""
        g = self._g
        ticket = self._tickets
        self._tickets += 1
        try:
            max_tokens = g._validate(prompt, sampling)
            g._reclaim()
            if not g._free_slots:
                raise RequestError("all sequence slots are busy; retry later", 503)
            slot = g._free_slots.pop()
            snap, tail = None, False
            try:
                reused, snap, tail = g._prefix_cache.lookup(prompt)
                if snap is not None:
                    if tail:  # a session tail is handed over: the slot takes its pages
                        g.fast.adopt(snap, slot)
                    else:
                        g.fast.restore(snap, slot)
            except BaseException:
                if snap is not None and tail:
                    g.fast.release(snap)
                g.fast.reset(slot)
                g._free_slots.append(slot)
                raise
        except Exception as exc:  # noqa: BLE001 - reported per request
            self._failed.append((ticket, exc))
            return ticket
        g.cached_tokens += reused
        g.prefilled_tokens += len(prompt) - reused
        self._pending.append(
            _Pending(ticket, prompt, sampling, max_tokens, slot, reused, g._chunks(prompt, reused))
        )
        return ticket

    def failed(self) -> list[tuple[int, Exception]]:
        """Requests ``add`` refused since the last call."""
        out, self._failed = self._failed, []
        return out

    def step(self) -> list[tuple[int, Generation | Exception]]:
        """Run one packed pass; the requests whose prefill it finished (or failed)."""
        g = self._g
        batch, used = [], 0
        for entry in self._pending:
            begin, end, _ = entry.spans[entry.at]
            if batch and used + (end - begin) > self._budget:
                continue
            batch.append(entry)
            used += end - begin
        segments, follows = [], []
        for entry in batch:
            begin, end, _ = entry.spans[entry.at]
            segments.append((entry.slot, entry.prompt[begin:end], begin))
            follows.append(entry.prompt[end] if end < len(entry.prompt) else None)
        g.prefill_passes += 1
        g.prefill_segments += len(segments)
        done: list[tuple[int, Generation | Exception]] = []
        try:
            logits = g.fast.prefill_packed(segments, follows)
            drafts = g.fast.last_drafts.tolist() if g.fast.mtp is not None else None
        except Exception as exc:  # noqa: BLE001 - the pass failed for every request in it
            for entry in batch:
                g.fast.reset(entry.slot)
                g._free_slots.append(entry.slot)
                done.append((entry.ticket, exc))
            self._pending = [e for e in self._pending if all(e is not b for b in batch)]
            return done
        for row, entry in enumerate(batch):
            _, end, store = entry.spans[entry.at]
            if store:
                g._prefix_cache.store(entry.prompt[:end], entry.slot)
            entry.at += 1
            if entry.at == len(entry.spans):
                generation = Generation(
                    g,
                    entry.sampling,
                    entry.reused,
                    None,
                    logits[row].clone(),
                    entry.max_tokens,
                    entry.prompt,
                    slot=entry.slot,
                    length=len(entry.prompt),
                )
                if drafts is not None:
                    generation.draft = drafts[row]
                done.append((entry.ticket, generation))
        self._pending = [e for e in self._pending if e.at < len(e.spans)]
        return done
