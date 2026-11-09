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

from engine import batching, experts, gdn, reqstamps, speculative
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


def use_tensor_parallel() -> bool:
    """``ENGINE_TP``: "2" (default) splits every layer over the replica's two GPUs
    (``engine.fast.tp``), "1" keeps the layer split; it applies only with exactly two visible
    GPUs (a replica of the default 4 x 2 layout)."""
    from engine.fast import tp

    return tp.size() == 2 and torch.cuda.is_available() and torch.cuda.device_count() == 2


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
            # Plan 46: cheap element probes first, so only a real match pays the slice.
            and prompt[length - 1] == key[-1]
            and prompt[length >> 1] == key[length >> 1]
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
        # Self-speculative decoding (``engine.speculative``): tokens a verify step picked,
        # the MTP head's draft of the token after ``last_token``, and whether the last step
        # flipped the slot's state bank.
        self.speculates = generator.fast is not None and generator.fast.mtp is not None and (
            speculative.eligible(sampling)
        )
        self.queue: list[int] = []
        self.draft = 0
        self.flipped = False
        # The chained verify (plan 31): the head's next drafts, ring entries the last step
        # advanced, and this request's own decayed counts of drafts checked and accepted.
        self.drafts: list[int] = []
        self.advanced = 0
        self.spec_checked = 0.0
        self.spec_accepted = 0.0
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

    def has_queued(self) -> bool:
        """Whether the next token is already picked (the server takes it without a decode)."""
        return bool(self.queue)

    def continues_after(self, token: int, ahead: int = 0) -> bool:
        """Whether, picking ``token`` next (after ``ahead`` already queued tokens), this request
        still needs another token."""
        if token in self._generator.eos_ids and not self._sampling.ignore_eos:
            return False
        return self._count + ahead + 1 < self._max_tokens

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


# Plan 46: tokenize completion prompts in the HTTP handler thread, incrementally (exact).
REQ_FAST = os.environ.get("ENGINE_REQ_FAST", "1") == "1"


def _prompt_encoder(tokenizer: Any) -> Any:
    from engine.inctok import PromptEncoder

    backend = None
    source = getattr(tokenizer, "backend_tokenizer", None)
    if source is not None:
        try:
            from tokenizers import Tokenizer

            backend = Tokenizer.from_str(source.to_str())  # its own instance for handler threads
        except Exception:  # noqa: BLE001 - full encodes only
            backend = None
    return PromptEncoder(lambda text: tokenizer.encode(text), backend)


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
        # Fast path: a prompt whose uncached part is at most this many tokens runs as ONE
        # prefill pass (last token included); 0 restores chunk-aligned passes plus a separate
        # last-token pass (``_chunks``).
        self._one_pass_max = int(os.environ.get("ENGINE_PREFILL_ONE_PASS_MAX", "4096"))
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
        # Plan 46: exact incremental tokenization of growing prompts (``engine.inctok``).
        self.prompt_encoder = _prompt_encoder(tokenizer)
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
            fast_options = {
                "slots": int(os.environ.get("ENGINE_SLOTS", "32")),
                # A verify step writes its draft one position past the last token.
                "max_len": options["max_model_len"] + (1 if head is not None else 0),
                "mtp": head,
            }
            try:
                if use_tensor_parallel():
                    # ENGINE_TP=2: every layer split over the replica's two GPUs (plan 13).
                    from engine.fast.tp import TPModel

                    fast = TPModel(lm, **fast_options)
                else:
                    fast = FastModel(lm, **fast_options)
            except FastPathUnavailable as exc:
                print(f"engine: fast path unavailable ({exc}); using Transformers", flush=True)
            else:
                # With the MTP head every step is a verify step: capture those, not plain decode.
                from engine.fast import mfma

                chain = head is not None and getattr(fast, "chain", False)
                fast.capture_all(decode=head is None, verify=not chain)
                if chain and fast.use_graphs:
                    # Plan 31: the chained verify's graphs, k + 1 = 8, 4, 2 rows per slot.
                    for groups in (8, 4, 2):
                        for size in sorted(fast.buckets, reverse=True):
                            if groups <= mfma.RING and groups * size <= 64 and fast.graph_error is None:
                                fast.capture(size, False, groups)
                    if hasattr(fast, "warm_chain_next") and fast.graph_error is None:
                        fast.warm_chain_next()  # plan 42's run-ahead input kernel
                print(
                    f"engine: fast path, triton={fast.use_triton} graphs={fast.use_graphs} "
                    f"graph_error={fast.graph_error} kv_pages={fast.pool.pages} "
                    f"mtp={head is not None} tp={2 if use_tensor_parallel() else 1}",
                    flush=True,
                )
        generator = cls(lm, tokenizer, eos_ids, fast=fast, **options)
        gdn.ready(generator, kernels)
        return generator

    def encode_text(self, prompt: str) -> tuple[int, ...]:
        """Exactly ``tokenizer.encode(prompt)``; with ``ENGINE_REQ_FAST`` (default on) reusing
        the ids of a recent prompt that ``prompt`` extends. Thread-safe."""
        if REQ_FAST:
            return self.prompt_encoder.encode(prompt)
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
        """``_prefill_fast``'s passes as (begin, end, snapshot after) spans.

        With ``ENGINE_PREFILL_ONE_PASS_MAX`` (default 4096) above 0, an uncached part of at
        most that many tokens is one span, the last token included: one pass and no
        snapshot (a multi-round session resumes from its session tail). A longer one runs
        in spans ending at multiples of the chunk size, the last running to the end of the
        prompt, with a snapshot after the last chunk boundary before the end (reused by a
        repeat of the prompt, at a length independent of where a reuse began). With 0:
        spans end at multiples of the chunk size, then the last token runs alone, with
        snapshots after the last boundary and after all but the last token.
        """
        n = len(prompt)
        if self._one_pass_max > 0:
            if n - start <= self._one_pass_max:
                return [(start, n, False)]
            last_boundary = ((n - 1) // self._chunk) * self._chunk
            spans = []
            position = start
            while position < n:
                end = min((position // self._chunk + 1) * self._chunk, n)
                spans.append((position, end, end == last_boundary))
                position = end
            return spans
        body = n - 1
        last_boundary = (body // self._chunk) * self._chunk
        spans = []
        position = start
        while position < body:
            end = min((position // self._chunk + 1) * self._chunk, body)
            store = (end == last_boundary and end != body) or end == body
            spans.append((position, end, store))
            position = end
        spans.append((body, n, False))
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
            **self.prompt_encoder.stats(),
        }

    # ---------------------------------------------------------------- fast path

    def warmup_fast(self, token_ids: list[int]) -> None:
        """Run the fast path's prefill variants once in a free slot (see ``FastModel.warmup``)."""
        self._reclaim()
        if self._free_slots:
            other = self._free_slots[-2] if self.packing and len(self._free_slots) > 1 else None
            # One-pass prefill runs spans up to ``_one_pass_max`` tokens: warm the bucketed
            # conv and prefill-expert shapes up to that length too.
            longest = max(self._chunk, self._one_pass_max)
            self.fast.warmup(self._free_slots[-1], longest, token_ids, other)

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
        """``prefill``'s passes (``_chunks``) and snapshots, into ``slot``; the last logits."""
        logits = None
        for begin, end, store in self._chunks(prompt, start):
            # ``prompt[end]`` follows the span (for the MTP head's last pair), None at the end.
            follow = prompt[end] if end < len(prompt) else None
            logits = self.fast.prefill(slot, prompt[begin:end], begin, follow)
            self.prefill_passes += 1
            self.prefill_segments += 1
            if store:
                self._prefix_cache.store(prompt[:end], slot)
        return logits

    def _decode_fast(self, generations: list[Generation], ahead_ok: bool = False) -> None:
        if self.fast.mtp is not None:
            speculative.verify_step(self, generations, ahead_ok)
            return
        logits = self.fast.decode(
            [g.last_token for g in generations],
            [g.slot for g in generations],
            [g.length for g in generations],
        )
        for g, row in zip(generations, logits, strict=True):
            g.logits = row.clone()
            g.length += 1

    def decode(self, generations: list[Generation], ahead_ok: bool = False) -> None:
        """Feed each generation's last token in one forward pass and store its next logits.

        ``ahead_ok``: the caller has no new request or prefill waiting, so the fast path may
        launch the next verify step before this one's picks are read (plan 42).

        One request runs alone on its own cache. Several are stacked into one
        batch: left-padded attention keys and values, a padding mask, and each
        request's own position.
        """
        self.decode_steps += 1
        self.decoded_tokens += len(generations)
        self.largest_batch = max(self.largest_batch, len(generations))
        if self.fast is not None:
            self._decode_fast(generations, ahead_ok)
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
                if reqstamps.ENABLED:
                    t_lookup = reqstamps.now()
                    reused, snap, tail = g._prefix_cache.lookup(prompt)
                    reqstamps.last_lookup = reqstamps.now() - t_lookup
                else:
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
