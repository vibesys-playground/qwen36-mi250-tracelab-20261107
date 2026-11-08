"""Exact-greedy self-speculative decoding with the checkpoint's own MTP head (``engine.mtp``).

Each decode step of a greedy request feeds ``[last, draft]`` instead of
``[last]``: ``last`` is the token the request just picked, ``draft`` the MTP
head's guess of the token after it. The target model returns logits after both
positions. The request then picks its next token from the logits after ``last``
exactly as it would without speculation; if that token equals the draft, the
logits after the draft are already computed and its following token needs no
forward pass. Every emitted token is the target model's own pick, so the
output is the plain greedy output up to the numerics of a 2-token pass.

Rollback. A rejected draft must leave no trace in the request's state. The
attention keys and values of the draft are cropped. Gated DeltaNet
convolution and recurrent states cannot be cut back, so during a verify pass
every Gated DeltaNet layer runs its two tokens one at a time, with the same
single-token kernels plain decode uses, and keeps a copy of its state after
the first token; a reject restores that copy. An accepted draft keeps the copy
until its follower has been picked: if the request ends on the draft itself,
``retire`` restores it, so the session tail (``PrefixCache.store_tail``)
always covers exactly the prompt and the accepted tokens but the last.

The MTP head keeps its own one-layer KV cache per request. It sees the pair
``(H[i], x[i+1])`` at position ``i`` for every position whose target hidden
state ``H[i]`` the engine computed: the prefilled prompt (up to
``MAX_PENDING`` positions), then each fed token. Pairs wait in ``MTPState.hidden``
until the next draft needs them, and the state travels with a session tail, so
the next round's head sees the earlier rounds too.

``ENGINE_MTP=0`` turns it off; requests that sample (temperature > 0) always
decode plainly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

from engine import batching
from engine.mtp import MTPHead

if TYPE_CHECKING:
    from engine.generator import Generation, Generator

ENV = "ENGINE_MTP"
# At most this many prompt positions wait for the MTP head (bounds memory on long prompts).
MAX_PENDING = 16384
MTP_CHUNK = 2048


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "off", "no")


# One Gated DeltaNet layer's (conv state, recurrent state) after the first token of a verify.
Snapshot = dict[int, tuple[torch.Tensor, torch.Tensor]]


@dataclass
class MTPState:
    """One request's MTP head state.

    ``cache`` covers head positions below ``pos``; ``hidden`` holds the target's
    final normed hidden states of positions ``pos ..``, which wait for the head.
    ``queued`` is the logits after an accepted draft, not yet picked from;
    ``snapshot`` restores the request's cache to before that draft.
    """

    cache: DynamicCache
    pos: int
    hidden: torch.Tensor | None
    queued: torch.Tensor | None = None
    snapshot: Snapshot | None = field(default=None, repr=False)

    def pending(self) -> int:
        return 0 if self.hidden is None else self.hidden.shape[1]

    def push(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden if self.hidden is None else torch.cat([self.hidden, hidden], dim=1)
        excess = self.pending() - MAX_PENDING
        if excess > 0:  # the head skips the oldest positions; positions stay absolute
            self.hidden = self.hidden[:, excess:]
            self.pos += excess


def _rows(snapshot: Snapshot, row: int) -> Snapshot:
    return {i: (c[row : row + 1], r[row : row + 1]) for i, (c, r) in snapshot.items()}


def rollback(cache: DynamicCache, snapshot: Snapshot) -> None:
    """Undo the last fed token of ``cache``: restore Gated DeltaNet states, crop attention by one."""
    for index, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            conv, recurrent = snapshot[index]
            layer.conv_states[0] = conv
            layer.recurrent_states[0] = recurrent
        elif getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys[:, :, :-1]
            layer.values = layer.values[:, :, :-1]


class Speculator:
    """The MTP head of one Generator, its hooks into the target model, and its counters."""

    def __init__(self, generator: Generator, head: MTPHead) -> None:
        self.g = generator
        self.head = head
        model = generator.model
        self.embed = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        self.proposed = 0
        self.accepted = 0
        self.verify_steps = 0
        self._captured: list[torch.Tensor] | None = None
        self._snapshot: Snapshot | None = None
        text = model.model
        norm = getattr(text, "norm", None) or text.language_model.norm
        norm.register_forward_hook(self._on_norm)
        for module in model.modules():
            if isinstance(module, Qwen3_5MoeGatedDeltaNet):
                module.forward = self._split_forward(module, module.forward)

    @classmethod
    def load(cls, generator: Generator, model: str, revision: str | None) -> Speculator | None:
        """The head of ``model``, or None (with a message) when it is off or the checkpoint has none."""
        if not enabled():
            print(f"engine: MTP speculative decoding off ({ENV}=0)", file=sys.stderr, flush=True)
            return None
        config = generator.model.config
        text_config = getattr(config, "text_config", config)
        device = generator.model.get_output_embeddings().weight.device
        try:
            head = MTPHead.load(model, revision, text_config, device)
        except (OSError, ValueError, KeyError) as exc:
            print(f"engine: MTP speculative decoding off: {exc}", file=sys.stderr, flush=True)
            return None
        print(f"engine: MTP speculative decoding on (head on {device})", file=sys.stderr, flush=True)
        return cls(generator, head)

    # ---------------------------------------------------------------- hooks

    def _on_norm(self, _module: Any, _inputs: Any, output: torch.Tensor) -> None:
        if self._captured is not None:
            self._captured.append(output)

    def _split_forward(self, module: Qwen3_5MoeGatedDeltaNet, forward: Any) -> Any:
        """During a verify, feed the layer one token at a time and keep its state after the first."""

        def split(hidden_states: torch.Tensor, cache_params: Any = None, attention_mask: Any = None, **kwargs: Any):
            length = hidden_states.shape[1]
            if self._snapshot is None or cache_params is None or length == 1:
                return forward(
                    hidden_states=hidden_states,
                    cache_params=cache_params,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            outputs = []
            for t in range(length):
                # Fed tokens are never padding, so no mask: the single-token decode path.
                outputs.append(
                    forward(
                        hidden_states=hidden_states[:, t : t + 1],
                        cache_params=cache_params,
                        attention_mask=None,
                        **kwargs,
                    )
                )
                if t == 0:
                    layer = cache_params.layers[module.layer_idx]
                    self._snapshot[module.layer_idx] = (
                        layer.conv_states[0].clone(),
                        layer.recurrent_states[0].clone(),
                    )
            return torch.cat(outputs, dim=1)

        return split

    def capture(self) -> None:
        """Start keeping the target's final normed hidden states."""
        self._captured = []

    def captured(self) -> torch.Tensor | None:
        """The hidden states kept since ``capture``, ``[B, L, D]`` on the head's device; stop keeping."""
        parts, self._captured = self._captured, None
        if not parts:
            return None
        device = self.head.device
        return torch.cat([p.to(device) for p in parts], dim=1)

    # ---------------------------------------------------------------- requests

    def new_state(self, carried: MTPState | None, start: int, hidden: torch.Tensor | None) -> MTPState:
        """The head state of a request whose prompt was prefilled from position ``start``.

        ``carried`` (from a session tail) is kept when it ends exactly at ``start``.
        """
        if carried is not None and carried.pos + carried.pending() == start:
            state = carried
            state.queued, state.snapshot = None, None
        else:
            state = MTPState(self.head.new_cache(), start, None)
        if hidden is not None:
            state.push(hidden)
        return state

    def _draft(self, g: Generation) -> int:
        """Feed the request's waiting pairs to the head; its guess of the token after ``last_token``."""
        state = g.mtp
        tokens = g.tokens()
        fed = len(tokens) - 1  # the cache covers every token but the last picked one
        assert state.pos + state.pending() == fed, (state.pos, state.pending(), fed)
        logits = None
        while state.pending():
            n = min(state.pending(), MTP_CHUNK)
            hidden, state.hidden = state.hidden[:, :n], state.hidden[:, n:]
            ids = torch.tensor([tokens[state.pos + 1 : state.pos + 1 + n]])
            positions = torch.arange(state.pos, state.pos + n)[None]
            logits = self.head(hidden, ids, positions, state.cache, self.embed, self.lm_head)
            state.pos += n
        state.hidden = None
        if logits is None:  # nothing new since the last draft: cannot happen after a fed step
            raise RuntimeError("MTP head has no pending position to draft from")
        return int(torch.argmax(logits[0, -1]))

    def step(self, generations: list[Generation]) -> None:
        """One verify forward for greedy requests; sets each request's logits (and queues a second)."""
        g0 = self.g
        drafts = [self._draft(g) for g in generations]
        self.verify_steps += 1
        self._snapshot = {}
        self.capture()
        try:
            if len(generations) == 1:
                g = generations[0]
                ids = torch.tensor([[g.last_token, drafts[0]]], device=g0._device)
                out = g0.model(input_ids=ids, past_key_values=g.cache, use_cache=True, logits_to_keep=2)
                caches = [g.cache]
                lengths = None
            else:
                merged, lengths = batching.merge([g.cache for g in generations])
                ids = torch.tensor([[g.last_token, d] for g, d in zip(generations, drafts, strict=True)], device=g0._device)
                out = g0.model(
                    input_ids=ids,
                    attention_mask=_padding_mask(lengths, 2, g0._device),
                    position_ids=torch.tensor([[n, n + 1] for n in lengths], device=g0._device),
                    past_key_values=merged,
                    use_cache=True,
                    logits_to_keep=2,
                )
                caches = batching.split(merged, lengths)
            hidden = self.captured()
            snapshot = self._snapshot
        finally:
            self._snapshot = None
            self._captured = None
        logits = out.logits.float()
        for row, (g, draft, cache) in enumerate(zip(generations, drafts, caches, strict=True)):
            g.cache = cache
            g.logits = logits[row, 0]
            mine = _rows(snapshot, row) if lengths is not None else snapshot
            pick = g0.pick(g.logits, g.sampling, g.picked_count(), None)
            self.proposed += 1
            if pick == draft:
                self.accepted += 1
                g.mtp.queued = logits[row, 1]
                g.mtp.snapshot = mine
                g.mtp.push(hidden[row : row + 1, 0:2])
            else:
                rollback(g.cache, mine)
                g.mtp.push(hidden[row : row + 1, 0:1])

    @staticmethod
    def take_queued(g: Generation) -> bool:
        """Use an accepted draft's follower logits instead of a forward pass; True if there were some."""
        state = g.mtp
        if state is None or state.queued is None:
            return False
        g.logits, state.queued, state.snapshot = state.queued, None, None
        return True

    @staticmethod
    def settle(g: Generation) -> None:
        """Before a request's cache is kept: drop a fed draft that the request ended on."""
        state = g.mtp
        if state is None or state.queued is None:
            return
        rollback(g.cache, state.snapshot)
        state.hidden = state.hidden[:, :-1]
        state.queued, state.snapshot = None, None

    def stats(self) -> dict[str, int]:
        return {
            "mtp_verify_steps": self.verify_steps,
            "mtp_proposed": self.proposed,
            "mtp_accepted": self.accepted,
        }


def _padding_mask(lengths: list[int], new: int, device: torch.device) -> torch.Tensor:
    """Attention mask for ``new`` tokens per request on top of ``batching.merge``'s left-padded cache."""
    longest = max(lengths)
    mask = torch.zeros((len(lengths), longest + new), dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, longest - length :] = 1
    return mask.to(device)
