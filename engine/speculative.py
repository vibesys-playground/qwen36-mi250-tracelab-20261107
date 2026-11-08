"""Exact-greedy self-speculative decoding with the checkpoint's own MTP head, on the fast path.

Each decode step of the fast path (``engine.fast.model.FastModel.verify``) feeds
two rows per running request: ``last``, the token it just picked, and
``draft``, the MTP head's guess of the token after it. The target model returns
the logits after both, in one captured graph per batch bucket, and the head
then runs on both positions so the next step's draft is ready whatever happens.

A greedy request takes its next token ``g1`` from the logits after ``last``,
exactly as plain decode would (the argmax of the same logits). If ``g1`` equals
the draft, the logits after the draft are those of the true sequence and its
pick ``g2`` is the token after ``g1``: both are emitted from one step. Every
emitted token is the target model's own greedy pick; the head only proposes.

State. The verify runs each request's first token on its slot's current Gated
DeltaNet state bank and the draft on a copy in the other bank, so a reject
leaves the current bank exactly one token on, and an accept flips the bank
(``FastModel.flip``). The draft's attention KV entry lies past the slot's
length after a reject and is overwritten by the next step. A request that ends
on ``g1`` (its last token, or an end-of-sequence token) takes the reject path,
and one stopped by a stop string after ``g1`` is flipped back before its slot
becomes a session tail (``settle``), so a tail always covers exactly the prompt
and every emitted token but the last.

Which requests speculate: temperature 0 with no ``min_tokens`` masking of the
end-of-sequence tokens (``eligible``), where the pick is the plain argmax.
Others ride along in the same step and pick from the logits after their first
row as before; their drafts are never accepted.

``ENGINE_MTP=0`` turns it off (plain captured decode), as does a checkpoint
without ``mtp.*`` tensors.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import torch

from engine.protocol import Sampling

if TYPE_CHECKING:
    from engine.generator import Generation, Generator

ENV = "ENGINE_MTP"


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def load_head(lm: Any, model: str, revision: str | None) -> torch.nn.Module | None:
    """The checkpoint's MTP head on the lm_head's device, or None (with a message)."""
    if not enabled():
        print(f"engine: MTP speculative decoding off ({ENV}=0)", file=sys.stderr, flush=True)
        return None
    from engine.mtp import MTPHead

    config = lm.config
    text_config = getattr(config, "text_config", config)
    device = lm.get_output_embeddings().weight.device
    try:
        head = MTPHead.load(model, revision, text_config, device)
    except (OSError, ValueError, KeyError) as exc:
        print(f"engine: MTP speculative decoding off: {exc}", file=sys.stderr, flush=True)
        return None
    print(f"engine: MTP speculative decoding on (head on {device})", file=sys.stderr, flush=True)
    return head


def eligible(sampling: Sampling) -> bool:
    """Whether ``Generator.pick`` is the plain argmax for this request at every step."""
    return sampling.temperature == 0.0 and (sampling.ignore_eos or sampling.min_tokens == 0)


def verify_step(generator: Generator, generations: list[Generation]) -> None:
    """One ``FastModel.verify`` for every running request; queues each one's next tokens."""
    fast = generator.fast
    logits, g1, g2, d1, d2 = fast.verify(
        [g.last_token for g in generations],
        [g.draft for g in generations],
        [g.slot for g in generations],
        [g.length for g in generations],
    )
    generator.verify_steps += 1
    apply(generator, generations, g1, g2, d1, d2, logits)


def apply(
    generator: Generator,
    generations: list[Generation],
    g1: list[int],
    g2: list[int],
    d1: list[int],
    d2: list[int],
    logits: torch.Tensor | None = None,
) -> None:
    """Take one verify step's picks: queue each request's next tokens, accept or reject its
    draft (flipping the slot's state bank on accept), and set its next draft. ``logits``
    (after each first token) are needed only for requests that do not speculate."""
    fast = generator.fast
    for i, g in enumerate(generations):
        g.flipped = False
        if not g.speculates:
            if logits is None:
                raise RuntimeError("a request that does not speculate needs the step's logits")
            g.logits = logits[i].clone()
            g.length += 1
            g.draft = d1[i]
            continue
        first = g1[i]
        g.queue.append(first)
        if g.continues_after(first):
            generator.mtp_proposed += 1
            if first == g.draft:
                generator.mtp_accepted += 1
                fast.flip(g.slot)
                g.queue.append(g2[i])
                g.length += 2
                g.draft = d2[i]
                g.flipped = True
                continue
        g.length += 1
        g.draft = d1[i]


def settle(generator: Generator, g: Generation, slot: int | None) -> None:
    """Before a request's slot is kept: undo an accepted draft whose follower was not emitted."""
    if g.queue and g.flipped and slot is not None:
        generator.fast.flip(slot)
        g.length -= 1
    g.queue.clear()
