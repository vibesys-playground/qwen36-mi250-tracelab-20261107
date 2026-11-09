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

Chained drafts (plan 31, ``ENGINE_SPEC_CHAIN=1``, the default): each step feeds every
running request's last token and k drafts (``FastModel.verify_chain``), the MTP head applied
to its own output k times. A request emits the target's picks up to and including the first
one that differs from the draft fed after it, so every emitted token is again the target's
own greedy pick, and its slot's state ring advances by the drafts it accepted. k is one per
step for the whole batch, chosen from {1, 3, 7} up to ``ENGINE_SPEC_KMAX`` (default 3;
``ENGINE_SPEC_K`` fixes it) to maximize the
predicted tokens per millisecond: per request the expected tokens per step at its own
running per-draft acceptance (an exponentially decayed count of its accepted and rejected
drafts, starting from a prior), over the step cost of k (``ENGINE_SPEC_COST``, ms per
2-slot step). Nothing but the request's own measured acceptance feeds the choice.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Any

import torch

from engine import looptimers
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


def chain_enabled(generator: Generator) -> bool:
    fast = generator.fast
    return fast is not None and fast.mtp is not None and getattr(fast, "chain", False) and (
        fast.fused is not None
    )


KS = (1, 3, 7)
PRIOR = (3.2, 4.0)  # (accepted, checked) pseudo-counts: a per-draft acceptance of 0.8
DECAY = 0.95


def _cost() -> dict[int, float]:
    """``ENGINE_SPEC_COST`` "1:9.2,3:13.7,7:23.2": ms per step of 2 slots by k (default: the
    measured ``verify_chain`` step under TP2, plan 31 stage 1, j161)."""
    spec = os.environ.get("ENGINE_SPEC_COST", "1:9.2,3:13.7,7:23.2")
    return {int(k): float(v) for k, v in (item.split(":") for item in spec.split(",") if item)}


def acceptance(g: Generation) -> float:
    """The request's running per-draft acceptance (its own counts and the prior)."""
    return (g.spec_accepted + PRIOR[0]) / (g.spec_checked + PRIOR[1])


def choose_k(generator: Generator, generations: list[Generation]) -> int:
    from engine.fast import mfma

    fast = generator.fast
    size = fast.bucket(len(generations)) if fast.use_graphs else len(generations)
    allowed = [k for k in KS if k <= mfma.KMAX and (k + 1) * size <= min(64, fast._rows)] or [1]
    fixed = os.environ.get("ENGINE_SPEC_K", "")
    if fixed:
        k = int(fixed)
        return max([a for a in allowed if a <= k] or [allowed[0]])
    cost = _cost()
    rates = [acceptance(g) for g in generations if g.speculates]
    if not rates:
        return allowed[0]

    def gain(k: int) -> float:
        return sum(sum(p ** i for i in range(k + 1)) for p in rates) / cost.get(k, 1.0)

    return max(allowed, key=gain)


RUNAHEAD_ENV = "ENGINE_RUNAHEAD"


def runahead_enabled() -> bool:
    """Plan 42 (``ENGINE_RUNAHEAD``, default on): launch the next chained verify step, its
    inputs built on the GPU, before the host waits for the current one, when that step is
    certain to run for the same requests (``_ahead_safe``)."""
    return os.environ.get(RUNAHEAD_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def _ahead_safe(g: Generation, k: int) -> bool:
    """Whether ``g`` surely runs the step after the in-flight one, whatever that step picks:
    a greedy request that cannot stop on a token (``ignore_eos``, no stop strings) and still
    needs a token after the in-flight step's at most ``k + 1``."""
    sampling = g._sampling
    return (
        g.speculates and sampling.ignore_eos and not sampling.stop and g.slot is not None
        and not g.queue and g._count + k + 1 < g._max_tokens
    )


class _Ahead:
    """A chained verify step launched before the host saw the step before it."""

    def __init__(self, generations: list[Generation], k: int, handle: dict) -> None:
        self.generations = generations
        self.k = k
        self.handle = handle


def _drafts(generations: list[Generation], k: int) -> list[list[int]]:
    drafts = []
    for g in generations:
        ds = list(g.drafts) if g.drafts and g.drafts[0] == g.draft else [g.draft]
        ds = (ds + [ds[-1]] * k)[:k]
        drafts.append(ds if g.speculates else [0] * k)
    return drafts


def _book(generator: Generator, generations: list[Generation], k: int, logits, picks, acc, nxt) -> None:
    """Queue each request's tokens from one chained verify step and advance its state ring."""
    fast = generator.fast
    generator.verify_steps += 1
    for i, g in enumerate(generations):
        g.flipped = False
        g.advanced = 0
        g.drafts = nxt[i]
        g.draft = nxt[i][0]
        if not g.speculates:
            g.logits = logits[i].clone()
            g.length += 1
            continue
        adv = 0
        for j in range(k + 1):
            token = picks[i][j]
            g.queue.append(token)
            if j >= acc[i] or not g.continues_after(token, len(g.queue) - 1):
                break
            adv += 1
        if adv:
            fast.advance(g.slot, adv)
        g.advanced = adv
        g.length += adv + 1
        if adv == acc[i] and len(g.queue) == adv + 1 and g.continues_after(g.queue[-1], adv):
            checked = min(k, acc[i] + 1)
            generator.mtp_proposed += checked
            generator.mtp_accepted += acc[i]
            g.spec_checked = g.spec_checked * DECAY + checked
            g.spec_accepted = g.spec_accepted * DECAY + acc[i]


MARGIN_S = float(os.environ.get("ENGINE_RUNAHEAD_MARGIN_MS", "1.2")) / 1e3
EST_ALPHA = 0.2


def _allowed(ahead_ok: Any) -> bool:
    return bool(ahead_ok() if callable(ahead_ok) else ahead_ok)


def _wait_late(generator: Generator, handle: dict) -> None:
    """Hold the run-ahead launch until ``MARGIN_S`` before the in-flight step should end (its
    start plus the learned step time of its shape), so a request arriving meanwhile is
    prefilled after this step rather than after the next one; the margin covers the launch's
    host time, so the GPU still does not idle. Without an estimate yet: launch now."""
    est = getattr(generator, "_ahead_est", {}).get((handle["size"], handle["groups"]))
    start = handle.get("t_start")
    if est is None or start is None:
        return
    target = start + est - MARGIN_S
    while True:
        dt = target - time.perf_counter()
        if dt <= 0 or handle["event"].query():
            return
        time.sleep(min(dt, 0.001))


def _learn(generator: Generator, handle: dict) -> None:
    """Update the shape's step time from a step that ran right after the previous one."""
    if not handle.get("b2b") or "t_done" not in handle or handle.get("t_start") is None:
        return
    est = generator.__dict__.setdefault("_ahead_est", {})
    key = (handle["size"], handle["groups"])
    dur = handle["t_done"] - handle["t_start"]
    est[key] = dur if key not in est else est[key] + EST_ALPHA * (dur - est[key])


def drain(generator: Generator) -> None:
    """Finish a step launched ahead: wait for it and queue its tokens (before the batch
    changes: an arrival, a prefill, a request leaving)."""
    ahead = getattr(generator, "_ahead", None)
    if ahead is None:
        return
    generator._ahead = None
    _book(generator, ahead.generations, ahead.k, *generator.fast.chain_collect(ahead.handle))


def pending(generator: Generator) -> bool:
    return getattr(generator, "_ahead", None) is not None


def chain_step(generator: Generator, generations: list[Generation], ahead_ok: bool = False) -> None:
    """One chained verify step for every running request; queues its next tokens.

    With ``ENGINE_RUNAHEAD`` (and ``ahead_ok``: the server has no arrival or prefill
    waiting), the step after this one is launched before this one's picks are read, when it
    surely runs for the same requests with the same k; the next call then only collects it."""
    fast = generator.fast
    timed = looptimers.ENABLED
    if not runahead_enabled() or not hasattr(fast, "chain_launch"):
        _chain_step_serial(generator, generations)
        return
    ahead = getattr(generator, "_ahead", None)
    generator._ahead = None
    if timed:
        t0 = looptimers.now()
    if ahead is not None and ahead.generations != generations:
        # The batch changed under a step launched ahead (a request left): queue its tokens
        # for the requests still here, then step whoever has none.
        _book(generator, ahead.generations, ahead.k, *fast.chain_collect(ahead.handle))
        rest = [g for g in generations if not g.queue]
        if rest:
            chain_step(generator, rest, False)
        return
    if ahead is not None:
        k, handle = ahead.k, ahead.handle
    else:
        k = choose_k(generator, generations)
        gather = not all(g.speculates for g in generations)
        handle = fast.chain_launch(
            [g.last_token for g in generations], _drafts(generations, k),
            [g.slot for g in generations], [g.length for g in generations], gather=gather,
        )
    nxt_handle = None
    if (
        _allowed(ahead_ok)
        and fast.chain_ready(handle["size"], handle["groups"])
        and all(_ahead_safe(g, k) for g in generations)
        and choose_k(generator, generations) == k
    ):
        if timed:
            looptimers.add("choosek", looptimers.now() - t0)
        _wait_late(generator, handle)
        if timed:
            t0 = looptimers.now()
        if _allowed(ahead_ok):  # still nothing new to start after the wait
            for g in generations:
                # Pages for the next step's rows: at most k + 1 tokens on, k + 1 rows, k drafts.
                fast._ensure_pages(g.slot, g.length + 3 * (k + 1))
            nxt_handle = fast.chain_launch_next(handle)
            # Launched while this step still ran: the next one starts when this one ends.
            nxt_handle["b2b"] = not handle["event"].query()
    if timed:
        looptimers.add("choosek", looptimers.now() - t0)
    result = fast.chain_collect(handle)
    _learn(generator, handle)
    if nxt_handle is not None and nxt_handle.get("b2b"):
        nxt_handle["t_start"] = handle["t_done"]
    if timed:
        t0 = looptimers.now()
    _book(generator, generations, k, *result)
    if nxt_handle is not None:
        generator._ahead = _Ahead(list(generations), k, nxt_handle)
    if timed:
        looptimers.add("book", looptimers.now() - t0)


def _chain_step_serial(generator: Generator, generations: list[Generation]) -> None:
    """One ``FastModel.verify_chain`` for every running request (``ENGINE_RUNAHEAD=0``)."""
    fast = generator.fast
    timed = looptimers.ENABLED
    if timed:
        t0 = looptimers.now()
    k = choose_k(generator, generations)
    drafts = _drafts(generations, k)
    if timed:
        looptimers.add("choosek", looptimers.now() - t0)
    logits, picks, acc, nxt = fast.verify_chain(
        [g.last_token for g in generations], drafts, [g.slot for g in generations],
        [g.length for g in generations],
    )
    if timed:
        t0 = looptimers.now()
    _book(generator, generations, k, logits, picks, acc, nxt)
    if timed:
        looptimers.add("book", looptimers.now() - t0)


def verify_step(generator: Generator, generations: list[Generation], ahead_ok: bool = False) -> None:
    """One ``FastModel.verify`` for every running request; queues each one's next tokens."""
    if chain_enabled(generator):
        chain_step(generator, generations, ahead_ok)
        return
    fast = generator.fast
    logits, g1, g2, d1, d2 = fast.verify(
        [g.last_token for g in generations],
        [g.draft for g in generations],
        [g.slot for g in generations],
        [g.length for g in generations],
    )
    generator.verify_steps += 1
    timed = looptimers.ENABLED
    if timed:
        t0 = looptimers.now()
    for i, g in enumerate(generations):
        g.flipped = False
        if not g.speculates:
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
    if timed:
        looptimers.add("book", looptimers.now() - t0)


def settle(generator: Generator, g: Generation, slot: int | None) -> None:
    """Before a request's slot is kept: undo an accepted draft whose follower was not emitted."""
    if chain_enabled(generator):
        n = min(len(g.queue), g.advanced)
        if n and slot is not None:
            generator.fast.advance(slot, -n)
            g.length -= n
        g.advanced = 0
        g.queue.clear()
        return
    if g.queue and g.flipped and slot is not None:
        generator.fast.flip(slot)
        g.length -= 1
    g.queue.clear()
