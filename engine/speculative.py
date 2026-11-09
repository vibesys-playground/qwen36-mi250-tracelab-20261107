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
step for the whole batch, chosen from {1, 3, 4, 5, 7} up to ``ENGINE_SPEC_KMAX`` (default 7;
``ENGINE_SPEC_K`` fixes it) to maximize the
predicted tokens per millisecond: per request the expected tokens per step at its own
running per-draft acceptance (an exponentially decayed count of its accepted and rejected
drafts, starting from a prior), over the step cost of k at the step's slot bucket
(``ENGINE_SPEC_COST``, ms per step by (slots, k), measured on the hardware). Nothing but the
request's own measured acceptance feeds the choice.

Depth by slot count (plan 58): rows are cheaper per token at 1 slot than at 2, so the table is
keyed by the slot bucket. Between k = 1 and 3 the choice is plan 31's (one acceptance per
request). Deeper k (4, 5, 7; only for buckets the table has rows for) is taken over k = 3 only
when it predicts ``ENGINE_SPEC_DEEP_MARGIN`` (5%) more tokens per ms from each request's own
acceptance BY DEPTH (decayed counts of drafts checked and accepted at each chain position;
a position it has not reached yet is assumed no better than its worst observed position past
the first), with the expected tokens capped by what the request still has to generate. Text
whose acceptance falls along the chain stays at k <= 3.

Run-ahead (plan 42, ``ENGINE_RUNAHEAD=1``, the default; tensor-parallel fast path): the next
chained step is launched before the host reads the current one's picks, its inputs written on
the GPU from the current step's outputs (``TPModel.chain_launch_next``), so the GPU does not
idle while the host builds inputs and handles tokens. Only when that step surely runs for the
same requests with the same k: every request greedy with ``ignore_eos``, no stop strings, and
a token still needed after the current step's at most k + 1; no job, prefill or arrival
waiting (asked again right before the launch). The launch waits until
``ENGINE_RUNAHEAD_MARGIN_MS`` before the current step's learned end, so a request arriving
meanwhile is prefilled after the current step. A batch change finishes the step launched
ahead first (``drain``). The steps, their inputs and kernels are those of the serial order.
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


KS = (1, 3, 4, 5, 7)
KBASE = 3  # deeper k needs the per-depth case and the margin (plan 58)
PRIOR = (3.2, 4.0)  # (accepted, checked) pseudo-counts: a per-draft acceptance of 0.8
DECAY = 0.95
MARGIN = float(os.environ.get("ENGINE_SPEC_DEEP_MARGIN", "0.05"))
# Slot buckets that may run k > 3 (default: every bucket with its own cost row).
DEEP_SIZES = {int(x) for x in os.environ.get("ENGINE_SPEC_DEEP_SIZES", "").split(",") if x.strip()}
# ms per chained verify step by (slot bucket, k): the captured ``verify_chain`` step under TP2
# with HIP GDN and IVF drafts on ROCm 7.0, median of 200 (plan 58, j303, the afb6be0 merge;
# j268's ROCm 6.4 table moved by up to 12.5%). "k:ms" items are size 2. Plan 73: rows 3 and 4
# (k 1, 3) for the exact 3-slot graph and the 4-slot one, from j339 (same job as a re-measure of
# rows 1 and 2, which moved < 5%: chain_gate 3:1 9.26, 3:3 13.63; row 4 from plan73_stage0 in that
# job scaled by its 2:3 to chain_gate's); larger buckets use row 4. Before, 3- and 4-slot steps
# were priced with the 2-slot row (k=3 at 1.39x k=1; it is 1.47x at 3 slots, 1.53x at 4).
COST = ("1:1:7.07,1:3:8.86,1:4:10.29,1:5:11.62,1:7:14.74,"
        "2:1:8.28,2:3:11.54,2:4:13.42,2:5:15.26,2:7:19.46,"
        "3:1:9.26,3:3:13.63,4:1:10.58,4:3:16.15")
_COST_CACHE: dict[str, dict[tuple[int, int], float]] = {}


def _cost() -> dict[tuple[int, int], float]:
    """``ENGINE_SPEC_COST`` "size:k:ms,...": ms per step by (slot bucket, k) (default ``COST``);
    the older "k:ms" items mean size 2."""
    spec = os.environ.get("ENGINE_SPEC_COST", COST)
    table = _COST_CACHE.get(spec)
    if table is None:
        table = {}
        for item in spec.split(","):
            parts = item.split(":") if item else []
            if len(parts) == 2:
                table[(2, int(parts[0]))] = float(parts[1])
            elif len(parts) == 3:
                table[(int(parts[0]), int(parts[1]))] = float(parts[2])
        _COST_CACHE[spec] = table
    return table


def _row(size: int) -> int | None:
    """The table row (slot bucket) whose costs a step of bucket ``size`` uses: its own, else the
    largest one below it (else the smallest)."""
    sizes = sorted({s for s, _ in _cost()})
    if not sizes:
        return None
    below = [s for s in sizes if s <= size]
    return below[-1] if below else sizes[0]


def cost(size: int, k: int) -> float:
    row = _row(size)
    return _cost().get((row, k), 1.0) if row is not None else 1.0


def depths(size: int) -> list[int]:
    """The k a step of slot bucket ``size`` may run: up to ``ENGINE_SPEC_KMAX``, (k + 1) x size
    rows within the 64-row cap, and beyond k = 3 only where the table has that bucket's own
    measured cost. The graphs of exactly these (size, k) are captured at load."""
    from engine.fast import mfma

    own = {k for s, k in _cost() if s == size}
    deep_ok = not DEEP_SIZES or size in DEEP_SIZES
    return [k for k in KS if k <= mfma.KMAX and (k + 1) * size <= 64
            and (k <= KBASE or (k in own and deep_ok))]


def acceptance(g: Generation) -> float:
    """The request's running per-draft acceptance (its own counts and the prior)."""
    return (g.spec_accepted + PRIOR[0]) / (g.spec_checked + PRIOR[1])


def _depth_book(g: Generation, k: int, accepted: int) -> None:
    """Count one step's chain positions: 1 .. min(k, accepted + 1) checked, 1 .. accepted
    accepted; each position's counts decay when it is checked again."""
    chk = g.__dict__.setdefault("spec_depth_checked", [0.0] * max(KS))
    acc = g.__dict__.setdefault("spec_depth_accepted", [0.0] * max(KS))
    for j in range(min(k, accepted + 1)):
        chk[j] = chk[j] * DECAY + 1.0
        acc[j] = acc[j] * DECAY + (1.0 if j < accepted else 0.0)


def expected_tokens(g: Generation, k: int) -> float:
    """Tokens a step of k drafts is expected to emit for ``g`` from its own acceptance by depth:
    1 + sum over positions i <= k of P(the first i drafts accepted), positions it never reached
    taken at its worst observed conditional past the first (or the first's, or the prior), and
    no more positions than tokens it still needs."""
    chk = g.__dict__.get("spec_depth_checked") or [0.0] * max(KS)
    acc = g.__dict__.get("spec_depth_accepted") or [0.0] * max(KS)
    rate = [(acc[j] + PRIOR[0]) / (chk[j] + PRIOR[1]) for j in range(len(chk))]
    seen = [j for j in range(len(chk)) if chk[j] >= 1.0]
    deeper = [rate[j] for j in seen if j >= 1]
    floor = min(deeper) if deeper else (rate[0] if seen else PRIOR[0] / PRIOR[1])
    need = g._max_tokens - g._count - len(g.queue)
    total = reach = 1.0
    for j in range(min(k, max(0, need - 1))):
        reach *= rate[j] if chk[j] >= 1.0 else floor
        total += reach
    return total


def choose_k(generator: Generator, generations: list[Generation]) -> int:
    fast = generator.fast
    size = fast.bucket(len(generations)) if fast.use_graphs else len(generations)
    allowed = [k for k in depths(size) if (k + 1) * size <= min(64, fast._rows)] or [1]
    fixed = os.environ.get("ENGINE_SPEC_K", "")
    if fixed:
        k = int(fixed)
        return max([a for a in allowed if a <= k] or [allowed[0]])
    spec = [g for g in generations if g.speculates]
    rates = [acceptance(g) for g in spec]
    if not rates:
        return allowed[0]

    def gain(k: int) -> float:
        return sum(sum(p ** i for i in range(k + 1)) for p in rates) / cost(size, k)

    base = [k for k in allowed if k <= KBASE]
    k0 = max(base, key=gain) if base else allowed[0]
    deep = [k for k in allowed if k > KBASE]
    if k0 != KBASE or not deep:
        return k0

    def gain_d(k: int) -> float:
        return sum(expected_tokens(g, k) for g in spec) / cost(size, k)

    # A step of k can only hand the next step k real drafts (the head's chain is k long), so a
    # deeper k than the batch's last one runs its extra positions on padding once: going up
    # needs the margin, and a batch already running deep stays at its k unless another deep k
    # beats it by the margin or k <= 3 beats it outright (going down costs nothing).
    have = min(len(g.drafts) if g.drafts and g.drafts[0] == g.draft else 1 for g in spec)
    best = max(deep, key=gain_d)
    if have in deep:
        if gain_d(have) < gain_d(KBASE):
            return k0
        return best if gain_d(best) >= (1.0 + MARGIN) * gain_d(have) else have
    return best if gain_d(best) >= (1.0 + MARGIN) * gain_d(KBASE) else k0


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
        g.spec_real = min(k, len(ds))  # drafts past these are padding (plan 58)
        ds = (ds + [ds[-1]] * k)[:k]
        drafts.append(ds if g.speculates else [0] * k)
    return drafts


def _book(generator: Generator, generations: list[Generation], k: int, logits, picks, acc, nxt) -> None:
    """Queue each request's tokens from one chained verify step and advance its state ring."""
    fast = generator.fast
    generator.verify_steps += 1
    hist = generator.__dict__.setdefault("spec_k_hist", {})
    key = f"{len(generations)}:{k}"  # requests in the step : drafts per request
    hist[key] = hist.get(key, 0) + 1
    # plan 58 counters for /health: tokens emitted by (requests : k), and k changes (a step whose
    # k differs from the step before for the same requests)
    emitted = generator.__dict__.setdefault("spec_k_tokens", {})
    last = generator.__dict__.get("_spec_last")
    ids = tuple(id(g) for g in generations)
    if last is not None and last[0] == ids and last[1] != k:
        generator.spec_k_changes = getattr(generator, "spec_k_changes", 0) + 1
    generator._spec_last = (ids, k)
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
        emitted[key] = emitted.get(key, 0) + adv + 1
        g.advanced = adv
        g.length += adv + 1
        if adv == acc[i] and len(g.queue) == adv + 1 and g.continues_after(g.queue[-1], adv):
            checked = min(k, acc[i] + 1)
            generator.mtp_proposed += checked
            generator.mtp_accepted += acc[i]
            g.spec_checked = g.spec_checked * DECAY + checked
            g.spec_accepted = g.spec_accepted * DECAY + acc[i]
            # positions past the real drafts held padding: they say nothing about acceptance
            _depth_book(g, min(k, getattr(g, "spec_real", k)), acc[i])


MARGIN_S = float(os.environ.get("ENGINE_RUNAHEAD_MARGIN_MS", "1.5")) / 1e3
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
        for g in generations:
            g.spec_real = k  # built on the GPU from the step before's k drafts: all real
    else:
        k = choose_k(generator, generations)
        gather = not all(g.speculates for g in generations)
        handle = fast.chain_launch(
            [g.last_token for g in generations], _drafts(generations, k),
            [g.slot for g in generations], [g.length for g in generations], gather=gather,
        )
    nxt_handle = None
    counts = generator.__dict__.setdefault("spec_ahead", {})
    if (
        _allowed(ahead_ok)
        and fast.chain_ready(handle["size"], handle["groups"])
        and all(_ahead_safe(g, k) for g in generations)
    ):
        same_k = choose_k(generator, generations) == k
        why = "eligible_same_k" if same_k else "eligible_k_changed"
    else:
        same_k, why = False, "not_eligible"
    counts[why] = counts.get(why, 0) + 1
    if k > KBASE:
        counts[f"{why}:deep"] = counts.get(f"{why}:deep", 0) + 1
    if same_k:
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
