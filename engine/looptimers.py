"""Per-stage timers of the engine worker loop, off unless ``ENGINE_LOOP_TIMERS=1``.

A probe for plan 32 step 0: how much host work runs between two decode/verify
steps while the GPU waits, and what it is. Only the engine worker thread
writes; ``summary()`` (read by ``/health``) copies the records first.

One record per loop iteration that launches a decode step. Its stages, in
loop order (all host wall time, ms):

- ``poll``: taking new jobs off the queue (each runs up to its prefill request);
- ``prefill``: the packed prefill pass (GPU work, not idle);
- ``adv``: per-request work for the previous step's tokens (``_advance_all`` of
  the running jobs), split into ``detok`` (Detokenizer), ``stop`` (StopMatcher),
  ``chunk`` (response chunk dicts), ``emit`` (``events.put`` to the handler
  thread) and ``resume`` (the rest: generator resume, ``next_step``, usage);
- ``adv_new``: the same for requests whose prefill just finished;
- ``pre``: from there to ``Generator.decode``;
- ``build``: inside the verify call, before the graph replays (page tables,
  host input tensor, H2D copies);
- ``launch``: the graph replays; ``gather``: the logits gather (host);
- ``wait``: ``out.cpu()`` until the picks are on the host;
- ``book``: ``verify_step``'s bookkeeping after the picks.

``gap`` is host time from the previous step's picks on the host to this step's
first replay: the GPU idles in it unless ``prefill`` ran. ``gap_cpu`` is the
worker thread's CPU time over the same span, so ``gap - gap_cpu`` is time the
thread was runnable but not running (mostly waiting for the GIL held by HTTP
handler threads), which reordering cannot hide. With CUDA events: ``gpu_step``
(rank 0, input copy start to gather end), ``gpu_gather`` (gather on the GPU,
including any wait for rank 1's graph), ``gpu_idle`` (rank 0 stream from the
previous step's gather end to this step's input copy).
"""

from __future__ import annotations

import os
import time
from typing import Any

ENABLED = os.environ.get("ENGINE_LOOP_TIMERS") == "1"
now = time.perf_counter
cpu = time.thread_time

# The record being filled (worker thread only).
cur: dict[str, Any] = {}
_records: list[dict[str, Any]] = []
_LIMIT = 400_000
# Carried from the previous step.
_last: dict[str, Any] = {"picks": None, "picks_cpu": None, "end_event": None, "batch": None}

_STAGES = (
    "poll", "prefill", "adv", "detok", "stop", "chunk", "emit", "resume", "adv_new", "pre",
    "build", "launch", "gather", "wait", "book", "gap", "gap_cpu", "gap_offcpu",
    "gpu_step", "gpu_gather", "gpu_idle", "wait_excess", "hideable",
    "chost", "h2d", "parse", "choosek", "gpu_idle_launch", "gpu_graph",
)


def add(key: str, seconds: float) -> None:
    cur[key] = cur.get(key, 0.0) + seconds


def begin() -> None:
    """Start a loop iteration's record."""
    cur.clear()


def replay_start() -> None:
    """Called just before the first graph replay of a step."""
    t, c = now(), cpu()
    cur["_replay_t"] = t
    if _last["picks"] is not None:
        cur["gap"] = t - _last["picks"]
        cur["gap_cpu"] = c - _last["picks_cpu"]


def picks_ready(batch: int, start_event: Any = None, mid_event: Any = None, end_event: Any = None,
                launch_event: Any = None) -> None:
    """Called once a step's picks are on the host (all of its GPU work has finished)."""
    t, c = now(), cpu()
    if start_event is not None:
        cur["gpu_step"] = start_event.elapsed_time(end_event) / 1e3
        cur["gpu_gather"] = mid_event.elapsed_time(end_event) / 1e3
        prev = _last["end_event"]
        if prev is not None:
            cur["gpu_idle"] = prev.elapsed_time(start_event) / 1e3
            if launch_event is not None:
                # Rank 0's stream from the previous step's last op to this step's first replay
                # (the input build and its H2D copies included).
                cur["gpu_idle_launch"] = prev.elapsed_time(launch_event) / 1e3
        if launch_event is not None:
            cur["gpu_graph"] = launch_event.elapsed_time(mid_event) / 1e3
        if "_build_t" in cur:
            # Host span (first replay -> picks) beyond the GPU's own step time: launch latency,
            # the picks D2H and reacquiring the GIL after the sync.
            cur["wait_excess"] = max(0.0, (t - cur["_build_t"]) - cur["gpu_step"])
    _last["picks"], _last["picks_cpu"], _last["end_event"] = t, c, end_event
    cur["_batch"] = batch


def finish(prev_batch: int | None, arrivals: int, prefilled: bool, tokens: int) -> None:
    """Close the record of an iteration whose decode step returned."""
    if "_batch" not in cur:
        return
    rec = {k: v for k, v in cur.items() if not k.startswith("_")}
    rec["b"] = cur["_batch"]
    rec["k"] = cur.get("_k")
    rec["prev_b"] = prev_batch
    rec["arrivals"] = arrivals
    rec["prefill_ran"] = prefilled
    rec["tokens"] = tokens
    if len(_records) < _LIMIT:
        _records.append(rec)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _derive(rec: dict[str, Any]) -> dict[str, Any]:
    r = dict(rec)
    adv = r.get("adv", 0.0)
    r["resume"] = max(0.0, adv - sum(r.get(k, 0.0) for k in ("detok", "stop", "chunk", "emit")))
    if "gap" in r and "gap_cpu" in r:
        r["gap_offcpu"] = max(0.0, r["gap"] - r["gap_cpu"])
    # What launch-first could move behind the next step: the previous step's per-request
    # work plus the logits gather's GPU time (which delays the picks).
    r["hideable"] = adv + r.get("gpu_gather", 0.0)
    return r


def summary() -> dict[str, Any]:
    if not ENABLED:
        return {}
    records = [_derive(r) for r in _records[:]]
    out: dict[str, Any] = {"steps": len(records)}
    # Pure steady-state steps: no arrival, no prefill in the gap, same batch as the step whose
    # tokens were handled (so the gap is per-request work + launch only).
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        pure = not r["arrivals"] and not r["prefill_ran"] and r["prev_b"] == r["b"]
        key = f"{'pure' if pure else 'mixed'}_b{r['b']}" + (f"_k{r['k']}" if r.get("k") else "")
        groups.setdefault(key, []).append(r)
    for key, rs in sorted(groups.items()):
        entry: dict[str, Any] = {"n": len(rs), "tokens_p50": _pct([r["tokens"] for r in rs], 0.5)}
        for stage in _STAGES:
            values = [r[stage] for r in rs if stage in r]
            if len(values) >= max(1, len(rs) // 10):
                entry[stage] = [round(1e3 * _pct(values, 0.5), 3), round(1e3 * _pct(values, 0.9), 3),
                                round(1e3 * sum(values) / len(values), 3)]
        out[key] = entry
    totals = {s: round(sum(r.get(s, 0.0) for r in records), 3) for s in _STAGES}
    out["totals_s"] = totals
    return {"loop_timers": out}
