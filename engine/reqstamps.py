"""Per-request stage stamps along the TTFT path, off unless ``ENGINE_REQ_STAMPS=1`` (plan 46).

The router and each replica stamp a request with ``time.monotonic()`` (one
clock for every process on the host) at each stage, and report per-stage
spans (p50/p90/mean ms) on ``/health`` as ``req_stamps``. The router forwards
its stamp of sending the request in the ``X-Engine-Sent`` header, so the
replica also sees the hop.

Replica stages, in order: ``hop`` (router sent -> replica handler start),
``parse`` (body read + JSON + request parse), ``tokenize`` (in the handler on
the fast path), ``queue`` (job submitted -> the engine thread takes it),
``run`` (job start up to its prefill request: tokenizing on the engine thread
when not done in the handler), ``add`` (``PackedPrefill.add``: slot, prefix
lookup, restore/adopt; ``lookup`` is its prefix-cache lookup alone),
``drain`` (finishing an in-flight run-ahead step), ``wait_pass`` (from there
to the start of the request's first prefill pass: the rest of the loop, other
requests' passes), ``pass`` (first pass start -> last pass end), ``emit``
(pass end -> first chunk queued: the running requests' tokens are handled
first), ``write`` (first chunk queued -> written by the handler thread), and
``total`` (handler start -> first write). The router adds ``r_parse`` (body
read + JSON + route), ``r_upstream`` (send -> first byte from the replica),
``r_relay`` (first byte -> written to the client) and ``r_total``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

ENABLED = os.environ.get("ENGINE_REQ_STAMPS") == "1"
now = time.monotonic

# The engine thread's last prefix-cache lookup time (s), read right after ``PackedPrefill.add``.
last_lookup = 0.0

_lock = threading.Lock()
_spans: dict[str, list[float]] = {}
_LIMIT = 100_000

REPLICA_ORDER = (
    ("hop", "sent", "handler"),
    ("parse", "handler", "parsed"),
    ("tokenize", "parsed", "tokenized"),
    ("queue", "submitted", "picked"),
    ("run", "picked", "prefill_req"),
    ("add", "prefill_req", "added"),
    ("drain", "added", "drained"),
    ("wait_pass", "drained", "pass0"),
    ("pass", "pass0", "pass1"),
    ("emit", "pass1", "emitted"),
    ("write", "emitted", "written"),
    ("total", "handler", "written"),
)


def record(name: str, seconds: float) -> None:
    with _lock:
        values = _spans.setdefault(name, [])
        if len(values) < _LIMIT:
            values.append(seconds)


def record_request(stamps: dict[str, float]) -> None:
    """Record a replica request's spans (the pairs whose two stamps it has)."""
    for name, a, b in REPLICA_ORDER:
        if a in stamps and b in stamps:
            record(name, stamps[b] - stamps[a])
    for name in ("lookup",):
        if name in stamps:
            record(name, stamps[name])


def summary() -> dict[str, Any]:
    if not ENABLED:
        return {}
    with _lock:
        spans = {k: list(v) for k, v in _spans.items()}
    out: dict[str, Any] = {}
    for name, values in spans.items():
        if not values:
            continue
        values.sort()
        n = len(values)
        out[name] = {
            "n": n,
            "p50": round(1e3 * values[n // 2], 3),
            "p90": round(1e3 * values[min(n - 1, int(0.9 * n))], 3),
            "mean": round(1e3 * sum(values) / n, 3),
        }
    return {"req_stamps": out}
