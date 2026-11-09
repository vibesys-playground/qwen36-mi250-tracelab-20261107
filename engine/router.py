"""Data-parallel replicas of the engine behind a session-affinity router.

``python -m engine.server`` runs this front when the layout has more than one
replica (by default one replica per two visible GPUs; see ``resolve_layout``).
The front starts each replica as its own process, ``python -m engine.server``
again with ``--replicas 1``, an internal port and ``HIP_VISIBLE_DEVICES`` (or
``CUDA_VISIBLE_DEVICES``) narrowed to that replica's devices. Every replica
loads the full BF16 model on its own devices and serves the whole HTTP
contract; the front only chooses a replica per request and copies the
replica's response bytes back unchanged, so usage (prompt, cached-prompt and
output tokens) is exactly what the serving replica reports.

Routing keeps a session on one replica. The replay sends no session id, so a
session is inferred from prompts: a session's next prompt starts with its
previous prompt (the earlier prompt plus the model output plus new input).
``Affinity`` remembers every routed prompt and the replica that served it; a
request whose prompt starts with a remembered prompt goes to that prompt's
replica, choosing the longest such prompt. A prompt that extends none goes to
the least-loaded replica, counting in-flight requests and sessions that ended a
round moments ago and are likely to continue. A prompt is compared as token
ids, as text, or for chat as the sequence of its messages, so a chat session's
growing history matches message by message. The replica holding a session holds
its prefix-cache snapshots, so affinity is also what makes cached tokens real.

Live moves (plan 63, ``ENGINE_MIGRATE``, off by default). Pinning a session to
its first replica lets load drift apart as sessions end at uneven times: one
replica carries three live sessions while another idles. So when a round ends
on a replica that carries ``ENGINE_MIGRATE_GAP`` (2) or more load units than
the least-loaded other replica (load: requests in flight plus sessions that
ended a round within ``ENGINE_MIGRATE_IDLE_SECONDS``, 3 s, counted from the
front's own request accounting only), the session's whole saved state (the
session tail: its attention KV pages and every Gated DeltaNet layer's
recurrent and convolution state) moves to that replica during the session's
tool wait: the source exports it into a shared memory channel, the target
imports it, and only then does the session's home change and the source drop
its copy. The next round is served by exactly one replica, from complete
state: if it arrives while the copy is in flight it waits for the copy (at most
``ENGINE_MIGRATE_WAIT_MS``, 100 ms) and otherwise goes to the source, which
still holds the state; the target's copy is then dropped. A replica runs at
most one move out and one move in at a time. Cached tokens reported for a moved
session's next round are the imported state the target actually reuses.

Two-holder sessions (plan 70, ``ENGINE_TWO_HOLDER``). Each round's saved state is
also copied, during the tool wait that follows the round, to one buddy replica
(the session's other previous holder, else the least-loaded other replica), with
the same export/import as a live move but the source keeps its copy. Every
holder's copy is tagged with the round (the router's ticket) it covers, and a
copy counts only once its import is acknowledged, whole and kept. The next
round goes to whichever holder of the latest round's state has fewer requests
in flight (the replica that served the last round on a tie): a copy still in
flight, failed, or of an older round never qualifies, and the round never waits
for one. A round sent to the buddy carries the number of tokens the buddy's copy
covers; the buddy refuses the round (before any output) unless its prefix cache
really reuses that many, and the round then goes to the replica that served the
last round. So exactly one replica serves each round, from complete state, and
its reported cached tokens are its own. Nothing here looks at prompt text or
guesses that a session has ended: every round of every session is copied.

The front opens its port only after every replica answers ``/health``, and
exits (stopping the others) if any replica exits.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from engine import protocol

Key = tuple[Any, ...] | str

# Request headers the front forwards; hop-by-hop ones are its own business.
_FORWARD_HEADERS = ("Content-Type", "Accept", "Authorization")
# Response headers copied back from a replica.
_RETURN_HEADERS = ("Content-Type", "Cache-Control")


# ----------------------------------------------------------------------------- layout


def env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def parse_groups(text: str) -> list[tuple[int, ...]]:
    """``"0,1;2,3"`` -> ``[(0, 1), (2, 3)]``: each replica's device indices."""
    groups = []
    for part in text.split(";"):
        part = part.strip()
        if part:
            groups.append(tuple(int(d) for d in part.split(",") if d.strip()))
    if not groups or any(not g for g in groups):
        raise ValueError(f"replica device groups {text!r} must be like '0,1;2,3'")
    return groups


def resolve_layout(
    gpus: int,
    replicas: int | None = None,
    gpus_per_replica: int | None = None,
    groups: str | None = None,
) -> list[tuple[int, ...]]:
    """Each replica's device indices (within the visible devices); ``()`` means CPU.

    ``groups`` lists them outright. Otherwise replicas take consecutive
    devices, ``gpus_per_replica`` each (default 2, so 4 x 2 on 8 GCDs; GCDs of
    one MI250 card enumerate next to each other), and as many replicas as fit
    unless ``replicas`` is given. With no GPU every replica runs on the CPU and
    the default is one replica. One replica on all devices is the classic
    single-process engine.
    """
    if groups:
        layout = parse_groups(groups)
        used = [d for g in layout for d in g]
        if len(set(used)) != len(used) or any(d < 0 or d >= gpus for d in used):
            raise ValueError(f"replica devices {groups!r} do not fit {gpus} visible GPUs")
        return layout
    if gpus == 0:
        return [()] * max(replicas or 1, 1)
    if replicas is not None and replicas < 1:
        raise ValueError("replicas must be at least 1")
    if gpus_per_replica is None:
        gpus_per_replica = gpus // replicas if replicas else min(2, gpus)
    gpus_per_replica = max(1, min(gpus_per_replica, gpus))
    count = replicas or max(gpus // gpus_per_replica, 1)
    if count == 1:
        return [tuple(range(gpus))]
    if count * gpus_per_replica > gpus:
        raise ValueError(f"{count} replicas of {gpus_per_replica} GPUs exceed {gpus} visible GPUs")
    return [
        tuple(range(i * gpus_per_replica, (i + 1) * gpus_per_replica)) for i in range(count)
    ]


def visible_gpus() -> int:
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def device_env(env: dict[str, str], devices: tuple[int, ...], hip: bool) -> dict[str, str]:
    """``env`` with the visible devices narrowed to ``devices`` (indices into what ``env`` shows).

    A visibility list already set (by the container runtime or the operator)
    is sliced, so the replica sees exactly those of its devices.
    """
    out = dict(env)
    if not devices:
        return out
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        current = env.get(name, "").strip()
        if current:
            ids = [d.strip() for d in current.split(",") if d.strip()]
            out[name] = ",".join(ids[d] for d in devices)
            return out
    out["HIP_VISIBLE_DEVICES" if hip else "CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    return out


# ----------------------------------------------------------------------------- affinity


def affinity_key(path: str, body: object) -> Key | None:
    """What a session's next request starts with, in comparable units; None when unknown."""
    if not isinstance(body, dict):
        return None
    if path == "/v1/completions":
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str | list):
            prompt = prompt[0]
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return tuple(prompt)
        return None
    if path == "/v1/chat/completions":
        messages = body.get("messages")
        if not isinstance(messages, list):
            return None
        head = json.dumps(
            [body.get("tools"), body.get("chat_template_kwargs")], sort_keys=True, default=str
        )
        return (head, *(json.dumps(m, sort_keys=True, default=str) for m in messages))
    return None


def _units(key: Key) -> int:
    """Rough size of a key in tokens: text (and chat messages) at about four characters each."""
    if isinstance(key, str):
        return len(key) // 4
    if key and isinstance(key[0], str):
        return sum(len(part) for part in key) // 4
    return len(key)


def _extends(key: Key, earlier: Key) -> bool:
    if type(key) is not type(earlier) or len(earlier) > len(key):
        return False
    if isinstance(key, str):
        return key.startswith(earlier)
    return key[: len(earlier)] == earlier


@dataclass
class Move:
    """A session's state on its way from ``src`` to ``dst`` (``Affinity.finish``)."""

    ticket: int
    src: int
    dst: int
    started: float
    done: threading.Event = field(default_factory=threading.Event)
    abandoned: bool = False


@dataclass
class Sync:
    """A copy of round ``ticket``'s saved state from ``src`` (which keeps it) to ``dst``."""

    ticket: int
    src: int
    dst: int
    started: float


@dataclass
class Entry:
    """One routed prompt: the replica that served it and when its request ended.

    ``move`` is set while the session's state moves away after this round; ``arrived`` once
    it has (the next round then runs on the new ``replica``). ``moved`` marks a round served
    from state that had moved."""

    key: Key
    replica: int
    ended_at: float | None = None
    continued: bool = False
    move: Move | None = None
    arrived: bool = False
    moved: bool = False
    # Two-holder sessions (plan 70): replica -> the ticket of the round whose complete saved
    # state it acknowledged holding; the tokens that state covers; the copy in flight; the
    # holders of the previous round when this round was routed; and, for a round sent to a
    # buddy, the replica that served the previous round (its fallback) and the tokens the
    # buddy must reuse.
    ticket: int = -1
    holders: dict[int, int] = field(default_factory=dict)
    tail_tokens: int | None = None
    sync: Sync | None = None
    prev_holders: tuple[int, ...] = ()
    home: int | None = None
    require: int | None = None


@dataclass
class Affinity:
    """Which replica each remembered prompt went to, and the replicas' load. Thread-safe.

    ``min_tokens`` keeps a short prompt (a probe, a shared opening) from pulling
    unrelated sessions onto its replica: a remembered prompt shorter than that
    attracts only an identical prompt. ``idle_seconds`` is how long a session
    that just finished a round still counts as load on its replica.

    With ``migrate``, ``finish`` may start a live move (module docstring): ``gap``,
    ``move_idle_seconds`` and ``move_wait_seconds`` are its rule's parameters. The caller
    runs the move and reports it with ``commit`` or ``fail``, then ``release``.

    It also keeps, per replica, the seconds spent with each number of requests in flight
    (plan 62's histogram) and the moves' counts and times: ``stats()`` reports them.
    """

    replicas: int
    max_entries: int = 4096
    min_tokens: int = 64
    idle_seconds: float = 10.0
    migrate: bool = False
    gap: int = 2
    move_idle_seconds: float = 3.0
    move_wait_seconds: float = 0.1
    two_holder: bool = False
    prefer_buddy: bool = False  # a test knob: a tie goes to the buddy, not home
    _entries: OrderedDict[int, Entry] = field(default_factory=OrderedDict)
    _inflight: list[int] = field(default_factory=list)
    _new_sessions: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_id: int = 0
    _seconds_at: list[list[float]] = field(default_factory=list)
    _clock: float | None = None
    _sending: set[int] = field(default_factory=set)
    _receiving: set[int] = field(default_factory=set)
    _moves: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._inflight = [0] * self.replicas
        self._new_sessions = [0] * self.replicas
        self._seconds_at = [[] for _ in range(self.replicas)]
        self._moves = {
            "started": 0, "committed": 0, "failed": 0, "abandoned": 0, "waited": 0,
            "moved_rounds": 0, "move_ms": [], "export_ms": [], "import_ms": [], "wait_ms": [],
            "errors": [],
        }
        self._syncs: dict[str, Any] = {
            "started": 0, "committed": 0, "failed": 0, "skipped_busy": 0, "buddy_rounds": 0,
            "home_rounds": 0, "in_flight_at_arrival": 0, "stale_refused": 0,
            "late_commits": 0, "sync_ms": [], "export_ms": [], "import_ms": [],
            "home_minus_buddy_inflight": [], "errors": [],
        }

    def _advance(self, now: float) -> None:
        """Add the time since the last change to each replica's current in-flight level."""
        if self._clock is not None:
            dt = max(now - self._clock, 0.0)
            for r, level in enumerate(self._inflight):
                hist = self._seconds_at[r]
                if len(hist) <= level:
                    hist.extend([0.0] * (level + 1 - len(hist)))
                hist[level] += dt
        self._clock = now

    def stats(self, now: float | None = None) -> dict[str, Any]:
        """Per replica, seconds at each in-flight count; the share of time above two; moves."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._clock is not None:
                self._advance(now)
            hist = [[round(t, 3) for t in h] for h in self._seconds_at]
            total = sum(sum(h) for h in self._seconds_at)
            busy = sum(sum(h[1:]) for h in self._seconds_at)
            over = sum(sum(h[3:]) for h in self._seconds_at)
            moves = {
                k: ([round(x, 2) for x in v] if k.endswith("_ms") else list(v[-8:]))
                if isinstance(v, list) else v
                for k, v in self._moves.items()
            }
            return {
                "inflight_seconds": hist,
                "share_over_2": round(over / total, 4) if total else 0.0,
                "share_over_2_of_busy": round(over / busy, 4) if busy else 0.0,
                "max_inflight": [len(h) - 1 if h else 0 for h in self._seconds_at],
                "idle_seconds": self.idle_seconds,
                "migrate": self.migrate,
                "moves": moves,
                "two_holder": self.two_holder,
                "syncs": {
                    k: ([round(x, 2) for x in v] if k.endswith("_ms") else list(v[-8:])
                        if k == "errors" else list(v))
                    if isinstance(v, list) else v
                    for k, v in self._syncs.items()
                },
            }

    def _match(self, key: Key) -> Entry | None:
        best: Entry | None = None
        for entry in self._entries.values():
            if best is not None and len(entry.key) <= len(best.key):
                continue
            if _units(entry.key) < self.min_tokens and entry.key != key:
                continue
            if _extends(key, entry.key):
                best = entry
        return best

    def load(self, now: float | None = None) -> list[int]:
        """Per replica: in-flight requests plus sessions that ended a round within ``idle_seconds``."""
        now = time.monotonic() if now is None else now
        with self._lock:
            return self._load(now)

    def _load(self, now: float, window: float | None = None) -> list[int]:
        """In-flight requests plus sessions resting within ``window`` (default
        ``idle_seconds``); a session whose state is moving counts on its target."""
        window = self.idle_seconds if window is None else window
        load = list(self._inflight)
        for entry in self._entries.values():
            if (
                not entry.continued
                and entry.ended_at is not None
                and now - entry.ended_at < window
            ):
                load[entry.move.dst if entry.move is not None else entry.replica] += 1
        return load

    def route(self, key: Key | None, now: float | None = None) -> tuple[int, int]:
        """(replica, ticket) for a request; pass the ticket to ``finish`` when it ends.

        A request continuing a session whose state is moving waits for the move, at most
        ``move_wait_seconds``; a move not finished by then is abandoned and the request goes
        to the source, which still holds the state."""
        waited = None
        if key is not None and self.migrate:
            with self._lock:
                match = self._match(key)
                move = match.move if match is not None else None
            if move is not None:
                t0 = time.monotonic()
                move.done.wait(self.move_wait_seconds)
                waited = time.monotonic() - t0
                with self._lock:
                    if match.move is move:  # still in flight: serve at the source
                        move.abandoned = True
                        match.move = None
                        self._moves["abandoned"] += 1
        with self._lock:
            now = time.monotonic() if now is None else now
            self._advance(now)
            match = self._match(key) if key is not None else None
            moved = False
            home = None
            require = None
            prev_holders: tuple[int, ...] = ()
            if waited is not None:
                self._moves["waited"] += 1
                self._moves["wait_ms"].append(waited * 1e3)
            if match is not None and self.two_holder:
                replica, home, require, prev_holders = self._pick_holder(match)
                match.continued = True
                moved = require is not None
            elif match is not None:
                replica = match.replica
                match.continued = True
                moved = match.arrived
                if moved:
                    self._moves["moved_rounds"] += 1
            else:
                load = self._load(now)
                replica = min(
                    range(self.replicas), key=lambda r: (load[r], self._new_sessions[r], r)
                )
                self._new_sessions[replica] += 1
            self._inflight[replica] += 1
            ticket = self._next_id
            self._next_id += 1
            if key is not None:
                self._entries[ticket] = Entry(
                    key, replica, moved=moved, ticket=ticket, prev_holders=prev_holders,
                    home=home if moved else None, require=require,
                )
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            return replica, ticket

    def _pick_holder(self, match: Entry) -> tuple[int, int, int | None, tuple[int, ...]]:
        """(replica, home, require, holders) for the next round of ``match``'s session.

        ``home`` served ``match``. The holders are the replicas whose acknowledged copy is of
        ``match``'s own round (version check: an older round's copy, or one still being
        copied, does not count). The round goes to the holder with the fewest requests in
        flight, ``home`` on a tie; ``require`` is the tokens a buddy must reuse (None at home).
        """
        home = match.replica
        if match.ended_at is None:  # the previous round has not ended: only home has it
            return home, home, None, (home,)
        holders = tuple(sorted(
            r for r, version in match.holders.items() if version == match.ticket
        )) or (home,)
        if match.sync is not None:
            self._syncs["in_flight_at_arrival"] += 1
        best = home
        if match.tail_tokens:
            for r in holders:
                if self._inflight[r] < self._inflight[best] or (
                    self.prefer_buddy and best == home and r != home
                    and self._inflight[r] <= self._inflight[home]
                ):
                    best = r
        if best == home:
            self._syncs["home_rounds"] += 1
            return home, home, None, holders
        self._syncs["buddy_rounds"] += 1
        self._syncs["home_minus_buddy_inflight"].append(self._inflight[home] - self._inflight[best])
        return best, home, match.tail_tokens, holders

    def required(self, ticket: int) -> int | None:
        """Tokens of saved state ``ticket``'s replica must reuse (a round sent to a buddy)."""
        with self._lock:
            entry = self._entries.get(ticket)
            return entry.require if entry is not None else None

    def reroute(self, ticket: int) -> int:
        """The buddy refused ``ticket``'s round (its copy was not there whole): send it to the
        replica that served the previous round instead. Returns that replica."""
        with self._lock:
            entry = self._entries[ticket]
            self._advance(time.monotonic())
            self._inflight[entry.replica] -= 1
            entry.replica = entry.home if entry.home is not None else entry.replica
            self._inflight[entry.replica] += 1
            entry.moved = False
            entry.require = None
            self._syncs["stale_refused"] += 1
            self._syncs["buddy_rounds"] -= 1
            self._syncs["home_rounds"] += 1
            return entry.replica

    def moved(self, ticket: int) -> bool:
        """Whether ``ticket``'s round runs on state that moved to its replica."""
        with self._lock:
            entry = self._entries.get(ticket)
            return entry is not None and entry.moved

    def finish(self, ticket: int, replica: int, now: float | None = None) -> Move | None:
        """``ticket``'s request has ended. Returns a move of its session's state to start
        now (the caller runs it), when ``migrate`` is on and the rule holds: the session's
        replica carries at least ``gap`` more load than the least-loaded other replica, not
        counting this session, and neither replica has a move in its direction running."""
        with self._lock:
            now = time.monotonic() if now is None else now
            self._advance(now)
            self._inflight[replica] -= 1
            entry = self._entries.get(ticket)
            if entry is None:
                return None
            entry.ended_at = now
            if self.two_holder:
                return self._start_sync(entry, replica, now)
            if (
                not self.migrate
                or entry.continued
                or entry.replica != replica
                or _units(entry.key) < self.min_tokens
                or replica in self._sending
            ):
                return None
            load = self._load(now, self.move_idle_seconds)
            load[replica] -= 1  # this session, resting here: it is what would move
            others = [r for r in range(self.replicas) if r != replica and r not in self._receiving]
            if not others:
                return None
            target = min(others, key=lambda r: (load[r], self._inflight[r], r))
            if load[replica] - load[target] < self.gap:
                return None
            move = Move(ticket, replica, target, now)
            entry.move = move
            self._sending.add(replica)
            self._receiving.add(target)
            self._moves["started"] += 1
            return move

    def _start_sync(self, entry: Entry, replica: int, now: float) -> Sync | None:
        """Two-holder: ``entry``'s round ended on ``replica``, which now holds the session's
        latest state (and no one else does). Returns the copy to a buddy to start now, if
        the session's prompt is long enough to be one and both ends are free."""
        if entry.replica != replica:
            return None
        entry.holders = {replica: entry.ticket}
        if _units(entry.key) < self.min_tokens:
            return None
        if replica in self._sending:
            self._syncs["skipped_busy"] += 1
            return None
        free = [r for r in range(self.replicas) if r != replica and r not in self._receiving]
        if not free:
            self._syncs["skipped_busy"] += 1
            return None
        again = [r for r in entry.prev_holders if r in free]
        if again:
            target = again[0]
        else:
            load = self._load(now)
            target = min(free, key=lambda r: (load[r], self._inflight[r], r))
        sync = Sync(entry.ticket, replica, target, now)
        entry.sync = sync
        self._sending.add(replica)
        self._receiving.add(target)
        self._syncs["started"] += 1
        return sync

    def sync_done(self, sync: Sync, tokens: int, export_ms: float, import_ms: float) -> bool:
        """``sync.dst`` acknowledged holding the whole state of round ``sync.ticket``
        (``tokens`` of it): it is now a holder of that round's version."""
        with self._lock:
            entry = self._entries.get(sync.ticket)
            if entry is None or entry.sync is not sync:
                self._syncs["late_commits"] += 1
                return False
            entry.sync = None
            if entry.holders.get(sync.src) != sync.ticket or (
                entry.tail_tokens is not None and entry.tail_tokens != tokens
            ):
                self._syncs["failed"] += 1
                self._syncs["errors"].append("version or length mismatch")
                return False
            entry.holders[sync.dst] = sync.ticket
            entry.tail_tokens = tokens
            self._syncs["committed"] += 1
            self._syncs["sync_ms"].append((time.monotonic() - sync.started) * 1e3)
            self._syncs["export_ms"].append(export_ms)
            self._syncs["import_ms"].append(import_ms)
            return True

    def sync_failed(self, sync: Sync, error: str) -> None:
        with self._lock:
            entry = self._entries.get(sync.ticket)
            if entry is not None and entry.sync is sync:
                entry.sync = None
            self._syncs["failed"] += 1
            self._syncs["errors"].append(error[:200])

    def commit(self, move: Move, export_ms: float, import_ms: float) -> bool:
        """The target holds the state: the session's home is now ``move.dst``. False if the
        move was abandoned (its next round went to the source) or forgotten."""
        with self._lock:
            entry = self._entries.get(move.ticket)
            ok = not move.abandoned and entry is not None and entry.move is move
            if ok:
                entry.replica = move.dst
                entry.move = None
                entry.arrived = True
                self._moves["committed"] += 1
                self._moves["move_ms"].append((time.monotonic() - move.started) * 1e3)
                self._moves["export_ms"].append(export_ms)
                self._moves["import_ms"].append(import_ms)
            move.done.set()
            return ok

    def fail(self, move: Move, error: str) -> None:
        """The move did not happen; the session stays where it was."""
        with self._lock:
            entry = self._entries.get(move.ticket)
            if entry is not None and entry.move is move:
                entry.move = None
            self._moves["failed"] += 1
            self._moves["errors"].append(error[:200])
            move.done.set()

    def release(self, move: Move) -> None:
        """The move's replicas may take part in another move."""
        with self._lock:
            self._sending.discard(move.src)
            self._receiving.discard(move.dst)


# ----------------------------------------------------------------------------- replicas


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _die_with_parent() -> None:
    """Child side of ``Popen``: get SIGTERM if the front dies, so no replica outlives it."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


@dataclass
class Replica:
    index: int
    devices: tuple[int, ...]
    port: int
    process: subprocess.Popen

    def connection(self, timeout: float | None = None) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def get_json(self, path: str, timeout: float = 10.0) -> dict[str, Any]:
        conn = self.connection(timeout)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                raise OSError(f"replica {self.index} {path}: HTTP {response.status}")
            return json.loads(body)
        finally:
            conn.close()

    def post_json(self, path: str, body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        conn = self.connection(timeout)
        try:
            payload = json.dumps(body).encode()
            conn.request("POST", path, body=payload, headers={"Content-Type": "application/json",
                                                              "Content-Length": str(len(payload))})
            response = conn.getresponse()
            data = response.read()
            if response.status != 200:
                raise OSError(f"replica {self.index} {path}: HTTP {response.status} {data[:300]!r}")
            return json.loads(data)
        finally:
            conn.close()


def migrate_enabled() -> bool:
    """``ENGINE_MIGRATE``: live moves of a session's state between replicas (default off)."""
    return os.environ.get("ENGINE_MIGRATE", "0").strip() not in ("", "0")


def two_holder_enabled() -> bool:
    """``ENGINE_TWO_HOLDER``: keep each session's state on two replicas (plan 70)."""
    return os.environ.get("ENGINE_TWO_HOLDER", "0").strip() not in ("", "0")


def _memfd(name: str) -> int:
    """An anonymous shared memory file (``os.memfd_create``, or libc's where Python lacks it)."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, 0)
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.memfd_create(name.encode(), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    return fd


def move_channels(count: int) -> list[int]:
    """One shared memory file per replica for its outgoing moves (``engine.migrate.HostChannel``),
    ``ENGINE_MIGRATE_CHANNEL_MIB`` each (512: a tail of about 16k tokens on a TP2 replica; a
    longer one does not move); the replicas inherit the descriptors."""
    nbytes = int(float(os.environ.get("ENGINE_MIGRATE_CHANNEL_MIB") or 512) * (1 << 20))
    fds = []
    for index in range(count):
        fd = _memfd(f"engine-move-{index}")
        os.ftruncate(fd, nbytes)
        fds.append(fd)
    return fds


def start_replicas(
    argv: Sequence[str], layout: list[tuple[int, ...]], hip: bool, channels: Sequence[int] = ()
) -> list[Replica]:
    """Launch one ``engine.server`` process per device group, all loading at once, each on its
    own slice of host cores (``engine.cpu_place``) when the topology allows. ``channels`` are
    the live-move channels' descriptors, passed to every replica."""
    from engine import cpu_place

    placed = cpu_place.replica_slices(layout) if layout and any(layout) else None
    replicas = []
    for index, devices in enumerate(layout):
        port = _free_port()
        command = [
            sys.executable,
            "-m",
            "engine.server",
            *argv,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--replicas",
            "1",
            "--replica-devices",
            "",
        ]
        env = device_env(dict(os.environ), devices, hip)
        env["ENGINE_REPLICA_INDEX"] = str(index)
        env.pop("ENGINE_MIGRATE_CHANNELS", None)
        if channels:
            env["ENGINE_MIGRATE_CHANNELS"] = ",".join(map(str, channels))
        preexec = _die_with_parent
        if placed is not None:
            piece = placed[0][index]
            env = cpu_place.child_env(env, piece)
            preexec = _chain(cpu_place.preexec(piece), _die_with_parent)
        else:
            env.pop("ENGINE_CPUS", None)
            env.pop("ENGINE_RANK_CPUS", None)
        where = ",".join(map(str, devices)) or "cpu"
        print(f"router: starting replica {index} on devices {where}, port {port}", flush=True)
        process = subprocess.Popen(  # noqa: PLW1509
            command, env=env, preexec_fn=preexec, pass_fds=tuple(channels)
        )
        replicas.append(Replica(index, devices, port, process))
    if placed is not None:
        cpu_place.pin_router(placed[1])
    return replicas


def _chain(*fns):
    def run() -> None:
        for fn in fns:
            fn()

    return run


def stop_replicas(replicas: list[Replica]) -> None:
    for replica in replicas:
        if replica.process.poll() is None:
            replica.process.terminate()
    deadline = time.monotonic() + 30
    for replica in replicas:
        try:
            replica.process.wait(timeout=max(deadline - time.monotonic(), 0.1))
        except subprocess.TimeoutExpired:
            replica.process.kill()


def wait_ready(replicas: list[Replica]) -> None:
    """Block until every replica answers ``/health``; raise if one exits first."""
    started = time.monotonic()
    pending = set(range(len(replicas)))
    while pending:
        for index in sorted(pending):
            replica = replicas[index]
            code = replica.process.poll()
            if code is not None:
                raise RuntimeError(f"replica {index} exited with {code} while loading")
            try:
                replica.get_json("/health", timeout=2)
            except (OSError, ValueError):
                continue
            pending.discard(index)
            print(
                f"router: replica {index} ready after {time.monotonic() - started:.0f} s",
                flush=True,
            )
        if pending:
            time.sleep(0.5)


# ----------------------------------------------------------------------------- front


class _Front(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], replicas: list[Replica], affinity: Affinity
    ) -> None:
        super().__init__(address, _FrontHandler)
        self.replicas = replicas
        self.affinity = affinity


class _FrontHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Front

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        replicas = self.server.replicas
        if self.path == "/health":
            try:
                states = [r.get_json("/health") for r in replicas]
            except (OSError, ValueError) as exc:
                self._send_json(503, {"status": "unavailable", "error": str(exc)})
                return
            counters = (
                "decode_steps",
                "decoded_tokens",
                "prefill_passes",
                "prefill_segments",
                "mtp_verify_steps",
                "mtp_proposed",
                "mtp_accepted",
                "prefilled_tokens",
                "cached_tokens",
                "prefix_cache_tail_hits",
                "prefix_cache_entries",
                "prefix_cache_bytes",
                "prefix_cache_evictions",
            )
            body: dict[str, Any] = {"status": "ok"}
            body.update({k: sum(s.get(k, 0) for s in states) for k in counters})
            body["largest_batch"] = max(s.get("largest_batch", 0) for s in states)
            body["replicas"] = [
                {"devices": list(r.devices), **s} for r, s in zip(replicas, states, strict=True)
            ]
            body["load"] = self.server.affinity.load()
            body["router"] = self.server.affinity.stats()
            self._send_json(200, body)
            return
        try:
            body = replicas[0].get_json(self.path)
        except (OSError, ValueError):
            self._send_json(404, protocol.error_body(f"no route {self.path}", 404))
            return
        self._send_json(200, body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"null")
        except json.JSONDecodeError:
            body = None  # the replica answers with its own error
        key = affinity_key(self.path, body)
        affinity = self.server.affinity
        replica_index, ticket = affinity.route(key)
        require = affinity.required(ticket)
        try:
            served = self._relay(self.server.replicas[replica_index], raw, ticket,
                                 affinity.moved(ticket), require)
            if not served:  # the buddy's copy was not there whole: serve where the last round ran
                replica_index = affinity.reroute(ticket)
                self._relay(self.server.replicas[replica_index], raw, ticket)
        finally:
            job = affinity.finish(ticket, replica_index)
            if job is not None:
                threading.Thread(
                    target=run_sync if isinstance(job, Sync) else run_move,
                    args=(affinity, self.server.replicas, job),
                    name="session-sync" if isinstance(job, Sync) else "session-move",
                    daemon=True,
                ).start()

    def _relay(
        self, replica: Replica, raw: bytes, ticket: int = -1, moved: bool = False,
        require: int | None = None,
    ) -> bool:
        """Send the request to ``replica`` and copy its response back as it arrives. With
        ``require`` (a round sent to a buddy), a refusal because the replica would not reuse
        that many tokens of saved state is not passed on: returns False, nothing sent."""
        conn = replica.connection()
        try:
            headers = {h: self.headers[h] for h in _FORWARD_HEADERS if self.headers.get(h)}
            headers["Content-Length"] = str(len(raw))
            if ticket >= 0:
                headers["X-Engine-Ticket"] = str(ticket)  # names the session tail it keeps
            if require is not None:
                headers["X-Engine-Require-Cached"] = str(require)
            try:
                conn.request("POST", self.path, body=raw, headers=headers)
                response = conn.getresponse()
            except OSError as exc:
                message = f"replica {replica.index} unavailable: {exc}"
                self._send_json(503, protocol.error_body(message, 503))
                return True
            if require is not None and response.getheader("X-Engine-Stale") == "1":
                response.read()
                return False
            self.send_response(response.status)
            for name in _RETURN_HEADERS:
                value = response.getheader(name)
                if value is not None:
                    self.send_header(name, value)
            self.send_header("X-Engine-Replica", str(replica.index))
            if moved:
                self.send_header("X-Engine-Moved", "1")
            size = response.getheader("Content-Length")
            if size is not None:
                self.send_header("Content-Length", size)
                self.end_headers()
                self.wfile.write(response.read())
                return True
            # A stream: no length, ends when the replica closes it. Pass each piece on at once.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while piece := response.read1(65536):
                    self.wfile.write(piece)
                    self.wfile.flush()
            except OSError:  # the client went away; closing our side cancels the replica's job
                pass
            return True
        finally:
            conn.close()


def run_move(affinity: Affinity, replicas: list[Replica], move: Move) -> None:
    """Move a resting session's state from ``move.src`` to ``move.dst`` (on its own thread):
    export into the source's channel, import on the target, then switch the session's home
    and drop the source's copy; if the session's next round went to the source meanwhile,
    drop the target's copy instead. On any error the session stays at the source."""
    src, dst = replicas[move.src], replicas[move.dst]
    key = None
    committed = None
    try:
        t0 = time.monotonic()
        exported = src.post_json("/internal/export", {"ticket": move.ticket})
        key = exported["key"]
        t1 = time.monotonic()
        if move.abandoned:
            raise RuntimeError("abandoned before the import")
        result = dst.post_json("/internal/import", {"source": move.src, "key": key})
        if not result.get("kept"):
            raise RuntimeError("the target did not keep the state")
        t2 = time.monotonic()
        committed = affinity.commit(move, (t1 - t0) * 1e3, (t2 - t1) * 1e3)
        if committed:
            src.post_json("/internal/drop", {"key": key})
        else:
            dst.post_json("/internal/drop", {"key": key})
    except Exception as exc:  # noqa: BLE001 - the session keeps its replica
        if committed is None:
            affinity.fail(move, f"{type(exc).__name__}: {exc}")
            if key is not None:
                try:  # an import that failed late may still have landed; the source has it all
                    dst.post_json("/internal/drop", {"key": key})
                except Exception:  # noqa: BLE001
                    pass
    finally:
        affinity.release(move)


def run_sync(affinity: Affinity, replicas: list[Replica], sync: Sync) -> None:
    """Copy round ``sync.ticket``'s saved state from ``sync.src`` (which keeps it) to
    ``sync.dst`` (on its own thread, during the session's tool wait); the buddy becomes a
    holder of that round only once it acknowledges keeping the whole state."""
    src, dst = replicas[sync.src], replicas[sync.dst]
    try:
        t0 = time.monotonic()
        exported = src.post_json("/internal/export", {"ticket": sync.ticket})
        t1 = time.monotonic()
        result = dst.post_json("/internal/import", {"source": sync.src, "key": exported["key"]})
        t2 = time.monotonic()
        if not result.get("kept") or int(result.get("tokens", -1)) != int(exported["tokens"]):
            raise RuntimeError(f"the buddy did not keep the whole state: {result}")
        affinity.sync_done(sync, int(exported["tokens"]), (t1 - t0) * 1e3, (t2 - t1) * 1e3)
    except Exception as exc:  # noqa: BLE001 - the session keeps its other holder
        affinity.sync_failed(sync, f"{type(exc).__name__}: {exc}")
    finally:
        affinity.release(sync)


def serve(
    argv: Sequence[str],
    host: str,
    port: int,
    layout: list[tuple[int, ...]],
    hip: bool,
) -> None:
    """Start the replicas, wait for all of them, then route requests until stopped."""
    two_holder = two_holder_enabled() and len(layout) > 1
    migrate = migrate_enabled() and len(layout) > 1 and not two_holder
    channels = move_channels(len(layout)) if migrate or two_holder else []
    replicas = start_replicas(argv, layout, hip, channels)

    def stop(signum: int, frame: object) -> None:
        stop_replicas(replicas)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        wait_ready(replicas)
    except BaseException:
        stop_replicas(replicas)
        raise
    affinity = Affinity(
        len(replicas),
        min_tokens=env_int("ENGINE_AFFINITY_MIN_TOKENS") or 64,
        idle_seconds=float(os.environ.get("ENGINE_SESSION_IDLE_SECONDS") or 10.0),
        migrate=migrate,
        gap=int(os.environ.get("ENGINE_MIGRATE_GAP") or 2),
        move_idle_seconds=float(os.environ.get("ENGINE_MIGRATE_IDLE_SECONDS") or 3.0),
        move_wait_seconds=float(os.environ.get("ENGINE_MIGRATE_WAIT_MS") or 100) / 1e3,
        two_holder=two_holder,
        prefer_buddy=os.environ.get("ENGINE_TWO_HOLDER_PREFER_BUDDY", "0").strip() == "1",
    )
    front = _Front((host, port), replicas, affinity)

    def watch() -> None:
        while True:
            for replica in replicas:
                code = replica.process.poll()
                if code is not None:
                    print(f"router: replica {replica.index} exited with {code}", flush=True)
                    stop_replicas(replicas)
                    os._exit(1)
            time.sleep(1)

    threading.Thread(target=watch, name="replica-watch", daemon=True).start()
    print(
        f"router: serving {len(replicas)} replicas on http://{host}:{port}",
        flush=True,
    )
    front.serve_forever()
