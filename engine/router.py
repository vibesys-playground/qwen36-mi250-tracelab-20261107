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
class Entry:
    """One routed prompt: the replica that served it and when its request ended."""

    key: Key
    replica: int
    ended_at: float | None = None
    continued: bool = False


@dataclass
class Affinity:
    """Which replica each remembered prompt went to, and the replicas' load. Thread-safe.

    ``min_tokens`` keeps a short prompt (a probe, a shared opening) from pulling
    unrelated sessions onto its replica: a remembered prompt shorter than that
    attracts only an identical prompt. ``idle_seconds`` is how long a session
    that just finished a round still counts as load on its replica.
    """

    replicas: int
    max_entries: int = 4096
    min_tokens: int = 64
    idle_seconds: float = 10.0
    _entries: OrderedDict[int, Entry] = field(default_factory=OrderedDict)
    _inflight: list[int] = field(default_factory=list)
    _new_sessions: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_id: int = 0

    def __post_init__(self) -> None:
        self._inflight = [0] * self.replicas
        self._new_sessions = [0] * self.replicas

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

    def _load(self, now: float) -> list[int]:
        load = list(self._inflight)
        for entry in self._entries.values():
            if (
                not entry.continued
                and entry.ended_at is not None
                and now - entry.ended_at < self.idle_seconds
            ):
                load[entry.replica] += 1
        return load

    def route(self, key: Key | None) -> tuple[int, int]:
        """(replica, ticket) for a request; pass the ticket to ``finish`` when it ends."""
        with self._lock:
            now = time.monotonic()
            match = self._match(key) if key is not None else None
            if match is not None:
                replica = match.replica
                match.continued = True
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
                self._entries[ticket] = Entry(key, replica)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            return replica, ticket

    def finish(self, ticket: int, replica: int) -> None:
        with self._lock:
            self._inflight[replica] -= 1
            entry = self._entries.get(ticket)
            if entry is not None:
                entry.ended_at = time.monotonic()


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


def start_replicas(
    argv: Sequence[str], layout: list[tuple[int, ...]], hip: bool
) -> list[Replica]:
    """Launch one ``engine.server`` process per device group, all loading at once."""
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
        where = ",".join(map(str, devices)) or "cpu"
        print(f"router: starting replica {index} on devices {where}, port {port}", flush=True)
        process = subprocess.Popen(command, env=env, preexec_fn=_die_with_parent)  # noqa: PLW1509
        replicas.append(Replica(index, devices, port, process))
    return replicas


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
        replica_index, ticket = self.server.affinity.route(key)
        try:
            self._relay(self.server.replicas[replica_index], raw)
        finally:
            self.server.affinity.finish(ticket, replica_index)

    def _relay(self, replica: Replica, raw: bytes) -> None:
        """Send the request to ``replica`` and copy its response back as it arrives."""
        conn = replica.connection()
        try:
            headers = {h: self.headers[h] for h in _FORWARD_HEADERS if self.headers.get(h)}
            headers["Content-Length"] = str(len(raw))
            try:
                conn.request("POST", self.path, body=raw, headers=headers)
                response = conn.getresponse()
            except OSError as exc:
                message = f"replica {replica.index} unavailable: {exc}"
                self._send_json(503, protocol.error_body(message, 503))
                return
            self.send_response(response.status)
            for name in _RETURN_HEADERS:
                value = response.getheader(name)
                if value is not None:
                    self.send_header(name, value)
            size = response.getheader("Content-Length")
            if size is not None:
                self.send_header("Content-Length", size)
                self.end_headers()
                self.wfile.write(response.read())
                return
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
        finally:
            conn.close()


def serve(
    argv: Sequence[str],
    host: str,
    port: int,
    layout: list[tuple[int, ...]],
    hip: bool,
) -> None:
    """Start the replicas, wait for all of them, then route requests until stopped."""
    replicas = start_replicas(argv, layout, hip)

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
