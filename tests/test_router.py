"""The replica router: layout, session affinity, and two CPU replicas of the tiny checkpoint.

Run with the CPU check's environment, from the project root:

    UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/vibesys/qwen36-cpu-check-venv" \\
      uv run --project .vibesys/shared/cpu_check python -m pytest tests
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / ".vibesys/shared/cpu_check")]

from engine import router  # noqa: E402
from engine.router import Affinity, affinity_key, device_env, resolve_layout  # noqa: E402
from conftest import repeat_reuse  # noqa: E402
from tiny_checkpoint import write_checkpoint  # noqa: E402

CHUNK = 8
TIMEOUT = 120


# ----------------------------------------------------------------------------- layout


def test_default_layout_is_four_pairs_on_eight_gpus() -> None:
    assert resolve_layout(8) == [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_layout_options() -> None:
    assert resolve_layout(8, gpus_per_replica=4) == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert resolve_layout(8, replicas=2) == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert resolve_layout(8, replicas=1) == [tuple(range(8))]
    assert resolve_layout(8, groups="0,2;1,3") == [(0, 2), (1, 3)]
    assert resolve_layout(1) == [(0,)]
    assert resolve_layout(0) == [()]
    assert resolve_layout(0, replicas=2) == [(), ()]
    with pytest.raises(ValueError):
        resolve_layout(4, replicas=3, gpus_per_replica=2)
    with pytest.raises(ValueError):
        resolve_layout(4, groups="0,1;1,2")


def test_device_env_slices_an_existing_visibility_list() -> None:
    assert device_env({}, (2, 3), hip=True)["HIP_VISIBLE_DEVICES"] == "2,3"
    env = device_env({"HIP_VISIBLE_DEVICES": "4,5,6,7"}, (2, 3), hip=True)
    assert env["HIP_VISIBLE_DEVICES"] == "6,7"
    env = device_env({"CUDA_VISIBLE_DEVICES": "a,b,c,d"}, (0, 1), hip=True)
    assert env["CUDA_VISIBLE_DEVICES"] == "a,b" and "HIP_VISIBLE_DEVICES" not in env
    assert device_env({"X": "1"}, (), hip=True) == {"X": "1"}


# ----------------------------------------------------------------------------- affinity


def _session(seed: int, length: int) -> tuple[int, ...]:
    return tuple(1000 * seed + i for i in range(length))


def test_a_growing_session_stays_on_its_replica_and_new_sessions_spread() -> None:
    affinity = Affinity(4, min_tokens=16)
    sessions = [_session(s, 100) for s in range(4)]
    placed = []
    for prompt in sessions:
        replica, ticket = affinity.route(prompt)
        placed.append(replica)
        affinity.finish(ticket, replica)
    assert sorted(placed) == [0, 1, 2, 3]
    for _ in range(3):  # every later round extends the previous prompt
        for s, prompt in enumerate(sessions):
            sessions[s] = prompt + tuple(range(50))
            replica, ticket = affinity.route(sessions[s])
            assert replica == placed[s]
            affinity.finish(ticket, replica)


def test_a_session_extending_another_does_not_steal_it() -> None:
    affinity = Affinity(2, min_tokens=16, idle_seconds=0)
    a = _session(1, 100)
    ra, t = affinity.route(a)
    affinity.finish(t, ra)
    b = a + _session(2, 50)  # a new session that happens to start with a's prompt
    rb, t = affinity.route(b)
    affinity.finish(t, rb)
    assert rb == ra
    a2 = a + _session(3, 20)  # a's own next round still finds a's prompt
    r, t = affinity.route(a2)
    assert r == ra


def test_short_prompts_attract_only_identical_ones() -> None:
    affinity = Affinity(2, min_tokens=64, idle_seconds=0)
    probe = _session(1, 5)
    r0, t = affinity.route(probe)
    affinity.finish(t, r0)
    r1, t = affinity.route(probe + _session(2, 200))
    affinity.finish(t, r1)
    assert r1 != r0  # balanced, not pulled onto the probe's replica
    again, _ = affinity.route(probe)
    assert again == r0


def test_in_flight_requests_count_as_load() -> None:
    affinity = Affinity(2, min_tokens=16)
    r0, _ = affinity.route(_session(1, 100))  # still running
    r1, _ = affinity.route(_session(2, 100))
    assert {r0, r1} == {0, 1}
    assert affinity.load() == [1, 1]


def test_chat_history_matches_message_by_message() -> None:
    messages = [{"role": "system", "content": "x" * 400}, {"role": "user", "content": "hi"}]
    first = affinity_key("/v1/chat/completions", {"messages": messages})
    grown = messages + [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
    ]
    second = affinity_key("/v1/chat/completions", {"messages": grown})
    affinity = Affinity(4, min_tokens=16)
    r, t = affinity.route(first)
    affinity.finish(t, r)
    assert affinity.route(second)[0] == r
    assert affinity_key("/v1/completions", {"prompt": [1, 2]}) == (1, 2)
    assert affinity_key("/v1/completions", {"prompt": "ab"}) == "ab"
    assert affinity_key("/v1/completions", None) is None


# ----------------------------------------------------------------------------- two CPU replicas


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def front(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    checkpoint = tmp_path_factory.mktemp("tiny-qwen36")
    write_checkpoint(checkpoint)
    log = tmp_path_factory.mktemp("logs") / "front.log"
    port = _free_port()
    with log.open("w") as sink:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "engine.server",
                "--model",
                str(checkpoint),
                "--port",
                str(port),
                "--prefill-chunk-tokens",
                str(CHUNK),
                "--replicas",
                "2",
            ],
            cwd=ROOT,
            stdout=sink,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "ENGINE_AFFINITY_MIN_TOKENS": "16"},
        )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 300
        while True:
            if process.poll() is not None:
                pytest.fail(f"front exited with {process.returncode}:\n{log.read_text()}")
            try:
                if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"front did not become healthy:\n{log.read_text()}")
            time.sleep(0.2)
        yield url
    finally:
        process.terminate()
        process.wait(timeout=60)
        # The replicas go down with the front.
        assert "replica 1 ready" in log.read_text()


def _completion(url: str, prompt: list[int], max_tokens: int) -> tuple[list[int], dict]:
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with httpx.stream("POST", f"{url}/v1/completions", json=body, timeout=TIMEOUT) as response:
        response.raise_for_status()
        lines = [x for x in response.iter_lines() if x.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    *chunks, last = [json.loads(x.removeprefix("data: ")) for x in lines[:-1]]
    return [t for c in chunks for t in c["choices"][0]["token_ids"]], last["usage"]


def _replica_tokens(url: str) -> list[int]:
    return [r["decoded_tokens"] for r in httpx.get(f"{url}/health").json()["replicas"]]


def test_two_replicas_keep_each_session_and_its_cache(front: str) -> None:
    health = httpx.get(f"{front}/health").json()
    assert len(health["replicas"]) == 2
    models = httpx.get(f"{front}/v1/models").json()
    assert len(models["data"]) == 1

    def session(seed: int) -> list[tuple[list[int], dict]]:
        prompt = [5 + (seed * 37 + i * 11) % 300 for i in range(3 * CHUNK + 4)]
        rounds = []
        for turn in range(3):
            output, usage = _completion(front, prompt, 6)
            rounds.append((output, usage))
            prompt = prompt + output + [7 + (seed + turn + i) % 300 for i in range(5)]
        return rounds

    before = _replica_tokens(front)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(session, range(4)))
    after = _replica_tokens(front)
    health = httpx.get(f"{front}/health").json()
    # The front sums the replicas' prefill and prefix-cache counters.
    for key in ("prefilled_tokens", "cached_tokens", "prefix_cache_tail_hits"):
        assert health[key] == sum(r[key] for r in health["replicas"]) > 0
    # Both replicas did work: the sessions were spread.
    assert all(a > b for a, b in zip(after, before, strict=True))
    for rounds in results:
        assert rounds[0][1]["prompt_tokens_details"]["cached_tokens"] == 0
        for output, usage in rounds:
            assert len(output) == usage["completion_tokens"] == 6
        # Later rounds land where the session's snapshots are.
        for _, usage in rounds[1:]:
            assert usage["prompt_tokens_details"]["cached_tokens"] > 0


def test_a_repeated_prompt_hits_the_cache_through_the_router(front: str) -> None:
    prompt = list(range(100, 100 + 3 * CHUNK + 5))
    first, first_usage = _completion(front, prompt, 8)
    second, second_usage = _completion(front, prompt, 8)
    assert first_usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert second_usage["prompt_tokens_details"]["cached_tokens"] == repeat_reuse(len(prompt), CHUNK)
    assert second == first


def test_errors_and_non_streaming_pass_through(front: str) -> None:
    bad = httpx.post(f"{front}/v1/completions", json={"prompt": [10**6]}, timeout=TIMEOUT)
    assert bad.status_code == 400 and bad.json()["error"]["message"]
    whole = httpx.post(
        f"{front}/v1/completions",
        json={"prompt": "def f():", "max_tokens": 5, "temperature": 0, "ignore_eos": True},
        timeout=TIMEOUT,
    ).json()
    assert whole["usage"]["completion_tokens"] == 5
    assert httpx.get(f"{front}/nowhere").status_code == 404
    _ = router  # imported for the module under test


# ----------------------------------------------------------------------------- live moves (plan 63)


def _finish_all(affinity: Affinity, routed: list[tuple[int, int]], now: float) -> list:
    return [affinity.finish(t, r, now=now) for r, t in routed]


def test_a_round_ending_on_a_crowded_replica_moves_its_session() -> None:
    affinity = Affinity(2, min_tokens=16, idle_seconds=0, migrate=True, gap=2)
    a, b, c, d, e = (_session(s, 100) for s in range(1, 6))
    routed = {name: affinity.route(p, now=0.0) for name, p in zip("abcde", (a, b, c, d, e))}
    assert [routed[n][0] for n in "abcde"] == [0, 1, 0, 1, 0]
    # b and d end on replica 1 (the less loaded side: nothing moves) and send nothing more.
    assert affinity.finish(routed["b"][1], 1, now=0.1) is None
    assert affinity.finish(routed["d"][1], 1, now=0.1) is None
    # Their resting window (3 s) lapses; a's round ends on 0 while c and e still run there.
    move = affinity.finish(routed["a"][1], 0, now=5.0)
    assert move is not None and (move.src, move.dst) == (0, 1)
    assert affinity._load(5.0, 3.0) == [2, 1]  # a counts on its target while it moves
    # One move out of a replica at a time.
    assert affinity.finish(routed["c"][1], 0, now=5.1) is None
    assert affinity.commit(move, 1.0, 2.0)
    affinity.release(move)
    r, t = affinity.route(a + _session(9, 10), now=5.2)
    assert r == 1 and affinity.moved(t)
    stats = affinity.stats(now=5.2)["moves"]
    assert stats["committed"] == 1 and stats["moved_rounds"] == 1 and stats["started"] == 1


def test_no_move_without_the_gap_or_when_off() -> None:
    for migrate in (False, True):
        affinity = Affinity(2, min_tokens=16, migrate=migrate, gap=2)
        r1 = affinity.route(_session(1, 100), now=0.0)
        r2 = affinity.route(_session(2, 100), now=0.0)
        assert _finish_all(affinity, [r1, r2], now=1.0) == [None, None]
    # Short prompts (probes) never move.
    affinity = Affinity(2, min_tokens=16, migrate=True, gap=0)
    r, t = affinity.route(_session(1, 4), now=0.0)
    assert affinity.finish(t, r, now=1.0) is None


def test_a_round_arriving_mid_move_waits_or_goes_to_the_source() -> None:
    affinity = Affinity(2, min_tokens=16, migrate=True, gap=0, move_wait_seconds=0.05)
    a = _session(1, 100)
    r, t = affinity.route(a)
    move = affinity.finish(t, r)
    assert move is not None and move.src == r
    # The copy does not finish in time: the round goes to the source, which has the state,
    # and the late commit is refused (the caller then drops the target's copy).
    r2, t2 = affinity.route(a + _session(2, 10))
    assert r2 == r and not affinity.moved(t2)
    assert not affinity.commit(move, 1.0, 1.0)
    affinity.release(move)
    assert affinity.stats()["moves"]["abandoned"] == 1
    # A move that finishes while the round waits: the round follows the state.
    move = affinity.finish(t2, r2)
    assert move is not None
    timer = threading.Timer(0.01, lambda: affinity.commit(move, 1.0, 1.0))
    affinity.move_wait_seconds = 5.0
    timer.start()
    r3, t3 = affinity.route(a + _session(2, 10) + _session(3, 10))
    timer.join()
    assert r3 == move.dst and affinity.moved(t3)
    affinity.release(move)
    # A failed move leaves the session where it was.
    move = affinity.finish(t3, r3)
    affinity.fail(move, "test")
    affinity.release(move)
    r4, _ = affinity.route(a + _session(2, 10) + _session(3, 10) + _session(4, 10))
    assert r4 == r3


def _front(checkpoint: Path, log: Path, env: dict[str, str]) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    with log.open("w") as sink:
        process = subprocess.Popen(
            [sys.executable, "-m", "engine.server", "--model", str(checkpoint), "--port",
             str(port), "--prefill-chunk-tokens", str(CHUNK), "--replicas", "2"],
            cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "ENGINE_AFFINITY_MIN_TOKENS": "16", **env},
        )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300
    while True:
        if process.poll() is not None:
            pytest.fail(f"front exited with {process.returncode}:\n{log.read_text()}")
        try:
            if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                return process, url
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            process.terminate()
            pytest.fail(f"front did not become healthy:\n{log.read_text()}")
        time.sleep(0.2)


def _moved_completion(url: str, prompt: list[int], n: int) -> tuple[list[int], dict, dict]:
    body = {"prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True, "stream": True, "stream_options": {"include_usage": True}}
    with httpx.stream("POST", f"{url}/v1/completions", json=body, timeout=TIMEOUT) as response:
        response.raise_for_status()
        headers = dict(response.headers)
        lines = [x for x in response.iter_lines() if x.startswith("data: ")]
    *chunks, last = [json.loads(x.removeprefix("data: ")) for x in lines[:-1]]
    return [t for c in chunks for t in c["choices"][0]["token_ids"]], last["usage"], headers


def test_a_moved_session_continues_token_for_token(tmp_path: Path) -> None:
    """Two CPU replicas on the engine's own forward: with a zero gap every round's state moves
    to the other replica; outputs and cached tokens match a front that never moves."""
    checkpoint = tmp_path / "tiny"
    write_checkpoint(checkpoint)
    common = {"ENGINE_FAST": "1", "ENGINE_MIGRATE_WAIT_MS": "20000"}
    runs = {}
    for name, env in (("off", {"ENGINE_MIGRATE": "0"}),
                      ("on", {"ENGINE_MIGRATE": "1", "ENGINE_MIGRATE_GAP": "0"})):
        process, url = _front(checkpoint, tmp_path / f"{name}.log", {**common, **env})
        try:
            prompt = [5 + (i * 13) % 300 for i in range(3 * CHUNK + 3)]
            rounds = []
            for turn in range(4):
                out, usage, headers = _moved_completion(url, prompt, 7)
                rounds.append((out, usage["prompt_tokens_details"]["cached_tokens"],
                               headers.get("x-engine-replica"), headers.get("x-engine-moved")))
                prompt = prompt + out + [9 + (turn * 7 + i) % 300 for i in range(4)]
                time.sleep(0.2)
            health = httpx.get(f"{url}/health", timeout=30).json()
        finally:
            process.terminate()
            process.wait(timeout=60)
        runs[name] = (rounds, health)
    off, on = runs["off"][0], runs["on"][0]
    assert [r[0] for r in on] == [r[0] for r in off]  # token-identical continuations
    assert [r[1] for r in on] == [r[1] for r in off]  # the same real reuse of the moved state
    assert all(r[1] > 0 for r in on[1:])
    assert len({r[2] for r in off}) == 1 and all(r[3] is None for r in off)
    replicas = [r[2] for r in on]
    assert all(x != y for x, y in zip(replicas, replicas[1:], strict=False))  # it moved each round
    assert all(r[3] == "1" for r in on[1:])
    moves = runs["on"][1]["router"]["moves"]
    assert moves["committed"] >= 3 and moves["failed"] == 0, moves
    assert sum(r["migrate"]["import"] for r in runs["on"][1]["replicas"]) >= 3


# ----------------------------------------------------------------------------- two holders (plan 70)


def _two(replicas: int = 3, **kw) -> Affinity:
    return Affinity(replicas, min_tokens=16, idle_seconds=0, two_holder=True, **kw)


def test_a_synced_round_goes_to_the_less_busy_holder() -> None:
    affinity = _two()
    a = _session(1, 100)
    r0, t0 = affinity.route(a, now=0.0)
    sync = affinity.finish(t0, r0, now=1.0)
    assert sync is not None and sync.src == r0 and sync.dst != r0
    assert affinity.sync_done(sync, 105, 1.0, 1.0)
    affinity.release(sync)
    # Two other sessions now run on the home replica; the buddy is idle.
    busy = [affinity.route(_session(s, 100), now=1.1) for s in (7, 8)]
    for r, t in busy:
        affinity._inflight[r] -= 1
        affinity._inflight[r0] += 1
    r1, t1 = affinity.route(a + _session(2, 10), now=1.2)
    assert r1 == sync.dst and affinity.moved(t1) and affinity.required(t1) == 105
    # With the home no busier than the buddy, the round stays home.
    affinity2 = _two()
    r0, t0 = affinity2.route(a, now=0.0)
    sync = affinity2.finish(t0, r0, now=1.0)
    affinity2.sync_done(sync, 105, 1.0, 1.0)
    affinity2.release(sync)
    r1, t1 = affinity2.route(a + _session(2, 10), now=1.2)
    assert r1 == r0 and not affinity2.moved(t1) and affinity2.required(t1) is None


def test_a_sync_in_flight_at_arrival_never_qualifies_nor_delays() -> None:
    affinity = _two(prefer_buddy=True)
    a = _session(1, 100)
    r0, t0 = affinity.route(a, now=0.0)
    sync = affinity.finish(t0, r0, now=1.0)
    assert sync is not None
    # The next round arrives while the copy is in flight: it goes home at once.
    started = time.monotonic()
    r1, t1 = affinity.route(a + _session(2, 10))
    assert time.monotonic() - started < 0.05
    assert r1 == r0 and not affinity.moved(t1)
    assert affinity.stats()["syncs"]["in_flight_at_arrival"] == 1
    # The copy lands late: the buddy holds round t0's state, an older version than the one
    # round t1 will leave, so the round after t1 cannot go there either.
    assert affinity.sync_done(sync, 105, 1.0, 1.0)
    affinity.release(sync)
    assert affinity._entries[t0].holders == {r0: t0, sync.dst: t0}
    affinity.finish(t1, r1)  # a new sync of t1 starts (to the old buddy again)
    entry = affinity._entries[t1]
    assert entry.holders == {r1: t1} and entry.sync is not None and entry.sync.dst == sync.dst
    r2, t2 = affinity.route(a + _session(2, 10) + _session(3, 10))
    assert r2 == r1 and not affinity.moved(t2)
    # A failed sync leaves only the home as a holder.
    affinity.sync_failed(entry.sync, "test")
    affinity.release(entry.sync if entry.sync else sync)


def test_versions_follow_the_serving_replica_and_stale_copies_never_qualify() -> None:
    affinity = _two(replicas=2, prefer_buddy=True)
    prompt = _session(1, 100)
    served = []
    replica, ticket = affinity.route(prompt, now=0.0)
    for turn in range(4):
        served.append(replica)
        sync = affinity.finish(ticket, replica)
        assert sync is not None and sync.src == replica
        assert affinity.sync_done(sync, 100 + turn, 1.0, 1.0)
        affinity.release(sync)
        prompt = prompt + _session(turn + 2, 10)
        replica, ticket = affinity.route(prompt)
        # Exactly one replica serves each round; with both holders current and idle, the
        # test knob alternates them.
        assert replica != served[-1] and affinity.moved(ticket)
    # The replica that served a round is its only holder until a copy is acknowledged.
    sync = affinity.finish(ticket, replica)
    assert affinity._entries[ticket].holders == {replica: ticket}
    affinity.sync_failed(sync, "lost")
    affinity.release(sync)
    nxt, t = affinity.route(prompt + _session(9, 10))
    assert nxt == replica and not affinity.moved(t)


def test_a_refused_buddy_round_goes_home() -> None:
    affinity = _two(replicas=2, prefer_buddy=True)
    a = _session(1, 100)
    r0, t0 = affinity.route(a, now=0.0)
    sync = affinity.finish(t0, r0)
    affinity.sync_done(sync, 105, 1.0, 1.0)
    affinity.release(sync)
    r1, t1 = affinity.route(a + _session(2, 10))
    assert r1 == sync.dst
    assert affinity.reroute(t1) == r0
    assert affinity._inflight == [1, 0] if r0 == 0 else [0, 1]
    assert not affinity.moved(t1) and affinity.required(t1) is None
    assert affinity.stats()["syncs"]["stale_refused"] == 1


def test_short_prompts_and_busy_ends_do_not_sync() -> None:
    affinity = _two(replicas=2)
    r, t = affinity.route(_session(1, 4))
    assert affinity.finish(t, r) is None
    r1, t1 = affinity.route(_session(2, 100))
    affinity.route(_session(3, 100))
    r2, t2 = affinity.route(_session(4, 100))
    assert r1 == r2
    s1 = affinity.finish(t1, r1)
    assert s1 is not None
    assert affinity.finish(t2, r2) is None  # one copy out of a replica at a time
    assert affinity.stats()["syncs"]["skipped_busy"] == 1


class _FakeReplica(BaseHTTPRequestHandler):
    """Answers like a replica: refuses (stale) any request that requires saved state."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        return

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append(self.headers.get("X-Engine-Require-Cached"))
        if self.headers.get("X-Engine-Require-Cached") is not None:
            status, body, extra = 409, b'{"error": "stale"}', {"X-Engine-Stale": "1"}
        else:
            status, body, extra = 200, json.dumps({"served_by": self.server.name}).encode(), {}
        self.send_response(status)
        for k, v in extra.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_the_front_reroutes_a_stale_refusal_home() -> None:
    fakes = []
    for name in ("r0", "r1"):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeReplica)
        server.name, server.seen = name, []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        fakes.append(server)
    replicas = [router.Replica(i, (), s.server_address[1], None) for i, s in enumerate(fakes)]
    affinity = _two(replicas=2, prefer_buddy=True)
    front = router._Front(("127.0.0.1", 0), replicas, affinity)
    threading.Thread(target=front.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{front.server_address[1]}/v1/completions"
    try:
        a = list(_session(1, 100))
        first = httpx.post(url, json={"prompt": a}, timeout=10)
        home = int(first.headers["x-engine-replica"])
        # Pretend the copy to the buddy was acknowledged (no real replicas here).
        (entry,) = [e for e in affinity._entries.values()]
        deadline = time.monotonic() + 5
        while entry.sync is None and time.monotonic() < deadline:
            time.sleep(0.01)
        # The front's own copy fails against the fakes; wait for it to give up.
        while affinity._sending and time.monotonic() < deadline:
            time.sleep(0.01)
        entry.holders = {home: entry.ticket, 1 - home: entry.ticket}
        entry.tail_tokens = 100
        second = httpx.post(url, json={"prompt": a + [7, 8, 9]}, timeout=10)
        assert second.status_code == 200
        assert second.json()["served_by"] == f"r{home}"
        assert second.headers["x-engine-replica"] == str(home)
        assert "x-engine-moved" not in second.headers
        assert fakes[1 - home].seen == ["100"]  # the buddy was asked, and refused
        assert affinity.stats()["syncs"]["stale_refused"] == 1
        while affinity._inflight != [0, 0] and time.monotonic() < deadline + 5:
            time.sleep(0.01)  # the front finishes the request after the client has it
        assert affinity._inflight == [0, 0]
    finally:
        front.shutdown()
        for s in fakes:
            s.shutdown()


def test_two_holder_sessions_continue_token_for_token(tmp_path: Path) -> None:
    """Two CPU replicas: every round's state is copied to the other replica and the test knob
    sends each next round there; outputs and cached tokens match a front with one holder."""
    checkpoint = tmp_path / "tiny"
    write_checkpoint(checkpoint)
    runs = {}
    for name, env in (("off", {"ENGINE_TWO_HOLDER": "0"}),
                      ("on", {"ENGINE_TWO_HOLDER": "1", "ENGINE_TWO_HOLDER_PREFER_BUDDY": "1"})):
        process, url = _front(checkpoint, tmp_path / f"{name}.log", {"ENGINE_FAST": "1", **env})
        try:
            prompt = [5 + (i * 13) % 300 for i in range(3 * CHUNK + 3)]
            rounds = []
            for turn in range(5):
                out, usage, headers = _moved_completion(url, prompt, 7)
                rounds.append((out, usage["prompt_tokens_details"]["cached_tokens"],
                               headers.get("x-engine-replica"), headers.get("x-engine-moved")))
                prompt = prompt + out + [9 + (turn * 7 + i) % 300 for i in range(4)]
                time.sleep(0.5)  # the tool wait: the copy lands in it
            health = httpx.get(f"{url}/health", timeout=30).json()
        finally:
            process.terminate()
            process.wait(timeout=60)
        runs[name] = (rounds, health)
    off, on = runs["off"][0], runs["on"][0]
    assert [r[0] for r in on] == [r[0] for r in off]  # token-identical continuations
    assert [r[1] for r in on] == [r[1] for r in off]  # the same real reuse, on the buddy
    assert all(r[1] > 0 for r in on[1:])
    assert len({r[2] for r in off}) == 1 and all(r[3] is None for r in off)
    replicas = [r[2] for r in on]
    assert all(x != y for x, y in zip(replicas, replicas[1:], strict=False))
    assert all(r[3] == "1" for r in on[1:])
    syncs = runs["on"][1]["router"]["syncs"]
    assert syncs["committed"] >= 4 and syncs["failed"] == 0 and syncs["stale_refused"] == 0, syncs
    assert syncs["buddy_rounds"] == 4
    assert all("decode_seconds_by_batch" in r for r in runs["on"][1]["replicas"])


def test_a_replica_refuses_a_round_without_the_required_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny"
    write_checkpoint(checkpoint)
    port = _free_port()
    log = tmp_path / "one.log"
    with log.open("w") as sink:
        process = subprocess.Popen(
            [sys.executable, "-m", "engine.server", "--model", str(checkpoint), "--port",
             str(port), "--prefill-chunk-tokens", str(CHUNK), "--replicas", "1"],
            cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "ENGINE_FAST": "1"},
        )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 300
        while True:
            assert process.poll() is None, log.read_text()
            try:
                if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline
            time.sleep(0.2)
        prompt = [5 + (i * 13) % 300 for i in range(3 * CHUNK + 3)]
        body = {"prompt": prompt, "max_tokens": 4, "temperature": 0, "ignore_eos": True,
                "stream": True}
        refused = httpx.post(f"{url}/v1/completions", json=body, timeout=TIMEOUT,
                             headers={"X-Engine-Require-Cached": str(len(prompt))})
        assert refused.status_code == 409 and refused.headers.get("x-engine-stale") == "1"
        ok = httpx.post(f"{url}/v1/completions", json=body, timeout=TIMEOUT,
                        headers={"X-Engine-Require-Cached": "0"})
        assert ok.status_code == 200 and "[DONE]" in ok.text
    finally:
        process.terminate()
        process.wait(timeout=60)
