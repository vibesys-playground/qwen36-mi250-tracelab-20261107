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


def test_pooled_replica_connections_survive_a_client_that_leaves(front: str) -> None:
    """Plan 46: router -> replica connections are reused; a client leaving mid-stream closes
    its upstream connection (cancelling the job) and later requests still stream whole."""
    prompt = [11 + (i * 7) % 300 for i in range(2 * CHUNK + 3)]
    want, _ = _completion(front, prompt, 12)
    body = {"prompt": prompt, "max_tokens": 64, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True, "stream": True}
    with httpx.stream("POST", f"{front}/v1/completions", json=body, timeout=TIMEOUT) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                break  # leave after the first chunk
    for _ in range(6):
        got, usage = _completion(front, prompt, 12)
        assert got == want and usage["completion_tokens"] == 12
    text = {"prompt": "def g(x):\n    return x\n", "max_tokens": 4, "temperature": 0,
            "ignore_eos": True, "stream": True}
    for _ in range(3):
        with httpx.stream("POST", f"{front}/v1/completions", json=text, timeout=TIMEOUT) as r:
            lines = [x for x in r.iter_lines() if x.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        text["prompt"] += "x = 1\n"
