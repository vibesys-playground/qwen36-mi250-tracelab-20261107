"""ENGINE_LOOKAHEAD: decode steps launched without waiting, and the one-step lookahead.

With the setting on, a step whose requests are all greedy is queued
(``Generator.launch``) and collected later (``Generator.collect``); when every
request of a collected step needs another step after the tokens it just got,
the next step is launched from those tokens before they are handed out. None of
this may change a token, a session tail or a response:

- driven like the server's worker, with MTP drafts forced right and wrong, the
  lookahead gives the tokens and session tails that synchronous steps give
  (and plain decode gives), over two rounds of different lengths;
- the same without the MTP head (plain launched decode steps);
- a request whose slot is still ahead of what the host saw keeps no tail;
- served end to end with the tiny MTP checkpoint, the setting on and off give
  identical responses (greedy, sampled, stop strings, min_tokens, chat, multi-
  round sessions, several at once), and the lookahead actually runs.

    UV_PROJECT_ENVIRONMENT=... uv run --project .vibesys/shared/cpu_check \
        python -m pytest tests/test_lookahead.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vibesys" / "shared" / "cpu_check"))

from engine.generator import Generator, lookahead_enabled  # noqa: E402
from engine.protocol import Sampling  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"engine_tests_{name}", Path(__file__).with_name(f"{name}.py"))
    module = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


_conftest = _load("conftest")
_spec = _load("test_speculative")
Server, serve = _conftest.Server, _conftest.serve
CHUNK = 8
_TIMEOUT = 120


@pytest.fixture(scope="module")
def mtp_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-mtp-lookahead")
    _spec.write_mtp_checkpoint(path)
    return path


def _generator(checkpoint: Path, monkeypatch: pytest.MonkeyPatch, mtp: str, lookahead: str) -> Generator:
    monkeypatch.setenv("ENGINE_FAST", "1")
    monkeypatch.setenv("ENGINE_WARMUP", "0")
    monkeypatch.setenv("ENGINE_MTP", mtp)
    monkeypatch.setenv("ENGINE_LOOKAHEAD", lookahead)
    g = Generator.load(
        str(checkpoint), None, max_model_len=4096, chunk_tokens=CHUNK,
        prefix_cache_tokens=100000, prefix_cache_entries=16,
    )
    assert (g.fast.mtp is not None) == (mtp == "1")
    assert g.lookahead == (lookahead == "1")
    return g


def _drive(
    g: Generator, prompts: list[list[int]], max_tokens: list[int], truth: list[list[int]] | None
) -> list[list[int]]:
    """One round, stepped as ``engine.server.Worker`` steps it: collect, launch ahead when
    every request runs past its picks, hand the tokens out, else launch after them. With
    ``truth`` (the right tokens), drafts are forced right at every other position."""
    gens = [
        g.start(tuple(p), Sampling(max_tokens=m, temperature=0.0, ignore_eos=True))
        for p, m in zip(prompts, max_tokens, strict=True)
    ]
    toks: list[list[int]] = [[] for _ in gens]

    def take(i: int) -> bool:  # the request's picked tokens; True while it needs a step
        while True:
            step = gens[i].next_step()
            toks[i].append(step.token_id)
            if step.finish is not None:
                gens[i].retire()
                return False
            if not gens[i].has_queued():
                return True

    def force(i: int) -> None:
        if truth is None or not gens[i].speculates:
            return
        gen = gens[i]
        nxt = len(toks[i]) + len(gen.queue) + (gen.next_token is not None)
        if nxt % 2 == 0 and nxt < len(truth[i]):
            gen.draft = truth[i][nxt]

    live = [i for i in range(len(gens)) if take(i)]
    launched, batch = None, []
    while True:
        ahead = False
        if launched is not None:
            g.collect(launched)
            if all(gens[i].runs_past_picked() for i in batch) and g.launchable([gens[i] for i in batch]):
                for i in batch:
                    force(i)
                launched = g.launch([gens[i] for i in batch], ahead=True)
                ahead = True
            else:
                launched = None
            live = [i for i in batch if take(i)]
            if ahead:
                assert live == batch
                continue
        if not live:
            return toks
        for i in live:
            force(i)
        batch = list(live)
        launched = g.launch([gens[i] for i in batch])
        if launched is None:  # ran synchronously
            live = [i for i in batch if take(i)]


def _rounds(g: Generator, truth: dict[int, list[list[int]]] | None) -> tuple[list, dict]:
    prompts = [[(3 * i + j) % 300 + 5 for i in range(n)] for j, n in enumerate([9, 40, 3])]
    out = []
    for rnd, lengths in enumerate(([11, 7, 14], [6, 13, 9])):
        toks = _drive(g, prompts, list(lengths), None if truth is None else truth[rnd])
        out.append(toks)
        prompts = [p + t + [17, 18] for p, t in zip(prompts, toks, strict=True)]
    return out, g.stats()


@pytest.mark.parametrize("mtp", ["1", "0"], ids=["mtp", "plain"])
def test_lookahead_gives_the_tokens_and_tails_of_synchronous_steps(
    mtp_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, mtp: str
) -> None:
    plain, _ = _rounds(_generator(mtp_checkpoint, monkeypatch, "0", "0"), None)
    truth = dict(enumerate(plain))
    sync, sync_stats = _rounds(_generator(mtp_checkpoint, monkeypatch, mtp, "0"), truth)
    ahead, stats = _rounds(_generator(mtp_checkpoint, monkeypatch, mtp, "1"), truth)
    assert sync == plain
    assert ahead == plain
    assert all(len(t) == m for r, ms in zip(ahead, ([11, 7, 14], [6, 13, 9])) for t, m in zip(r, ms))
    assert stats["lookahead_steps"] > 0 and stats["launched_steps"] == stats["decode_steps"]
    assert sync_stats["launched_steps"] == 0
    # Same steps, same session tails reused by round two.
    assert stats["decode_steps"] == sync_stats["decode_steps"]
    assert stats["prefix_cache_tail_hits"] == sync_stats["prefix_cache_tail_hits"] == 3
    if mtp == "1":
        assert stats["mtp_accepted"] == sync_stats["mtp_accepted"] > 0
        assert stats["mtp_proposed"] == sync_stats["mtp_proposed"]


def test_a_request_with_a_step_in_flight_keeps_no_tail(
    mtp_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = _generator(mtp_checkpoint, monkeypatch, "1", "1")
    prompt = tuple((5 * i + 2) % 300 + 5 for i in range(21))
    gen = g.start(prompt, Sampling(max_tokens=2, temperature=0.0, ignore_eos=True))
    entries = g.stats()["prefix_cache_entries"]  # the prefill's snapshots
    assert gen.next_step().finish is None
    launched = g.launch([gen])
    g.collect(launched)
    assert not gen.runs_past_picked()  # its last token is picked: no step ahead
    launched = g.launch([gen], ahead=True)  # forced here, as a cancelled batch mate would be
    step = gen.next_step()
    assert step.finish == "length"
    gen.retire()
    assert g.stats()["prefix_cache_entries"] == entries  # no tail
    g.collect(launched)
    # The same without the step ahead keeps the tail.
    g._reclaim()
    gen = g.start(prompt, Sampling(max_tokens=2, temperature=0.0, ignore_eos=True))
    gen.next_step()
    g.collect(g.launch([gen]))
    assert gen.next_step().finish == "length"
    gen.retire()
    assert g.stats()["prefix_cache_entries"] == entries + 1


def test_the_setting_defaults_on() -> None:
    assert lookahead_enabled("") and lookahead_enabled("1") and not lookahead_enabled("0")


# ---------------------------------------------------------------- served


def _serve_with(env: dict[str, str], checkpoint: Path, log: Path) -> Iterator[Server]:
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        servers = serve(checkpoint, log)
        server = next(servers)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    try:
        yield server
    finally:
        servers.close()


@pytest.fixture(scope="module", params=["1", "0"], ids=["mtp", "plain"])
def pair(request, mtp_checkpoint: Path, tmp_path_factory) -> Iterator[tuple[Server, Server]]:
    logs = tmp_path_factory.mktemp("lookahead-logs")
    mtp = request.param
    base = {"ENGINE_FAST": "1", "ENGINE_MTP": mtp, "ENGINE_WARMUP": "0"}
    off = _serve_with({**base, "ENGINE_LOOKAHEAD": "0"}, mtp_checkpoint, logs / f"off{mtp}.log")
    on = _serve_with({**base, "ENGINE_LOOKAHEAD": "1"}, mtp_checkpoint, logs / f"on{mtp}.log")
    a, b = next(off), next(on)
    try:
        yield a, b
    finally:
        off.close()
        on.close()


def _post(server: Server, path: str, body: dict[str, Any]) -> Any:
    body = {"model": server.model, **body}
    if not body.get("stream"):
        response = httpx.post(f"{server.url}{path}", json=body, timeout=_TIMEOUT)
        response.raise_for_status()
        out = response.json()
        out.pop("id", None)
        out.pop("created", None)
        return out
    with httpx.stream("POST", f"{server.url}{path}", json=body, timeout=_TIMEOUT) as response:
        response.raise_for_status()
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    for chunk in chunks:
        chunk.pop("id", None)
        chunk.pop("created", None)
    return chunks


def _prompt(seed: int, n: int) -> list[int]:
    return [7 + (seed * 37 + i * 11) % 240 for i in range(n)]


_REQUESTS: list[tuple[str, dict[str, Any]]] = [
    *[
        (
            "/v1/completions",
            {
                "prompt": _prompt(s, 5 + 3 * s),
                "max_tokens": 9 + 4 * s,
                "temperature": 0,
                "ignore_eos": True,
                "return_token_ids": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        for s in range(4)
    ],
    ("/v1/completions", {"prompt": "Hello there, tiny model", "max_tokens": 12, "temperature": 0}),
    (
        "/v1/completions",
        {"prompt": _prompt(9, 11), "max_tokens": 10, "temperature": 0, "min_tokens": 6, "stream": True},
    ),
    (
        "/v1/completions",
        {"prompt": _prompt(5, 7), "max_tokens": 10, "temperature": 0.8, "seed": 5, "top_k": 20},
    ),
    (
        "/v1/completions",
        {"prompt": "a b c d e f", "max_tokens": 16, "temperature": 0, "stop": ["e", " the"]},
    ),
    (
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Say something."}],
            "max_tokens": 14,
            "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    ),
]


def _session(server: Server, seed: int) -> list[Any]:
    """Three rounds, each extending the last with its output (session tails)."""
    prompt = _prompt(seed, 12 + seed)
    out = []
    for rnd in range(3):
        body = {
            "prompt": prompt, "max_tokens": 7 + 3 * rnd + seed, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True, "stream": True,
        }
        chunks = _post(server, "/v1/completions", body)
        ids = [t for c in chunks for ch in c["choices"] for t in (ch.get("token_ids") or [])]
        out.append(chunks)
        prompt = prompt + ids + [19 + rnd, 23]
    return out


def test_responses_are_identical_with_lookahead_on_and_off(pair: tuple[Server, Server]) -> None:
    off, on = pair
    for path, body in _REQUESTS:
        assert _post(off, path, body) == _post(on, path, body), (path, body)
    greedy = [r for r in _REQUESTS if r[1].get("temperature") == 0]
    with ThreadPoolExecutor(len(greedy)) as pool:
        a = list(pool.map(lambda r: _post(off, *r), greedy))
        b = list(pool.map(lambda r: _post(on, *r), greedy))
    for (path, body), x, y in zip(greedy, a, b, strict=True):
        assert x == y, (path, body)
    with ThreadPoolExecutor(3) as pool:
        a = list(pool.map(lambda s: _session(off, s), range(3)))
        b = list(pool.map(lambda s: _session(on, s), range(3)))
    assert a == b
    health_on = httpx.get(f"{on.url}/health", timeout=10).json()
    health_off = httpx.get(f"{off.url}/health", timeout=10).json()
    assert health_on["lookahead"] == 1 and health_off["lookahead"] == 0
    assert health_on["lookahead_steps"] > 0 and health_off["launched_steps"] == 0
    assert health_on["prefix_cache_tail_hits"] == health_off["prefix_cache_tail_hits"] > 0
