"""ENGINE_ZERO_SYNC: batched device-side greedy picks and text rendered on connection threads.

The setting must not change any response: with it on and off, the same
requests (greedy, sampled with a seed, with stop strings, min_tokens, chat with
reasoning, streamed or not, several at once so they share decode steps) return
the same text, token ids, finish reasons and usage, on the Transformers path
and on the fast path's CPU reference.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import importlib.util
import sys

import httpx
import pytest
import torch
from engine.generator import Generator, zero_sync_enabled
from engine.protocol import Sampling

# This directory's conftest by path: run together with the cpu_check suite, ``conftest`` is that one.
_spec = importlib.util.spec_from_file_location("engine_tests_conftest", Path(__file__).with_name("conftest.py"))
_conftest = sys.modules.setdefault(_spec.name, importlib.util.module_from_spec(_spec))
_spec.loader.exec_module(_conftest)
Server, serve = _conftest.Server, _conftest.serve

_TIMEOUT = 120


def _serve_with(env: dict[str, str], checkpoint: Path, log: Path) -> Iterator[Server]:
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        servers = serve(checkpoint, log)
        server = next(servers)  # the process is started (and healthy) here
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


@pytest.fixture(scope="module", params=["0", "1"], ids=["transformers", "fast"])
def pair(request, checkpoint: Path, tmp_path_factory) -> Iterator[tuple[Server, Server]]:
    logs = tmp_path_factory.mktemp("zero-sync-logs")
    fast = request.param
    off = _serve_with(
        {"ENGINE_FAST": fast, "ENGINE_ZERO_SYNC": "0"}, checkpoint, logs / f"off{fast}.log"
    )
    on = _serve_with(
        {"ENGINE_FAST": fast, "ENGINE_ZERO_SYNC": "1"}, checkpoint, logs / f"on{fast}.log"
    )
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
                "max_tokens": 9 + s,
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
    (
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8, "temperature": 0},
    ),
]


def test_responses_are_identical_with_zero_sync_on_and_off(pair: tuple[Server, Server]) -> None:
    off, on = pair
    # One at a time, then all at once (sharing decode steps; the prefix cache is warm). A
    # sampled request is compared only alone: its batch mates differ from run to run, and on
    # the fast path its logits may differ in the last bits with them, so its draw may too.
    for _ in range(2):
        for path, body in _REQUESTS:
            assert _post(off, path, body) == _post(on, path, body), (path, body)
    greedy = [r for r in _REQUESTS if r[1].get("temperature") == 0]
    with ThreadPoolExecutor(len(greedy)) as pool:
        a = list(pool.map(lambda r: _post(off, *r), greedy))
        b = list(pool.map(lambda r: _post(on, *r), greedy))
    for (path, body), x, y in zip(greedy, a, b, strict=True):
        assert x == y, (path, body)
    health = httpx.get(f"{on.url}/health", timeout=10).json()
    assert health["zero_sync"] == 1
    assert httpx.get(f"{off.url}/health", timeout=10).json()["zero_sync"] == 0


def test_the_setting_defaults_on() -> None:
    assert zero_sync_enabled("") and zero_sync_enabled("1") and not zero_sync_enabled("0")


def test_batched_greedy_picks_equal_per_row_picks() -> None:
    """``_select`` against ``pick`` on rows with ties, sampled rows and min_tokens masking."""
    torch.manual_seed(0)
    vocab = 300
    logits = torch.randn(6, vocab)
    logits[1, 17] = logits[1, 230] = logits[1].max() + 1  # a tie: the first index wins
    logits[2, 3] = 50.0  # the eos id is the argmax, masked while below min_tokens
    eos = frozenset({3})
    samplings = [
        Sampling(max_tokens=8, temperature=0.0, ignore_eos=True),
        Sampling(max_tokens=8, temperature=0.0),
        Sampling(max_tokens=8, temperature=0.0, min_tokens=4),
        Sampling(max_tokens=8, temperature=0.7, seed=1),
        Sampling(max_tokens=8, temperature=0.0, min_tokens=1),
        Sampling(max_tokens=8, temperature=0.0),
    ]
    gen = Generator.__new__(Generator)
    gen.eos_ids = eos
    gen.zero_sync = True
    gen._readback = {}
    gens = [
        SimpleNamespace(_sampling=s, _count=2, logits=None, next_token=None) for s in samplings
    ]
    gen._select(gens, logits, clone=True)
    for g, s, row in zip(gens, samplings, logits, strict=True):
        if s.temperature == 0.0 and not (s.min_tokens > g._count and not s.ignore_eos):
            assert g.next_token == gen.pick(row, s, g._count, None)
        else:
            assert g.next_token is None
            assert torch.equal(g.logits, row) and g.logits.data_ptr() != row.data_ptr()
    assert gens[1].next_token == 17
