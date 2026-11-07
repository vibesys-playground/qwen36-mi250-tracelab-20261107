"""Session tails: a finished round's model state is reused by the round that extends it.

A multi-round client (Request Factory's session replay) sends round r+1 as
round r's prompt, round r's output token ids and new tokens. The engine keeps
each finished round's cache, covering its prompt and all but its last output
token, so the next round reuses exactly that many tokens and prefills only the
rest. Outputs must not change: each round's greedy tokens equal those of a cold
server with the prefix cache off.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from conftest import CHUNK, Server
from engine.generator import Generation, Generator
from engine.protocol import Sampling

_TIMEOUT = 120


def _completion(server: Server, prompt: list[int], max_tokens: int) -> tuple[list[int], dict]:
    """Stream like Request Factory's session runner: token ids in, token ids and usage out."""
    body = {
        "model": server.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with httpx.stream(
        "POST", f"{server.url}/v1/completions", json=body, timeout=_TIMEOUT
    ) as response:
        response.raise_for_status()
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    *chunks, last = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    ids = [t for c in chunks for t in c["choices"][0]["token_ids"]]
    assert len(ids) == max_tokens == last["usage"]["completion_tokens"]
    return ids, last["usage"]


def _health(server: Server) -> dict[str, Any]:
    return httpx.get(f"{server.url}/health", timeout=10).json()


def _new_tokens(seed: int, count: int) -> list[int]:
    return [7 + (seed * 31 + i * 13) % 250 for i in range(count)]


# Prompt lengths straddle chunk boundaries; output lengths include 1 (an empty fed output).
_ROUNDS = [(2 * CHUNK + 5, 7), (CHUNK + 3, 1), (3, 12)]


def _session(server: Server, seed: int) -> list[tuple[list[int], list[int], int]]:
    """Run a 3-round session; (prompt, output ids, cached tokens) per round."""
    prompt: list[int] = []
    rounds = []
    for index, (new, out) in enumerate(_ROUNDS):
        prompt = prompt + _new_tokens(seed * 7 + index, new)
        ids, usage = _completion(server, prompt, out)
        assert usage["prompt_tokens"] == len(prompt)
        rounds.append((prompt, ids, usage["prompt_tokens_details"]["cached_tokens"]))
        prompt = prompt + ids
    return rounds


def _check_session(rounds: list[tuple[list[int], list[int], int]], cold: Server) -> None:
    for index, (prompt, ids, cached) in enumerate(rounds):
        if index:
            previous_prompt, previous_ids, _ = rounds[index - 1]
            # Exactly the reused tokens: the previous prompt and all but its last output token.
            assert cached == len(previous_prompt) + len(previous_ids) - 1
        cold_ids, cold_usage = _completion(cold, prompt, len(ids))
        assert cold_usage["prompt_tokens_details"]["cached_tokens"] == 0
        assert ids == cold_ids


def test_a_session_round_resumes_from_the_previous_round(server: Server, cold_server: Server) -> None:
    before = _health(server)
    rounds = _session(server, seed=1)
    after = _health(server)
    _check_session(rounds, cold_server)
    sent = sum(len(p) for p, _, _ in rounds)
    reused = sum(c for _, _, c in rounds)
    assert after["prefix_cache_tail_hits"] - before["prefix_cache_tail_hits"] == len(rounds) - 1
    assert after["cached_tokens"] - before["cached_tokens"] == reused
    assert after["prefilled_tokens"] - before["prefilled_tokens"] == sent - reused


def test_repeating_a_resumed_round_repeats_its_tokens(server: Server) -> None:
    rounds = _session(server, seed=2)
    prompt, ids, _ = rounds[-1]
    again, usage = _completion(server, prompt, len(ids))
    assert again == ids
    assert usage["prompt_tokens_details"]["cached_tokens"] == len(prompt) - 1


def test_a_diverging_prompt_does_not_reuse_a_tail(server: Server, cold_server: Server) -> None:
    rounds = _session(server, seed=3)
    prompt, ids, _ = rounds[-1]
    # The same history except one fed output token: the tail (prompt + ids[:-1]) does not match.
    other = prompt + ids[:-2] + [(ids[-2] + 1) % 250 + 5] + _new_tokens(9, 4)
    warm, usage = _completion(server, other, 5)
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    assert cached < len(prompt) + len(ids) - 2
    assert cached % CHUNK == 0  # only a chunk-boundary snapshot can apply
    assert warm == _completion(cold_server, other, 5)[0]


def test_concurrent_sessions_resume_in_one_batch(server: Server, cold_server: Server) -> None:
    before = _health(server)
    seeds = [10, 11, 12, 13]
    with ThreadPoolExecutor(len(seeds)) as pool:
        sessions = list(pool.map(lambda s: _session(server, s), seeds))
    after = _health(server)
    assert after["decoded_tokens"] - before["decoded_tokens"] > (
        after["decode_steps"] - before["decode_steps"]
    )  # the sessions shared decode steps
    for rounds in sessions:
        _check_session(rounds, cold_server)


# In-process: eviction under a byte budget, and compact storage after a batched decode.


def _generator(model: Generator, budget: int | None, entries: int = 64) -> Generator:
    return Generator(
        model.model,
        model.tokenizer,
        model.eos_ids,
        max_model_len=4096,
        chunk_tokens=CHUNK,
        prefix_cache_tokens=1 << 20,
        prefix_cache_entries=entries,
        prefix_cache_bytes=budget,
    )


@pytest.fixture(scope="module")
def loaded(checkpoint: Path) -> Generator:
    torch.manual_seed(0)
    return Generator.load(
        str(checkpoint),
        None,
        max_model_len=4096,
        chunk_tokens=CHUNK,
        prefix_cache_tokens=1 << 20,
        prefix_cache_entries=0,
    )


def _run(gen: Generator, prompts: list[list[int]], n: int) -> tuple[list[list[int]], list[int]]:
    """Serve ``prompts`` together, as the server's worker does; outputs and cached tokens."""
    sampling = Sampling(max_tokens=n, temperature=0.0, ignore_eos=True)
    live: list[Generation] = [gen.start(tuple(p), sampling) for p in prompts]
    cached = [g.cached_tokens for g in live]
    outputs: list[list[int]] = [[] for _ in prompts]
    running = list(range(len(prompts)))
    while running:
        waiting = []
        for i in running:
            step = live[i].next_step()
            outputs[i].append(step.token_id)
            if step.finish is None:
                waiting.append(i)
            else:
                live[i].retire()
        running = waiting
        if running:
            gen.decode([live[i] for i in running])
    return outputs, cached


def _tail(gen: Generator, key: list[int]):
    entry = gen._prefix_cache._entries[tuple(key)]
    assert entry.tail
    return entry


def test_tails_are_evicted_least_recently_used_under_a_byte_budget(loaded: Generator) -> None:
    prompts = [_new_tokens(20 + s, 2 * CHUNK + 3) for s in range(3)]
    n = 6
    # Size one tail with an unbounded cache.
    probe = _generator(loaded, None)
    outs, _ = _run(probe, [prompts[0]], n)
    tail_bytes = _tail(probe, prompts[0] + outs[0][:-1]).nbytes
    assert tail_bytes > 0
    budget = int(2.5 * tail_bytes)
    gen = _generator(loaded, budget)
    first = {}
    for p in prompts:  # one session after another: A, B, C
        out, _ = _run(gen, [p], n)
        first[tuple(p)] = out[0]
        assert gen._prefix_cache.bytes <= budget
    assert gen._prefix_cache.evictions > 0
    tails = {
        k for k, e in gen._prefix_cache._entries.items() if e.tail
    }
    a, c = prompts[0], prompts[2]
    assert tuple(a + first[tuple(a)][:-1]) not in tails  # oldest: evicted
    assert tuple(c + first[tuple(c)][:-1]) in tails  # newest: kept
    cold = _generator(loaded, None, entries=0)
    for p, expect_hit in ((c, True), (a, False)):  # C first: its new entries would evict
        grown = p + first[tuple(p)] + _new_tokens(40, 5)
        out, cached = _run(gen, [grown], n)
        resumed = len(p) + n - 1
        assert (cached[0] == resumed) is expect_hit
        assert cached[0] <= resumed
        assert out == _run(cold, [grown], n)[0]
        assert gen._prefix_cache.bytes <= budget


def test_a_tail_from_a_batched_decode_holds_only_its_own_rows(loaded: Generator) -> None:
    prompts = [_new_tokens(30, 5), _new_tokens(31, 3 * CHUNK + 1)]
    n = 4
    gen = _generator(loaded, None)
    outs, _ = _run(gen, prompts, n)
    solo = _generator(loaded, None)
    for p, out in zip(prompts, outs, strict=True):
        assert _run(solo, [p], n)[0] == [out]
        entry = _tail(gen, p + out[:-1])
        assert entry.nbytes == _tail(solo, p + out[:-1]).nbytes
        for layer in entry.cache.layers:
            for value in vars(layer).values():
                for t in value.values() if isinstance(value, dict) else (value,):
                    if isinstance(t, torch.Tensor):
                        assert t._base is None
                        assert t.untyped_storage().nbytes() == t.nbytes
