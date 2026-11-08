"""Exact-greedy MTP speculative decoding (engine.speculative) on a tiny random checkpoint.

The tiny checkpoint gets random ``mtp.*`` tensors, so the real head's drafts are
nearly always rejected; an oracle that drafts the reference's own next token
(or a wrong one on purpose) drives the accept path, the reject path and a
request that ends on an accepted draft. In every case the tokens must equal
plain greedy decoding, and session tails must cover exactly the accepted tokens.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
import torch
from conftest import CHUNK, serve
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from engine.generator import Generation, Generator
from engine.mtp import MTPHead, mtp_config
from engine.protocol import Sampling
from engine.speculative import ENV


@pytest.fixture(scope="module")
def mtp_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tiny_checkpoint import write_checkpoint

    path = tmp_path_factory.mktemp("tiny-qwen36-mtp")
    write_checkpoint(path)
    config = AutoConfig.from_pretrained(path).text_config
    config._attn_implementation = "sdpa"
    torch.manual_seed(3)
    head = MTPHead(mtp_config(config))
    weights = load_file(path / "model.safetensors")
    for name, tensor in head.state_dict().items():
        weights["mtp." + name] = (torch.randn_like(tensor) * 0.2).to(torch.bfloat16).contiguous()
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    return path


def _load(path: Path, monkeypatch: pytest.MonkeyPatch, on: bool, entries: int = 16) -> Generator:
    monkeypatch.setenv(ENV, "1" if on else "0")
    g = Generator.load(
        str(path),
        None,
        max_model_len=4096,
        chunk_tokens=CHUNK,
        prefix_cache_tokens=1 << 20,
        prefix_cache_entries=entries,
    )
    assert (g.spec is not None) == on
    return g


def _sampling(max_tokens: int) -> Sampling:
    return Sampling(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)


def _serve(g: Generator, requests: list[tuple[list[int], int]]) -> list[tuple[list[int], int]]:
    """Run requests together as the server's worker does; (tokens, cached prompt tokens) each."""
    gens: list[Generation] = [g.start(tuple(p), _sampling(n)) for p, n in requests]
    outs: list[list[int]] = [[] for _ in gens]
    live = list(range(len(gens)))
    while live:
        waiting = []
        for i in live:
            step = gens[i].next_step()
            outs[i].append(step.token_id)
            if step.finish is not None:
                gens[i].retire()
            else:
                waiting.append(i)
        live = waiting
        if waiting:
            g.decode([gens[i] for i in waiting])
    return [(o, gen.cached_tokens) for o, gen in zip(outs, gens, strict=True)]


def _prompt(seed: int, n: int) -> list[int]:
    return [7 + (seed * 31 + i * 13) % 250 for i in range(n)]


def _oracle(g: Generator, references: dict[tuple[int, ...], list[int]], wrong_every: int = 0) -> None:
    """Draft the reference continuation's next token (wrong on purpose every ``wrong_every``-th)."""
    original = g.spec._draft
    calls = [0]

    def draft(gen: Generation) -> int:
        original(gen)  # keep the head's own cache in step
        tokens = gen.tokens()
        for prompt, ref in references.items():
            if tuple(tokens[: len(prompt)]) == prompt:
                k = len(tokens) - len(prompt)  # the draft is ref[k]
                calls[0] += 1
                if k < len(ref) and not (wrong_every and calls[0] % wrong_every == 0):
                    return ref[k]
                return (ref[k] + 1) % 250 if k < len(ref) else 0
        raise AssertionError("no reference for this prompt")

    g.spec._draft = draft


def test_real_head_drafts_keep_greedy_output(mtp_checkpoint: Path, monkeypatch) -> None:
    plain = _load(mtp_checkpoint, monkeypatch, on=False)
    spec = _load(mtp_checkpoint, monkeypatch, on=True)
    requests = [(_prompt(1, 2 * CHUNK + 3), 20), (_prompt(2, 5), 9)]
    assert _serve(spec, requests[:1]) == _serve(plain, requests[:1])
    assert _serve(spec, requests[1:]) == _serve(plain, requests[1:])
    assert spec.stats()["mtp_proposed"] > 0


@pytest.mark.parametrize("wrong_every", [0, 3])
def test_accepts_and_rejects_alone_and_batched(mtp_checkpoint: Path, monkeypatch, wrong_every: int) -> None:
    plain = _load(mtp_checkpoint, monkeypatch, on=False, entries=0)
    spec = _load(mtp_checkpoint, monkeypatch, on=True, entries=0)
    requests = [(_prompt(3, CHUNK + 1), 17), (_prompt(4, 3 * CHUNK + 2), 12), (_prompt(5, 4), 2)]
    want = [_serve(plain, [r])[0][0] for r in requests]
    _oracle(spec, {tuple(p): w for (p, _), w in zip(requests, want, strict=True)}, wrong_every)
    for request, expected in zip(requests, want, strict=True):
        assert _serve(spec, [request])[0][0] == expected
    assert [o for o, _ in _serve(spec, requests)] == want
    stats = spec.stats()
    assert stats["mtp_accepted"] > 0
    assert (stats["mtp_accepted"] < stats["mtp_proposed"]) == bool(wrong_every)
    # Accepted drafts save forward passes: fewer decode steps than tokens fed one at a time.
    assert stats["decode_steps"] < sum(n - 1 for _, n in requests) * 2


def test_session_tails_hold_only_accepted_tokens(mtp_checkpoint: Path, monkeypatch) -> None:
    """Rounds resume from tails (some ending on an accepted draft) and match a cold plain server."""
    cold = _load(mtp_checkpoint, monkeypatch, on=False, entries=0)
    spec = _load(mtp_checkpoint, monkeypatch, on=True)
    references: dict[tuple[int, ...], list[int]] = {}
    _oracle(spec, references)
    prompt: list[int] = []
    for index, (new, out) in enumerate([(2 * CHUNK + 5, 7), (CHUNK + 3, 2), (3, 12), (6, 5)]):
        prompt = prompt + _prompt(10 + index, new)
        (expected, _), = _serve(cold, [(prompt, out)])
        references.clear()
        references[tuple(prompt)] = expected
        (got, cached), = _serve(spec, [(prompt, out)])
        assert got == expected
        if index:
            assert cached == len(previous) - 1  # the tail: last prompt and all but its last token
        previous = prompt + got
        prompt = previous
    assert spec.stats()["prefix_cache_tail_hits"] == 3
    assert spec.stats()["mtp_accepted"] > 0


@pytest.fixture(scope="module")
def mtp_front(mtp_checkpoint: Path, tmp_path_factory: pytest.TempPathFactory):
    """Two CPU replicas of the MTP checkpoint behind the router, speculation on by default."""
    patch = pytest.MonkeyPatch()
    patch.delenv(ENV, raising=False)
    patch.setenv("ENGINE_AFFINITY_MIN_TOKENS", "16")  # the tiny prompts are short
    try:
        yield from serve(mtp_checkpoint, tmp_path_factory.mktemp("logs") / "mtp-front.log", "--replicas", "2")
    finally:
        patch.undo()


def _http_completion(url: str, prompt: list[int], max_tokens: int) -> tuple[list[int], dict]:
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with httpx.stream("POST", f"{url}/v1/completions", json=body, timeout=120) as response:
        response.raise_for_status()
        lines = [x for x in response.iter_lines() if x.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    *chunks, last = [json.loads(x.removeprefix("data: ")) for x in lines[:-1]]
    return [t for c in chunks for t in c["choices"][0]["token_ids"]], last["usage"]


def test_served_sessions_through_the_router_match_plain_greedy(
    mtp_front, mtp_checkpoint: Path, monkeypatch
) -> None:
    def session(seed: int) -> list[tuple[list[int], list[int], dict]]:
        prompt = _prompt(seed, 2 * CHUNK + seed)
        rounds = []
        for turn in range(3):
            out, usage = _http_completion(mtp_front.url, prompt, 5 + 3 * turn)
            rounds.append((prompt, out, usage))
            prompt = prompt + out + _prompt(seed * 5 + turn, 4)
        return rounds

    with ThreadPoolExecutor(4) as pool:
        sessions = list(pool.map(session, range(20, 24)))
    plain = _load(mtp_checkpoint, monkeypatch, on=False, entries=0)
    for rounds in sessions:
        for index, (prompt, out, usage) in enumerate(rounds):
            assert out == _serve(plain, [(prompt, len(out))])[0][0]
            assert usage["completion_tokens"] == len(out)
            if index:
                assert usage["prompt_tokens_details"]["cached_tokens"] > 0
    health = httpx.get(f"{mtp_front.url}/health", timeout=10).json()
    assert all(r["mtp"] == 1 for r in health["replicas"])
    assert health["mtp_proposed"] == sum(r["mtp_proposed"] for r in health["replicas"]) > 0
