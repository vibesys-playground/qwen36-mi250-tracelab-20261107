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
from conftest import CHUNK, Server, load_on_cpu, load_weights, repeat_reuse
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
    assert usage["prompt_tokens_details"]["cached_tokens"] == repeat_reuse(len(prompt))


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
    return load_weights(
        str(checkpoint),
        max_model_len=4096,
        chunk_tokens=CHUNK,
        prefix_cache_tokens=1 << 20,
        prefix_cache_entries=0,
    )


@pytest.fixture(scope="module")
def loaded_cpu(checkpoint: Path, loaded: Generator) -> Generator:
    """``loaded``, on the CPU. A Transformers-path decode step of two requests together equals
    each alone exactly only on the CPU: on a GPU the BF16 GEMMs of one and two rows run other
    kernels, and a one-ulp difference can flip the tiny random model's top-2-of-8 expert routing
    or a top-1 tie (plan 74, j340: logits 2.5 apart at a step whose solo top-2 margin was 0)."""
    if not any(p.is_cuda for p in loaded.model.parameters()):
        return loaded
    torch.manual_seed(0)
    return load_on_cpu(
        str(checkpoint),
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


def test_a_tail_from_a_batched_decode_holds_only_its_own_rows(loaded_cpu: Generator) -> None:
    loaded = loaded_cpu
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


# The engine's own forward (engine.fast): a tail is the finished request's slot state.


def _fast_generator(
    model: Generator, budget: int | None, entries: int = 64, park: int = 0, slots: int = 4
) -> Generator:
    """``park``: tails parked in their slots at most (plan 69); 0 detaches every tail."""
    from engine.fast.model import FastModel

    fast = FastModel(model.model, slots=slots, max_len=4096, use_triton=False, use_graphs=False)
    gen = Generator(
        model.model,
        model.tokenizer,
        model.eos_ids,
        max_model_len=4096,
        chunk_tokens=CHUNK,
        prefix_cache_tokens=1 << 20,
        prefix_cache_entries=entries,
        prefix_cache_bytes=budget,
        fast=fast,
    )
    gen._prefix_cache.park_max = park
    return gen


def test_fast_path_rounds_resume_from_the_slot_state(loaded: Generator) -> None:
    gen = _fast_generator(loaded, None)
    cold = _fast_generator(loaded, None, entries=0)
    free_at_start = gen.fast.pool.free
    sessions = {s: _new_tokens(50 + s, 2 * CHUNK + 5) for s in range(2)}
    outputs: dict[int, list[int]] = {}
    hits = 0
    for index, (new, n) in enumerate(_ROUNDS):
        prompts = [
            sessions[s] if index == 0 else sessions[s] + outputs[s] + _new_tokens(60 + s + index, new)
            for s in sessions
        ]
        outs, cached = _run(gen, prompts, n)  # both sessions decode together
        for s, prompt, out, c in zip(sessions, prompts, outs, cached, strict=True):
            if index:
                assert c == len(sessions[s]) + len(outputs[s]) - 1
                hits += 1
            assert out == _run(cold, [prompt], n)[0][0]
            assert cold._prefix_cache.tail_hits == 0
            sessions[s], outputs[s] = prompt, out
            tail = _tail(gen, prompt + out[:-1])
            assert tail.snap.length == len(prompt) + len(out) - 1
            assert tail.nbytes == gen.fast.snapshot_bytes(tail.snap) > 0
    assert gen._prefix_cache.tail_hits == hits == 2 * (len(_ROUNDS) - 1)
    # Every page comes back once the cache lets go of its entries.
    gen._reclaim()
    while gen._prefix_cache.evict_one():
        pass
    assert gen.fast.pool.free == free_at_start


def test_fast_path_tails_respect_the_byte_budget(loaded: Generator) -> None:
    prompts = [_new_tokens(70 + s, 2 * CHUNK + 3) for s in range(3)]
    n = 6
    probe = _fast_generator(loaded, None)
    outs, _ = _run(probe, [prompts[0]], n)
    tail_bytes = _tail(probe, prompts[0] + outs[0][:-1]).nbytes
    gen = _fast_generator(loaded, int(2.5 * tail_bytes), entries=64)
    for p in prompts:
        _run(gen, [p], n)
        assert gen._prefix_cache.bytes <= int(2.5 * tail_bytes)
    assert gen._prefix_cache.evictions > 0


# Plan 69: a session tail parked in its own slot (no clone, no adopt copy) must resume exactly
# as a detached tail does: the same state, logits, tokens and cached-token counts.


def _rounds(gen: Generator, sessions: dict[int, list[int]], plan: list[tuple[int, int]],
            packed: bool) -> tuple[list, list, list]:
    """Rounds of all ``sessions`` together (round r+1 = prompt + output + new tokens); per
    round the outputs, cached counts and each request's first logits."""
    outs_all, cached_all, logits_all = [], [], []
    prompts = dict(sessions)
    for index, (new, n) in enumerate(plan):
        sampling = Sampling(max_tokens=n, temperature=0.0, ignore_eos=True)
        keys = sorted(prompts)
        reqs = [(tuple(prompts[s]), sampling) for s in keys]
        live = gen.start_many(reqs) if packed else [gen.start(p, smp) for p, smp in reqs]
        assert not any(isinstance(g, Exception) for g in live), live
        logits_all.append([g.logits.clone() for g in live])
        cached_all.append([g.cached_tokens for g in live])
        outputs: list[list[int]] = [[] for _ in keys]
        running = list(range(len(keys)))
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
        outs_all.append(outputs)
        for i, s in enumerate(keys):
            prompts[s] = prompts[s] + outputs[i] + _new_tokens(90 + 7 * s + index, new)
    return outs_all, cached_all, logits_all


def _slot_state(gen: Generator, slot: int, length: int):
    fast = gen.fast
    row = fast._g(slot)
    gdn = [(c[row].clone(), r[row].clone()) for c, r in fast.gdn_state.values()]
    return gdn, _kv(gen, fast.slot_pages[slot], length)


def _kv(gen: Generator, pages: list[int], length: int) -> list[torch.Tensor]:
    from engine.fast.model import PAGE

    out = []
    for k_pool, v_pool in gen.fast.kv.values():
        for pool in (k_pool, v_pool):
            rows = [pool[p][:, : min(PAGE, length - j * PAGE)] for j, p in enumerate(pages)
                    if j * PAGE < length]
            out.append(torch.cat(rows, dim=1))
    return out


@pytest.mark.parametrize("packed", [False, True])
def test_parked_tails_resume_bitwise_as_detached_tails(loaded: Generator, packed: bool) -> None:
    plan = [(0, 9), (2 * CHUNK + 3, 5), (1, 1), (70, 12), (3, 4)]
    sessions = {s: _new_tokens(80 + s, 60 + 9 * s) for s in range(3)}  # tails cross pages
    parked = _fast_generator(loaded, None, park=4, slots=6)
    detached = _fast_generator(loaded, None, park=0, slots=6)
    a = _rounds(parked, sessions, plan, packed)
    b = _rounds(detached, sessions, plan, packed)
    assert a[0] == b[0]  # tokens
    assert a[1] == b[1]  # cached prompt tokens, every round
    assert all(c > 0 for c in a[1][1])
    for la, lb in zip(a[2], b[2], strict=True):
        for x, y in zip(la, lb, strict=True):
            assert torch.equal(x, y)  # first logits after each prefill, bit for bit
    pc = parked._prefix_cache
    assert pc.parked_hits == 3 * (len(plan) - 1) == detached._prefix_cache.tail_hits
    assert pc.unparks == 0 and detached._prefix_cache.parked_hits == 0
    # The last round's tails: parked state == detached snapshot, GDN rows and KV, exactly.
    tails = {k: e for k, e in pc._entries.items() if e.tail}
    assert len(tails) == 3 and all(e.slot is not None for e in tails.values())
    for key, entry in tails.items():
        snap = detached._prefix_cache._entries[key].snap
        gdn, kv = _slot_state(parked, entry.slot, len(key))
        for (c1, r1), (c2, r2) in zip(gdn, snap.gdn, strict=True):
            assert torch.equal(c1, c2) and torch.equal(r1, r2)
        for x, y in zip(kv, _kv(detached, snap.pages, len(key)), strict=True):
            assert torch.equal(x, y)


def test_a_parked_tail_never_keeps_a_request_out(loaded: Generator) -> None:
    """All slots parked: a new session takes the oldest parked slot (detached on demand),
    and the session whose tail that was resumes from the detached snapshot."""
    gen = _fast_generator(loaded, None, park=3, slots=3)
    cold = _fast_generator(loaded, None, entries=0, slots=3)
    free_at_start = gen.fast.pool.free
    prompts = {s: _new_tokens(110 + s, 40 + 30 * s) for s in range(5)}
    outs = {}
    for s in range(3):
        outs[s] = _run(gen, [prompts[s]], 5)[0][0]
    pc = gen._prefix_cache
    assert pc.parked == 3 and not gen._free_slots and not gen._released
    out, cached = _run(gen, [prompts[3]], 5)  # needs a slot: session 0's is unparked
    assert out[0] == _run(cold, [prompts[3]], 5)[0][0] and cached == [0]
    assert pc.unparks == 1
    for s in (0, 1):  # 0 resumes from its detached snapshot, 1 from its parked slot
        grown = prompts[s] + outs[s] + _new_tokens(120 + s, 11)
        out, cached = _run(gen, [grown], 6)
        assert cached == [len(prompts[s]) + 4]
        assert out[0] == _run(cold, [grown], 6)[0][0]
    # Two requests at once, more than the free slots: both served.
    many = [prompts[4], prompts[2] + outs[2] + [5, 6, 7]]
    got, cached = _run(gen, many, 4)
    assert got == [_run(cold, [p], 4)[0][0] for p in many]
    assert cached == [0, len(prompts[2]) + 4]
    # Every page comes back once the cache lets go of its entries (parked ones included).
    gen._reclaim()
    while pc.evict_one():
        pass
    gen._reclaim()
    assert pc.parked == 0 and sorted(gen._free_slots) == [0, 1, 2]
    assert gen.fast.pool.free == free_at_start


def test_parked_tails_respect_budgets_and_page_pressure(loaded: Generator) -> None:
    """Evicting a parked tail (entry count or KV pages) frees its pages and its slot."""
    gen = _fast_generator(loaded, None, entries=2, park=4, slots=6)
    cold = _fast_generator(loaded, None, entries=0, slots=6)
    prompts = [_new_tokens(130 + s, 70) for s in range(4)]
    outs = [_run(gen, [p], 3)[0][0] for p in prompts]
    pc = gen._prefix_cache
    assert len(pc) <= 2 and pc.parked <= 2 and pc.evictions >= 2
    gen._reclaim()
    assert len(gen._free_slots) == 6 - pc.parked
    # Page pressure: take pages until the pool evicts the parked tails.
    fast = gen.fast
    taken = []
    while pc.parked:
        taken.append(fast.pool.alloc())
    for page in taken:
        fast.pool.unref(page)
    gen._reclaim()
    assert sorted(gen._free_slots) == list(range(6))
    grown = prompts[3] + outs[3] + [9]
    out, cached = _run(gen, [grown], 4)
    assert cached == [0] and out[0] == _run(cold, [grown], 4)[0][0]
