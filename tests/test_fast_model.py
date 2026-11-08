"""The engine's own forward (``engine.fast``) against the Transformers path, on a tiny checkpoint.

Runs on the CPU with the PyTorch references of the kernels, or with the Triton
kernels under ``TRITON_INTERPRET=1`` when Triton is installed:

    UV_PROJECT_ENVIRONMENT=... uv run --project .vibesys/shared/cpu_check \
        python -m pytest tests/test_fast_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vibesys" / "shared" / "cpu_check"))

from tiny_checkpoint import write_checkpoint  # noqa: E402

from engine.fast import kernels  # noqa: E402
from engine.fast.model import FastModel  # noqa: E402
from engine.generator import Generator  # noqa: E402
from transformers import DynamicCache  # noqa: E402

CHUNK = 8
TRITON = kernels.triton is not None and kernels.INTERPRET


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> Generator:
    path = tmp_path_factory.mktemp("tiny")
    write_checkpoint(path)
    return Generator.load(
        str(path),
        None,
        max_model_len=4096,
        chunk_tokens=CHUNK,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )


def _fast(generator: Generator, **kw) -> FastModel:
    return FastModel(generator.model, slots=4, max_len=4096, use_triton=TRITON, **kw)


def _hf_run(generator: Generator, prompt: list[int], steps: int):
    """Greedy tokens and per-step logits from the Transformers path."""
    cache = DynamicCache(config=generator.model.config)
    with torch.inference_mode():
        logits = generator.prefill(tuple(prompt), 0, cache)
        out, all_logits = [], [logits]
        for _ in range(steps):
            token = int(torch.argmax(logits))
            out.append(token)
            logits = generator.forward([token], cache)
            all_logits.append(logits)
    return out, all_logits


def _fast_prefill(fast: FastModel, slot: int, prompt: list[int], start: int = 0) -> torch.Tensor:
    """Prefill in the same chunks as ``Generator.prefill``."""
    body = len(prompt) - 1
    position = start
    while position < body:
        end = min((position // CHUNK + 1) * CHUNK, body)
        fast.prefill(slot, prompt[position:end], position)
        position = end
    return fast.prefill(slot, prompt[body:], body)


def _close(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


@pytest.mark.parametrize("length", [1, 5, 13, 70])
def test_fast_prefill_and_decode_match_transformers(generator: Generator, length: int) -> None:
    prompt = [(7 * i + 3) % 300 + 5 for i in range(length)]
    tokens, hf_logits = _hf_run(generator, prompt, 12)
    fast = _fast(generator)
    fast.reset(0)
    logits = _fast_prefill(fast, 0, prompt)
    assert _close(logits, hf_logits[0]) < 0.03
    for step, token in enumerate(tokens):
        logits = fast.decode([token], [0], [length + step])[0]
        assert _close(logits, hf_logits[step + 1]) < 0.03, step
        assert int(torch.argmax(logits)) == int(torch.argmax(hf_logits[step + 1]))


def test_batched_decode_matches_each_sequence_alone(generator: Generator) -> None:
    prompts = [[(5 * i + j) % 300 + 5 for i in range(n)] for j, n in enumerate([3, 66, 17])]
    fast = _fast(generator)
    solo = []
    for p in prompts:
        fast.reset(0)
        _fast_prefill(fast, 0, p)
        solo.append([fast.decode([11 + s], [0], [len(p) + s])[0].clone() for s in range(3)])
    for slot in range(3):
        fast.reset(slot + 1)
        _fast_prefill(fast, slot + 1, prompts[slot])
    for step in range(3):
        out = fast.decode(
            [11 + step] * 3, [1, 2, 3], [len(p) + step for p in prompts]
        )
        for row in range(3):
            assert _close(out[row], solo[row][step]) < 1e-2


def test_snapshot_restore_continues_like_a_cold_prefill(generator: Generator) -> None:
    prompt = [(3 * i + 1) % 300 + 5 for i in range(150)]
    fast = _fast(generator)
    fast.reset(0)
    cold = _fast_prefill(fast, 0, prompt)
    fast.reset(0)
    _fast_prefill(fast, 0, prompt[:130])  # covers chunks up to 128 then 129 + last
    fast.reset(0)
    position = 0
    while position < 128:
        fast.prefill(0, prompt[position : position + CHUNK], position)
        position += CHUNK
    snap = fast.snapshot(0, 128)
    fast.reset(0)
    fast.reset(1)
    fast.restore(snap, 1)
    warm = _fast_prefill(fast, 1, prompt, start=128)
    assert torch.equal(warm, cold)
    free = fast.pool.free
    fast.reset(1)
    fast.release(snap)
    assert fast.pool.free > free


@pytest.mark.skipif(kernels.triton is None, reason="needs Triton (TRITON_INTERPRET=1 on the CPU)")
def test_attention_reads_the_last_page_of_the_pool() -> None:
    """Page ids near the pool's end address the right memory (int64 offsets)."""
    torch.manual_seed(0)
    pages, heads, page, dim = 40, 2, 16, 32
    k_pool = torch.randn(pages, heads, page, dim).to(torch.bfloat16)
    v_pool = torch.randn(pages, heads, page, dim).to(torch.bfloat16)
    table = torch.zeros(2, 4, dtype=torch.int32)
    table[0, :3] = torch.tensor([pages - 1, pages - 2, 0])
    table[1, :1] = torch.tensor([pages - 1])
    q = torch.randn(2, 8, dim).to(torch.bfloat16)
    slots = torch.tensor([0, 1])
    lengths = torch.tensor([40, 16], dtype=torch.int32)
    ref = kernels.paged_decode_attention(q, k_pool, v_pool, table, slots, lengths, 0.2, use_triton=False)
    got = kernels.paged_decode_attention(q, k_pool, v_pool, table, slots, lengths, 0.2, use_triton=True)
    assert float((ref.float() - got.float()).abs().max()) < 2e-2
