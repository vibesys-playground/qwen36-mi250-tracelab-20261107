"""Packed prefill (``FastModel.prefill_packed``, ``Generator.start_many``) against serial prefill.

Tiny random-weight checkpoint on the CPU, PyTorch references of the kernels
(or Triton under ``TRITON_INTERPRET=1``):

    UV_PROJECT_ENVIRONMENT=... uv run --project .vibesys/shared/cpu_check \
        python -m pytest tests/test_prefill_packed.py
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
from engine.generator import Generator, PackedPrefill  # noqa: E402
from engine.protocol import Sampling  # noqa: E402

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


def _fast(generator: Generator) -> FastModel:
    return FastModel(generator.model, slots=6, max_len=4096, use_triton=TRITON)


def _close(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def _prompt(n: int, seed: int) -> list[int]:
    return [(7 * i + 13 * seed + 3) % 300 + 5 for i in range(n)]


def test_packed_prefill_matches_serial(generator: Generator) -> None:
    """Segments with fresh and carried state, one-token segments, and a 70-token one."""
    fast = _fast(generator)
    prompts = [_prompt(n, s) for s, n in enumerate([20, 70, 9, 33, 1])]
    # Slots 0-4 first get a prefix (so state and KV carry into the segment), except slot 4.
    prefix = [5, 30, 0, 17, 0]

    def setup() -> None:
        for slot, (p, k) in enumerate(zip(prompts, prefix, strict=True)):
            fast.reset(slot)
            if k:
                fast.prefill(slot, p[:k], 0)

    setup()
    serial = [fast.prefill(s, p[k:], k) for s, (p, k) in enumerate(zip(prompts, prefix, strict=True))]
    # One decode step on top: checks conv, recurrent state and KV the prefill left behind.
    serial_next = fast.decode([11] * 5, list(range(5)), [len(p) for p in prompts]).clone()
    setup()
    packed = fast.prefill_packed([(s, p[k:], k) for s, (p, k) in enumerate(zip(prompts, prefix, strict=True))])
    packed_next = fast.decode([11] * 5, list(range(5)), [len(p) for p in prompts]).clone()
    assert packed.shape[0] == 5
    for row in range(5):
        assert _close(packed[row], serial[row]) < 2e-2, row
        assert int(packed[row].argmax()) == int(serial[row].argmax())
        assert _close(packed_next[row], serial_next[row]) < 2e-2, row
        assert int(packed_next[row].argmax()) == int(serial_next[row].argmax())


def test_start_many_matches_start(generator: Generator, monkeypatch: pytest.MonkeyPatch) -> None:
    fast = _fast(generator)
    generator.fast = fast
    generator._free_slots = list(range(fast.slots - 1, -1, -1))
    generator._released = []
    from engine.generator import FastPrefixCache

    generator._prefix_cache = FastPrefixCache(fast, CHUNK, 4096, 16)
    try:
        sampling = Sampling(max_tokens=4, temperature=0.0, ignore_eos=True)
        prompts = [tuple(_prompt(n, s)) for s, n in enumerate([21, 3, 40, 1, 17])]
        serial = []
        for p in prompts:
            g = generator.start(p, sampling)
            serial.append((g.logits.clone(), g.cached_tokens))
            generator.finished(g)
        generator._reclaim()
        # A second time packed: every prompt now hits the snapshots the serial run stored.
        monkeypatch.setenv("ENGINE_PREFILL_PACK_TOKENS", "24")
        results = generator.start_many([(p, sampling) for p in prompts] + [((), sampling)])
        assert isinstance(results[-1], Exception)
        for (logits, _), g, p in zip(serial, results[:-1], prompts, strict=True):
            assert not isinstance(g, Exception), g
            assert _close(g.logits, logits) < 2e-2
            assert int(g.logits.argmax()) == int(logits.argmax())
            body = len(p) - 1
            expect = body if body > 0 else 0
            assert g.cached_tokens == expect, (len(p), g.cached_tokens)
            generator.finished(g)
        generator._reclaim()
        # Cold, packed, then decode a few greedy tokens: equal to the serial path's tokens.
        generator._prefix_cache = FastPrefixCache(fast, CHUNK, 4096, 16)
        packer = PackedPrefill(generator)
        tickets = [packer.add(p, sampling) for p in prompts]
        done = dict(packer.failed())
        passes = 0
        while packer.busy:
            done.update(packer.step())
            passes += 1
        gens = [done[t] for t in tickets]
        assert passes < sum(len(generator._chunks(p, 0)) for p in prompts)
        for (logits, _), g in zip(serial, gens, strict=True):
            assert int(g.logits.argmax()) == int(logits.argmax())
            assert g.cached_tokens == 0
    finally:
        generator.fast = None
