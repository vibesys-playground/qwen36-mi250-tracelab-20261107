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


def test_packed_gdn_varlen_kernel_path_matches_the_padded_path(
    generator: Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a varlen chunk kernel bound (fla under ENGINE_GDN=fla), the packed GDN layers pass
    the concatenated segments with ``cu_seqlens`` and per-segment initial states. Here a torch
    stand-in for fla's varlen kernel runs each span on its own, so the plumbing is checked on
    the CPU against the padded path."""
    from engine import gdn
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    torch_chunk = hf.torch_chunk_gated_delta_rule
    calls = []

    def varlen(q, k, v, g, beta, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
        if cu_seqlens is None:  # a single sequence's prefill (``FastModel.prefill``)
            return torch_chunk(q, k, v, g=g, beta=beta, initial_state=initial_state,
                               output_final_state=output_final_state,
                               use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        assert q.shape[0] == 1
        assert initial_state.shape[0] == len(cu_seqlens) - 1
        calls.append(len(cu_seqlens) - 1)
        cores, finals = [], []
        bounds = cu_seqlens.tolist()
        for n, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            core, final = torch_chunk(
                q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g=g[:, lo:hi], beta=beta[:, lo:hi],
                initial_state=initial_state[n : n + 1], output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            cores.append(core)
            finals.append(final)
        return torch.cat(cores, dim=1), torch.cat(finals, dim=0)

    fast = _fast(generator)
    prompts = [_prompt(n, s) for s, n in enumerate([20, 70, 9, 1])]
    prefix = [5, 30, 0, 0]
    segments = [(s, p[k:], k) for s, (p, k) in enumerate(zip(prompts, prefix, strict=True))]

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        for slot, (p, k) in enumerate(zip(prompts, prefix, strict=True)):
            fast.reset(slot)
            if k:
                fast.prefill(slot, p[:k], 0)
        logits = fast.prefill_packed(segments)
        after = fast.decode([11] * 4, list(range(4)), [len(p) for p in prompts]).clone()
        return logits, after

    padded = run()
    monkeypatch.setattr(gdn, "fla_chunk", varlen)
    monkeypatch.setattr(hf, "torch_chunk_gated_delta_rule", varlen)
    packed = run()
    assert calls and all(n == 3 for n in calls)  # the three longer segments, one call per layer
    for a, b in zip(packed, padded, strict=True):
        assert _close(a, b) < 1e-3
        assert torch.equal(a.argmax(-1), b.argmax(-1))


def test_packed_rounds_resume_from_session_tails(generator: Generator) -> None:
    """Every waiting request is looked up first (a tail is adopted into its slot), then packed."""
    from engine.generator import FastPrefixCache

    fast = _fast(generator)
    generator.fast = fast
    generator._free_slots = list(range(fast.slots - 1, -1, -1))
    generator._released = []
    generator._prefix_cache = FastPrefixCache(fast, CHUNK, 1 << 20, 64)
    try:
        sampling = Sampling(max_tokens=3, temperature=0.0, ignore_eos=True)
        prompts = [tuple(_prompt(n, s)) for s, n in enumerate([21, 13, 30])]
        for round_ in range(3):
            gens = generator.start_many([(p, sampling) for p in prompts])
            for g, p in zip(gens, prompts, strict=True):
                assert not isinstance(g, Exception), g
                if round_:
                    assert g.cached_tokens == len(p) - 3  # previous prompt + 2 of its 3 outputs
            live = list(gens)
            outputs = [[] for _ in gens]
            while live:
                for g, out in zip(gens, outputs, strict=True):
                    if g in live:
                        step = g.next_step()
                        out.append(step.token_id)
                        if step.finish is not None:
                            g.retire()
                            live.remove(g)
                if live:
                    generator.decode(live)
            prompts = [p + tuple(o) + tuple(_prompt(2, 9 + round_)) for p, o in zip(prompts, outputs)]
        assert generator._prefix_cache.tail_hits == 2 * len(prompts)
    finally:
        generator.fast = None
