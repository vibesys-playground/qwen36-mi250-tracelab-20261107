"""Fused decode layers (``engine.fast.fused``) against the per-op decode path, on a tiny checkpoint.

Needs Triton's CPU interpreter (``TRITON_INTERPRET=1``); skipped otherwise.
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

pytestmark = pytest.mark.skipif(
    kernels.triton is None or not kernels.INTERPRET, reason="needs TRITON_INTERPRET=1"
)


@pytest.fixture(scope="module")
def fast(tmp_path_factory: pytest.TempPathFactory) -> FastModel:
    path = tmp_path_factory.mktemp("tiny")
    write_checkpoint(path)
    gen = Generator.load(str(path), None, max_model_len=1024, chunk_tokens=8,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    model = FastModel(gen.model, slots=8, max_len=1024, use_triton=True, use_graphs=False)
    assert model.fused is not None, model.fused_error
    return model


def _state(fast: FastModel):
    gdn = {i: (c.clone(), r.clone()) for i, (c, r) in fast.gdn_state.items()}
    kv = {i: (k.clone(), v.clone()) for i, (k, v) in fast.kv.items()}
    return gdn, kv


def _set(fast: FastModel, saved) -> None:
    gdn, kv = saved
    for i, (c, r) in gdn.items():
        fast.gdn_state[i][0].copy_(c)
        fast.gdn_state[i][1].copy_(r)
    for i, (k, v) in kv.items():
        fast.kv[i][0].copy_(k)
        fast.kv[i][1].copy_(v)


@pytest.mark.parametrize("rows", [1, 3, 5])
def test_each_layer_matches_the_per_op_path(fast: FastModel, rows: int) -> None:
    gen = torch.Generator().manual_seed(rows)
    hidden = fast.embed.shape[1]
    slots = list(range(rows))
    positions = [7 + 13 * r for r in range(rows)]
    for s, p in zip(slots, positions):
        fast._ensure_pages(s, p + 1)
    meta = next(iter(fast.devices.values()))
    slots_t = torch.tensor(slots)
    pos_t = torch.tensor(positions)
    # A non-zero state, so the conv shift and the recurrent decay are exercised.
    for c, r in fast.gdn_state.values():
        c.copy_(torch.randn(c.shape, generator=gen).to(c.dtype))
        r.copy_(torch.randn(r.shape, generator=gen) * 0.1)
    for k, v in fast.kv.values():
        k.copy_(torch.randn(k.shape, generator=gen).to(k.dtype))
        v.copy_(torch.randn(v.shape, generator=gen).to(v.dtype))
    start = _state(fast)
    for i in range(len(fast.layers)):
        x = (torch.randn(rows, hidden, generator=gen) * 2).to(torch.bfloat16)
        _set(fast, start)
        fast.triton_kernels["fused_decode"] = False
        with torch.inference_mode():
            ref = fast.layer_decode(i, x, slots_t, pos_t, meta)
        ref_state = _state(fast)
        _set(fast, start)
        fast.triton_kernels["fused_decode"] = True
        with torch.inference_mode():
            got = fast.layer_decode(i, x, slots_t, pos_t, meta)
        got_state = _state(fast)
        scale = float(ref.float().abs().max())
        diff = float((got.float() - ref.float()).abs().max())
        print("LAYER", rows, i, fast.layer_types[i], "diff", diff, "scale", scale, "exact", bool(torch.equal(got, ref)))
        assert diff <= 0.02 * scale, (i, fast.layer_types[i], diff, scale)
        for a, b in zip(ref_state[0].get(i, ()), got_state[0].get(i, ())):
            assert float((a.float() - b.float()).abs().max()) <= 1e-2, (i, "gdn state")
        for a, b in zip(ref_state[1].get(i, ()), got_state[1].get(i, ())):
            assert float((a.float() - b.float()).abs().max()) <= 0.05, (i, "kv")
    fast.triton_kernels["fused_decode"] = True
    for s in slots:
        fast.reset(s)


def test_greedy_decode_matches_the_per_op_path(fast: FastModel) -> None:
    prompts = [[5, 9, 22, 31, 7, 3, 11], [40, 2, 17, 8, 90, 61, 33, 14, 70]]
    runs = []
    for flag in (False, True):
        fast.triton_kernels["fused_decode"] = flag
        tokens, logits = [], []
        with torch.inference_mode():
            current = []
            for slot, prompt in enumerate(prompts):
                fast.reset(slot)
                current.append(int(fast.prefill(slot, prompt, 0).argmax()))
            for step in range(6):
                out = fast.decode(current, [0, 1], [len(p) + step for p in prompts]).float()
                logits.append(out.clone())
                current = [int(t) for t in out.argmax(-1).tolist()]
                tokens.append(current)
        runs.append((tokens, logits))
    fast.triton_kernels["fused_decode"] = True
    assert runs[0][0] == runs[1][0]
    for a, b in zip(runs[0][1], runs[1][1]):
        assert float((a - b).abs().max()) <= 0.05 * float(a.abs().max()) + 1e-3
