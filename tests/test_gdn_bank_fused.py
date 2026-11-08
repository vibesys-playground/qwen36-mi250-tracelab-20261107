"""The verify step's GDN layer with the state written to both banks inside its kernels
(``ENGINE_GDN_BANK_FUSED``, ``FusedDecode._gdn_verify(bank_fused=True)``) against the old
half-by-half path with ``index_select``/``index_copy_`` bank copies: outputs and both banks
must be bitwise equal, per layer at B=1..4 (either bank current) and through whole verify
steps with accepted and rejected drafts. Needs Triton's CPU interpreter (``TRITON_INTERPRET=1``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / ".vibesys" / "shared" / "cpu_check"))

from engine.fast import fused as fused_mod  # noqa: E402
from engine.fast import kernels  # noqa: E402
from engine.fast.model import FastModel  # noqa: E402
from engine.generator import Generator  # noqa: E402
from engine.mtp import MTPHead  # noqa: E402
from test_speculative import write_mtp_checkpoint  # noqa: E402

pytestmark = pytest.mark.skipif(
    kernels.triton is None or not kernels.INTERPRET, reason="needs TRITON_INTERPRET=1"
)
CHUNK = 8
SLOTS = 4


@pytest.fixture(scope="module")
def fast(tmp_path_factory: pytest.TempPathFactory) -> FastModel:
    path = tmp_path_factory.mktemp("tiny-mtp-banks")
    write_mtp_checkpoint(path)
    gen = Generator.load(str(path), None, max_model_len=1024, chunk_tokens=CHUNK,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    config = gen.model.config
    head = MTPHead.load(str(path), None, getattr(config, "text_config", config), "cpu")
    model = FastModel(gen.model, slots=SLOTS, max_len=1024, use_triton=True, use_graphs=False,
                      mtp=head)
    assert model.fused is not None, model.fused_error
    assert model.triton_kernels["fused_verify"]
    return model


def _gdn_layers(fast: FastModel) -> list[int]:
    return [i for i in range(len(fast.layers)) if fast.layer_types[i] == "linear_attention"]


def _snapshot(fast: FastModel) -> list[torch.Tensor]:
    return [t.clone() for pair in fast.gdn_state.values() for t in pair]


def _restore(fast: FastModel, snap: list[torch.Tensor]) -> None:
    for t, v in zip((t for pair in fast.gdn_state.values() for t in pair), snap, strict=True):
        t.copy_(v)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("current", ["bank0", "bank1", "mixed"])
def test_layer_bank_fused_equals_bank_copies(fast: FastModel, batch: int, current: str) -> None:
    """One GDN layer on random states: the fused path's output and every state row (both
    banks, all slots) equal the old path's bit for bit."""
    torch.manual_seed(batch * 7 + len(current))
    rows = fast.gdn_rows
    for t in (t for pair in fast.gdn_state.values() for t in pair):
        t.copy_((torch.randn(t.shape) * 0.5).to(t.dtype))
    snap = _snapshot(fast)
    slots = list(range(SLOTS - batch, SLOTS))  # not starting at 0
    banks = {"bank0": [0] * batch, "bank1": [1] * batch, "mixed": [s % 2 for s in slots]}[current]
    cur = [s + b * rows for s, b in zip(slots, banks)]
    other = [s + (1 - b) * rows for s, b in zip(slots, banks)]
    states = torch.tensor(cur + other, dtype=torch.long)
    i = _gdn_layers(fast)[0]
    h = (torch.randn((2 * batch, fast.config.hidden_size)) * 0.5).to(fast.embed.dtype)
    outs = []
    for bank_fused in (False, True):
        _restore(fast, snap)
        with torch.inference_mode(), kernels.on_device(h):
            out = fast.fused._gdn_verify(i, h, states, bank_fused=bank_fused)
        outs.append((out.clone(), _snapshot(fast)))
    (oa, sa), (ob, sb) = outs
    assert torch.equal(oa, ob)
    for a, b in zip(sa, sb, strict=True):
        assert torch.equal(a, b)
    # The draft rows' banks really moved (the test is not vacuous).
    conv, rec = fast.gdn_state[i]
    assert not torch.equal(rec[other], snap[2 * list(fast.gdn_state).index(i) + 1][other])


PROMPTS = [[(7 * i + 3 + j) % 300 + 5 for i in range(n)] for j, n in enumerate([13, 30, 9, 21])]


def _prefill(fast: FastModel, slot: int, prompt: list[int]) -> int:
    fast.reset(slot)
    body = len(prompt) - 1
    position = 0
    while position < body:
        end = min((position // CHUNK + 1) * CHUNK, body)
        fast.prefill(slot, prompt[position:end], position, prompt[end])
        position = end
    return int(fast.prefill(slot, prompt[body:], body).argmax())


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_verify_steps_bank_fused_equal_bank_copies(
    fast: FastModel, batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three verify steps per slot with the flag on and off; even slots are fed their true next
    token (accepted, the banks flip), odd slots a wrong one (rejected). Logits, picks, the
    head's drafts, the bank choice and both banks of every slot are bitwise equal."""
    fast.triton_kernels["fused_verify"] = True
    slots = list(range(batch))
    prompts = PROMPTS[:batch]
    steps = 3
    want = []  # greedy continuation of each slot (plain fused decode)
    with torch.inference_mode():
        for s, p in zip(slots, prompts):
            seq = [_prefill(fast, s, p)]
            for k in range(2 * steps + 1):
                seq.append(int(fast.decode([seq[-1]], [s], [len(p) + k])[0].argmax()))
            want.append(seq)
    runs = []
    for bank_fused in (False, True):
        monkeypatch.setattr(fused_mod, "BANK_FUSED", bank_fused)
        trace = []
        with torch.inference_mode():
            lasts = [_prefill(fast, s, p) for s, p in zip(slots, prompts)]
            done = [0] * batch  # tokens of want[s] already emitted after the first
            lengths = [len(p) for p in prompts]
            for _ in range(steps):
                drafts = [want[s][done[s] + 1] if s % 2 == 0 else (want[s][done[s] + 1] + 1) % 300 + 5
                          for s in slots]
                logits, g1, g2, d1, d2 = fast.verify(lasts, drafts, slots, lengths)
                trace.append((logits.clone(), g1, g2, d1, d2, drafts))
                for k, s in enumerate(slots):
                    assert g1[k] == want[s][done[s] + 1]
                    if g1[k] == drafts[k]:
                        fast.flip(s)
                        lasts[k], lengths[k], done[k] = g2[k], lengths[k] + 2, done[k] + 2
                    else:
                        lasts[k], lengths[k], done[k] = g1[k], lengths[k] + 1, done[k] + 1
        runs.append((trace, _snapshot(fast), [fast.bank[s] for s in slots]))
    (ta, sa, ba), (tb, sb, bb) = runs
    assert ba == bb
    accepted = rejected = 0
    for (la, *pa), (lb, *pb) in zip(ta, tb, strict=True):
        assert torch.equal(la, lb)
        assert pa == pb
        g1, _, _, _, fed = pa
        accepted += sum(a == b for a, b in zip(g1, fed))
        rejected += sum(a != b for a, b in zip(g1, fed))
    for a, b in zip(sa, sb, strict=True):
        assert torch.equal(a, b)
    assert accepted == steps * ((batch + 1) // 2)
    assert rejected == steps * (batch // 2)
