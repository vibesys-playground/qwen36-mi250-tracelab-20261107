"""Verify steps (``FastModel.verify``, two rows per slot) and the MTP head through the fused
decode layers (``engine.fast.fused``) against the per-op kernels, on a tiny checkpoint with a
random MTP head. Needs Triton's CPU interpreter (``TRITON_INTERPRET=1``); skipped otherwise.
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

from engine.fast import kernels  # noqa: E402
from engine.fast.model import FastModel  # noqa: E402
from engine.generator import Generator  # noqa: E402
from engine.mtp import MTPHead  # noqa: E402
from test_speculative import write_mtp_checkpoint  # noqa: E402

pytestmark = pytest.mark.skipif(
    kernels.triton is None or not kernels.INTERPRET, reason="needs TRITON_INTERPRET=1"
)
CHUNK = 8


@pytest.fixture(scope="module")
def fast(tmp_path_factory: pytest.TempPathFactory) -> FastModel:
    path = tmp_path_factory.mktemp("tiny-mtp")
    write_mtp_checkpoint(path)
    gen = Generator.load(str(path), None, max_model_len=1024, chunk_tokens=CHUNK,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    config = gen.model.config
    head = MTPHead.load(str(path), None, getattr(config, "text_config", config), "cpu")
    model = FastModel(gen.model, slots=4, max_len=1024, use_triton=True, use_graphs=False,
                      mtp=head)
    assert model.fused is not None, model.fused_error
    assert model.triton_kernels["fused_verify"]
    return model


def _prefill(fast: FastModel, slot: int, prompt: list[int]) -> int:
    fast.reset(slot)
    body = len(prompt) - 1
    position = 0
    while position < body:
        end = min((position // CHUNK + 1) * CHUNK, body)
        fast.prefill(slot, prompt[position:end], position, prompt[end])
        position = end
    return int(fast.prefill(slot, prompt[body:], body).argmax())


def _banks(fast: FastModel, slot: int) -> list[torch.Tensor]:
    rows = (slot, slot + fast.gdn_rows)
    return [t[r].clone() for pair in fast.gdn_state.values() for t in pair for r in rows]


PROMPTS = [[(7 * i + 3 + j) % 300 + 5 for i in range(n)] for j, n in enumerate([13, 30, 9])]


@pytest.mark.parametrize("accept", [False, True])
def test_a_fused_verify_step_matches_the_per_op_one(fast: FastModel, accept: bool) -> None:
    """Logits, picks, drafts, both GDN banks and the KV after one verify step of 3 slots."""
    results = []
    for fused in (False, True):
        fast.triton_kernels["fused_verify"] = fused
        lasts = [_prefill(fast, s, p) for s, p in enumerate(PROMPTS)]
        drafts = [fast.last_drafts[0].item() if accept else 1] * 3
        if accept:  # the true next token (plain decode's pick) as the draft
            drafts = []
            for s, p in enumerate(PROMPTS):
                fast.triton_kernels["fused_decode"] = False
                saved = _banks(fast, s)
                drafts.append(int(fast.decode([lasts[s]], [s], [len(p)])[0].argmax()))
                # undo that decode: restore the current bank and rewrite nothing else read later
                for t, v in zip((t for pair in fast.gdn_state.values() for t in pair), saved[::2]):
                    t[fast._g(s)].copy_(v)
                fast.triton_kernels["fused_decode"] = True
        with torch.inference_mode():
            logits, g1, g2, d1, d2 = fast.verify(lasts, drafts, [0, 1, 2], [len(p) for p in PROMPTS])
        state = [_banks(fast, s) for s in range(3)]
        kv = {i: (k.clone(), v.clone()) for i, (k, v) in fast.kv.items()}
        results.append((logits.clone(), g1, g2, d1, d2, drafts, state, kv))
    fast.triton_kernels["fused_verify"] = True
    (la, *pa, sa, ka), (lb, *pb, sb, kb) = results
    # The target's picks after each token, and the drafts fed. The head's next drafts (d1, d2)
    # may differ: the fused final norm rounds the head's input hidden differently from the
    # per-op one, which can flip the random tiny head's near-ties; drafts only propose.
    g1a, g2a, _, _, fed_a = pa
    g1b, g2b, _, _, fed_b = pb
    assert (g1a, g2a, fed_a) == (g1b, g2b, fed_b)
    if accept:
        assert g1a == fed_a  # every draft accepted
    assert float((la - lb).abs().max()) <= 0.02 * float(la.abs().max()) + 1e-3
    for x, y in zip(sa, sb, strict=True):
        for a, b in zip(x, y, strict=True):
            assert float((a.float() - b.float()).abs().max()) <= 1e-2
    for i in ka:
        for a, b in zip(ka[i], kb[i], strict=True):
            # BF16 KV entries: one ulp at the entries' scale (the MTP head's layer is large).
            tol = 0.01 * float(a.float().abs().max()) + 1e-3
            assert float((a.float() - b.float()).abs().max()) <= tol, i


def test_greedy_through_fused_verify_steps_equals_plain_decode(fast: FastModel) -> None:
    prompt = PROMPTS[1]
    steps = 8
    fast.triton_kernels["fused_verify"] = True
    last = _prefill(fast, 0, prompt)
    want = [last]
    with torch.inference_mode():
        for step in range(2 * steps + 2):
            want.append(int(fast.decode([want[-1]], [0], [len(prompt) + step])[0].argmax()))
    last = _prefill(fast, 0, prompt)
    out, length, draft = [last], len(prompt), int(fast.last_drafts[0])
    with torch.inference_mode():
        for step in range(steps):
            _, g1, g2, d1, d2 = fast.verify([last], [draft], [0], [length])
            out.append(g1[0])
            if g1[0] == draft:
                fast.flip(0)
                out.append(g2[0])
                length, last, draft = length + 2, g2[0], d2[0]
            else:
                length, last, draft = length + 1, g1[0], d1[0]
    assert out == want[: len(out)]


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_fused_verify_logits_equal_fused_decode_logits(fast: FastModel, batch: int) -> None:
    """The fused GEMV, final norm and lm_head sum each row the same way whatever the row count,
    so a verify step (2 rows per slot) gives each slot's first token bit-identical logits to a
    decode step (1 row per slot), and its draft row those of the next decode step."""
    fast.triton_kernels["fused_verify"] = True
    prompts = PROMPTS[:batch]
    slots = list(range(batch))
    with torch.inference_mode():
        lasts = [_prefill(fast, s, p) for s, p in zip(slots, prompts)]
        lengths = [len(p) for p in prompts]
        first = fast.decode(lasts, slots, lengths).clone()
        nexts = [int(t) for t in first.argmax(-1).tolist()]
        second = fast.decode(nexts, slots, [n + 1 for n in lengths]).clone()
        lasts = [_prefill(fast, s, p) for s, p in zip(slots, prompts)]
        logits, g1, g2, _, _ = fast.verify(lasts, nexts, slots, lengths)
    assert torch.equal(logits.float(), first.float())
    assert g1 == nexts
    assert g2 == [int(t) for t in second.argmax(-1).tolist()]
