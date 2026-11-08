"""Plan 31's chained speculation (``ENGINE_SPEC_CHAIN=1``): the 16-row MFMA GEMV
(``engine.fast.mfma.gemv``), the chained verify step (``FastModel.verify_chain``, its GDN
ring) and the generator's chained path.

- the MFMA GEMV's row outputs do not depend on the row count or the row's place in the tile
  (with and without a K split), and match an fp32 reference to BF16 rounding;
- chained verify steps with drafts right, partly right and wrong reproduce plain greedy
  decode token for token, their first rows' logits equal plain decode's bit for bit, and
  after ``advance`` the slot's state equals the plain path's state bit for bit;
- served through ``Generator`` (two rounds, the second extending the first through session
  tails, drafts forced right at times): greedy outputs equal those without the MTP head.

Needs Triton's CPU interpreter (``TRITON_INTERPRET=1``) on the CPU, or a GPU.
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

from engine.fast import kernels, mfma  # noqa: E402

pytestmark = pytest.mark.skipif(
    kernels.triton is None or not (kernels.INTERPRET or torch.cuda.is_available()),
    reason="needs TRITON_INTERPRET=1 or a GPU",
)
DEV = "cuda" if torch.cuda.is_available() and not kernels.INTERPRET else "cpu"
CHUNK = 8


@pytest.mark.parametrize("shape,tiles", [
    ((40, 64), (16, 32, 4, 1)),
    ((40, 64), (16, 16, 4, 2)),
    ((96, 128), (32, 32, 4, 4)),
])
def test_mfma_gemv_row_invariant(shape, tiles) -> None:
    torch.manual_seed(sum(shape) + sum(tiles))
    n, k = shape
    w = (torch.randn(n, k) * k ** -0.5).to(torch.bfloat16).to(DEV)
    x = torch.randn(16, k).to(torch.bfloat16).to(DEV)
    with kernels.on_device(x):
        ref = mfma.gemv(x, w, tiles=tiles).clone()
        exact = x.float() @ w.float().T
        assert float((ref.float() - exact).abs().max()) <= 2 * float(exact.abs().max()) * 2 ** -8
        for rows in (1, 2, 3, 4, 8):
            assert torch.equal(mfma.gemv(x[:rows].contiguous(), w, tiles=tiles), ref[:rows])
        for perm in ([(i + 5) % 16 for i in range(16)], list(range(15, -1, -1))):
            p = torch.tensor(perm, device=x.device)
            assert torch.equal(mfma.gemv(x[p].contiguous(), w, tiles=tiles), ref[p])
        # more than 16 rows: further launches, same bits per row
        x2 = torch.cat([x, x[:5]]).contiguous()
        assert torch.equal(mfma.gemv(x2, w, tiles=tiles), torch.cat([ref, ref[:5]]))


@pytest.fixture(scope="module")
def chain_on():
    saved = (mfma.CHAIN, mfma.GEMV)
    mfma.CHAIN, mfma.GEMV = True, True
    yield
    mfma.CHAIN, mfma.GEMV = saved


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_speculative import write_mtp_checkpoint

    path = tmp_path_factory.mktemp("tiny-mtp-chain")
    write_mtp_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def fast(checkpoint: Path, chain_on):
    from engine.fast.model import FastModel
    from engine.generator import Generator
    from engine.mtp import MTPHead

    gen = Generator.load(str(checkpoint), None, max_model_len=1024, chunk_tokens=CHUNK,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    config = gen.model.config
    head = MTPHead.load(str(checkpoint), None, getattr(config, "text_config", config), "cpu")
    model = FastModel(gen.model, slots=4, max_len=1024, use_triton=True, use_graphs=False, mtp=head)
    assert model.fused is not None, model.fused_error
    assert model.ring == mfma.RING
    return model


def _prefill(fast, slot: int, prompt: list[int]) -> torch.Tensor:
    fast.reset(slot)
    body = len(prompt) - 1
    position = 0
    while position < body:
        end = min((position // CHUNK + 1) * CHUNK, body)
        fast.prefill(slot, prompt[position:end], position, prompt[end])
        position = end
    return fast.prefill(slot, prompt[body:], body)


def _state(fast, slot: int) -> list[torch.Tensor]:
    row = fast._g(slot)
    return [t[row].clone() for pair in fast.gdn_state.values() for t in pair]


PROMPTS = [[(7 * i + 3 + j) % 300 + 5 for i in range(n)] for j, n in enumerate([13, 30, 9])]


@pytest.mark.parametrize("k", [1, 3, 7])
def test_chain_verify_reproduces_plain_decode(fast, k: int) -> None:
    """Three slots, drafts all right (slot 0), right up to a wrong one (slot 1), all wrong
    (slot 2), for several steps: every emitted token, each step's first-row logits and the
    final state equal plain decode."""
    steps = 3
    total = steps * (k + 1) + 2
    want, want_logits, want_state = [], [], []
    with torch.inference_mode():
        for s, p in enumerate(PROMPTS):
            seq = [int(_prefill(fast, s, p).argmax())]
            lg = []
            for t in range(total):
                out = fast.decode([seq[-1]], [s], [len(p) + t])[0]
                lg.append(out.clone())
                seq.append(int(out.argmax()))
            want.append(seq)
            want_logits.append(lg)
        lasts = [int(_prefill(fast, s, p).argmax()) for s, p in enumerate(PROMPTS)]
        done = [0] * 3  # index in want[s] of the last token fed
        lengths = [len(p) for p in PROMPTS]
        accepted = []
        for _ in range(steps):
            drafts = []
            for s in range(3):
                true = want[s][done[s] + 1 : done[s] + 1 + k]
                if s == 1:
                    true = true[: k // 2] + [(t + 1) % 300 + 5 for t in true[k // 2 :]]
                elif s == 2:
                    true = [(t + 1) % 300 + 5 for t in true]
                drafts.append(true)
            logits, picks, acc, nxt = fast.verify_chain(lasts, drafts, [0, 1, 2], lengths)
            assert len(nxt) == 3 and all(len(d) == k for d in nxt)
            for s in range(3):
                assert torch.equal(logits[s], want_logits[s][done[s]]), (s, done[s])
                a = acc[s]
                assert a == {0: k, 1: k // 2, 2: 0}[s]
                for j in range(a + 1):
                    assert picks[s][j] == want[s][done[s] + 1 + j]
                fast.advance(s, a)
                lasts[s] = picks[s][a]
                lengths[s] += a + 1
                done[s] += a + 1
            accepted.append(acc)
        for s in range(3):
            # plain decode's state after the same tokens
            got = _state(fast, s)
            seq = want[s]
            _prefill(fast, s, PROMPTS[s])
            for t in range(done[s]):
                fast.decode([seq[t]], [s], [len(PROMPTS[s]) + t])
            for a, b in zip(got, _state(fast, s), strict=True):
                assert torch.equal(a, b)


def test_generator_chain_equals_plain(checkpoint: Path, chain_on, monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.generator import Generator
    from engine.protocol import Sampling

    monkeypatch.setenv("ENGINE_FAST", "1")
    monkeypatch.setenv("ENGINE_WARMUP", "0")
    outs, stats = {}, {}
    for setting in ("0", "1"):
        monkeypatch.setenv("ENGINE_MTP", setting)
        for k in (("",) if setting == "0" else ("3", "7", "")):
            monkeypatch.setenv("ENGINE_SPEC_K", k)
            g = Generator.load(str(checkpoint), None, max_model_len=4096, chunk_tokens=CHUNK,
                               prefix_cache_tokens=100000, prefix_cache_entries=16)
            results = []
            prompts = [[(3 * i + j) % 300 + 5 for i in range(n)] for j, n in enumerate([9, 40, 3])]
            for rnd in range(2):
                gens = [g.start(tuple(p), Sampling(max_tokens=13, temperature=0.0, ignore_eos=True))
                        for p in prompts]
                toks = [[] for _ in gens]
                live = list(range(len(gens)))
                while live:
                    waiting = []
                    for i in live:
                        while True:
                            step = gens[i].next_step()
                            toks[i].append(step.token_id)
                            if step.finish is not None:
                                gens[i].retire()
                                break
                            if not gens[i].has_queued():
                                waiting.append(i)
                                break
                    live = waiting
                    if setting == "1":  # the random head never guesses: force drafts at times
                        for i in live:
                            n = len(toks[i])
                            if n % 3 != 2:
                                ref = outs["0"][rnd][i]
                                right = ref[n : n + 7] + [5] * 7
                                if n % 3 == 1:  # right for two, then wrong
                                    right = right[:2] + [(t + 1) % 300 + 5 for t in right[2:]]
                                gens[i].drafts = right[:7]
                                gens[i].draft = right[0]
                    if live:
                        g.decode([gens[i] for i in live])
                results.append(toks)
                prompts = [p + t + [17, 18] for p, t in zip(prompts, toks, strict=True)]
            outs[setting if setting == "0" else f"1-{k}"] = results
            stats[setting if setting == "0" else f"1-{k}"] = g.stats()
    for key, res in outs.items():
        assert res == outs["0"], key
    for key in ("1-3", "1-7"):
        assert stats[key]["mtp_verify_steps"] > 0
        assert stats[key]["mtp_accepted"] > 0
