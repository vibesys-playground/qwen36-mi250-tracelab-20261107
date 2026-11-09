"""Self-speculative decoding on the fast path (``FastModel.verify``, ``engine.speculative``).

A tiny random checkpoint with a random MTP head, on the CPU (PyTorch references):

- a verify step's pick after the first token is plain decode's, and with an
  accepted draft the pick after it is plain decode's next one, so greedy tokens
  driven through verify steps (drafts forced right or wrong) equal plain decode;
- a rejected draft leaves the slot's state exactly one token on (the current bank)
  and an accepted one two tokens on (after ``flip``);
- the head's fast forward equals ``engine.mtp.MTPHead`` (Transformers modules) on
  the same hidden states, prompt pairs included;
- served end to end: greedy outputs with the head on equal those with it off.

    UV_PROJECT_ENVIRONMENT=... uv run --project .vibesys/shared/cpu_check \
        python -m pytest tests/test_speculative.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vibesys" / "shared" / "cpu_check"))

from conftest import load_weights, model_device  # noqa: E402
from tiny_checkpoint import write_checkpoint  # noqa: E402

from engine.fast.model import FastModel  # noqa: E402
from engine.generator import Generator  # noqa: E402
from engine.mtp import MTPHead, mtp_config  # noqa: E402

CHUNK = 8


def _mtp_state(text_config, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    head = MTPHead(mtp_config(text_config))
    return {k: (torch.randn_like(v) * 0.2).to(torch.bfloat16) for k, v in head.state_dict().items()}


def write_mtp_checkpoint(path: Path) -> None:
    """The tiny checkpoint plus random ``mtp.*`` tensors in a second shard, with an index."""
    write_checkpoint(path)
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path)
    text = config.text_config
    text._attn_implementation = "sdpa"
    main = load_file(path / "model.safetensors")
    mtp = {"mtp." + k: v.contiguous() for k, v in _mtp_state(text).items()}
    save_file(mtp, path / "mtp.safetensors", metadata={"format": "pt"})
    weight_map = {k: "model.safetensors" for k in main} | {k: "mtp.safetensors" for k in mtp}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-mtp")
    write_mtp_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def generator(checkpoint: Path) -> Generator:
    return load_weights(
        str(checkpoint),
        max_model_len=4096,
        chunk_tokens=CHUNK,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )


@pytest.fixture(scope="module")
def head(generator: Generator, checkpoint: Path) -> MTPHead:
    config = generator.model.config
    return MTPHead.load(str(checkpoint), None, getattr(config, "text_config", config), model_device(generator))


def _fast(generator: Generator, head: MTPHead | None) -> FastModel:
    return FastModel(generator.model, slots=4, max_len=4096, use_triton=False, mtp=head)


def _prefill(fast: FastModel, slot: int, prompt: list[int]) -> torch.Tensor:
    body = len(prompt) - 1
    position = 0
    while position < body:
        end = min((position // CHUNK + 1) * CHUNK, body)
        fast.prefill(slot, prompt[position:end], position, prompt[end])
        position = end
    return fast.prefill(slot, prompt[body:], body)


def _plain(generator: Generator, prompt: list[int], steps: int) -> list[int]:
    fast = _fast(generator, None)
    last = int(torch.argmax(_prefill(fast, 0, prompt)))
    out = [last]
    for step in range(steps):
        last = int(torch.argmax(fast.decode([last], [0], [len(prompt) + step])[0]))
        out.append(last)
    return out


def _gdn(fast: FastModel, slot: int) -> list[torch.Tensor]:
    row = fast._g(slot)
    return [t[row].clone() for pair in fast.gdn_state.values() for t in pair]


def test_verify_steps_reproduce_plain_greedy_decode(generator: Generator, head: MTPHead) -> None:
    prompts = [[(7 * i + 3 + j) % 300 + 5 for i in range(n)] for j, n in enumerate([13, 70])]
    steps = 14
    want = [_plain(generator, p, 2 * steps + 2) for p in prompts]
    fast = _fast(generator, head)
    lasts, lengths, outs = [], [], []
    for slot, p in enumerate(prompts):
        lasts.append(int(torch.argmax(_prefill(fast, slot, p))))
        lengths.append(len(p))
        outs.append([lasts[-1]])
    accepted = 0
    for step in range(steps):
        # Alternate right and wrong drafts, out of phase between the two requests.
        drafts = []
        for r in range(2):
            truth = want[r][len(outs[r])]
            drafts.append(truth if (step + r) % 2 == 0 else (truth + 1) % 300)
        _, g1, g2, _, _ = fast.verify(lasts, drafts, [0, 1], lengths)
        for r in range(2):
            assert g1[r] == want[r][len(outs[r])], (step, r)
            outs[r].append(g1[r])
            if g1[r] == drafts[r]:
                accepted += 1
                fast.flip(r)
                outs[r].append(g2[r])
                lengths[r] += 2
                lasts[r] = g2[r]
            else:
                lengths[r] += 1
                lasts[r] = g1[r]
    assert accepted == steps
    for r in range(2):
        n = len(outs[r])
        assert outs[r] == want[r][:n]


def test_reject_and_accept_leave_the_state_plain_decode_reaches(
    generator: Generator, head: MTPHead
) -> None:
    prompt = [(5 * i + 2) % 300 + 5 for i in range(21)]
    plain = _fast(generator, head)  # same model, stepped by plain decode
    last = int(torch.argmax(_prefill(plain, 0, prompt)))
    after1 = int(torch.argmax(plain.decode([last], [0], [len(prompt)])[0]))
    one = _gdn(plain, 0)
    plain.decode([after1], [0], [len(prompt) + 1])
    two = _gdn(plain, 0)

    fast = _fast(generator, head)
    _prefill(fast, 0, prompt)
    fast.verify([last], [(after1 + 1) % 300], [0], [len(prompt)])  # rejected
    for a, b in zip(_gdn(fast, 0), one, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-2)

    fast.reset(0)
    _prefill(fast, 0, prompt)
    _, g1, _, _, _ = fast.verify([last], [after1], [0], [len(prompt)])  # accepted
    assert g1[0] == after1
    fast.flip(0)
    for a, b in zip(_gdn(fast, 0), two, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-2)


def test_fast_head_matches_transformers_head(generator: Generator, head: MTPHead) -> None:
    """Prefill pairs, then one verify step's second-row pair: the same logits as MTPHead."""
    prompt = [(11 * i + 1) % 300 + 5 for i in range(19)]
    fast = _fast(generator, head)
    seen: list[torch.Tensor] = []
    original = fast._mtp_logits
    fast._mtp_logits = lambda z: seen.append(original(z).float()) or seen[-1]
    last = int(torch.argmax(_prefill(fast, 0, prompt)))
    prefill_logits = seen[-1][-1]
    _, g1, g2, _, _ = fast.verify([last], [0], [0], [len(prompt)])
    verify_logits = seen[-1]  # rows: (H[L], g1) at L, (H[L+1], g2) at L+1

    # Reference: the Transformers model's final normed hidden states over the sequence the
    # fast head saw, then MTPHead over every pair.
    seq = prompt + [last, 0]  # the verify fed ``last`` then the (wrong) draft 0
    model = generator.model
    with torch.inference_mode():
        ids = torch.tensor([seq], device=model.get_input_embeddings().weight.device)
        out = model.model(input_ids=ids, use_cache=False)
        hidden = out.last_hidden_state
        nxt = seq[1:len(prompt)] + [last, g1[0], g2[0]]
        embed = model.get_input_embeddings()
        lm_head = model.get_output_embeddings()
        positions = torch.arange(len(seq))[None]
        want = head(hidden, torch.tensor([nxt]), positions, head.new_cache(), embed, lm_head, len(seq))
    # Row L-1 is the prefill's last pair; rows L and L+1 the verify's (the draft row past L
    # uses a rejected draft's hidden state, as the fast head does).
    assert float((prefill_logits - want[0, len(prompt) - 1]).abs().max()) < 0.05 * float(
        want.abs().max()
    )
    torch.testing.assert_close(verify_logits[0], want[0, len(prompt)], rtol=0.05, atol=0.05 * float(want.abs().max()))
    torch.testing.assert_close(verify_logits[1], want[0, len(prompt) + 1], rtol=0.05, atol=0.05 * float(want.abs().max()))


def test_generator_greedy_outputs_equal_with_and_without_mtp(
    generator: Generator, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from engine.protocol import Sampling

    monkeypatch.setenv("ENGINE_FAST", "1")
    monkeypatch.setenv("ENGINE_WARMUP", "0")
    outs = {}
    stats = {}
    for setting in ("0", "1"):
        monkeypatch.setenv("ENGINE_MTP", setting)
        g = Generator.load(
            str(checkpoint), None, max_model_len=4096, chunk_tokens=CHUNK,
            prefix_cache_tokens=100000, prefix_cache_entries=16,
        )
        assert (g.fast.mtp is not None) == (setting == "1")
        results = []
        prompts = [[(3 * i + j) % 300 + 5 for i in range(n)] for j, n in enumerate([9, 40, 3])]
        # Two rounds, the second extending the first (session tails), all three decoding together.
        for rnd in range(2):
            gens = [g.start(tuple(p), Sampling(max_tokens=11, temperature=0.0, ignore_eos=True)) for p in prompts]
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
                if setting == "1":  # the random head never guesses: force some right drafts
                    for i in live:
                        if len(toks[i]) % 2 == 0:
                            gens[i].draft = outs["0"][rnd][i][len(toks[i])]
                if live:
                    g.decode([gens[i] for i in live])
            results.append(toks)
            prompts = [p + t + [17, 18] for p, t in zip(prompts, toks, strict=True)]
        outs[setting] = results
        stats[setting] = g.stats()
    assert outs["0"] == outs["1"]
    assert all(len(t) == 11 for r in outs["1"] for t in r)
    assert stats["1"]["mtp_verify_steps"] > 0
    assert stats["1"]["mtp_accepted"] > 0
    assert stats["1"]["prefix_cache_tail_hits"] == stats["0"]["prefix_cache_tail_hits"] == 3


def test_parked_session_tails_equal_detached_ones_with_mtp(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 69: with the MTP head (state rings, banks left wherever the last verify put
    them), a round resumed in its parked slot gives the same tokens, cached counts and first
    logits as one resumed from a detached tail."""
    from engine.protocol import Sampling

    monkeypatch.setenv("ENGINE_FAST", "1")
    monkeypatch.setenv("ENGINE_WARMUP", "0")
    monkeypatch.setenv("ENGINE_MTP", "1")
    runs = {}
    for park in ("0", "1"):
        monkeypatch.setenv("ENGINE_SLOT_PARK", park)
        g = Generator.load(
            str(checkpoint), None, max_model_len=4096, chunk_tokens=CHUNK,
            prefix_cache_tokens=100000, prefix_cache_entries=16,
        )
        assert (g._prefix_cache.park_max > 0) == (park == "1")
        prompts = [[(5 * i + j) % 300 + 5 for i in range(n)] for j, n in enumerate([9, 70, 33])]
        toks_all, cached_all, logits_all = [], [], []
        for rnd in range(4):
            sampling = Sampling(max_tokens=7 + 3 * rnd, temperature=0.0, ignore_eos=True)
            gens = g.start_many([(tuple(p), sampling) for p in prompts])
            cached_all.append([x.cached_tokens for x in gens])
            logits_all.append([x.logits.clone() for x in gens])
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
                for i in live:  # force some right drafts (the random head never guesses)
                    if len(toks[i]) % 2 == 0 and park == "1":
                        gens[i].draft = runs["0"][0][rnd][i][len(toks[i])]
                    elif len(toks[i]) % 2 == 0:
                        gens[i].draft = toks[i][-1]
                if live:
                    g.decode([gens[i] for i in live])
            toks_all.append(toks)
            prompts = [p + t + [17, 18 + rnd] for p, t in zip(prompts, toks, strict=True)]
        runs[park] = (toks_all, cached_all, logits_all, g.stats())
    assert runs["0"][0] == runs["1"][0]
    assert runs["0"][1] == runs["1"][1]
    assert all(c > 0 for r in runs["1"][1][1:] for c in r)
    for la, lb in zip(runs["0"][2], runs["1"][2], strict=True):
        for x, y in zip(la, lb, strict=True):
            assert torch.equal(x, y)
    assert runs["1"][3]["parked_hits"] == 9 and runs["0"][3]["parked_hits"] == 0
    assert runs["1"][3]["mtp_accepted"] > 0
