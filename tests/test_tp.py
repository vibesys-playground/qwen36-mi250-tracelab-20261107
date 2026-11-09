"""Tensor parallelism over two ranks (``engine.fast.tp``, ENGINE_TP=2) against the one-device
fast path, on a tiny checkpoint with a random MTP head. The ranks run on two GPUs when two are
visible, else both on the CPU (the eager all-reduce through the host mailbox); the in-graph
all-reduce needs two GPUs.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / ".vibesys" / "shared" / "cpu_check"))

from engine.fast import tp  # noqa: E402
from engine.fast.model import FastModel  # noqa: E402
from engine.mtp import MTPHead  # noqa: E402
from conftest import load_on_cpu, load_weights, model_device  # noqa: E402
from test_speculative import write_mtp_checkpoint  # noqa: E402

CHUNK = 8
CPU = [torch.device("cpu"), torch.device("cpu")]
TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
RANKS = [torch.device("cuda", 0), torch.device("cuda", 1)] if TWO_GPUS else CPU


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-mtp-tp")
    write_mtp_checkpoint(path)
    return path


def _load(checkpoint: Path, parallel: bool):
    # Without two GPUs the ranks run on the CPU, and the one-device model with them.
    load = load_weights if TWO_GPUS else load_on_cpu
    gen = load(str(checkpoint), max_model_len=1024, chunk_tokens=CHUNK,
               prefix_cache_tokens=0, prefix_cache_entries=0)
    config = gen.model.config
    head = MTPHead.load(str(checkpoint), None, getattr(config, "text_config", config), model_device(gen))
    options = dict(slots=4, max_len=1024, use_triton=False, use_graphs=False, mtp=head)
    if parallel:
        return tp.TPModel(gen.model, devices=RANKS, **options)
    return FastModel(gen.model, **options)


@pytest.fixture(scope="module")
def one(checkpoint: Path) -> FastModel:
    return _load(checkpoint, False)


@pytest.fixture(scope="module")
def two(checkpoint: Path) -> tp.TPModel:
    return _load(checkpoint, True)


def _prefill(fast, slot: int, prompt: list[int]) -> torch.Tensor:
    fast.reset(slot)
    body = len(prompt) - 1
    position = 0
    while position < body:
        end = min((position // CHUNK + 1) * CHUNK, body)
        fast.prefill(slot, prompt[position:end], position, prompt[end])
        position = end
    return fast.prefill(slot, prompt[body:], body)


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    b = b.to(a.device)  # with two GPUs the one-device model's logits may sit on the other
    scale = float(a.abs().max())
    assert a.shape == b.shape
    assert float((a - b).abs().max()) <= 0.03 * scale + 1e-3, float((a - b).abs().max())


PROMPTS = [[(7 * i + 3 + j) % 300 + 5 for i in range(n)] for j, n in enumerate([13, 40, 70])]


def test_shards_split_every_layer(two: tp.TPModel) -> None:
    r0, r1 = two.ranks
    assert two.vocab == 320 and r0.vocab + r1.vocab == 320 and r1.vocab_offset == r0.vocab
    for i, t in enumerate(r0.layer_types):
        a, b = r0.layers[i], r1.layers[i]
        if t == "linear_attention":
            assert a.linear_attn.num_v_heads == b.linear_attn.num_v_heads == 2
        else:
            assert a.self_attn.k_proj.weight.shape[0] == a.self_attn.head_dim  # one KV head
        assert a.mlp.experts.gate_up_proj.shape[1] == 32  # 2 x 16 of 2 x 32
    assert r0.pool is r1.pool and r0.slot_pages is r1.slot_pages and r0.bank is r1.bank


def test_prefill_and_decode_match_one_device(one: FastModel, two: tp.TPModel) -> None:
    agree = total = 0
    for slot, prompt in enumerate(PROMPTS):
        a, b = _prefill(one, slot, prompt), _prefill(two, slot, prompt)
        _close(a, b)
        token = int(a.argmax())
        for step in range(6):  # teacher-forced on the one-device tokens
            la = one.decode([token], [slot], [len(prompt) + step])[0]
            lb = two.decode([token], [slot], [len(prompt) + step])[0]
            _close(la, lb)
            agree += int(la.argmax()) == int(lb.argmax())
            total += 1
            token = int(la.argmax())
    assert agree >= total - 2, (agree, total)


def test_packed_prefill_equals_separate(two: tp.TPModel) -> None:
    sep = [_prefill(two, s, p) for s, p in enumerate(PROMPTS[:2])]
    for s in range(2):
        two.reset(s)
    packed = two.prefill_packed([(0, PROMPTS[0], 0), (1, PROMPTS[1], 0)])
    for s in range(2):
        _close(sep[s], packed[s])


def test_verify_matches_decode_and_snapshots(two: tp.TPModel) -> None:
    prompt = PROMPTS[2]
    last = int(_prefill(two, 0, prompt).argmax())
    snap = two.snapshot(0, len(prompt))
    plain = two.decode([last], [0], [len(prompt)])[0].clone()
    want = int(plain.argmax())
    # Restore into another slot and verify (last, draft=want): the first pick is the decode's.
    two.reset(1)
    two.restore(snap, 1)
    logits, g1, g2, d1, d2 = two.verify([last], [want], [1], [len(prompt)])
    assert g1[0] == want
    _close(plain, logits[0])
    # A session tail handed over (detach/adopt) continues identically.
    two.reset(2)
    two.restore(snap, 2)
    tail = two.detach(2, len(prompt))
    two.reset(3)
    two.adopt(tail, 3)
    again = two.decode([last], [3], [len(prompt)])[0]
    assert torch.equal(again, plain)
    two.release(snap)
    for s in range(4):
        two.reset(s)


def test_argmax_ties_go_to_the_lowest_index() -> None:
    group = tp.Group(CPU)
    group.start()
    shards = [torch.tensor([[1.0, 5.0, 5.0], [0.0, 1.0, 2.0]]), torch.tensor([[5.0, 0.0], [3.0, 3.0]])]
    out: list[torch.Tensor | None] = [None, None]

    def run(rank: int) -> None:
        out[rank] = group.argmax(rank, shards[rank], 0 if rank == 0 else 3)

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    full = torch.cat(shards, dim=-1)
    assert out[0].tolist() == out[1].tolist() == full.argmax(-1).tolist() == [1, 3]


def test_warmup_leaves_the_pool_and_results_unchanged(one: FastModel, two: tp.TPModel) -> None:
    free = two.pool.free
    two.warmup(0, CHUNK, [(5 * i + 7) % 300 + 5 for i in range(160)], other=1)
    assert two.pool.free == free and not two.group.solo
    _close(_prefill(one, 0, PROMPTS[1]), _prefill(two, 0, PROMPTS[1]))
    one.reset(0)
    two.reset(0)


def test_lockstep_prefill_matches_threads(two: tp.TPModel, monkeypatch: pytest.MonkeyPatch) -> None:
    want = [_prefill(two, s, p) for s, p in enumerate(PROMPTS[:2])]
    drafts = []
    for s, p in enumerate(PROMPTS[:2]):
        _prefill(two, s, p)
        drafts.append(int(two.last_drafts[0]))
    monkeypatch.setattr(tp, "LOCKSTEP", True)
    for s, p in enumerate(PROMPTS[:2]):
        got = _prefill(two, s, p)
        _close(want[s], got)
        assert int(got.argmax()) == int(want[s].argmax())
        assert int(two.last_drafts[0]) == drafts[s]
    for s in range(2):
        two.reset(s)
    packed = two.prefill_packed([(0, PROMPTS[0], 0), (1, PROMPTS[1], 0)])
    for s in range(2):
        _close(want[s], packed[s])
        two.reset(s)
