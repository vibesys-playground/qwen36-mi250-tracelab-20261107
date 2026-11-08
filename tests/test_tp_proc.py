"""Rank 1 in its own process (``engine.fast.tp_proc``, ENGINE_TP_PROC=1) against the threads
of ``engine.fast.tp``, on the tiny checkpoint of ``test_tp``. On the CPU both ranks' eager
all-reduces exchange through a pipe with the same sum as the host mailbox, so every result
must be bitwise the same; the commands (page tables, resets, flips, snapshots, steps) are the
ones a GPU replica sends.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / ".vibesys" / "shared" / "cpu_check"))

from engine.fast import tp, tp_proc  # noqa: E402
from engine.generator import Generator  # noqa: E402
from engine.mtp import MTPHead  # noqa: E402
from test_speculative import write_mtp_checkpoint  # noqa: E402
from test_tp import CHUNK, CPU, PROMPTS, _prefill  # noqa: E402

os.environ["ENGINE_TP_PROC_EXIT"] = "0"  # a dead child must raise here, not end pytest


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-mtp-tp-proc")
    write_mtp_checkpoint(path)
    return path


def _load(checkpoint: Path, proc: bool):
    gen = Generator.load(str(checkpoint), None, max_model_len=1024, chunk_tokens=CHUNK,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    config = gen.model.config
    head = MTPHead.load(str(checkpoint), None, getattr(config, "text_config", config), "cpu")
    options = dict(slots=4, max_len=1024, use_triton=False, use_graphs=False, mtp=head)
    cls = tp_proc.ProcTPModel if proc else tp.TPModel
    return cls(gen.model, devices=CPU, **options)


@pytest.fixture(scope="module")
def threads(checkpoint: Path) -> tp.TPModel:
    return _load(checkpoint, False)


@pytest.fixture(scope="module")
def proc(checkpoint: Path):
    model = _load(checkpoint, True)
    yield model
    model.close()


def _same(a: torch.Tensor, b) -> None:
    b = b if isinstance(b, torch.Tensor) else b[: len(b)]
    assert a.shape == b.shape and torch.equal(a, b), float((a - b).abs().max())


def test_shapes(threads: tp.TPModel, proc) -> None:
    assert proc.vocab == threads.vocab == 320
    assert proc.pool.pages == threads.pool.pages
    assert len(proc.ranks) == 1


def test_prefill_decode_verify_and_snapshots_match_threads(threads: tp.TPModel, proc) -> None:
    for model in (threads, proc):
        for s in range(4):
            model.reset(s)
    for slot, prompt in enumerate(PROMPTS):
        a, b = _prefill(threads, slot, prompt), _prefill(proc, slot, prompt)
        _same(a, b)
        assert int(threads.last_drafts[0]) == int(proc.last_drafts[0])
        token = int(a.argmax())
        for step in range(4):
            la = threads.decode([token], [slot], [len(prompt) + step])
            lb = proc.decode([token], [slot], [len(prompt) + step])
            _same(la, lb)
            token = int(la[0].argmax())
    # Packed prefill of two fresh sequences.
    for model in (threads, proc):
        model.reset(0)
        model.reset(1)
    pa = threads.prefill_packed([(0, PROMPTS[0], 0), (1, PROMPTS[1], 0)])
    pb = proc.prefill_packed([(0, PROMPTS[0], 0), (1, PROMPTS[1], 0)])
    _same(pa, pb)
    assert threads.last_drafts.tolist() == proc.last_drafts.tolist()
    # Snapshots, restore, verify with flips, detach/adopt, release.
    prompt = PROMPTS[2]
    outs = []
    for model in (threads, proc):
        last = int(_prefill(model, 0, prompt).argmax())
        snap = model.snapshot(0, len(prompt))
        plain = model.decode([last], [0], [len(prompt)])[0].clone()
        want = int(plain.argmax())
        model.reset(1)
        model.restore(snap, 1)
        logits, g1, g2, d1, d2 = model.verify([last], [want], [1], [len(prompt)])
        row = logits[0].clone()
        model.flip(1)  # accept the draft as the generator does
        after = model.decode([g2[0]], [1], [len(prompt) + 2])[0].clone()
        model.reset(2)
        model.restore(snap, 2)
        tail = model.detach(2, len(prompt))
        model.reset(3)
        model.adopt(tail, 3)
        again = model.decode([last], [3], [len(prompt)])[0].clone()
        model.release(snap)
        outs.append((plain, row, (g1, g2, d1, d2), after, again))
        for s in range(4):
            model.reset(s)
    (p0, r0, k0, a0, g0), (p1, r1, k1, a1, g1_) = outs
    _same(p0, p1)
    _same(r0, r1)
    assert k0 == k1
    _same(a0, a1)
    _same(g0, g1_)
    _same(p1, g1_)
    assert proc.snapshot_bytes(proc.snapshot(0, 5)) == threads.snapshot_bytes(threads.snapshot(0, 5))


def test_warmup_and_stale_verify_logits(threads: tp.TPModel, proc) -> None:
    free = proc.pool.free
    ids = [(5 * i + 7) % 300 + 5 for i in range(160)]
    proc.warmup(0, CHUNK, ids, other=1)
    threads.warmup(0, CHUNK, ids, other=1)
    assert proc.pool.free == free and not proc.group.solo
    _same(_prefill(threads, 0, PROMPTS[1]), _prefill(proc, 0, PROMPTS[1]))
    logits, *_ = proc.verify([7], [9], [0], [len(PROMPTS[1])])
    proc.decode([7], [0], [len(PROMPTS[1])])
    with pytest.raises(RuntimeError, match="after the next step"):
        logits[0]
    for model in (threads, proc):
        model.reset(0)
    assert proc.sync() == 0


def test_child_death_raises_and_close_leaves_no_process(checkpoint: Path) -> None:
    model = _load(checkpoint, True)
    proc = model._proc
    assert proc.is_alive()
    proc.kill()
    proc.join(10)
    deadline = time.monotonic() + 10
    while model._dead is None and time.monotonic() < deadline:
        time.sleep(0.05)
    with pytest.raises(RuntimeError, match="rank 1"):
        _prefill(model, 0, PROMPTS[0])
    model.close()
    other = _load(checkpoint, True)
    child = other._proc
    other.close()
    assert not child.is_alive()


def test_failed_load_stops_the_child(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rank 0's load failing after the child started (j157: the shard's memory pool) leaves
    no rank-1 process behind."""
    import multiprocessing

    before = {p.pid for p in multiprocessing.active_children()}

    def fail(self, lm, mtp):  # noqa: ANN001, ANN202
        raise RuntimeError("test: shard failed")

    monkeypatch.setattr(tp_proc.ProcTPModel, "_shard", fail)
    with pytest.raises(RuntimeError, match="shard failed"):
        _load(checkpoint, True)
    left = [p for p in multiprocessing.active_children() if p.pid not in before]
    assert not left, left
