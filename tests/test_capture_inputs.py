"""A decode step whose bucket is captured inside the step must run on its own inputs.

Capturing (``FastModel._capture``, ``TPModel._warm``) first runs the step on the dummy slot
through the same input buffers the step had just loaded. Before the fix the step then ran
(replayed, or eagerly after a failed capture) on the dummy slot's inputs: wrong logits and a
lost token for the real slots. On the CPU there are no graphs, so the capture here does what
a capture does to the inputs and then refuses, and the step falls back to eager.
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

from test_speculative import write_mtp_checkpoint  # noqa: E402
from test_tp import PROMPTS, _load, _prefill  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-mtp-capture")
    write_mtp_checkpoint(path)
    return path


def _clobber(owner, ranks) -> None:
    """``owner.capture`` that overwrites the step inputs as the warm-up does, then fails."""

    def capture(size: int, verify: bool = False) -> None:
        for model in ranks:
            for meta in model.devices.values():
                meta.slots.fill_(model.dummy)
                meta.positions.zero_()
                meta.gdn.fill_(model.dummy)
        owner.graph_error = "test: capture refused after its warm-up"

    owner.capture = capture


@pytest.mark.parametrize("parallel", [False, True])
def test_step_after_in_step_capture_uses_its_inputs(checkpoint: Path, parallel: bool) -> None:
    model = _load(checkpoint, parallel)
    ranks = model.ranks if parallel else [model]
    prompt = PROMPTS[1]
    want, got = [], []
    for slot, sink in ((0, want), (1, got)):
        last = int(_prefill(model, slot, prompt).argmax())
        if slot == 1:  # the second slot's first step captures its bucket inside the step
            for m in ranks:
                m.use_graphs = True
            model.graph_error = None
            _clobber(model, ranks)
        for step in range(3):
            row = model.decode([last], [slot], [len(prompt) + step])[0].clone()
            sink.append(row)
            last = int(row.argmax())
    for a, b in zip(want, got, strict=True):
        assert torch.equal(a, b)
