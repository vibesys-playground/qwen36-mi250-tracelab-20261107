"""Plan 61: measured BLAS solutions for the dense GEMMs of the captured prefill graphs only.

``TPModel.capture_prefill`` records one graph per padded prefill size (64..2,048 rows). Its
dense projections are plain ``F.linear`` calls, for which hipBLASLt's default heuristic picks
one solution per shape. Here PyTorch TunableOp swaps in the solution that was fastest on the
hardware for that exact GEMM (op, transposes, m, n, k, leading dimensions), chosen offline by
``tools/blas_tune_probe.py`` over hipBLASLt and rocBLAS candidates for the shapes those graphs
run (model shapes only; synthetic inputs). The table ships in the repository and is read
read-only at startup; nothing is tuned while the server starts.

TunableOp is on only while ``capture_prefill`` warms and captures (``prefill_blas``); before
it the decode and verify graphs are captured, after it the chained verify graphs, and every
eager call (packed or longer prefills, fallbacks) runs as before. A graph replays the kernel
it recorded, so serving needs no TunableOp state.

``ENGINE_PREFILL_BLAS_TUNE``:
- ``1`` (default): use the table (``ENGINE_PREFILL_BLAS_TABLE``, default
  ``blas_tuned_gfx90a.csv`` next to this file) when its validators (torch, ROCm, hipBLASLt,
  rocBLAS versions, GPU arch) equal this process's; otherwise, or if anything fails, the
  default heuristic (today's kernels), with one log line saying why. A shape missing from the
  table also runs the default solution.
- ``0``: TunableOp is never touched (exactly the previous behaviour).
- ``record``: TunableOp on with tuning off (default solutions) and the untuned GEMMs written
  to ``PYTORCH_TUNABLEOP_UNTUNED_FILENAME`` (the probe's input for offline tuning).

Tuning is not done online in the warm passes: the two ranks' passes meet at device-side
all-reduces, and one rank timing hundreds of candidates while its peer spins would stall it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import torch

MODE = os.environ.get("ENGINE_PREFILL_BLAS_TUNE", "1")
TABLE = os.environ.get("ENGINE_PREFILL_BLAS_TABLE", str(Path(__file__).with_name("blas_tuned_gfx90a.csv")))
# Where TunableOp would read or write its own results file: never the read-only checkout.
SCRATCH = "/tmp/engine_tunableop_results.csv"


def read_table(path: str) -> tuple[dict[str, str], list[list[str]]]:
    """(validators, result rows) of a TunableOp results CSV."""
    validators: dict[str, str] = {}
    rows: list[list[str]] = []
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 3 and cells[0] == "Validator":
            validators[cells[1]] = cells[2]
        elif len(cells) >= 4:
            rows.append(cells)
    return validators, rows


def current_validators() -> dict[str, str]:
    got = torch.cuda.tunable.get_validators()
    if isinstance(got, dict):
        return {str(k): str(v) for k, v in got.items()}
    return {str(k): str(v) for k, v in got}


def _register_validators() -> None:
    """The ROCm BLAS validators are registered when TunableOp builds its first GEMM op: run one
    tiny GEMM with TunableOp on and tuning off (the default solution)."""
    a = torch.zeros(8, 8, dtype=torch.bfloat16, device="cuda")
    torch.mm(a, a.t())
    torch.cuda.synchronize()


def _quiet_base() -> None:
    tunable = torch.cuda.tunable
    tunable.tuning_enable(False)
    getattr(tunable, "write_file_on_exit", lambda v: None)(False)  # gone in torch 2.10
    tunable.set_filename(SCRATCH, False)


@contextlib.contextmanager
def prefill_blas() -> Iterator[str]:
    """TunableOp around the prefill graphs' warm-up and capture; yields a one-line state."""
    if MODE == "0" or not torch.cuda.is_available() or torch.version.hip is None:
        yield "off"
        return
    tunable = torch.cuda.tunable
    state = "off"
    try:
        _quiet_base()
        if MODE == "record":
            tunable.record_untuned_enable(True)
            tunable.enable(True)
            state = f"record -> {os.environ.get('PYTORCH_TUNABLEOP_UNTUNED_FILENAME', '(torch default)')}"
        else:
            validators, rows = read_table(TABLE)
            tunable.enable(True)
            _register_validators()
            have = current_validators()
            wrong = {k: (v, have.get(k)) for k, v in validators.items() if have.get(k) != v}
            if not rows or not validators or wrong:
                tunable.enable(False)
                state = f"default (table {Path(TABLE).name} does not match: {wrong or 'empty'})"
            elif not tunable.read_file(TABLE):
                tunable.enable(False)
                state = f"default (TunableOp refused {Path(TABLE).name})"
            else:
                state = f"tuned ({len(rows)} solutions from {Path(TABLE).name})"
    except Exception as exc:  # noqa: BLE001 - the default heuristic is always a valid answer
        with contextlib.suppress(Exception):
            tunable.enable(False)
        state = f"default ({type(exc).__name__}: {exc})"
    print(f"engine: prefill BLAS {state}", flush=True)
    try:
        yield state
    finally:
        with contextlib.suppress(Exception):
            tunable.record_untuned_enable(False)
        with contextlib.suppress(Exception):
            tunable.enable(False)
