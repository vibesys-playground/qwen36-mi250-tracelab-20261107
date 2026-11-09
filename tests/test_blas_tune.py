"""Plan 61: the shipped prefill BLAS table and the TunableOp window (engine/fast/blas_tune.py)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from engine.fast import blas_tune

TABLE = Path(blas_tune.__file__).with_name("blas_tuned_gfx90a.csv")


def test_shipped_table_parses():
    validators, rows = blas_tune.read_table(str(TABLE))
    assert validators.get("GCN_ARCH_NAME", "").startswith("gfx90a")
    assert {"PT_VERSION", "ROCM_VERSION", "HIPBLASLT_VERSION"} <= set(validators)
    assert rows and all(r[0].startswith("Gemm") and len(r) >= 4 for r in rows)
    assert len({(r[0], r[1]) for r in rows}) == len(rows)  # one solution per GEMM


def test_mismatched_table_falls_back(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("needs a ROCm GPU")
    _, rows = blas_tune.read_table(str(TABLE))
    bad = tmp_path / "t.csv"
    bad.write_text("Validator,HIPBLASLT_VERSION,0-not-this\n" + ",".join(rows[0]) + "\n")
    monkeypatch.setenv("ENGINE_PREFILL_BLAS_TABLE", str(bad))
    monkeypatch.setenv("ENGINE_PREFILL_BLAS_TUNE", "1")
    mod = importlib.reload(blas_tune)
    try:
        with mod.prefill_blas() as state:
            assert state.startswith("default"), state
            assert not torch.cuda.tunable.is_enabled()
        assert not torch.cuda.tunable.is_enabled()
    finally:
        monkeypatch.undo()
        importlib.reload(blas_tune)


def test_window_closes():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("needs a ROCm GPU")
    with blas_tune.prefill_blas() as state:
        assert state.startswith("tuned"), state  # the image the table was tuned on
        assert torch.cuda.tunable.is_enabled() and not torch.cuda.tunable.tuning_is_enabled()
    assert not torch.cuda.tunable.is_enabled()
