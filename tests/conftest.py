"""Fixtures for the engine's own CPU tests: tiny random checkpoints, served as ``python -m engine.server``.

Run with the CPU-check environment:

    UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/vibesys/qwen36-cpu-check-venv" \\
      uv run --project .vibesys/shared/cpu_check python -m pytest tests
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / ".vibesys" / "shared" / "cpu_check"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tiny_checkpoint import write_checkpoint  # noqa: E402

CHUNK = 8  # small prefill chunks, so short prompts cross chunk boundaries
_STARTUP_GUARD_SECONDS = 180


def repeat_reuse(n: int, chunk: int = CHUNK, gpu: bool | None = None) -> int:
    """Tokens a repeat of an ``n``-token prompt reuses from its first run's snapshots.

    The Transformers path (and the fast path with ``ENGINE_PREFILL_ONE_PASS_MAX=0``)
    snapshots all but the last token. The fast path's one-pass prefill stores no
    snapshot for an uncached part within the limit, and the last chunk boundary
    before the end for a longer one. Exact for prompts within the default limit,
    or for a prompt whose first run started cold.

    Which path the server runs is ``engine.generator.use_fast_path``: ``ENGINE_FAST``,
    and with it unset, whether the server sees a GPU (``gpu``; default: whether this
    process does, as a server started from it with the same environment).
    """
    from engine.generator import use_fast_path

    if gpu is None:
        gpu = torch.cuda.is_available()
    one_pass_max = int(os.environ.get("ENGINE_PREFILL_ONE_PASS_MAX", "4096"))
    if not use_fast_path(os.environ.get("ENGINE_FAST"), gpu) or one_pass_max == 0:
        return max(n - 1, 0)
    if n <= one_pass_max:
        return 0
    return ((n - 1) // chunk) * chunk


def load_weights(model: str, **options: int):
    """``Generator.load`` on the Transformers path (``ENGINE_FAST=0``), for a test that builds
    its own fast models over ``generator.model``.

    With ``ENGINE_FAST`` unset a GPU host loads the fast path, and with two visible GPUs that
    is a ``TPModel``, whose sharding frees the loaded weights (``engine.fast.tp.shard_model``):
    every model built on them afterwards would read empty tensors. On the CPU this is the
    default load.
    """
    from engine.generator import Generator

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ENGINE_FAST", "0")
        return Generator.load(model, None, **options)


def load_on_cpu(model: str, **options: int):
    """``load_weights`` with the weights on the CPU, in a process that may see GPUs: for a test
    whose premise holds only on the CPU (exact agreement of two numerically different paths).

    The GDN kernels stay as a GPU load installs them (``engine.gdn``), which on CPU tensors
    run Transformers' torch rules, as on a CPU host.
    """
    from engine import gdn
    from engine.generator import Generator

    load_weights(model, **options)  # the GDN kernels as this host serves them
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ENGINE_FAST", "0")
        patch.setattr(torch.cuda, "is_available", lambda: False)
        patch.setattr(gdn, "install", lambda *a, **k: {"mode": "as installed", "layers": 0,
                                                       "fla": None, "fla_error": None})
        return Generator.load(model, None, **options)


def model_device(generator) -> torch.device:
    """The device of the loaded model's lm_head: where its MTP head must sit
    (``engine.speculative.load_head``, ``FastModel``)."""
    return generator.model.get_output_embeddings().weight.device


@dataclass(frozen=True)
class Server:
    url: str
    model: str


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(checkpoint: Path, log: Path, *extra: str) -> Iterator[Server]:
    port = _free_port()
    with log.open("w") as sink:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "engine.server",
                "--model",
                str(checkpoint),
                "--port",
                str(port),
                "--prefill-chunk-tokens",
                str(CHUNK),
                *extra,
            ],
            cwd=PROJECT_ROOT,
            stdout=sink,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + _STARTUP_GUARD_SECONDS
        while True:
            if process.poll() is not None:
                pytest.fail(f"engine exited with {process.returncode}:\n{log.read_text()}")
            try:
                if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"engine did not become healthy:\n{log.read_text()}")
            time.sleep(0.2)
        yield Server(url=url, model=str(checkpoint))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-qwen36")
    write_checkpoint(path)
    return path


@pytest.fixture(scope="session")
def server(checkpoint: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Server]:
    yield from serve(checkpoint, tmp_path_factory.mktemp("logs") / "server.log")


@pytest.fixture(scope="session")
def cold_server(checkpoint: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Server]:
    """The same weights with the prefix cache off: every prompt is prefilled from scratch."""
    yield from serve(
        checkpoint, tmp_path_factory.mktemp("logs") / "cold.log", "--prefix-cache-entries", "0"
    )
