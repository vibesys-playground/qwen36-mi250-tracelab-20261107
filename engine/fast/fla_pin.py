"""Fixed launch configs for fla's autotuned Triton kernels (the chunked gated delta rule).

fla autotunes its kernels by timing every candidate config the first time a shape key is
seen, in each process; with ``FLA_CACHE_RESULTS`` (fla's default) the pick is also written
to, and read back from, Triton's on-disk cache, which concurrent processes (the replicas)
write in no fixed order. Candidates differ in block sizes and warps, so in reduction order:
two processes can pick differently and give slightly different logits, which flips greedy
picks at near-ties. Pinning each kernel to one config (``fla_configs.json``: kernel ->
config string, as ``str(triton.Config)`` prints it) makes the choice the same in every
process and skips the timing. ``ENGINE_FLA_PIN=0`` restores fla's autotuning;
``ENGINE_FLA_PINS`` names another JSON file.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PINS = Path(__file__).with_name("fla_configs.json")


def _tuner(obj: Any) -> Any:
    """The Triton ``Autotuner`` inside ``obj`` (through ``heuristics`` wrappers), or None."""
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:  # pragma: no cover - no Triton
        return None
    for _ in range(4):
        if isinstance(obj, Autotuner):
            return obj
        obj = getattr(obj, "fn", None)
        if obj is None:
            return None
    return None


def autotuners() -> Iterator[tuple[str, Any]]:
    """(``module.attr``, autotuner) for every autotuned kernel in the loaded fla modules."""
    seen = set()
    for name, module in sorted(list(sys.modules.items())):
        if not (name == "fla" or name.startswith("fla.")) or module is None:
            continue
        for attr, value in sorted(vars(module).items()):
            tuner = _tuner(value)
            if tuner is not None and id(tuner) not in seen:
                seen.add(id(tuner))
                yield f"{name}.{attr}", tuner


def snapshot() -> dict[str, dict[str, Any]]:
    """Per autotuned kernel: its candidate count and the config picked per tuned key."""
    return {
        name: {"configs": len(t.configs), "picked": {repr(k): str(c) for k, c in t.cache.items()}}
        for name, t in autotuners()
    }


def apply(path: str | os.PathLike | None = None) -> dict[str, int]:
    """Pin every listed kernel to its config (one candidate left, so no timing). Returns
    counts: pinned, not found among the candidates, kernels without an entry."""
    if os.environ.get("ENGINE_FLA_PIN", "1") == "0":
        return {}
    path = Path(path or os.environ.get("ENGINE_FLA_PINS") or PINS)
    if not path.exists():
        return {}
    pins = json.loads(path.read_text())
    counts = {"pinned": 0, "missing_config": 0, "unlisted": 0}
    for name, tuner in autotuners():
        want = pins.get(name)
        if want is None:
            counts["unlisted"] += 1
            continue
        match = [c for c in tuner.configs if str(c) == want]
        if not match:
            counts["missing_config"] += 1
            continue
        tuner.configs = [match[0]]
        tuner.cache.clear()
        counts["pinned"] += 1
    return counts
