"""Plan 74 stage 0: where device_map="auto" places the tiny checkpoints on the visible GPUs.

Conftest-free. Prints hf_device_map, the placed devices and accelerate's balanced max_memory for
the plain tiny checkpoint, the MTP tiny checkpoint (index without metadata) and the MTP one with
``metadata.total_size`` in its index.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / ".vibesys" / "shared" / "cpu_check")]

from tiny_checkpoint import write_checkpoint  # noqa: E402
from test_speculative import write_mtp_checkpoint  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


def probe(name: str, path: Path) -> None:
    print(f"--- {name}: files {sorted(p.name for p in path.iterdir())}", flush=True)
    try:
        from accelerate.utils import get_balanced_memory, get_max_memory

        print(f"    max_memory {get_max_memory()}", flush=True)
        lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
        print(f"    hf_device_map {getattr(lm, 'hf_device_map', None)}", flush=True)
        print(f"    placed {sorted({str(p.device) for p in lm.parameters()})}", flush=True)
        try:
            bal = get_balanced_memory(lm, dtype=torch.bfloat16, no_split_module_classes=lm._no_split_modules)
            print(f"    balanced {bal}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    balanced failed: {exc!r}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def main() -> None:
    print(f"devices {torch.cuda.device_count()}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        plain, mtp, meta = Path(tmp, "plain"), Path(tmp, "mtp"), Path(tmp, "meta")
        for p in (plain, mtp, meta):
            p.mkdir()
        write_checkpoint(plain)
        write_mtp_checkpoint(mtp)
        write_mtp_checkpoint(meta)
        index = json.loads((meta / "model.safetensors.index.json").read_text())
        total = sum((meta / f).stat().st_size for f in set(index["weight_map"].values()))
        index["metadata"] = {"total_size": total}
        (meta / "model.safetensors.index.json").write_text(json.dumps(index))
        probe("plain", plain)
        probe("mtp", mtp)
        probe("mtp+metadata", meta)


if __name__ == "__main__":
    main()
