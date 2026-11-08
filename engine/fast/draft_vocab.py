"""A static draft vocabulary for the MTP head (ENGINE_MTP_VOCAB).

The MTP head only proposes a draft; verify picks every served token with an exact greedy
argmax over the whole vocabulary, so a draft taken from a subset of the vocabulary changes
which drafts are proposed (and how many are accepted), never the served tokens. The head's
lm_head then reads ``n`` rows instead of all ~248k (FR-Spec style).

The subset is built once at load from generic text that ships in the image, never from
requests or a workload's prompts:

- the Python standard library's source of the running interpreter (code, comments and
  docstrings), files in sorted order, test suites left out;
- the license texts of the base OS (``/usr/share/common-licenses``), generic English prose.

Its ids: the ``n // 2`` most frequent tokens of that text (ties to the lower id), filled up
to ``n`` with the lowest ids not yet chosen (the tokenizer's earliest merges: bytes and the
commonest pieces), plus every special / added token (chat template markers), sorted.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sysconfig
from pathlib import Path
from typing import Any, Sequence

SKIP_DIRS = {"test", "tests", "idle_test", "site-packages", "dist-packages", "__pycache__"}
LICENSES = Path("/usr/share/common-licenses")
CACHE_DIR = Path(os.environ.get("ENGINE_DRAFT_VOCAB_CACHE", "/tmp/engine-draft-vocab"))


def size_from_env() -> int:
    """ENGINE_MTP_VOCAB: the draft vocabulary's size; 0 (the default) keeps the full head."""
    return max(0, int(os.environ.get("ENGINE_MTP_VOCAB", "0") or 0))


def corpus_files() -> list[Path]:
    """The generic text the frequencies come from, in a fixed order."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = []
    for root, dirs, names in os.walk(stdlib):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        files.extend(Path(root) / n for n in sorted(names) if n.endswith(".py"))
    if LICENSES.is_dir():
        files.extend(p for p in sorted(LICENSES.rglob("*")) if p.is_file() and not p.is_symlink())
    return files


def token_counts(tokenizer: Any, files: Sequence[Path], batch: int = 256) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    texts: list[str] = []

    def flush() -> None:
        if texts:
            for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
                counts.update(ids)
            texts.clear()

    for path in files:
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if len(texts) >= batch:
            flush()
    flush()
    return counts


def special_ids(tokenizer: Any) -> set[int]:
    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    ids.update(int(i) for i in decoder)
    return ids


def select(counts: collections.Counter, n: int, vocab: int, always: set[int]) -> list[int]:
    """``n // 2`` most frequent ids, the always-ids, then the lowest ids up to ``n``."""
    chosen = {i for i in always if 0 <= i < vocab}
    ranked = sorted(((c, i) for i, c in counts.items() if 0 <= i < vocab), key=lambda t: (-t[0], t[1]))
    for _, i in ranked[: n // 2]:
        chosen.add(i)
    low = 0
    while len(chosen) < n and low < vocab:
        chosen.add(low)
        low += 1
    return sorted(chosen)


def build(tokenizer: Any, n: int, vocab: int) -> list[int] | None:
    """The sorted draft ids (about ``n``; a few more when the special ids sit above the fill),
    or None for the full vocabulary (``n`` 0 or not smaller than ``vocab``). Cached under /tmp
    by the corpus, the tokenizer and ``n``."""
    if n <= 0 or n >= vocab:
        return None
    files = corpus_files()
    key = hashlib.sha256()
    key.update(f"v1 {n} {vocab} {len(tokenizer)}".encode())
    key.update(getattr(tokenizer, "name_or_path", "").encode())
    for f in files:
        try:
            key.update(f"{f}:{f.stat().st_size}".encode())
        except OSError:
            pass
    cache = CACHE_DIR / f"{key.hexdigest()[:24]}.json"
    try:
        ids = json.loads(cache.read_text())
        if isinstance(ids, list) and ids:
            return ids
    except (OSError, ValueError):
        pass
    counts = token_counts(tokenizer, files)
    ids = select(counts, n, vocab, special_ids(tokenizer))
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(ids))
    except OSError:
        pass
    return ids
