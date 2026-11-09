"""Exact incremental tokenization of growing prompts (plan 46).

A session's next prompt is its previous prompt plus the model's output plus new
input, so re-encoding the whole text every round repeats work whose answer is
known. ``PromptEncoder.encode(text)`` returns exactly
``tokenizer.encode(text)`` (the full encode), but when ``text`` starts with a
recently encoded prompt it reuses that prompt's ids up to a safe cut and
encodes only the rest.

Why a cut is safe (for the checkpoint's tokenizer, checked at construction):
the tokenizer splits off added tokens, normalizes with NFC, splits the text
into pre-tokens with one regex (left to right, no look-behind), byte-maps and
BPE-merges each pre-token on its own, and adds no special tokens. In that
regex no pre-token runs from a line feed into a following non-whitespace
character (every alternative that can match ``\\n`` ends at a line break or
holds only whitespace, and the letter, number and punctuation alternatives
cannot start with one), so a position ``c`` with ``text[c - 1] == "\\n"`` and
an ASCII non-whitespace ``text[c]`` is a pre-token boundary of any text with
that prefix, and matching the pre-tokens before ``c`` reads no character past
``c``. NFC does not compose across an ASCII starter, no added token contains a
line feed (so none spans ``c``), and none strips whitespace to its left, so
the cut is also a boundary for added-token splitting. Hence for ``c`` inside
the cached prompt, ``encode(new) == ids_of(prompt[:c]) + encode(new[c:])``.
A tokenizer whose configuration does not match these conditions gets the full
encode always; ``ENGINE_TOKENIZE_CHECK=1`` compares every incremental result
with the full encode and raises on any difference.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from typing import Any

# The Qwen2/Qwen3 family's pre-tokenizer pattern (with and without combining marks).
_SAFE_PATTERNS = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)


def safe_config(backend: Any) -> str | None:
    """None when ``backend`` (a ``tokenizers.Tokenizer``) satisfies the cut argument, else why not."""
    try:
        spec = json.loads(backend.to_str())
    except Exception as exc:  # noqa: BLE001
        return f"cannot read the tokenizer config: {exc}"
    if spec.get("truncation") is not None or spec.get("padding") is not None:
        return "truncation or padding is set"
    normalizer = spec.get("normalizer")
    if normalizer not in (None, {"type": "NFC"}):
        return f"normalizer {normalizer}"
    pre = spec.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers") if pre.get("type") == "Sequence" else [pre]
    if not parts or len(parts) != 2:
        return "pre-tokenizer is not Split + ByteLevel"
    split, byte_level = parts
    if (
        split.get("type") != "Split"
        or split.get("behavior") != "Isolated"
        or split.get("invert")
        or (split.get("pattern") or {}).get("Regex") not in _SAFE_PATTERNS
    ):
        return "pre-tokenizer split is not the known pattern"
    if byte_level.get("type") != "ByteLevel" or byte_level.get("add_prefix_space") or byte_level.get(
        "use_regex"
    ):
        return "pre-tokenizer byte level differs"
    post = spec.get("post_processor")
    if post is not None and post.get("type") != "ByteLevel":
        return f"post-processor {post.get('type')} may add tokens"
    model = spec.get("model") or {}
    if model.get("type") != "BPE":
        return "model is not BPE"
    for added in spec.get("added_tokens") or []:
        content = added.get("content", "")
        if "\n" in content or "\r" in content or added.get("lstrip"):
            return f"added token {content!r} may span a cut"
    return None


class PromptEncoder:
    """Thread-safe exact prompt encoder with a small cache of recent prompts.

    ``full`` is the reference encode (``tokenizer.encode``); ``backend`` is a
    ``tokenizers.Tokenizer`` used for suffixes (its own instance, so handler
    threads never share the Transformers wrapper's mutable state).
    """

    def __init__(self, full: Any, backend: Any | None, entries: int = 256) -> None:
        self._full = full
        self._backend = backend
        self._entries = entries
        self._cache: OrderedDict[str, tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self.check = os.environ.get("ENGINE_TOKENIZE_CHECK") == "1"
        self.why_full = "no backend tokenizer" if backend is None else safe_config(backend)
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0
        if self.why_full is None:
            sample = "def f(x):\n    return x  # é中 <think>\n</think>\n\nok\n"
            if tuple(full(sample)) != tuple(backend.encode(sample, add_special_tokens=True).ids) or (
                tuple(backend.encode(sample, add_special_tokens=True).ids)
                != tuple(backend.encode(sample, add_special_tokens=False).ids)
            ):
                self.why_full = "backend encode differs from the full encode"

    @property
    def incremental(self) -> bool:
        return self.why_full is None

    @staticmethod
    def _cuts(text: str, base: int, offsets: list[tuple[int, int]], first_token: int) -> list[tuple[int, int]]:
        """Safe cuts (char position, token index) among the tokens of ``text[base:]``."""
        cuts = []
        for index, (start, _) in enumerate(offsets):
            c = base + start
            if c > 0 and text[c - 1] == "\n" and c < len(text):
                ch = text[c]
                if ch < "\x7f" and not ch.isspace():
                    cuts.append((c, first_token + index))
        return cuts

    def _lookup(self, text: str) -> tuple[str, tuple[int, ...], tuple[tuple[int, int], ...]] | None:
        best = None
        with self._lock:
            for key, (ids, cuts) in self._cache.items():
                if (best is None or len(key) > len(best[0])) and text.startswith(key):
                    best = (key, ids, cuts)
            if best is not None:
                self._cache.move_to_end(best[0])
        return best

    def _store(self, text: str, ids: tuple[int, ...], cuts: tuple[tuple[int, int], ...]) -> None:
        with self._lock:
            self._cache[text] = (ids, cuts)
            self._cache.move_to_end(text)
            while len(self._cache) > self._entries:
                self._cache.popitem(last=False)

    def encode(self, text: str) -> tuple[int, ...]:
        """Exactly ``tuple(full(text))``."""
        if not self.incremental:
            return tuple(self._full(text))
        found = self._lookup(text)
        cut = None
        if found is not None:
            key, ids, cuts = found
            # The last safe cut strictly inside the cached prompt (c < len(key): the character
            # at the cut is the prompt's own, so the cached ids cover the same pre-tokens).
            for c, k in reversed(cuts):
                if c < len(key):
                    cut = (c, k, ids, cuts)
                    break
        if cut is None:
            enc = self._backend.encode(text, add_special_tokens=False)
            ids = tuple(enc.ids)
            cuts = tuple(self._cuts(text, 0, enc.offsets, 0))
            self.misses += 1
        else:
            c, k, old_ids, old_cuts = cut
            enc = self._backend.encode(text[c:], add_special_tokens=False)
            ids = old_ids[:k] + tuple(enc.ids)
            cuts = tuple(x for x in old_cuts if x[0] <= c) + tuple(
                x for x in self._cuts(text, c, enc.offsets, k) if x[0] > c
            )
            self.hits += 1
            self.reused_tokens += k
        if self.check:
            want = tuple(self._full(text))
            if want != ids:
                raise AssertionError(
                    f"incremental tokenization differs from the full encode ({len(ids)} vs "
                    f"{len(want)} ids)"
                )
        self._store(text, ids, cuts)
        return ids

    def stats(self) -> dict[str, Any]:
        return {
            "tokenize_incremental": self.incremental,
            "tokenize_hits": self.hits,
            "tokenize_misses": self.misses,
            "tokenize_reused_tokens": self.reused_tokens,
        }
