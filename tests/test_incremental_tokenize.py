"""Plan 46: incremental prompt tokenization must equal the full encode exactly.

Runs with the checkpoint's tokenizer when it is in the Hugging Face cache
(``ENGINE_TOKENIZER`` may name another tokenizer directory or file), else skips.
"""

from __future__ import annotations

import os
import random

import pytest

from engine.inctok import PromptEncoder


def _tokenizer():
    path = os.environ.get("ENGINE_TOKENIZER")
    try:
        from transformers import AutoTokenizer

        if path:
            return AutoTokenizer.from_pretrained(path)
        from engine.generator import CHECKPOINT

        return AutoTokenizer.from_pretrained(CHECKPOINT["model"], revision=CHECKPOINT["revision"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no tokenizer: {exc}")


@pytest.fixture(scope="module")
def tok():
    return _tokenizer()


def _encoder(tok):
    from tokenizers import Tokenizer

    backend = Tokenizer.from_str(tok.backend_tokenizer.to_str())
    enc = PromptEncoder(lambda t: tok.encode(t), backend)
    assert enc.incremental, enc.why_full
    return enc


PIECES = [
    "def case_{i}(items):\n    return sorted(set(items))  # {i} 0f3a9c\n",
    "\n",
    "\n\n",
    "  \n",
    "\r\n",
    "x",
    " ",
    "\t",
    "<think>",
    "</think>\n\n",
    "<|im_end|>\n<|im_start|>user\n",
    "<tool_call>\n{\"name\": \"f\"}\n</tool_call>",
    "café",
    "é",
    "́",
    "中文",
    "\U0001f600",
    "'s",
    "'ll",
    "123456",
    "  # comment\n",
    "\n'",
    "\nI'm",
    "\n ",
    "\n.",
    "\né",
    "aaaa",
    "===",
    " ",
]


def _rand_text(rng: random.Random, n: int) -> str:
    out = []
    for _ in range(n):
        p = rng.choice(PIECES)
        out.append(p.format(i=rng.randrange(10**6)) if "{i}" in p else p)
    return "".join(out)


def test_sessions_like_replay(tok):
    """Growing prompts (prompt + output + new input), as the replay sends, incl. junctions
    that fall inside words, number runs, whitespace runs and multi-byte characters."""
    enc = _encoder(tok)
    rng = random.Random(46)
    checked = 0
    for session in range(40):
        text = _rand_text(rng, 60)
        for _ in range(6):
            assert enc.encode(text) == tuple(tok.encode(text)), (session, len(text))
            checked += 1
            text = text + _rand_text(rng, rng.randrange(1, 30))
    # Cut points chosen right at the junction with arbitrary characters before/after it.
    for _ in range(300):
        base = _rand_text(rng, rng.randrange(1, 20))
        enc.encode(base)
        for extra in ("\n", "x", " y", "́", "\n\n", "'s", "1", "中", "<think>", "\r"):
            text = base + extra + _rand_text(rng, 3)
            assert enc.encode(text) == tuple(tok.encode(text))
            checked += 1
    assert enc.hits > 0
    assert checked > 3000


def test_synthetic_corpus_long(tok):
    """The replay's synthetic code lines at replay sizes (~1.8k to ~8k tokens)."""
    enc = _encoder(tok)
    lines = [
        f"def case_{i}(items):\n    return sorted(set(items))  # {i} 6c1f0a9b\n" for i in range(3000)
    ]
    text = "".join(lines[:200])
    for step in range(12):
        assert enc.encode(text) == tuple(tok.encode(text))
        text += " sorted(set(x)) for x in items:\n" + "".join(lines[200 + 60 * step : 260 + 60 * step])
    assert enc.hits >= 11
