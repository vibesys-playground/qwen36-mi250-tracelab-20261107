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


def test_random_character_soup(tok):
    """Short random texts over characters that stress the cut rules (spaces, tabs, line
    breaks, combining marks, no-break spaces, apostrophes, added tokens), grown in rounds."""
    enc = _encoder(tok)
    rng = random.Random(7)
    alphabet = ["a", "b", " ", "  ", "\t", ".", "(", ")", "_", "1", "2", "\n", "é", "中",
                "'", "́", " ", "<think>", "x y", "\r\n", "  \n", "'s ", " '"]
    checked = 0
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 200)))
        for _ in range(5):
            assert enc.encode(text) == tuple(tok.encode(text)), repr(text)
            checked += 1
            text += "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 40)))
    assert enc.hits > checked // 2


# Plan 69: more corpora. Natural text (this repository's own code and prose, multilingual
# prose) and replay-like sessions whose "output" is the decode of arbitrary token ids, as a
# model's streamed output is; junctions anywhere, multi-byte characters at the cut.

_MULTILINGUAL = (
    "Grüße aus Köln! Ça va? Ñandú, façade, naïve coöperation.\n"
    "日本語のテキストと中文混合，还有标点。\n"
    "한국어 문장입니다. Русский текст с ёжиком. Ελληνικά γράμματα.\n"
    "العربية تكتب من اليمين إلى اليسار. עברית גם.\n"
    "Emoji: 👨‍👩‍👧‍👦 🏳️‍🌈 👍🏽 ❤️ — dashes – and “quotes” ‘single’ …\n"
    "Combining: é ä ñ; NBSP: x; ideographic space:　y; tab:\tz\r\n"
    "Zero-width: a​b‍c﻿d; math: ∀x∈ℝ, x²≥0; ½ ¾ ⅓.\n"
)


def _natural_corpus() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "engine").glob("*.py")) + [root / "README.md"]
    text = "".join(f.read_text(encoding="utf-8") for f in files if f.exists())
    return text + _MULTILINGUAL * 3


def test_natural_corpus_sessions(tok):
    """Code and prose grown at random character positions (cuts anywhere, mid-word, mid
    whitespace run, between a base and its combining mark, inside emoji sequences)."""
    enc = _encoder(tok)
    corpus = _natural_corpus()
    rng = random.Random(69)
    checked = 0
    for session in range(24):
        start = rng.randrange(0, len(corpus) - 30000)
        end = start + rng.randrange(2000, 12000)
        text = corpus[start:end]
        for _ in range(6):
            assert enc.encode(text) == tuple(tok.encode(text)), (session, len(text))
            checked += 1
            step = rng.randrange(1, 3000)
            text = text + corpus[end : end + step]
            end += step
    # Multilingual text cut at every character position, then extended by itself.
    ml = _MULTILINGUAL
    for cut in range(1, len(ml)):
        enc.encode(ml[:cut])
        text = ml[:cut] + ml[cut:] + ml[: cut % 50]
        assert enc.encode(text) == tuple(tok.encode(text)), cut
        checked += 1
    assert enc.hits > checked // 2


def test_replay_like_sessions_with_decoded_outputs(tok):
    """prompt + decode(output ids) + new input text, as the replay sends; outputs are
    arbitrary vocabulary ids (byte-level pieces, partial UTF-8 -> replacement characters,
    added tokens) and real code continuations."""
    enc = _encoder(tok)
    rng = random.Random(4669)
    vocab = len(tok)
    lines = [
        f"def case_{i}(items):\n    return sorted(set(items))  # {i} 9e1d77ab\n" for i in range(4000)
    ]
    checked = 0
    for session in range(16):
        at = rng.randrange(0, 3000)
        text = "".join(lines[at : at + 120])
        for rnd in range(7):
            assert enc.encode(text) == tuple(tok.encode(text)), (session, rnd)
            checked += 1
            if rng.random() < 0.5:
                ids = [rng.randrange(vocab) for _ in range(rng.randrange(1, 200))]
            else:  # a plausible continuation: re-encoded corpus, decoded back
                ids = list(tok.encode("".join(lines[at + 120 + rnd * 7 : at + 127 + rnd * 7])))
            out = tok.decode(ids)
            text = text + out + "".join(lines[at + 200 + rnd * 9 : at + 200 + rnd * 9 + rng.randrange(0, 9)])
    assert checked == 16 * 7
    assert enc.hits >= 16 * 5
