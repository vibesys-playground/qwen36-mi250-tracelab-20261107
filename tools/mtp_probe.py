"""Offline acceptance probe for the checkpoint's MTP head (plan 5, stage A).

    python tools/mtp_probe.py [--max-new 256] [--sessions 4] [--rounds 2]

Loads the model as the engine does (``Generator.load``, sharded over every
visible GPU, prefix cache off) and the ``mtp.*`` head (``engine.mtp``). For
each prompt it greedy-decodes one request at a time with the engine's own
``prefill``/``forward``. At every step the MTP head reads the base model's
final normed hidden state at position t and the greedy token t+1, and proposes
t+2; the probe counts how often that proposal equals the base model's next
greedy token (the acceptance of one draft under exact greedy verification).
The MTP head keeps its own KV cache over the whole prompt and output, as a
deployed drafter would.

Prompts:
- synthetic: rebuilt as Request Factory's session_runner builds them for the
  replay (.vibesys/tasks/mi2508x-tracelab/replay.py and Request Factory's
  src/tokens.rs at the pinned revision): the same corpus lines with a fresh
  nonce, each line tokenized without special tokens and concatenated, each
  session reading the pool from offset ordinal*9973; round r's prompt is the
  previous prompt, the model's real output, then input_len new pool tokens.
  Lengths come from recorded_sessions.csv; outputs are capped at --max-new
  and run past EOS, as the replay's ignore_eos does.
- natural: a few chat prompts (code and prose) through the checkpoint's chat
  template with thinking off, stopping at EOS or --max-new.

It also times, with all devices synchronized: one decode step (1 token), one
MTP head step, and every 32 steps a 2-token forward on a copy of the cache (the
cost of a verify step that feeds [token, draft]).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import secrets
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.generator import CHECKPOINT, Generator  # noqa: E402
from engine.mtp import MTPHead  # noqa: E402

TRACE = ROOT / ".vibesys/tasks/mi2508x-tracelab/recorded_sessions.csv"
SESSION_STRIDE = 9_973  # Request Factory src/executor/session.rs: seed offset per session ordinal

NATURAL = [
    "Write a Python function that parses an ISO-8601 timestamp string into a timezone-aware "
    "datetime, with input validation, a docstring and three doctest examples.",
    "Implement a thread-safe LRU cache class in Python with get, put and a max size, and "
    "briefly explain the design choices.",
    "Here is a buggy function:\n\n```python\ndef mean(xs):\n    total = 0\n    for i in range(1, len(xs)):\n"
    "        total += xs[i]\n    return total / len(xs)\n```\n\nFind the bugs, fix them, and add "
    "unit tests with pytest.",
    "Explain the difference between TCP and UDP to a new backend engineer, with concrete "
    "examples of when to use each.",
    "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
]


def sync() -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - start) * 1e3


def synthetic_pool(tokenizer, limit: int) -> list[int]:
    nonce = secrets.token_hex(16)
    pool: list[int] = []
    index = 0
    while len(pool) < limit:
        # replay.py writes these two lines per index; session_runner reads lines without '\n'.
        for line in (f"def case_{index}(items):", f"    return sorted(set(items))  # {index} {nonce}"):
            pool.extend(tokenizer.encode(line, add_special_tokens=False))
        index += 1
    return pool[:limit]


def synthetic_rounds(tokenizer, sessions: int, rounds: int):
    """(label, prompt_fn, output_len) per round; prompt_fn takes the previous round's context."""
    with TRACE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_session: dict[str, list[dict]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).append(row)
    chosen = sorted(by_session, key=int)[:sessions]
    need = max(len(chosen) - 1, 0) * SESSION_STRIDE + sum(
        int(r["input_len"]) + int(r["prefix_len"]) for s in chosen for r in by_session[s][:rounds]
    )
    pool = synthetic_pool(tokenizer, need + 1)
    plans = []
    for ordinal, session in enumerate(chosen):
        steps = sorted(by_session[session], key=lambda r: int(r["round_idx"]))[:rounds]
        plans.append((ordinal, session, steps))
    return pool, plans


class Probe:
    def __init__(self, generator: Generator, head: MTPHead, timing_every: int) -> None:
        self.g = generator
        self.head = head
        self.lm = generator.model
        self.embed = self.lm.get_input_embeddings()
        self.lm_head = self.lm.get_output_embeddings()
        self.timing_every = timing_every
        # (input, output) of the base model's final norm: pre-norm and normed last hidden states.
        self.captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        text = self.lm.model
        norm = getattr(text, "norm", None) or text.language_model.norm
        norm.register_forward_hook(lambda _m, i, out: self.captured.append((i[0], out)))
        self.decode_ms: list[float] = []
        self.mtp_ms: list[float] = []
        self.verify2_ms: list[float] = []

    def take_hidden(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(normed, pre-norm) hidden states captured since the last call, ``[1, L, D]`` each."""
        device = self.head.device
        normed = torch.cat([o.to(device) for _, o in self.captured], dim=1)
        raw = torch.cat([i.to(device) for i, _ in self.captured], dim=1)
        self.captured.clear()
        return normed, raw

    def mtp(self, hidden, ids, positions, cache):
        return self.head(hidden, ids, positions, cache, self.embed, self.lm_head)[0, -1]

    @torch.inference_mode()
    def run(self, prompt: list[int], max_new: int, stop_at_eos: bool) -> dict:
        """Greedy-decode ``prompt``; the MTP head (normed hidden, as SGLang/vLLM wire it) drafts each
        next-but-one token. A second head cache fed the pre-norm hidden is a wiring control only."""
        g = self.g
        cache = DynamicCache(config=g.model.config)
        self.captured.clear()
        logits = g.prefill(tuple(prompt), 0, cache)
        normed, raw = self.take_hidden()  # positions 0..n-1
        n = len(prompt)
        assert normed.shape[1] == n, (normed.shape, n)
        token = int(torch.argmax(logits))
        check = int(torch.argmax(self.lm_head(normed[:, -1:].to(self.lm_head.weight.device))))
        hidden_ok = check == token

        caches = {"normed": self.head.new_cache(), "prenorm": self.head.new_cache()}
        sources = {"normed": normed, "prenorm": raw}
        ids = torch.tensor([list(prompt[1:]) + [token]])
        positions = torch.arange(n)[None]
        drafts: dict[str, torch.Tensor] = {}
        for name, mtp_cache in caches.items():
            for start in range(0, n, 2048):
                end = min(start + 2048, n)
                last = self.mtp(sources[name][:, start:end], ids[:, start:end],
                                positions[:, start:end], mtp_cache)
            drafts[name] = last

        output = [token]
        proposed = 0
        accepted = {"normed": 0, "prenorm": 0, "normed_top5": 0}
        position = n
        while len(output) < max_new and not (stop_at_eos and token in g.eos_ids):
            draft = int(torch.argmax(drafts["normed"]))
            if self.timing_every and len(output) % self.timing_every == 0:
                trial = copy.deepcopy(cache)
                _, ms = timed(lambda: g.forward([token, draft], trial))
                self.verify2_ms.append(ms)
                del trial
                self.captured.clear()
            logits, ms = timed(lambda: g.forward([token], cache))
            self.decode_ms.append(ms)
            normed, raw = self.take_hidden()
            nxt = int(torch.argmax(logits))
            proposed += 1
            accepted["normed"] += int(draft == nxt)
            accepted["normed_top5"] += int(nxt in torch.topk(drafts["normed"], 5).indices.tolist())
            accepted["prenorm"] += int(int(torch.argmax(drafts["prenorm"])) == nxt)
            nxt_ids, pos = torch.tensor([[nxt]]), torch.tensor([[position]])
            drafts["normed"], ms = timed(lambda: self.mtp(normed, nxt_ids, pos, caches["normed"]))
            self.mtp_ms.append(ms)
            drafts["prenorm"] = self.mtp(raw, nxt_ids, pos, caches["prenorm"])
            output.append(nxt)
            token = nxt
            position += 1
        return {
            "prompt_tokens": n,
            "output_tokens": len(output),
            "proposed": proposed,
            "accepted": accepted["normed"],
            "acceptance": accepted["normed"] / proposed if proposed else None,
            "top5_hit": accepted["normed_top5"] / proposed if proposed else None,
            "prenorm_control_acceptance": accepted["prenorm"] / proposed if proposed else None,
            "hidden_check": hidden_ok,
            "output": output,
        }


def summarize(name: str, results: list[dict]) -> dict:
    proposed = sum(r["proposed"] for r in results)
    accepted = sum(r["accepted"] for r in results)
    per = [round(r["acceptance"], 4) for r in results if r["acceptance"] is not None]
    weighted = lambda key: round(  # noqa: E731
        sum(r[key] * r["proposed"] for r in results if r[key] is not None) / proposed, 4
    ) if proposed else None
    return {
        "top5_hit": weighted("top5_hit"),
        "prenorm_control_acceptance": weighted("prenorm_control_acceptance"),
        "set": name,
        "prompts": len(results),
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": round(accepted / proposed, 4) if proposed else None,
        "per_prompt": per,
    }


def ms_stats(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "p50": round(statistics.median(ordered), 2),
        "p10": round(ordered[len(ordered) // 10], 2),
        "p90": round(ordered[(len(ordered) * 9) // 10], 2),
        "mean": round(statistics.fmean(ordered), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timing-every", type=int, default=32)
    parser.add_argument("--model", default=CHECKPOINT["model"], help="hub id or local directory")
    parser.add_argument("--revision", default=CHECKPOINT["revision"])
    args = parser.parse_args()

    t0 = time.perf_counter()
    g = Generator.load(
        args.model, args.revision or None,
        max_model_len=262144, chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0,
    )
    text_config = getattr(g.model.config, "text_config", g.model.config)
    device = g.model.get_output_embeddings().weight.device
    head = MTPHead.load(args.model, args.revision or None, text_config, device)
    print(json.dumps({"event": "loaded", "seconds": round(time.perf_counter() - t0, 1),
                      "mtp_device": str(device), "devices": torch.cuda.device_count(),
                      "experts_impl": getattr(text_config, "_experts_implementation", None)}), flush=True)
    probe = Probe(g, head, args.timing_every)
    tok = g.tokenizer

    synthetic = []
    pool, plans = synthetic_rounds(tok, args.sessions, args.rounds)
    for ordinal, session, steps in plans:
        cursor = (ordinal * SESSION_STRIDE) % len(pool)
        context: list[int] = []
        for step in steps:
            prefix = min(int(step["prefix_len"]), len(context))
            take = int(step["input_len"]) + int(step["prefix_len"]) - prefix
            fresh = [pool[(cursor + i) % len(pool)] for i in range(take)]
            cursor = (cursor + take) % len(pool)
            prompt = context[:prefix] + fresh
            out_len = min(int(step["output_len"]), args.max_new)
            r = probe.run(prompt, out_len, stop_at_eos=False)
            r.update(set="synthetic", session=session, round=int(step["round_idx"]))
            synthetic.append(r)
            context = prompt + r["output"]
            print(json.dumps({k: v for k, v in r.items() if k != "output"}
                             | {"text_head": tok.decode(r["output"][:48])}), flush=True)

    natural = []
    for i, text in enumerate(NATURAL):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        prompt = tok.encode(rendered, add_special_tokens=False)
        r = probe.run(prompt, args.max_new, stop_at_eos=True)
        r.update(set="natural", index=i)
        natural.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "output"}
                         | {"text_head": tok.decode(r["output"][:48])}), flush=True)

    summary = {
        "event": "summary",
        "synthetic": summarize("synthetic", synthetic),
        "natural": summarize("natural", natural),
        "all": summarize("all", synthetic + natural),
        "hidden_check_all": all(r["hidden_check"] for r in synthetic + natural),
        "decode_step_ms": ms_stats(probe.decode_ms),
        "mtp_step_ms": ms_stats(probe.mtp_ms),
        "verify2_step_ms": ms_stats(probe.verify2_ms),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
