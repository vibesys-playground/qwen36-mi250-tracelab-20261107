"""Plan 13: teacher-forced top-1 agreement of ENGINE_TP=2 against ENGINE_TP=1 on one replica.

    ENGINE_TP=1 python tools/tp_parity.py --mode ref --out /tmp/tp_ref.pt
    ENGINE_TP=2 python tools/tp_parity.py --mode check --ref /tmp/tp_ref.pt

``ref`` loads the replica as the server does, prefills 8 prompts (4 replay-like synthetic
coding prompts of the recorded first-round lengths, 4 chat prompts), and decodes ``--steps``
greedy tokens per prompt with captured plain decode steps, 4 prompts per batch; it saves the
prompts, every step's input token, greedy pick and top-2 margin. ``check`` (another process:
both models do not fit at once) feeds the same tokens (teacher forcing) and counts how often
its greedy pick equals the reference's, the prefill picks included; it also reports the
free-running identical prefix and the margins at disagreements. Prints ``RESULT`` lines.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ref", "check", "repeat"), required=True)
    parser.add_argument("--out", default="/tmp/tp_ref.pt")
    parser.add_argument("--ref", default="/tmp/tp_ref.pt")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--lengths", default=None,
                        help="LO:HI:N: N synthetic prompts evenly spaced LO..HI tokens instead of the default 8")
    parser.add_argument("--one-pass", action="store_true",
                        help="prefill each prompt in one pass (the server's ENGINE_PREFILL_ONE_PASS_MAX path)")
    parser.add_argument("--dump-fla", default=None, help="write fla's autotune picks (JSON) here")
    args = parser.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    os.environ.setdefault("ENGINE_MTP", "0")  # plain decode steps; the head does not change picks

    import torch

    from engine.generator import CHECKPOINT, Generator

    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    tok = g.tokenizer
    print(f"RESULT {args.mode} tp={os.environ.get('ENGINE_TP', '1')} graphs={fast.use_graphs} "
          f"graph_error={fast.graph_error} kv_pages={fast.pool.pages}", flush=True)
    print(f"RESULT fla_pins={getattr(fast, 'fla_pins', None)} "
          f"prefill_fused={os.environ.get('ENGINE_PREFILL_FUSED', '1')}", flush=True)

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        if args.one_pass:
            return fast.prefill(slot, prompt, 0)
        body = len(prompt) - 1
        position = 0
        while position < body:
            end = min((position // args.chunk + 1) * args.chunk, body)
            fast.prefill(slot, prompt[position:end], position, prompt[end])
            position = end
        return fast.prefill(slot, prompt[body:], body)

    if args.mode in ("ref", "repeat"):
        ids: list[int] = []
        i = 0
        spread = [int(x) for x in args.lengths.split(":")] if args.lengths else None
        while len(ids) < 9973 * (spread[2] if spread else 4) + 20000:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} 5f0c2a9e"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        with (ROOT / ".vibesys/tasks/mi2508x-tracelab/recorded_sessions.csv").open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["round_idx"]) == 0]
        rows.sort(key=lambda r: int(r["session_id"]))
        prompts = [ids[9973 * k : 9973 * k + int(r["input_len"])] for k, r in enumerate(rows[:4])]
        if spread:
            lo, hi, n = spread
            prompts = [ids[9973 * k : 9973 * k + lo + round(k * (hi - lo) / max(1, n - 1))] for k in range(n)]
        for text in () if spread else (
            "Write a Python function that parses an ISO-8601 timestamp into a timezone-aware datetime.",
            "Implement a thread-safe LRU cache class in Python and explain the design.",
            "Explain the difference between TCP and UDP to a new backend engineer.",
            "Write a short story (about 200 words) about a lighthouse keeper.",
        ):
            prompts.append(tok.encode(
                tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False),
                add_special_tokens=False,
            ))
    else:
        saved = torch.load(args.ref)
        prompts = saved["prompts"]
    if args.mode != "check":
        saved = None
    # ``repeat``: the reference and the check in this one process (greedy repeatability).
    for mode in (["ref", "check"] if args.mode == "repeat" else [args.mode]):
        groups = [list(range(i, min(i + 4, len(prompts)))) for i in range(0, len(prompts), 4)]
        record = {"prompts": prompts, "inputs": {}, "picks": {}, "margins": {}}
        agree = total = 0
        close_misses = []
        free_identical = []
        for group in groups:
            first = []
            for slot, k in enumerate(group):
                logits = prefill(slot, prompts[k]).float()
                first.append(logits)
            lasts = [int(x.argmax()) for x in first]
            inputs = {k: [] for k in group}
            picks = {k: [lasts[j]] for j, k in enumerate(group)}
            margins = {k: [float(torch.topk(first[j], 2).values.diff().abs())] for j, k in enumerate(group)}
            if mode == "check":
                ref_picks = saved["picks"]
                for j, k in enumerate(group):
                    total += 1
                    agree += int(lasts[j] == ref_picks[k][0])
                feed = [saved["inputs"][k] for k in group]
            for step in range(args.steps):
                tokens = lasts if mode == "ref" else [feed[j][step] for j in range(len(group))]
                out = fast.decode(tokens, list(range(len(group))), [len(prompts[k]) + step for k in group]).float()
                top = torch.topk(out, 2, dim=-1).values
                lasts = out.argmax(-1).tolist()
                for j, k in enumerate(group):
                    inputs[k].append(tokens[j])
                    picks[k].append(lasts[j])
                    margins[k].append(float(top[j, 0] - top[j, 1]))
                    if mode == "check":
                        want = saved["picks"][k][step + 1]
                        total += 1
                        if lasts[j] == want:
                            agree += 1
                        else:
                            close_misses.append(round(saved["margins"][k][step + 1], 4))
            for k in group:
                record["inputs"][k] = inputs[k]
                record["picks"][k] = picks[k]
                record["margins"][k] = margins[k]
                if mode == "check":
                    ref = saved["picks"][k]
                    n = 0  # free-running: the pick equals the reference and every earlier one did
                    while n < len(ref) and picks[k][n] == ref[n] and (n == 0 or inputs[k][n - 1] == ref[n - 1]):
                        n += 1
                    free_identical.append(n)
            for slot in range(len(group)):
                fast.reset(slot)
        if mode == "ref":
            torch.save(record, args.out)
            print(f"RESULT ref saved prompts={len(prompts)} steps={args.steps} "
                  f"lengths={[len(p) for p in prompts]}", flush=True)
        else:
            print(f"RESULT teacher_forced top1_pct={100.0 * agree / total:.2f} ({agree}/{total}) "
                  f"ref_margin_at_misses={sorted(close_misses)[:20]}", flush=True)
            print(f"RESULT identical_greedy_prefix={free_identical} of {args.steps + 1}", flush=True)
            if hasattr(fast, "timeouts"):
                print(f"RESULT tp_allreduce_timeouts={fast.timeouts()}", flush=True)
            print(f"RESULT prefill_graphs={len(getattr(fast, 'pg', {}) or {})} "
                  f"replays={getattr(fast, 'pg_replays', None)} error={getattr(fast, 'pg_error', None)}",
                  flush=True)

        if mode == "ref" and args.mode == "repeat":
            saved = record

    if args.dump_fla:
        import json

        from engine.fast import fla_pin

        Path(args.dump_fla).write_text(json.dumps(fla_pin.snapshot(), indent=1, sort_keys=True))
        print(f"RESULT fla_snapshot written {args.dump_fla}", flush=True)


if __name__ == "__main__":
    main()
