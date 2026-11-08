"""Plan 10's kill test: is fast-path prefill fixed-cost bound, so that packing pays?

    python tools/prefill_pack_bench.py [--reps 3]

Loads the checkpoint as the server does (``Generator.load``: BF16 over every
visible GPU, the fast path with its captured decode graphs), then on
``FastModel`` directly:

1. ``prefill`` alone at 1, 64, 256, 700 and 2048 tokens (fresh slot, median of
   ``--reps``), and a least-squares fit ``ms = a + b * tokens`` over T >= 64.
2. Packed against serial: n segments of T tokens as n ``prefill`` calls and as
   one ``prefill_packed`` pass, for 2x700, 4x700, 8x700, 4x2048 and 8x64. Each
   packed result is checked against the serial one: max |logit diff|, top-1
   agreement, and the logits of one decode step after it (the conv, recurrent
   and KV state the pass left).
3. Requests: ``Generator.start`` per prompt against ``Generator.start_many``
   (chunking, snapshot boundaries and the final one-token pass included) for
   4 prompts of 701 tokens and 8 mixed prompts.

Prompts are the source of the installed Transformers package (nothing from the
benchmark). It prints ``RESULT`` lines and a verdict: KILL when 4x700 packed is
less than 1.8x faster than four serial prefills (exit 3), else PASS (exit 0).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.generator import CHECKPOINT, Generator  # noqa: E402
from engine.protocol import Sampling  # noqa: E402

KILL_SPEEDUP = 1.8
BASE_SLOT = 16  # direct FastModel tests use slots 16..; Generator.start pops 0.. first


def _sync() -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def _prompts(generator: Generator, lengths: list[int], skip: int = 0) -> list[list[int]]:
    import transformers

    files = sorted(Path(transformers.__file__).parent.glob("models/*/modeling_*.py"))[skip:]
    out, index = [], 0
    for length in lengths:
        while True:
            ids = list(generator.tokenizer.encode(files[index].read_text()))
            index += 1
            if len(ids) >= length:
                out.append(ids[:length])
                break
    return out


def _timed(fn, reps: int, prepare=None) -> tuple[float, object]:
    """Median ms of ``fn`` over ``reps`` runs (``prepare`` untimed before each); its last result."""
    times, result = [], None
    for _ in range(reps):
        if prepare is not None:
            prepare()
        _sync()
        t0 = time.perf_counter()
        result = fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--max-len", type=int, default=262144)
    parser.add_argument("--model", default=None, help="another checkpoint (a CPU smoke test)")
    args = parser.parse_args()
    print("torch", torch.__version__, "hip", torch.version.hip, "gpus", torch.cuda.device_count(), flush=True)
    t0 = time.perf_counter()
    generator = Generator.load(
        args.model or CHECKPOINT["model"],
        None if args.model else CHECKPOINT["revision"],
        max_model_len=args.max_len,
        chunk_tokens=args.chunk,
        prefix_cache_tokens=0,
        prefix_cache_entries=0,
    )
    fast = generator.fast
    assert fast is not None, "the fast path did not load"
    print(f"RESULT load_s={time.perf_counter() - t0:.1f} graphs={fast.use_graphs} graph_error={fast.graph_error}", flush=True)
    reps = args.reps

    def reset(slots) -> None:
        for s in slots:
            fast.reset(s)

    # Warm up: kernels compile, BLAS handles and allocator settle.
    warm = _prompts(generator, [2048, 700, 700], skip=200)
    for _ in range(2):
        reset(range(BASE_SLOT, BASE_SLOT + 3))
        fast.prefill(BASE_SLOT, warm[0], 0)
        fast.prefill(BASE_SLOT + 1, warm[1][:64], 0)
        fast.prefill(BASE_SLOT + 1, warm[1][64:65], 64)
        fast.prefill_packed([(BASE_SLOT + 1, warm[1], 0), (BASE_SLOT + 2, warm[2], 0)])
    _sync()
    reset(range(BASE_SLOT, BASE_SLOT + 3))

    # ---------------------------------------------------------------- 1. prefill alone
    singles = {}
    source = _prompts(generator, [2048])[0]
    for count in (1, 64, 256, 700, 2048):
        ms, _ = _timed(
            lambda count=count: fast.prefill(BASE_SLOT, source[:count], 0),
            reps,
            prepare=lambda: reset([BASE_SLOT]),
        )
        singles[count] = ms
        print(f"RESULT prefill_ms tokens={count} ms={ms:.1f} ms_per_token={ms / count:.3f}", flush=True)
    xs = [t for t in singles if t >= 64]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(singles[t] for t in xs) / len(xs)
    b = sum((t - mean_x) * (singles[t] - mean_y) for t in xs) / sum((t - mean_x) ** 2 for t in xs)
    a = mean_y - b * mean_x
    print(
        f"RESULT prefill_fit ms = {a:.1f} + {b:.4f} * tokens  "
        f"(fixed share of a 700-token pass {a / (a + 700 * b):.0%})",
        flush=True,
    )

    # ---------------------------------------------------------------- 2. packed vs serial
    speedups = {}
    cases = [(2, 700), (4, 700), (8, 700), (4, 2048), (8, 64)]
    for n, count in cases:
        prompts = _prompts(generator, [count + 1] * n, skip=20)
        slots = list(range(BASE_SLOT, BASE_SLOT + n))
        segs = [(s, p[:count], 0) for s, p in zip(slots, prompts, strict=True)]
        serial_ms, serial = _timed(
            lambda segs=segs: [fast.prefill(s, ids, st).clone() for s, ids, st in segs],
            reps,
            prepare=lambda slots=slots: reset(slots),
        )
        nxt = [p[count] for p in prompts]
        serial_next = fast.decode(nxt, slots, [count] * n).float().clone()
        packed_ms, packed = _timed(
            lambda segs=segs: fast.prefill_packed(segs).clone(),
            reps,
            prepare=lambda slots=slots: reset(slots),
        )
        packed_next = fast.decode(nxt, slots, [count] * n).float().clone()
        serial_t = torch.stack(serial).float()
        diff = float((packed.float() - serial_t).abs().max())
        top1 = int((packed.argmax(-1) == serial_t.argmax(-1)).sum())
        ndiff = float((packed_next - serial_next).abs().max())
        ntop1 = int((packed_next.argmax(-1) == serial_next.argmax(-1)).sum())
        speedup = serial_ms / packed_ms
        speedups[(n, count)] = speedup
        print(
            f"RESULT packed n={n} tokens={count} serial_ms={serial_ms:.1f} packed_ms={packed_ms:.1f} "
            f"speedup={speedup:.2f}x logits_max_diff={diff:.3f} top1_equal={top1}/{n} "
            f"next_step_max_diff={ndiff:.3f} next_top1_equal={ntop1}/{n}",
            flush=True,
        )
        reset(slots)

    # ---------------------------------------------------------------- 3. requests
    sampling = Sampling(max_tokens=8, temperature=0.0, ignore_eos=True)
    for label, lengths in (("4x701", [701] * 4), ("mixed8", [701, 1500, 2900, 300, 701, 64, 4000, 900])):
        prompts = [tuple(p) for p in _prompts(generator, lengths, skip=60)]
        _sync()
        t0 = time.perf_counter()
        solo = []
        for p in prompts:
            g = generator.start(p, sampling)
            solo.append(g)
        _sync()
        serial_ms = (time.perf_counter() - t0) * 1e3
        solo_logits = torch.stack([g.logits for g in solo]).float()
        for g in solo:
            generator.finished(g)
        generator._reclaim()
        _sync()
        t0 = time.perf_counter()
        packed = generator.start_many([(p, sampling) for p in prompts])
        _sync()
        packed_ms = (time.perf_counter() - t0) * 1e3
        errors = [r for r in packed if isinstance(r, Exception)]
        assert not errors, errors
        packed_logits = torch.stack([g.logits for g in packed]).float()
        top1 = int((packed_logits.argmax(-1) == solo_logits.argmax(-1)).sum())
        diff = float((packed_logits - solo_logits).abs().max())
        for g in packed:
            generator.finished(g)
        generator._reclaim()
        print(
            f"RESULT requests {label} tokens={sum(lengths)} start_serial_ms={serial_ms:.1f} "
            f"start_many_ms={packed_ms:.1f} speedup={serial_ms / packed_ms:.2f}x "
            f"logits_max_diff={diff:.3f} top1_equal={top1}/{len(prompts)} "
            f"passes={generator.prefill_passes} segments={generator.prefill_segments}",
            flush=True,
        )

    key = speedups[(4, 700)]
    verdict = "PASS" if key >= KILL_SPEEDUP else "KILL"
    print(f"RESULT kill_test 4x700_speedup={key:.2f}x threshold={KILL_SPEEDUP}x verdict={verdict}", flush=True)
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
