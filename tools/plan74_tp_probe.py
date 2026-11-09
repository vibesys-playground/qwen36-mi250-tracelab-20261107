"""Plan 74: the tiny checkpoint's tensor-parallel load on two GPUs, one variant per process.

    python tools/plan74_tp_probe.py tp <max_len>      # TPModel + capture_all, tracebacks printed
    python tools/plan74_tp_probe.py margin            # Transformers path: batched vs solo logits

``tp``: loads the tiny checkpoint on the Transformers path, builds ``TPModel`` (graphs and Triton
as served) and captures; every exception inside a warm or captured pass prints its traceback.
Then one prefill and three decode steps against the one-device fast path's argmax.
``margin``: the session-tail test's two prompts decoded together and alone; per step the max
logit difference and the solo top-1/top-2 margin.
"""
from __future__ import annotations

import functools
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / ".vibesys" / "shared" / "cpu_check")]
os.environ["ENGINE_FAST"] = "0"

from tiny_checkpoint import write_checkpoint  # noqa: E402

from engine.fast import tp  # noqa: E402
from engine.fast.model import FastModel  # noqa: E402
from engine.generator import Generator  # noqa: E402
from engine.protocol import Sampling  # noqa: E402


def traced(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            print(f"--- traceback in {fn.__qualname__}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            raise
    return wrap


def run_tp(max_len: int) -> None:
    FastModel._pg_body = traced(FastModel._pg_body)
    tp.TPModel._warm = traced(tp.TPModel._warm)
    FastModel._capture = traced(FastModel._capture)
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(Path(tmp))
        gen = Generator.load(tmp, None, max_model_len=max_len, chunk_tokens=8,
                             prefix_cache_tokens=0, prefix_cache_entries=0)
        one = FastModel(gen.model, slots=4, max_len=max_len)
        prompt = [(7 * i + 3) % 300 + 5 for i in range(100)]
        ref = [int(one.prefill(0, prompt, 0).argmax())]
        for s in range(3):
            ref.append(int(one.decode([ref[-1]], [0], [len(prompt) + s])[0].argmax()))
        print(f"one-device: {ref}", flush=True)
        t0 = time.time()
        model = tp.TPModel(gen.model, slots=4, max_len=max_len)
        print(f"TPModel: triton={model.use_triton} graphs={model.use_graphs} "
              f"pg_supported={[m.pg_supported() for m in model.ranks]} ll={model.group.ll is not None}", flush=True)
        try:
            model.capture_all(decode=True, verify=False)
        except Exception:
            print("--- capture_all raised:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            os._exit(3)
        print(f"captured in {time.time() - t0:.1f}s: graph_error={model.graph_error} "
              f"pg_error={model.pg_error} pg={sorted(model.pg)}", flush=True)
        got = [int(model.prefill(0, prompt, 0).argmax())]
        for s in range(3):
            got.append(int(model.decode([got[-1]], [0], [len(prompt) + s])[0].argmax()))
        torch.cuda.synchronize()
        print(f"tp: {got} pg_replays={model.pg_replays} equal={got == ref}", flush=True)


def run_margin() -> None:
    from conftest import CHUNK  # noqa: F401
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(Path(tmp))
        torch.manual_seed(0)
        loaded = Generator.load(tmp, None, max_model_len=4096, chunk_tokens=8,
                                prefix_cache_tokens=1 << 20, prefix_cache_entries=0)

        def new_tokens(seed, count):
            return [7 + (seed * 31 + i * 13) % 250 for i in range(count)]

        def generator():
            return Generator(loaded.model, loaded.tokenizer, loaded.eos_ids, max_model_len=4096,
                             chunk_tokens=8, prefix_cache_tokens=1 << 20, prefix_cache_entries=64,
                             prefix_cache_bytes=None)

        def run(gen, prompts, n):
            sampling = Sampling(max_tokens=n, temperature=0.0, ignore_eos=True)
            live = [gen.start(tuple(p), sampling) for p in prompts]
            logits = [[] for _ in prompts]
            outs = [[] for _ in prompts]
            running = list(range(len(prompts)))
            while running:
                waiting = []
                for i in running:
                    logits[i].append(live[i].logits.float().flatten().cpu().clone())
                    step = live[i].next_step()
                    outs[i].append(step.token_id)
                    if step.finish is None:
                        waiting.append(i)
                    else:
                        live[i].retire()
                running = waiting
                if running:
                    gen.decode([live[i] for i in running])
            return outs, logits

        prompts = [new_tokens(30, 5), new_tokens(31, 25)]
        outs, logits = run(generator(), prompts, 4)
        for i, p in enumerate(prompts):
            souts, slog = run(generator(), [p], 4)
            print(f"prompt {i}: batched {outs[i]} solo {souts[0]}", flush=True)
            for step, (a, b) in enumerate(zip(logits[i], slog[0])):
                top = torch.topk(b, 2).values
                print(f"  step {step}: max|diff| {float((a - b).abs().max()):.5f} scale "
                      f"{float(b.abs().max()):.3f} solo margin {float(top[0] - top[1]):.5f} "
                      f"argmax {int(a.argmax())}/{int(b.argmax())}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "tp":
        run_tp(int(sys.argv[2]))
    else:
        run_margin()
