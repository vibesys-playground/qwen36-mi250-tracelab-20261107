"""Served A/B of engine settings: by default the replica layouts 4x2, 2x4 and 1x8 GCDs.

    python tools/serve_probe.py [--sessions 8] [--rounds 3] [--first 2000] [--grow 300] [--max-new 128]
        [--concurrency N] [--settings ENGINE_REPLICAS=4,ENGINE_FAST=0,KEY=V+KEY=V,...]

For each setting it starts ``python -m engine.server`` exactly as the task does,
waits for /health, and runs ``--sessions`` multi-round sessions, ``--concurrency``
at a time (all by default), ``--rounds`` each (fractional: some one more):
streaming /v1/completions with token ids, temperature 0, ignore_eos and exact
max_tokens; round r+1's prompt is round r's prompt, its output and ``--grow``
new tokens, so the prefix cache is exercised. Prompt text is the source of the
installed Transformers package (nothing from the benchmark). It prints, per
setting, output tokens/s over the wall time, TPOT and TTFT percentiles, cached
prompt tokens and /health, then (for two settings) how many requests produced
identical tokens in both. ``--corpus synthetic`` uses the replay's own synthetic
code lines instead of the Transformers source. A hint for the lead, not the
task's metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def token_pool(model: str, revision: str | None, need: int, corpus: str = "source") -> list[int]:
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    if corpus == "synthetic":
        # The TraceLab replay's token pool: Request Factory's synthetic coding lines (two per
        # index, a fixed nonce here), each tokenized without special tokens, concatenated.
        ids = []
        index = 0
        while len(ids) < need:
            for line in (f"def case_{index}(items):", f"    return sorted(set(items))  # {index} 0f3a9c"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            index += 1
        return ids
    files = sorted(Path(transformers.__file__).parent.glob("models/*/modeling_*.py"))
    ids: list[int] = []
    for f in files:
        ids.extend(tok.encode(f.read_text()))
        if len(ids) >= need:
            break
    return ids


def stream(url: str, body: dict) -> dict:
    start = time.perf_counter()
    ids: list[int] = []
    times: list[float] = []
    usage = {}
    with httpx.stream("POST", url + "/v1/completions", json=body, timeout=3600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                got = choice.get("token_ids") or []
                ids.extend(got)
                times.extend([time.perf_counter()] * len(got))
    return {"ids": ids, "times": times, "start": start, "usage": usage}


def rounds_of(index: int, args) -> int:
    """Session ``index``'s rounds: ``--rounds`` may be fractional (6.3: some sessions 6, some 7)."""
    return max(1, math.floor((index + 1) * args.rounds) - math.floor(index * args.rounds))


def session(url: str, pool: list[int], index: int, args) -> list[dict]:
    rounds = rounds_of(index, args)
    span = args.first + rounds * args.grow
    stride = args.first + math.ceil(args.rounds) * args.grow  # sessions do not share tokens
    offset = (index * stride) % max(1, len(pool) - span)
    fresh = pool[offset : offset + span]
    prompt = fresh[: args.first]
    used = args.first
    out = []
    for r in range(rounds):
        body = {
            "prompt": prompt,
            "max_tokens": args.max_new,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        res = stream(url, body)
        assert len(res["ids"]) == args.max_new, (len(res["ids"]), args.max_new)
        res["key"] = f"s{index}r{r}"
        out.append(res)
        prompt = prompt + res["ids"] + fresh[used : used + args.grow]
        used += args.grow
    return out


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def serve_and_run(setting: str | None, port: int, args, pool) -> dict[str, list[int]]:
    env = dict(os.environ)
    env.pop("ENGINE_FAST", None)
    if setting is not None and "=" in setting:  # KEY=VALUE[+KEY=VALUE...]
        for pair in setting.split("+"):
            key, _, value = pair.partition("=")
            env[key] = value
    elif setting is not None:
        env["ENGINE_FAST"] = setting
    name = "".join(c if c.isalnum() else "_" for c in setting or "default")
    log = Path(f"/tmp/serve-probe-{name}.log")
    with log.open("w") as sink:
        proc = subprocess.Popen(
            [sys.executable, "-m", "engine.server", "--host", "127.0.0.1", "--port", str(port),
             *(["--model", args.model] if args.model else [])],
            cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT,
        )
    url = f"http://127.0.0.1:{port}"
    try:
        t0 = time.perf_counter()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}:\n{log.read_text()[-4000:]}")
            try:
                if httpx.get(url + "/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(2)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency or args.sessions) as ex:
            results = [r for s in ex.map(lambda i: session(url, pool, i, args), range(args.sessions)) for r in s]
        wall = time.perf_counter() - t0
        health = httpx.get(url + "/health", timeout=30).json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    tpot = [(r["times"][-1] - r["times"][0]) * 1e3 / (len(r["times"]) - 1)
            for r in results if len(r["times"]) > 1]
    ttft = [(r["times"][0] - r["start"]) * 1e3 for r in results if r["times"]]
    tokens = sum(len(r["ids"]) for r in results)
    cached = sum(int((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                 for r in results)
    engine_lines = [x for x in log.read_text().splitlines() if x.startswith("engine:")]
    print(json.dumps({
        "event": "setting", "setting": setting or "default", "load_s": round(load_s, 1),
        "wall_s": round(wall, 2), "requests": len(results), "output_tokens": tokens,
        "output_tok_per_s": round(tokens / wall, 2),
        "tpot_ms_p50": pct(tpot, 0.5), "tpot_ms_p90": pct(tpot, 0.9),
        "ttft_ms_p50": pct(ttft, 0.5), "ttft_ms_p90": pct(ttft, 0.9),
        "cached_prompt_tokens": cached, "health": health, "engine": engine_lines[:4],
    }), flush=True)
    return {r["key"]: r["ids"] for r in results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--rounds", type=float, default=3, help="per session; may be fractional")
    parser.add_argument(
        "--concurrency", type=int, default=None, help="sessions in flight at once (default: all)"
    )
    parser.add_argument("--first", type=int, default=2000)
    parser.add_argument("--grow", type=int, default=300)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument(
        "--corpus",
        choices=("source", "synthetic"),
        default="source",
        help="prompt text: the installed Transformers source, or the replay's synthetic code lines",
    )
    parser.add_argument(
        "--settings",
        default="ENGINE_REPLICAS=4,ENGINE_REPLICAS=2,ENGINE_REPLICAS=1",
        help="comma-separated, in order: an ENGINE_FAST value, 'default', or KEY=VALUE[+KEY=VALUE]",
    )
    args = parser.parse_args()
    from engine.generator import CHECKPOINT

    model = args.model or CHECKPOINT["model"]
    revision = None if args.model else CHECKPOINT["revision"]
    need = args.sessions * (args.first + math.ceil(args.rounds) * args.grow) + 1
    pool = token_pool(model, revision, need, args.corpus)
    print(json.dumps({"event": "shape", **vars(args)}), flush=True)
    outputs = {}
    for setting in args.settings.split(","):
        outputs[setting] = serve_and_run(None if setting == "default" else setting, args.port, args, pool)
    names = list(outputs)
    if len(names) == 2:
        a, b = outputs[names[0]], outputs[names[1]]
        same = sum(a[k] == b[k] for k in a)
        first_div = []
        for k in a:
            n = next((i for i, (x, y) in enumerate(zip(a[k], b[k])) if x != y), None)
            if n is not None:
                first_div.append(n)
        print(json.dumps({"event": "agreement", "identical_requests": same, "requests": len(a),
                          "first_divergence_positions": sorted(first_div)[:16]}), flush=True)


if __name__ == "__main__":
    main()
