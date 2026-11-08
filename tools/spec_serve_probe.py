"""Served-path A/B of MTP speculative decoding: the same workload with ENGINE_MTP=0, then =1.

    python tools/spec_serve_probe.py [--sessions 8] [--rounds 2] [--max-new 256]

For each setting it starts ``python -m engine.server`` (the router and its
replicas, as the task runs it), waits for /health, and runs replay-like
sessions concurrently (prompts rebuilt as Request Factory builds them, see
tools/mtp_probe.py; streaming /v1/completions with token ids, temperature 0,
ignore_eos, exact max_tokens; round r+1 = round r's prompt + its output + new
tokens) plus a few natural chat prompts. It prints, per setting, output
tokens/s over the wall time, TPOT and TTFT percentiles and the server's /health
counters, then how many requests produced identical tokens in both settings and
where the first divergence is. A hint for the lead, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mtp_probe import NATURAL, SESSION_STRIDE, synthetic_rounds  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from engine.generator import CHECKPOINT  # noqa: E402


def stream(url: str, path: str, body: dict) -> tuple[list[int], list[float], float]:
    """(token ids, arrival times of token chunks, start time)."""
    start = time.perf_counter()
    ids: list[int] = []
    times: list[float] = []
    with httpx.stream("POST", url + path, json=body, timeout=1800) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            for choice in chunk.get("choices", []):
                got = choice.get("token_ids") or []
                if got:
                    ids.extend(got)
                    times.extend([time.perf_counter()] * len(got))
    return ids, times, start


def run_session(url: str, pool: list[int], ordinal: int, steps: list[dict], cap: int) -> list[dict]:
    cursor = (ordinal * SESSION_STRIDE) % len(pool)
    context: list[int] = []
    out = []
    for step in steps:
        prefix = min(int(step["prefix_len"]), len(context))
        take = int(step["input_len"]) + int(step["prefix_len"]) - prefix
        fresh = [pool[(cursor + i) % len(pool)] for i in range(take)]
        cursor = (cursor + take) % len(pool)
        prompt = context[:prefix] + fresh
        n = min(int(step["output_len"]), cap)
        body = {"prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True,
                "return_token_ids": True, "stream": True, "stream_options": {"include_usage": True}}
        ids, times, start = stream(url, "/v1/completions", body)
        assert len(ids) == n, (len(ids), n)
        out.append({"key": f"s{ordinal}r{step['round_idx']}", "ids": ids, "start": start, "times": times})
        context = prompt + ids
    return out


def run_chat(url: str, index: int, text: str, cap: int) -> dict:
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": cap, "temperature": 0,
            "return_token_ids": True, "stream": True}
    ids, times, start = stream(url, "/v1/chat/completions", body)
    return {"key": f"chat{index}", "ids": ids, "start": start, "times": times}


def workload(url: str, pool, plans, cap: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(plans)) as ex:
        sessions = list(ex.map(lambda p: run_session(url, pool, p[0], p[2], cap), plans))
    with ThreadPoolExecutor(len(NATURAL)) as ex:
        chats = list(ex.map(lambda a: run_chat(url, a[0], a[1], cap), enumerate(NATURAL)))
    return [r for s in sessions for r in s] + chats, time.perf_counter() - t0


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def serve_and_run(flag: str, port: int, args, pool, plans) -> dict:
    env = {**os.environ, "ENGINE_MTP": flag}
    log = Path(f"/tmp/spec-probe-server-{flag}.log")
    with log.open("w") as sink:
        proc = subprocess.Popen([sys.executable, "-m", "engine.server", "--host", "127.0.0.1",
                                 "--port", str(port), *(["--model", args.model] if args.model else [])], cwd=ROOT, env=env, stdout=sink,
                                stderr=subprocess.STDOUT)
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
        results, wall = workload(url, pool, plans, args.max_new)
        health = httpx.get(url + "/health", timeout=30).json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    lines = [x for x in log.read_text().splitlines() if "MTP" in x]
    tpot = [(r["times"][-1] - r["times"][0]) * 1e3 / (len(r["times"]) - 1)
            for r in results if len(r["times"]) > 1]
    ttft = [(r["times"][0] - r["start"]) * 1e3 for r in results if r["times"]]
    tokens = sum(len(r["ids"]) for r in results)
    summary = {
        "event": "setting", "ENGINE_MTP": flag, "load_s": round(load_s, 1), "wall_s": round(wall, 2),
        "requests": len(results), "output_tokens": tokens,
        "output_tok_per_s": round(tokens / wall, 2),
        "tpot_ms_p50": pct(tpot, 0.5), "tpot_ms_p90": pct(tpot, 0.9),
        "ttft_ms_p50": pct(ttft, 0.5), "ttft_ms_p90": pct(ttft, 0.9),
        "health": {k: v for k, v in health.items() if k != "replicas"},
        "server_mtp_lines": lines[:8],
    }
    print(json.dumps(summary), flush=True)
    return {r["key"]: r["ids"] for r in results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--model", default=None, help="local checkpoint (default: the pinned one)")
    args = parser.parse_args()
    if args.model:
        tok = AutoTokenizer.from_pretrained(args.model)
    else:
        tok = AutoTokenizer.from_pretrained(CHECKPOINT["model"], revision=CHECKPOINT["revision"])
    pool, plans = synthetic_rounds(tok, args.sessions, args.rounds)
    outputs = {flag: serve_and_run(flag, args.port + i, args, pool, plans) for i, flag in enumerate(("0", "1"))}
    same, diverged = 0, []
    for key, base in outputs["0"].items():
        spec = outputs["1"].get(key)
        if spec == base:
            same += 1
        else:
            first = next((i for i, (a, b) in enumerate(zip(base, spec or [])) if a != b), None)
            diverged.append({"key": key, "first_divergence": first, "len": len(base)})
    print(json.dumps({"event": "parity", "requests": len(outputs["0"]), "identical": same,
                      "diverged": diverged}), flush=True)


if __name__ == "__main__":
    main()
