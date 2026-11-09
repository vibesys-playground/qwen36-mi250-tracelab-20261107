"""Plan 38: Triton compiles after /health while serving a replay-like load.

    HIP_VISIBLE_DEVICES=4,5 python tools/live_compile_probe.py [--sessions 24]

Starts ``python -m engine.server`` as the task does, with ``tools/compile_hook`` on PYTHONPATH
(its ``sitecustomize`` logs every Triton JIT cache miss of every server process to
``TRITON_COMPILE_LOG``), waits for /health, then runs multi-round streaming sessions
(``serve_probe.session``: temperature 0, ignore_eos, growing prompts that reuse the prefix
cache) whose first prompts span 20..7,000 tokens, 8 at a time, so prefill passes of many sizes
(captured graph buckets, packed and long eager passes), decode and speculative verify batches
all run; then a few chat completions. Prints the compiles logged before and after /health,
by kernel. Any compile after /health is a stall while serving.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

FIRSTS = (20, 61, 100, 130, 250, 561, 700, 1000, 1100, 1650, 2048, 2300, 3000, 4500, 7000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--grow", type=int, default=480)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--port", type=int, default=18533)
    args = parser.parse_args()
    from engine.generator import CHECKPOINT
    from serve_probe import session, token_pool

    log = Path("/tmp/triton-compiles.jsonl")
    log.unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "tools" / "compile_hook"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["TRITON_COMPILE_LOG"] = str(log)
    server_log = Path("/tmp/live-compile-server.log")
    with server_log.open("w") as sink:
        proc = subprocess.Popen([sys.executable, "-m", "engine.server", "--host", "127.0.0.1", "--port", str(args.port)],
                                cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{args.port}"
    pool = token_pool(CHECKPOINT["model"], CHECKPOINT["revision"], 400_000, "synthetic")
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                print(server_log.read_text()[-4000:], flush=True)
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                if httpx.get(url + "/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(2)
        t_health = time.time()
        print(f"RESULT {json.dumps({'kind': 'health', 'load_s': round(t_health - t0, 1)})}", flush=True)

        def one(i: int):
            a = SimpleNamespace(first=FIRSTS[i % len(FIRSTS)], rounds=args.rounds, grow=args.grow,
                                max_new=args.max_new)
            return session(url, pool, i, a)

        t1 = time.time()
        with ThreadPoolExecutor(8) as ex:
            results = [r for s in ex.map(one, range(args.sessions)) for r in s]
        for text in ("What is 17 * 23? Answer with the number only.", "Name the capital of France."):
            r = httpx.post(url + "/v1/chat/completions", json={
                "messages": [{"role": "user", "content": text}], "max_tokens": 16, "temperature": 0}, timeout=600)
            r.raise_for_status()
        health = httpx.get(url + "/health", timeout=30).json()
        print(f"RESULT {json.dumps({'kind': 'load', 'requests': len(results), 'wall_s': round(time.time() - t1, 1), 'health': health})}"[:3000], flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    entries = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    before = [e for e in entries if e["t"] < t_health]
    after = [e for e in entries if e["t"] >= t_health]
    print("RESULT " + json.dumps({
        "kind": "compiles", "hook_working": bool(entries), "before_health": len(before),
        "before_health_s": round(sum(e["s"] for e in before), 1), "after_health": len(after),
        "after_health_s": round(sum(e["s"] for e in after), 2),
        "after_by_kernel": Counter(e["name"] for e in after).most_common(),
        "before_by_kernel_top": Counter(e["name"] for e in before).most_common(12),
        "processes": len({e["pid"] for e in entries}),
    }), flush=True)
    for line in server_log.read_text().splitlines():
        if line.startswith("engine:") or "Traceback" in line or "Error" in line:
            print(line[:300], flush=True)
    return 0 if after == [] and entries else 1


if __name__ == "__main__":
    sys.exit(main())
