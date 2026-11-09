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

``--shapes uniform,uneven`` runs several load passes on each started server
(``--pass-gap`` seconds apart) and reports each pass on its own, including the
router's per-replica in-flight histogram for that pass. ``uneven`` draws each
session's shape from a seeded generic distribution (``--seed``): rounds
geometric with mean ``--rounds``, first prompt log-normal around ``--first``,
each round's output log-normal around ``--max-new``, and an exponential tool
wait of mean ``--tool-wait`` seconds before each later round. These are
synthetic, chosen without any recorded trace; every setting sees the same draws.
``replay`` keeps the uniform shape's lengths and adds the same kind of exponential
tool wait (mean ``--replay-wait``) before each later round. ``trace`` (plan 70) takes
each session's rounds, new input tokens, output lengths and tool waits from the task's
recorded session table (``--trace``; lengths and waits only, the tokens are the pool's),
so the load has the replay's own shape; only this client reads it, never the server.
Each pass also reports, per replica, the share of decode-step time spent at each batch
size (``decode_seconds_by_batch`` of the replicas' /health).

Each request records the router's ``X-Engine-Replica`` and ``X-Engine-Moved``
headers (plan 63): a pass reports how many rounds ran on state that moved
between replicas, their TTFT against the other later rounds, and the router's
move counts and times; at the end every setting's tokens are compared with the
first setting's, overall and for the moved rounds alone. Settings may repeat
(``ENGINE_MIGRATE=0,ENGINE_MIGRATE=1,ENGINE_MIGRATE=0,...``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
        moved = response.headers.get("x-engine-moved") == "1"
        replica = response.headers.get("x-engine-replica")
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                got = choice.get("token_ids") or []
                ids.extend(got)
                times.extend([time.perf_counter()] * len(got))
    return {"ids": ids, "times": times, "start": start, "usage": usage, "moved": moved,
            "replica": replica}


def rounds_of(index: int, args) -> int:
    """Session ``index``'s rounds: ``--rounds`` may be fractional (6.3: some sessions 6, some 7)."""
    return max(1, math.floor((index + 1) * args.rounds) - math.floor(index * args.rounds))


def plan_sessions(shape: str, args) -> list[dict]:
    """Per session: first-prompt length, each round's output length and the wait before it,
    and the session's token offset in the pool (sessions do not share tokens)."""
    rng = random.Random(args.seed)
    plans = []
    offset = 0
    if shape == "trace":
        return trace_sessions(args)
    for index in range(args.sessions):
        if shape == "replay":
            rounds = rounds_of(index, args)
            first = args.first
            outs = [args.max_new] * rounds
            waits = [0.0] + [min(rng.expovariate(1 / args.replay_wait), 3 * args.replay_wait)
                             if args.replay_wait > 0 else 0.0 for _ in range(rounds - 1)]
        elif shape == "uneven":
            rounds = 1
            while rounds < 4 * args.rounds and rng.random() > 1 / args.rounds:
                rounds += 1
            first = int(min(max(rng.lognormvariate(math.log(args.first), 0.6), 256), 4 * args.first))
            outs = [int(min(max(rng.lognormvariate(math.log(args.max_new), 0.8), 16),
                            4 * args.max_new)) for _ in range(rounds)]
            waits = [0.0] + [min(rng.expovariate(1 / args.tool_wait), 3 * args.tool_wait)
                             if args.tool_wait > 0 else 0.0 for _ in range(rounds - 1)]
        else:
            rounds = rounds_of(index, args)
            first = args.first
            outs = [args.max_new] * rounds
            waits = [0.0] * rounds
        plans.append({"first": first, "outs": outs, "waits": waits, "offset": offset})
        # uniform: the original fixed stride per session; uneven: packed one after another
        offset += (first + rounds * args.grow if shape == "uneven"
                   else args.first + math.ceil(args.rounds) * args.grow)
    return plans


def trace_sessions(args) -> list[dict]:
    """The recorded table's sessions in id order: first input, each later round's new input,
    output lengths, and the tool wait before each later round."""
    import csv

    rows: dict[int, list[dict]] = {}
    with open(args.trace) as f:
        for row in csv.DictReader(f):
            rows.setdefault(int(row["session_id"]), []).append(row)
    plans = []
    offset = 0
    for sid in sorted(rows)[: args.sessions]:
        rs = sorted(rows[sid], key=lambda r: int(r["round_idx"]))
        grows = [int(r["input_len"]) for r in rs[1:]]
        plans.append({
            "first": int(rs[0]["input_len"]),
            "outs": [int(r["output_len"]) for r in rs],
            "waits": [0.0] + [int(r["tool_wait_after_ms"]) / 1e3 for r in rs[:-1]],
            "grows": grows,
            "offset": offset,
        })
        offset += int(rs[0]["input_len"]) + sum(grows)
    return plans


def session(url: str, pool: list[int], index: int, args, plan: dict | None = None,
            shape: str = "uniform") -> list[dict]:
    if plan is None:  # the uniform shape (tools/live_compile_probe.py passes only lengths)
        rounds = rounds_of(index, args)
        plan = {"first": args.first, "outs": [args.max_new] * rounds, "waits": [0.0] * rounds,
                "offset": index * (args.first + math.ceil(args.rounds) * args.grow)}
    grows = plan.get("grows") or [args.grow] * len(plan["outs"])
    span = plan["first"] + sum(grows) + args.grow
    offset = plan["offset"] % max(1, len(pool) - span)
    fresh = pool[offset : offset + span]
    prompt = fresh[: plan["first"]]
    used = plan["first"]
    out = []
    for r, (max_new, wait) in enumerate(zip(plan["outs"], plan["waits"], strict=True)):
        if wait:
            time.sleep(wait)
        body = {
            "prompt": prompt,
            "max_tokens": max_new,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        res = stream(url, body)
        assert len(res["ids"]) == max_new, (len(res["ids"]), max_new)
        res["key"] = f"{shape}:s{index}r{r}"
        out.append(res)
        grow = grows[r] if r < len(grows) else args.grow
        prompt = prompt + res["ids"] + fresh[used : used + grow]
        used += grow
    return out


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def per_request(health: dict, key: str, requests: int) -> float | None:
    """A /health counter (the router's total over replicas) per request."""
    value = health.get(key)
    return round(value / requests, 3) if isinstance(value, int) and requests else None


def serve_and_run(setting: str | None, port: int, args, pool) -> dict[str, dict]:
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
        outputs: dict[str, dict] = {}
        before = httpx.get(url + "/health", timeout=30).json()
        for number, shape in enumerate(args.shapes.split(",")):
            if number:
                time.sleep(args.pass_gap)  # let the router's idle window lapse between passes
                before = httpx.get(url + "/health", timeout=30).json()
            plans = plan_sessions(shape, args)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(args.concurrency or args.sessions) as ex:
                results = [r for s in ex.map(lambda i: session(url, pool, i, args, plans[i], shape),
                                             range(args.sessions)) for r in s]
            wall = time.perf_counter() - t0
            health = httpx.get(url + "/health", timeout=30).json()
            report(setting, shape, load_s, wall, results, health, before, log)
            outputs.update({r["key"]: {"ids": r["ids"], "moved": r["moved"]} for r in results})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    return outputs


def router_delta(health: dict, before: dict) -> dict | None:
    """The router's in-flight histogram over one pass: ``after - before``, per replica."""
    after, start = health.get("router"), before.get("router")
    if not after:
        return None
    hist = []
    for r, h in enumerate(after["inflight_seconds"]):
        b = start["inflight_seconds"][r] if start else []
        hist.append([round(t - (b[i] if i < len(b) else 0.0), 2) for i, t in enumerate(h)])
    total = sum(sum(h) for h in hist)
    busy = sum(sum(h[1:]) for h in hist)
    over = sum(sum(h[3:]) for h in hist)
    return {
        "inflight_seconds": hist,
        "share_over_2": round(over / total, 4) if total else 0.0,
        "share_over_2_of_busy": round(over / busy, 4) if busy else 0.0,
        "max_inflight": [max((i for i, t in enumerate(h) if t > 0), default=0) for h in hist],
        "idle_seconds": after.get("idle_seconds"),
        "moves": moves_delta(after.get("moves"), (start or {}).get("moves")),
    }


def moves_delta(after: dict | None, before: dict | None) -> dict | None:
    """The router's live-move counters and times over one pass (plan 63)."""
    if not after:
        return None
    before = before or {}
    out = {}
    for key, value in after.items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            out[key] = value - before.get(key, 0)
        elif key.endswith("_ms") and isinstance(value, list):
            fresh = value[len(before.get(key, [])):]
            out[key] = {"n": len(fresh), "p50": pct(fresh, 0.5), "p90": pct(fresh, 0.9),
                        "max": round(max(fresh), 2) if fresh else None}
        elif key == "errors":
            out[key] = value[-4:]
    return out


def report(setting, shape, load_s, wall, results, health, before, log) -> None:
    tpot = [(r["times"][-1] - r["times"][0]) * 1e3 / (len(r["times"]) - 1)
            for r in results if len(r["times"]) > 1]
    ttft = [(r["times"][0] - r["start"]) * 1e3 for r in results if r["times"]]
    # TTFT by round: a session's first round (cold, the longest prefill) vs its later rounds
    # (resumed from the session tail).
    ttft_first = [(r["times"][0] - r["start"]) * 1e3 for r in results
                  if r["times"] and r["key"].endswith("r0")]
    ttft_later = [(r["times"][0] - r["start"]) * 1e3 for r in results
                  if r["times"] and not r["key"].endswith("r0")]
    tokens = sum(len(r["ids"]) for r in results)
    moved = [r for r in results if r["moved"]]
    ttft_moved = [(r["times"][0] - r["start"]) * 1e3 for r in moved if r["times"]]
    ttft_stayed = [(r["times"][0] - r["start"]) * 1e3 for r in results
                   if r["times"] and not r["moved"] and not r["key"].endswith("r0")]
    tpot_moved = [(r["times"][-1] - r["times"][0]) * 1e3 / (len(r["times"]) - 1)
                  for r in moved if len(r["times"]) > 1]
    migrate = [rep.get("migrate") for rep in health.get("replicas", [])]
    # Per replica: decode-step seconds by batch size over this pass, and each size's share.
    decode_by_batch = []
    for index, rep in enumerate(health.get("replicas", [])):
        now = rep.get("decode_seconds_by_batch") or {}
        was = ((before.get("replicas") or [{}] * (index + 1))[index] or {}).get(
            "decode_seconds_by_batch") or {}
        secs = {b: now[b][0] - (was.get(b) or [0.0, 0])[0] for b in now}
        steps = {b: now[b][1] - (was.get(b) or [0.0, 0])[1] for b in now}
        total = sum(secs.values())
        decode_by_batch.append({
            "seconds": round(total, 3),
            "share": {b: round(t / total, 4) for b, t in sorted(secs.items(), key=lambda x: int(x[0]))
                      if total and t > 0},
            "steps": {b: n for b, n in steps.items() if n},
            "share_ge3": round(sum(t for b, t in secs.items() if int(b) >= 3) / total, 4)
            if total else None,
        })
    cached = sum(int((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                 for r in results)
    engine_lines = [x for x in log.read_text().splitlines() if x.startswith("engine:")]
    # ENGINE_LOOP_TIMERS=1: each replica's per-stage loop timers, one line each.
    timer_sets = [health.pop("loop_timers", None)] + [
        r.pop("loop_timers", None) for r in health.get("replicas", [])
    ]
    for index, timers in enumerate(t for t in timer_sets if t):
        print(json.dumps({"event": "loop_timers", "setting": setting or "default",
                          "replica": index, **timers}), flush=True)
    print(json.dumps({
        "event": "setting", "setting": setting or "default", "shape": shape,
        "router_pass": router_delta(health, before), "load_s": round(load_s, 1),
        "wall_s": round(wall, 2), "requests": len(results), "output_tokens": tokens,
        "output_tok_per_s": round(tokens / wall, 2),
        "tpot_ms_p50": pct(tpot, 0.5), "tpot_ms_p90": pct(tpot, 0.9),
        "ttft_ms_p50": pct(ttft, 0.5), "ttft_ms_p90": pct(ttft, 0.9),
        "ttft_first_round_ms_p50": pct(ttft_first, 0.5),
        "ttft_first_round_ms_p90": pct(ttft_first, 0.9),
        "ttft_later_rounds_ms_p50": pct(ttft_later, 0.5),
        "ttft_later_rounds_ms_p90": pct(ttft_later, 0.9),
        "moved_rounds": len(moved),
        "ttft_moved_ms_p50": pct(ttft_moved, 0.5), "ttft_moved_ms_p90": pct(ttft_moved, 0.9),
        "ttft_unmoved_later_ms_p50": pct(ttft_stayed, 0.5),
        "ttft_unmoved_later_ms_p90": pct(ttft_stayed, 0.9),
        "tpot_moved_ms_p50": pct(tpot_moved, 0.5),
        "replica_migrate_cumulative": migrate if any(migrate) else None,
        "decode_by_batch": decode_by_batch,
        # Segments: a request's prefill spans (passes it takes part in); passes: shared passes.
        "prefill_segments_per_request": per_request(health, "prefill_segments", len(results)),
        "prefill_passes_per_request": per_request(health, "prefill_passes", len(results)),
        "cached_prompt_tokens": cached, "health": health, "engine": engine_lines[:4],
    }), flush=True)


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
    parser.add_argument("--shapes", default="uniform",
                        help="comma-separated load passes per server: uniform or uneven")
    parser.add_argument("--pass-gap", type=float, default=12.0,
                        help="seconds between passes on one server")
    parser.add_argument("--seed", type=int, default=0, help="uneven shape's draws")
    parser.add_argument("--tool-wait", type=float, default=0.0,
                        help="uneven shape: mean seconds before each later round")
    parser.add_argument("--trace", default=str(ROOT / ".vibesys/tasks/mi2508x-tracelab/recorded_sessions.csv"),
                        help="trace shape: the recorded session table (lengths and waits)")
    parser.add_argument("--replay-wait", type=float, default=0.3,
                        help="replay shape: mean seconds before each later round")
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
    need = max(
        args.sessions * (args.first + math.ceil(args.rounds) * args.grow),
        *(sum(p["first"] + sum(p.get("grows") or [args.grow] * len(p["outs"])) + args.grow
              for p in plan_sessions(shape, args))
          for shape in args.shapes.split(",")),
    ) + 1
    pool = token_pool(model, revision, need, args.corpus)
    print(json.dumps({"event": "shape", **vars(args)}), flush=True)
    runs = []
    for setting in args.settings.split(","):
        runs.append((setting, serve_and_run(None if setting == "default" else setting, args.port,
                                            args, pool)))
    # Every run's tokens against the first run's, overall and for the rounds served from moved
    # state (temperature 0: a moved continuation must be token-identical).
    base_name, base = runs[0]
    for number, (name, out) in enumerate(runs[1:], start=1):
        keys = [k for k in out if k in base]
        moved = [k for k in keys if out[k]["moved"]]
        diverged = [k for k in keys if out[k]["ids"] != base[k]["ids"]]
        first_div = sorted(
            next(i for i, (x, y) in enumerate(zip(out[k]["ids"], base[k]["ids"])) if x != y)
            for k in diverged if any(x != y for x, y in zip(out[k]["ids"], base[k]["ids"]))
        )
        print(json.dumps({
            "event": "agreement", "run": number, "setting": name, "against": base_name,
            "identical_requests": len(keys) - len(diverged), "requests": len(keys),
            "moved_requests": len(moved),
            "moved_identical": sum(out[k]["ids"] == base[k]["ids"] for k in moved),
            "diverged_keys": diverged[:12], "first_divergence_positions": first_div[:16],
        }), flush=True)


if __name__ == "__main__":
    main()
