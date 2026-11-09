"""Plan 49 step 0: move a session's cached state between two replica processes, as served.

    HIP_VISIBLE_DEVICES=0,1,2,3 python tools/plan49_migrate_probe.py

Starts two replicas the way ``engine.router`` does: each its own process, ``Generator.load``
exactly as ``engine.server`` (TP2, chained MTP, prefix cache, warmup), with its visible GPUs
narrowed to its pair (A: 0,1; B: 2,3) and its CPU slice from ``engine.cpu_place``. Then, for
several synthetic prompts at about 4k and 8k tokens:

1. A serves round 1 (``--decode1`` greedy tokens, ``ignore_eos``), keeping the session tail.
2. The tail moves to B (``engine.migrate``), timed, ``--reps`` times per transport:
   ``host``: A packs per rank on its GPUs and copies into a shared memory file (pinned in both
   processes), B copies to its GPUs and scatters into its own pages; ``ipc``: A packs into a
   device buffer shared by HIP IPC handle, B copies from it over the GPUs' links and scatters.
   The SHA-256 of every rank's bytes is compared between A's pack and B's landing buffers.
   The last rep of one transport (alternating per prompt) is kept by B as a session tail.
3. Round 2 (round 1's prompt + its output + ``--new`` new tokens) runs on A (control, its own
   tail) and on B (the imported tail), ``--decode2`` greedy tokens each; tokens must be
   identical, and both must report the tail as cached tokens.

Then the source-side stall: A decodes ``--bg`` background requests step by step (as the
server's worker, run-ahead off) and, between steps, exports an ~8k tail by each transport
(host inline, host on a side stream overlapped with decoding, ipc inline) while B imports
concurrently. Reported: the export's time on A's engine thread and A's decode step times
during the whole move against the steps around it.

Prints ``RESULT {json}`` lines; ``RESULT {"kind": "summary"...}`` carries the verdict inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import statistics
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHM_BYTES = 1 << 30
CAPACITY_TOKENS = 16384


def result(**kw) -> None:
    print("RESULT " + json.dumps(kw), flush=True)


def now_ns() -> int:
    return time.monotonic_ns()  # CLOCK_MONOTONIC: comparable across processes


def ms(ns: int) -> float:
    return round(ns / 1e6, 3)


# ============================================================================ replica side


def _wire(manifest) -> dict:
    """A manifest for the orchestrator to pass on unopened (it never imports torch)."""
    return {"manifest": pickle.dumps(manifest), "rank_bytes": list(manifest.rank_bytes)}


class Replica:
    def __init__(self, role: str, shm_fd: int, peer: Connection) -> None:
        import torch

        from engine import cpu_place, migrate
        from engine.generator import CHECKPOINT, Generator

        self.torch, self.migrate = torch, migrate
        self.role, self.peer = role, peer
        cpu_place.pin_rank(0)  # this thread drives rank 0, as the server's worker does
        t = time.time()
        self.g = Generator.load(
            CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144, chunk_tokens=2048,
            prefix_cache_tokens=4194304, prefix_cache_entries=128, prefix_cache_bytes=16 * 2**30,
        )
        self.load_s = time.time() - t
        self.fast = self.g.fast
        self.devices = [migrate._device(m) for m in migrate.ranks(self.fast)]
        self.channel = migrate.HostChannel(SHM_BYTES, fd=shm_fd, ranks=len(self.devices))
        self.staging = migrate.staging(self.fast, CAPACITY_TOKENS)
        self.ipc_export = None
        self.ipc_import = None
        self.side = [torch.cuda.Stream(d) for d in self.devices]

    # ------------------------------------------------------------------ helpers

    def sync(self) -> None:
        for d in self.devices:
            self.torch.cuda.synchronize(d)

    def sha(self, manifest, bufs) -> list[str]:
        self.sync()
        return [
            hashlib.sha256(bufs[r][:n].cpu().numpy().tobytes()).hexdigest()[:16]
            for r, n in enumerate(manifest.rank_bytes)
        ]

    def ready(self) -> dict:
        torch = self.torch
        return {
            "role": self.role, "load_s": round(self.load_s, 1),
            "devices": [str(d) for d in self.devices],
            "visible": os.environ.get("HIP_VISIBLE_DEVICES"),
            "tp": len(self.devices), "mtp": self.fast.mtp is not None,
            "chain": bool(getattr(self.fast, "chain", False)),
            "graphs": self.fast.use_graphs, "graph_error": self.fast.graph_error,
            "kv_pages": self.fast.pool.pages, "pinned_channel": self.channel.pinned,
            "staging_bytes": [int(b.numel()) for b in self.staging],
            "reserved_gib": [round(torch.cuda.memory_reserved(d) / 2**30, 2) for d in self.devices],
        }

    # ------------------------------------------------------------------ serving one request

    def run(self, prompt: list[int], max_tokens: int) -> dict:
        from engine.generator import PackedPrefill
        from engine.protocol import Sampling

        g = self.g
        sampling = Sampling(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        before = g.cached_tokens
        t0 = now_ns()
        packer = PackedPrefill(g)
        packer.add(tuple(prompt), sampling)
        for _, exc in packer.failed():
            raise exc
        gen = None
        while packer.busy:
            for _, res in packer.step():
                if isinstance(res, Exception):
                    raise res
                gen = res
        t1 = now_ns()
        out: list[int] = []
        while True:
            while True:
                step = gen.next_step()  # retires the request at its last token
                out.append(step.token_id)
                if gen.done or not gen.has_queued():
                    break
            if gen.done:
                break
            g.decode([gen], lambda: False)
        gen.retire()  # the session tail, as the server keeps it after the last token
        t2 = now_ns()
        return {"tokens": out, "cached": g.cached_tokens - before, "prefill_ms": ms(t1 - t0),
                "decode_ms": ms(t2 - t1)}

    # ------------------------------------------------------------------ source side

    def tail(self, key: list[int]):
        snap = self.g.export_tail(tuple(key), keep=True)
        if snap is None:
            raise RuntimeError(f"no session tail at the {len(key)}-token key")
        return snap

    def export_host(self, key: list[int], sha: bool = True) -> dict:
        snap = self.tail(key)
        t0 = now_ns()
        with self.torch.inference_mode():
            manifest = self.migrate.export_into(self.fast, snap, self.staging)
            self.sync()
            t1 = now_ns()
            self.channel.put(manifest, self.staging)
        t2 = now_ns()
        return {**_wire(manifest), "t0": t0, "t_end": t2, "pack_ms": ms(t1 - t0),
                "d2h_ms": ms(t2 - t1), "export_ms": ms(t2 - t0),
                "sha": self.sha(manifest, self.staging) if sha else None}

    def ipc_handles(self) -> dict:
        self.ipc_export = self.migrate.IpcExport(self.fast, CAPACITY_TOKENS)
        return {"handles": self.ipc_export.handles(), "wrapped": self.ipc_export.wrapped,
                "wrap_error": self.ipc_export.wrap_error}

    def export_ipc(self, key: list[int], sha: bool = True) -> dict:
        snap = self.tail(key)
        ipc = self.ipc_export
        t0 = now_ns()
        with self.torch.inference_mode():
            manifest = self.migrate.export_into(self.fast, snap, ipc.bufs)
            ipc.publish(manifest)
        t1 = now_ns()
        return {**_wire(manifest), "t0": t0, "t_end": t1, "export_ms": ms(t1 - t0),
                "sha": self.sha(manifest, ipc.bufs) if sha else None}

    # ------------------------------------------------------------------ target side

    def open_ipc(self, handles: list[bytes]) -> dict:
        t0 = now_ns()
        self.ipc_import = self.migrate.IpcImport(self.fast, handles)
        return {"open_ms": ms(now_ns() - t0)}

    def import_(self, kind: str, manifest, key: list[int] | None, keep: bool, sha: bool = True) -> dict:
        torch = self.torch
        if isinstance(manifest, bytes):
            manifest = pickle.loads(manifest)
        t0 = now_ns()
        with torch.inference_mode():
            if kind == "host":
                self.channel.get(manifest, self.staging)
            else:
                self.ipc_import.pull(manifest, self.staging)
            self.sync()
            t1 = now_ns()
            snap = self.migrate.import_from(self.fast, manifest, self.staging)
            self.sync()
        t2 = now_ns()
        digest = self.sha(manifest, self.staging) if sha else None
        kept = False
        if keep:
            kept = self.g.import_tail(tuple(key), snap)
        else:
            self.fast.release(snap)
        return {"t0": t0, "t_end": t2, "copy_ms": ms(t1 - t0), "scatter_ms": ms(t2 - t1),
                "import_ms": ms(t2 - t0), "sha": digest, "kept": kept}

    def serve_peer(self) -> dict:
        """Import whatever the source announces on the peer pipe until it says stop."""
        log = []
        while True:
            msg = self.peer.recv()
            if msg[0] == "stop":
                return {"imports": log}
            _, kind, manifest = msg
            r = self.import_(kind, manifest, None, keep=False, sha=False)
            log.append({"kind": kind, "t0": r["t0"], "t_end": r["t_end"], "import_ms": r["import_ms"]})
            self.peer.send(("done", r["t_end"]))

    # ------------------------------------------------------------------ the stall test

    def stall(self, key: list[int], bg_prompts: list[list[int]], gap: int, reps: int) -> dict:
        """Decode the background requests step by step; between steps, export ``key``'s tail
        by each transport while the target imports it; time every step."""
        from engine.generator import PackedPrefill
        from engine.protocol import Sampling

        torch, g = self.torch, self.g
        snap = self.tail(key)
        sampling = Sampling(max_tokens=100000, temperature=0.0, ignore_eos=True)
        packer = PackedPrefill(g)
        for p in bg_prompts:
            packer.add(tuple(p), sampling)
        gens = []
        while packer.busy:
            gens += [res for _, res in packer.step()]
        plan = ["host", "host_async", "ipc"] * reps
        steps: list[tuple[int, int]] = []  # (start, end) of each decode step, ns
        events: list[dict] = []
        pending = None  # the move in flight
        index = 0
        total = gap * (len(plan) + 2)
        for i in range(total):
            if pending is None and i >= gap and (i % gap) == 0 and index < len(plan):
                kind = plan[index]
                index += 1
                t0 = now_ns()
                with torch.inference_mode():
                    if kind == "host":
                        manifest = self.migrate.export_into(self.fast, snap, self.staging)
                        self.channel.put(manifest, self.staging)
                        pending = {"kind": kind, "t0": t0, "inline_ns": now_ns() - t0, "announced": True}
                        self.peer.send(("import", "host", manifest))
                    elif kind == "ipc":
                        manifest = self.migrate.export_into(self.fast, snap, self.ipc_export.bufs)
                        self.ipc_export.publish(manifest)
                        pending = {"kind": kind, "t0": t0, "inline_ns": now_ns() - t0, "announced": True}
                        self.peer.send(("import", "ipc", manifest))
                    else:  # host_async: pack and copy out on side streams while decoding
                        for s, d in zip(self.side, self.devices, strict=True):
                            s.wait_stream(torch.cuda.current_stream(d))
                        ctx = [torch.cuda.stream(s) for s in self.side]
                        for c in ctx:
                            c.__enter__()
                        try:
                            manifest = self.migrate.export_into(self.fast, snap, self.staging)
                            for r, n in enumerate(manifest.rank_bytes):
                                self.channel.rank(r, n).copy_(self.staging[r][:n], non_blocking=True)
                            done = [s.record_event() for s in self.side]
                        finally:
                            for c in reversed(ctx):
                                c.__exit__(None, None, None)
                        pending = {"kind": kind, "t0": t0, "inline_ns": now_ns() - t0,
                                   "announced": False, "events": done, "manifest": manifest}
            if pending is not None and not pending["announced"]:
                if all(e.query() for e in pending["events"]):
                    pending["ready_ns"] = now_ns() - pending["t0"]
                    pending["announced"] = True
                    self.peer.send(("import", "host", pending.pop("manifest")))
                    pending.pop("events")
            # One decode step for every background request, as the server's worker runs it.
            s0 = now_ns()
            for gen in gens:
                while True:
                    gen.next_step()
                    if gen.done or not gen.has_queued():
                        break
            g.decode(gens, lambda: False)
            s1 = now_ns()
            steps.append((s0, s1))
            if pending is not None and pending["announced"] and self.peer.poll():
                _, t_end = self.peer.recv()
                pending["t_end"] = t_end
                events.append(pending)
                pending = None
        while pending is not None:  # not expected: the loop leaves room
            if not pending["announced"]:
                torch.cuda.synchronize()
                self.peer.send(("import", "host", pending.pop("manifest")))
                pending["announced"] = True
            _, t_end = self.peer.recv()
            pending["t_end"] = t_end
            events.append(pending)
            pending = None
        self.peer.send(("stop",))
        for gen in gens:
            gen.retire()
        return {"steps": steps, "events": [
            {k: v for k, v in e.items() if k not in ("events", "manifest", "announced")} for e in events
        ], "bytes": self.migrate.plan(self.fast, snap.length).nbytes, "tokens": snap.length}


def replica_main(args) -> None:
    ctl = Connection(args.ctl)
    peer = Connection(args.peer)
    try:
        rep = Replica(args.role, args.shm, peer)
    except BaseException as exc:
        ctl.send(("error", f"{type(exc).__name__}: {exc}"))
        raise
    ctl.send(("ok", rep.ready()))
    while True:
        try:
            msg = ctl.recv()
        except EOFError:
            return
        name, kw = msg
        if name == "quit":
            ctl.send(("ok", None))
            return
        try:
            out = getattr(rep, name)(**kw)
            ctl.send(("ok", out))
        except BaseException as exc:  # noqa: BLE001 - report and keep serving
            import traceback

            traceback.print_exc()
            ctl.send(("error", f"{type(exc).__name__}: {exc}"))


# ============================================================================ orchestration


class Child:
    def __init__(self, role: str, devices: tuple[int, ...], piece, shm_fd: int, peer_fd: int) -> None:
        from engine import cpu_place, router

        here, there = socket.socketpair()
        env = router.device_env(dict(os.environ), devices, hip=True)
        env["ENGINE_REPLICA_INDEX"] = "0" if role == "A" else "1"
        preexec = None
        if piece is not None:
            env = cpu_place.child_env(env, piece)
            preexec = cpu_place.preexec(piece)
        else:
            env.pop("ENGINE_CPUS", None)
            env.pop("ENGINE_RANK_CPUS", None)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--role", role,
                   "--ctl", str(there.fileno()), "--peer", str(peer_fd), "--shm", str(shm_fd)]
        self.role = role
        self.proc = subprocess.Popen(  # noqa: PLW1509
            command, env=env, cwd=str(ROOT), preexec_fn=preexec,
            pass_fds=(there.fileno(), peer_fd, shm_fd),
        )
        there.close()
        self.conn = Connection(here.detach())

    def recv(self, timeout: float):
        if not self.conn.poll(timeout):
            raise TimeoutError(f"replica {self.role}: no answer in {timeout:.0f}s")
        status, out = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"replica {self.role}: {out}")
        return out

    def call(self, name: str, timeout: float = 600.0, **kw):
        self.conn.send((name, kw))
        return self.recv(timeout)

    def send(self, name: str, **kw) -> None:
        self.conn.send((name, kw))


def synthetic(tok, seed: int, count: int) -> list[int]:
    """Coding-like text, distinct per seed from its first token on."""
    ids: list[int] = []
    i = 0
    head = f"# module_{seed:04d}: helpers ({seed * 7919 % 100003})\n"
    ids.extend(tok.encode(head, add_special_tokens=False))
    while len(ids) < count:
        lines = (
            f"def handler_{seed}_{i}(items, limit={i % 17 + 1}):\n",
            f"    out = [x for x in items if x % {i % 13 + 2} == {seed % 3}]\n",
            f"    return sorted(out)[:limit]  # {seed}-{i} {(seed * 31 + i * 7) % 997:03d}\n",
        )
        for line in lines:
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1
    return ids[:count]


def summarize_stall(st: dict) -> list[dict]:
    steps = st["steps"]
    durs = [e - s for s, e in steps]
    rows = []
    for ev in st["events"]:
        t0, t1 = ev["t0"], ev["t_end"]
        inside = [d for (s, e), d in zip(steps, durs, strict=True) if e > t0 and s < t1]
        first = next(i for i, (s, e) in enumerate(steps) if s >= t0)
        last = max(i for i, (s, e) in enumerate(steps) if s < t1) if any(s < t1 for s, _ in steps) else first
        around = durs[max(0, first - 12) : max(0, first - 2)] + durs[last + 3 : last + 13]
        base = statistics.median(around) if around else float("nan")
        extra = sum(d - base for d in inside)
        rows.append({
            "kind": ev["kind"], "inline_ms": ms(ev["inline_ns"]),
            "ready_ms": ms(ev["ready_ns"]) if "ready_ns" in ev else None,
            "move_ms": ms(t1 - t0), "steps_during": len(inside),
            "step_base_ms": round(base / 1e6, 3),
            "step_during_max_ms": round(max(inside) / 1e6, 3) if inside else None,
            "extra_step_ms_during": round(extra / 1e6, 3),
            "stall_ms": round((ev["inline_ns"] + max(extra, 0)) / 1e6, 3),
        })
    return rows


def _memfd(name: str) -> int:
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, 0)
    import ctypes

    fd = ctypes.CDLL(None, use_errno=True).memfd_create(name.encode(), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    return fd


def orchestrate(args) -> int:
    from engine import cpu_place

    layout = [(0, 1), (2, 3)]
    placed = cpu_place.replica_slices(layout)
    shm = _memfd("plan49-probe")  # torch stays out of this process
    os.ftruncate(shm, SHM_BYTES)
    pa, pb = socket.socketpair()
    a = Child("A", layout[0], placed[0][0] if placed else None, shm, pa.fileno())
    b = Child("B", layout[1], placed[0][1] if placed else None, shm, pb.fileno())
    pa.close()
    pb.close()
    children = [a, b]
    try:
        for c in children:
            result(kind="ready", **c.recv(1500))
        from transformers import AutoTokenizer

        from engine.generator import CHECKPOINT

        tok = AutoTokenizer.from_pretrained(CHECKPOINT["model"], revision=CHECKPOINT["revision"])
        ipc_ok, ipc_error = False, None
        try:
            h = a.call("ipc_handles")
            result(kind="ipc_export", wrapped=h["wrapped"], wrap_error=h["wrap_error"])
            result(kind="ipc_open", **b.call("open_ipc", handles=h["handles"]))
            ipc_ok = True
        except Exception as exc:  # noqa: BLE001 - IPC is optional; report it
            ipc_error = f"{type(exc).__name__}: {exc}"
            result(kind="ipc_unavailable", error=ipc_error)
        kinds = ["host"] + (["ipc"] if ipc_ok else [])

        lengths = [int(x) for x in args.lengths.split(",")]
        moves: dict[tuple[int, str], list[dict]] = {}
        identical, compared, last_key = True, 0, None
        sha_ok = True
        for L in lengths:
            for p in range(args.prompts):
                seed = L * 10 + p
                prompt = synthetic(tok, seed, L)
                r1 = a.call("run", prompt=prompt, max_tokens=args.decode1)
                key = prompt + r1["tokens"][:-1]
                keep_kind = kinds[p % len(kinds)]
                for kind in kinds:
                    for rep in range(args.reps):
                        keep = kind == keep_kind and rep == args.reps - 1
                        ex = a.call("export_host" if kind == "host" else "export_ipc", key=key)
                        im = b.call("import_", kind=kind, manifest=ex["manifest"], key=key, keep=keep)
                        same = ex["sha"] == im["sha"]
                        sha_ok &= same
                        row = {
                            "kind": "move", "transport": kind, "tokens": len(key), "rep": rep,
                            "bytes_per_rank": ex["rank_bytes"], "mb_total": round(sum(ex["rank_bytes"]) / 1e6, 2),
                            "export_ms": ex["export_ms"], "import_ms": im["import_ms"],
                            "end_to_end_ms": ms(im["t_end"] - ex["t0"]),
                            "bytes_equal": same, "kept_by_B": im["kept"],
                        }
                        for k in ("pack_ms", "d2h_ms"):
                            if k in ex:
                                row["A_" + k] = ex[k]
                        row["B_copy_ms"], row["B_scatter_ms"] = im["copy_ms"], im["scatter_ms"]
                        result(**row)
                        moves.setdefault((L, kind), []).append(row)
                round2 = prompt + r1["tokens"] + synthetic(tok, seed + 5, args.new)
                ca = a.call("run", prompt=round2, max_tokens=args.decode2)
                cb = b.call("run", prompt=round2, max_tokens=args.decode2)
                same = ca["tokens"] == cb["tokens"]
                first = next((i for i, (x, y) in enumerate(zip(ca["tokens"], cb["tokens"])) if x != y), None)
                identical &= same
                compared += 1
                result(kind="continuation", prompt_tokens=L, seed=seed, tail_tokens=len(key),
                       imported_via=keep_kind, round2_tokens=len(round2), decode2=args.decode2,
                       identical=same, first_diff=first, A_cached=ca["cached"], B_cached=cb["cached"],
                       A_prefill_ms=ca["prefill_ms"], B_prefill_ms=cb["prefill_ms"],
                       A_head=ca["tokens"][:8], B_head=cb["tokens"][:8])
                last_key = round2 + ca["tokens"][:-1]

        # Source stall: A decodes background requests while the latest (~8k) tail moves.
        bg = [synthetic(tok, 900 + i, args.bg_len) for i in range(args.bg)]
        b.send("serve_peer")
        st = a.call("stall", key=last_key, bg_prompts=bg, gap=args.gap, reps=args.stall_reps, timeout=900)
        peer_log = b.recv(120)
        rows = summarize_stall(st)
        for row in rows:
            result(kind="stall", tokens=st["tokens"], mb_total=round(st["bytes"] / 1e6, 2), bg=args.bg, **row)
        result(kind="stall_imports", imports=[{"kind": x["kind"], "import_ms": x["import_ms"]} for x in peer_log["imports"]])

        summary = {"kind": "summary", "identical_all": identical, "continuations": compared,
                   "bytes_equal_all": sha_ok, "ipc": ipc_ok, "ipc_error": ipc_error}
        for (L, kind), rs in sorted(moves.items()):
            summary[f"e2e_ms_{kind}_{L}"] = round(statistics.median(r["end_to_end_ms"] for r in rs), 3)
            summary[f"export_ms_{kind}_{L}"] = round(statistics.median(r["export_ms"] for r in rs), 3)
            summary[f"mb_{kind}_{L}"] = rs[0]["mb_total"]
        for kind in ("host", "host_async", "ipc"):
            sel = [r for r in rows if r["kind"] == kind]
            if sel:
                summary[f"stall_ms_{kind}_8k"] = round(statistics.median(r["stall_ms"] for r in sel), 3)
                summary[f"move_ms_{kind}_8k"] = round(statistics.median(r["move_ms"] for r in sel), 3)
        result(**summary)
        return 0 if identical and sha_ok else 3
    finally:
        for c in children:
            try:
                c.call("quit", timeout=30)
            except Exception:  # noqa: BLE001
                pass
            try:
                c.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                c.proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("A", "B"))
    parser.add_argument("--ctl", type=int)
    parser.add_argument("--peer", type=int)
    parser.add_argument("--shm", type=int)
    parser.add_argument("--lengths", default="4050,8130")
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--decode1", type=int, default=50)
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--decode2", type=int, default=200)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--bg", type=int, default=4)
    parser.add_argument("--bg-len", type=int, default=1500)
    parser.add_argument("--gap", type=int, default=25)
    parser.add_argument("--stall-reps", type=int, default=3)
    args = parser.parse_args()
    if args.role:
        replica_main(args)
        return
    sys.exit(orchestrate(args))


if __name__ == "__main__":
    main()
