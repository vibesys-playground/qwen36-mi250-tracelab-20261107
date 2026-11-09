"""Plan 45 stage 0: each slot's chained verify step as its own concurrently running graph "lane".

    python tools/lane_probe.py [--prompt-len 4000] [--iters 200] [--stress 2000] [--timeout 2700]

No served-path change. One TP2 replica on GCDs 0,1 loaded as the server loads it
(``Generator.load``: ``TPModel``, chained verify graphs captured at load). Then, from the
engine's own capture code (``TPModel.capture``), this tool builds "lanes": per lane and per rank
its own static step inputs (a ``_Device`` with its own ``packed`` buffer, the page table shared),
its own chained-verify segments and captured 1-slot graph, its own graph memory pool, its own
in-graph all-reduce state (a second ``tp.Group``: LL receive buffers and epochs) and its own
replay stream. It swaps those into the rank models for the capture and restores the engine's
state after it; the engine's graphs are untouched.

Measured (replay of the captured graphs only, inputs loaded before, every GCD synchronized,
host wall per step, median / p90 over ``--iters`` after warm-up; per-lane GPU time from events
on its stream, both ranks):

- ``batched``: today's batched 2-slot (and 3-slot) chained verify graph (``TPModel._run``);
  also the full ``verify_chain`` call (host inputs and parse included).
- ``solo``: one lane alone (slot 0, slot 1), and the engine's own 1-slot graph.
- ``concurrent``: lanes A (slot 0) and B (slot 1) launched back to back on their own streams,
  wall to both done; launch-order and stream-priority variants; 3 lanes (info).
- the same at k = 1 (2 rows per slot).

Checks: every concurrent lane's logits (both ranks' vocab shards) and chain output (picks,
accepted count, next drafts; both ranks) bitwise equal to the same lane run alone; the lanes'
rows against the batched graph's rows (reported); no in-graph all-reduce timeout (the kernels'
spin bound) over a ``--stress``-step concurrent loop; the part runs under ``timeout``.

Kill rule (plan 45 stage 0): concurrent 2-lane k=3 step > 12.5 ms, any bit difference against
the lane alone, any all-reduce timeout or hang. Pass: <= 11.5 ms.

Prints ``RESULT <tag> <json>`` lines. A plan hint, not the task's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SLOT_STRIDE = 9973
PASS_MS = 11.5
KILL_MS = 12.5


def out(tag: str, **fields) -> None:
    print(f"RESULT {tag} " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


# ================================================================================ driver


def driver(args: argparse.Namespace) -> None:
    env = dict(os.environ)
    env.update(ENGINE_MTP="1", HIP_VISIBLE_DEVICES="0,1", ENGINE_TP="2")
    argv = [__file__, "--part", "lanes", "--prompt-len", str(args.prompt_len), "--iters",
            str(args.iters), "--stress", str(args.stress)]
    print(f"===== part: {' '.join(argv)} (timeout {args.timeout}s)", flush=True)
    t0 = time.time()
    # ``timeout`` kills a part that hangs (a spinning all-reduce that never completes, say).
    proc = subprocess.Popen(["timeout", "-k", "30", str(args.timeout), sys.executable, "-u", *argv],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd=str(ROOT))
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        if "Loading weights" in line or "amdgpu.ids" in line:
            continue
        print(line, end="", flush=True)
        if line.startswith("RESULT "):
            lines.append(line.strip())
    rc = proc.wait()
    hung = rc in (124, 137)
    print(f"===== part done rc={rc}{' (TIMEOUT: hang)' if hung else ''} s={time.time() - t0:.0f}",
          flush=True)

    res: dict[str, list[dict]] = {}
    for line in lines:
        _, tag, payload = line.split(" ", 2)
        try:
            res.setdefault(tag, []).append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    walls = {r["name"]: r["wall_p50_ms"] for r in res.get("time", [])}
    k3 = {n: v for n, v in walls.items() if n.startswith("k3_concurrent2_")}
    best = min(k3, key=k3.get) if k3 else None
    failed = []
    if rc != 0:
        failed.append(f"part rc={rc}{' (hang)' if hung else ''}")
    for c in res.get("check", []):
        if not c.get("ok") and c.get("rule", True):
            failed.append(f"check {c.get('what')}")
    if best is None:
        failed.append("no concurrent k=3 time")
    conc = k3.get(best) if best else None
    kill = bool(failed) or (conc is not None and conc > KILL_MS)
    passed = not kill and conc is not None and conc <= PASS_MS
    out("verdict", batched2_k3_ms=walls.get("k3_batched2_replay"),
        solo_k3_ms=[walls.get("k3_solo_slot0"), walls.get("k3_solo_slot1")],
        concurrent2_k3_ms=k3, best=best, best_ms=conc,
        concurrent2_k1_ms=walls.get("k1_concurrent2_lane_major"),
        batched2_k1_ms=walls.get("k1_batched2_replay"),
        pass_rule=f"best concurrent 2-lane k=3 <= {PASS_MS} ms",
        kill_rule=f"> {KILL_MS} ms, any bit difference vs the lane alone, any AR timeout/hang",
        failed=failed, verdict="pass" if passed else ("kill" if kill else "neither"))


# ================================================================================ lanes


def sync_all(torch) -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


class Lane:
    """One slot's own 1-slot chained verify graph per rank, built with the engine's capture
    (``TPModel.capture``) on swapped-in per-lane state."""

    def __init__(self, fast, groups: int, slot: int, name: str, group=None) -> None:
        import torch

        from engine.fast import tp
        from engine.fast.model import _Device

        self.fast, self.groups, self.slot, self.name = fast, groups, slot, name
        self.group = group if group is not None else tp.Group(fast.devices_tp)
        self.devs = [m.embed.device for m in fast.ranks]
        self.metas, self.graphs, self.outs, self.chains = [], [], [], []
        self.capture_streams = []
        saved = []
        for m, dev in zip(fast.ranks, self.devs, strict=True):
            old = m.devices[dev]
            packed = torch.zeros_like(old.packed)
            packed[1].fill_(m.slots)
            packed[3].fill_(m.slots)
            meta = _Device(device=dev, packed=packed, page_table=old.page_table)
            self.metas.append(meta)
            keys = ("_csegments", "_pools", "_capture_streams", "_chain_static")
            saved.append((m.devices, m.tp, {k: m.__dict__.get(k) for k in keys}))
            m.devices = {dev: meta}
            m.tp = self.group
            m.__dict__["_csegments"] = {groups: m._segments(chain=groups)}
            m.__dict__["_pools"] = {}
            m.__dict__["_capture_streams"] = {}
            m.__dict__["_chain_static"] = {}
        saved_group = fast.group
        fast.group = self.group
        try:
            fast.capture(1, False, groups)
            if fast.graph_error is not None:
                raise RuntimeError(f"lane {name} capture failed: {fast.graph_error}")
            for m in fast.ranks:
                seg = m.__dict__["_csegments"][groups][0]
                self.graphs.append(seg.graphs[1])
                self.outs.append(seg.outputs[1])
                self.chains.append(m.__dict__["_chain_static"][(1, groups)])
                self.capture_streams.append(m.__dict__["_capture_streams"])
            self._keep = [(m.__dict__["_csegments"], m.__dict__["_pools"]) for m in fast.ranks]
        finally:
            fast.group = saved_group
            for m, (devices, group, extra) in zip(fast.ranks, saved, strict=True):
                m.devices, m.tp = devices, group
                for k, v in extra.items():
                    if v is None:
                        m.__dict__.pop(k, None)
                    else:
                        m.__dict__[k] = v
        self.streams = [torch.cuda.Stream(d) for d in self.devs]
        self.ev = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in self.devs]

    def set_streams(self, priority: int = 0) -> None:
        import torch

        self.streams = [torch.cuda.Stream(d, priority=priority) for d in self.devs]

    def load(self, token: int, drafts: list[int], position: int) -> None:
        """The step inputs of this lane's slot (``_chain_host``, as ``verify_chain`` builds them)."""
        r0 = self.fast.ranks[0]
        host = r0._chain_host([token], [drafts], [self.slot], [position], self.groups, 1)
        for meta in self.metas:
            meta.packed.copy_(host)

    def launch_rank(self, r: int) -> None:
        import torch

        with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
            self.ev[r][0].record()
            self.graphs[r].replay()
            self.ev[r][1].record()

    def gpu_ms(self) -> list[float]:
        return [self.ev[r][0].elapsed_time(self.ev[r][1]) for r in range(len(self.devs))]

    def snapshot(self):
        return [o.clone() for o in self.outs] + [c.clone() for c in self.chains]

    def timeouts(self) -> int:
        return self.group.timeouts()


def same(a, b) -> bool:
    import torch

    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def timed(torch, launch, iters: int, lanes=()) -> dict:
    """Host wall of ``launch()`` to every GPU synchronized; per-lane GPU ms from events."""
    for _ in range(5):
        launch()
        sync_all(torch)
    walls, gpus, hosts = [], {ln.name: [] for ln in lanes}, []
    for _ in range(iters):
        sync_all(torch)
        t0 = time.perf_counter()
        launch()
        t1 = time.perf_counter()
        sync_all(torch)
        walls.append((time.perf_counter() - t0) * 1e3)
        hosts.append((t1 - t0) * 1e3)
        for ln in lanes:
            gpus[ln.name].append(max(ln.gpu_ms()))
    walls.sort()
    p90 = walls[min(len(walls) - 1, int(0.9 * len(walls)))]
    r = dict(wall_p50_ms=round(statistics.median(walls), 3), wall_p90_ms=round(p90, 3),
             launch_host_p50_ms=round(statistics.median(hosts), 3))
    if lanes:
        r["lane_gpu_p50_ms"] = {n: round(statistics.median(v), 3) for n, v in gpus.items()}
    return r


def part_lanes(args: argparse.Namespace) -> None:
    import torch

    from engine.fast.model import FastModel
    from engine.generator import CHECKPOINT, Generator

    t0 = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    if not hasattr(fast, "ranks") or not fast.chain:
        raise SystemExit(f"expected the TP2 chained path, got {type(fast).__name__}")
    out("load", seconds=round(time.perf_counter() - t0), graph_error=fast.graph_error,
        kv_pages=fast.pool.pages, buckets=fast.buckets, torch=torch.__version__,
        hip=torch.version.hip, max_hw_queues=os.environ.get("GPU_MAX_HW_QUEUES"))

    tok = g.tokenizer
    ids: list[int] = []
    i = 0
    while len(ids) < SLOT_STRIDE * 3 + 1000:  # Request Factory-like synthetic coding lines
        for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} 5f0c2a9e1b7d"):
            ids.extend(tok.encode(line, add_special_tokens=False))
        i += 1
    plen = args.prompt_len
    nslots = 3
    for s in range(nslots):
        fast.reset(s)
        prompt = ids[SLOT_STRIDE * s : SLOT_STRIDE * s + plen]
        for b in range(0, plen, 2048):
            e = min(plen, b + 2048)
            fast.prefill(s, prompt[b:e], b, prompt[e] if e < plen else None)
    sync_all(torch)
    pos = plen

    def inputs(s: int, groups: int) -> tuple[int, list[int]]:
        base = SLOT_STRIDE * s + pos
        return ids[base], [ids[base + j] for j in range(1, groups)]

    rule_checks: list[bool] = []

    def check(what: str, ok: bool, rule: bool = True, **kw) -> None:
        if rule:
            rule_checks.append(ok)
        out("check", what=what, ok=bool(ok), rule=rule, **kw)

    group_a = group_b = group_c = None
    for groups in (4, 2):
        k = groups - 1
        tag = f"k{k}"
        # ---------------------------------------------------------------- (a) batched reference
        batched, bwall = {}, {}
        for nb in (2, 3):
            slots = list(range(nb))
            toks = [inputs(s, groups)[0] for s in slots]
            drs = [inputs(s, groups)[1] for s in slots]
            full = timed(torch, lambda: fast.verify_chain(toks, drs, slots, [pos] * nb), args.iters)
            out("time", name=f"{tag}_batched{nb}_full_call", **full)
            fast.verify_chain(toks, drs, slots, [pos] * nb)  # leaves these inputs loaded
            size = fast.bucket(nb)
            rep = timed(torch, lambda: fast._run(size, groups * size, False, chain=groups), args.iters)
            out("time", name=f"{tag}_batched{nb}_replay", bucket=size, **rep)
            bwall[nb] = rep["wall_p50_ms"]
            outs, chain = fast._run(size, groups * size, False, chain=groups)
            sync_all(torch)
            batched[nb] = ([o.clone() for o in outs], chain.clone(), size)
        # The engine's own 1-slot graph (shared inputs / pool / AR set), slot 0.
        t, d = inputs(0, groups)
        fast.verify_chain([t], [d], [0], [pos])
        eng1 = timed(torch, lambda: fast._run(1, groups, False, chain=groups), args.iters)
        out("time", name=f"{tag}_engine_solo_slot0", **eng1)
        outs, chain = fast._run(1, groups, False, chain=groups)
        sync_all(torch)
        engine_solo0 = [o.clone() for o in outs] + [chain.clone()]

        # ---------------------------------------------------------------- lanes
        t0 = time.perf_counter()
        la = Lane(fast, groups, 0, "A", group_a)
        lb = Lane(fast, groups, 1, "B", group_b)
        lc = Lane(fast, groups, 2, "C", group_c)
        group_a, group_b, group_c = la.group, lb.group, lc.group
        free = [torch.cuda.mem_get_info(d)[0] >> 20 for d in la.devs]
        out("lanes_built", k=k, seconds=round(time.perf_counter() - t0, 1), free_mib=free)
        lanes = (la, lb, lc)
        for ln in lanes:
            ln.load(*inputs(ln.slot, groups), pos)
        sync_all(torch)

        # ---------------------------------------------------------------- (b) solo
        solo = {}
        for ln in lanes:
            r = timed(torch, lambda ln=ln: [ln.launch_rank(0), ln.launch_rank(1)], args.iters, [ln])
            out("time", name=f"{tag}_solo_slot{ln.slot}", **r)
            solo[ln.name] = r
            ln.launch_rank(0)
            ln.launch_rank(1)
            sync_all(torch)
            ref = ln.snapshot()
            ln.launch_rank(0)
            ln.launch_rank(1)
            sync_all(torch)
            check(f"{tag}_solo_repeat_bitwise_{ln.name}", same(ref, ln.snapshot()))
            ln.ref = ref
            # Rank agreement of the chain output (picks, accepted, drafts).
            check(f"{tag}_ranks_agree_{ln.name}", torch.equal(ref[2].cpu(), ref[3].cpu()))
        check(f"{tag}_lane_vs_engine_1slot_graph_bitwise", same(la.ref, engine_solo0), rule=False)

        # Lane rows against the batched graph's rows (rows j * size + b).
        for nb in (2, 3):
            b_outs, b_chain, size = batched[nb]
            for b, ln in enumerate(lanes[:nb]):
                rows = [j * size + b for j in range(groups)]
                lg_same = all(torch.equal(bo[rows], lo) for bo, lo in zip(b_outs, ln.ref[:2], strict=True))
                bp = FastModel._chain_parse(b_chain, nb, size, groups)
                lp = FastModel._chain_parse(ln.ref[2], 1, 1, groups)
                picks_same = all(x[b] == y[0] for x, y in zip(bp[:1] + bp[2:], lp[:1] + lp[2:], strict=True)) \
                    and bp[1][b] == lp[1][0]
                drift = max(float((bo[rows] - lo).abs().max()) for bo, lo in zip(b_outs, ln.ref[:2], strict=True))
                check(f"{tag}_lane{ln.name}_vs_batched{nb}", lg_same and picks_same, rule=False,
                      logits_bitwise=lg_same, picks_drafts_same=picks_same, max_abs_logit_diff=drift)

        # ---------------------------------------------------------------- (c) concurrent
        orders = {
            "lane_major": lambda: [la.launch_rank(0), la.launch_rank(1), lb.launch_rank(0), lb.launch_rank(1)],
            "rank_major": lambda: [la.launch_rank(0), lb.launch_rank(0), la.launch_rank(1), lb.launch_rank(1)],
            "b_first": lambda: [lb.launch_rank(0), lb.launch_rank(1), la.launch_rank(0), la.launch_rank(1)],
        }
        if k != 3:
            orders = {"lane_major": orders["lane_major"]}
        variants = [(n, f, 0) for n, f in orders.items()]
        if k == 3:
            variants.append(("prio_a_high", orders["lane_major"], -1))
        for name, launch, prio in variants:
            la.set_streams(prio)
            lb.set_streams(0)
            r = timed(torch, launch, args.iters, (la, lb))
            slow = {ln.name: round(r["lane_gpu_p50_ms"][ln.name] / max(solo[ln.name]["lane_gpu_p50_ms"][ln.name], 1e-9), 3)
                    for ln in (la, lb)}
            out("time", name=f"{tag}_concurrent2_{name}", lane_slowdown_vs_solo=slow,
                vs_batched2=round(r["wall_p50_ms"] / bwall[2], 3), **r)
            launch()
            sync_all(torch)
            check(f"{tag}_concurrent2_{name}_bitwise_vs_solo",
                  same(la.ref, la.snapshot()) and same(lb.ref, lb.snapshot()))
        la.set_streams(0)
        # Three lanes (info: the replay's rarer 3-live-slot steps).
        r = timed(torch, lambda: [ln.launch_rank(q) for ln in lanes for q in (0, 1)], args.iters, lanes)
        out("time", name=f"{tag}_concurrent3_lane_major", vs_batched3=round(r["wall_p50_ms"] / bwall[3], 3), **r)
        sync_all(torch)
        check(f"{tag}_concurrent3_bitwise_vs_solo", all(same(ln.ref, ln.snapshot()) for ln in lanes))

        # ---------------------------------------------------------------- stress (hang guard)
        if k == 3 and args.stress:
            t0 = time.perf_counter()
            bad = 0
            for it in range(args.stress):
                orders["lane_major"]()
                sync_all(torch)
                if it % 100 == 99 and not (same(la.ref, la.snapshot()) and same(lb.ref, lb.snapshot())):
                    bad += 1
            sync_all(torch)
            check("k3_stress_bitwise", bad == 0, iters=args.stress, bad_checks=bad,
                  seconds=round(time.perf_counter() - t0, 1))
        to = {"engine": fast.timeouts(), "A": la.timeouts(), "B": lb.timeouts(), "C": lc.timeouts()}
        check(f"{tag}_allreduce_timeouts_zero", sum(to.values()) == 0, timeouts=to)
    out("rule_checks", all_ok=all(rule_checks), count=len(rule_checks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all", choices=("all", "lanes"))
    parser.add_argument("--prompt-len", type=int, default=4000)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--stress", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=2700)
    args = parser.parse_args()
    if args.part == "all":
        driver(args)
    else:
        part_lanes(args)


if __name__ == "__main__":
    main()
