"""Plan 55 stage 0: per-rank timestamps of the in-graph all-reduces, off unless ``ENGINE_AR_STAMP=1``.

How long does one GCD of a TP2 pair wait for the other at the in-graph LL all-reduces
(``tp._ll_allreduce_kernel``) on the served path, and why? With the flag on:

- the all-reduce kernel is compiled with ``STAMP``: program 0 reads the GPU's constant
  25 MHz clock (``s_memrealtime``) at entry, when its spin for the peer's words ends and at
  its end, and the shader clock (``s_memtime``) at entry, and stores them at index ``e`` (its
  own epoch: a per-rank count of all-reduce calls) of this rank's ring; the last program
  stores its entry, spin end and end at the index an atomic counter gives (the same count);
- ``TPModel`` launches a one-program stamp kernel on each rank's stream just before and just
  after each graph replay (decode, verify, chained verify, prefill), tagged with the step's
  kind, batch, k, bucket and whether it was a run-ahead launch; it records the clock and the
  all-reduce count at that point.

Only differences of one GCD's clock are used, plus one alignment per step: when an all-reduce's
spin ends on both ranks, both have the other's words, so the two spin ends of one call happen
within a microsecond of each other; the median over a step's calls of (rank-1 spin end -
rank-0 spin end) is the offset between the two clocks for that step.

``summary()`` (read by ``/health``) pairs the two ranks' steps and reports per step class:

- ``D``: rank 1's graph start minus rank 0's (aligned clocks; signed, us);
- ``wait0``/``wait1``: the sum over the step's all-reduces of each rank's spin (us);
  ``floor``: the sum of min(wait0, wait1) per call (what the later rank waits: the transfer
  and visibility floor); ``excess0``/``excess1``: each rank's wait beyond it;
- ``L``: the critical-path loss: segment i's time (from the previous all-reduce's end, or
  the graph start, to all-reduce i's entry) costs the step max(d0, d1); balanced it would
  cost mean(d0, d1); the first segment also carries the start offset. ``L_start`` is that
  first segment's part, ``L_imb`` the rest;
- per segment class (the all-reduce that ends it: the GDN or attention mixer's, the MoE's
  after a GDN or an attention layer, the MTP head's, the fp32 argmax's) the mean signed
  ``d0 - d1`` per step, its spread over steps, and its share of ``L_imb``;
- ``mhz0``/``mhz1``: each GCD's shader clock over the step (``s_memtime`` against
  ``s_memrealtime``), and the steps' span and the idle gap before them per rank.

A hint for the plan, not the task's metric. Only the instrumentation: nothing changes the
kernels' results, and with the flag off the stamp code is compiled out.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from typing import Any, Sequence

import torch

from engine.fast import kernels

triton, tl = kernels.triton, kernels.tl

ENABLED = os.environ.get("ENGINE_AR_STAMP", "0") == "1"
RING = 1 << int(os.environ.get("ENGINE_AR_STAMP_LOG2", "20"))  # all-reduce records per rank
HRING = 1 << 17  # graph head/tail records per rank
FIELDS = 8  # per all-reduce: tag, entry0, spin0, end0, memtime0, entryL, spinL, endL
HFIELDS = 6  # per head/tail: kind, tag, realtime, memtime, all-reduce count, 0

# Step kinds in a head/tail tag.
DECODE, VERIFY, CHAIN, PREFILL = 1, 2, 3, 4
_KIND_NAMES = {DECODE: "decode", VERIFY: "verify", CHAIN: "chain", PREFILL: "prefill"}


def tag(kind: int, size: int, batch: int, groups: int = 0, ahead: bool = False) -> int:
    """A step's tag: kind, run-ahead, groups (k + 1), batch (slots or tokens), bucket."""
    return ((kind << 28) | (int(ahead) << 27) | (min(groups, 15) << 23)
            | (min(batch, 2047) << 12) | min(size, 4095))


def untag(t: int) -> tuple[int, bool, int, int, int]:
    return (t >> 28) & 0xF, bool((t >> 27) & 1), (t >> 23) & 0xF, (t >> 12) & 0x7FF, t & 0xFFF


def _key(t: int) -> str:
    """A step class: ``chain_b2_k3_ahead``, ``prefill_s512``, ..."""
    kind, ahead, g, batch, size = untag(t)
    if kind == PREFILL:
        return f"prefill_s{size}"
    return (f"{_KIND_NAMES.get(kind, kind)}_b{batch}" + (f"_k{g - 1}" if g else "")
            + ("_ahead" if ahead else "_serial"))


if triton is not None:

    @triton.jit
    def realtime():
        """The GPU's constant-rate clock (25 MHz on MI250: j261), as a scalar int64."""
        return tl.inline_asm_elementwise("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [],
                                          dtype=tl.int64, is_pure=False, pack=1)

    @triton.jit
    def memtime():
        """The shader clock counter, as a scalar int64."""
        return tl.inline_asm_elementwise("s_memtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [],
                                         dtype=tl.int64, is_pure=False, pack=1)

    @triton.jit(do_not_specialize=["tag"])
    def _stamp_kernel(ring, ctr, epoch_ptr, tag, KIND: tl.constexpr, HR: tl.constexpr):
        t = realtime()
        c = memtime()
        e = tl.load(epoch_ptr, volatile=True)
        i = tl.atomic_add(ctr, 1)
        if i < HR:
            base = ring + i.to(tl.int64) * 6
            tl.store(base, KIND)
            tl.store(base + 1, tag)
            tl.store(base + 2, t)
            tl.store(base + 3, c)
            tl.store(base + 4, e)


class Stamps:
    """The stamp rings of one TP2 pair (``tp._LL`` owns one when ``ENABLED``)."""

    def __init__(self, devices: Sequence[torch.device], epochs: Sequence[torch.Tensor]) -> None:
        self.devices = list(devices)
        self.epochs = list(epochs)  # each rank's all-reduce epochs (the count is entry 0)
        self.ring = [torch.zeros((RING, FIELDS), dtype=torch.int64, device=d) for d in self.devices]
        self.ctr = [torch.zeros(1, dtype=torch.int32, device=d) for d in self.devices]
        # The loopback warm-up calls (each rank against itself) write here, not in the ring.
        self.loop_ring = [torch.zeros((RING, FIELDS), dtype=torch.int64, device=d) for d in self.devices]
        self.loop_ctr = [torch.zeros(1, dtype=torch.int32, device=d) for d in self.devices]
        self.hring = [torch.zeros((HRING, HFIELDS), dtype=torch.int64, device=d) for d in self.devices]
        self.hctr = [torch.zeros(1, dtype=torch.int32, device=d) for d in self.devices]
        self.layer_types: list[str] = []
        self.rt_hz = 25e6
        self.calibration: dict[str, Any] = {}
        self.host: dict[int, list] = {}  # tag -> [n, rank-0 replay s, rank-1 replay s, stamp s]
        self._lock = threading.Lock()
        self._cache: tuple[tuple[int, int], dict[str, Any]] | None = None
        PAIRS.append(self)

    def ar_args(self, rank: int, loop: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if loop:
            return self.loop_ring[rank], self.loop_ctr[rank]
        return self.ring[rank], self.ctr[rank]

    def stamp(self, rank: int, kind: int, t: int) -> None:
        """A head (kind 1) or tail (kind 2) record on ``rank``'s current stream."""
        _stamp_kernel[(1,)](self.hring[rank], self.hctr[rank], self.epochs[rank], t,
                            KIND=kind, HR=HRING, num_warps=1)

    def replay(self, graphs: Sequence[Any], devices: Sequence[torch.device], t: int) -> None:
        """Both ranks' graph replays as ``TPModel`` issues them (rank 0's, then rank 1's), a
        head stamp before each and the tail stamps after both, so a stamp launch sits between
        the two replays only once (rank 1's head); the host time of each call is kept per tag."""
        now = time.perf_counter
        ts = [now()]
        for rank, (g, dev) in enumerate(zip(graphs, devices, strict=True)):
            with torch.cuda.device(dev):
                self.stamp(rank, 1, t)
                ts.append(now())
                g.replay()
                ts.append(now())
        for rank, dev in enumerate(devices):
            with torch.cuda.device(dev):
                self.stamp(rank, 2, t)
        h = self.host.setdefault(t, [0, 0.0, 0.0, 0.0])
        h[0] += 1
        h[1] += ts[2] - ts[1]  # rank 0's replay call
        h[2] += ts[4] - ts[3]  # rank 1's replay call
        h[3] += (ts[1] - ts[0] + ts[3] - ts[2]) / 2  # one head stamp launch

    def calibrate(self) -> None:
        """Compile the stamp kernel and measure the realtime clock's rate against the host's."""
        rates = []
        for rank, dev in enumerate(self.devices):
            with torch.cuda.device(dev):
                # Compile both kinds first: j261 timed rank 0's first tail-kind compile into
                # the host interval and read 20 MHz for a 25 MHz clock.
                self.stamp(rank, 1, 0)
                self.stamp(rank, 2, 0)
                torch.cuda.synchronize(dev)
                self.hring[rank].zero_()
                self.hctr[rank].zero_()
                self.stamp(rank, 1, 0)
                torch.cuda.synchronize(dev)
                h0 = time.perf_counter()
                self.stamp(rank, 2, 0)
                torch.cuda.synchronize(dev)
                time.sleep(0.05)
                self.stamp(rank, 1, 0)
                torch.cuda.synchronize(dev)
                h1 = time.perf_counter()
                self.stamp(rank, 2, 0)
                torch.cuda.synchronize(dev)
                rec = self.hring[rank][:4].cpu()
                rt = (int(rec[3, 2]) + int(rec[2, 2]) - int(rec[1, 2]) - int(rec[0, 2])) / 2
                mt = (int(rec[3, 3]) + int(rec[2, 3]) - int(rec[1, 3]) - int(rec[0, 3])) / 2
                rates.append((rt / (h1 - h0), mt / (h1 - h0)))
                self.hring[rank].zero_()
                self.hctr[rank].zero_()
                torch.cuda.synchronize(dev)
        self.rt_hz = statistics.mean(r for r, _ in rates)
        self.calibration = {
            "realtime_mhz": [round(r / 1e6, 2) for r, _ in rates],
            "memtime_mhz_idle": [round(m / 1e6, 1) for _, m in rates],
        }
        print(f"engine: AR stamps on (plan 55): {self.calibration}", flush=True)

    # ------------------------------------------------------------ readout

    def _read(self) -> tuple[list, list]:
        ars, heads = [], []
        for rank, dev in enumerate(self.devices):
            torch.cuda.synchronize(dev)
            n = min(int(self.epochs[rank][0].item()) + 1, RING)
            h = min(int(self.hctr[rank].item()), HRING)
            ars.append(self.ring[rank][:n].cpu().tolist())
            heads.append(self.hring[rank][:h].cpu().tolist())
        return ars, heads

    def summary(self) -> dict[str, Any]:
        with self._lock:
            key = tuple(int(self.hctr[r].item()) for r in range(2))
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
            try:
                out = self._summary()
            except Exception as exc:  # noqa: BLE001 - instrumentation must not fail /health
                out = {"error": f"{type(exc).__name__}: {exc}"}
            self._cache = (key, out)
            return out

    def _classes(self, n: int, bf16: list[bool]) -> list[str]:
        """The class of each all-reduce of a step (the segment it ends)."""
        out = []
        layers = len(self.layer_types)
        for i in range(n):
            if i < 2 * layers and bf16[i]:
                gdn = self.layer_types[i // 2] == "linear_attention"
                out.append(("gdn" if gdn else "attn") if i % 2 == 0 else ("moe_g" if gdn else "moe_a"))
            else:
                out.append("mtp" if bf16[i] else "argmax")
        return out

    def _summary(self) -> dict[str, Any]:
        ars, heads = self._read()
        us = 1e6 / self.rt_hz
        h0, h1 = heads
        steps = []
        bad = 0
        prev_tail = [None, None]
        j = 0
        while j + 1 < min(len(h0), len(h1)):
            a, b = h0[j], h1[j]
            ta, tb = h0[j + 1], h1[j + 1]
            j += 2
            if a[0] != 1 or ta[0] != 2 or b[0] != 1 or tb[0] != 2 or a[1] != b[1] or a[1] == 0:
                bad += 1
                if a[0] != 1 or b[0] != 1:  # lost alignment of the head/tail sequence
                    break
                continue
            lo = [int(a[4]), int(b[4])]
            hi = [int(ta[4]), int(tb[4])]
            n = hi[0] - lo[0]
            if n <= 0 or n != hi[1] - lo[1] or hi[0] >= RING or hi[1] >= RING:
                bad += 1
                prev_tail = [ta[2], tb[2]]
                continue
            recs = [ars[0][lo[0] + 1 : hi[0] + 1], ars[1][lo[1] + 1 : hi[1] + 1]]
            if any(r[1] == 0 or r[7] == 0 for rr in recs for r in rr):
                bad += 1
                prev_tail = [ta[2], tb[2]]
                continue
            steps.append((a[1], (a, b), (ta, tb), recs, list(prev_tail)))
            prev_tail = [ta[2], tb[2]]
        groups: dict[str, list[dict[str, Any]]] = {}
        seqs: dict[str, list[str]] = {}
        for t, head, tail, recs, prev in steps:
            key = _key(int(t))
            n = len(recs[0])
            r0, r1 = recs
            spin0 = [r[2] for r in r0]
            spin1 = [r[2] for r in r1]
            c = statistics.median(s1 - s0 for s0, s1 in zip(spin0, spin1))  # rank-1 clock - rank-0
            ent = [[r[1] for r in r0], [r[1] - c for r in r1]]
            spn = [spin0, [s - c for s in spin1]]
            end = [[max(r[3], r[7]) for r in r0], [max(r[3], r[7]) - c for r in r1]]
            hd = [head[0][2], head[1][2] - c]
            tl_ = [tail[0][2], tail[1][2] - c]
            d = [[ent[r][0] - hd[r]] + [ent[r][i] - end[r][i - 1] for i in range(1, n)] for r in range(2)]
            w = [[spn[r][i] - ent[r][i] for i in range(n)] for r in range(2)]
            floor = [min(w[0][i], w[1][i]) for i in range(n)]
            start = min(hd)
            seg0 = max(hd[0] + d[0][0], hd[1] + d[1][0]) - start - (d[0][0] + d[1][0]) / 2
            imb = [max(d[0][i], d[1][i]) - (d[0][i] + d[1][i]) / 2 for i in range(1, n)]
            bf16 = [bool((r[0] >> 32) & 1) for r in r0]
            classes = self._classes(n, bf16)
            seqs.setdefault(key, classes)
            by_class: dict[str, list[float]] = {}
            for i in range(n):
                cl = by_class.setdefault(classes[i], [0.0, 0.0])
                cl[0] += (d[0][i] - d[1][i]) * us
                if i:
                    cl[1] += imb[i - 1] * us
            mt = [(recs[r][-1][4] - recs[r][0][4]) / max(1, recs[r][-1][1] - recs[r][0][1]) for r in range(2)]
            rec = {
                "D": (hd[1] - hd[0]) * us,
                "absD": abs(hd[1] - hd[0]) * us,
                "wait0": sum(w[0]) * us, "wait1": sum(w[1]) * us, "floor": sum(floor) * us,
                "excess0": (sum(w[0]) - sum(floor)) * us, "excess1": (sum(w[1]) - sum(floor)) * us,
                "L": (seg0 + sum(imb)) * us, "L_start": seg0 * us, "L_imb": sum(imb) * us,
                "span0": (tl_[0] - hd[0]) * us, "span1": (tl_[1] - hd[1]) * us,
                "tail0": (tl_[0] - end[0][-1]) * us, "tail1": (tl_[1] - end[1][-1]) * us,
                "mhz0": mt[0] * self.rt_hz / 1e6, "mhz1": mt[1] * self.rt_hz / 1e6,
                "nar": n, "_cls": by_class,
                "_d": [(d[0][i] - d[1][i]) * us for i in range(n)],
            }
            if prev[0] is not None:
                rec["gap0"] = (head[0][2] - prev[0]) * us
                rec["gap1"] = (head[1][2] - prev[1]) * us
            groups.setdefault(key, []).append(rec)
        out: dict[str, Any] = {"steps": len(steps), "unpaired": bad, "rt_hz": round(self.rt_hz),
                               **self.calibration}
        metrics = ("D", "absD", "L", "L_start", "L_imb", "wait0", "wait1", "floor", "excess0",
                   "excess1", "span0", "span1", "tail0", "tail1", "gap0", "gap1", "mhz0", "mhz1", "nar")

        def stats(values: list[float]) -> list[float]:
            ordered = sorted(values)
            q = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]  # noqa: E731
            return [round(q(0.5), 1), round(q(0.9), 1), round(sum(values) / len(values), 1)]

        prefill_all: list[dict[str, Any]] = []
        for key, rs in sorted(groups.items()):
            if key.startswith("prefill"):
                prefill_all.extend(rs)
        if prefill_all:
            groups["prefill_all"] = prefill_all
        for key, rs in sorted(groups.items()):
            entry: dict[str, Any] = {"n": len(rs)}
            for m in metrics:
                vals = [r[m] for r in rs if m in r]
                if vals:
                    entry[m] = stats(vals)
            cls: dict[str, list[float]] = {}
            for name in sorted({c for r in rs for c in r["_cls"]}):
                signed = [r["_cls"][name][0] for r in rs if name in r["_cls"]]
                share = [r["_cls"][name][1] for r in rs if name in r["_cls"]]
                cls[name] = [round(statistics.mean(signed), 1),
                             round(statistics.pstdev(signed), 1), round(statistics.mean(share), 1)]
            entry["class_d0_minus_d1_mean_sd_Limb"] = cls
            if not key.startswith("prefill") and len(rs) >= 20 and len({len(r["_d"]) for r in rs}) == 1:
                n = len(rs[0]["_d"])
                entry["per_ar_d0_minus_d1"] = [round(statistics.mean(r["_d"][i] for r in rs), 1)
                                               for i in range(n)]
                entry["classes"] = "".join(_LETTER[c] for c in seqs[key])
            out[key] = entry
        host: dict[str, list[float]] = {}
        for t, h in list(self.host.items()):
            for key in (_key(t), "prefill_all") if _key(t).startswith("prefill") else (_key(t),):
                acc = host.setdefault(key, [0, 0.0, 0.0, 0.0])
                for i in range(4):
                    acc[i] += h[i]
        for key, (n, r0, r1, st) in host.items():
            if key in out and n:
                out[key]["host_us_replay0_replay1_stamp"] = [round(1e6 * r0 / n, 1), round(1e6 * r1 / n, 1),
                                                            round(1e6 * st / n, 1)]
        return out


PAIRS: list[Stamps] = []
# One letter per all-reduce in ``classes``: G/A the GDN/attention mixer's, g/a the MoE's after
# a GDN/attention layer, T the MTP head's, X the fp32 argmax's.
_LETTER = {"gdn": "G", "attn": "A", "moe_g": "g", "moe_a": "a", "mtp": "T", "argmax": "X"}


def summary() -> dict[str, Any]:
    if not ENABLED or not PAIRS:
        return {}
    return {"ar_stamp": [p.summary() for p in PAIRS]}
