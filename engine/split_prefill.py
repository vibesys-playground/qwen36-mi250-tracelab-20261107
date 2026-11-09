"""Split one long uncached prefill across two GCD pairs (plan 41).

A first-round prompt has no cached prefix, so its whole pass runs on the replica (a TP2 pair)
that holds the session, while the other pairs keep decoding. Here a process hosts two pairs
(``engine.pair_host``), and a single-segment prefill of at least ``ENGINE_SPLIT_MIN_TOKENS``
uncached tokens (default 1,500; ``ENGINE_SPLIT_PREFILL=0`` turns all of this off) runs as two
sequence halves: the owner pair (the session's replica) runs positions ``0 .. n1`` into its
slot, and the other pair (the helper) runs ``n1 .. n`` as a continuation segment over the
first half's state, one layer behind. That is the math every later round of a session already
runs (a prefill segment with ``start > 0``), with the same weights and kernels.

Each half pass is the captured prefill pass (``FastModel._pg_body``) captured as a chain of
one graph per layer (``Chains``), so the two pairs can exchange state at every layer boundary.
After the owner's layer ``i`` (on both ranks), rank ``r`` of the owner copies that layer's
state to rank ``r`` of the helper (the same head and expert shard): a Gated DeltaNet layer's
convolution and recurrent rows, an attention layer's KV pages of the first half (through a
staging buffer, so fragmented page lists cost three copies, not one per run), and after the
last layer the MTP head's KV. The helper's layer ``i`` waits for that copy only (cross-device
events on side streams; the host never waits). After the helper's layer ``i``, its final rows
and its second-half KV pages are copied back into the owner's slot, so the owner ends holding
the state of the whole prompt, as after a single pass; the logits and the MTP draft of the
last token come from the helper. The helper's slot is a reserved one, reset afterwards.

Scheduling: each pair's worker thread runs its loop iterations as turns of its ``Gate``. An
owner borrows the other pair only between that pair's turns: it waits for the helper's current
turn (a decode step) to end rather than preempting it, and does not split when the helper is in
a prefill pass. One split runs per process at a time (``Splitter``'s token), so a pair is never
booked by two owners, and two pairs that want each other at once cannot deadlock: the one that
does not get the token runs its pass unsplit.

Numerics: the split equals the continuation prefill bit for bit (plan 41 step 0); against the
single pass it differs as a pass of another padded size does (plan 41 profile j203).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from typing import Any

import torch

PAGE = 64  # engine.fast.model.PAGE (checked when a Pair is made)

ENABLED = os.environ.get("ENGINE_SPLIT_PREFILL", "1") != "0"
MIN_TOKENS = int(os.environ.get("ENGINE_SPLIT_MIN_TOKENS", "1500"))


def enabled() -> bool:
    return ENABLED


# ----------------------------------------------------------------------------- turns


class Gate:
    """A pair's worker loop turns, and lending the pair to another pair's owner between them.

    The worker calls ``enter`` when it starts an iteration (after any idle wait for a job) and
    ``leave`` before the next one. ``borrow`` waits until the worker is between turns and keeps
    it there until ``give_back``. ``may_prefill`` tells the worker whether to run a prefill pass
    in this turn: not while someone waits to borrow the pair (the pass waits for the next turn)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._in_turn = False
        self._prefill = False
        self._lent = False
        self._wanted = False

    def enter(self) -> None:
        with self._cond:
            while self._lent or self._wanted:
                self._cond.wait()
            self._in_turn = True

    def leave(self) -> None:
        with self._cond:
            if self._in_turn:
                self._in_turn = False
                self._prefill = False
                self._cond.notify_all()

    def may_prefill(self) -> bool:
        with self._cond:
            if self._wanted:
                return False
            self._prefill = True
            return True

    def borrow(self) -> bool:
        """Book the pair for one split: False (at once) if it is in a prefill pass or lent."""
        with self._cond:
            if self._lent or self._wanted or (self._in_turn and self._prefill):
                return False
            self._wanted = True
            while self._in_turn:
                self._cond.wait()
            self._wanted = False
            self._lent = True
            return True

    def give_back(self) -> None:
        with self._cond:
            self._lent = False
            self._cond.notify_all()


class NoGate:
    """``Gate`` for a worker that is never lent."""

    def enter(self) -> None:
        pass

    def leave(self) -> None:
        pass

    def may_prefill(self) -> bool:
        return True


# ----------------------------------------------------------------------------- chained passes


class Chains:
    """A pair's captured prefill pass of each size as a chain of one graph per layer (the last
    graph also holds the head: logits and the MTP draft), per rank."""

    def __init__(self, tp: Any) -> None:
        self.tp = tp
        self.layers = len(tp.ranks[0].layers)
        self.graphs: dict[int, list[list[torch.cuda.CUDAGraph]]] = {}
        self.bufs: dict[int, list[torch.Tensor]] = {}
        self.outs: dict[int, list[tuple[torch.Tensor, torch.Tensor | None]]] = {}
        self._carries: dict[int, list[list[Any]]] = {}
        self.pools = {m.embed.device: torch.cuda.graph_pool_handle() for m in tp.ranks}
        self.streams = {m.embed.device: torch.cuda.Stream(m.embed.device) for m in tp.ranks}
        self.error: str | None = None

    def _part(self, m: Any, buf: torch.Tensor, size: int, i: int, carry: Any):
        if i == 0:
            st, x, h = m._pg_begin(buf, size)
        else:
            st, x, h = carry
        x, h = m._pg_layers(i, i + 1, st, x, h)
        out = m._pg_end(st, h) if i == self.layers - 1 else None
        return (st, x, h), out

    def capture(self, sizes: Sequence[int]) -> None:
        """Capture every size of ``sizes`` the pair also captured as a whole pass (largest
        first). Warm-up runs touch the dummy slot only. A failure leaves ``error`` set and the
        sizes captured so far usable."""
        tp = self.tp
        r0, r1 = tp.ranks
        dummy = r0.dummy
        scratch = r0.slot_pages[dummy][0]
        from engine.fast.model import FastModel

        t0 = time.perf_counter()
        try:
            for m in tp.ranks:
                for meta in m.devices.values():
                    meta.page_table[dummy].fill_(scratch)
            for size in sorted(sizes, reverse=True):
                if size not in tp.pg:
                    continue
                count = size - 3
                host = FastModel.pg_host(size, [(7 * j) % 1000 + 100 for j in range(count)], 0, None,
                                         dummy, r0._g(dummy))
                bufs = [m._pg_inputs(size) for m in tp.ranks]
                for b in bufs:
                    b.copy_(host)

                def warm(m: Any, size: int = size, bufs: list = bufs) -> None:
                    dev = m.embed.device
                    s = self.streams[dev]
                    s.wait_stream(torch.cuda.current_stream(dev))
                    with torch.cuda.stream(s):
                        carry = None
                        for i in range(self.layers):
                            carry, _ = self._part(m, bufs[m.rank], size, i, carry)
                    torch.cuda.current_stream(dev).wait_stream(s)
                    torch.cuda.synchronize(dev)

                tp._both(warm)

                def cap(m: Any, mode: str, size: int = size, bufs: list = bufs):
                    dev = m.embed.device
                    graphs, carries, out, carry = [], [], None, None
                    with torch.cuda.device(dev), torch.no_grad():
                        for i in range(self.layers):
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g, pool=self.pools[dev], stream=self.streams[dev],
                                                  capture_error_mode=mode):
                                carry, out = self._part(m, bufs[m.rank], size, i, carry)
                            graphs.append(g)
                            carries.append(carry)
                    return graphs, carries, out

                errors = []
                for mode in ("global", "thread_local", "relaxed"):
                    try:
                        c0 = cap(r0, mode)
                        c1 = tp._pool.submit(tp._on_rank1, lambda: cap(r1, mode)).result()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{mode}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")
                        for d in tp.devices_tp:
                            try:
                                torch.cuda.synchronize(d)
                            except Exception:  # noqa: BLE001
                                pass
                        continue
                    self.graphs[size] = [c0[0], c1[0]]
                    self._carries[size] = [c0[1], c1[1]]
                    self.outs[size] = [c0[2], c1[2]]
                    self.bufs[size] = bufs
                    break
                else:
                    raise RuntimeError(f"size {size}: {' | '.join(errors)}")
        except Exception as exc:  # noqa: BLE001 - those sizes stay unsplit
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"engine: split prefill chain capture failed: {self.error}", flush=True)
        finally:
            r0._push_table(dummy)
            for m in tp.ranks:
                for conv, rec in m.gdn_state.values():
                    for row in (dummy, dummy + m.gdn_rows):
                        if row < conv.shape[0]:
                            conv[row].zero_()
                            rec[row].zero_()
            for d in tp.devices_tp:
                torch.cuda.synchronize(d)
        print(f"engine: split prefill chains for sizes {sorted(self.graphs)} on "
              f"{[str(d) for d in tp.devices_tp]} in {time.perf_counter() - t0:.1f}s; reserved "
              f"{[round(torch.cuda.memory_reserved(d) / 2**30, 2) for d in tp.devices_tp]} GiB",
              flush=True)

    def fill(self, size: int, ids: Sequence[int], start: int, follow: int | None, slot: int) -> None:
        from engine.fast.model import FastModel

        host = FastModel.pg_host(size, ids, start, follow, slot, self.tp.ranks[0]._g(slot))
        for b in self.bufs[size]:
            b.copy_(host)

    def replay(self, size: int, i: int) -> list[torch.cuda.Event]:
        """Layer ``i``'s graph on both ranks (on their current streams); an event per rank."""
        events = []
        for r, m in enumerate(self.tp.ranks):
            dev = m.embed.device
            with torch.cuda.device(dev):
                self.graphs[size][r][i].replay()
                e = torch.cuda.Event()
                e.record(torch.cuda.current_stream(dev))
                events.append(e)
        return events


# ----------------------------------------------------------------------------- the pairs


class Pair:
    """One pair of a split-hosting process: its model, gate, chains, reserved helper slot and
    the per-device staging buffers and side streams of the state copies."""

    def __init__(self, index: int, tp: Any, gate: Gate) -> None:
        from engine.fast import model as fm

        if fm.PAGE != PAGE:
            raise RuntimeError(f"KV page size {fm.PAGE} != {PAGE}")
        self.index = index
        self.tp = tp
        self.gate = gate
        self.chains = Chains(tp)
        self.slot = tp.slots - 1  # reserved: the slot this pair computes a helper half in
        self.splits_owned = 0
        self.splits_helped = 0
        self.split_ms = 0.0
        self._stage: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.fwd = {m.embed.device: torch.cuda.Stream(m.embed.device) for m in tp.ranks}
        self.back = {m.embed.device: torch.cuda.Stream(m.embed.device) for m in tp.ranks}

    def stage(self, rank: int, kind: str, pages: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank ``rank``'s (K, V) staging buffers of ``kind`` ('send' or 'recv') for ``pages``."""
        key = (rank, kind)
        held = self._stage.get(key)
        if held is None or held[0].shape[0] < pages:
            m = self.tp.ranks[rank]
            k_pool, v_pool = next(iter(m.kv.values()))
            n = max(pages, 40)
            held = (torch.empty((n, *k_pool.shape[1:]), dtype=k_pool.dtype, device=k_pool.device),
                    torch.empty((n, *v_pool.shape[1:]), dtype=v_pool.dtype, device=v_pool.device))
            self._stage[key] = held
        return held[0][:pages], held[1][:pages]


def _copy_layer(src_m: Any, dst_m: Any, layer: int, src_row: int, dst_row: int,
                src_idx: torch.Tensor | None, dst_idx: torch.Tensor | None,
                send: tuple[torch.Tensor, torch.Tensor] | None,
                recv: tuple[torch.Tensor, torch.Tensor] | None) -> None:
    """Layer ``layer``'s state from ``src_m`` to ``dst_m`` (the current streams of both devices
    are the copy streams): GDN rows, or the KV pages ``src_idx`` -> ``dst_idx`` via staging."""
    if layer in src_m.gdn_state:
        for ts, td in zip(src_m.gdn_state[layer], dst_m.gdn_state[layer], strict=True):
            td[dst_row].copy_(ts[src_row])
    if layer in src_m.kv and src_idx is not None:
        for ts, td, sb, rb in zip(src_m.kv[layer], dst_m.kv[layer], send, recv, strict=True):
            torch.index_select(ts, 0, src_idx, out=sb)
            rb.copy_(sb)
            td.index_copy_(0, dst_idx, rb)


class Splitter:
    """The pairs of one process and the split itself; see the module doc."""

    def __init__(self, pairs: list[Pair]) -> None:
        self.pairs = pairs
        self._token = threading.Lock()

    def helper_of(self, owner: Pair) -> Pair | None:
        others = [p for p in self.pairs if p is not owner]
        return others[0] if others else None

    def wants(self, owner: Pair, segments, follows) -> bool:
        if len(segments) != 1:
            return False
        slot, ids, start = segments[0]
        if start != 0 or len(ids) < MIN_TOKENS:
            return False
        helper = self.helper_of(owner)
        if helper is None:
            return False
        n1 = len(ids) // 2
        s1, s2 = owner.tp._pg_size(n1), helper.tp._pg_size(len(ids) - n1)
        return s1 in owner.chains.graphs and s2 in helper.chains.graphs

    def try_split(self, owner: Pair, slot: int, ids: Sequence[int], start: int,
                  follow: int | None) -> torch.Tensor | None:
        """The split pass's logits ``[1, vocab]`` on the owner's first GPU, or None (run it
        unsplit) when another split runs or the helper is busy with a prefill."""
        if not self._token.acquire(blocking=False):
            return None
        try:
            helper = self.helper_of(owner)
            if helper is None or not helper.gate.borrow():
                return None
            try:
                try:
                    helper.tp.ranks[0]._ensure_pages(helper.slot, start + len(ids))
                except RuntimeError:  # the helper's KV cache is full: unsplit
                    helper.tp.reset(helper.slot)
                    return None
                t0 = time.perf_counter()
                try:
                    out = self._run(owner, helper, slot, ids, start, follow)
                finally:
                    helper.tp.reset(helper.slot)
                owner.splits_owned += 1
                helper.splits_helped += 1
                owner.split_ms += (time.perf_counter() - t0) * 1e3
                return out
            finally:
                helper.gate.give_back()
        finally:
            self._token.release()

    def _run(self, owner: Pair, helper: Pair, slot: int, ids: Sequence[int], start: int,
             follow: int | None) -> torch.Tensor:
        if start != 0:
            raise ValueError("only an uncached segment is split")
        o, h = owner.tp, helper.tp
        n = len(ids)
        n1 = n // 2
        s1, s2 = o._pg_size(n1), h._pg_size(n - n1)
        hs = helper.slot
        oc, hc = owner.chains, helper.chains
        # Pages: half 1 (the boundary page in full) forward; half 2 (from the boundary page) back.
        p1 = (n1 + PAGE - 1) // PAGE
        p_lo, p_hi = n1 // PAGE, (n + PAGE - 1) // PAGE
        o_pages, h_pages = o.ranks[0].slot_pages[slot], h.ranks[0].slot_pages[hs]
        idx = []  # per rank: (fwd src on owner, fwd dst on helper, back src on helper, back dst on owner)
        for r in range(2):
            od, hd = o.ranks[r].embed.device, h.ranks[r].embed.device
            idx.append((
                torch.tensor(o_pages[:p1], dtype=torch.long, device=od),
                torch.tensor(h_pages[:p1], dtype=torch.long, device=hd),
                torch.tensor(h_pages[p_lo:p_hi], dtype=torch.long, device=hd),
                torch.tensor(o_pages[p_lo:p_hi], dtype=torch.long, device=od),
            ))
        oc.fill(s1, ids[:n1], 0, ids[n1], slot)
        hc.fill(s2, ids[n1:], n1, follow, hs)
        streams = []
        for r in range(2):
            od, hd = o.ranks[r].embed.device, h.ranks[r].embed.device
            fs, fd = owner.fwd[od], helper.fwd[hd]
            bs, bd = helper.back[hd], owner.back[od]
            for s, dev in ((fs, od), (fd, hd), (bs, hd), (bd, od)):
                s.wait_stream(torch.cuda.current_stream(dev))
            streams.append((fs, fd, bs, bd))
        layers = oc.layers
        mtp = o.ranks[0].mtp_index if o.ranks[0].mtp is not None else None
        for i in range(layers):
            todo = [i] + ([mtp] if i == layers - 1 and mtp is not None else [])
            ea = oc.replay(s1, i)
            for r in range(2):
                om, hm = o.ranks[r], h.ranks[r]
                fs, fd, _, _ = streams[r]
                fs.wait_event(ea[r])
                with torch.cuda.stream(fd), torch.cuda.stream(fs):
                    send = owner.stage(r, "send", p1)
                    recv = helper.stage(r, "recv", p1)
                    for j in todo:
                        _copy_layer(om, hm, j, om._g(slot), hm._g(hs), idx[r][0], idx[r][1], send, recv)
                done = torch.cuda.Event()
                done.record(fd)
                torch.cuda.current_stream(hm.embed.device).wait_event(done)
            eb = hc.replay(s2, i)
            for r in range(2):
                om, hm = o.ranks[r], h.ranks[r]
                _, _, bs, bd = streams[r]
                bs.wait_event(eb[r])
                with torch.cuda.stream(bd), torch.cuda.stream(bs):
                    send = helper.stage(r, "send", p_hi - p_lo)
                    recv = owner.stage(r, "recv", p_hi - p_lo)
                    for j in todo:
                        _copy_layer(hm, om, j, hm._g(hs), om._g(slot), idx[r][2], idx[r][3], send, recv)
        # Every GPU's own stream waits for its side streams: the owner's slot is whole, the
        # helper's slot may be reset, and the next split's copies come after these.
        for r in range(2):
            od, hd = o.ranks[r].embed.device, h.ranks[r].embed.device
            fs, fd, bs, bd = streams[r]
            for s, dev in ((fs, od), (fd, hd), (bs, hd), (bd, od)):
                torch.cuda.current_stream(dev).wait_stream(s)
        outs = hc.outs[s2]
        logits = h._gather([outs[0][0], outs[1][0]]).to(o.ranks[0].embed.device)
        o.last_drafts = outs[0][1].clone() if outs[0][1] is not None else None
        return logits[None]


class SplitFast:
    """A pair's ``TPModel`` as the ``Generator`` sees it: everything is the model's own, except
    that a long uncached single-segment prefill may run split (``Splitter.try_split``)."""

    def __init__(self, splitter: Splitter, pair: Pair) -> None:
        self.__dict__["_splitter"] = splitter
        self.__dict__["_pair"] = pair
        self.__dict__["_tp"] = pair.tp

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_tp"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["_tp"], name, value)

    def _split(self, segments, follows) -> torch.Tensor | None:
        sp, pair = self._splitter, self._pair
        if not sp.wants(pair, segments, follows):
            return None
        slot, ids, start = segments[0]
        pair.tp.ranks[0]._ensure_pages(slot, start + len(ids))
        follow = follows[0] if follows is not None else None
        with torch.inference_mode():
            return sp.try_split(pair, slot, ids, start, follow)

    def prefill_packed(self, segments, follows=None) -> torch.Tensor:
        out = self._split(segments, follows)
        if out is not None:
            return out
        return self._tp.prefill_packed(segments, follows)

    def prefill(self, slot: int, ids: Sequence[int], start: int, follow: int | None = None) -> torch.Tensor:
        out = self._split([(slot, ids, start)], [follow])
        if out is not None:
            return out[0]
        return self._tp.prefill(slot, ids, start, follow)

    def split_stats(self) -> dict[str, int]:
        p = self._pair
        return {"split_prefills_owned": p.splits_owned, "split_prefills_helped": p.splits_helped,
                "split_prefill_ms": int(p.split_ms)}
