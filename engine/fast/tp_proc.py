"""Tensor parallelism with rank 1 in its own host process (plan 30): ``ENGINE_TP_PROC=1``.

Under ``engine.fast.tp`` both ranks of a replica run in one process, rank 1 on a worker
thread, so the two ranks take turns on one GIL: an eager prefill pass costs the host time of
both ranks' ~5k kernel launches one after the other (j139: a 561-token pass 196 ms with both
ranks, 130 ms for one rank alone). Here rank 1 is a child process with its own interpreter:

- **Load.** The parent loads and shards the checkpoint as before (``shard_model``), with
  rank 1's shard copies allocated from a memory pool of their own (``torch.cuda.MemPool``),
  so they share no allocator segment with the checkpoint's tensors: once those are freed,
  every other segment on rank 1's GPU goes back to the driver. The child maps the shard
  through torch's CUDA IPC (no copy) and builds its ``FastModel`` on it; the parent keeps
  the pool alive. (j150/j153: shard tensors carved out of the loader's segments pinned
  ~14 GB of freed blocks in them, and rank 1's KV pool no longer fit.)
- **Commands.** The parent keeps every piece of host state (page pool and page lists, state
  banks, prefix-cache snapshots) and sends rank 1 what changes, in program order, over a
  pipe: page-table rows, resets, bank flips, page copies, snapshot copies of rank 1's Gated
  DeltaNet rows (kept in the child under the snapshot's id), prefill passes, step inputs,
  graph replays and captures. The child runs each one as rank 1 of ``TPModel`` would; nothing
  waits for an answer except a capture's result.
- **Collectives.** Each process allocates the in-graph all-reduce's fine-grained receive
  buffers on its GPU and opens the peer's through ``hipIpcGetMemHandle`` /
  ``hipIpcOpenMemHandle`` (j147: opened on both GPUs, 1,100 checked calls, 2.5 us in a graph).
  The captured decode and verify graphs and ``_ll_allreduce_kernel`` are ``engine.fast.tp``'s,
  unchanged. Eager all-reduces (prefill, capture warm-up, the eager step) run the same kernel
  on a second, larger pair of buffers instead of the host mailbox: the same sum,
  ``bf16(fp32(a) + fp32(b))`` then ``bf16(residual + that)``, so the same bits. Rank 1's
  vocabulary shard of the logits reaches rank 0 through it too (an all-reduce with zeros).
  On the CPU (tests) the eager exchange goes through a second pipe.
- **Failure.** A command that fails in the child reports its traceback and the child exits;
  the parent raises it at its next command. The parent watches the child: when it dies the
  parent says so and (``ENGINE_TP_PROC_EXIT``, default on) exits, so the router stops serving
  instead of waiting on all-reduces that will never complete. The child exits when the parent's
  pipe closes or the parent is gone. ``close`` stops it.

``ENGINE_TP_PROC=0`` keeps the threads of ``engine.fast.tp``.
"""

from __future__ import annotations

import atexit
import ctypes
import math
import multiprocessing.connection
import os
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.multiprocessing  # noqa: F401 - registers the CUDA tensor reductions for the pipe

from engine.fast import kernels
from engine.fast import tp as tpm
from engine.fast.model import PAGE, FastModel, Snapshot

triton = tpm.triton

# Words (32-bit) per parity buffer of the eager all-reduce: a 2,048-token prefill pass of
# hidden 2,048 BF16 in one launch; larger tensors run in pieces.
EAGER_CAPACITY = int(os.environ.get("ENGINE_TP_PROC_CAPACITY", str(1 << 21)))
# Spin iterations before an eager call gives up. A rank's host may lag the other's GPU by a
# Triton compile or an autotune; a dead peer is caught by the process watch instead.
EAGER_MAXIT = int(os.environ.get("ENGINE_TP_PROC_MAXIT", str(1 << 30)))
# Seconds to wait for the child's answer at load or capture before failing.
REPLY_S = float(os.environ.get("ENGINE_TP_PROC_REPLY_S", "1800"))


def _exit_on_death() -> bool:
    return os.environ.get("ENGINE_TP_PROC_EXIT", "1") == "1"


# ================================================================================ IPC buffers


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]  # hipIpcMemHandle_t


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {rc}")


def _alloc_fine(hip: ctypes.CDLL, dev: torch.device, nbytes: int) -> int:
    """Fine-grained device memory (``hipDeviceMallocFinegrained``), zeroed."""
    ptr = ctypes.c_void_p()
    _check(hip.hipSetDevice(dev.index), "hipSetDevice")
    _check(hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1)),
           "hipExtMallocWithFlags")
    _check(hip.hipMemset(ptr, 0, ctypes.c_size_t(nbytes)), "hipMemset")
    _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
    return ptr.value


def _export(hip: ctypes.CDLL, ptr: int) -> bytes:
    h = _Handle()
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_Handle), ctypes.c_void_p]
    hip.hipIpcGetMemHandle.restype = ctypes.c_int
    _check(hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)), "hipIpcGetMemHandle")
    return ctypes.string_at(ctypes.addressof(h), 64)  # all 64 bytes (``.reserved`` stops at a NUL)


def _open(hip: ctypes.CDLL, handle: bytes) -> int:
    fn = hip.hipIpcOpenMemHandle
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), _Handle, ctypes.c_uint]
    fn.restype = ctypes.c_int
    h = _Handle()
    ctypes.memmove(ctypes.addressof(h), handle, 64)
    ptr = ctypes.c_void_p()
    _check(fn(ctypes.byref(ptr), h, 1), "hipIpcOpenMemHandle")  # hipIpcMemLazyEnablePeerAccess
    return ptr.value


class _LLProc(tpm._LL):
    """``engine.fast.tp._LL`` for one rank of a pair of processes: this rank's receive buffers
    are its own, the peer's are opened from the peer's IPC handles (``connect``). The in-graph
    call is ``_LL._launch`` as is; ``eager`` runs the same kernel on larger buffers."""

    def __init__(self, rank: int, devices: Sequence[torch.device]) -> None:  # noqa: D107
        self.hip = hip = tpm._hip()
        self.devices = list(devices)
        self.rank = rank
        me, peer = self.devices[rank], self.devices[1 - rank]
        _check(hip.hipSetDevice(me.index), "hipSetDevice")
        rc = hip.hipDeviceEnablePeerAccess(peer.index, 0)
        if rc not in (0, 704):  # 704: hipErrorPeerAccessAlreadyEnabled
            raise RuntimeError(f"hipDeviceEnablePeerAccess({me.index}->{peer.index}) failed: {rc}")
        hip.hipGetLastError()
        self.cap_eager = EAGER_CAPACITY - EAGER_CAPACITY % tpm.CHUNK
        # Plan 29's fused all-reduce + RMSNorm (ENGINE_GLUE_ARRMS, off by default) has no
        # cross-process buffers: ``Group.allreduce_rms`` falls back to all-reduce + add_rms.
        self.rms_cap = tpm.RMS_ROWS * tpm.RMS_WIDTH
        self.rms_ready = False
        own = _alloc_fine(hip, me, 2 * tpm.CAPACITY * 8)
        own_eager = _alloc_fine(hip, me, 2 * self.cap_eager * 8)
        loop = _alloc_fine(hip, me, 2 * tpm.CAPACITY * 8)
        torch.cuda.set_device(me)
        self._handles = {"recv": _export(hip, own), "eager": _export(hip, own_eager)}
        self.recv: list[Any] = [None, None]
        self.recv[rank] = tpm._Raw(own, torch.int64)
        self.eager_recv: list[Any] = [None, None]
        self.eager_recv[rank] = tpm._Raw(own_eager, torch.int64)
        self.loop: list[Any] = [None, None]
        self.loop[rank] = tpm._Raw(loop, torch.int64)
        ctas = tpm.CAPACITY // tpm.CHUNK
        self.epoch: list[Any] = [None, None]
        self.epoch[rank] = torch.zeros(ctas, dtype=torch.int32, device=me)
        self.loop_epoch: list[Any] = [None, None]
        self.loop_epoch[rank] = torch.zeros(ctas, dtype=torch.int32, device=me)
        self.err: list[Any] = [None, None]
        self.err[rank] = torch.zeros(1, dtype=torch.int32, device=me)
        self.eager_epoch = torch.zeros(self.cap_eager // tpm.CHUNK, dtype=torch.int32, device=me)
        # Compile every variant before any rendezvous, against this rank's own loop buffer.
        for dtype, residual in ((torch.bfloat16, False), (torch.bfloat16, True), (torch.float32, False),
                                (torch.float32, True)):
            x = torch.zeros((1, 256), dtype=dtype, device=me)
            self._launch(rank, x, x if residual else None, loop=True)
            self._eager_launch(x, x if residual else None, self.loop[rank], self.loop[rank],
                               self.loop_epoch[rank], tpm.CAPACITY)
        torch.cuda.synchronize(me)
        self.err[rank].zero_()

    def handles(self) -> dict[str, bytes]:
        return dict(self._handles)

    def connect(self, handles: dict[str, bytes]) -> None:
        """Open the peer's receive buffers (its ``handles()``) on this rank's GPU."""
        me = self.devices[self.rank]
        _check(self.hip.hipSetDevice(me.index), "hipSetDevice")
        self.recv[1 - self.rank] = tpm._Raw(_open(self.hip, handles["recv"]), torch.int64)
        self.eager_recv[1 - self.rank] = tpm._Raw(_open(self.hip, handles["eager"]), torch.int64)
        torch.cuda.set_device(me)

    def _eager_launch(self, x: torch.Tensor, residual: torch.Tensor | None, local: Any, peer: Any,
                      epoch: torch.Tensor, cap: int) -> torch.Tensor:
        x = x.contiguous()
        nbytes = x.numel() * x.element_size()
        if nbytes % 4 or x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"all-reduce of {tuple(x.shape)} {x.dtype} is not supported")
        out = torch.empty_like(x)
        if not nbytes:
            return out
        if residual is not None:
            residual = residual.contiguous()
            if residual.dtype != x.dtype or residual.shape != x.shape:
                raise ValueError("the residual must match the all-reduced tensor")
        xi = x.view(-1).view(torch.int32)
        oi = out.view(-1).view(torch.int32)
        ri = residual.view(-1).view(torch.int32) if residual is not None else xi
        words = nbytes // 4
        with torch.cuda.device(x.device):
            for lo in range(0, words, cap):
                nw = min(cap, words - lo)
                tpm._ll_allreduce_kernel[(triton.cdiv(nw, tpm.CHUNK),)](
                    xi[lo : lo + nw], ri[lo : lo + nw], oi[lo : lo + nw], local, peer, epoch,
                    self.err[self.rank], nw, cap, CHUNK=tpm.CHUNK, BF16=x.dtype == torch.bfloat16,
                    RES=residual is not None, MAXIT=EAGER_MAXIT, num_warps=1,
                )
        return out

    def eager(self, x: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        """``[residual +] (x_self + x_peer)`` outside a graph, stream-ordered on both GPUs."""
        r = self.rank
        return self._eager_launch(x, residual, self.eager_recv[r], self.eager_recv[1 - r],
                                  self.eager_epoch, self.cap_eager)

    def timeouts(self) -> int:
        return int(self.err[self.rank].item())


# ================================================================================ weights


class _Leaf:
    """Where a CUDA tensor of rank 1's shard sat in the tree (sent separately)."""

    __slots__ = ("i",)

    def __init__(self, i: int) -> None:  # noqa: D107
        self.i = i


def _strip(obj: Any, leaves: list[torch.Tensor], index: dict[int, int]) -> Any:
    """``obj`` with every CUDA tensor replaced by a ``_Leaf`` (namespaces edited in place)."""
    if isinstance(obj, torch.Tensor):
        if not obj.is_cuda:
            return obj
        if id(obj) not in index:
            index[id(obj)] = len(leaves)
            leaves.append(obj)
        return _Leaf(index[id(obj)])
    if isinstance(obj, list):
        return [_strip(x, leaves, index) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_strip(x, leaves, index) for x in obj)
    if isinstance(obj, dict):
        return {k: _strip(v, leaves, index) for k, v in obj.items()}
    if isinstance(obj, (tpm.NS, tpm._Experts)):
        for k, v in list(vars(obj).items()):
            setattr(obj, k, _strip(v, leaves, index))
    return obj


def _fill(obj: Any, table: dict[int, torch.Tensor]) -> Any:
    """``_strip`` undone with this process's copies."""
    if isinstance(obj, _Leaf):
        return table[obj.i]
    if isinstance(obj, list):
        return [_fill(x, table) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_fill(x, table) for x in obj)
    if isinstance(obj, dict):
        return {k: _fill(v, table) for k, v in obj.items()}
    if isinstance(obj, (tpm.NS, tpm._Experts)):
        for k, v in list(vars(obj).items()):
            setattr(obj, k, _fill(v, table))
    return obj


# ================================================================================ the group


def _to_wire(t: torch.Tensor) -> tuple[str, tuple[int, ...], bytes]:
    t = t.detach().contiguous().cpu()
    return str(t.dtype).split(".")[-1], tuple(t.shape), t.view(-1).view(torch.uint8).numpy().tobytes()


def _from_wire(w: tuple[str, tuple[int, ...], bytes]) -> torch.Tensor:
    dtype, shape, data = w
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8) if data else torch.empty(0, dtype=torch.uint8)
    return raw.view(getattr(torch, dtype)).reshape(shape)


class ProcGroup(tpm.Group):
    """``Group`` for one rank of a pair of processes (same ``allreduce`` and ``argmax``)."""

    # No ``_exchange`` (the peer's own tensor) across processes: callers that would add the
    # peer's partial themselves (``FastModel._reduce_rms``) take the all-reduce's sum instead.
    sums_only = True

    def _exchange(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("tensor-parallel ranks in two processes exchange sums only")

    def __init__(self, rank: int, devices: Sequence[torch.device], ll: _LLProc | None, data: Any) -> None:  # noqa: D107
        self.devices = list(devices)
        self.rank = rank
        self.ll = ll
        self.data = data  # the CPU exchange's pipe (tests)
        self.solo = False
        self._broken = None

    def start(self) -> None:
        pass

    def abort(self, why: str) -> None:
        pass

    def _swap(self, x: torch.Tensor) -> torch.Tensor:
        """The peer's ``x`` of the same call (CPU): rank 0 sends first, rank 1 receives first."""
        if self.rank == 0:
            self.data.send(_to_wire(x))
            return _from_wire(self.data.recv()).to(x.device)
        theirs = _from_wire(self.data.recv()).to(x.device)
        self.data.send(_to_wire(x))
        return theirs

    def allreduce(self, rank: int, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        x = x.contiguous()
        if self.solo:
            return x if residual is None else residual + x
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            if self.ll is None:
                raise RuntimeError("no in-graph all-reduce on this platform")
            return self.ll._launch(rank, x, residual)
        if x.is_cuda and self.ll is not None:
            if residual is None or (residual.dtype == x.dtype and residual.shape == x.shape):
                return self.ll.eager(x, residual)
            return residual + self.ll.eager(x, None)
        theirs = self._swap(x)
        out = (x.float() + theirs.float()).to(x.dtype)
        return out if residual is None else residual + out

    def peer_rows(self, mine: torch.Tensor) -> torch.Tensor:
        """Rank 1's fp32 tensor on rank 0 (rank 0 passes zeros of its shape, rank 1 its own):
        an all-reduce with zeros on the GPU (``a + 0.0``, exact), a swap on the CPU."""
        mine = mine.float().contiguous()
        if mine.is_cuda and self.ll is not None:
            return self.ll.eager(mine, None)
        return self._swap(mine)

    def timeouts(self) -> int:
        return self.ll.timeouts() if self.ll is not None else 0


# ================================================================================ rank 1


def _gdn_globals() -> list[tuple[str, str, str | None]]:
    """How the parent bound the Gated DeltaNet functions the fast path reads from the
    Transformers module (``engine.gdn.install``), so the child binds the same."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    from engine import gdn

    out = []
    saved = gdn._ORIGINAL.get(hf.__name__, {})
    for g in gdn._GLOBALS:
        v = getattr(hf, g, None)
        if getattr(v, "__module__", "") == gdn.__name__:
            out.append((g, "gdn", v.__name__))
        elif g in saved and v is gdn._unwrapped(saved[g]) and v is not saved[g]:
            out.append((g, "unwrapped", None))
        else:
            out.append((g, "orig", None))
    return out


def _bind_gdn(spec: list[tuple[str, str, str | None]]) -> None:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    from engine import gdn

    for g, how, name in spec:
        if how == "gdn":
            setattr(hf, g, getattr(gdn, name))
        elif how == "unwrapped":
            setattr(hf, g, gdn._unwrapped(getattr(hf, g)))


def _row_bytes(model: FastModel) -> int:
    """Bytes of one slot's Gated DeltaNet state copy (what a snapshot keeps per rank)."""
    return sum(c[0].nbytes + r[0].nbytes for c, r in model.gdn_state.values())


class _Rank1:
    """The child's side: rank 1's ``FastModel`` and the commands that drive it."""

    def __init__(self, conn: Any, data: Any, init: dict[str, Any]) -> None:
        self.conn = conn
        devices = [torch.device(d) for d in init["devices"]]
        self.dev = devices[1]
        if self.dev.type == "cuda":
            torch.cuda.set_device(self.dev)
        _bind_gdn(init["gdn"])
        if init["attn_splits"] is not None:
            kernels.ATTN_SPLITS = init["attn_splits"]
        self.ll = None
        if self.dev.type == "cuda":
            self.ll = _LLProc(1, devices)
            self.ll.connect(init["handles"])
        self.group = ProcGroup(1, devices, self.ll, data)
        table = self._weights()
        shard, head = _fill(init["shard"], table), _fill(init["head"], table)
        del table
        model = FastModel(shard, mtp=head, **init["options"])
        model.tp, model.rank, model.vocab_offset = self.group, 1, shard.vocab_offset
        if len(model.segments) != 1:
            raise RuntimeError("a tensor-parallel rank must hold its whole shard on one device")
        model._ensure_pages = self._ensure_pages  # pages come from rank 0's commands
        self.model = model
        self.snaps: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.last: torch.Tensor | None = None
        self.ready = {
            "handles": self.ll.handles() if self.ll is not None else None,
            "pages": model.pool.pages,
            "vocab": model.vocab,
            "row_bytes": _row_bytes(model),
            "pid": os.getpid(),
            "fla_pins": model.fla_pins,
            "kernels": dict(model.triton_kernels),
        }

    def _weights(self) -> dict[int, torch.Tensor]:
        """Rank 1's weights, mapped from the parent's memory (``weights`` batches)."""
        table: dict[int, torch.Tensor] = {}
        while True:
            msg = self.conn.recv()
            if msg[0] == "weights_end":
                return table
            if msg[0] != "weights":
                raise RuntimeError(f"rank 1 expected weights, got {msg[0]!r}")
            for i, t in msg[2]:
                table[i] = t
            self.conn.send(("weights_done", None))

    def _ensure_pages(self, slot: int, length: int) -> None:
        if len(self.model.slot_pages[slot]) * PAGE < length:
            raise RuntimeError(f"rank 1: slot {slot} holds {len(self.model.slot_pages[slot])} pages, "
                               f"{length} tokens need more (rank 0 sends the page tables)")

    # ------------------------------------------------------------ commands

    def op_table(self, slot: int, pages: list[int]) -> None:
        m = self.model
        m.slot_pages[slot] = list(pages)
        m._push_table(slot)

    def op_reset(self, slot: int) -> None:
        m = self.model
        m.slot_pages[slot] = []
        m.bank[slot] = 0
        for conv, rec in m.gdn_state.values():
            conv[slot].zero_()
            rec[slot].zero_()

    def op_flip(self, slot: int) -> None:
        self.model.flip(slot)

    def op_copy_page(self, src: int, dst: int) -> None:
        for k_pool, v_pool in self.model.kv.values():
            k_pool[dst].copy_(k_pool[src])
            v_pool[dst].copy_(v_pool[src])

    def op_snap(self, sid: int, slot: int) -> None:
        row = self.model._g(slot)
        self.snaps[sid] = [(c[row].clone(), r[row].clone()) for c, r in self.model.gdn_state.values()]

    def op_gdn_load(self, sid: int, slot: int) -> None:
        row = self.model._g(slot)
        for (conv, rec), (c, r) in zip(self.model.gdn_state.values(), self.snaps[sid], strict=True):
            conv[row].copy_(c)
            rec[row].copy_(r)

    def op_gdn_release(self, sid: int) -> None:
        self.snaps.pop(sid, None)

    def op_solo(self, on: bool) -> None:
        self.group.solo = on

    def op_call(self, name: str, args: tuple) -> None:
        getattr(self.model, name)(*args)

    def op_prefill(self, slot: int, ids: list[int], start: int, follow: int | None) -> None:
        out = self.model.prefill(slot, ids, start, follow)
        self.group.peer_rows(out)

    def op_prefill_packed(self, segments: list, follows: list | None) -> None:
        out = self.model.prefill_packed(segments, follows)
        self.group.peer_rows(out)

    def op_inputs(self, host: Any) -> None:
        t = torch.from_numpy(host)
        for meta in self.model.devices.values():
            meta.packed.copy_(t)

    def op_warm(self, size: int, verify: bool) -> None:
        with torch.inference_mode():
            tpm.TPModel._warm(None, self.model, size, verify)  # type: ignore[arg-type]

    def op_capture(self, size: int, mode: str, verify: bool) -> None:
        self.conn.send(("captured", self.model._capture(size, mode, verify)))

    def op_clear_graphs(self) -> None:
        m = self.model
        for seg in m.segments + m.vsegments:
            seg.graphs.clear()
            seg.inputs.clear()
            seg.outputs.clear()
        m.__dict__.pop("_capture_streams", None)
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def op_replay(self, size: int, verify: bool) -> None:
        seg = (self.model.vsegments if verify else self.model.segments)[0]
        with torch.cuda.device(seg.device):
            seg.graphs[size].replay()
        self.last = seg.outputs[size]

    def op_eager_step(self, size: int, rows: int, verify: bool) -> None:
        seg = (self.model.vsegments if verify else self.model.segments)[0]
        meta = self.model.devices[seg.device]
        self.last = seg.run(meta.tokens[:rows], meta)

    def op_fetch(self, rows: int) -> None:
        self.group.peer_rows(self.last[:rows])

    def op_ping(self) -> None:
        self.conn.send(("pong", self.group.timeouts()))

    def serve(self) -> None:
        self.conn.send(("ready", self.ready))
        while True:
            msg = self.conn.recv()
            op, inference = msg[0], msg[1]
            if op == "stop":
                self.conn.send(("stopped", None))
                return
            handler = getattr(self, "op_" + op)
            with torch.inference_mode(inference):
                handler(*msg[2:])


def _watch_parent(ppid: int) -> None:
    while True:
        time.sleep(0.5)
        if os.getppid() != ppid:
            os._exit(0)


def child_main(conn: Any, data: Any, ppid: int) -> None:
    """Rank 1's process (``multiprocessing`` spawn target)."""
    threading.Thread(target=_watch_parent, args=(ppid,), daemon=True, name="tp-parent-watch").start()
    try:
        msg = conn.recv()
        if msg[0] != "init":
            raise RuntimeError(f"rank 1 expected init, got {msg[0]!r}")
        _Rank1(conn, data, msg[1]).serve()
    except (EOFError, ConnectionError, BrokenPipeError):
        os._exit(0)  # the parent is gone
    except BaseException:  # noqa: BLE001 - report it, then leave: the pair is out of step
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass
        sys.stderr.write("engine: tensor-parallel rank 1 failed:\n" + traceback.format_exc())
        sys.stderr.flush()
        os._exit(1)
    os._exit(0)


# ================================================================================ rank 0


@dataclass
class _ProcSnapshot(Snapshot):
    """A snapshot whose rank-1 Gated DeltaNet copies live in the child, under ``remote``."""

    remote: int = -1


class _PeerLogits:
    """A verify step's logits over the whole vocabulary, fetched from rank 1 when first used
    (most steps only need the picks): indexes and behaves as the tensor. Valid until the next
    step, which overwrites rank 1's outputs."""

    def __init__(self, owner: ProcTPModel, own: torch.Tensor, step: int) -> None:  # noqa: D107
        self._owner, self._own, self._step, self._full = owner, own, step, None

    def _tensor(self) -> torch.Tensor:
        if self._full is None:
            if self._owner._step != self._step:
                raise RuntimeError("verify logits read after the next step")
            self._full = self._owner._gather([self._own])
        return self._full

    def __getitem__(self, index: Any) -> torch.Tensor:
        return self._tensor()[index]

    def __len__(self) -> int:
        return self._own.shape[0]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tensor(), name)


def _stop(proc: Any, conn: Any) -> None:
    if proc is None or not proc.is_alive():
        return
    try:
        conn.send(("stop", False))
    except Exception:  # noqa: BLE001
        pass
    proc.join(10)
    if proc.is_alive():
        proc.kill()
        proc.join(5)


def _watch_child(model_ref: Any, proc: Any) -> None:
    multiprocessing.connection.wait([proc.sentinel])
    model = model_ref()
    if model is None or model._closing:
        return
    why = f"tensor-parallel rank 1 process {proc.pid} exited with code {proc.exitcode}"
    model._dead = why
    print(f"engine: {why}", file=sys.stderr, flush=True)
    if _exit_on_death():
        print("engine: exiting: this replica's all-reduces cannot complete without rank 1",
              file=sys.stderr, flush=True)
        time.sleep(1.0)
        os._exit(70)


class ProcTPModel(tpm.TPModel):
    """``TPModel`` with rank 1 in its own process (see the module doc). ``ranks`` holds rank 0
    alone; rank 1 is reached through ``_send``."""

    def __init__(
        self,
        lm: Any,
        *,
        slots: int,
        max_len: int,
        mtp: Any = None,
        devices: Sequence[torch.device] | None = None,
        kv_tokens: int | None = None,
        use_triton: bool | None = None,
        use_graphs: bool | None = None,
        buckets: Sequence[int] = (1, 2, 4, 8, 16, 32),
    ) -> None:  # noqa: D107
        if devices is None:
            devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
        self.devices_tp = list(devices)
        if len(self.devices_tp) != 2:
            raise ValueError("tensor parallelism runs over exactly two devices")
        self._closing = False
        self._dead: str | None = None
        self._proc = None
        self._io = threading.RLock()
        self._sid = 0
        self._step = 0
        self.config = (lm.model.language_model if hasattr(lm.model, "language_model") else lm.model).config
        cuda = self.devices_tp[0].type == "cuda"
        attn_splits = None
        if cuda:
            attn_splits = int(os.environ.get("ENGINE_TP_ATTN_SPLITS", "64"))
            kernels.ATTN_SPLITS = attn_splits
        # Start the child first: it imports while rank 0 builds.
        ctx = torch.multiprocessing.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        self._data, child_data = ctx.Pipe()
        self._proc = proc = ctx.Process(target=child_main, args=(child_conn, child_data, os.getpid()),
                                        daemon=True, name="tp-rank1")
        proc.start()
        child_conn.close()
        child_data.close()
        atexit.register(self.close)
        threading.Thread(target=_watch_child, args=(weakref.ref(self), proc), daemon=True,
                         name="tp-rank1-watch").start()
        shards = self._shard(lm, mtp)
        ll = _LLProc(0, self.devices_tp) if cuda else None
        self.group = ProcGroup(0, self.devices_tp, ll, self._data)
        shard0, head0 = shards[0]
        r0 = FastModel(shard0, slots=slots, max_len=max_len, kv_tokens=kv_tokens, use_triton=use_triton,
                       use_graphs=use_graphs, buckets=buckets, mtp=head0)
        r0.tp, r0.rank, r0.vocab_offset = self.group, 0, shard0.vocab_offset
        if len(r0.segments) != 1:
            raise RuntimeError("a tensor-parallel rank must hold its whole shard on one device")
        self.ranks = [r0]
        options = dict(slots=slots, max_len=max_len, use_triton=use_triton, use_graphs=use_graphs,
                       buckets=tuple(buckets),
                       kv_tokens=(r0.pool.pages - 1) * PAGE if kv_tokens is None else kv_tokens)
        leaves: list[torch.Tensor] = []
        index: dict[int, int] = {}
        skeleton = _strip(list(shards[1]), leaves, index)
        del index
        init = {
            "devices": [str(d) for d in self.devices_tp],
            "shard": skeleton[0],
            "head": skeleton[1],
            "options": options,
            "gdn": _gdn_globals(),
            "attn_splits": attn_splits,
            "handles": ll.handles() if ll is not None else None,
        }
        with self._io:
            self._conn.send(("init", init))
        del init, shards, skeleton
        moved = sum(t.untyped_storage().nbytes() for t in leaves)
        if leaves:
            self._send("weights", list(enumerate(leaves)))
            self._reply("weights_done", REPLY_S)
        self._keep = leaves  # rank 1's weights: the child maps them; they must outlive it
        self._send("weights_end")
        tpm._empty_cache(self.devices_tp)
        ready = self._reply("ready", REPLY_S)
        if ll is not None:
            ll.connect(ready["handles"])
        if ready["pages"] != r0.pool.pages:
            raise RuntimeError(f"ranks hold {r0.pool.pages} and {ready['pages']} KV pages")
        self._peer_vocab = ready["vocab"]
        self._row_bytes1 = ready["row_bytes"]
        self.vocab = r0.vocab + self._peer_vocab
        r0._peers = []
        push = r0._push_table

        def push_both(slot: int) -> None:  # every page-table change goes to rank 1 as well
            push(slot)
            self._send("table", slot, list(r0.slot_pages[slot]))

        r0._push_table = push_both
        r0._push_table(r0.dummy)
        self.last_drafts: torch.Tensor | None = None
        self.graph_error: str | None = None
        self.capture_mode: str | None = None
        if ready["fla_pins"] != r0.fla_pins:  # both ranks must run fla with the same configs
            raise RuntimeError(f"fla pins differ: rank 0 {r0.fla_pins}, rank 1 {ready['fla_pins']}")
        if ready["kernels"] != dict(r0.triton_kernels):  # and the same kernel paths
            raise RuntimeError(f"kernel paths differ: rank 0 {r0.triton_kernels}, rank 1 {ready['kernels']}")
        print(f"engine: tensor-parallel rank 1 in process {ready['pid']} (ENGINE_TP_PROC=1), "
              f"fla pins {ready['fla_pins']}, prefill_fused {r0.triton_kernels.get('prefill_fused')}, "
              f"{moved / 2**30:.1f} GiB of weights handed over", flush=True)

    def _shard(self, lm: Any, mtp: Any) -> list:
        """``shard_model`` with rank 1's copies in their own memory pool (see the module doc)."""
        dev1 = self.devices_tp[1]
        pool = None
        if dev1.type == "cuda" and hasattr(torch.cuda, "MemPool"):
            # A MemPool registers (and holds a reference) on the device current when it is made;
            # made on another device, each ``use_mem_pool`` exit on ``dev1`` drops the pool there
            # and the next entry trips the allocator's ``use_count > 0`` assert.
            with torch.cuda.device(dev1):
                pool = torch.cuda.MemPool()
        self._mem_pool = pool  # kept: the child maps memory of this pool
        if pool is None:
            return tpm.shard_model(lm, mtp, self.devices_tp)
        put = tpm._put

        def put_pooled(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
            if dev != dev1:
                return put(t, dev)
            with torch.cuda.use_mem_pool(pool, device=dev):
                out = torch.empty(tuple(t.shape), dtype=t.dtype, device=dev)
            out.copy_(t.detach())
            return out

        tpm._put = put_pooled
        try:
            return tpm.shard_model(lm, mtp, self.devices_tp)
        finally:
            tpm._put = put

    # ------------------------------------------------------------ the child

    def _raise_dead(self) -> None:
        raise RuntimeError(self._dead or "tensor-parallel rank 1 failed")

    def _take(self, msg: Any) -> Any:
        if msg[0] == "error":
            self._dead = f"tensor-parallel rank 1 failed:\n{msg[1]}"
            self._raise_dead()
        return msg

    def _send(self, op: str, *args: Any) -> None:
        """Queue one command for rank 1 (in program order; no answer)."""
        with self._io:
            if self._dead is not None:
                self._raise_dead()
            while self._conn.poll():
                self._take(self._conn.recv())
                raise RuntimeError("rank 1 sent an unexpected message")
            try:
                self._conn.send((op, torch.is_inference_mode_enabled(), *args))
            except (BrokenPipeError, ConnectionError, OSError) as exc:
                self._dead = self._dead or f"tensor-parallel rank 1 is gone ({exc})"
                self._raise_dead()

    def _reply(self, kind: str, timeout: float) -> Any:
        """Rank 1's next answer, which must be ``kind``; fails if it dies or times out."""
        deadline = time.monotonic() + timeout
        with self._io:
            while True:
                if self._conn.poll(0.2):
                    try:
                        msg = self._take(self._conn.recv())
                    except (EOFError, ConnectionError, OSError) as exc:
                        self._dead = self._dead or f"tensor-parallel rank 1 is gone ({exc})"
                        self._raise_dead()
                    if msg[0] != kind:
                        raise RuntimeError(f"rank 1 answered {msg[0]!r}, expected {kind!r}")
                    return msg[1]
                if self._dead is not None or (self._proc is not None and not self._proc.is_alive()):
                    self._dead = self._dead or f"tensor-parallel rank 1 exited with code {self._proc.exitcode}"
                    self._raise_dead()
                if time.monotonic() > deadline:
                    raise RuntimeError(f"rank 1 did not answer {kind!r} within {timeout:.0f} s")

    def close(self) -> None:
        """Stop rank 1's process (idempotent; also at exit)."""
        self._closing = True
        proc, self._proc = self._proc, None
        _stop(proc, self._conn)

    def sync(self) -> int:
        """Wait until rank 1 has run every command so far; its all-reduce timeouts."""
        self._send("ping")
        return self._reply("pong", REPLY_S)

    def timeouts(self) -> int:
        return self.group.timeouts()

    def _check_timeouts(self) -> None:
        """After a pass whose results the caller reads at once: no all-reduce gave up."""
        if self.group.timeouts():
            raise RuntimeError(f"{self.group.timeouts()} tensor-parallel all-reduce(s) timed out "
                               "waiting for rank 1")

    def _both(self, fn: Any) -> list[Any]:
        raise NotImplementedError("ENGINE_TP_PROC=1 runs rank 1 in its own process")

    def _with_peer(self, own: torch.Tensor) -> torch.Tensor:
        """Rank 0's logits rows with rank 1's shard of them (rank 1 sends its own side)."""
        peer = self.group.peer_rows(torch.zeros((own.shape[0], self._peer_vocab), dtype=torch.float32,
                                                device=own.device))
        return torch.cat([own, peer.to(own.dtype)], dim=-1)

    # ------------------------------------------------------------ slots, pages, snapshots

    def reset(self, slot: int) -> None:
        self.ranks[0].reset(slot)
        self._send("reset", slot)

    def flip(self, slot: int) -> None:
        self.ranks[0].flip(slot)
        self._send("flip", slot)

    def _copy_page(self, src: int) -> int:
        r0 = self.ranks[0]
        dst = r0.pool.alloc()
        for k_pool, v_pool in r0.kv.values():
            k_pool[dst].copy_(k_pool[src])
            v_pool[dst].copy_(v_pool[src])
        self._send("copy_page", src, dst)
        return dst

    def _copies(self, slot: int) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        r0 = self.ranks[0]
        row = r0._g(slot)
        gdn = [(c[row].clone(), r[row].clone()) for c, r in r0.gdn_state.values()]
        self._sid += 1
        self._send("snap", self._sid, slot)
        return gdn, self._sid

    def _load(self, slot: int, snap: Snapshot) -> None:
        r0 = self.ranks[0]
        row = r0._g(slot)
        for (conv, rec), (c, r) in zip(r0.gdn_state.values(), snap.gdn, strict=True):
            conv[row].copy_(c)
            rec[row].copy_(r)
        self._send("gdn_load", snap.remote, slot)

    def snapshot(self, slot: int, length: int) -> Snapshot:
        r0 = self.ranks[0]
        gdn, sid = self._copies(slot)
        used = math.ceil(length / PAGE)
        full = length // PAGE
        pages = []
        for index, page in enumerate(r0.slot_pages[slot][:used]):
            if index < full:
                r0.pool.ref(page)
                pages.append(page)
            else:
                pages.append(self._copy_page(page))
        return _ProcSnapshot(length=length, gdn=gdn, pages=pages, remote=sid)

    def restore(self, snap: Snapshot, slot: int) -> None:
        r0 = self.ranks[0]
        self._load(slot, snap)
        full = snap.length // PAGE
        held = list(snap.pages)
        for page in held:
            r0.pool.ref(page)
        pages = []
        for index, page in enumerate(held):
            if index < full:
                pages.append(page)
            else:
                pages.append(self._copy_page(page))
                r0.pool.unref(page)
        r0.slot_pages[slot] = pages
        r0._push_table(slot)

    def detach(self, slot: int, length: int) -> Snapshot:
        r0 = self.ranks[0]
        gdn, sid = self._copies(slot)
        used = math.ceil(length / PAGE)
        pages = r0.slot_pages[slot]
        for page in pages[used:]:
            r0.pool.unref(page)
        r0.slot_pages[slot] = []
        return _ProcSnapshot(length=length, gdn=gdn, pages=pages[:used], remote=sid)

    def adopt(self, snap: Snapshot, slot: int) -> None:
        if snap.released:
            raise RuntimeError("adopting a released snapshot")
        r0 = self.ranks[0]
        self._load(slot, snap)
        for page in r0.slot_pages[slot]:
            r0.pool.unref(page)
        r0.slot_pages[slot] = list(snap.pages)
        snap.pages = []
        snap.gdn = []
        snap.released = True
        self._send("gdn_release", snap.remote)
        r0._push_table(slot)

    def release(self, snap: Snapshot) -> None:
        if not snap.released:
            self._send("gdn_release", snap.remote)
        self.ranks[0].release(snap)

    def snapshot_bytes(self, snap: Snapshot) -> int:  # type: ignore[override]
        """Both ranks' state copies (rank 1's are the child's, the same size per row)."""
        return FastModel.snapshot_bytes(snap) + (self._row_bytes1 if snap.gdn else 0)

    # ------------------------------------------------------------ prefill

    @torch.inference_mode()
    def prefill(self, slot: int, ids: Sequence[int], start: int, follow: int | None = None) -> torch.Tensor:
        r0 = self.ranks[0]
        r0._ensure_pages(slot, start + len(ids))
        self._send("prefill", slot, list(ids), start, follow)
        with tpm._on(r0):
            own = r0.prefill(slot, ids, start, follow)
            out = self._with_peer(own)
        self.last_drafts = r0.last_drafts
        self._check_timeouts()
        return out

    @torch.inference_mode()
    def prefill_packed(
        self,
        segments: Sequence[tuple[int, Sequence[int], int]],
        follows: Sequence[int | None] | None = None,
    ) -> torch.Tensor:
        r0 = self.ranks[0]
        for slot, ids, start in segments:
            r0._ensure_pages(slot, start + len(ids))
        self._send("prefill_packed", [(s, list(ids), st) for s, ids, st in segments],
                   list(follows) if follows is not None else None)
        with tpm._on(r0):
            own = r0.prefill_packed(segments, follows)
            out = self._with_peer(own)
        self.last_drafts = r0.last_drafts
        self._check_timeouts()
        return out

    def _solo(self, name: str, *args: Any) -> None:
        """``rank.name(*args)`` on rank 0 here, then on rank 1 (``Group.solo``: no rendezvous)."""
        r0 = self.ranks[0]
        with tpm._on(r0):
            getattr(r0, name)(*args)
        self._send("call", name, args)

    @torch.inference_mode()
    def warmup(self, slot: int, chunk: int, token_ids: Sequence[int], other: int | None = None) -> None:
        """``TPModel.warmup``: each rank's kernels alone first (``Group.solo``), then together."""
        ids = list(token_ids)
        self._solo("_warm_kernels", chunk)
        self.group.solo = True
        self._send("solo", True)
        try:
            self._solo("prefill", slot, ids[:37], 0)
            self._solo("prefill", slot, ids[37:101], 37)
            self.reset(slot)
            self._solo("prefill", slot, ids[:chunk], 0)
            if other is not None and len(ids) >= 102:
                self.reset(slot)
                self._solo("prefill", slot, ids[:37], 0)
                self._solo("prefill_packed", [(slot, ids[37:101], 37), (other, ids[:64], 0)])
                end = min(len(ids), max(chunk, 70))
                self._solo("prefill_packed", [(slot, ids[101:102], 101), (other, ids[64:end], 64)])
        finally:
            self.group.solo = False
            self._send("solo", False)
            self.reset(slot)
            if other is not None:
                self.reset(other)
        try:
            self.prefill(slot, ids[:37], 0)
            self.prefill(slot, ids[37:101], 37)
            self.reset(slot)
            self.prefill(slot, ids[:chunk], 0)
            if other is not None:
                self.reset(slot)
                self.reset(other)
                if len(ids) < 102:
                    raise ValueError("warmup needs at least 102 token ids")
                self.prefill(slot, ids[:37], 0)
                self.prefill_packed([(slot, ids[37:101], 37), (other, ids[:64], 0)])
                end = min(len(ids), max(chunk, 70))
                self.prefill_packed([(slot, ids[101:102], 101), (other, ids[64:end], 64)])
        finally:
            self.reset(slot)
            if other is not None:
                self.reset(other)
        if self.devices_tp[0].type == "cuda":
            torch.cuda.synchronize(self.devices_tp[0])
        self.sync()

    # ------------------------------------------------------------ decode and verify

    def _load_inputs(self, host: torch.Tensor) -> None:
        self._host = host
        for meta in self.ranks[0].devices.values():
            meta.packed.copy_(host)
        self._send("inputs", host.numpy())

    def _run(self, size: int, rows: int, verify: bool) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Rank 0's output (its logits shard) and, for a verify step, the picks; rank 1 runs
        the same step in its process."""
        r0 = self.ranks[0]
        if r0.use_graphs and self.graph_error is None:
            segs = r0.vsegments if verify else r0.segments
            if size not in segs[0].graphs:
                self.capture(size, verify)
                self._load_inputs(self._host)  # the warm-up overwrote them, on both ranks
            if self.graph_error is None:
                self._send("replay", size, verify)
                self._step += 1
                with torch.cuda.device(segs[0].device):
                    segs[0].graphs[size].replay()
                return [segs[0].outputs[size]], (r0._verify_static[size] if verify else None)
        seg = (r0.vsegments if verify else r0.segments)[0]
        meta = r0.devices[seg.device]
        self._send("eager_step", size, rows, verify)
        self._step += 1
        with tpm._on(r0):
            out = seg.run(meta.tokens[:rows], meta)
        return [out], (r0._verify_out if verify else None)

    def _gather(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """Rank 0's rows (``parts[0]``) with rank 1's shard of the last step's outputs."""
        own = parts[0]
        self._send("fetch", own.shape[0])
        with tpm._on(self.ranks[0]):
            out = self._with_peer(own)
        return out

    @torch.inference_mode()
    def decode(self, tokens: Sequence[int], slots: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        out = super().decode(tokens, slots, positions)
        self._check_timeouts()
        return out

    def _verify_logits(self, outs: Sequence[torch.Tensor], batch: int) -> Any:
        return _PeerLogits(self, outs[0][:batch], self._step)

    @torch.inference_mode()
    def verify(self, tokens, drafts, slots, positions):  # noqa: ANN001, ANN201 - TPModel.verify's
        out = super().verify(tokens, drafts, slots, positions)
        self._check_timeouts()
        return out

    # ------------------------------------------------------------ graphs

    def capture(self, size: int, verify: bool = False) -> None:
        """Both ranks' graphs for bucket ``size`` (each process captures its own), or record
        why not (then steps run eagerly)."""
        modes = getattr(self, "_capture_modes", None) or os.environ.get(
            "ENGINE_CAPTURE_MODES", "global,thread_local,relaxed"
        ).split(",")
        r0 = self.ranks[0]
        errors = []
        for mode in modes:
            try:
                self._send("warm", size, verify)
                with torch.inference_mode():
                    self._warm(r0, size, verify)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"warm: {type(exc).__name__}: {exc}")
                break
            self._send("capture", size, mode, verify)
            results = [r0._capture(size, mode, verify), self._reply("captured", REPLY_S)]
            if all(r is None for r in results):
                self._capture_modes = [mode]
                self.capture_mode = mode
                return
            for seg in r0.segments + r0.vsegments:
                seg.graphs.clear()
                seg.inputs.clear()
                seg.outputs.clear()
            r0.__dict__.pop("_capture_streams", None)
            self._send("clear_graphs")
            for dev in self.devices_tp:
                torch.cuda.synchronize(dev)
            errors.append(f"{mode}: {[r for r in results if r]}")
            print(f"engine: tensor-parallel capture of bucket {size} (verify={verify}) failed in "
                  f"mode {mode}: {[r for r in results if r]}", flush=True)
        self.graph_error = " | ".join(errors)


__all__ = ["ProcTPModel", "ProcGroup", "child_main"]
