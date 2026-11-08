"""Tensor parallelism over the two GCDs of one replica (plan 13): ``ENGINE_TP=2``.

Without it a replica splits its 40 layers between its two GCDs and runs them one after the
other, so one GCD always idles. Here every layer is split in halves along heads and the
expert intermediate dimension, each GCD (rank) runs all 40 layers on its half, and the two
partial sums of each layer's mixer output and MoE output are added by an all-reduce. Nothing
changes in the math but the summation order of those two projections (a sum of two fp32
accumulated, BF16-rounded halves), and both ranks add in the same order, so their hidden
states are bitwise identical.

Splits (rank r of 2, ``shard_model``):
- Gated DeltaNet: key heads ``[8r, 8r+8)`` and value heads ``[16r, 16r+16)`` (value head h
  reads key head h // 2, so a rank's value heads read only its key heads): the q|k|v rows of
  ``in_proj_qkv`` and the conv channels, ``in_proj_z/b/a`` rows, ``A_log``, ``dt_bias``, the
  per-head recurrent and conv state; ``out_proj`` columns (partial sum). The gated norm is
  per head: local.
- Attention: query heads (with their gates) ``[8r, 8r+8)`` and KV head r (query head h reads
  KV head h // 8): ``q/k/v_proj`` rows, the KV pool, ``o_proj`` columns (partial sum).
- MoE: the router and the shared expert's sigmoid gate replicated (both ranks route the same
  hidden state identically), every routed expert's and the shared expert's intermediate
  dimension halved: gate/up rows, down columns (partial sum).
- Embedding, norms and the MTP head's ``fc`` replicated; ``lm_head`` split by vocabulary rows.
  Greedy picks combine each rank's (max, first argmax) with ties to rank 0, i.e. to the lowest
  index, as ``argmax`` over the whole vocabulary; full logits are gathered when returned.

The all-reduce (``Group.allreduce``):
- Inside a captured decode graph: one Triton kernel per call, the step-0 winner of plan 13
  (j86: 2.5 us for 8 KB in a graph, no stale read over 1,900 checked iterations). Each rank
  writes its partial into the peer's receive buffer as 8-byte words of 4 data bytes and the
  4-byte call epoch (NCCL's LL protocol), write-through (``.wt``), and spins on its own receive
  buffer with uncached volatile loads (``.cv``) until every word carries this epoch; buffers
  are fine-grained (``hipExtMallocWithFlags``) and double-buffered by epoch parity; 64 words
  per one-wavefront CTA, each CTA with its own epoch. The kernel keeps its epochs in device
  memory, so the graphs need no host input and the ranks rendezvous inside them.
- Eagerly (prefill, capture warm-up, the eager decode fallback): the ranks run in two host
  threads (``TPModel._both``) and exchange partials through a host mailbox and a
  stream-ordered peer copy. No device spin, so no host ordering can deadlock it.
"""

from __future__ import annotations

import concurrent.futures
import copy
import ctypes
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import torch
from torch.nn import functional as F

from engine.fast import kernels
from engine.fast.model import DECODE_EXPERT_TOKENS, PAGE, FastModel, Snapshot, _PackPlan, _rms

triton, tl = kernels.triton, kernels.tl

CHUNK = 64  # 32-bit words per CTA of the in-graph all-reduce
# Plan 33 (ENGINE_PREFILL_GRAPH, default on; 0 for eager prefill): single-segment passes of up to the
# largest bucket replay one captured graph per rank and size (``TPModel.capture_prefill``),
# their all-reduces in the graph too, so the all-reduce buffer must hold a pass's hidden state.
PREFILL_GRAPH = os.environ.get("ENGINE_PREFILL_GRAPH", "1") == "1"
# Padded pass sizes (rows); a pass of n tokens (DECODE_EXPERT_TOKENS < n <= the largest, 2,048
# = the server's prefill chunk) runs the smallest size >= n; longer or packed passes run eagerly.
PG_BUCKETS = tuple(sorted(int(b) for b in os.environ.get(
    "ENGINE_PREFILL_GRAPH_BUCKETS",
    "64,80,96,112,128,160,192,224,256,288,320,384,448,512,576,640,704,768,896,1024,1152,1280,"
    "1408,1536,1792,2048",
).split(",")))
# Words per parity buffer: 64 rows x 2048 BF16 (a verify step of bucket 32); with prefill graphs
# 2^22 (4,096 rows), which costs 2 x (2 x 2^22 x 8 B) = 128 MiB of fine-grained memory per GCD
# (receive + loopback buffers) and 256 KiB of epochs, against 2 MiB at 2^16.
CAPACITY = 1 << (22 if PREFILL_GRAPH else 16)
MAXIT = 4_000_000  # spin iterations before a call gives up (counted in ``timeouts``)
# Python's thread switch interval while both ranks run eagerly (prefill): the threads hand the
# GIL to each other at every all-reduce, so a short interval only bounds the odd stall.
SWITCH_S = float(os.environ.get("ENGINE_TP_SWITCH_S", "0.0002"))
# ENGINE_TP_PREFILL: "lockstep" runs prefill passes from one host thread, both ranks stage by
# stage; "threads" runs each rank's own pass on its own thread, meeting at every all-reduce.
LOCKSTEP = os.environ.get("ENGINE_TP_PREFILL", "threads") == "lockstep"
# Seconds an eager all-reduce waits for the peer thread before failing the pass.
WAIT_S = float(os.environ.get("ENGINE_TP_WAIT_S", "300"))
# Plan 29 item 1 (ENGINE_GLUE_ARRMS=1, default off): in a captured graph, the all-reduce of a
# partial sum, the residual add and the following RMSNorm in one kernel (``_ll_ar_rms_kernel``,
# one program per row), meant to be bitwise ``add_rms(res, allreduce(x), w)``. Off: in j141 the
# 80 fused calls took ~1,020 us per GCD per verify B=2 step against ~820 us for the all-reduce
# and add_rms launches they replace, and tests/test_fused_glue.py did not pass for it.
GLUE_ARRMS = os.environ.get("ENGINE_GLUE_ARRMS", "0") == "1"
# Plan 29 item 4 (ENGINE_GLUE_ARGMAX=1, default on): ``Group.argmax`` as three launches: one
# Triton pass per row over this rank's logits shard (its max and first index, packed with the
# zeros of the peer's columns), the fp32 all-reduce, one Triton pick. The same bits as the torch
# ops (cast, argmax, gather, zeros, 2 stores, add, all-reduce, where, long): a max and the first
# index of it do not depend on the order of the scan.
GLUE_ARGMAX = os.environ.get("ENGINE_GLUE_ARGMAX", "1") == "1"
ARGMAX_BLOCK = 8192
RMS_ROWS = 64  # rows per call of the fused all-reduce + RMSNorm
RMS_WIDTH = 2048  # BF16 values per row (the hidden size)


def size() -> int:
    """``ENGINE_TP``: 2 (default) for tensor parallelism over a replica's two GPUs, 1 for the
    layer split (one GPU's layers after the other's)."""
    return int(os.environ.get("ENGINE_TP", "2"))


if triton is not None:

    @triton.jit(do_not_specialize=["NW"])
    def _ll_allreduce_kernel(
        x_ptr, res_ptr, out_ptr, recv_local, recv_peer, epoch_ptr, err_ptr, NW, CAP,
        CHUNK: tl.constexpr, BF16: tl.constexpr, RES: tl.constexpr, MAXIT: tl.constexpr,
    ):
        """out = [res +] (x_self + x_peer) over 32-bit words of BF16 pairs or fp32 values,
        rounded as ``bf16(fp32(a) + fp32(b))`` (then ``bf16(res + that)``); see the module doc."""
        p = tl.program_id(0)
        e = tl.load(epoch_ptr + p) + 1
        buf = (e % 2).to(tl.int64) * CAP
        offs = p * CHUNK + tl.arange(0, CHUNK)
        m = offs < NW
        xi = tl.load(x_ptr + offs, mask=m, other=0)
        word = (e.to(tl.int64) << 32) | (xi.to(tl.int64) & 0xFFFFFFFF)
        tl.store(recv_peer + buf + offs, word, mask=m, cache_modifier=".wt")
        src = recv_local + buf + offs
        w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
        pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
        it = pending * 0
        while (pending > 0) & (it < MAXIT):
            w = tl.load(src, mask=m, other=0, volatile=True, cache_modifier=".cv")
            pending = tl.sum((((w >> 32).to(tl.int32) != e) & m).to(tl.int32), axis=0)
            it += 1
        if pending > 0:
            tl.atomic_add(err_ptr, 1)
        yi = (w & 0xFFFFFFFF).to(tl.int32)
        if BF16:
            s_lo = (((xi & 0xFFFF) << 16).to(tl.float32, bitcast=True)
                    + ((yi & 0xFFFF) << 16).to(tl.float32, bitcast=True))
            s_hi = ((xi & -65536).to(tl.float32, bitcast=True)
                    + (yi & -65536).to(tl.float32, bitcast=True))
            s_lo = s_lo.to(tl.bfloat16).to(tl.float32)
            s_hi = s_hi.to(tl.bfloat16).to(tl.float32)
            if RES:
                ri = tl.load(res_ptr + offs, mask=m, other=0)
                s_lo = (((ri & 0xFFFF) << 16).to(tl.float32, bitcast=True) + s_lo).to(tl.bfloat16).to(tl.float32)
                s_hi = ((ri & -65536).to(tl.float32, bitcast=True) + s_hi).to(tl.bfloat16).to(tl.float32)
            lo = s_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            hi = s_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
            res = lo | hi
        else:
            s = xi.to(tl.float32, bitcast=True) + yi.to(tl.float32, bitcast=True)
            if RES:
                s = tl.load(res_ptr + offs, mask=m, other=0).to(tl.float32, bitcast=True) + s
            res = s.to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, res, mask=m)
        tl.store(epoch_ptr + p, e)


    @triton.jit(do_not_specialize=["eps"])
    def _ll_ar_rms_kernel(
        x_ptr, res_ptr, w_ptr, sum_ptr, out_ptr, recv_local, recv_peer, epoch_ptr, err_ptr,
        eps, CAP, N: tl.constexpr, BLOCK: tl.constexpr, MAXIT: tl.constexpr,
    ):
        """One row per program: s = bf16(fp32(x_self) + fp32(x_peer)) over the LL exchange
        (one 8-byte word per BF16 value: its bits and the row's epoch), then exactly
        ``kernels._add_rms_kernel`` with ``h = s``: sum = bf16(res + s), out = RMSNorm(sum)."""
        row = tl.program_id(0)
        e = tl.load(epoch_ptr + row) + 1
        buf = (e % 2).to(tl.int64) * CAP
        cols = tl.arange(0, BLOCK)
        ok = cols < N
        x = tl.load(res_ptr + row * N + cols, mask=ok, other=0.0).to(tl.float32)
        mine = tl.load(x_ptr + row * N + cols, mask=ok, other=0.0)
        bits = mine.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        word = (e.to(tl.int64) << 32) | bits.to(tl.int64)
        offs = row * N + cols
        tl.store(recv_peer + buf + offs, word, mask=ok, cache_modifier=".wt")
        src = recv_local + buf + offs
        w = tl.load(src, mask=ok, other=0, volatile=True, cache_modifier=".cv")
        pending = tl.sum((((w >> 32).to(tl.int32) != e) & ok).to(tl.int32), axis=0)
        it = pending * 0
        while (pending > 0) & (it < MAXIT):
            w = tl.load(src, mask=ok, other=0, volatile=True, cache_modifier=".cv")
            pending = tl.sum((((w >> 32).to(tl.int32) != e) & ok).to(tl.int32), axis=0)
            it += 1
        if pending > 0:
            tl.atomic_add(err_ptr, 1)
        theirs = ((w & 0xFFFF).to(tl.int32) << 16).to(tl.float32, bitcast=True)
        h = (mine.to(tl.float32) + theirs).to(tl.bfloat16).to(tl.float32)
        x = (x + h).to(tl.bfloat16).to(tl.float32)
        tl.store(sum_ptr + row * N + cols, x.to(tl.bfloat16), mask=ok)
        var = tl.sum(x * x, axis=0) / N
        wt = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
        y = x * tl.rsqrt(var + eps) * (1.0 + wt)
        tl.store(out_ptr + row * N + cols, y.to(tl.bfloat16), mask=ok)
        tl.store(epoch_ptr + row, e)


    @triton.jit
    def _argmax_pack_kernel(l_ptr, both_ptr, V, stride, offset, RANK: tl.constexpr,
                            BLOCK: tl.constexpr):
        """both[row] = this rank's (max, first index + offset) in columns 2 RANK, 2 RANK + 1 of
        four, zeros in the peer's two (``Group.argmax``)."""
        row = tl.program_id(0)
        base = l_ptr + row.to(tl.int64) * stride
        lanes = tl.arange(0, BLOCK)
        best = tl.full([BLOCK], float("-inf"), tl.float32)
        bidx = tl.zeros([BLOCK], dtype=tl.int32)
        for v0 in range(0, V, BLOCK):
            cols = v0 + lanes
            x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
            upd = x > best  # strict: each lane keeps the first column of its max
            best = tl.where(upd, x, best)
            bidx = tl.where(upd, cols, bidx)
        m = tl.max(best, axis=0)
        i = tl.min(tl.where(best == m, bidx, V), axis=0)
        i = tl.where(m == float("-inf"), 0, i)  # an all -inf row: index 0, as torch.argmax
        k = tl.arange(0, 4)
        val = tl.where(k == 2 * RANK, m, tl.where(k == 2 * RANK + 1, (i + offset).to(tl.float32), 0.0))
        tl.store(both_ptr + row * 4 + k, val)

    @triton.jit
    def _argmax_pick_kernel(both_ptr, out_ptr, R, RB: tl.constexpr):
        """out[row] = int64(both[row, 1] if both[row, 0] >= both[row, 2] else both[row, 3])."""
        r = tl.arange(0, RB)
        ok = r < R
        a0 = tl.load(both_ptr + r * 4, mask=ok, other=0.0)
        a1 = tl.load(both_ptr + r * 4 + 1, mask=ok, other=0.0)
        a2 = tl.load(both_ptr + r * 4 + 2, mask=ok, other=0.0)
        a3 = tl.load(both_ptr + r * 4 + 3, mask=ok, other=0.0)
        tl.store(out_ptr + r, tl.where(a0 >= a2, a1, a3).to(tl.int64), mask=ok)


def argmax_pack(logits: torch.Tensor, rank: int, offset: int) -> torch.Tensor:
    """The ``[rows, 4]`` fp32 block ``Group.argmax`` all-reduces, in one launch."""
    logits = logits.contiguous()
    rows, v = logits.shape
    both = torch.empty((rows, 4), dtype=torch.float32, device=logits.device)
    with kernels.on_device(logits):
        _argmax_pack_kernel[(rows,)](logits, both, v, logits.stride(0), offset, RANK=rank,
                                     BLOCK=ARGMAX_BLOCK, num_warps=8)
    return both


def argmax_pick(both: torch.Tensor) -> torch.Tensor:
    rows = both.shape[0]
    out = torch.empty((rows,), dtype=torch.long, device=both.device)
    with kernels.on_device(both):
        _argmax_pick_kernel[(1,)](both, out, rows, RB=max(16, triton.next_power_of_2(rows)),
                                  num_warps=1)
    return out


def _hip() -> ctypes.CDLL:
    path = Path(torch.__file__).parent / "lib" / "libamdhip64.so"
    return ctypes.CDLL(str(path) if path.exists() else "libamdhip64.so")


class _Raw:
    """A device pointer Triton takes as a tensor argument (``data_ptr`` and ``dtype``)."""

    def __init__(self, ptr: int, dtype: torch.dtype) -> None:
        self.ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.ptr


class _LL:
    """State of the in-graph all-reduce between two GPUs (see the module doc)."""

    def __init__(self, devices: Sequence[torch.device]) -> None:
        self.hip = hip = _hip()
        self.devices = list(devices)
        for a, b in ((0, 1), (1, 0)):
            if hip.hipSetDevice(self.devices[a].index) != 0:
                raise RuntimeError("hipSetDevice failed")
            rc = hip.hipDeviceEnablePeerAccess(self.devices[b].index, 0)
            if rc not in (0, 704):  # 704: hipErrorPeerAccessAlreadyEnabled
                raise RuntimeError(f"hipDeviceEnablePeerAccess({a}->{b}) failed: {rc}")
        torch.cuda.set_device(self.devices[0])
        self.recv = [self._fine(d, 2 * CAPACITY * 8) for d in self.devices]
        self.loop = [self._fine(d, 2 * CAPACITY * 8) for d in self.devices]  # loopback warm-up
        ctas = CAPACITY // CHUNK
        self.epoch = [torch.zeros(ctas, dtype=torch.int32, device=d) for d in self.devices]
        self.loop_epoch = [torch.zeros(ctas, dtype=torch.int32, device=d) for d in self.devices]
        self.err = [torch.zeros(1, dtype=torch.int32, device=d) for d in self.devices]
        self.rms_cap = RMS_ROWS * RMS_WIDTH
        self.rms_ready = False
        if GLUE_ARRMS:
            self.rms_recv = [self._fine(d, 2 * self.rms_cap * 8) for d in self.devices]
            self.rms_loop = [self._fine(d, 2 * self.rms_cap * 8) for d in self.devices]
            self.rms_epoch = [torch.zeros(RMS_ROWS, dtype=torch.int32, device=d) for d in self.devices]
            self.rms_loop_epoch = [torch.zeros(RMS_ROWS, dtype=torch.int32, device=d)
                                   for d in self.devices]
            for rank, dev in enumerate(self.devices):
                x = torch.zeros((1, RMS_WIDTH), dtype=torch.bfloat16, device=dev)
                self._launch_rms(rank, x, x, x[0], 1e-6, loop=True)
                torch.cuda.synchronize(dev)
            self.rms_ready = True
        # Compile and load every variant on both GPUs before any capture: each rank against
        # itself (its own buffers as the "peer"), so no rendezvous is needed.
        for rank, dev in enumerate(self.devices):
            for dtype, residual in ((torch.bfloat16, False), (torch.bfloat16, True), (torch.float32, False)):
                x = torch.zeros((1, 256), dtype=dtype, device=dev)
                self._launch(rank, x, x if residual else None, loop=True)
            torch.cuda.synchronize(dev)

    def _fine(self, dev: torch.device, nbytes: int) -> _Raw:
        ptr = ctypes.c_void_p()
        hip = self.hip
        if hip.hipSetDevice(dev.index) != 0:
            raise RuntimeError("hipSetDevice failed")
        # hipDeviceMallocFinegrained = 1: peer writes are visible to the owner's loads.
        rc = hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1))
        if rc != 0:
            raise RuntimeError(f"hipExtMallocWithFlags failed: {rc}")
        if hip.hipMemset(ptr, 0, ctypes.c_size_t(nbytes)) != 0 or hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("hipMemset failed")
        torch.cuda.set_device(self.devices[0])
        return _Raw(ptr.value, torch.int64)

    def _launch(self, rank: int, x: torch.Tensor, residual: torch.Tensor | None,
                loop: bool = False) -> torch.Tensor:
        x = x.contiguous()
        nbytes = x.numel() * x.element_size()
        nw = nbytes // 4
        if nbytes % 4 or nw > CAPACITY or x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"all-reduce of {tuple(x.shape)} {x.dtype} is not supported in a graph")
        out = torch.empty_like(x)
        res = residual.contiguous() if residual is not None else x
        if residual is not None and (res.dtype != x.dtype or res.shape != x.shape):
            raise ValueError("the residual must match the all-reduced tensor")
        local = (self.loop if loop else self.recv)[rank]
        peer = local if loop else self.recv[1 - rank]
        epoch = (self.loop_epoch if loop else self.epoch)[rank]
        with torch.cuda.device(x.device):
            _ll_allreduce_kernel[(triton.cdiv(nw, CHUNK),)](
                x.view(-1).view(torch.int32), res.view(-1).view(torch.int32),
                out.view(-1).view(torch.int32), local, peer, epoch, self.err[rank], nw, CAPACITY,
                CHUNK=CHUNK, BF16=x.dtype == torch.bfloat16, RES=residual is not None,
                MAXIT=MAXIT, num_warps=1,
            )
        return out

    def rms_fits(self, x: torch.Tensor) -> bool:
        return (self.rms_ready and x.dim() == 2 and x.dtype == torch.bfloat16
                and x.shape[1] == RMS_WIDTH and x.shape[0] <= RMS_ROWS)

    def _launch_rms(self, rank: int, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                    eps: float, loop: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """(bf16(res + allreduce(x)), its RMSNorm times 1 + weight): ``_ll_ar_rms_kernel``."""
        x = x.contiguous()
        residual = residual.contiguous()
        rows, n = x.shape
        if n != RMS_WIDTH or rows > RMS_ROWS or x.dtype != torch.bfloat16 or residual.shape != x.shape:
            raise ValueError(f"fused all-reduce + RMSNorm of {tuple(x.shape)} {x.dtype} is not supported")
        total = torch.empty_like(x)
        out = torch.empty_like(x)
        local = (self.rms_loop if loop else self.rms_recv)[rank]
        peer = local if loop else self.rms_recv[1 - rank]
        epoch = (self.rms_loop_epoch if loop else self.rms_epoch)[rank]
        with torch.cuda.device(x.device):
            _ll_ar_rms_kernel[(rows,)](
                x, residual, weight, total, out, local, peer, epoch, self.err[rank], eps,
                self.rms_cap, N=n, BLOCK=triton.next_power_of_2(n), MAXIT=MAXIT,
            )
        return total, out

    def timeouts(self) -> int:
        return sum(int(e.item()) for e in self.err)


class Group:
    """Two ranks' all-reduce and greedy pick, in a captured graph (``_LL``) or eagerly."""

    def __init__(self, devices: Sequence[torch.device]) -> None:
        self.devices = list(devices)
        self._cv = threading.Condition()
        self._box: dict[tuple[int, int], tuple[torch.Tensor, Any]] = {}
        self._seq = [0, 0]
        self._broken: str | None = None
        # Rank 0 alone (``TPModel.warmup``): the all-reduces return the local partial, so one
        # rank can run a prefill by itself (wrong numbers, right kernels: autotuning once).
        self.solo = False
        self.ll: _LL | None = None
        if all(d.type == "cuda" for d in self.devices) and triton is not None:
            self.ll = _LL(self.devices)

    # ------------------------------------------------------------ eager rendezvous

    def start(self) -> None:
        """Begin a pair of eager rank programs (``TPModel._both``)."""
        with self._cv:
            self._box.clear()
            self._seq = [0, 0]
            self._broken = None

    def abort(self, why: str) -> None:
        with self._cv:
            self._broken = why
            self._cv.notify_all()

    def _exchange(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        """The peer's ``x`` of the same call, on this rank's device, ordered on its stream."""
        event = None
        if x.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(x.device))
        with self._cv:
            seq = self._seq[rank]
            self._seq[rank] += 1
            self._box[(seq, rank)] = (x, event)
            self._cv.notify_all()
            # The peer got here first and waits on us: after this hand it the GIL at once, or it
            # would wait for Python's switch interval (5 ms by default) at every all-reduce.
            handoff = (seq, 1 - rank) in self._box
            while (seq, 1 - rank) not in self._box:
                if self._broken is not None:
                    raise RuntimeError(f"tensor-parallel peer failed: {self._broken}")
                if not self._cv.wait(timeout=WAIT_S):
                    raise RuntimeError("tensor-parallel peer did not reach the all-reduce")
            theirs, their_event = self._box.pop((seq, 1 - rank))
        if handoff:
            time.sleep(0)
        if not theirs.is_cuda or theirs.device == x.device:
            return theirs
        # The copy runs on the source device's current stream (torch's cross-device copy),
        # after the event that marks the peer's partial ready; the peer's allocator must not
        # reuse that memory before the copy ran.
        src_stream = torch.cuda.current_stream(theirs.device)
        if their_event is not None:
            src_stream.wait_event(their_event)
        theirs.record_stream(src_stream)
        return theirs.to(x.device)

    # ------------------------------------------------------------ collectives

    def allreduce(self, rank: int, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """``[residual +] (x_rank0 + x_rank1)``, the sum in fp32 rounded to ``x``'s dtype; the
        same bits on both ranks."""
        x = x.contiguous()
        if self.solo:
            return x if residual is None else residual + x
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            if self.ll is None:
                raise RuntimeError("no in-graph all-reduce on this platform")
            return self.ll._launch(rank, x, residual)
        theirs = self._exchange(rank, x)
        out = (x.float() + theirs.float()).to(x.dtype)
        return out if residual is None else residual + out

    def allreduce_rms(self, rank: int, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """``kernels.add_rms(residual, allreduce(x), weight, eps)``: in a captured graph one
        kernel (``GLUE_ARRMS``), bitwise the two launches it replaces."""
        x = x.contiguous()
        if (not self.solo and x.is_cuda and torch.cuda.is_current_stream_capturing()
                and self.ll is not None and self.ll.rms_fits(x)):
            return self.ll._launch_rms(rank, x, residual, weight, eps)
        return kernels.add_rms(residual, self.allreduce(rank, x), weight, eps)

    def argmax(self, rank: int, logits: torch.Tensor, offset: int) -> torch.Tensor:
        """The greedy pick per row over both ranks' vocabulary shards (``logits`` is this
        rank's, its first row ``offset``): the larger max, ties to rank 0 (the lower index)."""
        if (GLUE_ARGMAX and logits.is_cuda and triton is not None and logits.dim() == 2
                and logits.shape[0] > 0 and logits.dtype in (torch.float32, torch.bfloat16)):
            return argmax_pick(self.allreduce(rank, argmax_pack(logits, rank, offset)))
        logits = logits.float()
        index = logits.argmax(-1)
        best = logits.gather(-1, index[:, None])[:, 0]
        both = torch.zeros((logits.shape[0], 4), dtype=torch.float32, device=logits.device)
        both[:, 2 * rank] = best
        both[:, 2 * rank + 1] = (index + offset).float()
        both = self.allreduce(rank, both)
        return torch.where(both[:, 0] >= both[:, 2], both[:, 1], both[:, 3]).long()

    def timeouts(self) -> int:
        return self.ll.timeouts() if self.ll is not None else 0


# ================================================================================ sharding


def _free(module: Any) -> None:
    """Drop a loaded module's weights (keeping each parameter's dtype and device)."""
    if module is None:
        return
    for p in list(module.parameters()) + list(module.buffers()):
        if p.numel():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)


def _put(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """A contiguous copy of ``t`` on ``dev`` (never a view of a weight about to be freed)."""
    return t.detach().to(dev, copy=True).contiguous()


def _norm(m: Any, dev: torch.device) -> NS:
    out = NS(weight=_put(m.weight, dev))
    for name in ("eps", "variance_epsilon"):
        if hasattr(m, name):
            setattr(out, name, getattr(m, name))
    return out


class _Experts:
    """A rank's routed experts: the tensors the kernels read, and the registered prefill
    implementation (``engine.experts``) as a fallback when the grouped kernel cannot run."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        self.gate_up_proj = gate_up
        self.down_proj = down
        self.num_experts = gate_up.shape[0]
        self.act_fn = F.silu

    def __call__(self, x: torch.Tensor, top_i: torch.Tensor, top_w: torch.Tensor) -> torch.Tensor:
        from engine import experts

        return experts._padded_experts_forward(self, x, top_i, top_w)


def _gdn_shard(m: Any, rank: int, ranks: int, dev: torch.device) -> NS:
    nk, nv, dk, dv = m.num_k_heads, m.num_v_heads, m.head_k_dim, m.head_v_dim
    if nk % ranks or nv % ranks:
        raise ValueError(f"Gated DeltaNet heads {nk}/{nv} do not split over {ranks} ranks")
    key_dim, value_dim = nk * dk, nv * dv
    kd, vd = key_dim // ranks, value_dim // ranks
    nk2, nv2 = nk // ranks, nv // ranks
    dev0 = m.in_proj_qkv.weight.device
    channels = torch.cat([
        torch.arange(rank * kd, (rank + 1) * kd),
        torch.arange(key_dim + rank * kd, key_dim + (rank + 1) * kd),
        torch.arange(2 * key_dim + rank * vd, 2 * key_dim + (rank + 1) * vd),
    ]).to(dev0)
    heads = slice(rank * nv2, (rank + 1) * nv2)
    return NS(
        in_proj_qkv=NS(weight=_put(m.in_proj_qkv.weight.index_select(0, channels), dev)),
        in_proj_z=NS(weight=_put(m.in_proj_z.weight[rank * vd : (rank + 1) * vd], dev)),
        in_proj_b=NS(weight=_put(m.in_proj_b.weight[heads], dev)),
        in_proj_a=NS(weight=_put(m.in_proj_a.weight[heads], dev)),
        conv1d=NS(weight=_put(m.conv1d.weight.index_select(0, channels), dev)),
        out_proj=NS(weight=_put(m.out_proj.weight[:, rank * vd : (rank + 1) * vd], dev)),
        A_log=_put(m.A_log[heads], dev),
        dt_bias=_put(m.dt_bias[heads], dev),
        norm=_norm(m.norm, dev),
        num_k_heads=nk2, num_v_heads=nv2, head_k_dim=dk, head_v_dim=dv,
        key_dim=kd, value_dim=vd, conv_dim=2 * kd + vd, conv_kernel_size=m.conv_kernel_size,
    )


def _attn_shard(m: Any, q_heads: int, kv_heads: int, rank: int, ranks: int, dev: torch.device) -> NS:
    if q_heads % ranks or kv_heads % ranks:
        raise ValueError(f"attention heads {q_heads}/{kv_heads} do not split over {ranks} ranks")
    d = m.head_dim
    q2, kv2 = q_heads // ranks, kv_heads // ranks
    return NS(
        q_proj=NS(weight=_put(m.q_proj.weight[rank * q2 * 2 * d : (rank + 1) * q2 * 2 * d], dev)),
        k_proj=NS(weight=_put(m.k_proj.weight[rank * kv2 * d : (rank + 1) * kv2 * d], dev)),
        v_proj=NS(weight=_put(m.v_proj.weight[rank * kv2 * d : (rank + 1) * kv2 * d], dev)),
        o_proj=NS(weight=_put(m.o_proj.weight[:, rank * q2 * d : (rank + 1) * q2 * d], dev)),
        q_norm=_norm(m.q_norm, dev),
        k_norm=_norm(m.k_norm, dev),
        scaling=m.scaling,
        head_dim=d,
    )


def _moe_shard(mlp: Any, rank: int, ranks: int, dev: torch.device) -> NS:
    ex, sh = mlp.experts, mlp.shared_expert
    gu, dn = ex.gate_up_proj, ex.down_proj
    inter = gu.shape[1] // 2
    shared = sh.gate_proj.weight.shape[0]
    if inter % ranks or shared % ranks:
        raise ValueError(f"expert intermediate {inter}/{shared} does not split over {ranks} ranks")
    i2, s2 = inter // ranks, shared // ranks
    lo, hi = rank * i2, (rank + 1) * i2
    gate_up = _put(torch.cat([gu[:, lo:hi], gu[:, inter + lo : inter + hi]], dim=1), dev)
    down = _put(dn[:, :, lo:hi], dev)
    return NS(
        gate=NS(weight=_put(mlp.gate.weight, dev)),
        shared_expert=NS(
            gate_proj=NS(weight=_put(sh.gate_proj.weight[rank * s2 : (rank + 1) * s2], dev)),
            up_proj=NS(weight=_put(sh.up_proj.weight[rank * s2 : (rank + 1) * s2], dev)),
            down_proj=NS(weight=_put(sh.down_proj.weight[:, rank * s2 : (rank + 1) * s2], dev)),
        ),
        shared_expert_gate=NS(weight=_put(mlp.shared_expert_gate.weight, dev)),
        experts=_Experts(gate_up, down),
    )


def _layer_shard(layer: Any, kind: str, cfg: Any, rank: int, ranks: int, dev: torch.device) -> NS:
    out = NS(
        input_layernorm=_norm(layer.input_layernorm, dev),
        post_attention_layernorm=_norm(layer.post_attention_layernorm, dev),
        mlp=_moe_shard(layer.mlp, rank, ranks, dev),
    )
    if kind == "linear_attention":
        out.linear_attn = _gdn_shard(layer.linear_attn, rank, ranks, dev)
    else:
        out.self_attn = _attn_shard(
            layer.self_attn, cfg.num_attention_heads, cfg.num_key_value_heads, rank, ranks, dev
        )
    return out


def shard_model(lm: Any, mtp: Any, devices: Sequence[torch.device]) -> list[tuple[NS, NS | None]]:
    """Per rank, a stand-in for the loaded model (what ``FastModel`` reads) holding that rank's
    shard on its device, and the MTP head's shard. The loaded weights are freed layer by
    layer as their shards are made, so peak memory stays near the loaded model's."""
    ranks = len(devices)
    text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
    cfg = text.config
    rank_cfg = copy.copy(cfg)
    rank_cfg.num_attention_heads = cfg.num_attention_heads // ranks
    rank_cfg.num_key_value_heads = cfg.num_key_value_heads // ranks
    vocab = lm.lm_head.weight.shape[0]
    bounds = [math.ceil(vocab * r / ranks) for r in range(ranks + 1)]
    layers: list[list[NS]] = [[] for _ in devices]
    for i, layer in enumerate(text.layers):
        for r, dev in enumerate(devices):
            layers[r].append(_layer_shard(layer, cfg.layer_types[i], cfg, r, ranks, dev))
        _free(layer)
        _empty_cache(devices)
    heads: list[NS | None] = [None] * ranks
    if mtp is not None:
        for r, dev in enumerate(devices):
            heads[r] = NS(
                fc=NS(weight=_put(mtp.fc.weight, dev)),
                pre_fc_norm_embedding=_norm(mtp.pre_fc_norm_embedding, dev),
                pre_fc_norm_hidden=_norm(mtp.pre_fc_norm_hidden, dev),
                layers=[_layer_shard(mtp.layers[0], "full_attention", cfg, r, ranks, dev)],
                norm=_norm(mtp.norm, dev),
            )
        _free(mtp)
    out = []
    for r, dev in enumerate(devices):
        out.append(NS(
            model=NS(language_model=NS(
                config=rank_cfg,
                embed_tokens=NS(weight=_put(text.embed_tokens.weight, dev)),
                norm=NS(weight=_put(text.norm.weight, dev)),
                layers=layers[r],
                rotary_emb=NS(inv_freq=text.rotary_emb.inv_freq.detach().clone(),
                              attention_scaling=text.rotary_emb.attention_scaling),
            )),
            lm_head=NS(weight=_put(lm.lm_head.weight[bounds[r] : bounds[r + 1]], dev)),
            vocab_offset=bounds[r],
        ))
    for t in (text.embed_tokens.weight, text.norm.weight, lm.lm_head.weight):
        t.data = torch.empty(0, dtype=t.dtype, device=t.device)
    _empty_cache(devices)
    return [(o, h) for o, h in zip(out, heads, strict=True)]


def _on(model: FastModel):
    """Make ``model``'s GPU current (Triton launches on the current device)."""
    return kernels.on_device(model.embed)


def _empty_cache(devices: Sequence[torch.device]) -> None:
    if any(d.type == "cuda" for d in devices):
        torch.cuda.empty_cache()


# ================================================================================ the model


class TPModel:
    """``FastModel``'s interface over two ranks, each a ``FastModel`` of its shard.

    The ranks share the host state: the KV page pool and page lists (page ids are the same on
    both, each rank holding its KV head's half of every page), the GDN state banks, and the
    page tables (pushed to both). Decode and verify steps replay one full-depth graph per rank;
    prefill and every eager pass run the two ranks in two host threads.
    """

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
    ) -> None:
        if devices is None:
            devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
        self.devices_tp = list(devices)
        if len(self.devices_tp) != 2:
            raise ValueError("tensor parallelism runs over exactly two devices")
        self.config = (lm.model.language_model if hasattr(lm.model, "language_model") else lm.model).config
        if self.devices_tp[0].type == "cuda":
            # The 1-KV-head shard has half the attention CTAs per split: use more splits.
            kernels.ATTN_SPLITS = int(os.environ.get("ENGINE_TP_ATTN_SPLITS", "64"))
        shards = shard_model(lm, mtp, self.devices_tp)
        self.group = Group(self.devices_tp)
        self.ranks: list[FastModel] = []
        for r, (shard, head) in enumerate(shards):
            if r == 1 and kv_tokens is None:
                kv_tokens = (self.ranks[0].pool.pages - 1) * PAGE  # the same page count
            model = FastModel(
                shard, slots=slots, max_len=max_len, kv_tokens=kv_tokens, use_triton=use_triton,
                use_graphs=use_graphs, buckets=buckets, mtp=head,
            )
            model.tp, model.rank, model.vocab_offset = self.group, r, shard.vocab_offset
            self.ranks.append(model)
        r0, r1 = self.ranks
        if r1.pool.pages != r0.pool.pages:
            raise RuntimeError(f"ranks hold {r0.pool.pages} and {r1.pool.pages} KV pages")
        if len(r0.segments) != 1 or len(r1.segments) != 1:
            raise RuntimeError("a tensor-parallel rank must hold its whole shard on one device")
        r1.pool, r1.slot_pages, r1.bank, r1._lock = r0.pool, r0.slot_pages, r0.bank, r0._lock
        r0._peers, r1._peers = [r1], [r0]
        r0._push_table(r0.dummy)
        self.vocab = sum(r.vocab for r in self.ranks)
        self.last_drafts: torch.Tensor | None = None
        self.graph_error: str | None = None
        self.capture_mode: str | None = None
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tp-rank1")
        # Captured prefill passes (plan 33): size -> (graphs, input buffers, logits, drafts) per
        # rank; filled at load by ``capture_prefill``, never while serving.
        self.pg: dict[int, tuple[list, list[torch.Tensor], list[torch.Tensor], list]] = {}
        self.pg_error: str | None = None
        self.pg_replays = 0

    # ------------------------------------------------------------ shared attributes

    def __getattr__(self, name: str) -> Any:
        # Host-side state and configuration are rank 0's (shared by the ranks).
        if name in ("ranks",):
            raise AttributeError(name)
        return getattr(self.ranks[0], name)

    @property
    def tp(self) -> Group:
        return self.group

    def timeouts(self) -> int:
        """In-graph all-reduce calls that gave up waiting for the peer (0 when healthy)."""
        return self.group.timeouts()

    # ------------------------------------------------------------ running both ranks

    def _both(self, fn: Callable[[FastModel], Any]) -> list[Any]:
        """``fn(rank)`` for both ranks at once: rank 1 on the worker thread, rank 0 here."""
        group = self.group
        group.start()
        r0, r1 = self.ranks

        def run1() -> Any:
            if r1.embed.device.type == "cuda":
                torch.cuda.set_device(r1.embed.device)
            try:
                with torch.inference_mode():
                    return fn(r1)
            except BaseException as exc:
                group.abort(f"rank 1: {type(exc).__name__}: {exc}")
                raise

        # Both threads launch kernels between rendezvous: switch between them quickly.
        interval = sys.getswitchinterval()
        sys.setswitchinterval(SWITCH_S)
        try:
            return self._both_run(fn, run1)
        finally:
            sys.setswitchinterval(interval)

    def _both_run(self, fn: Callable[[FastModel], Any], run1: Callable[[], Any]) -> list[Any]:
        group, r0 = self.group, self.ranks[0]
        future = self._pool.submit(run1)
        try:
            if r0.embed.device.type == "cuda":
                torch.cuda.set_device(r0.embed.device)
            with torch.inference_mode():
                out0 = fn(r0)
        except BaseException as exc:
            group.abort(f"rank 0: {type(exc).__name__}: {exc}")
            try:
                future.result()
            except BaseException:  # noqa: BLE001 - rank 0's error is the one to report
                pass
            raise
        return [out0, future.result()]

    def _on_rank1(self, fn: Callable[[], Any]) -> Any:
        """``fn()`` on the worker thread with rank 1's GPU current."""
        with _on(self.ranks[1]):
            return fn()

    def _gather(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concatenate the ranks' vocabulary shards of fp32 logits on rank 0's device."""
        dev = parts[0].device
        return torch.cat([p.to(dev) for p in parts], dim=-1)

    # ------------------------------------------------------------ slots, pages, snapshots

    def reset(self, slot: int) -> None:
        for model in self.ranks:
            model.reset(slot)

    def flip(self, slot: int) -> None:
        self.ranks[0].flip(slot)  # the banks are shared

    def _copy_page(self, src: int) -> int:
        dst = self.ranks[0].pool.alloc()
        for model in self.ranks:
            for k_pool, v_pool in model.kv.values():
                k_pool[dst].copy_(k_pool[src])
                v_pool[dst].copy_(v_pool[src])
        return dst

    def _gdn_copies(self, slot: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        row = self.ranks[0]._g(slot)
        return [(c[row].clone(), r[row].clone()) for m in self.ranks for c, r in m.gdn_state.values()]

    def _gdn_load(self, slot: int, gdn: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        row = self.ranks[0]._g(slot)
        states = [pair for m in self.ranks for pair in m.gdn_state.values()]
        for (conv, rec), (c, r) in zip(states, gdn, strict=True):
            conv[row].copy_(c)
            rec[row].copy_(r)

    def snapshot(self, slot: int, length: int) -> Snapshot:
        r0 = self.ranks[0]
        gdn = self._gdn_copies(slot)
        used = math.ceil(length / PAGE)
        full = length // PAGE
        pages = []
        for index, page in enumerate(r0.slot_pages[slot][:used]):
            if index < full:
                r0.pool.ref(page)
                pages.append(page)
            else:
                pages.append(self._copy_page(page))
        return Snapshot(length=length, gdn=gdn, pages=pages)

    def restore(self, snap: Snapshot, slot: int) -> None:
        r0 = self.ranks[0]
        self._gdn_load(slot, snap.gdn)
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
        gdn = self._gdn_copies(slot)
        used = math.ceil(length / PAGE)
        pages = r0.slot_pages[slot]
        for page in pages[used:]:
            r0.pool.unref(page)
        r0.slot_pages[slot] = []
        return Snapshot(length=length, gdn=gdn, pages=pages[:used])

    def adopt(self, snap: Snapshot, slot: int) -> None:
        if snap.released:
            raise RuntimeError("adopting a released snapshot")
        r0 = self.ranks[0]
        self._gdn_load(slot, snap.gdn)
        for page in r0.slot_pages[slot]:
            r0.pool.unref(page)
        r0.slot_pages[slot] = list(snap.pages)
        snap.pages = []
        snap.gdn = []
        snap.released = True
        r0._push_table(slot)

    def release(self, snap: Snapshot) -> None:
        self.ranks[0].release(snap)

    snapshot_bytes = staticmethod(FastModel.snapshot_bytes)

    # ------------------------------------------------------------ prefill

    @torch.inference_mode()
    def prefill(self, slot: int, ids: Sequence[int], start: int, follow: int | None = None) -> torch.Tensor:
        self.ranks[0]._ensure_pages(slot, start + len(ids))
        size = self._pg_size(len(ids))
        if size is not None:
            return self._pg_run(size, slot, ids, start, follow)
        if LOCKSTEP:
            return self._lockstep([(slot, ids, start)], [follow])[0]
        parts = self._both(lambda m: m.prefill(slot, ids, start, follow))
        self.last_drafts = self.ranks[0].last_drafts
        return self._gather(parts)

    @torch.inference_mode()
    def prefill_packed(
        self,
        segments: Sequence[tuple[int, Sequence[int], int]],
        follows: Sequence[int | None] | None = None,
    ) -> torch.Tensor:
        for slot, ids, start in segments:
            self.ranks[0]._ensure_pages(slot, start + len(ids))
        if len(segments) == 1:
            size = self._pg_size(len(segments[0][1]))
            if size is not None:
                slot, ids, start = segments[0]
                follow = follows[0] if follows is not None else None
                return self._pg_run(size, slot, ids, start, follow)[None]
        if LOCKSTEP:
            if len({slot for slot, _, _ in segments}) != len(segments):
                raise ValueError("a packed prefill holds at most one segment per slot")
            if any(len(ids) == 0 for _, ids, _ in segments):
                raise ValueError("an empty prefill segment")
            return self._lockstep(segments, list(follows) if follows is not None else [None] * len(segments))
        parts = self._both(lambda m: m.prefill_packed(segments, follows))
        self.last_drafts = self.ranks[0].last_drafts
        return self._gather(parts)

    # ------------------------------------------------------------ captured prefill

    def _pg_size(self, count: int) -> int | None:
        """The captured size a single-segment pass of ``count`` tokens replays, or None."""
        if not self.pg or count <= DECODE_EXPERT_TOKENS:
            return None
        for size in PG_BUCKETS:
            if size >= count:
                return size if size in self.pg else None
        return None

    def _pg_run(self, size: int, slot: int, ids: Sequence[int], start: int,
                follow: int | None) -> torch.Tensor:
        """One prefill pass by replaying the size's graphs: the inputs in one copy per rank."""
        r0 = self.ranks[0]
        graphs, bufs, outs, drafts = self.pg[size]
        host = FastModel.pg_host(size, ids, start, follow, slot, r0._g(slot))
        for buf in bufs:
            buf.copy_(host)
        for g, buf in zip(graphs, bufs, strict=True):
            with torch.cuda.device(buf.device):
                g.replay()
        self.pg_replays += 1
        self.last_drafts = drafts[0].clone() if drafts[0] is not None else None
        return self._gather(outs)

    def capture_prefill(self) -> None:
        """Capture one prefill graph per rank for every size of ``PG_BUCKETS`` (largest first,
        sharing one memory pool per GPU; one pass replays at a time). Each size is first run
        eagerly twice through the same code on the capture streams (compiling the kernels);
        those passes write the dummy slot's state and page only. Any failure leaves prefill
        eager (``pg_error``)."""
        r0, r1 = self.ranks
        if not PREFILL_GRAPH or self.group.ll is None or not all(m.pg_supported() for m in self.ranks):
            return
        dummy = r0.dummy
        scratch = r0.slot_pages[dummy][0]
        pools = {m.embed.device: torch.cuda.graph_pool_handle() for m in self.ranks}
        streams = {m.embed.device: torch.cuda.Stream(m.embed.device) for m in self.ranks}
        modes = getattr(self, "_capture_modes", None) or ["global", "thread_local", "relaxed"]
        t0 = time.perf_counter()
        try:
            # The warm passes' page table: every entry the dummy page.
            for m in self.ranks:
                for meta in m.devices.values():
                    meta.page_table[dummy].fill_(scratch)
            for size in sorted(PG_BUCKETS, reverse=True):
                count = max(DECODE_EXPERT_TOKENS + 1, size - 3)
                host = FastModel.pg_host(size, [(7 * j) % 1000 + 100 for j in range(count)], 0, None,
                                         dummy, r0._g(dummy))
                bufs = [m._pg_inputs(size) for m in self.ranks]
                for buf in bufs:
                    buf.copy_(host)

                def warm(m: FastModel, size: int = size, bufs: list = bufs) -> None:
                    dev = m.embed.device
                    stream = streams[dev]
                    stream.wait_stream(torch.cuda.current_stream(dev))
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            m._pg_body(bufs[m.rank], size)
                    torch.cuda.current_stream(dev).wait_stream(stream)
                    torch.cuda.synchronize(dev)

                self._both(warm)

                def cap(m: FastModel, mode: str, size: int = size, bufs: list = bufs):
                    dev = m.embed.device
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.device(dev), torch.inference_mode():
                        with torch.cuda.graph(graph, pool=pools[dev], stream=streams[dev],
                                              capture_error_mode=mode):
                            logits, drafts = m._pg_body(bufs[m.rank], size)
                    return graph, logits, drafts

                errors = []
                for mode in modes:
                    try:
                        got = [cap(r0, mode)]
                        got.append(self._pool.submit(self._on_rank1, lambda: cap(r1, mode)).result())
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{mode}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")
                        for dev in self.devices_tp:
                            try:
                                torch.cuda.synchronize(dev)
                            except Exception:  # noqa: BLE001
                                pass
                        continue
                    self.pg[size] = ([g for g, _, _ in got], bufs, [lg for _, lg, _ in got],
                                     [d for _, _, d in got])
                    modes = [mode]
                    break
                else:
                    raise RuntimeError(f"size {size}: {' | '.join(errors)}")
        except Exception as exc:  # noqa: BLE001 - prefill stays eager
            self.pg.clear()
            self.pg_error = f"{type(exc).__name__}: {exc}"
            print(f"engine: prefill graph capture failed, prefill stays eager: {self.pg_error}", flush=True)
        finally:
            r0._push_table(dummy)
            for m in self.ranks:
                for conv, rec in m.gdn_state.values():
                    for row in (dummy, dummy + m.gdn_rows):
                        if row < conv.shape[0]:
                            conv[row].zero_()
                            rec[row].zero_()
            for dev in self.devices_tp:
                torch.cuda.synchronize(dev)
        if self.pg:
            print(f"engine: prefill graphs for {len(self.pg)} sizes {min(self.pg)}..{max(self.pg)} "
                  f"in {time.perf_counter() - t0:.1f}s; reserved "
                  f"{[round(torch.cuda.memory_reserved(d) / 2**30, 2) for d in self.devices_tp]} GiB",
                  flush=True)

    # ------------------------------------------------------------ lock-step prefill

    def _pair(self, parts: list[torch.Tensor], residuals: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """The all-reduce from one thread: per rank ``[residual +] (own + peer)``, the sum in
        fp32 in the same order as ``Group.allreduce`` (so the same bits on both ranks)."""
        out = []
        for r, part in enumerate(parts):
            theirs = parts[1 - r].to(part.device)
            total = (part.float() + theirs.float()).to(part.dtype)
            out.append(total if residuals is None else residuals[r] + total)
        return out

    def _argmax_all(self, logits: list[torch.Tensor]) -> torch.Tensor:
        """Greedy picks over both ranks' vocabulary shards (ties to the lowest index), on rank
        0's device."""
        return self._gather([lg.float() for lg in logits]).argmax(-1)

    def _lockstep(self, segments: Sequence[tuple[int, Sequence[int], int]],
                  follows: Sequence[int | None]) -> torch.Tensor:
        """``FastModel.prefill_packed``'s pass over both ranks from this one thread, stage by
        stage (every launch is asynchronous, so the two GPUs still run side by side): no host
        threads to hand over, the cross-rank sums as stream-ordered peer copies."""
        ranks = self.ranks
        plans = [_PackPlan(m, segments) for m in ranks]
        ons = [plan.on(m.embed.device) for plan, m in zip(plans, ranks, strict=True)]
        xs = []
        for m, plan in zip(ranks, plans, strict=True):
            with _on(m):
                xs.append(F.embedding(plan.tokens.to(m.embed.device), m.embed))
        for i in range(len(ranks[0].layers)):
            gdn = ranks[0].layer_types[i] == "linear_attention"
            hs = []
            for m, x, on in zip(ranks, xs, ons, strict=True):
                with _on(m):
                    h = _rms(x, m.layers[i].input_layernorm.weight, m.eps)
                    hs.append(m._gdn_prefill_packed(i, h, on) if gdn else m._attn_prefill_packed(i, h, on))
            xs = self._pair(hs, xs)
            hs = []
            for m, x in zip(ranks, xs, strict=True):
                with _on(m):
                    h = _rms(x, m.layers[i].post_attention_layernorm.weight, m.eps)
                    hs.append(m._moe(m.layers[i].mlp, h))
            xs = self._pair(hs, xs)
        logits = []
        hidden = []
        for m, x, on in zip(ranks, xs, ons, strict=True):
            with _on(m):
                last = _rms(x.index_select(0, on.last), m.final_norm, m.eps)
                logits.append(F.linear(last, m.lm_head).float())
                if m.mtp is not None:
                    hidden.append(_rms(x, m.final_norm, m.eps))
        out = self._gather(logits)
        if ranks[0].mtp is not None:
            self.last_drafts = self._mtp_lockstep(hidden, segments, follows, out, ons)
        return out

    def _mtp_lockstep(self, hidden: list[torch.Tensor], segments, follows, logits: torch.Tensor,
                      ons: list) -> torch.Tensor:
        """``FastModel._mtp_prefill`` over both ranks, stage by stage; the drafts (rank 0)."""
        ranks = self.ranks
        nxt: list[int] = []
        picked: list[tuple[int, int]] = []
        for k, ((_, ids, _), follow) in enumerate(zip(segments, follows, strict=True)):
            nxt.extend(ids[1:])
            if follow is None:
                picked.append((len(nxt), k))
            nxt.append(0 if follow is None else follow)
        tokens = torch.tensor(nxt, dtype=torch.long, device=logits.device)
        if picked:
            rows = torch.tensor([r for r, _ in picked], dtype=torch.long, device=logits.device)
            segs = torch.tensor([k for _, k in picked], dtype=torch.long, device=logits.device)
            tokens.index_copy_(0, rows, logits.index_select(0, segs).argmax(-1))
        zs, hs = [], []
        for m, h, on in zip(ranks, hidden, ons, strict=True):
            with _on(m):
                z = m._mtp_in(h, tokens.to(h.device))
                layer = m.mtp.layers[0]
                zs.append(z)
                hs.append(m._attn_prefill_packed(m.mtp_index, _rms(z, layer.input_layernorm.weight, m.eps), on))
        zs = self._pair(hs, zs)
        hs = []
        for m, z in zip(ranks, zs, strict=True):
            with _on(m):
                layer = m.mtp.layers[0]
                hs.append(m._moe(layer.mlp, _rms(z, layer.post_attention_layernorm.weight, m.eps)))
        zs = self._pair(hs, zs)
        parts = []
        for m, z, on in zip(ranks, zs, ons, strict=True):
            with _on(m):
                parts.append(m._mtp_logits(z.index_select(0, on.last)))
        return self._argmax_all(parts)

    @torch.inference_mode()
    def warmup(self, slot: int, chunk: int, token_ids: Sequence[int], other: int | None = None) -> None:
        """``FastModel.warmup`` with the prefills and resets driven from here (pages are
        allocated and released once for both ranks).

        First rank 0 runs the prefills alone (``Group.solo``): autotuned kernels (fla's chunked
        delta rule) tune once, from one thread, and their choice, keyed by shapes, serves both
        ranks; per-size kernels compile one rank at a time. Then both ranks run them together.
        """
        ids = list(token_ids)
        for model in self.ranks:
            with _on(model):
                model._warm_kernels(chunk)
        r0 = self.ranks[0]
        self.group.solo = True
        try:
            with _on(r0):
                r0.prefill(slot, ids[:37], 0)
                r0.prefill(slot, ids[37:101], 37)
                self.reset(slot)
                r0.prefill(slot, ids[:chunk], 0)
                if other is not None and len(ids) >= 102:
                    self.reset(slot)
                    r0.prefill(slot, ids[:37], 0)
                    r0.prefill_packed([(slot, ids[37:101], 37), (other, ids[:64], 0)])
                    end = min(len(ids), max(chunk, 70))
                    r0.prefill_packed([(slot, ids[101:102], 101), (other, ids[64:end], 64)])
        finally:
            self.group.solo = False
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
        for dev in self.devices_tp:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

    # ------------------------------------------------------------ decode and verify

    def _load_inputs(self, host: torch.Tensor) -> None:
        for model in self.ranks:
            for meta in model.devices.values():
                meta.packed.copy_(host)

    def _run(self, size: int, rows: int, verify: bool) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Both ranks' outputs (their logits shards) and, for a verify step, the picks."""
        r0 = self.ranks[0]
        if r0.use_graphs and self.graph_error is None:
            segs = [m.vsegments if verify else m.segments for m in self.ranks]
            if size not in segs[0][0].graphs:
                self.capture(size, verify)
            if self.graph_error is None:
                outs = []
                for model, seg in zip(self.ranks, segs, strict=True):
                    with torch.cuda.device(seg[0].device):
                        seg[0].graphs[size].replay()
                    outs.append(seg[0].outputs[size])
                return outs, (r0._verify_static[size] if verify else None)

        def eager(model: FastModel) -> tuple[torch.Tensor, torch.Tensor | None]:
            seg = (model.vsegments if verify else model.segments)[0]
            meta = model.devices[seg.device]
            out = seg.run(meta.tokens[:rows], meta)
            return out, (model._verify_out if verify else None)

        results = self._both(eager)
        return [o for o, _ in results], results[0][1]

    @torch.inference_mode()
    def decode(self, tokens: Sequence[int], slots: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        r0 = self.ranks[0]
        batch = len(tokens)
        for slot, position in zip(slots, positions, strict=True):
            r0._ensure_pages(slot, position + 1)
        size = r0.bucket(batch) if r0.use_graphs else batch
        host = torch.zeros((4, r0._rows), dtype=torch.long)
        host[1].fill_(r0.dummy)
        host[3].fill_(r0.dummy)
        host[0, :batch] = torch.tensor(list(tokens), dtype=torch.long)
        host[1, :batch] = torch.tensor(list(slots), dtype=torch.long)
        host[2, :batch] = torch.tensor(list(positions), dtype=torch.long)
        host[3, :batch] = torch.tensor([r0._g(s) for s in slots], dtype=torch.long)
        self._load_inputs(host)
        outs, _ = self._run(size, size, verify=False)
        return self._gather([o[:batch] for o in outs])

    @torch.inference_mode()
    def verify(
        self,
        tokens: Sequence[int],
        drafts: Sequence[int],
        slots: Sequence[int],
        positions: Sequence[int],
    ) -> tuple[torch.Tensor, list[int], list[int], list[int], list[int]]:
        """``FastModel.verify`` over both ranks (same inputs, outputs and contract)."""
        r0 = self.ranks[0]
        if r0.mtp is None:
            raise RuntimeError("verify needs the MTP head")
        batch = len(tokens)
        for slot, position in zip(slots, positions, strict=True):
            r0._ensure_pages(slot, position + 2)
        size = r0.bucket(batch) if r0.use_graphs else batch
        rows = 2 * size
        host = torch.zeros((4, r0._rows), dtype=torch.long)
        host[1].fill_(r0.dummy)
        host[3].fill_(r0.dummy)
        host[3, size:rows] = r0.dummy + r0.gdn_rows
        cur = [r0._g(s) for s in slots]
        other = [s + (1 - r0.bank[s]) * r0.gdn_rows for s in slots]
        for half, toks, pos, states in (
            (0, tokens, positions, cur),
            (size, drafts, [p + 1 for p in positions], other),
        ):
            host[0, half : half + batch] = torch.tensor(list(toks), dtype=torch.long)
            host[1, half : half + batch] = torch.tensor(list(slots), dtype=torch.long)
            host[2, half : half + batch] = torch.tensor(list(pos), dtype=torch.long)
            host[3, half : half + batch] = torch.tensor(states, dtype=torch.long)
        self._load_inputs(host)
        outs, out = self._run(size, rows, verify=True)
        logits = self._gather([o[:batch] for o in outs])
        picks = out.cpu().tolist()
        g, d = picks[0], picks[1]
        return logits, g[:batch], g[size : size + batch], d[:batch], d[size : size + batch]

    # ------------------------------------------------------------ graphs

    def _warm(self, model: FastModel, size: int, verify: bool) -> None:
        """What ``FastModel._capture`` runs before capturing (both ranks together)."""
        rows = 2 * size if verify else size
        for meta in model.devices.values():
            meta.slots[:rows].fill_(model.dummy)
            meta.positions[:rows].zero_()
            meta.gdn[:rows].fill_(model.dummy)
            if verify:
                meta.gdn[size:rows].fill_(model.dummy + model.gdn_rows)
        seg = (model.vsegments if verify else model.segments)[0]
        meta = model.devices[seg.device]
        streams = model.__dict__.setdefault("_capture_streams", {})
        if seg.device not in streams:
            streams[seg.device] = torch.cuda.Stream(seg.device)
        stream = streams[seg.device]
        stream.wait_stream(torch.cuda.current_stream(seg.device))
        with torch.cuda.stream(stream):
            for _ in range(2):  # compile the Triton kernels, warm the BLAS handles
                seg.run(meta.tokens[:rows], meta)
        torch.cuda.current_stream(seg.device).wait_stream(stream)
        torch.cuda.synchronize(seg.device)

    def capture(self, size: int, verify: bool = False) -> None:
        """Both ranks' graphs for bucket ``size``, or record why not (then steps run eagerly)."""
        modes = getattr(self, "_capture_modes", None) or os.environ.get(
            "ENGINE_CAPTURE_MODES", "global,thread_local,relaxed"
        ).split(",")
        errors = []
        for mode in modes:
            try:
                self._both(lambda m: self._warm(m, size, verify))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"warm: {type(exc).__name__}: {exc}")
                break
            # Each rank captures on the thread that warmed it (BLAS handles are per thread), one
            # after the other (a capture must not overlap another thread's HIP calls).
            r1 = self.ranks[1]
            results = [self.ranks[0]._capture(size, mode, verify)]
            results.append(self._pool.submit(self._on_rank1, lambda: r1._capture(size, mode, verify)).result())
            if all(r is None for r in results):
                self._capture_modes = [mode]
                self.capture_mode = mode
                return
            # A failed capture clears that rank's graphs of every bucket: clear both ranks' (the
            # next steps recapture what they need), and drop the capture streams, which a
            # broken capture may leave unusable.
            for m in self.ranks:
                for seg in m.segments + m.vsegments:
                    seg.graphs.clear()
                    seg.inputs.clear()
                    seg.outputs.clear()
                m.__dict__.pop("_capture_streams", None)
            for dev in self.devices_tp:
                torch.cuda.synchronize(dev)
            errors.append(f"{mode}: {[r for r in results if r]}")
            print(f"engine: tensor-parallel capture of bucket {size} (verify={verify}) failed in "
                  f"mode {mode}: {[r for r in results if r]}", flush=True)
        self.graph_error = " | ".join(errors)

    def capture_all(self, decode: bool = True, verify: bool = True) -> None:
        r0 = self.ranks[0]
        if not r0.use_graphs:
            return
        kinds = [k for k, on in ((True, verify and r0.mtp is not None), (False, decode)) if on]
        for is_verify in kinds:
            for size in sorted(r0.buckets, reverse=True):
                segs = r0.vsegments if is_verify else r0.segments
                if self.graph_error is None and size not in segs[0].graphs:
                    self.capture(size, is_verify)
        if not self.pg and self.pg_error is None:
            self.capture_prefill()
