"""Move a session's cached state between replicas (plan 49): export and import of a snapshot.

A session tail (``FastPrefixCache.store_tail``) is everything a replica holds for a session
between rounds: the KV pages of its ten attention layers and the MTP head's layer, and one
copy of each Gated DeltaNet layer's convolution and recurrent state. Under tensor parallelism
each rank holds its own half of every page (its KV head) and of every GDN state (its heads).

``export_into`` packs a snapshot into one flat byte buffer per rank on that rank's device
(``plan`` gives the layout: every KV pool's pages in the snapshot's page order, then every
GDN state, each part 256-byte aligned). ``import_from`` allocates pages from the importing
replica's pool, scatters the buffer into them and into fresh GDN state copies, and returns a
``Snapshot`` the prefix cache takes as a session tail (``Generator.import_tail``), so the
session's next round adopts it exactly as it would have on the source replica. Both sides
must run the same model with the same tensor-parallel split: ``import_from`` checks the
layout.

Transports between processes (each replica is its own process with only its own GPUs
visible): ``HostChannel``, a shared memory file both processes map and register as pinned
host memory (device to host on the source, host to device on the target), and
``IpcExport`` / ``IpcImport``, a device buffer on the source whose HIP IPC handle the target
opens and copies from directly over the GPUs' links.

Serving (plan 63) uses the host transport through ``Mover``: the router gives every replica
one ``HostChannel`` per replica (the memfds it passes down), a replica writes only its own and
reads the others'. A live move runs on the engine thread at step boundaries and never blocks
it on a copy: the pack and the scatter are queued on the replica's own streams, the copies
through host memory run on side streams, and the engine polls their events between steps.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from engine.fast.model import PAGE, Snapshot

ALIGN = 256


def ranks(fast: Any) -> list[Any]:
    """The per-device models of ``fast``: a ``TPModel``'s ranks, or a ``FastModel`` itself."""
    return list(fast.__dict__.get("ranks") or [fast])


@dataclass(frozen=True)
class Part:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Manifest:
    """The layout of one exported snapshot: ``length`` tokens in ``pages`` pages."""

    length: int
    pages: int
    parts: tuple[tuple[Part, ...], ...]  # per rank
    rank_bytes: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return sum(self.rank_bytes)


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def plan(fast: Any, length: int) -> Manifest:
    """Where each piece of a ``length``-token snapshot sits in each rank's buffer."""
    pages = math.ceil(length / PAGE)
    all_parts, sizes = [], []
    for model in ranks(fast):
        parts, offset = [], 0

        def add(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
            nonlocal offset
            nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            parts.append(Part(name, shape, dtype, offset, nbytes))
            offset = _align(offset + nbytes)

        for i, (k_pool, v_pool) in model.kv.items():
            add(f"kv{i}.k", (pages, *k_pool.shape[1:]), k_pool.dtype)
            add(f"kv{i}.v", (pages, *v_pool.shape[1:]), v_pool.dtype)
        for i, (conv, rec) in model.gdn_state.items():
            add(f"gdn{i}.conv", tuple(conv.shape[1:]), conv.dtype)
            add(f"gdn{i}.rec", tuple(rec.shape[1:]), rec.dtype)
        all_parts.append(tuple(parts))
        sizes.append(offset)
    return Manifest(length, pages, tuple(all_parts), tuple(sizes))


def _view(buf: torch.Tensor, part: Part) -> torch.Tensor:
    return buf[part.offset : part.offset + part.nbytes].view(part.dtype).view(part.shape)


def _gdn_per_rank(fast: Any, snap: Snapshot) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """``snap.gdn`` (rank 0's layers, then rank 1's) split per rank."""
    out, at = [], 0
    for model in ranks(fast):
        n = len(model.gdn_state)
        out.append(snap.gdn[at : at + n])
        at += n
    if at != len(snap.gdn):
        raise ValueError(f"snapshot holds {len(snap.gdn)} GDN states, the model {at}")
    return out


def export_into(fast: Any, snap: Snapshot, bufs: Sequence[torch.Tensor]) -> Manifest:
    """Pack ``snap`` into ``bufs`` (one uint8 buffer per rank, on its device); returns the
    layout. The copies are queued on each device's current stream: synchronize (or record an
    event) before reading the buffers, and keep ``snap`` alive until then."""
    if snap.released:
        raise RuntimeError("exporting a released snapshot")
    manifest = plan(fast, snap.length)
    if len(snap.pages) != manifest.pages:
        raise ValueError(f"snapshot of {snap.length} tokens holds {len(snap.pages)} pages")
    gdn = _gdn_per_rank(fast, snap)
    for r, model in enumerate(ranks(fast)):
        buf = bufs[r]
        if buf.numel() < manifest.rank_bytes[r]:
            raise ValueError(f"rank {r} buffer of {buf.numel()} B < {manifest.rank_bytes[r]} B")
        with torch.cuda.device(buf.device) if buf.is_cuda else _null():
            idx = torch.tensor(snap.pages, dtype=torch.long, device=buf.device)
            parts = iter(manifest.parts[r])
            for k_pool, v_pool in model.kv.values():
                torch.index_select(k_pool, 0, idx, out=_view(buf, next(parts)))
                torch.index_select(v_pool, 0, idx, out=_view(buf, next(parts)))
            for conv, rec in gdn[r]:
                _view(buf, next(parts)).copy_(conv)
                _view(buf, next(parts)).copy_(rec)
    return manifest


def import_from(fast: Any, manifest: Manifest, bufs: Sequence[torch.Tensor]) -> Snapshot:
    """A new snapshot of this replica holding what ``bufs`` (laid out as ``manifest``) holds:
    fresh pages from the pool and fresh GDN state copies. Queued on the current streams."""
    mine = plan(fast, manifest.length)
    if mine != manifest:
        raise ValueError("the exported layout does not match this replica's model")
    r0 = ranks(fast)[0]
    pages: list[int] = []
    try:
        for _ in range(manifest.pages):
            pages.append(r0.pool.alloc())
    except BaseException:
        for page in pages:
            r0.pool.unref(page)
        raise
    gdn: list[tuple[torch.Tensor, torch.Tensor]] = []
    for r, model in enumerate(ranks(fast)):
        buf = bufs[r]
        with torch.cuda.device(buf.device) if buf.is_cuda else _null():
            idx = torch.tensor(pages, dtype=torch.long, device=buf.device)
            parts = iter(manifest.parts[r])
            for k_pool, v_pool in model.kv.values():
                k_pool.index_copy_(0, idx, _view(buf, next(parts)))
                v_pool.index_copy_(0, idx, _view(buf, next(parts)))
            for _ in model.gdn_state:
                conv = _view(buf, next(parts)).clone()
                rec = _view(buf, next(parts)).clone()
                gdn.append((conv, rec))
    return Snapshot(length=manifest.length, gdn=gdn, pages=pages)


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def staging(fast: Any, tokens: int) -> list[torch.Tensor]:
    """One uint8 buffer per rank, on its device, large enough for a ``tokens``-token snapshot."""
    manifest = plan(fast, tokens)
    return [
        torch.empty(n, dtype=torch.uint8, device=_device(model))
        for n, model in zip(manifest.rank_bytes, ranks(fast), strict=True)
    ]


def _device(model: Any) -> torch.device:
    pools = next(iter(model.kv.values()))
    return pools[0].device


# ----------------------------------------------------------------------------- HIP runtime


class Hip:
    """The few HIP runtime calls torch does not expose, from the library torch loaded."""

    _lib: Any = None

    class IpcHandle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_char * 64)]

    def __init__(self) -> None:
        if Hip._lib is None:
            torch.cuda.init()
            path = None
            with open("/proc/self/maps") as maps:
                for line in maps:
                    if "libamdhip64.so" in line:
                        path = line.split()[-1]
                        break
            if path is None:
                raise RuntimeError("libamdhip64.so is not loaded (not a ROCm build of torch?)")
            lib = ctypes.CDLL(path)
            lib.hipGetErrorString.restype = ctypes.c_char_p
            lib.hipIpcOpenMemHandle.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), Hip.IpcHandle, ctypes.c_uint,
            ]
            lib.hipMemcpyAsync.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
            ]
            lib.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            lib.hipHostUnregister.argtypes = [ctypes.c_void_p]
            lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            lib.hipFree.argtypes = [ctypes.c_void_p]
            lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(Hip.IpcHandle), ctypes.c_void_p]
            lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
            Hip._lib = lib
        self.lib = Hip._lib

    def check(self, code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what}: {self.lib.hipGetErrorString(code).decode()} ({code})")


# ----------------------------------------------------------------------------- host transport


def _memfd(name: str) -> int:
    """An anonymous shared memory file (not bounded by the container's /dev/shm size)."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, 0)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.memfd_create(name.encode(), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    return fd


class HostChannel:
    """A shared memory file of ``nbytes`` (``os.memfd_create``; another process maps it from
    the inherited descriptor ``fd``), registered as pinned host memory for every device of
    this process. A header page holds one progress counter per rank; rank ``r`` of a snapshot
    goes at ``HEADER + r * stride``.

    ``put``/``get`` move a whole snapshot (the target starts after the source has finished).
    ``put_stream``/``get_stream`` overlap the two: the source copies each rank in ``chunk``
    pieces and publishes, per rank, how many bytes of move ``seq`` have landed; the target,
    started at the same time, copies each piece to its GPUs as soon as it is published. A
    move's ``seq`` must exceed every earlier one on the channel.
    """

    HEADER = 4096

    def __init__(
        self, nbytes: int, fd: int | None = None, ranks: int = 2, pinned_optional: bool = False
    ) -> None:
        import numpy as np

        if fd is None:
            fd = _memfd("engine-migrate")
            os.ftruncate(fd, nbytes)
        self.fd = fd
        self.nbytes = nbytes
        self.stride = ((nbytes - self.HEADER) // ranks) // ALIGN * ALIGN
        self._map = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.host = torch.frombuffer(self._map, dtype=torch.uint8)
        self._ready = np.frombuffer(self._map, dtype=np.int64, count=ranks)
        self.pinned = False
        self.pin_error: str | None = None
        if torch.cuda.is_available():
            hip = Hip()
            # hipHostRegisterPortable: pinned for every device of this process.
            try:
                hip.check(hip.lib.hipHostRegister(self.host.data_ptr(), nbytes, 1), "hipHostRegister")
                self.pinned = True
            except RuntimeError as exc:  # still correct, only slower (pageable copies)
                if not pinned_optional:
                    raise
                self.pin_error = str(exc)

    def rank(self, r: int, nbytes: int) -> torch.Tensor:
        if nbytes > self.stride:
            raise ValueError(f"{nbytes} B do not fit the channel's {self.stride} B per rank")
        at = self.HEADER + r * self.stride
        return self.host[at : at + nbytes]

    def put(self, manifest: Manifest, bufs: Sequence[torch.Tensor]) -> None:
        """Device buffers to the channel; returns when the bytes are in host memory."""
        for r, n in enumerate(manifest.rank_bytes):
            self.rank(r, n).copy_(bufs[r][:n], non_blocking=True)
        for buf in bufs:
            if buf.is_cuda:
                torch.cuda.current_stream(buf.device).synchronize()

    def get(self, manifest: Manifest, bufs: Sequence[torch.Tensor]) -> None:
        """The channel to device buffers, queued on the current streams."""
        for r, n in enumerate(manifest.rank_bytes):
            bufs[r][:n].copy_(self.rank(r, n), non_blocking=True)

    @staticmethod
    def _pieces(manifest: Manifest, chunk: int) -> list[tuple[int, int, int]]:
        """(rank, begin, end) in the order both sides walk them: piece by piece, ranks inside."""
        out = []
        for begin in range(0, max(manifest.rank_bytes), chunk):
            for r, n in enumerate(manifest.rank_bytes):
                if begin < n:
                    out.append((r, begin, min(begin + chunk, n)))
        return out

    def put_stream(self, manifest: Manifest, bufs: Sequence[torch.Tensor], seq: int, chunk: int = 8 << 20) -> None:
        """``put``, publishing each piece as it lands; returns when all have landed."""
        self.publish(self.launch(manifest, bufs, chunk), seq)

    def launch(self, manifest: Manifest, bufs: Sequence[torch.Tensor], chunk: int = 8 << 20) -> list:
        """Queue ``put_stream``'s copies on the current streams; ``publish`` the result (from
        any thread) to tell the target as each piece lands."""
        marks = []
        for r, begin, end in self._pieces(manifest, chunk):
            host = self.rank(r, manifest.rank_bytes[r])
            host[begin:end].copy_(bufs[r][begin:end], non_blocking=True)
            if bufs[r].is_cuda:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(bufs[r].device))
            else:
                event = None
            marks.append((r, end, event))
        return marks

    def publish(self, marks: list, seq: int) -> None:
        for r, end, event in marks:
            if event is not None:
                event.synchronize()
            self._ready[r] = (seq << 40) + end

    def get_stream(
        self, manifest: Manifest, bufs: Sequence[torch.Tensor], seq: int, chunk: int = 8 << 20,
        timeout: float = 5.0,
    ) -> None:
        """``get`` of move ``seq``, each piece queued as soon as the source has published it."""
        import time

        deadline = time.monotonic() + timeout
        for r, begin, end in self._pieces(manifest, chunk):
            want = (seq << 40) + end
            while int(self._ready[r]) < want:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"move {seq}: rank {r} stalled at {int(self._ready[r]) & ((1 << 40) - 1)} B")
            host = self.rank(r, manifest.rank_bytes[r])
            bufs[r][begin:end].copy_(host[begin:end], non_blocking=True)


# ----------------------------------------------------------------------------- IPC transport


class IpcExport:
    """Per-rank device buffers of the source replica, shared by HIP IPC handle.

    ``handles()`` (bytes, picklable) go to the target once; afterwards each export packs into
    ``bufs``, calls ``publish`` and the target pulls with ``IpcImport.pull``. The source must not pack the next
    export before the target has pulled the previous one.
    """

    def __init__(self, fast: Any, tokens: int) -> None:
        self.hip = Hip()
        manifest = plan(fast, tokens)
        self.capacity = manifest.rank_bytes
        self._ptrs: list[int] = []
        self._handles: list[bytes] = []
        for n, model in zip(self.capacity, ranks(fast), strict=True):
            dev = _device(model)
            with torch.cuda.device(dev):
                ptr = ctypes.c_void_p()
                self.hip.check(self.hip.lib.hipMalloc(ctypes.byref(ptr), n), "hipMalloc")
                handle = Hip.IpcHandle()
                self.hip.check(self.hip.lib.hipIpcGetMemHandle(ctypes.byref(handle), ptr), "hipIpcGetMemHandle")
            self._ptrs.append(ptr.value)
            self._handles.append(bytes(handle.reserved))
        # Pack straight into the shared allocations through torch views of them; where torch
        # cannot wrap foreign device memory, pack into torch buffers and copy (``publish``).
        self.wrapped = True
        self.wrap_error: str | None = None
        try:
            self.bufs = [
                _wrap(ptr, n, _device(m))
                for ptr, n, m in zip(self._ptrs, self.capacity, ranks(fast), strict=True)
            ]
        except Exception as exc:  # noqa: BLE001 - fall back to a local copy
            self.wrapped = False
            self.wrap_error = f"{type(exc).__name__}: {exc}"
            self.bufs = staging(fast, tokens)

    def publish(self, manifest: Manifest) -> None:
        """Make the packed bytes visible in the shared allocations and wait for them."""
        for r, n in enumerate(manifest.rank_bytes):
            buf = self.bufs[r]
            with torch.cuda.device(buf.device):
                stream = torch.cuda.current_stream(buf.device)
                if not self.wrapped:
                    self.hip.check(
                        self.hip.lib.hipMemcpyAsync(self._ptrs[r], buf.data_ptr(), n, 3, stream.cuda_stream),
                        "hipMemcpyAsync",
                    )
        for buf in self.bufs:
            torch.cuda.current_stream(buf.device).synchronize()

    def handles(self) -> list[bytes]:
        return list(self._handles)


def _wrap(ptr: int, nbytes: int, device: torch.device) -> torch.Tensor:
    """A uint8 tensor over device memory torch does not own (``__cuda_array_interface__``)."""

    class _Mem:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
            "strides": None,
        }

    with torch.cuda.device(device):
        t = torch.as_tensor(_Mem(), device=device)
    if t.data_ptr() != ptr or t.device != device:
        raise RuntimeError(f"wrapping device memory gave {t.device}@{t.data_ptr():#x}")
    return t


class IpcImport:
    """The target's side of ``IpcExport``: the source's buffers opened on this process's
    devices (rank ``r`` on its own rank-``r`` device), copied from with one DMA per rank."""

    def __init__(self, fast: Any, handles: Sequence[bytes], devices: Sequence[torch.device] | None = None) -> None:
        self.hip = Hip()
        self._ptrs: list[int] = []
        self.devices = list(devices) if devices is not None else [_device(m) for m in ranks(fast)]
        for raw, dev in zip(handles, self.devices, strict=True):
            handle = Hip.IpcHandle()
            ctypes.memmove(ctypes.addressof(handle), raw, 64)
            with torch.cuda.device(dev):
                ptr = ctypes.c_void_p()
                # hipIpcMemLazyEnablePeerAccess, then no flags
                code = self.hip.lib.hipIpcOpenMemHandle(ctypes.byref(ptr), handle, 1)
                if code != 0:
                    code = self.hip.lib.hipIpcOpenMemHandle(ctypes.byref(ptr), handle, 0)
                self.hip.check(code, f"hipIpcOpenMemHandle on {dev}")
            self._ptrs.append(ptr.value)

    def pull(self, manifest: Manifest, bufs: Sequence[torch.Tensor]) -> None:
        """The source's packed buffers into ``bufs``, queued on the current streams."""
        for r, n in enumerate(manifest.rank_bytes):
            dev = bufs[r].device
            with torch.cuda.device(dev):
                stream = torch.cuda.current_stream(dev).cuda_stream
                # hipMemcpyDeviceToDevice (3)
                self.hip.check(
                    self.hip.lib.hipMemcpyAsync(bufs[r].data_ptr(), self._ptrs[r], n, 3, stream),
                    "hipMemcpyAsync",
                )

    def close(self) -> None:
        for ptr in self._ptrs:
            self.hip.lib.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))
        self._ptrs = []


# ----------------------------------------------------------------------------- live moves


class _Op:
    """One step-boundary operation of ``Mover``: ``start`` queues it, ``poll`` says whether it
    has finished (and then sets ``result``). Both run on the engine thread."""

    kind = "op"

    def __init__(self, mover: Mover) -> None:
        self.mover = mover
        self.result: dict[str, Any] | None = None
        self.t0 = time.perf_counter()
        self.inline = 0.0  # seconds of the engine thread spent in start and poll

    def start(self) -> None:
        raise NotImplementedError

    def poll(self) -> bool:
        return True


def _done(events: list[Any]) -> bool:
    return all(e is None or e.query() for e in events)


class _Export(_Op):
    """Pack the session tail a request kept into this replica's channel."""

    kind = "export"

    def __init__(self, mover: Mover, ticket: int) -> None:
        super().__init__(mover)
        self.ticket = ticket
        self.events: list[Any] = []

    def start(self) -> None:
        m = self.mover
        key = m.tails.get(self.ticket)
        if key is None:
            raise LookupError(f"no session tail recorded for ticket {self.ticket}")
        snap = m.generator.export_tail(key, keep=True)
        if snap is None:
            raise LookupError(f"the {len(key)}-token session tail is no longer cached")
        manifest = plan(m.fast, snap.length)
        channel = m.channels[m.index]
        if max(manifest.rank_bytes) > channel.stride:
            raise ValueError(f"a {snap.length}-token tail does not fit the move channel")
        bufs = m.buffers("out", manifest)
        with torch.inference_mode():
            # Packed on the replica's own streams, so it reads the pages before any later step
            # can reuse them; the copy to host memory then runs beside the next steps.
            export_into(m.fast, snap, bufs)
            self.events = m.on_side(
                lambda r, n: channel.rank(r, n).copy_(bufs[r][:n], non_blocking=True), manifest
            )
        self.key = key
        self.manifest = manifest

    def poll(self) -> bool:
        if not _done(self.events):
            return False
        self.result = {"key": list(self.key), "tokens": self.manifest.length,
                       "bytes": self.manifest.nbytes}
        return True


class _Import(_Op):
    """Copy a tail another replica exported (its channel) to this replica, then keep it."""

    kind = "import"

    def __init__(self, mover: Mover, source: int, key: Sequence[int]) -> None:
        super().__init__(mover)
        self.source = source
        self.key = tuple(int(t) for t in key)
        self.events: list[Any] = []

    def start(self) -> None:
        m = self.mover
        if self.source == m.index or not 0 <= self.source < len(m.channels):
            raise ValueError(f"no move channel from replica {self.source}")
        self.manifest = plan(m.fast, len(self.key))
        channel = m.channels[self.source]
        if max(self.manifest.rank_bytes) > channel.stride:
            raise ValueError(f"a {len(self.key)}-token tail does not fit the move channel")
        bufs = m.buffers("in", self.manifest)
        self.bufs = bufs
        self.events = m.on_side(
            lambda r, n: bufs[r][:n].copy_(channel.rank(r, n), non_blocking=True), self.manifest
        )

    def poll(self) -> bool:
        if not _done(self.events):
            return False
        if self.result is not None:  # scattered, and the scatter has finished on every device
            return True
        m = self.mover
        for event, dev in zip(self.events, m.devices, strict=True):
            if event is not None:
                torch.cuda.current_stream(dev).wait_event(event)
        with torch.inference_mode():
            # Fresh pages and GDN state copies, scattered on the replica's own streams.
            snap = import_from(m.fast, self.manifest, self.bufs)
            kept = m.generator.import_tail(self.key, snap)
        self.result = {"kept": bool(kept), "tokens": len(self.key), "bytes": self.manifest.nbytes}
        # Answer only once the state is in place on every device, whichever stream reads it.
        self.events = [
            torch.cuda.current_stream(dev).record_event() if dev.type == "cuda" else None
            for dev in m.devices
        ]
        return _done(self.events)


class _Drop(_Op):
    """Forget the session tail at ``key`` (the source after a move, or a target whose copy the
    router no longer wants)."""

    kind = "drop"

    def __init__(self, mover: Mover, key: Sequence[int]) -> None:
        super().__init__(mover)
        self.key = tuple(int(t) for t in key)

    def start(self) -> None:
        snap = self.mover.generator.export_tail(self.key, keep=False)
        if snap is not None:
            self.mover.fast.release(snap)
        self.result = {"dropped": snap is not None}


class Mover:
    """A replica's side of live session moves (plan 63).

    ``channels[i]`` is replica ``i``'s ``HostChannel`` (this replica's is ``channels[index]``).
    ``tails`` maps the router's request tickets to the session tail each request kept. The
    operations (``export``, ``import_``, ``drop``) are ``_Op`` objects for the engine thread
    (``engine.server.Worker``), which starts them between steps and polls them until done.
    """

    def __init__(self, generator: Any, index: int, fds: Sequence[int]) -> None:
        self.generator = generator
        self.fast = generator.fast
        if self.fast is None:
            raise RuntimeError("live moves need the fast path")
        self.index = index
        self.devices = [_device(m) for m in ranks(self.fast)]
        self.channels = [
            HostChannel(os.fstat(fd).st_size, fd=fd, ranks=len(self.devices), pinned_optional=True)
            for fd in fds
        ]
        cuda = all(d.type == "cuda" for d in self.devices)
        self.side = [torch.cuda.Stream(d) for d in self.devices] if cuda else None
        self.tails: OrderedDict[int, tuple[int, ...]] = OrderedDict()
        self._bufs: dict[str, list[torch.Tensor]] = {}
        self.counts = {"export": 0, "import": 0, "drop": 0, "failed": 0}
        self.op_ms: dict[str, list[float]] = {"export": [], "import": []}
        self.inline_ms: dict[str, list[float]] = {"export": [], "import": []}
        # Engine loop period of decode-only iterations, by batch size: [sum ms, count] without
        # and with a move's copy in flight on this replica (the stall a move causes).
        self.loop_ms: dict[int, list[float]] = {}

    # -- engine thread

    def remember(self, ticket: int, key: tuple[int, ...]) -> None:
        self.tails[ticket] = key
        while len(self.tails) > 4096:
            self.tails.popitem(last=False)

    def buffers(self, which: str, manifest: Manifest) -> list[torch.Tensor]:
        """Device staging buffers (one per rank) for ``which`` side, grown to fit ``manifest``."""
        have = self._bufs.get(which)
        if have is None or any(b.numel() < n for b, n in zip(have, manifest.rank_bytes, strict=True)):
            grain = 64 << 20
            self._bufs.pop(which, None)
            have = [
                torch.empty(-(-n // grain) * grain, dtype=torch.uint8, device=dev)
                for n, dev in zip(manifest.rank_bytes, self.devices, strict=True)
            ]
            self._bufs[which] = have
        return have

    def on_side(self, copy, manifest: Manifest) -> list[Any]:
        """Run ``copy(rank, nbytes)`` per rank on that device's side stream, after the work
        queued so far on its own stream; one event per rank (None on the CPU)."""
        if self.side is None:
            for r, n in enumerate(manifest.rank_bytes):
                copy(r, n)
            return [None] * len(manifest.rank_bytes)
        events = []
        for r, (n, st, dev) in enumerate(zip(manifest.rank_bytes, self.side, self.devices, strict=True)):
            st.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(st):
                copy(r, n)
                events.append(st.record_event())
        return events

    def export(self, ticket: int) -> _Op:
        return _Export(self, ticket)

    def import_(self, source: int, key: Sequence[int]) -> _Op:
        return _Import(self, source, key)

    def drop(self, key: Sequence[int]) -> _Op:
        return _Drop(self, key)

    def finished(self, op: _Op, ok: bool) -> None:
        if not ok:
            self.counts["failed"] += 1
            return
        self.counts[op.kind] += 1
        if op.kind in self.op_ms:
            self.op_ms[op.kind].append((time.perf_counter() - op.t0) * 1e3)
            self.inline_ms[op.kind].append(op.inline * 1e3)

    def loop(self, batch: int, seconds: float, moving: bool) -> None:
        row = self.loop_ms.setdefault(batch, [0.0, 0, 0.0, 0])
        at = 2 if moving else 0
        row[at] += seconds * 1e3
        row[at + 1] += 1

    def stats(self) -> dict[str, Any]:
        def summary(values: list[float]) -> dict[str, float]:
            if not values:
                return {"n": 0}
            v = sorted(values)
            return {"n": len(v), "p50": round(v[len(v) // 2], 2),
                    "p90": round(v[min(len(v) - 1, int(0.9 * len(v)))], 2),
                    "max": round(v[-1], 2), "mean": round(sum(v) / len(v), 2)}

        return {
            **self.counts,
            "pinned": [c.pinned for c in self.channels],
            "op_ms": {k: summary(v) for k, v in self.op_ms.items()},
            "inline_ms": {k: summary(v) for k, v in self.inline_ms.items()},
            "loop_ms": {
                str(b): {"idle_n": int(r[1]), "idle_mean": round(r[0] / r[1], 3) if r[1] else None,
                         "moving_n": int(r[3]), "moving_mean": round(r[2] / r[3], 3) if r[3] else None}
                for b, r in sorted(self.loop_ms.items())
            },
        }
