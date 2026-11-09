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

Nothing here is wired into serving yet.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
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
    this process. Rank ``r`` of a snapshot goes at ``r * stride``."""

    def __init__(self, nbytes: int, fd: int | None = None, ranks: int = 2) -> None:
        if fd is None:
            fd = _memfd("engine-migrate")
            os.ftruncate(fd, nbytes)
        self.fd = fd
        self.nbytes = nbytes
        self.stride = (nbytes // ranks) // ALIGN * ALIGN
        self._map = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.host = torch.frombuffer(self._map, dtype=torch.uint8)
        self.pinned = False
        if torch.cuda.is_available():
            hip = Hip()
            # hipHostRegisterPortable: pinned for every device of this process.
            hip.check(hip.lib.hipHostRegister(self.host.data_ptr(), nbytes, 1), "hipHostRegister")
            self.pinned = True

    def rank(self, r: int, nbytes: int) -> torch.Tensor:
        if nbytes > self.stride:
            raise ValueError(f"{nbytes} B do not fit the channel's {self.stride} B per rank")
        return self.host[r * self.stride : r * self.stride + nbytes]

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

    def __init__(self, fast: Any, handles: Sequence[bytes]) -> None:
        self.hip = Hip()
        self._ptrs: list[int] = []
        self.devices = [_device(m) for m in ranks(fast)]
        for raw, dev in zip(handles, self.devices, strict=True):
            handle = Hip.IpcHandle()
            ctypes.memmove(ctypes.addressof(handle), raw, 64)
            with torch.cuda.device(dev):
                ptr = ctypes.c_void_p()
                # hipIpcMemLazyEnablePeerAccess
                self.hip.check(self.hip.lib.hipIpcOpenMemHandle(ctypes.byref(ptr), handle, 1), "hipIpcOpenMemHandle")
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
