"""Plan 39 step 0: compile HIP C++ at run time and launch it, with nothing but torch's ROCm wheel.

The candidate image has no ``hipcc`` (no ROCm SDK, only ``hsa-rocr``), but torch's ROCm wheel
ships ``libhiprtc.so`` (HIP's runtime compiler, built on ``libamd_comgr.so``'s clang) and
``libamdhip64.so`` in ``torch/lib``. This module drives both through ctypes:

- ``compile_hip(src, arch)``: hipRTC -> a code object (bytes), cached under /tmp by a hash of the
  source and options, with the compiler's kernel-resource remarks (VGPRs, spills, occupancy)
  parsed from the log (``-Rpass-analysis=kernel-resource-usage``).
- ``Module(code, device)``: ``hipModuleLoadData`` on a device, ``fn(name)`` a launchable
  function, ``Function.launch(grid, block, args, stream)`` with ``hipModuleLaunchKernel`` (args
  as ctypes values; torch tensors become their device pointers). A launch on a stream that is
  being captured is recorded into the graph like any other kernel.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
from pathlib import Path

import torch

LIB = Path(torch.__file__).parent / "lib"
CACHE = Path(os.environ.get("ENGINE_HIPMEGA_CACHE", "/tmp/hipmega-cache"))


def _load(name: str, mode: int = ctypes.RTLD_GLOBAL) -> ctypes.CDLL:
    path = LIB / name
    return ctypes.CDLL(str(path) if path.exists() else name, mode=mode)


_hip: ctypes.CDLL | None = None
_rtc: ctypes.CDLL | None = None


def hip() -> ctypes.CDLL:
    global _hip
    if _hip is None:
        _hip = _load("libamdhip64.so")
    return _hip


def rtc() -> ctypes.CDLL:
    global _rtc
    if _rtc is None:
        # Local symbols: comgr carries its own LLVM, which must not interpose on Triton's.
        _rtc = _load("libhiprtc.so", ctypes.RTLD_LOCAL)
    return _rtc


def check(rc: int, what: str) -> None:
    if rc != 0:
        h = hip()
        h.hipGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(f"{what} failed: {rc} ({h.hipGetErrorString(rc).decode()})")


def resources(log: str) -> dict[str, dict[str, str]]:
    """Per kernel, the compiler's resource remarks (``VGPRs``, ``VGPRs Spill``, ``Occupancy
    [waves/SIMD]``, ``LDS Size [bytes/block]``, ...)."""
    out: dict[str, dict[str, str]] = {}
    name = None
    for line in log.splitlines():
        m = re.search(r"remark:\s+Function Name: (\S+)", line)
        if m:
            name = m.group(1)
            out[name] = {}
            continue
        m = re.search(r"remark:\s+([^:\[]+(?:\[[^\]]*\])?):\s*(\S+)\s+\[-Rpass", line)
        if m and name is not None:
            out[name][m.group(1).strip()] = m.group(2)
    return out


def compile_hip(src: str, arch: str = "gfx90a", options: tuple[str, ...] = ()) -> tuple[bytes, str]:
    """(code object, compiler log) of ``src`` for ``arch``; cached under ``CACHE``."""
    opts = ["--offload-arch=" + arch, "-O3", "-std=c++17", "-Rpass-analysis=kernel-resource-usage",
            *options]
    key = hashlib.sha256((src + "\0" + "\0".join(opts)).encode()).hexdigest()[:24]
    co, lg = CACHE / f"{key}.co", CACHE / f"{key}.log"
    if co.exists() and lg.exists():
        return co.read_bytes(), lg.read_text()
    r = rtc()
    prog = ctypes.c_void_p()
    rc = r.hiprtcCreateProgram(ctypes.byref(prog), src.encode(), b"hipmega.hip", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"hiprtcCreateProgram failed: {rc}")
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = r.hiprtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    r.hiprtcGetProgramLogSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value + 1)
    r.hiprtcGetProgramLog(prog, buf)
    log = buf.value.decode(errors="replace")
    if rc != 0:
        raise RuntimeError(f"hiprtcCompileProgram failed: {rc}\n{log[-8000:]}")
    r.hiprtcGetCodeSize(prog, ctypes.byref(n))
    code = ctypes.create_string_buffer(n.value)
    r.hiprtcGetCode(prog, code)
    r.hiprtcDestroyProgram(ctypes.byref(prog))
    data = code.raw
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        co.write_bytes(data)
        lg.write_text(log)
    except OSError:
        pass
    return data, log


def compile_hip_subprocess(path: str, arch: str = "gfx90a", options: tuple[str, ...] = ()) -> tuple[bytes, str]:
    """``compile_hip`` of the file at ``path`` in a child process (hipRTC and comgr's LLVM never
    load into this one), then the cached result."""
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, %r); from engine.fast import hipmega_rtc as r; "
            "r.compile_hip(open(%r).read(), %r, %r)" % (str(Path(__file__).resolve().parents[2]), path, arch,
                                                          tuple(options)))
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)
    src = Path(path).read_text()
    opts = ["--offload-arch=" + arch, "-O3", "-std=c++17", "-Rpass-analysis=kernel-resource-usage", *options]
    key = hashlib.sha256((src + "\0" + "\0".join(opts)).encode()).hexdigest()[:24]
    return (CACHE / f"{key}.co").read_bytes(), (CACHE / f"{key}.log").read_text()


def _arg(a) -> ctypes._SimpleCData:
    if isinstance(a, torch.Tensor):
        return ctypes.c_void_p(a.data_ptr())
    if isinstance(a, ctypes._SimpleCData):
        return a
    if isinstance(a, bool):
        return ctypes.c_int(int(a))
    if isinstance(a, int):
        return ctypes.c_int64(a)
    if isinstance(a, float):
        return ctypes.c_float(a)
    if hasattr(a, "data_ptr"):
        return ctypes.c_void_p(a.data_ptr())
    raise TypeError(f"kernel argument of type {type(a)}")


class Function:
    def __init__(self, handle: ctypes.c_void_p, name: str, device: torch.device) -> None:
        self.handle, self.name, self.device = handle, name, device

    def attr(self, which: int) -> int:
        v = ctypes.c_int()
        check(hip().hipFuncGetAttribute(ctypes.byref(v), ctypes.c_int(which), self.handle),
              "hipFuncGetAttribute")
        return v.value

    def occupancy(self, block: int, smem: int = 0) -> int:
        """Resident blocks per CU (``hipModuleOccupancyMaxActiveBlocksPerMultiprocessor``)."""
        v = ctypes.c_int()
        check(hip().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(v), self.handle, ctypes.c_int(block), ctypes.c_size_t(smem)),
            "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor")
        return v.value

    def launch(self, grid, block, args, stream=None, smem: int = 0) -> None:
        if isinstance(grid, int):
            grid = (grid, 1, 1)
        if isinstance(block, int):
            block = (block, 1, 1)
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        vals = [_arg(a) for a in args]
        ptrs = (ctypes.c_void_p * max(1, len(vals)))(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        check(hip().hipModuleLaunchKernel(
            self.handle, ctypes.c_uint(grid[0]), ctypes.c_uint(grid[1]), ctypes.c_uint(grid[2]),
            ctypes.c_uint(block[0]), ctypes.c_uint(block[1]), ctypes.c_uint(block[2]),
            ctypes.c_uint(smem), ctypes.c_void_p(stream.cuda_stream), ptrs, None),
            f"hipModuleLaunchKernel({self.name})")


class Module:
    """A code object loaded on one device."""

    def __init__(self, code: bytes, device: torch.device) -> None:
        self.device = device
        torch.cuda.set_device(device)
        torch.cuda.current_stream(device)  # make torch create the context first
        check(hip().hipSetDevice(device.index), "hipSetDevice")
        self.handle = ctypes.c_void_p()
        self._code = ctypes.create_string_buffer(code, len(code))
        check(hip().hipModuleLoadData(ctypes.byref(self.handle), self._code), "hipModuleLoadData")
        self._fns: dict[str, Function] = {}

    def fn(self, name: str) -> Function:
        if name not in self._fns:
            h = ctypes.c_void_p()
            check(hip().hipModuleGetFunction(ctypes.byref(h), self.handle, name.encode()),
                  f"hipModuleGetFunction({name})")
            self._fns[name] = Function(h, name, self.device)
        return self._fns[name]
