"""Plan 50 (stage 0, not on the served path): the decode / chained-verify GDN step as one
hand-written HIP kernel per layer (``engine/fast/hip/gdn_chain.hip``), compiled with hipRTC
from torch's wheel (``hipmega_rtc``, in a child process) and launched on the current stream
(graph-capturable).

Layouts (``LAYOUTS``: name -> (LPC lanes per column group, CPL columns per lane)); conv modes:

- ``sep``: today's Triton conv kernel (``mfma._gdn_conv_chain_kernel``, conv ring + in-place
  conv of q|k|v), then the HIP chain (CONV=0, one head per block);
- ``f1``: conv in the HIP chain's prologue, one head per block (CONV=1, HPW=1), then the tiny
  ``gdn_qk_ring`` kernel writes the q / k conv ring rows (shared by the k-head's two blocks);
- ``f2``: conv in the prologue, one block per k-head (both v-heads, HPW=2): one launch.
"""

from __future__ import annotations

import ctypes
import tempfile
from pathlib import Path

import torch

from engine.fast import hipmega_rtc as rtc

SRC = Path(__file__).parent / "hip" / "gdn_chain.hip"
LAYOUTS = {"L4": (4, 1), "L16x4": (16, 4), "L2": (2, 1), "L8x4": (8, 4), "L4x4": (4, 4)}
MODES = {"sep": (0, 1), "f1": (1, 1), "f2": (1, 2)}  # mode -> (CONV, HPW)
ARGS = ("u16* conv, float* rec, const i64* slots, const u16* proj, const float* cw, "
        "const float* alog, const float* dtb, const float* nw, u16* core, i64* prof, i64 N, "
        "i64 stride_in, i64 stride_cslot, i64 stride_slot, float eps, float scale")
CALL = "conv, rec, slots, proj, cw, alog, dtb, nw, core, prof, N, stride_in, stride_cslot, stride_slot, eps, scale"


def threads(layout: str, hpw: int, dv: int = 128) -> int:
    lpc, cpl = LAYOUTS[layout]
    return dv // ((64 // lpc) * cpl) * hpw * 64


def kname(layout: str, mode: str, g: int, prof: int = 0) -> str:
    return f"gdn_{layout}_{mode}_g{g}_p{prof}"


def source(layout: str, groups=(2, 4), prof_groups=(4,), dv: int = 128) -> str:
    lpc, cpl = LAYOUTS[layout]
    out = [SRC.read_text(), "\n// ---- generated entry points\n"]
    for g in groups:
        for mode, (conv, hpw) in MODES.items():
            for p in (0, 1) if g in prof_groups else (0,):
                nt = threads(layout, hpw, dv)
                out.append(f'extern "C" __global__ void __launch_bounds__({nt}) {kname(layout, mode, g, p)}({ARGS}) '
                           f"{{ gdn_chain_impl<{lpc}, {cpl}, {hpw}, {conv}, {g}, {p}>({CALL}); }}\n")
        out.append(f'extern "C" __global__ void __launch_bounds__(256) gdn_qk_ring_g{g}(u16* conv, const i64* slots, '
                   f"const u16* proj, i64 N, i64 stride_in, i64 stride_cslot) "
                   f"{{ gdn_qk_ring_impl<{g}>(conv, slots, proj, N, stride_in, stride_cslot); }}\n")
    return "".join(out)


def compile_layout(layout: str, dims: dict, groups=(2, 4)) -> tuple[bytes, str]:
    """(code object, compiler log) of ``layout``'s kernels for the shard ``dims``."""
    src = source(layout, groups, dv=dims["head_v_dim"])
    path = Path(tempfile.gettempdir()) / f"gdn_chain_{layout}.hip"
    path.write_text(src)
    opts = (f"-DKEY_DIM={dims['key_dim']}", f"-DV_HEADS={dims['num_v_heads']}",
            f"-DDK={dims['head_k_dim']}", f"-DDV={dims['head_v_dim']}")
    return rtc.compile_hip_subprocess(str(path), options=opts)


def prepack(L) -> None:
    """fp32 copies (exact) of a layer's small weights, made once at load."""
    if getattr(L, "cw32", None) is None:
        conv_dim = 2 * L.key_dim + L.value_dim
        L.cw32 = L.conv_w.reshape(conv_dim, 4).float().contiguous()
        L.alog32 = L.A_log.float().contiguous()
        L.dtb32 = L.dt_bias.float().contiguous()
        L.nw32 = L.norm_w.float().contiguous()


class GdnHip:
    """One layout's compiled kernels on one device."""

    def __init__(self, layout: str, code: bytes, dev: torch.device) -> None:
        self.layout, self.dev = layout, dev
        self.mod = rtc.Module(code, dev)
        self.prof = torch.zeros(4096, dtype=torch.int64, device=dev)

    def chain(self, L, proj, states, core, n: int, groups: int, mode: str, prof: int = 0) -> None:
        conv, hpw = MODES[mode]
        prepack(L)
        heads = L.num_v_heads
        fn = self.mod.fn(kname(self.layout, mode, groups, prof))
        fn.launch((n, heads // hpw, 1), threads(self.layout, hpw, L.head_v_dim), [
            L.conv, L.rec, states, proj, L.cw32, L.alog32, L.dtb32, L.nw32, core, self.prof,
            n, proj.stride(0), L.conv.stride(0), L.rec.stride(0),
            ctypes.c_float(L.eps), ctypes.c_float(L.head_k_dim ** -0.5)])

    def qk_ring(self, L, proj, states, n: int, groups: int) -> None:
        fn = self.mod.fn(f"gdn_qk_ring_g{groups}")
        fn.launch((n, (2 * L.key_dim + 255) // 256, 1), 256,
                  [L.conv, states, proj, n, proj.stride(0), L.conv.stride(0)])

    def step(self, L, proj, states, core, n: int, groups: int, mode: str, prof: int = 0) -> None:
        """One GDN layer's conv + chain + norm for ``n`` slots x ``groups`` rows (rows
        ``j * n + r``, ``states`` each row's ring state row), ``core`` the bf16 output."""
        if mode == "sep":
            from engine.fast import mfma
            conv_dim = 2 * L.key_dim + L.value_dim
            block = 256
            mfma._gdn_conv_chain_kernel[(n, (conv_dim + block - 1) // block)](
                L.conv, states, proj, L.conv_w, n, conv_dim, proj.stride(0), L.conv.stride(0),
                G=groups, BLOCK=block,
            )
        self.chain(L, proj, states, core, n, groups, mode, prof)
        if mode == "f1":
            self.qk_ring(L, proj, states, n, groups)
