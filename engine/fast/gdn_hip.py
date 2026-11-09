"""Plan 50: the decode / verify / chained-verify GDN step (short conv, delta rule, gated norm)
as one hand-written HIP kernel per layer (``engine/fast/hip/gdn_chain.hip``), compiled with hipRTC
from torch's wheel (``hipmega_rtc``, in a child process) and launched on the current stream
(graph-capturable).

Layouts (``LAYOUTS``: name -> (LPC lanes per column group, CPL columns per lane)); conv modes:

- ``sep``: today's Triton conv kernel (``mfma._gdn_conv_chain_kernel``, conv ring + in-place
  conv of q|k|v), then the HIP chain (CONV=0, one head per block);
- ``f1``: conv in the HIP chain's prologue, one head per block (CONV=1, HPW=1), then the tiny
  ``gdn_qk_ring`` kernel writes the q / k conv ring rows (shared by the k-head's two blocks);
- ``f2``: conv in the prologue, one block per k-head (both v-heads, HPW=2): one launch;
- ``v2`` / ``v4`` / ``v8``: as ``f1`` with each head's value columns split over 2 / 4 / 8 blocks
  (VS); the pre-norm o goes to a buffer and ``gdn_post`` runs the gated norm and the q / k ring.

Served (stage 1, ``ENGINE_GDN_HIP=1``, default on): ``FusedDecode`` runs every decode (G = 1),
verify (G = 2) and chained-verify (G = k + 1) GDN layer through ``layer_core`` with one layout
(``ENGINE_GDN_HIP_LAYOUT``, default L4) and the column split chosen by the step's slot count
(``vs_for``: 1 slot -> 4 blocks per head, 2 -> 2, more -> 1). The layout fixes every fp32
summation order, and the split and G change none, so a slot's state, conv ring and output bits
are the same in a decode, a verify and a chained-verify step and at any slot count (spec ==
plain stays bitwise). Kernels are compiled once per process at load (``prepare``, from
``FusedDecode.__init__``) for G = 1..8 and loaded on each rank's device; nothing compiles
after /health.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
import threading
import time
from pathlib import Path

import torch

from engine.fast import hipmega_rtc as rtc

SRC = Path(__file__).parent / "hip" / "gdn_chain.hip"
LAYOUTS = {"L4": (4, 1), "L16x4": (16, 4), "L2": (2, 1), "L8x4": (8, 4), "L4x4": (4, 4)}
MODES = {"sep": (0, 1, 1), "f1": (1, 1, 1), "f2": (1, 2, 1), "v2": (1, 1, 2), "v4": (1, 1, 4),
         "v8": (1, 1, 8)}  # mode -> (CONV, HPW, VS)
ARGS = ("u16* conv, float* rec, const i64* slots, const u16* proj, const float* cw, "
        "const float* alog, const float* dtb, const float* nw, u16* core, float* obuf, i64* prof, "
        "i64 N, i64 stride_in, i64 stride_cslot, i64 stride_slot, float eps, float scale")
CALL = ("conv, rec, slots, proj, cw, alog, dtb, nw, core, obuf, prof, N, stride_in, stride_cslot, "
        "stride_slot, eps, scale")


def threads(layout: str, hpw: int, vs: int = 1, dv: int = 128) -> int:
    lpc, cpl = LAYOUTS[layout]
    return dv // vs // ((64 // lpc) * cpl) * hpw * 64


def valid(layout: str, mode: str, dv: int = 128) -> bool:
    lpc, cpl = LAYOUTS[layout]
    _, hpw, vs = MODES[mode]
    cw = (64 // lpc) * cpl
    return (dv // vs) % cw == 0 and 64 <= threads(layout, hpw, vs, dv) <= 1024


def modes(layout: str, dv: int = 128) -> list[str]:
    return [m for m in MODES if valid(layout, m, dv)]


def kname(layout: str, mode: str, g: int, prof: int = 0) -> str:
    return f"gdn_{layout}_{mode}_g{g}_p{prof}"


def source(layout: str, groups=(2, 4), prof_groups=(4,), dv: int = 128, only=None) -> str:
    lpc, cpl = LAYOUTS[layout]
    out = [SRC.read_text(), "\n// ---- generated entry points\n"]
    for g in groups:
        for mode in modes(layout, dv):
            if only is not None and mode not in only:
                continue
            conv, hpw, vs = MODES[mode]
            for p in (0, 1) if g in prof_groups else (0,):
                nt = threads(layout, hpw, vs, dv)
                out.append(f'extern "C" __global__ void __launch_bounds__({nt}) {kname(layout, mode, g, p)}({ARGS}) '
                           f"{{ gdn_chain_impl<{lpc}, {cpl}, {hpw}, {conv}, {vs}, {g}, {p}>({CALL}); }}\n")
        out.append(f'extern "C" __global__ void __launch_bounds__(256) gdn_qk_ring_g{g}(u16* conv, const i64* slots, '
                   f"const u16* proj, i64 N, i64 stride_in, i64 stride_cslot) "
                   f"{{ gdn_qk_ring_impl<{g}>(conv, slots, proj, N, stride_in, stride_cslot); }}\n")
        out.append(f'extern "C" __global__ void __launch_bounds__(256) gdn_post_g{g}(const float* obuf, const u16* proj, '
                   f"const float* nw, u16* core, u16* conv, const i64* slots, i64 N, i64 stride_in, "
                   f"i64 stride_cslot, float eps) "
                   f"{{ gdn_post_impl<{g}>(obuf, proj, nw, core, conv, slots, N, stride_in, stride_cslot, eps); }}\n")
    return "".join(out)


def compile_layout(layout: str, dims: dict, groups=(2, 4), prof_groups=(4,), only=None) -> tuple[bytes, str]:
    """(code object, compiler log) of ``layout``'s kernels for the shard ``dims``."""
    src = source(layout, groups, prof_groups, dv=dims["head_v_dim"], only=only)
    path = Path(tempfile.gettempdir()) / f"gdn_chain_{layout}_{os.getpid()}.hip"
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

    def __init__(self, layout: str, code: bytes, dev: torch.device, max_rows: int = 16,
                 value_dim: int = 2048) -> None:
        self.layout, self.dev = layout, dev
        self.mod = rtc.Module(code, dev)
        self.prof = torch.zeros(8192, dtype=torch.int64, device=dev)
        self.obuf = torch.zeros((max_rows, value_dim), dtype=torch.float32, device=dev)

    def chain(self, L, proj, states, core, n: int, groups: int, mode: str, prof: int = 0) -> None:
        _, hpw, vs = MODES[mode]
        prepack(L)
        fn = self.mod.fn(kname(self.layout, mode, groups, prof))
        fn.launch((n, L.num_v_heads // hpw, vs), threads(self.layout, hpw, vs, L.head_v_dim), [
            L.conv, L.rec, states, proj, L.cw32, L.alog32, L.dtb32, L.nw32, core, self.obuf, self.prof,
            n, proj.stride(0), L.conv.stride(0), L.rec.stride(0),
            ctypes.c_float(L.eps), ctypes.c_float(L.head_k_dim ** -0.5)])

    def qk_ring(self, L, proj, states, n: int, groups: int) -> None:
        fn = self.mod.fn(f"gdn_qk_ring_g{groups}")
        fn.launch((n, (2 * L.key_dim + 255) // 256, 1), 256,
                  [L.conv, states, proj, n, proj.stride(0), L.conv.stride(0)])

    def post(self, L, proj, states, core, n: int, groups: int) -> None:
        vecs = n * groups * L.num_v_heads
        blocks = (vecs + 3) // 4 + n * ((2 * L.key_dim + 255) // 256)
        self.mod.fn(f"gdn_post_g{groups}").launch(
            blocks, 256, [self.obuf, proj, L.nw32, core, L.conv, states, n, proj.stride(0),
                          L.conv.stride(0), ctypes.c_float(L.eps)])

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
        elif MODES[mode][2] > 1:
            self.post(L, proj, states, core, n, groups)


# ====================================================================== served path (stage 1)
ENABLED = os.environ.get("ENGINE_GDN_HIP", "1") == "1"
LAYOUT = os.environ.get("ENGINE_GDN_HIP_LAYOUT", "L4")
SERVED_GROUPS = tuple(range(1, 9))
SERVED_MODES = ("f1", "v2", "v4", "v8")
_lock = threading.Lock()
_codes: dict[tuple, bytes] = {}
_mods: dict[tuple, rtc.Module] = {}


def _jit_log(what: str, seconds: float) -> None:
    """Reported like a Triton compile: ``ENGINE_LOG_JIT`` lines and ``TRITON_COMPILE_LOG``
    entries (tools/compile_hook, tools/live_compile_probe.py count them)."""
    log = os.environ.get("TRITON_COMPILE_LOG")
    if log:
        import json
        try:
            with open(log, "a") as f:
                f.write(json.dumps({"t": time.time() - seconds, "pid": os.getpid(), "name": f"gdn_hip:{what}",
                                    "s": round(seconds, 3)}) + "\n")
        except OSError:
            pass
    if os.environ.get("ENGINE_LOG_JIT") == "1":
        from engine.fast import jit_log
        print(f"engine-jit: compile pid={os.getpid()} serving={jit_log.SERVING} s={seconds:.3f} "
              f"fn=gdn_hip:{what}", flush=True)


def vs_for(n: int) -> int:
    """Value-column blocks per head for a step of ``n`` slots (``ENGINE_GDN_HIP_VS``, e.g.
    ``1:4,2:2,*:1``, overrides). Any choice gives the same bits."""
    table = {1: 4, 2: 2}
    default = 1
    for item in filter(None, os.environ.get("ENGINE_GDN_HIP_VS", "").split(",")):
        k, v = item.split(":")
        if k == "*":
            default = int(v)
        else:
            table[int(k)] = int(v)
    vs = table.get(n, default)
    mode = {1: "f1", 2: "v2", 4: "v4", 8: "v8"}.get(vs, "f1")
    return vs if valid(LAYOUT, mode) else 1


def dims_of(m) -> dict:
    return dict(key_dim=m.key_dim, num_v_heads=m.num_v_heads, num_k_heads=m.num_k_heads,
                head_k_dim=m.head_k_dim, head_v_dim=m.head_v_dim)


def supported(m) -> bool:
    d = dims_of(m)
    return (d["head_k_dim"] == 128 and d["head_v_dim"] == 128 and d["num_v_heads"] == 2 * d["num_k_heads"]
            and d["key_dim"] == d["num_k_heads"] * 128 and m.conv1d.weight.shape[-1] == 4
            and m.in_proj_z.weight.shape[0] == d["num_v_heads"] * 128
            and m.in_proj_b.weight.shape[0] == d["num_v_heads"]
            and m.in_proj_a.weight.shape[0] == d["num_v_heads"])


def _module(dev: torch.device, d: dict) -> rtc.Module:
    key = (LAYOUT, d["key_dim"], d["num_v_heads"], d["head_k_dim"], d["head_v_dim"])
    with _lock:
        if key not in _codes:
            t0 = time.perf_counter()
            code, log = compile_layout(LAYOUT, d, groups=SERVED_GROUPS, prof_groups=(), only=SERVED_MODES)
            _codes[key] = code
            res = rtc.resources(log)
            spills = {k: v.get("VGPRs Spill") for k, v in res.items() if v.get("VGPRs Spill") not in (None, "0")}
            print(f"engine: gdn_hip compiled layout={LAYOUT} dims={d} kernels={len(res)} "
                  f"s={time.perf_counter() - t0:.1f} spills={spills or 0}", flush=True)
            _jit_log(f"compile:{LAYOUT}", time.perf_counter() - t0)
        mk = key + (dev.index,)
        if mk not in _mods:
            t0 = time.perf_counter()
            prev = torch.cuda.current_device()
            try:
                _mods[mk] = rtc.Module(_codes[key], dev)
            finally:  # the loader makes ``dev`` current; the caller's device stays current
                torch.cuda.set_device(prev)
                rtc.check(rtc.hip().hipSetDevice(prev), "hipSetDevice")
            _jit_log(f"load:{LAYOUT}:cuda{dev.index}", time.perf_counter() - t0)
        return _mods[mk]


def prepare(m) -> None:
    """Load time: compile (once per process) and load the kernels on ``m``'s device, and the
    exact fp32 copies of the layer's small weights."""
    dev = m.conv1d.weight.device
    conv_dim = 2 * m.key_dim + m.value_dim
    m._gdn_hip = dict(
        mod=_module(dev, dims_of(m)),
        cw=m.conv1d.weight.reshape(conv_dim, 4).float().contiguous(),
        alog=m.A_log.float().contiguous(), dtb=m.dt_bias.float().contiguous(),
        nw=m.norm.weight.float().contiguous(),
    )


def layer_core(m, conv_state: torch.Tensor, rec_state: torch.Tensor, proj: torch.Tensor,
               states: torch.Tensor, groups: int) -> torch.Tensor:
    """The GDN step of ``proj``'s rows (raw q|k|v|z|b|a projections, rows ``j * n + r`` = slot
    r's j-th token), state rows ``states`` (ring entry j of slot r at ``states[j * n + r]``;
    entry 0 is the slot's current row): conv and recurrent ring states written, returns the
    gated-norm output (bf16 [rows, heads * DV]) for ``out_proj``."""
    hp = m._gdn_hip
    rows = proj.shape[0]
    n = rows // groups
    assert n * groups == rows and states.shape[0] >= rows and states.dtype == torch.int64
    assert states.stride(0) == 1
    assert proj.stride(1) == 1 and conv_state.is_contiguous() and rec_state.is_contiguous()
    heads = m.num_v_heads
    hv = heads * m.head_v_dim
    core = torch.empty((rows, hv), dtype=proj.dtype, device=proj.device)
    vs = vs_for(n)
    mode = {1: "f1", 2: "v2", 4: "v4", 8: "v8"}[vs]
    obuf = torch.empty((rows, hv), dtype=torch.float32, device=proj.device) if vs > 1 else core
    mod = hp["mod"]
    mod.fn(kname(LAYOUT, mode, groups)).launch(
        (n, heads, vs), threads(LAYOUT, 1, vs, m.head_v_dim), [
            conv_state, rec_state, states, proj, hp["cw"], hp["alog"], hp["dtb"], hp["nw"], core, obuf,
            core, n, proj.stride(0), conv_state.stride(0), rec_state.stride(0),
            ctypes.c_float(m.norm.variance_epsilon), ctypes.c_float(m.head_k_dim ** -0.5)])
    qkb = (2 * m.key_dim + 255) // 256
    if vs > 1:
        mod.fn(f"gdn_post_g{groups}").launch(
            (rows * heads + 3) // 4 + n * qkb, 256,
            [obuf, proj, hp["nw"], core, conv_state, states, n, proj.stride(0), conv_state.stride(0),
             ctypes.c_float(m.norm.variance_epsilon)])
    else:
        mod.fn(f"gdn_qk_ring_g{groups}").launch(
            (n, qkb, 1), 256, [conv_state, states, proj, n, proj.stride(0), conv_state.stride(0)])
    return core
