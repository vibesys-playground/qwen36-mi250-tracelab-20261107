"""Plan 27 step 0: lossless ~12-bit storage of the BF16 weights, decoded in registers.

    python tools/pack_probe.py [--devices 6,7] [--parts 1,2,3] [--layers 4] [--samples 6]
    TRITON_INTERPRET=1 python tools/pack_probe.py --smoke      # CPU: round trip + kernel bits

Format (exact; every BF16 bit pattern is kept). A weight is viewed as [N, K] with K the
reduction dim (contiguous). K is cut in chunks of C (the kernel's K step, so the decode needs
one per-row scalar per chunk), and every (row, chunk) block gets an 8-bit ``base`` and an 8-bit
``alt`` exponent:
- ``sm`` uint8 [N, K]: sign << 7 | mantissa (the BF16 low 7 bits);
- ``cx`` uint8 [N, K/2]: a 4-bit exponent code per weight. Within chunk j the byte at
  j*C/2 + c holds column c (low nibble) and column c + C/2 (high nibble), so the kernel loads
  it with the same [rows, C] shape as ``sm``;
- ``ba`` uint16 [N, K/C]: base | alt << 8. Code 1..15 -> exponent field base + code - 1,
  code 0 -> exponent field ``alt``. A block fits if all its exponent fields but at most one
  distinct value lie in a 15-wide window; the window top is the block max, or the second
  largest distinct value when the max is the lone outlier (it then becomes ``alt``).
- Tiles of BN rows x C columns (the kernel's work tile): if any block of a tile does not fit,
  the tile is kept as plain BF16 in a side table and ``tmap`` [ceil(N/BN), K/C] int32 gives its
  element offset (-1: packed). The kernels branch on that scalar per tile, so no weight is
  ever rounded: a packed tile decodes to the exact BF16 bits and a raw tile is the BF16 itself.
Streamed bytes per packed weight: 1 + 0.5 + 2/C (+ 4/(BN*C) map) ~ 12.0 bits at C=1024.

Part 1: ratio, fallback and a bitwise round trip (pack -> planes -> torch decode -> equal to
the original int16 bits) over every 2-D/3-D BF16 parameter of the loaded model (as the engine
loads it, on its GCDs), by tensor class, at C in {64, 256, 1024 (or K)}.
Part 2: Triton copies of ``fused._moe_up_kernel`` / ``fused._moe_down_kernel`` whose routed
weight loads read the packed planes (the shared expert down stays BF16), at the engine's
current work order and grid (plan 22: ORDER 1 from 4 rows, BN 8, BK 1024, 4 warps, 8
programs/CU), on real weights and real routing (``tools/moe_decode_bench.py``'s sets) at
R = 2, 4, 8, for whole experts and for rank 0's TP2 shard. Timed as CUDA graphs against
today's kernels launched as ``FusedDecode._moe`` launches them; outputs compared with
torch.equal. Also prints the TTGIR layouts of both up kernels (the fp32 sum order follows them).
Part 3: the fused GEMV (``fused._gemv_kernel``, pairs tiles 2 x 4 x 2048, no conv epilogue)
on a GDN layer's fused in_proj (and its row half), packed vs today's, at 4 rows.
Prints ``RESULT`` lines. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

import moe_decode_bench as mdb  # noqa: E402
from engine.fast import fused  # noqa: E402
from engine.fast import kernels as _kernels  # noqa: E402

triton, tl = _kernels.triton, _kernels.tl
out = mdb.out

# ============================================================================ format (torch)


def _exp_fields(w16: torch.Tensor) -> torch.Tensor:
    """int16 BF16 bits -> int32 exponent fields."""
    return (w16.to(torch.int32) >> 7) & 0xFF


def block_params(e: torch.Tensor):
    """e: int32 [..., C] exponent fields of blocks. Returns (base, alt, fits) per block."""
    hi = e.amax(-1)
    lo1 = (hi - 14).clamp(min=0)
    below = e < lo1[..., None]
    bmax = torch.where(below, e, torch.full_like(e, -1)).amax(-1)
    bmin = torch.where(below, e, torch.full_like(e, 999)).amin(-1)
    fit1 = (bmax < 0) | (bmax == bmin)
    alt1 = torch.where(bmax < 0, lo1, bmax)
    # option 2: the max is a lone outlier; window under the second distinct max
    m2 = torch.where(e < hi[..., None], e, torch.full_like(e, -1)).amax(-1)
    lo2 = (m2 - 14).clamp(min=0)
    fit2 = (m2 >= 0) & (e.amin(-1) >= lo2)
    use2 = ~fit1 & fit2
    base = torch.where(use2, lo2, lo1)
    alt = torch.where(use2, hi, alt1)
    return base, alt, fit1 | fit2


def block_fits_base_only(e: torch.Tensor) -> torch.Tensor:
    """The plan's original scheme (no alt): code 0 = exponent field 0, window 15 under max."""
    hi = e.amax(-1)
    lo = (hi - 14).clamp(min=1)
    return ((e >= lo[..., None]) | (e == 0)).all(-1)


SLAB = 1 << 25  # weights per slab while packing / checking (bounds the int32 temporaries)


def _slab_rows(K: int, BN: int) -> int:
    return max(BN, (SLAB // K) // BN * BN)


def pack(w: torch.Tensor, C: int, BN: int, aligned: bool = False) -> dict:
    """w: bf16 [N, K] (K % C == 0). Returns the planes (see the module doc). ``smp``/``cxp``
    hold ``sm``/``cx`` from byte 8 of a 16-byte aligned buffer (the kernels' HINT 1 reads them
    at +8); with ``aligned`` also 16-byte aligned copies ``sma``/``cxa`` for HINT 0."""
    N, K = w.shape
    assert K % C == 0 and C % 2 == 0, (K, C)
    nch, half = K // C, C // 2
    dev = w.device
    nt = (N + BN - 1) // BN
    smp = torch.zeros(N * K + 16, dtype=torch.uint8, device=dev)
    cxp = torch.zeros(N * K // 2 + 16, dtype=torch.uint8, device=dev)
    ba = torch.empty((N, nch), dtype=torch.int16, device=dev)
    tfit = torch.ones((nt, nch), dtype=torch.bool, device=dev)
    nunfit = 0
    step = _slab_rows(K, BN)
    for r0 in range(0, N, step):
        r1 = min(N, r0 + step)
        n = r1 - r0
        w16 = w[r0:r1].contiguous().view(torch.int16)
        e = _exp_fields(w16)
        s = w16.to(torch.int32) & 0xFFFF
        smp[8 + r0 * K : 8 + r1 * K] = (((s >> 8) & 0x80) | (s & 0x7F)).to(torch.uint8).reshape(-1)
        del s
        eb = e.view(n, nch, C)
        base, alt, fits = block_params(eb)
        code = eb - base[..., None] + 1
        code = torch.where((code >= 1) & (code <= 15), code, torch.zeros_like(code))
        cb = code.view(n, nch, 2, half)
        cxp[8 + r0 * K // 2 : 8 + r1 * K // 2] = (cb[:, :, 0, :] | (cb[:, :, 1, :] << 4)).to(torch.uint8).reshape(-1)
        del e, eb, code, cb
        v = (base | (alt << 8)).to(torch.int32)
        ba[r0:r1] = torch.where(v >= 32768, v - 65536, v).to(torch.int16)  # uint16 bits
        nunfit += int((~fits).sum())
        tn = (n + BN - 1) // BN
        fp = torch.ones((tn * BN, nch), dtype=torch.bool, device=dev)
        fp[:n] = fits
        tfit[r0 // BN : r0 // BN + tn] &= fp.view(tn, BN, nch).all(1)
    esc = (~tfit).nonzero()
    tmap = torch.full((nt, nch), -1, dtype=torch.int32, device=dev)
    side = torch.zeros((max(1, esc.shape[0]), BN, C), dtype=torch.bfloat16, device=dev)
    for j, (ti, ch) in enumerate(esc.tolist()):
        rows = w[ti * BN : min(N, ti * BN + BN), ch * C : (ch + 1) * C]
        side[j, : rows.shape[0]] = rows
        tmap[ti, ch] = j * BN * C
    p = dict(smp=smp, cxp=cxp, sm=smp[8 : 8 + N * K].view(N, K), cx=cxp[8 : 8 + N * K // 2].view(N, K // 2),
             ba=ba, tmap=tmap, side=side.reshape(-1).contiguous(), N=N, K=K, C=C, BN=BN,
             nesc=int(esc.shape[0]), nblk_unfit=nunfit, nblk=N * nch, ntile=nt * nch)
    if aligned:
        p["sma"], p["cxa"] = p["sm"].clone(), p["cx"].clone()
    return p


def unpack(p: dict, r0: int = 0, r1: int | None = None) -> torch.Tensor:
    """Rows r0:r1 of the planes -> bf16 in torch (the reference decode; raw tiles from the
    side table). r0 must be a multiple of BN."""
    N, K, C, BN = p["N"], p["K"], p["C"], p["BN"]
    r1 = N if r1 is None else r1
    n = r1 - r0
    nch, half = K // C, C // 2
    s = p["sm"][r0:r1].to(torch.int32)
    cx = p["cx"][r0:r1].to(torch.int32).view(n, nch, half)
    code = torch.stack([cx & 15, cx >> 4], 2).view(n, nch, C)
    del cx
    ba = p["ba"][r0:r1].to(torch.int32) & 0xFFFF
    base, alt = (ba & 0xFF)[..., None], (ba >> 8)[..., None]
    e = torch.where(code == 0, alt, base - 1 + code).view(n, K)
    del code
    bits = ((s & 0x80) << 8) | (e << 7) | (s & 0x7F)
    del s, e
    w = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16).view(torch.bfloat16)
    tm = p["tmap"][r0 // BN : (r1 + BN - 1) // BN]
    side = p["side"].view(-1, BN, C)
    for ti, ch in (tm >= 0).nonzero().tolist():
        a = ti * BN
        b = min(n, a + BN)
        w[a:b, ch * C : (ch + 1) * C] = side[int(tm[ti, ch]) // (BN * C), : b - a]
    return w


def roundtrip(p: dict, w: torch.Tensor) -> bool:
    """Every weight of ``w`` decodes from the planes to its exact BF16 bits."""
    step = _slab_rows(p["K"], p["BN"])
    for r0 in range(0, p["N"], step):
        r1 = min(p["N"], r0 + step)
        if not torch.equal(unpack(p, r0, r1).view(torch.int16), w[r0:r1].contiguous().view(torch.int16)):
            return False
    return True


def streamed_bytes(p: dict) -> tuple[int, int]:
    """(bytes a decode pass reads, bytes stored) for the planes."""
    N, K, C, BN = p["N"], p["K"], p["C"], p["BN"]
    per_tile_packed = BN * C * 1.5 + BN * 2 + 4
    per_tile_raw = BN * C * 2 + 4
    ntile, nesc = p["ntile"], p["nesc"]
    streamed = (ntile - nesc) * per_tile_packed + nesc * per_tile_raw
    stored = N * K * 1.5 + N * (K // C) * 2 + ntile * 4 + nesc * BN * C * 2
    return int(streamed), int(stored)


# ============================================================================ kernels

if triton is not None:

    @triton.jit
    def _wdec(sm_ptr, cx_ptr, ba_ptr, tid, map_ptr, side_ptr, grow, rloc, rok, chunk, cc, cok,
              K: tl.constexpr, BK: tl.constexpr, NCH: tl.constexpr, HINT: tl.constexpr):
        """fp32 [BN, BK] of packed weight rows ``grow`` (global), columns chunk * BK + cc."""
        m = rok[:, None] & cok[None, :]
        HALF: tl.constexpr = BK // 2
        off = tl.load(map_ptr + tid)
        if off < 0:
            # HINT 1: the planes start 8 bytes past a 16-byte aligned pointer, so the axis
            # analysis sees 8-byte alignment and the uint8 loads take 8 per thread, the same
            # layout as today's BF16 tile load (and so the same fp32 sum order).
            PAD: tl.constexpr = 8 if HINT == 1 else 0
            c1 = PAD + cc
            c2 = PAD + cc % HALF
            s = tl.load(sm_ptr + grow[:, None] * K + (chunk * BK + c1)[None, :], mask=m, other=0).to(tl.int32)
            b = tl.load(cx_ptr + grow[:, None] * (K // 2) + (chunk * HALF + c2)[None, :], mask=m,
                        other=0).to(tl.int32)
            code = (b >> ((cc // HALF) * 4)[None, :]) & 15
            ba = tl.load(ba_ptr + grow * NCH + chunk, mask=rok, other=0).to(tl.int32)
            base = (ba & 0xFF) - 1
            alt = (ba >> 8) & 0xFF
            e = tl.where(code == 0, alt[:, None], base[:, None] + code)
            bits = ((s & 0x80) << 24) | (e << 23) | ((s & 0x7F) << 16)
            w = bits.to(tl.float32, bitcast=True)
        else:
            w = tl.load(side_ptr + off + rloc[:, None] * BK + cc[None, :], mask=m, other=0.0).to(tl.float32)
        return w

    @triton.jit
    def _moe_up_packed(
        h_ptr, idx_ptr, sm_ptr, cx_ptr, ba_ptr, map_ptr, side_ptr, act_ptr, R,
        H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr,
        BK: tl.constexpr, ORDER: tl.constexpr, HINT: tl.constexpr,
    ):
        """``fused._moe_up_kernel`` with the routed weights read from the packed planes of
        gate_up viewed as [E * 2I, H] (chunk BK, tile BN)."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ntile = tl.cdiv(I, BN)
        P = R * TOPK
        NCH: tl.constexpr = H // BK
        for t in range(pid, P * ntile, nprog):
            if ORDER == 1:
                pair = t % P
                tile = t // P
            else:
                pair = t // ntile
                tile = t % ntile
            row = pair // TOPK
            expert = tl.load(idx_ptr + pair).to(tl.int64)
            rloc = tl.arange(0, BN)
            rows = tile * BN + rloc
            rok = rows < I
            g_row0 = expert * (2 * I) + tile * BN
            u_row0 = expert * (2 * I) + I + tile * BN
            acc_g = tl.zeros([BN], dtype=tl.float32)
            acc_u = tl.zeros([BN], dtype=tl.float32)
            for k0 in range(0, H, BK):
                cc = tl.arange(0, BK)
                cols = k0 + cc
                cok = cols < H
                chunk = k0 // BK
                xv = tl.load(h_ptr + row * H + cols, mask=cok, other=0.0).to(tl.float32)
                wg = _wdec(sm_ptr, cx_ptr, ba_ptr, (g_row0 // BN) * NCH + chunk, map_ptr, side_ptr,
                           g_row0 + rloc, rloc, rok, chunk, cc, cok, H, BK, NCH, HINT)
                wu = _wdec(sm_ptr, cx_ptr, ba_ptr, (u_row0 // BN) * NCH + chunk, map_ptr, side_ptr,
                           u_row0 + rloc, rloc, rok, chunk, cc, cok, H, BK, NCH, HINT)
                acc_g += tl.sum(wg * xv[None, :], axis=1)
                acc_u += tl.sum(wu * xv[None, :], axis=1)
            g = acc_g.to(tl.bfloat16).to(tl.float32)
            u = acc_u.to(tl.bfloat16).to(tl.float32)
            a = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
            tl.store(act_ptr + pair * I + rows, (a * u).to(tl.bfloat16), mask=rok)

    @triton.jit
    def _moe_down_packed(
        act_ptr, r_ptr, idx_ptr, wt_ptr, sm_ptr, cx_ptr, ba_ptr, map_ptr, side_ptr,
        sd_ptr, res_ptr, out_ptr, R, N3,
        H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
        TOPK: tl.constexpr, BN: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr,
        ORDER: tl.constexpr, HINT: tl.constexpr,
    ):
        """``fused._moe_down_kernel`` with the routed down weights read from the packed planes
        of down viewed as [E * H, I] (chunk BKI, tile BN); the shared expert stays BF16."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ntile = tl.cdiv(H, BN)
        NCH: tl.constexpr = I // BKI
        for t in range(pid, R * ntile, nprog):
            if ORDER == 1:
                row = t % R
                tile = t // R
            else:
                row = t // ntile
                tile = t % ntile
            rloc = tl.arange(0, BN)
            rows = tile * BN + rloc
            rok = rows < H
            rbase = r_ptr + row * N3
            acc = tl.zeros([BN], dtype=tl.float32)
            for j in tl.static_range(TOPK):
                pair = row * TOPK + j
                expert = tl.load(idx_ptr + pair).to(tl.int64)
                weight = tl.load(wt_ptr + pair)
                row0 = expert * H + tile * BN
                part = tl.zeros([BN], dtype=tl.float32)
                for k0 in range(0, I, BKI):
                    cc = tl.arange(0, BKI)
                    cols = k0 + cc
                    cok = cols < I
                    chunk = k0 // BKI
                    hv = tl.load(act_ptr + pair * I + cols, mask=cok, other=0.0).to(tl.float32)
                    w = _wdec(sm_ptr, cx_ptr, ba_ptr, (row0 // BN) * NCH + chunk, map_ptr, side_ptr,
                              row0 + rloc, rloc, rok, chunk, cc, cok, I, BKI, NCH, HINT)
                    part += tl.sum(w * hv[None, :], axis=1)
                acc += (part.to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
            routed = acc.to(tl.bfloat16).to(tl.float32)
            shared = tl.zeros([BN], dtype=tl.float32)
            for k0 in range(0, IS, BKS):
                sc = k0 + tl.arange(0, BKS)
                sok = sc < IS
                g = tl.load(rbase + E + sc, mask=sok, other=0.0).to(tl.float32)
                u = tl.load(rbase + E + IS + sc, mask=sok, other=0.0).to(tl.float32)
                sa = ((g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16).to(tl.float32)
                sw = tl.load(sd_ptr + rows[:, None] * IS + sc[None, :],
                             mask=rok[:, None] & sok[None, :], other=0.0).to(tl.float32)
                shared += tl.sum(sw * sa[None, :], axis=1)
            shared = shared.to(tl.bfloat16).to(tl.float32)
            sg = tl.load(rbase + E + 2 * IS).to(tl.float32)
            sg = tl.sigmoid(sg).to(tl.bfloat16).to(tl.float32)
            moe = (routed + (sg * shared).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            res = tl.load(res_ptr + row * H + rows, mask=rok, other=0.0).to(tl.float32)
            tl.store(out_ptr + row * H + rows, (res + moe).to(tl.bfloat16), mask=rok)

    @triton.jit
    def _gemv_packed(
        x_ptr, sm_ptr, cx_ptr, ba_ptr, map_ptr, side_ptr, out_ptr, R, N, stride_out,
        K: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HINT: tl.constexpr,
    ):
        """``fused._gemv_kernel`` (CONV off) with W read from packed planes (chunk BK, tile BN)."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        NCH: tl.constexpr = K // BK
        for t in range(pid, tl.cdiv(N, BN) * tl.cdiv(R, BP), nprog):
            ti = t // tl.cdiv(R, BP)
            rloc = tl.arange(0, BN)
            rows = ti * BN + rloc
            rok = rows < N
            bs = (t % tl.cdiv(R, BP)) * BP + tl.arange(0, BP)
            bok = bs < R
            acc = tl.zeros([BN, BP], dtype=tl.float32)
            for k0 in range(0, K, BK):
                cc = tl.arange(0, BK)
                cols = k0 + cc
                cok = cols < K
                chunk = k0 // BK
                w = _wdec(sm_ptr, cx_ptr, ba_ptr, ti * NCH + chunk, map_ptr, side_ptr,
                          rows.to(tl.int64), rloc, rok, chunk, cc, cok, K, BK, NCH, HINT)
                x = tl.load(x_ptr + bs[:, None] * K + cols[None, :],
                            mask=bok[:, None] & cok[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(w[:, None, :] * x[None, :, :], axis=2)
            y = acc.to(tl.bfloat16)
            omask = rok[:, None] & bok[None, :]
            tl.store(out_ptr + bs[None, :] * stride_out + rows[:, None], y, mask=omask)


p2 = mdb.p2

# ============================================================================ launchers


class PackedLayer:
    """Packed planes of one MoE layer's routed experts (gate_up as [E*2I, H], down [E*H, I])."""

    def __init__(self, L: mdb.Layer, bn: int, bk: int, bki: int, aligned: bool = False):
        self.L = L
        E, H, I = L.E, L.H, L.I
        self.bn, self.bk, self.bki = bn, bk, bki
        gu, dn = L.gu.reshape(E * 2 * I, H), L.dn.reshape(E * H, I)
        self.up = pack(gu, bk, bn, aligned=aligned)
        self.dn = pack(dn, bki, bn, aligned=aligned)
        self.up["roundtrip"] = roundtrip(self.up, gu)
        self.dn["roundtrip"] = roundtrip(self.dn, dn)


def engine_order(R: int) -> tuple[int, int]:
    """(ORDER, programs per CU or 0 = one per item for up) as ``FusedDecode._moe``."""
    new = fused.MOE_ORDER == 1
    order = 1 if new and R >= fused.MOE_TILE_ROWS else 0
    grid = fused.MOE_GRID if new else 2
    return order, grid


def launch_cur_up(s: mdb.Set, cus: int, act=None):
    L = s.layer
    act = s.act if act is None else act
    order, grid = engine_order(s.R)
    bn = fused.TILES["moe_bn"]
    items = s.R * s.topk * triton.cdiv(L.I, bn)
    fused._moe_up_kernel[(items if order else min(items, grid * cus),)](
        s.h, s.idx, L.gu, act, s.R, L.gu.stride(0),
        H=L.H, I=L.I, TOPK=s.topk, BN=bn, BK=min(1024, p2(L.H)), ORDER=order, num_warps=4,
    )
    return act


def launch_cur_dn(s: mdb.Set, cus: int, act=None, outt=None):
    L = s.layer
    act = s.act if act is None else act
    outt = s.out if outt is None else outt
    order, grid = engine_order(s.R)
    bn = fused.TILES["moe_bn"]
    items = s.R * triton.cdiv(L.H, bn)
    fused._moe_down_kernel[(min(items, grid * cus),)](
        act, s.r, s.idx, s.wt, L.dn, L.sd, s.x, outt, s.R, L.dn.stride(0), L.N3,
        H=L.H, I=L.I, IS=L.IS, E=L.E, TOPK=s.topk, BN=bn,
        BKI=min(1024, p2(L.I)), BKS=min(1024, p2(L.IS)), ORDER=order, num_warps=4,
    )
    return outt


def launch_pk_up(s: mdb.Set, P: PackedLayer, cus: int, hint: int, act=None):
    L = s.layer
    act = s.act if act is None else act
    order, grid = engine_order(s.R)
    bn = P.bn
    items = s.R * s.topk * triton.cdiv(L.I, bn)
    u = P.up
    sm, cx = (u["smp"], u["cxp"]) if hint == 1 else (u["sma"], u["cxa"])
    _moe_up_packed[(items if order else min(items, grid * cus),)](
        s.h, s.idx, sm, cx, u["ba"], u["tmap"], u["side"], act, s.R,
        H=L.H, I=L.I, TOPK=s.topk, BN=bn, BK=P.bk, ORDER=order, HINT=hint, num_warps=4,
    )
    return act


def launch_pk_dn(s: mdb.Set, P: PackedLayer, cus: int, hint: int, act=None, outt=None):
    L = s.layer
    act = s.act if act is None else act
    outt = s.out if outt is None else outt
    order, grid = engine_order(s.R)
    bn = P.bn
    items = s.R * triton.cdiv(L.H, bn)
    d = P.dn
    sm, cx = (d["smp"], d["cxp"]) if hint == 1 else (d["sma"], d["cxa"])
    _moe_down_packed[(min(items, grid * cus),)](
        act, s.r, s.idx, s.wt, sm, cx, d["ba"], d["tmap"], d["side"], L.sd, s.x, outt,
        s.R, L.N3, H=L.H, I=L.I, IS=L.IS, E=L.E, TOPK=s.topk, BN=bn,
        BKI=P.bki, BKS=min(1024, p2(L.IS)), ORDER=order, HINT=hint, num_warps=4,
    )
    return outt


def ttgir_layouts(compiled) -> str:
    try:
        ir = compiled.asm["ttgir"]
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"
    lays = re.findall(r"#blocked\d* = #ttg\.blocked<\{[^}]*\}>", ir)
    return " ; ".join(sorted(set(lays)))[:1500]


# ============================================================================ part 2 / 3


def bench_moe(sets_by_r: dict[int, list[mdb.Set]], packed: dict[str, PackedLayer], cus: int,
              timing: bool, shape: str, hints=(0, 1)) -> dict:
    """Bitwise and time of packed vs today's MoE up/down on every set."""
    res = {}
    for R, sets in sets_by_r.items():
        for s in sets:
            s.ref_act = launch_cur_up(s, cus).clone()
            s.ref_out = launch_cur_dn(s, cus, act=s.ref_act).clone()
            s.nu = len(set(s.idx.flatten().tolist()))
    for hint in hints:
        up_ok = dn_ok = chain_ok = True
        try:
            for R, sets in sets_by_r.items():
                for s in sets:
                    P = packed[s.layer.name]
                    a = torch.full_like(s.act, float("nan"))
                    launch_pk_up(s, P, cus, hint, act=a)
                    up_ok &= bool(torch.equal(a, s.ref_act))
                    o = torch.full_like(s.out, float("nan"))
                    launch_pk_dn(s, P, cus, hint, act=s.ref_act, outt=o)
                    dn_ok &= bool(torch.equal(o, s.ref_out))
                    o2 = torch.full_like(s.out, float("nan"))
                    launch_pk_dn(s, P, cus, hint, act=a, outt=o2)
                    chain_ok &= bool(torch.equal(o2, s.ref_out))
        except Exception as exc:  # noqa: BLE001
            out(f"RESULT error shape={shape} hint={hint} {type(exc).__name__}: {str(exc)[:400]}")
            continue
        nsets = sum(len(v) for v in sets_by_r.values())
        out(f"RESULT bitwise shape={shape} hint={hint} sets={nsets} up_identical={up_ok} "
            f"down_identical={dn_ok} chain_identical={chain_ok}")
        res[hint] = (up_ok, dn_ok, chain_ok)
    if not timing:
        return res
    for hint in res:
        for R, sets in sets_by_r.items():
            t = {}
            t["cur_up"] = mdb.time_us([lambda s=s: launch_cur_up(s, cus) for s in sets])
            t["pk_up"] = mdb.time_us([lambda s=s: launch_pk_up(s, packed[s.layer.name], cus, hint) for s in sets])
            t["cur_dn"] = mdb.time_us([lambda s=s: launch_cur_dn(s, cus, act=s.ref_act) for s in sets])
            t["pk_dn"] = mdb.time_us([lambda s=s: launch_pk_dn(s, packed[s.layer.name], cus, hint, act=s.ref_act)
                                      for s in sets])
            t["cur_updn"] = mdb.time_us([lambda s=s: (launch_cur_up(s, cus), launch_cur_dn(s, cus)) for s in sets])
            t["pk_updn"] = mdb.time_us([lambda s=s: (launch_pk_up(s, packed[s.layer.name], cus, hint),
                                                     launch_pk_dn(s, packed[s.layer.name], cus, hint))
                                        for s in sets])
            L0 = sets[0].layer
            uq = statistics.mean(s.nu for s in sets)
            raw_b = uq * 3 * L0.I * L0.H * 2 + L0.H * L0.IS * 2
            ok = res[hint]
            out(f"RESULT time shape={shape} hint={hint} R={R} order={engine_order(R)[0]} "
                f"cur_up_us={t['cur_up']:.1f} pk_up_us={t['pk_up']:.1f} up_ratio={t['pk_up'] / t['cur_up']:.3f} "
                f"cur_dn_us={t['cur_dn']:.1f} pk_dn_us={t['pk_dn']:.1f} dn_ratio={t['pk_dn'] / t['cur_dn']:.3f} "
                f"cur_updn_us={t['cur_updn']:.1f} pk_updn_us={t['pk_updn']:.1f} "
                f"updn_ratio={t['pk_updn'] / t['cur_updn']:.3f} cur_GBps_unique={raw_b / t['cur_updn'] / 1e3:.0f} "
                f"bitwise_up={ok[0]} bitwise_dn={ok[1]} bitwise_chain={ok[2]}")
    return res


def bench_gemv(w: torch.Tensor, fd, cus: int, label: str, rows: int = 4, timing: bool = True) -> None:
    """Packed vs today's fused GEMV (pairs tiles, CONV off) on weight ``w`` at ``rows`` rows."""
    dev = w.device
    N, K = w.shape
    bp, bn, bk = fd._tiles(rows, K)
    assert K % bk == 0, (K, bk)
    P = pack(w, bk, bn, aligned=True)
    rt = roundtrip(P, w)
    xs = [(torch.randn(rows, K, device=dev) * 0.5).to(torch.bfloat16) for _ in range(4)]
    grid = min(triton.cdiv(N, bn) * triton.cdiv(rows, bp), fused.GRID * cus)

    def cur(x, o):
        return fd._gemv(x, w, out=o)

    def pk(x, o, hint):
        sm, cx = (P["smp"], P["cxp"]) if hint == 1 else (P["sma"], P["cxa"])
        _gemv_packed[(grid,)](x, sm, cx, P["ba"], P["tmap"], P["side"], o, rows, N, o.stride(0),
                              K=K, BP=bp, BN=bn, BK=bk, HINT=hint, num_warps=4)
        return o

    outs = [torch.empty((rows, N), dtype=torch.bfloat16, device=dev) for _ in xs]
    refs = [cur(x, torch.empty_like(o)).clone() for x, o in zip(xs, outs, strict=True)]
    for hint in (0, 1):
        try:
            same = True
            for x, o, r in zip(xs, outs, refs, strict=True):
                o.fill_(float("nan"))
                pk(x, o, hint)
                same &= bool(torch.equal(o, r))
        except Exception as exc:  # noqa: BLE001
            out(f"RESULT error gemv {label} hint={hint} {type(exc).__name__}: {str(exc)[:300]}")
            continue
        line = (f"RESULT gemv {label} N={N} K={K} rows={rows} tiles=({bp},{bn},{bk}) roundtrip={rt} "
                f"esc_tiles={P['nesc']}/{P['ntile']} hint={hint} identical={same}")
        if timing:
            tc = mdb.time_us([lambda x=x, o=o: cur(x, o) for x, o in zip(xs, outs, strict=True)])
            tp = mdb.time_us([lambda x=x, o=o: pk(x, o, hint) for x, o in zip(xs, outs, strict=True)])
            line += (f" cur_us={tc:.1f} pk_us={tp:.1f} ratio={tp / tc:.3f} "
                     f"cur_GBps={N * K * 2 / tc / 1e3:.0f} pk_GBps_raw_equiv={N * K * 2 / tp / 1e3:.0f}")
        out(line)


# ============================================================================ part 1


def classify(name: str) -> str:
    n = name.lower()
    if "embed" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    if "mtp" in n:
        return "mtp"
    if "experts.gate_up" in n:
        return "routed_gate_up"
    if "experts.down" in n:
        return "routed_down"
    if "shared_expert_gate" in n:
        return "shared_expert_gate"
    if "shared_expert" in n:
        return "shared_expert"
    if "mlp.gate" in n:
        return "router"
    if "linear_attn" in n:
        return "gdn_proj"
    if "self_attn" in n:
        return "attn_proj"
    return "other"


def part1(named: list[tuple[str, torch.Tensor]]) -> None:
    """Per class: ratio, fallback and round trip at several chunk sizes."""
    t0 = time.perf_counter()
    agg: dict = {}
    hist = torch.zeros(256, dtype=torch.float64)
    skipped = 0
    seen = set()
    for name, p in named:
        if p.dtype != torch.bfloat16 or p.dim() < 2:
            skipped += p.numel() * p.element_size()
            continue
        key = (p.data_ptr(), tuple(p.shape))
        if key in seen:
            continue
        seen.add(key)
        cls = classify(name)
        K = p.shape[-1]
        w2 = p.detach().reshape(-1, K)
        N = w2.shape[0]
        hist += torch.bincount(_exp_fields(w2[: min(N, 4096)].reshape(-1).view(torch.int16)),
                               minlength=256).double().cpu()
        # (C, tile rows) variants: the kernel format first (C = min(1024, K), BN 8)
        variants = [(min(1024, K), 8, "kern")]
        if cls in ("routed_down", "shared_expert") and K % 2 == 0:
            variants.append((K // 2, 8, "tp2"))
        for C in (64, 256):
            if K % C == 0 and C < K:
                variants.append((C, 8, f"c{C}"))
        for C, BN, tag in variants:
            if K % C:
                continue
            a = agg.setdefault((cls, tag), dict(raw=0, streamed=0, stored=0, nblk=0, unfit=0, unfit_base=0,
                                                ntile=0, nesc=0, rt=True, ntens=0, C=set()))
            a["ntens"] += 1
            a["C"].add(C)
            P = pack(w2, C, BN)
            sb, st = streamed_bytes(P)
            a["raw"] += w2.numel() * 2
            a["streamed"] += sb
            a["stored"] += st
            a["nblk"] += P["nblk"]
            a["unfit"] += P["nblk_unfit"]
            a["ntile"] += P["ntile"]
            a["nesc"] += P["nesc"]
            step = _slab_rows(K, 1)
            for r0 in range(0, N, step):
                eb = _exp_fields(w2[r0 : r0 + step].contiguous().view(torch.int16)).view(-1, K // C, C)
                a["unfit_base"] += int((~block_fits_base_only(eb)).sum())
                del eb
            if tag == "kern":
                a["rt"] &= roundtrip(P, w2)
            del P
    tot = dict(raw=0, streamed=0)
    for (cls, tag), a in sorted(agg.items()):
        out(f"RESULT ratio class={cls} fmt={tag} C={sorted(a['C'])} tensors={a['ntens']} raw_GB={a['raw'] / 1e9:.3f} "
            f"streamed_ratio={a['streamed'] / a['raw']:.4f} stored_ratio={a['stored'] / a['raw']:.4f} "
            f"block_unfit={a['unfit'] / max(1, a['nblk']):.2e} block_unfit_no_alt={a['unfit_base'] / max(1, a['nblk']):.2e} "
            f"tile_fallback={a['nesc'] / max(1, a['ntile']):.2e} ({a['nesc']}/{a['ntile']})"
            + (f" roundtrip_bitexact={a['rt']}" if tag == "kern" else ""))
        if tag == "kern" and cls != "embed":
            tot["raw"] += a["raw"]
            tot["streamed"] += a["streamed"]
    rt_all = all(a["rt"] for (c, t), a in agg.items() if t == "kern")
    out(f"RESULT ratio_total streamed_classes(all but embed) raw_GB={tot['raw'] / 1e9:.2f} "
        f"streamed_ratio={tot['streamed'] / max(1, tot['raw']):.4f} roundtrip_all_bitexact={rt_all} "
        f"skipped_non2d_or_nonbf16_MB={skipped / 1e6:.1f} s={time.perf_counter() - t0:.0f}")
    h = hist / hist.sum()
    top = sorted(((float(h[i]), i) for i in range(256) if h[i] > 0), reverse=True)[:24]
    ent = -sum(v * torch.log2(torch.tensor(v)).item() for v, _ in top if v > 0)
    out("RESULT exp_hist (sampled 4096 rows/tensor) entropy_bits~" + f"{ent:.2f} "
        + " ".join(f"e{i}:{v:.4f}" for v, i in sorted(top, key=lambda z: z[1])))




# ============================================================================ modes


def smoke(args) -> None:
    torch.manual_seed(0)
    # round trip on adversarial blocks
    w = (torch.randn(64, 2048) * 0.02).to(torch.bfloat16)
    w[0, :5] = torch.tensor([0.0, -0.0, 1e-40, -3e-39, 1e-30])  # zero, subnormal, tiny normal
    w[1, 7] = 3e4  # a lone huge outlier
    w[2, :64] = torch.logspace(-30, 30, 64).to(torch.bfloat16)  # wide range -> escape
    w[3, 100] = float("inf")
    for C in (64, 256, 1024):
        P = pack(w, C, 8)
        ok = roundtrip(P, w) and torch.equal(unpack(P).view(torch.int16), w.view(torch.int16))
        out(f"RESULT smoke roundtrip C={C} bitexact={ok} esc_tiles={P['nesc']}/{P['ntile']} "
            f"unfit_blocks={P['nblk_unfit']}/{P['nblk']}")
        assert ok
    # kernels vs today's on small random experts with adversarial values
    E, H, I, IS, topk = 16, 2048, 64, 64, 8
    gu = (torch.randn(E, 2 * I, H) * 0.02).to(torch.bfloat16)
    dn = (torch.randn(E, H, I) * 0.05).to(torch.bfloat16)
    gu[3, 5, :64] = torch.logspace(-20, 1, 64).to(torch.bfloat16)  # an escaped tile
    dn[2, 9, :64] = torch.logspace(-20, 1, 64).to(torch.bfloat16)
    gu[1, 0, :3] = torch.tensor([0.0, 1e-40, -0.0])
    sd = (torch.randn(H, IS) * 0.05).to(torch.bfloat16)
    L = mdb.Layer("L0", gu, dn, sd)
    P = PackedLayer(L, 8, 1024, min(1024, p2(I)), aligned=True)
    out(f"RESULT smoke packed up_esc={P.up['nesc']}/{P.up['ntile']} dn_esc={P.dn['nesc']}/{P.dn['ntile']} "
        f"roundtrip up={P.up['roundtrip']} dn={P.dn['roundtrip']}")
    sets_by_r = {}
    for R in (2, 4):
        sets = []
        for k in range(2):
            h = torch.randn(R, H).to(torch.bfloat16)
            x = torch.randn(R, H).to(torch.bfloat16)
            r = torch.randn(R, L.N3).to(torch.bfloat16)
            r[:, 3] += 20.0  # route every row to the escaped experts too
            r[:, 2] += 20.0
            sets.append(mdb.Set(L, h, x, r, topk))
        sets_by_r[R] = sets
    bench_moe(sets_by_r, {"L0": P}, cus=4, timing=False, shape="smoke", hints=(0, 1))

    class FD:
        def _tiles(self, rows, k):
            return 2, 4, min(2048, p2(k))

        def _gemv(self, x, w, out=None):
            o = out
            _, bn, bk = self._tiles(x.shape[0], x.shape[1])
            fused._gemv_kernel[(4,)](x, w, o, x.shape[0], w.shape[0], o.stride(0), o, o, o, 0, 0,
                                     K=w.shape[1], BP=2, BN=bn, BK=bk, CONV=False, KC=2, num_warps=4)
            return o
    wg = (torch.randn(44, 2048) * 0.02).to(torch.bfloat16)
    wg[5, :64] = torch.logspace(-20, 1, 64).to(torch.bfloat16)
    bench_gemv(wg, FD(), cus=2, label="smoke", rows=4, timing=False)


def real(args) -> None:
    os.environ.setdefault("ENGINE_MTP", "1")
    os.environ.setdefault("ENGINE_TP", "1")  # whole layers per GCD; the TP2 shard is built below
    from engine.generator import CHECKPOINT, Generator

    parts = {int(v) for v in args.parts.split(",")}
    hints = tuple(int(v) for v in args.hints.split(","))
    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    fd = fast.fused
    if fd is None:
        raise SystemExit(f"needs the fused layers: {getattr(fast, 'fused_error', None)}")
    dev0 = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev0)
    cus = props.multi_processor_count
    out(f"RESULT env load_s={time.perf_counter() - started:.0f} device={props.name} "
        f"gcn={getattr(props, 'gcnArchName', '?')} cus={cus} torch={torch.__version__} "
        f"triton={triton.__version__} visible={os.environ.get('HIP_VISIBLE_DEVICES')} "
        f"moe_order={fused.MOE_ORDER} moe_grid={fused.MOE_GRID} moe_tile_rows={fused.MOE_TILE_ROWS} "
        f"tiles={fused.TILES} row_tile={fused.ROW_TILE} grid={fused.GRID}")

    if 1 in parts:
        named = list(g.model.named_parameters())
        if getattr(fast, "mtp", None) is not None:
            named += [("mtp." + n, p) for n, p in fast.mtp.named_parameters()]
        with torch.inference_mode():
            try:
                part1(named)
            except Exception:  # noqa: BLE001
                out("RESULT part1 error " + traceback.format_exc().replace("\n", " | "))
        torch.cuda.empty_cache()

    if not ({2, 3} & parts):
        return
    # routing sets as tools/moe_decode_bench.py builds them (real prompts, eager decode)
    on0 = [i for i in range(len(fast.layers)) if fast.layers[i].mlp.experts.gate_up_proj.device == dev0]
    pick = [on0[round(j * (len(on0) - 1) / max(1, args.layers - 1))] for j in range(args.layers)]
    pick = list(dict.fromkeys(pick))
    out(f"RESULT layers on_gcd0={len(on0)} picked={pick}")
    tok = g.tokenizer

    if 3 in parts:
        torch.cuda.set_device(dev0)
        gi = next(i for i in on0 if fast.layer_types[i] == "linear_attention")
        w_in = fd.w_in[gi]
        with torch.inference_mode():
            try:
                bench_gemv(w_in, fd, cus, f"in_proj_full(L{gi})", rows=4)
                half = w_in[: (w_in.shape[0] // 2) // 4 * 4].contiguous()
                bench_gemv(half, fd, cus, f"in_proj_rowhalf(L{gi})", rows=4)
            except Exception:  # noqa: BLE001
                out("RESULT part3 error " + traceback.format_exc().replace("\n", " | "))
        torch.cuda.empty_cache()

    if 2 not in parts:
        return

    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    S = args.sessions
    corpus = pool(9973 * S + args.prompt_len + 10)
    prompts = [corpus[9973 * k : 9973 * k + args.prompt_len] for k in range(S)]

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(prompt) - 1
        position = 0
        while position < body:
            end = min((position // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[position:end], position, prompt[end])
            position = end
        return fast.prefill(slot, prompt[body:], body)

    rec: dict[int, list] = {i: [] for i in pick}
    state = {"on": False}
    moe0 = fd._moe

    def moe_rec(i, x, h):
        if state["on"] and i in rec:
            rec[i].append((h.detach().clone(), x.detach().clone()))
        return moe0(i, x, h)

    fd._moe = moe_rec
    t0 = time.perf_counter()
    with torch.inference_mode():
        slots = list(range(S))
        lasts = [int(torch.argmax(prefill(s, p))) for s, p in zip(slots, prompts, strict=True)]
        fast.use_graphs = False
        for step in range(args.steps):
            state["on"] = True
            logits = fast.decode(lasts, slots, [args.prompt_len + step] * S)
            state["on"] = False
            lasts = logits.argmax(-1).tolist()
        for s in slots:
            fast.reset(s)
    fd._moe = moe0
    out(f"RESULT decode sessions={S} steps={args.steps} s={time.perf_counter() - t0:.1f}")

    T = args.steps
    layers = {}
    for i in pick:
        mlp = fast.layers[i].mlp
        layers[i] = mdb.Layer(f"L{i}", mlp.experts.gate_up_proj, mlp.experts.down_proj,
                              mlp.shared_expert.down_proj.weight, w3=fd.w_moe[i])

    def make_sets(lay: dict) -> dict[int, list[mdb.Set]]:
        sets_by_r: dict[int, list[mdb.Set]] = {}
        for R in (2, 4, 8):
            nslot = max(1, R // 2)
            sets = []
            for i in pick:
                hs = rec[i]
                for k in range(args.samples):
                    t = (k * 5 + 3) % (T - 1)
                    sess = [(k * nslot + j) % S for j in range(nslot)]
                    spec = [(t, s) for s in sess] + [(t + 1, s) for s in sess]
                    h = torch.stack([hs[tt][0][s] for tt, s in spec]).contiguous()
                    x = torch.stack([hs[tt][1][s] for tt, s in spec]).contiguous()
                    r = fd._gemv(h, lay[i].w3)
                    sets.append(mdb.Set(lay[i], h, x, r, fast.top_k))
            sets_by_r[R] = sets
        return sets_by_r

    torch.cuda.set_device(dev0)
    with torch.inference_mode():
        bn = fused.TILES["moe_bn"]
        L0 = layers[pick[0]]
        # full experts
        packed = {L.name: PackedLayer(L, bn, min(1024, p2(L.H)), min(1024, p2(L.I)), aligned=0 in hints) for L in layers.values()}
        for P in packed.values():
            out(f"RESULT pack shape=full {P.L.name} up_esc_tiles={P.up['nesc']}/{P.up['ntile']} "
                f"dn_esc_tiles={P.dn['nesc']}/{P.dn['ntile']} roundtrip_up={P.up['roundtrip']} "
                f"roundtrip_dn={P.dn['roundtrip']}")
        sets = make_sets(layers)
        torch.cuda.synchronize()
        try:
            bench_moe(sets, packed, cus, timing=True, shape=f"full I={L0.I}", hints=hints)
        except Exception:  # noqa: BLE001
            out("RESULT part2 full error " + traceback.format_exc().replace("\n", " | "))
        # layouts of the compiled up kernels (cur vs packed)
        try:
            s = sets[4][0]
            c1 = fused._moe_up_kernel.warmup(
                s.h, s.idx, L0.gu, s.act, s.R, L0.gu.stride(0), H=L0.H, I=L0.I, TOPK=s.topk, BN=bn,
                BK=1024, ORDER=1, num_warps=4, grid=(1,))
            out("RESULT ttgir cur_up " + ttgir_layouts(c1))
            P = packed[L0.name]
            for hint in hints:
                c2 = _moe_up_packed.warmup(
                    s.h, s.idx, P.up["smp" if hint else "sma"], P.up["cxp" if hint else "cxa"], P.up["ba"], P.up["tmap"], P.up["side"], s.act, s.R,
                    H=L0.H, I=L0.I, TOPK=s.topk, BN=bn, BK=1024, ORDER=1, HINT=hint, num_warps=4, grid=(1,))
                out(f"RESULT ttgir pk_up hint={hint} " + ttgir_layouts(c2))
        except Exception as exc:  # noqa: BLE001
            out(f"RESULT ttgir error {type(exc).__name__}: {str(exc)[:300]}")
        del packed, sets
        torch.cuda.empty_cache()
        # rank 0's TP2 shard (engine.fast.tp._moe_shard shapes)
        shards = {}
        for i in pick:
            L = layers[i]
            mlp = fast.layers[i].mlp
            i2, s2 = L.I // 2, L.IS // 2
            gu = torch.cat([L.gu[:, :i2], L.gu[:, L.I : L.I + i2]], dim=1).contiguous()
            dn = L.dn[:, :, :i2].contiguous()
            sd = L.sd[:, :s2].contiguous()
            sh = mlp.shared_expert
            w3 = torch.cat([mlp.gate.weight, sh.gate_proj.weight[:s2], sh.up_proj.weight[:s2],
                            mlp.shared_expert_gate.weight]).contiguous()
            shards[i] = mdb.Layer(f"L{i}", gu, dn, sd, w3=w3)
        packed = {L.name: PackedLayer(L, bn, min(1024, p2(L.H)), min(1024, p2(L.I)), aligned=0 in hints) for L in shards.values()}
        for P in packed.values():
            out(f"RESULT pack shape=tp2 {P.L.name} up_esc_tiles={P.up['nesc']}/{P.up['ntile']} "
                f"dn_esc_tiles={P.dn['nesc']}/{P.dn['ntile']} roundtrip_up={P.up['roundtrip']} "
                f"roundtrip_dn={P.dn['roundtrip']}")
        sets = make_sets(shards)
        torch.cuda.synchronize()
        try:
            bench_moe(sets, packed, cus, timing=True, shape=f"tp2 I={shards[pick[0]].I}", hints=hints)
        except Exception:  # noqa: BLE001
            out("RESULT part2 tp2 error " + traceback.format_exc().replace("\n", " | "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="6,7")
    parser.add_argument("--parts", default="1,3,2")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--hints", default="1,0", help="1: 8-byte plane offset (today's layout); 0: aligned")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    try:
        if args.smoke:
            smoke(args)
        else:
            real(args)
    except Exception:  # noqa: BLE001
        out("RESULT fatal " + traceback.format_exc().replace("\n", " | "))
        raise


if __name__ == "__main__":
    main()
