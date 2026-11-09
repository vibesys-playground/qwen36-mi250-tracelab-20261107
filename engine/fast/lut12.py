"""Plan 44: LUT-12, a bit-exact 12.5-bit storage of BF16 weights for weight-streaming kernels.

A weight matrix ``[..., N, K]`` (K the reduction dim, a multiple of 128) is cut into tiles of
128 consecutive weights of one row. Each tile is stored as 200 bytes:

- ``lo[128]``: the low byte of every BF16 (exponent LSB and 7 mantissa bits), verbatim;
- ``nib[64]``: per weight a nibble ``sign << 3 | code``. A lane decodes 8 weights j=0..7 (one
  "slot", 16 slots per tile) from 4 bytes: weight j<4 in the low nibble of byte j, weight j+4
  in its high nibble;
- ``lut[8]``: code -> the top 7 exponent bits (BF16 bits 14..8). 8 entries cover 8 exponent
  pairs (16 binades); zero and subnormal pairs are ordinary entries.

A tile whose weights span more than 8 exponent pairs is a fallback tile: its lut holds the
little-endian index of a 128-byte row of ``side`` (bytes 0..3) and the flag 0x80 in byte 7
(a real lut entry is 7 bits, so bit 7 of byte 7 is free). ``side[i]`` holds the full high
byte (sign and 7 exponent bits) of each weight; ``lo`` stays valid. Every BF16 bit pattern,
including +-0, subnormals, inf and NaN, round-trips exactly.

Bytes per 128 weights: 200 (+128 for a fallback tile) against 256: 0.781x.

The HIP kernels (``engine/fast/hip/lut12_moe.hip``) rebuild the fp32 value of each weight
(``bf16_bits << 16``, what ``.to(float32)`` gives) with v_perm_b32 byte selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

TILE = 128
TILE_BYTES = 200
FLAG = 0x80
HIP_SRC = Path(__file__).resolve().parent / "hip" / "lut12_moe.hip"


@dataclass
class Packed:
    """A LUT-12 weight: ``planes`` uint8 ``[..., N, K // 128, 200]``, ``side`` uint8
    ``[nfb, 128]`` (at least one row, so the pointer is valid), and the source shape."""

    planes: torch.Tensor
    side: torch.Tensor
    shape: tuple[int, ...]
    fallback: int

    @property
    def tiles(self) -> int:
        return self.planes.numel() // TILE_BYTES

    @property
    def nbytes(self) -> int:
        """Bytes a kernel streams (planes plus the fallback rows actually used)."""
        return self.planes.numel() + self.fallback * TILE

    def ratio(self) -> float:
        return self.nbytes / (self.tiles * TILE * 2)


def _pack_tiles(bits: torch.Tensor, side_base: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``bits`` int32 [T, 128] (BF16 bit patterns, 0..65535) -> (planes uint8 [T, 200],
    side uint8 [F, 128], F)."""
    T = bits.shape[0]
    dev = bits.device
    lo = bits & 0xFF
    hi = bits >> 8
    sign = hi >> 7
    h7 = hi & 0x7F
    srt = torch.sort(h7, dim=1).values
    new = torch.ones_like(srt, dtype=torch.bool)
    new[:, 1:] = srt[:, 1:] != srt[:, :-1]
    ndist = new.sum(dim=1)
    ok = ndist <= 8
    rank = torch.cumsum(new.to(torch.int64), dim=1) - 1
    lut = srt[:, -1:].expand(T, 8).clone()
    sel = new & (rank < 8)
    rows = torch.arange(T, device=dev).unsqueeze(1).expand_as(srt)
    lut[rows[sel], rank[sel]] = srt[sel]
    code = torch.searchsorted(lut.contiguous(), h7.contiguous())
    code = torch.where(ok.unsqueeze(1), code, torch.zeros_like(code)).clamp_(0, 7)
    nib = (sign << 3) | code  # [T, 128]
    nib = nib.view(T, 16, 8)
    nibb = (nib[:, :, 0:4] | (nib[:, :, 4:8] << 4)).reshape(T, 64)
    fb = (~ok).nonzero().flatten()
    F = int(fb.numel())
    lutb = lut.clone()
    if F:
        ids = torch.arange(side_base, side_base + F, device=dev, dtype=torch.int64)
        fl = torch.zeros((F, 8), dtype=lut.dtype, device=dev)
        for b in range(4):
            fl[:, b] = (ids >> (8 * b)) & 0xFF
        fl[:, 7] = FLAG
        lutb[fb] = fl
        side = hi[fb]
    else:
        side = torch.zeros((0, TILE), dtype=hi.dtype, device=dev)
    planes = torch.cat([lo, nibb, lutb], dim=1).to(torch.uint8)
    return planes, side.to(torch.uint8), F


def pack(w: torch.Tensor, chunk_tiles: int = 1 << 17) -> Packed:
    """LUT-12 of a BF16 tensor ``[..., N, K]`` with K % 128 == 0, on ``w``'s device."""
    if w.dtype != torch.bfloat16:
        raise TypeError(f"pack needs bfloat16, got {w.dtype}")
    K = w.shape[-1]
    if K % TILE:
        raise ValueError(f"K={K} is not a multiple of {TILE}")
    flat = w.contiguous().view(torch.int16).reshape(-1, TILE)
    T = flat.shape[0]
    planes = torch.empty((T, TILE_BYTES), dtype=torch.uint8, device=w.device)
    sides = []
    nfb = 0
    for s in range(0, T, chunk_tiles):
        bits = flat[s : s + chunk_tiles].to(torch.int32) & 0xFFFF
        p, side, F = _pack_tiles(bits, nfb)
        planes[s : s + chunk_tiles] = p
        if F:
            sides.append(side)
        nfb += F
    side = torch.cat(sides) if sides else torch.zeros((1, TILE), dtype=torch.uint8, device=w.device)
    return Packed(planes.view(*w.shape[:-1], K // TILE, TILE_BYTES), side.contiguous(),
                  tuple(w.shape), nfb)


def unpack(p: Packed, chunk_tiles: int = 1 << 17) -> torch.Tensor:
    """The BF16 tensor (reference decode in torch; the kernels decode in registers)."""
    planes = p.planes.reshape(-1, TILE_BYTES)
    T = planes.shape[0]
    out = torch.empty((T, TILE), dtype=torch.int32, device=planes.device)
    side = p.side.to(torch.int32)
    for s in range(0, T, chunk_tiles):
        pl = planes[s : s + chunk_tiles].to(torch.int32)
        n = pl.shape[0]
        lo = pl[:, :TILE]
        nibb = pl[:, TILE : TILE + 64].view(n, 16, 4)
        nib = torch.cat([nibb & 0xF, nibb >> 4], dim=2).reshape(n, TILE)
        lut = pl[:, TILE + 64 :]
        code = nib & 7
        sign = nib >> 3
        hi = (sign << 7) | torch.gather(lut, 1, code)
        fbm = (lut[:, 7] & FLAG) != 0
        if bool(fbm.any()):
            fb = fbm.nonzero().flatten()
            ids = lut[fb, 0] | (lut[fb, 1] << 8) | (lut[fb, 2] << 16) | (lut[fb, 3] << 24)
            hi[fb] = side[ids.long()]
        out[s : s + chunk_tiles] = (hi << 8) | lo
    bits = out.to(torch.int16)  # wraps 32768..65535 to the same 16-bit pattern
    return bits.view(torch.bfloat16).reshape(p.shape)


def hip_source() -> str:
    return HIP_SRC.read_text()
