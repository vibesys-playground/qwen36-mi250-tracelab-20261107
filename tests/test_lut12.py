"""LUT-12 (plan 44): bit-exact round trip of the storage format on CPU, and a bit-level model of
the HIP kernels' v_perm decoder against the format."""

from __future__ import annotations

import struct

import torch

from engine.fast import lut12


def _bits(w: torch.Tensor) -> torch.Tensor:
    return w.contiguous().view(torch.int16)


def _weights(n: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)


def test_roundtrip_gaussian():
    w = _weights(64, 2048)
    p = lut12.pack(w)
    assert p.planes.shape == (64, 16, 200)
    assert torch.equal(_bits(lut12.unpack(p)), _bits(w))
    assert p.ratio() < 0.80


def test_roundtrip_special_and_wide():
    w = _weights(8, 256, seed=1)
    bits = _bits(w).clone()
    # tile 0 of row 0: every special value; tile 1 of row 1: 40 binades -> fallback
    special = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x3F80]
    for i, v in enumerate(special):
        bits[0, i] = v - 65536 if v >= 32768 else v
    for i in range(128):
        e = 60 + (i % 40) * 2
        bits[1, 128 + i] = (e << 7) | (i & 0x7F)
    w2 = bits.view(torch.bfloat16)
    p = lut12.pack(w2)
    assert p.fallback >= 1
    assert torch.equal(_bits(lut12.unpack(p)), bits)


def test_all_patterns():
    allbits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).view(512, 128)
    p = lut12.pack(allbits)
    assert torch.equal(_bits(lut12.unpack(p)), _bits(allbits))


def _perm(s0: int, s1: int, sel: int) -> int:
    """v_perm_b32: bytes of {s0, s1} (s1 low) chosen by each selector byte."""
    src = struct.pack("<II", s1, s0)
    out = 0
    for b in range(4):
        c = (sel >> (8 * b)) & 0xFF
        if c < 8:
            v = src[c]
        elif c == 0x0C:
            v = 0
        elif c >= 0x0D:
            v = 0xFF
        else:
            raise AssertionError(c)
        out |= v << (8 * b)
    return out


def _decode_slot(tile: bytes, side: torch.Tensor, s: int) -> list[int]:
    """The HIP decoder (``cvt(PkLd)`` in lut12_moe.hip) on slot ``s`` of a tile: fp32 words."""
    lo0, lo1 = struct.unpack_from("<II", tile, s * 8)
    (nib,) = struct.unpack_from("<I", tile, 128 + s * 4)
    l0, l1 = struct.unpack_from("<II", tile, 192)
    if l1 & 0x80000000:
        HA, HB = struct.unpack_from("<II", bytes(side[l0].tolist()), s * 8)
    else:
        A, B = nib & 0x0F0F0F0F, (nib >> 4) & 0x0F0F0F0F
        HA = _perm(l1, l0, A & 0x07070707) | ((A & 0x08080808) << 4)
        HB = _perm(l1, l0, B & 0x07070707) | ((B & 0x08080808) << 4)
    sels = (0x04000C0C, 0x05010C0C, 0x06020C0C, 0x07030C0C)
    return [_perm(HA, lo0, c) for c in sels] + [_perm(HB, lo1, c) for c in sels]


def test_kernel_decoder_model():
    w = _weights(4, 256, seed=2)
    bits = _bits(w).clone()
    for i in range(128):
        bits[3, i] = ((30 + (i % 30) * 3) << 7) | i
    w = bits.view(torch.bfloat16)
    p = lut12.pack(w)
    assert p.fallback >= 1
    planes = p.planes.reshape(-1, 200)
    ref = (bits.to(torch.int32) & 0xFFFF).reshape(-1, 128)
    for t in range(planes.shape[0]):
        tile = bytes(planes[t].tolist())
        for s in range(16):
            got = _decode_slot(tile, p.side, s)
            want = [int(ref[t, s * 8 + j]) << 16 for j in range(8)]
            assert got == want, (t, s)
