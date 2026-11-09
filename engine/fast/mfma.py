"""Plan 31: matrix-core (MFMA, ``tl.dot``) decode GEMVs and the chained verify's GDN kernels.

On by default (``ENGINE_SPEC_CHAIN=1``; =0 restores the two-row verify), with at most
``ENGINE_SPEC_KMAX`` (default 3) drafts per slot; it turns on:

- ``gemv``: every fused decode GEMV (the dense projections, the router|shared GEMV, the
  lm_head and the MTP head's fc and lm_head) as one zero-padded 16-row ``tl.dot`` tile per
  launch, its tiles chosen per weight shape (``TILES``) and never by the row count, K summed
  in a fixed order (with a K split, fp32 partials summed in part order by a second kernel).
  A row's output therefore does not depend on how many rows share the launch nor on its
  position in the tile: plain decode and every verify shape give the same bits for a row.
  More than 16 rows run as further launches. Plan 31 step 0 (j143): in_proj 0.99-1.11 TB/s
  and the lm_head shard 1.24-1.33 TB/s at 2-16 rows, bitwise row-count/position invariant.
  ``ENGINE_MFMA_GEMV=1`` turns on only this part. The routed experts keep today's kernels.
- ``gdn_chain``: a chained verify step's Gated DeltaNet layer, G rows per slot (its last
  token and G - 1 drafts at consecutive positions, row ``j * N + slot``): the short
  convolution and the delta rule of each slot run row after row in one launch each, and
  the state after row j is written to the slot's ring entry j (``FastModel._ring_row``), so
  accepting a drafts makes entry a current without a copy. Per-row math and order are
  ``_gdn_conv_kernel``'s / ``fused._gdn_step_kernel``'s, so a verified row equals plain
  decode bit for bit.
"""

from __future__ import annotations

import os

import torch

from engine.fast import kernels

triton, tl = kernels.triton, kernels.tl

CHAIN = os.environ.get("ENGINE_SPEC_CHAIN", "1") == "1"
GEMV = CHAIN or os.environ.get("ENGINE_MFMA_GEMV", "0") == "1"
# Most drafts per slot of a chained verify step (the GDN ring holds KMAX + 1 states per slot).
# 3 by default (plan 31 stage 1, j161: k = 3 the best tokens/ms on replay-like text); 7 for
# experiments.
KMAX = int(os.environ.get("ENGINE_SPEC_KMAX", "3"))
RING = KMAX + 1
BM = 16  # the fixed row tile
# Plan 37: the MoE router top-k inside the router|shared GEMV's K-sum kernel, so no
# ``fused._topk_kernel`` launch per layer (on by default; ENGINE_TOPK_EPI=0 restores the
# separate launch). R extra programs of the K-sum launch, one per row, each re-sum that row's
# E router columns from the same fp32 K parts in the same order and run the top-k on those
# values in registers (no cross-program dependency). The softmax/top-k math is
# ``_topk_vals``, the body of ``fused._topk``, in one wave (num_warps=1 as ``_topk_kernel``),
# so experts, order and weights are bitwise equal.
# Step 0 (j183, TP2 GCD 2,3, served verify_chain B=2): bitwise over 1000 graph replays x 3
# layers x R 1-32; 9.155 -> 8.809 ms (k=1), 13.648 -> 13.311 ms (k=3). A last-program variant
# (atomic tile counter, one wave doing every row's top-k) was slower (9.594, 14.385): dropped.
# Step 1 (j192, on f4b2962): bitwise again (1000 replays x 3 layers x R 1-32); chain_gate
# 9.285 -> 9.002 ms (k=1), 13.75 -> 13.427 ms (k=3); served A/B 0/1/0/1 302/302 requests
# identical across all four processes; 0 Triton compiles after /health.
TOPK_EPI = os.environ.get("ENGINE_TOPK_EPI", "1") == "1"
# Plan 64 (a): how the top-k program of ``_ksum_router_kernel`` selects (ENGINE_TOPK_SEL):
# 0 (default) today's TOPK rounds of a float max and an index min (2 TOPK serial one-wave
# reductions); 1 TOPK rounds of one int64 max over keys ``p bits << 32 | (E - 1 - e)``; 2 one
# reduction whose combine keeps the top 8 keys (a bitonic merge of two sorted 8-lists; TOPK 8
# only). All three pick the same experts in the same order (max p first, ties to the lowest
# index: p >= 0, so its bits order as its values) and build and renormalize the weights with
# the same code, so idx/wt are bitwise equal.
TOPK_SEL = int(os.environ.get("ENGINE_TOPK_SEL", "0"))

# Tiles per weight shape (N, K): outputs per tile, K chunk, warps, K split. Never by row count.
# in_proj and lm_head from j143's sweep (TP2 shard); the small shapes split K so a launch has
# a few hundred programs.
TILES: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (6176, 2048): (64, 128, 4, 1),      # TP2 GDN in_proj
    (124160, 2048): (32, 256, 4, 1),    # TP2 lm_head shard
    (2048, 2048): (16, 128, 4, 4),      # TP2 out_proj, o_proj
    (4608, 2048): (32, 128, 4, 2),      # TP2 attention q|k|v
    (769, 2048): (16, 128, 4, 8),       # TP2 router|shared gate|up|gate
    (2048, 4096): (16, 128, 4, 4),      # MTP fc (replicated)
    (12352, 2048): (64, 128, 4, 1),     # whole in_proj
    (248320, 2048): (32, 256, 4, 1),    # whole lm_head
    (9216, 2048): (32, 128, 4, 2),      # whole q|k|v
    (1281, 2048): (16, 128, 4, 8),      # whole router|shared
}
DEFAULT = (32, 128, 4, 1)


def _tiles_env(spec: str) -> dict:
    """``ENGINE_MFMA_TILES``: ``"NxK=bn,bk,warps,ksplit;..."`` entries replacing ``TILES``'."""
    table = {}
    for item in filter(None, (p.strip() for p in spec.split(";"))):
        shape, cfg = item.split("=")
        n, k = (int(v) for v in shape.split("x"))
        table[(n, k)] = tuple(int(v) for v in cfg.split(","))
    return table


TILES.update(_tiles_env(os.environ.get("ENGINE_MFMA_TILES", "")))


if triton is not None:

    @triton.jit(do_not_specialize=["R", "N"])
    def _mfma_gemv_kernel(
        x_ptr, w_ptr, out_ptr, part_ptr, R, N, stride_out,
        K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        KSPLIT: tl.constexpr, DOT_F32: tl.constexpr,
    ):
        """out[r, n] = bf16(x[r, :] . W[n, :]) for R <= BM rows in one zero-padded BM-row tile.
        Program (N tile, K part); K in BK chunks in a fixed order (``tl.dot``)."""
        pid = tl.program_id(0)
        nb = pid // KSPLIT
        kp = pid % KSPLIT
        KP: tl.constexpr = K // KSPLIT
        rm = tl.arange(0, BM)
        mok = rm < R
        rn = nb * BN + tl.arange(0, BN)
        nok = rn < N
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(kp * KP, kp * KP + KP, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
            b = tl.load(w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None], mask=nok[None, :], other=0.0)
            if DOT_F32:
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
            else:
                acc = tl.dot(a, b, acc)
        m = mok[:, None] & nok[None, :]
        if KSPLIT == 1:
            tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)
        else:
            tl.store(part_ptr + (kp * BM + rm[:, None]) * N + rn[None, :], acc, mask=m)

    @triton.jit(do_not_specialize=["R", "N"])
    def _ksum_kernel(part_ptr, out_ptr, R, N, stride_out, KSPLIT: tl.constexpr,
                     BM: tl.constexpr, BLOCK: tl.constexpr):
        """out[r, n] = bf16(sum over K parts p, in order, of part[p, r, n])."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        r = i // N
        n = i % N
        m = r < R
        s = tl.load(part_ptr + r * N + n, mask=m, other=0.0)
        for p in tl.static_range(1, KSPLIT):
            s += tl.load(part_ptr + (p * BM + r) * N + n, mask=m, other=0.0)
        tl.store(out_ptr + r * stride_out + n, s.to(tl.bfloat16), mask=m)

    @triton.jit
    def _topk_vals(p, ee, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr):
        """fp32 softmax over E values (BF16 logits as fp32), top-k, renormalized, rounded to
        BF16: ``fused._topk``'s math on values already loaded (``ee = tl.arange(0, E)``)."""
        p = tl.exp(p - tl.max(p, axis=0))
        p = p / tl.sum(p, axis=0)
        kk = tl.arange(0, TOPKP)
        sel_w = tl.zeros([TOPKP], dtype=tl.float32)
        sel_i = tl.zeros([TOPKP], dtype=tl.int32)
        for j in tl.static_range(TOPK):
            mx = tl.max(p, axis=0)
            i = tl.min(tl.where(p == mx, ee, E), axis=0)
            sel_w = tl.where(kk == j, mx, sel_w)
            sel_i = tl.where(kk == j, i, sel_i)
            p = tl.where(ee == i, -1.0, p)
        sel_w = sel_w / tl.sum(sel_w, axis=0)
        return sel_i, sel_w.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _cx(a, b):
        return tl.maximum(a, b), tl.minimum(a, b)

    @triton.jit
    def _top8_combine(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7):
        """The 8 largest of two descending 8-lists, descending: max(a_i, b_{7-i}) is bitonic
        and holds them; a bitonic merge (distances 4, 2, 1) sorts it."""
        c0 = tl.maximum(a0, b7)
        c1 = tl.maximum(a1, b6)
        c2 = tl.maximum(a2, b5)
        c3 = tl.maximum(a3, b4)
        c4 = tl.maximum(a4, b3)
        c5 = tl.maximum(a5, b2)
        c6 = tl.maximum(a6, b1)
        c7 = tl.maximum(a7, b0)
        c0, c4 = _cx(c0, c4)
        c1, c5 = _cx(c1, c5)
        c2, c6 = _cx(c2, c6)
        c3, c7 = _cx(c3, c7)
        c0, c2 = _cx(c0, c2)
        c1, c3 = _cx(c1, c3)
        c4, c6 = _cx(c4, c6)
        c5, c7 = _cx(c5, c7)
        c0, c1 = _cx(c0, c1)
        c2, c3 = _cx(c2, c3)
        c4, c5 = _cx(c4, c5)
        c6, c7 = _cx(c6, c7)
        return c0, c1, c2, c3, c4, c5, c6, c7

    @triton.jit
    def _key_pick(sel_i, sel_w, kk, key, j: tl.constexpr, E: tl.constexpr):
        i = (E - 1 - (key & 0xFFFFFFFF)).to(tl.int32)
        w = (key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        return tl.where(kk == j, i, sel_i), tl.where(kk == j, w, sel_w)

    @triton.jit
    def _topk_vals_sel(p, ee, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr,
                       SEL: tl.constexpr):
        """``_topk_vals`` with the selection of ``TOPK_SEL`` (1 or 2): the same softmax, the
        same experts in the same order, the same renormalization code."""
        p = tl.exp(p - tl.max(p, axis=0))
        p = p / tl.sum(p, axis=0)
        kk = tl.arange(0, TOPKP)
        sel_w = tl.zeros([TOPKP], dtype=tl.float32)
        sel_i = tl.zeros([TOPKP], dtype=tl.int32)
        key = (p.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (E - 1 - ee).to(tl.int64)
        if SEL == 2:
            neg = tl.full([E], -1, tl.int64)
            k0, k1, k2, k3, k4, k5, k6, k7 = tl.reduce(
                (key, neg, neg, neg, neg, neg, neg, neg), 0, _top8_combine)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k0, 0, E)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k1, 1, E)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k2, 2, E)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k3, 3, E)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k4, 4, E)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k5, 5, E)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k6, 6, E)
            sel_i, sel_w = _key_pick(sel_i, sel_w, kk, k7, 7, E)
        else:
            for j in tl.static_range(TOPK):
                mk = tl.max(key, axis=0)
                sel_i, sel_w = _key_pick(sel_i, sel_w, kk, mk, j, E)
                key = tl.where(key == mk, -1, key)
        sel_w = sel_w / tl.sum(sel_w, axis=0)
        return sel_i, sel_w.to(tl.bfloat16).to(tl.float32)

    @triton.jit(do_not_specialize=["R", "N"])
    def _ksum_router_kernel(part_ptr, out_ptr, idx_ptr, wt_ptr, R, N, stride_out,
                            KSPLIT: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr,
                            E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr,
                            SEL: tl.constexpr = 0):
        """``_ksum_kernel`` (programs < cdiv(R N, BLOCK)) plus, in R extra programs, each
        row's router top-k (``TOPK_EPI``). One wave per program (num_warps=1)."""
        pid = tl.program_id(0)
        nmain = tl.cdiv(R * N, BLOCK)
        if pid < nmain:
            i = pid * BLOCK + tl.arange(0, BLOCK)
            r = i // N
            n = i % N
            m = r < R
            s = tl.load(part_ptr + r * N + n, mask=m, other=0.0)
            for p in tl.static_range(1, KSPLIT):
                s += tl.load(part_ptr + (p * BM + r) * N + n, mask=m, other=0.0)
            tl.store(out_ptr + r * stride_out + n, s.to(tl.bfloat16), mask=m)
        else:
            ee = tl.arange(0, E)
            kk = tl.arange(0, TOPKP)
            row = pid - nmain
            s = tl.load(part_ptr + row * N + ee)
            for p in tl.static_range(1, KSPLIT):
                s += tl.load(part_ptr + (p * BM + row) * N + ee)
            v = s.to(tl.bfloat16).to(tl.float32)
            if SEL == 0:
                sel_i, sel_w = _topk_vals(v, ee, E, TOPK, TOPKP)
            else:
                sel_i, sel_w = _topk_vals_sel(v, ee, E, TOPK, TOPKP, SEL)
            tl.store(idx_ptr + row * TOPK + kk, sel_i, mask=kk < TOPK)
            tl.store(wt_ptr + row * TOPK + kk, sel_w, mask=kk < TOPK)

    @triton.jit
    def _gdn_conv_chain_kernel(
        state_ptr, slot_ptr, io_ptr, w_ptr, N, C, stride_io, stride_slot, G: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """The short convolution (kernel size 4) of slot ``r``'s G rows ``j * N + r`` in turn:
        the window is read from state row ``slots[r]``, and after row j it is stored to state
        row ``slots[j * N + r]`` (the ring entry j). Per row ``_gdn_conv_kernel``'s math and
        order. In place on ``io`` (the raw q|k|v columns of the projection rows)."""
        r = tl.program_id(0)
        block = tl.program_id(1)
        first = tl.load(slot_ptr + r).to(tl.int64)
        c = block * BLOCK + tl.arange(0, BLOCK)
        ok = c < C
        a = state_ptr + first * stride_slot + c * 4
        s1 = tl.load(a + 1, mask=ok, other=0.0)
        s2 = tl.load(a + 2, mask=ok, other=0.0)
        s3 = tl.load(a + 3, mask=ok, other=0.0)
        w0 = tl.load(w_ptr + c * 4, mask=ok, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + c * 4 + 1, mask=ok, other=0.0).to(tl.float32)
        w2 = tl.load(w_ptr + c * 4 + 2, mask=ok, other=0.0).to(tl.float32)
        w3 = tl.load(w_ptr + c * 4 + 3, mask=ok, other=0.0).to(tl.float32)
        for j in tl.static_range(G):  # unrolled: straight-line code per row, as the decode kernels
            row = j * N + r
            x = tl.load(io_ptr + row * stride_io + c, mask=ok, other=0.0)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            acc += s1.to(tl.float32) * w0
            acc += s2.to(tl.float32) * w1
            acc += s3.to(tl.float32) * w2
            acc += x.to(tl.float32) * w3
            y = acc.to(tl.bfloat16).to(tl.float32)
            y = (y * tl.sigmoid(y)).to(tl.bfloat16)
            dst = tl.load(slot_ptr + row).to(tl.int64)
            b = state_ptr + dst * stride_slot + c * 4
            tl.store(b, s1, mask=ok)
            tl.store(b + 1, s2, mask=ok)
            tl.store(b + 2, s3, mask=ok)
            tl.store(b + 3, x, mask=ok)
            tl.store(io_ptr + row * stride_io + c, y, mask=ok)
            s1 = s2
            s2 = s3
            s3 = x

    @triton.jit
    def _gdn_row(s, base_in, nw_ptr, alog_ptr, dtb_ptr, head, k_head, eps, scale,
                 Z_OFF, B_OFF, A_OFF, KEY_DIM: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr):
        """``fused._gdn_step_kernel``'s math for one row on the fp32 state tile ``s``."""
        kk = tl.arange(0, DK)
        vv = tl.arange(0, DV)
        q = tl.load(base_in + k_head * DK + kk).to(tl.float32)
        k = tl.load(base_in + KEY_DIM + k_head * DK + kk).to(tl.float32)
        v = tl.load(base_in + 2 * KEY_DIM + head * DV + vv).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        beta = tl.sigmoid(tl.load(base_in + B_OFF + head).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        a = tl.load(base_in + A_OFF + head).to(tl.float32) + tl.load(dtb_ptr + head).to(tl.float32)
        softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        g = -tl.exp(tl.load(alog_ptr + head).to(tl.float32)) * softplus
        s = s * tl.exp(g)
        kv_mem = tl.sum(s * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        s = s + k[:, None] * delta[None, :]
        o = tl.sum(s * q[:, None], axis=0)
        o = o.to(tl.bfloat16).to(tl.float32)
        n = (o * tl.rsqrt(tl.sum(o * o, axis=0) / DV + eps)).to(tl.bfloat16).to(tl.float32)
        y = (tl.load(nw_ptr + vv).to(tl.float32) * n).to(tl.bfloat16).to(tl.float32)
        zz = tl.load(base_in + Z_OFF + head * DV + vv).to(tl.float32)
        return s, (y * zz * tl.sigmoid(zz)).to(tl.bfloat16)

    @triton.jit
    def _gdn_chain_step_kernel(
        state_ptr, slot_ptr, in_ptr, alog_ptr, dtb_ptr, nw_ptr, out_ptr, N,
        stride_in, V_HEADS, stride_slot, stride_head, eps, scale,
        Z_OFF, B_OFF, A_OFF,
        RATIO: tl.constexpr, KEY_DIM: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
        G: tl.constexpr,
    ):
        """The delta rule of one v-head of slot ``r`` over its G rows ``j * N + r`` in turn:
        the state tile is loaded once from row ``slots[r]``, and after row j stored to row
        ``slots[j * N + r]`` (ring entry j). Per-row math as ``fused._gdn_step_kernel``."""
        r = tl.program_id(0)
        head = tl.program_id(1)
        k_head = head // RATIO
        first = tl.load(slot_ptr + r).to(tl.int64)
        kk = tl.arange(0, DK)
        vv = tl.arange(0, DV)
        tile = head * stride_head + kk[:, None] * DV + vv[None, :]
        s = tl.load(state_ptr + first * stride_slot + tile)
        for j in tl.static_range(G):
            row = j * N + r
            s, y = _gdn_row(s, in_ptr + row * stride_in, nw_ptr, alog_ptr, dtb_ptr, head, k_head,
                            eps, scale, Z_OFF, B_OFF, A_OFF, KEY_DIM=KEY_DIM, DK=DK, DV=DV)
            dst = tl.load(slot_ptr + row).to(tl.int64)
            tl.store(state_ptr + dst * stride_slot + tile, s)
            tl.store(out_ptr + (row * V_HEADS + head) * DV + vv, y)
            s = s.to(state_ptr.dtype.element_ty).to(tl.float32)


def tiles_for(n: int, k: int) -> tuple[int, int, int, int]:
    bn, bk, warps, ksplit = TILES.get((n, k), DEFAULT)
    bk = min(bk, triton.next_power_of_2(k))
    if k % (ksplit * bk):
        ksplit = 1
        while k % bk:
            bk //= 2
    return bn, bk, warps, ksplit


class KParts:
    """Plan 51: a K-split GEMV's fp32 parts, not yet summed (``gemv(..., defer=True)``):
    ``part[p, r, n]`` at ``(p * BM + r) * n``. The consumer sums them in part order and rounds
    to BF16, as ``_ksum_kernel`` does (``tp._ll_allreduce_kernel``'s KSPLIT prologue);
    ``materialize`` runs ``_ksum_kernel`` itself."""

    __slots__ = ("part", "out", "ksplit")

    def __init__(self, part: torch.Tensor, out: torch.Tensor, ksplit: int) -> None:
        self.part, self.out, self.ksplit = part, out, ksplit

    @property
    def shape(self) -> torch.Size:
        return self.out.shape

    @property
    def device(self) -> torch.device:
        return self.out.device

    @property
    def is_cuda(self) -> bool:
        return self.out.is_cuda

    def materialize(self) -> torch.Tensor:
        rows, n = self.out.shape
        with kernels.on_device(self.out):
            _ksum_kernel[(triton.cdiv(rows * n, 1024),)](
                self.part, self.out, rows, n, self.out.stride(0), KSPLIT=self.ksplit, BM=BM,
                BLOCK=1024, num_warps=4)
        return self.out


def gemv(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None,
         tiles: tuple[int, int, int, int] | None = None, defer: bool = False):
    """``out = bf16(x @ w.T)`` in 16-row MFMA launches (see the module doc); the current
    device must be ``x``'s. With ``defer``, a K-split shape of at most ``BM`` rows returns its
    unsummed ``KParts`` (no ``_ksum_kernel`` launch) for the consumer to sum."""
    rows, k = x.shape
    n = w.shape[0]
    assert w.shape[1] == k and x.stride(1) == 1 and x.stride(0) == k
    if out is None:
        out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    bn, bk, warps, ksplit = tiles or tiles_for(n, k)
    part = out
    if ksplit > 1:
        part = torch.empty((ksplit * BM * n,), dtype=torch.float32, device=x.device)
    if defer and ksplit > 1 and 0 < rows <= BM and out.is_contiguous():
        _mfma_gemv_kernel[(triton.cdiv(n, bn) * ksplit,)](
            x, w, out, part, rows, n, out.stride(0), K=k, BM=BM, BN=bn, BK=bk, KSPLIT=ksplit,
            DOT_F32=kernels.INTERPRET, num_warps=warps,
        )
        return KParts(part, out, ksplit)
    for r0 in range(0, rows, BM):
        r1 = min(rows, r0 + BM)
        xr, orr = x[r0:r1], out[r0:r1]
        _mfma_gemv_kernel[(triton.cdiv(n, bn) * ksplit,)](
            xr, w, orr, part, r1 - r0, n, orr.stride(0), K=k, BM=BM, BN=bn, BK=bk, KSPLIT=ksplit,
            DOT_F32=kernels.INTERPRET, num_warps=warps,
        )
        if ksplit > 1:
            m = (r1 - r0) * n
            _ksum_kernel[(triton.cdiv(m, 1024),)](part, orr, r1 - r0, n, orr.stride(0),
                                                  KSPLIT=ksplit, BM=BM, BLOCK=1024, num_warps=4)
    return out


def gemv_topk(x: torch.Tensor, w: torch.Tensor, idx: torch.Tensor, wt: torch.Tensor,
              experts: int, top_k: int) -> torch.Tensor | None:
    """The router|shared GEMV ``bf16(x @ w.T)`` (``gemv``'s tiles and bits) with each row's
    router top-k written to ``idx``/``wt`` (``fused._topk_kernel``'s values) by R extra
    programs of the K-sum kernel. None when the shape runs without a K split (no K-sum
    kernel): the caller runs the GEMV and top-k as before."""
    rows, k = x.shape
    n = w.shape[0]
    assert w.shape[1] == k and x.stride(1) == 1 and x.stride(0) == k
    bn, bk, warps, ksplit = tiles_for(n, k)
    if ksplit == 1 or experts & (experts - 1) or n < experts:
        return None
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    part = torch.empty((ksplit * BM * n,), dtype=torch.float32, device=x.device)
    block = 256
    sel = TOPK_SEL if (TOPK_SEL != 2 or top_k == 8) else 1
    for r0 in range(0, rows, BM):
        r1 = min(rows, r0 + BM)
        xr, orr = x[r0:r1], out[r0:r1]
        _mfma_gemv_kernel[(triton.cdiv(n, bn) * ksplit,)](
            xr, w, orr, part, r1 - r0, n, orr.stride(0), K=k, BM=BM, BN=bn, BK=bk, KSPLIT=ksplit,
            DOT_F32=kernels.INTERPRET, num_warps=warps,
        )
        _ksum_router_kernel[(triton.cdiv((r1 - r0) * n, block) + (r1 - r0),)](
            part, orr, idx[r0:r1], wt[r0:r1], r1 - r0, n, orr.stride(0), KSPLIT=ksplit,
            BM=BM, BLOCK=block, E=experts, TOPK=top_k, TOPKP=triton.next_power_of_2(top_k),
            SEL=sel, num_warps=1,
        )
    return out


def gdn_chain(fd, i: int, h: torch.Tensor, states: torch.Tensor, groups: int,
              defer: bool = False) -> torch.Tensor:
    """A chained verify step's GDN layer ``i`` (``FusedDecode``'s weights and state): rows
    ``j * n + s`` are slot s's j-th token, ``states`` each row's ring state row."""
    fast = fd.fast
    m = fast.layers[i].linear_attn
    conv_state, rec_state = fast.gdn_state[i]
    assert conv_state.shape[-1] == 4 and conv_state.stride(1) == 4 and conv_state.stride(2) == 1
    conv_dim = 2 * m.key_dim + m.value_dim
    proj = fd._gemv(h, fd.w_in[i])  # raw projections; the convolution runs per slot below
    core = torch.empty((h.shape[0], m.num_v_heads * m.head_v_dim), dtype=h.dtype, device=h.device)
    n = h.shape[0] // groups
    block = 256
    _gdn_conv_chain_kernel[(n, triton.cdiv(conv_dim, block))](
        conv_state, states, proj, m.conv1d.weight, n, conv_dim, proj.stride(0),
        conv_state.stride(0), G=groups, BLOCK=block,
    )
    heads, dk, dv = m.num_v_heads, m.head_k_dim, m.head_v_dim
    z_off = 2 * m.key_dim + m.value_dim
    b_off = z_off + m.in_proj_z.weight.shape[0]
    a_off = b_off + m.in_proj_b.weight.shape[0]
    _gdn_chain_step_kernel[(n, heads)](
        rec_state, states, proj, m.A_log, m.dt_bias, m.norm.weight, core, n,
        proj.stride(0), heads, rec_state.stride(0), rec_state.stride(1),
        m.norm.variance_epsilon, dk ** -0.5, z_off, b_off, a_off,
        RATIO=heads // m.num_k_heads, KEY_DIM=m.key_dim, DK=dk, DV=dv, G=groups, num_warps=8,
    )
    return fd._gemv(core, m.out_proj.weight, defer=defer)


def groups_of(verify) -> int:
    """Rows per slot of a step: 1 (decode), 2 (``verify=True``) or the chain's G."""
    if verify is True:
        return 2
    return int(verify) if verify else 1
