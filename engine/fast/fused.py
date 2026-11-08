"""Fused decode layers (plan 11): the same math as ``FastModel.layer_decode`` in fewer launches.

A decode layer of the per-op path issues about 29 kernels (GDN) or 45 (attention), most of
them tiny: four separate input projections, a 1-row ``shared_expert_gate`` GEMV, softmax and
top-k sorts, rotary elementwise ops, the split-K attention combine in PyTorch. Each costs
10-50 us on gfx90a whatever its size. Here a layer is:

GDN layer (9 launches)
    add_rms -> GEMV of ``in_proj_qkv|z|b|a`` with the short convolution in its epilogue
    -> delta-rule step + gated norm -> ``out_proj`` GEMV -> add_rms -> MoE (4 launches)
Attention layer (11 launches)
    add_rms -> GEMV of ``q|k|v`` -> q/k RMSNorm + rotary + KV-cache write -> paged attention
    (``kernels._paged_attn_kernel``) -> split combine + sigmoid gate -> ``o_proj`` GEMV
    -> add_rms -> MoE (4 launches)
MoE (4 launches)
    GEMV of ``router|shared gate|shared up|shared_expert_gate`` -> softmax top-k per row
    -> routed gate_up -> routed down + combine + shared expert +
    its sigmoid gate + the residual add.

Every rounding point of the per-op path is kept (BF16 after each projection, activation and
product, fp32 softmax over BF16 logits, renormalized top-k weights rounded to BF16), so the
outputs equal the per-op path's up to fp32 summation order. The concatenated weights are one
tensor per layer, and the module's own parameters become views of it, so no memory is added.

Rows are a runtime count (``x.shape[0]``, at most ``MAX_ROWS``), one per sequence as in the
per-op path: the recurrent/conv state and KV writes follow ``slots``/``positions`` row by
row exactly as there. Rotary cos/sin come from a per-device table built with the same
PyTorch ops as ``FastModel._qkv``.
"""

from __future__ import annotations

import os

import torch

from engine.fast import kernels

triton, tl = kernels.triton, kernels.tl

MAX_ROWS = int(os.environ.get("ENGINE_FUSED_MAX_ROWS", "8"))
# Row tiling of the fused GEMVs (ENGINE_FUSED_ROW_TILE):
# - "pairs" (default): rows run in blocks of 2 with the B=1 tiles (4 x 2048), a block per
#   work item, so a row's sum order never depends on how many rows share the launch;
# - "fixed": one block of pow2(MAX_ROWS) rows with 4 x 1024 tiles (also row-count invariant);
# - "rows": the block covers all rows and its tiles follow the row count (plan 11's
#   original; a verify step then sums a slot's first token differently from a decode step).
ROW_TILE = os.environ.get("ENGINE_FUSED_ROW_TILE", "pairs")
# The final norm and lm_head of a fused step through the per-row Triton RMSNorm and the fused
# GEMV (ENGINE_FUSED_HEAD=1), so a slot's logits are the same in decode and verify steps; its
# own tiles (ENGINE_FUSED_HEAD_TILE="bn,bk") and grid (programs per CU, ENGINE_FUSED_HEAD_GRID).
HEAD = os.environ.get("ENGINE_FUSED_HEAD", "1") == "1"
HEAD_TILE = tuple(int(v) for v in os.environ.get("ENGINE_FUSED_HEAD_TILE", "4,2048").split(","))
HEAD_GRID = int(os.environ.get("ENGINE_FUSED_HEAD_GRID", "2"))
GRID = int(os.environ.get("ENGINE_FUSED_GRID", "2"))
# Tiles: GEMV outputs per tile, K chunk for up to 4 rows and for 5-8 rows; expert rows per tile.
# From a sweep on one gfx90a GCD (j71): 4 x 2048 beat 8 x 1024 and 16 x 512 at 4 and 8 rows.
TILES = {"bn": 4, "bk4": 2048, "bk8": 1024, "moe_bn": 8}
# The multi-row GEMV (plan 19, ENGINE_FUSED_GEMV=1, default): each weight chunk is loaded once
# for every row of the launch, each row has its own per-lane fp32 accumulator of one fixed
# shape, reduced once per tile, so a row's sum never depends on the row count. =0 keeps the
# row-pair kernel above (``_gemv_kernel``) for the lm_head and every fused GEMV.
GEMV = os.environ.get("ENGINE_FUSED_GEMV", "1") == "1"
# Rows per launch of the multi-row GEMV (more rows run as further launches of this many).
GEMV_MAX_ROWS = 8
# Tiles per weight shape (N, K): outputs per tile, K chunk, warps, K split (>1: fp32 partials
# summed in a fixed order by a second kernel), pipelining stages, and programs per CU by row
# count (0: one program per tile). The tiles must not depend on the row count (they fix the sum
# order); the grid may. From a sweep on one gfx90a GCD (j100, tools/gemv_bw_bench.py).
GEMV_TILES = {
    (12352, 2048): (32, 128, 4, 1, 2, {1: 2, 2: 2, 4: 8, 8: 0}),     # GDN in_proj (+ conv)
    (2048, 4096): (32, 128, 4, 4, 2, {1: 4, 2: 8, 4: 0, 8: 2}),      # out_proj, o_proj, MTP fc
    (9216, 2048): (32, 256, 8, 1, 1, {1: 2, 2: 2, 4: 2, 8: 2}),      # attention q|k|v
    (1281, 2048): (16, 256, 8, 1, 2, {1: 2, 2: 1, 4: 4, 8: 0}),      # router|shared
    (248320, 2048): (64, 128, 8, 1, 1, {1: 1, 2: 1, 4: 0, 8: 2}),    # lm_head
}
GEMV_DEFAULT = (32, 128, 4, 1, 2, {})


def enabled() -> bool:
    return os.environ.get("ENGINE_FUSED_DECODE", "1") == "1"


def verify_enabled() -> bool:
    """Verify steps (two rows per slot) and the MTP head through the fused layers too."""
    return os.environ.get("ENGINE_FUSED_VERIFY", "1") == "1"


if triton is not None:

    @triton.jit
    def _gemv_kernel(
        x_ptr, w_ptr, out_ptr, R, N, stride_out,
        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N,
        K: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        CONV: tl.constexpr, KC: tl.constexpr,
    ):
        """out[r, n] = bf16(W[n, :] . x[r, :]) for R rows; grid-stride over tiles of BN outputs.

        CONV: outputs n < CONV_N are the GDN conv inputs; the epilogue runs the short causal
        convolution step on the slot's state (shift, append, silu) as ``_gdn_conv_kernel``.
        """
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        for t in range(pid, tl.cdiv(N, BN) * tl.cdiv(R, BP), nprog):
            # Tile t: BN outputs of one block of BP rows. A row is always summed with the
            # same [BN, BP, BK] shape and K chunks, whichever block it falls in.
            rows = (t // tl.cdiv(R, BP)) * BN + tl.arange(0, BN)
            rok = rows < N
            bs = (t % tl.cdiv(R, BP)) * BP + tl.arange(0, BP)
            bok = bs < R
            acc = tl.zeros([BN, BP], dtype=tl.float32)
            for k0 in range(0, K, BK):
                cols = k0 + tl.arange(0, BK)
                cok = cols < K
                w = tl.load(w_ptr + rows[:, None].to(tl.int64) * K + cols[None, :],
                            mask=rok[:, None] & cok[None, :], other=0.0).to(tl.float32)
                x = tl.load(x_ptr + bs[:, None] * K + cols[None, :],
                            mask=bok[:, None] & cok[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(w[:, None, :] * x[None, :, :], axis=2)
            y = acc.to(tl.bfloat16)
            omask = rok[:, None] & bok[None, :]
            if CONV:
                crow = rok & (rows < CONV_N)
                cm = crow[:, None] & bok[None, :]
                slot = tl.load(slot_ptr + bs, mask=bok, other=0).to(tl.int64)
                sbase = state_ptr + slot[None, :] * stride_slot + rows[:, None] * KC
                cacc = tl.zeros([BN, BP], dtype=tl.float32)
                for j in tl.static_range(KC - 1):
                    sj = tl.load(sbase + j + 1, mask=cm, other=0.0)
                    wj = tl.load(cw_ptr + rows * KC + j, mask=crow, other=0.0).to(tl.float32)
                    cacc += sj.to(tl.float32) * wj[:, None]
                    tl.store(sbase + j, sj, mask=cm)
                wl = tl.load(cw_ptr + rows * KC + KC - 1, mask=crow, other=0.0).to(tl.float32)
                cacc += y.to(tl.float32) * wl[:, None]
                tl.store(sbase + KC - 1, y, mask=cm)
                c = cacc.to(tl.bfloat16).to(tl.float32)
                c = (c * tl.sigmoid(c)).to(tl.bfloat16)
                y = tl.where(cm, c, y)
            tl.store(out_ptr + bs[None, :] * stride_out + rows[:, None], y, mask=omask)

    @triton.jit
    def _mr_zero(BN: tl.constexpr, BK: tl.constexpr):
        return tl.zeros([BN, BK], dtype=tl.float32)

    @triton.jit
    def _mr_row(a, w, x_ptr, r: tl.constexpr, K: tl.constexpr, cols):
        xr = tl.load(x_ptr + r * K + cols).to(tl.float32)
        return a + w * xr[None, :]

    @triton.jit
    def _mr_put(a, r: tl.constexpr, n, nok, out_ptr, stride_out, part_ptr, kp, N,
                NR: tl.constexpr, KSPLIT: tl.constexpr, CONV: tl.constexpr,
                state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC: tl.constexpr):
        """Row ``r``'s outputs of a tile: one cross-lane reduction, then BF16 (with the conv
        epilogue of ``_gemv_kernel``) or, with a K split, the fp32 partial."""
        s = tl.sum(a, axis=1)
        if KSPLIT == 1:
            y = s.to(tl.bfloat16)
            if CONV:
                crow = nok & (n < CONV_N)
                slot = tl.load(slot_ptr + r).to(tl.int64)
                sbase = state_ptr + slot * stride_slot + n * KC
                cacc = tl.zeros_like(s)
                for j in tl.static_range(KC - 1):
                    sj = tl.load(sbase + j + 1, mask=crow, other=0.0)
                    wj = tl.load(cw_ptr + n * KC + j, mask=crow, other=0.0).to(tl.float32)
                    cacc += sj.to(tl.float32) * wj
                    tl.store(sbase + j, sj, mask=crow)
                wl = tl.load(cw_ptr + n * KC + KC - 1, mask=crow, other=0.0).to(tl.float32)
                cacc += y.to(tl.float32) * wl
                tl.store(sbase + KC - 1, y, mask=crow)
                c = cacc.to(tl.bfloat16).to(tl.float32)
                c = (c * tl.sigmoid(c)).to(tl.bfloat16)
                y = tl.where(crow, c, y)
            tl.store(out_ptr + r * stride_out + n, y, mask=nok)
        else:
            tl.store(part_ptr + (kp * NR + r) * N + n, s, mask=nok)

    @triton.jit
    def _mr_gemv_kernel(
        x_ptr, w_ptr, out_ptr, part_ptr, N, ntiles, stride_out,
        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N,
        K: tl.constexpr, NR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        KSPLIT: tl.constexpr, CONV: tl.constexpr, KC: tl.constexpr,
    ):
        """out[r, n] = bf16(W[n, :] . x[r, :]) for NR <= 8 rows (plan 19); grid-stride over
        tiles of BN outputs x one of KSPLIT K parts. Each weight chunk [BN, BK] is loaded once
        and multiplied into every row's own [BN, BK] fp32 accumulator; one reduction per row
        per tile. A row's arithmetic is the same for every NR. CONV as ``_gemv_kernel``."""
        KP: tl.constexpr = K // KSPLIT
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        for t in range(pid, ntiles, nprog):
            nb = t // KSPLIT
            kp = t % KSPLIT
            n = nb * BN + tl.arange(0, BN)
            nok = n < N
            wbase = w_ptr + n[:, None].to(tl.int64) * K
            a0 = _mr_zero(BN, BK)
            a1 = _mr_zero(BN, BK)
            a2 = _mr_zero(BN, BK)
            a3 = _mr_zero(BN, BK)
            a4 = _mr_zero(BN, BK)
            a5 = _mr_zero(BN, BK)
            a6 = _mr_zero(BN, BK)
            a7 = _mr_zero(BN, BK)
            for k0 in range(kp * KP, kp * KP + KP, BK):
                cols = k0 + tl.arange(0, BK)
                w = tl.load(wbase + cols[None, :], mask=nok[:, None], other=0.0).to(tl.float32)
                a0 = _mr_row(a0, w, x_ptr, 0, K, cols)
                if NR > 1:
                    a1 = _mr_row(a1, w, x_ptr, 1, K, cols)
                if NR > 2:
                    a2 = _mr_row(a2, w, x_ptr, 2, K, cols)
                if NR > 3:
                    a3 = _mr_row(a3, w, x_ptr, 3, K, cols)
                if NR > 4:
                    a4 = _mr_row(a4, w, x_ptr, 4, K, cols)
                if NR > 5:
                    a5 = _mr_row(a5, w, x_ptr, 5, K, cols)
                if NR > 6:
                    a6 = _mr_row(a6, w, x_ptr, 6, K, cols)
                if NR > 7:
                    a7 = _mr_row(a7, w, x_ptr, 7, K, cols)
            _mr_put(a0, 0, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                    state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)
            if NR > 1:
                _mr_put(a1, 1, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)
            if NR > 2:
                _mr_put(a2, 2, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)
            if NR > 3:
                _mr_put(a3, 3, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)
            if NR > 4:
                _mr_put(a4, 4, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)
            if NR > 5:
                _mr_put(a5, 5, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)
            if NR > 6:
                _mr_put(a6, 6, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)
            if NR > 7:
                _mr_put(a7, 7, n, nok, out_ptr, stride_out, part_ptr, kp, N, NR, KSPLIT, CONV,
                        state_ptr, slot_ptr, cw_ptr, stride_slot, CONV_N, KC)

    @triton.jit
    def _mr_ksum_kernel(part_ptr, out_ptr, N, stride_out, M, KSPLIT: tl.constexpr,
                        BLOCK: tl.constexpr):
        """out[r, n] = bf16(sum over the K parts p, in order, of part[p, r, n]); M = rows * N."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < M
        s = tl.load(part_ptr + i, mask=m, other=0.0)
        for p in tl.static_range(1, KSPLIT):
            s += tl.load(part_ptr + p * M + i, mask=m, other=0.0)
        tl.store(out_ptr + (i // N) * stride_out + i % N, s.to(tl.bfloat16), mask=m)

    @triton.jit
    def _gdn_step_kernel(
        state_ptr, slot_ptr, in_ptr, alog_ptr, dtb_ptr, nw_ptr, out_ptr,
        stride_in, V_HEADS, stride_slot, stride_head, eps, scale,
        Z_OFF, B_OFF, A_OFF,
        RATIO: tl.constexpr, KEY_DIM: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    ):
        """``_gdn_fused_kernel`` reading the convolved q|k|v and raw z, b, a from one row of the
        fused projection output (row stride ``stride_in``)."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        k_head = head // RATIO
        slot = tl.load(slot_ptr + row).to(tl.int64)
        kk = tl.arange(0, DK)
        vv = tl.arange(0, DV)
        base_in = in_ptr + row * stride_in
        q = tl.load(base_in + k_head * DK + kk).to(tl.float32)
        k = tl.load(base_in + KEY_DIM + k_head * DK + kk).to(tl.float32)
        v = tl.load(base_in + 2 * KEY_DIM + head * DV + vv).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        beta = tl.sigmoid(tl.load(base_in + B_OFF + head).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        a = tl.load(base_in + A_OFF + head).to(tl.float32) + tl.load(dtb_ptr + head).to(tl.float32)
        softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        g = -tl.exp(tl.load(alog_ptr + head).to(tl.float32)) * softplus
        sbase = state_ptr + slot * stride_slot + head * stride_head + kk[:, None] * DV + vv[None, :]
        s = tl.load(sbase) * tl.exp(g)
        kv_mem = tl.sum(s * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        s = s + k[:, None] * delta[None, :]
        o = tl.sum(s * q[:, None], axis=0)
        tl.store(sbase, s)
        o = o.to(tl.bfloat16).to(tl.float32)
        n = (o * tl.rsqrt(tl.sum(o * o, axis=0) / DV + eps)).to(tl.bfloat16).to(tl.float32)
        y = (tl.load(nw_ptr + vv).to(tl.float32) * n).to(tl.bfloat16).to(tl.float32)
        zz = tl.load(base_in + Z_OFF + head * DV + vv).to(tl.float32)
        tl.store(out_ptr + (row * V_HEADS + head) * DV + vv, (y * zz * tl.sigmoid(zz)).to(tl.bfloat16))

    @triton.jit
    def _attn_prep_kernel(
        qkv_ptr, q_ptr, k_pool, v_pool, table_ptr, slot_ptr, pos_ptr, len_ptr,
        cos_ptr, sin_ptr, qn_ptr, kn_ptr, q_eps, k_eps,
        stride_in, stride_table, stride_page, stride_head, stride_tok,
        QH: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
        PAGE: tl.constexpr,
    ):
        """One head of one row: q heads get RMSNorm + rotary (to ``q``), k heads RMSNorm +
        rotary, v heads nothing, both written to the slot's KV page; head 0 writes the
        attention length. ``FastModel._qkv`` / ``_attn_decode`` math and rounding."""
        row = tl.program_id(0)
        h = tl.program_id(1)
        pos = tl.load(pos_ptr + row)
        d = tl.arange(0, D)
        rot = d < 2 * HALF
        partner = tl.where(d < HALF, d + HALF, d - HALF)
        sign = tl.where(d < HALF, -1.0, 1.0)
        base = qkv_ptr + row * stride_in
        c = tl.load(cos_ptr + pos * HALF + (d % HALF), mask=rot, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + pos * HALF + (d % HALF), mask=rot, other=0.0).to(tl.float32)
        if h == 0:
            tl.store(len_ptr + row, (pos + 1).to(tl.int32))
        if h < QH + KVH:
            if h < QH:
                off = h * 2 * D
                nw_ptr = qn_ptr
                eps = q_eps
            else:
                off = QH * 2 * D + (h - QH) * D
                nw_ptr = kn_ptr
                eps = k_eps
            x = tl.load(base + off + d).to(tl.float32)
            xp = tl.load(base + off + partner).to(tl.float32)
            r = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
            y = (x * r * (1.0 + tl.load(nw_ptr + d).to(tl.float32))).to(tl.bfloat16).to(tl.float32)
            yp = (xp * r * (1.0 + tl.load(nw_ptr + partner).to(tl.float32))).to(tl.bfloat16).to(tl.float32)
            t1 = (y * c).to(tl.bfloat16).to(tl.float32)
            t2 = ((sign * yp) * s).to(tl.bfloat16).to(tl.float32)
            y = tl.where(rot, (t1 + t2), y).to(tl.bfloat16)
            if h < QH:
                tl.store(q_ptr + (row * QH + h) * D + d, y)
            else:
                slot = tl.load(slot_ptr + row).to(tl.int64)
                page = tl.load(table_ptr + slot * stride_table + pos // PAGE).to(tl.int64)
                dst = page * stride_page + (h - QH) * stride_head + (pos % PAGE) * stride_tok + d
                tl.store(k_pool + dst, y)
        else:
            v = tl.load(base + QH * 2 * D + KVH * D + (h - QH - KVH) * D + d)
            slot = tl.load(slot_ptr + row).to(tl.int64)
            page = tl.load(table_ptr + slot * stride_table + pos // PAGE).to(tl.int64)
            dst = page * stride_page + (h - QH - KVH) * stride_head + (pos % PAGE) * stride_tok + d
            tl.store(v_pool + dst, v)

    @triton.jit
    def _attn_combine_kernel(
        acc_ptr, ml_ptr, qkv_ptr, out_ptr, stride_in,
        QH: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
    ):
        """Merge the split-K partials as ``paged_decode_attention`` does, round to BF16, times
        BF16 sigmoid(gate) (``_attn_decode``)."""
        row = tl.program_id(0)
        h = tl.program_id(1)
        sp = tl.arange(0, SPLITS)
        d = tl.arange(0, D)
        rows = (row * QH + h) * SPLITS + sp
        m = tl.load(ml_ptr + rows * 2)
        l = tl.load(ml_ptr + rows * 2 + 1)
        top = tl.max(m, axis=0)
        w = tl.exp(m - top)
        acc = tl.load(acc_ptr + rows[:, None] * D + d[None, :])
        o = tl.sum(acc * w[:, None], axis=0) / tl.sum(l * w, axis=0)
        o = o.to(tl.bfloat16).to(tl.float32)
        g = tl.load(qkv_ptr + row * stride_in + h * 2 * D + D + d).to(tl.float32)
        g = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + row * (QH * D) + h * D + d, (o * g).to(tl.bfloat16))

    @triton.jit
    def _topk(logit_ptr, E: tl.constexpr, TOPK: tl.constexpr, TOPKP: tl.constexpr):
        """fp32 softmax over E BF16 logits, top-k, renormalized, rounded to BF16 (``_moe``)."""
        ee = tl.arange(0, E)
        p = tl.load(logit_ptr + ee).to(tl.float32)
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
    def _topk_kernel(r_ptr, idx_ptr, wt_ptr, N3, E: tl.constexpr, TOPK: tl.constexpr,
                     TOPKP: tl.constexpr):
        row = tl.program_id(0)
        sel_i, sel_w = _topk(r_ptr + row * N3, E, TOPK, TOPKP)
        kk = tl.arange(0, TOPKP)
        tl.store(idx_ptr + row * TOPK + kk, sel_i, mask=kk < TOPK)
        tl.store(wt_ptr + row * TOPK + kk, sel_w, mask=kk < TOPK)

    @triton.jit
    def _moe_up_kernel(
        h_ptr, idx_ptr, w_ptr, act_ptr, R, stride_we,
        H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """act[p, n] = bf16(silu(bf16(Wg[e_p, n] . h)) * bf16(Wu[e_p, n] . h)) over work items
        (pair p = row * TOPK + j, tile of BN rows); e_p from ``_topk_kernel``."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ntile = tl.cdiv(I, BN)
        for t in range(pid, R * TOPK * ntile, nprog):
            pair = t // ntile
            tile = t % ntile
            row = pair // TOPK
            expert = tl.load(idx_ptr + pair).to(tl.int64)
            rows = tile * BN + tl.arange(0, BN)
            rok = rows < I
            wbase = w_ptr + expert * stride_we
            acc_g = tl.zeros([BN], dtype=tl.float32)
            acc_u = tl.zeros([BN], dtype=tl.float32)
            for k0 in range(0, H, BK):
                cols = k0 + tl.arange(0, BK)
                cok = cols < H
                m = rok[:, None] & cok[None, :]
                xv = tl.load(h_ptr + row * H + cols, mask=cok, other=0.0).to(tl.float32)
                wg = tl.load(wbase + rows[:, None] * H + cols[None, :], mask=m, other=0.0).to(tl.float32)
                wu = tl.load(wbase + (rows[:, None] + I) * H + cols[None, :], mask=m, other=0.0).to(tl.float32)
                acc_g += tl.sum(wg * xv[None, :], axis=1)
                acc_u += tl.sum(wu * xv[None, :], axis=1)
            g = acc_g.to(tl.bfloat16).to(tl.float32)
            u = acc_u.to(tl.bfloat16).to(tl.float32)
            a = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
            tl.store(act_ptr + pair * I + rows, (a * u).to(tl.bfloat16), mask=rok)

    @triton.jit
    def _moe_down_kernel(
        act_ptr, r_ptr, idx_ptr, wt_ptr, w_ptr, sd_ptr, res_ptr, out_ptr, R, stride_we, N3,
        H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
        TOPK: tl.constexpr, BN: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr,
    ):
        """out = bf16(res + bf16(routed + bf16(bf16(sigmoid(seg)) * shared))): the routed down
        projection with its weighted combine (``_moe_down_kernel``), the shared expert's down
        projection of bf16(silu(gate) * up), its sigmoid gate and the layer's residual add."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ntile = tl.cdiv(H, BN)
        for t in range(pid, R * ntile, nprog):
            row = t // ntile
            rows = (t % ntile) * BN + tl.arange(0, BN)
            rok = rows < H
            rbase = r_ptr + row * N3
            acc = tl.zeros([BN], dtype=tl.float32)
            for j in tl.static_range(TOPK):
                pair = row * TOPK + j
                expert = tl.load(idx_ptr + pair).to(tl.int64)
                weight = tl.load(wt_ptr + pair)
                wbase = w_ptr + expert * stride_we
                part = tl.zeros([BN], dtype=tl.float32)
                for k0 in range(0, I, BKI):
                    cols = k0 + tl.arange(0, BKI)
                    cok = cols < I
                    hv = tl.load(act_ptr + pair * I + cols, mask=cok, other=0.0).to(tl.float32)
                    w = tl.load(wbase + rows[:, None] * I + cols[None, :],
                                mask=rok[:, None] & cok[None, :], other=0.0).to(tl.float32)
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


def _pow2(n: int) -> int:
    return triton.next_power_of_2(n)


def _pack(params: list[torch.nn.Parameter]) -> torch.Tensor:
    """One contiguous [sum(N), K] tensor; each parameter becomes a row-slice view of it."""
    with torch.no_grad():
        cat = torch.cat([p.data for p in params]).contiguous()
        start = 0
        for p in params:
            n = p.shape[0]
            p.data = cat[start : start + n]
            start += n
    return cat


class FusedDecode:
    """Per-layer packed weights and the fused decode of one layer (see the module doc)."""

    def __init__(self, fast) -> None:
        self.fast = fast
        self.max_rows = MAX_ROWS
        self.w_in: dict[int, torch.Tensor] = {}
        self.w_moe: dict[int, torch.Tensor] = {}
        self.rope: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cus: dict[torch.device, int] = {}
        indices = list(range(len(fast.layers)))
        if getattr(fast, "mtp", None) is not None:
            indices.append(fast.mtp_index)  # the MTP head's decoder layer (full attention)
        for i in indices:
            layer = self._layer(i)
            if self._is_gdn(i):
                m = layer.linear_attn
                self.w_in[i] = _pack([m.in_proj_qkv.weight, m.in_proj_z.weight,
                                      m.in_proj_b.weight, m.in_proj_a.weight])
            else:
                m = layer.self_attn
                self.w_in[i] = _pack([m.q_proj.weight, m.k_proj.weight, m.v_proj.weight])
                dev = m.q_proj.weight.device
                if dev not in self.rope:
                    inv = fast.inv_freq[dev]
                    freqs = torch.arange(fast.max_len, device=dev).float()[:, None] * inv[None, :]
                    self.rope[dev] = (freqs.cos().to(fast.embed.dtype).contiguous(),
                                      freqs.sin().to(fast.embed.dtype).contiguous())
            mlp = layer.mlp
            sh = mlp.shared_expert
            self.w_moe[i] = _pack([mlp.gate.weight, sh.gate_proj.weight, sh.up_proj.weight,
                                   mlp.shared_expert_gate.weight])
            ex = mlp.experts
            assert ex.gate_up_proj.stride(1) == ex.gate_up_proj.shape[2]
            assert ex.down_proj.stride(1) == ex.down_proj.shape[2]
            dev = layer.input_layernorm.weight.device
            if dev not in self.cus:
                self.cus[dev] = (torch.cuda.get_device_properties(dev).multi_processor_count
                                 if dev.type == "cuda" else 4)

    def _layer(self, i: int) -> torch.nn.Module:
        fast = self.fast
        if getattr(fast, "mtp", None) is not None and i == fast.mtp_index:
            return fast.mtp.layers[0]
        return fast.layers[i]

    def _cus(self, dev: torch.device) -> int:
        if dev not in self.cus:
            self.cus[dev] = (torch.cuda.get_device_properties(dev).multi_processor_count
                             if dev.type == "cuda" else 4)
        return self.cus[dev]

    def head(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """bf16(x @ w.T) for the lm_head: its own tiles and grid (``HEAD_TILE``,
        ``HEAD_GRID``), rows in blocks of 2 (or as ``ROW_TILE`` says for "fixed"/"rows")."""
        x = x.contiguous()
        if GEMV:
            return self._gemv(x, w)
        bp, _, _ = self._tiles(x.shape[0], x.shape[1])
        bn, bk = HEAD_TILE
        return self._gemv(x, w, tiles=(bp, bn, min(bk, _pow2(x.shape[1]))), grid_per_cu=HEAD_GRID)

    def _is_gdn(self, i: int) -> bool:
        return i < len(self.fast.layers) and self.fast.layer_types[i] == "linear_attention"

    # ---------------------------------------------------------------- pieces

    def _tiles(self, rows: int, k: int) -> tuple[int, int, int]:
        """(row block, outputs per tile, K chunk) of a GEMV (see ``ROW_TILE``)."""
        if ROW_TILE == "pairs":
            bp = 2
        elif ROW_TILE == "fixed":
            bp = max(2, _pow2(self.max_rows))
            assert rows <= bp, (rows, bp)
        else:
            bp = max(2, _pow2(rows))
        bk = min(TILES["bk4"] if bp <= 4 else TILES["bk8"], _pow2(k))
        return bp, TILES["bn"], bk

    def _gemv(self, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None,
              conv: tuple | None = None, tiles: tuple[int, int, int] | None = None,
              grid_per_cu: int = GRID) -> torch.Tensor:
        rows, k = x.shape
        n = w.shape[0]
        assert w.shape[1] == k and x.stride(1) == 1 and x.stride(0) == k
        if out is None:
            out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
        if GEMV and tiles is None:
            for r0 in range(0, rows, GEMV_MAX_ROWS):
                r1 = min(rows, r0 + GEMV_MAX_ROWS)
                cv = None if conv is None else (conv[0], conv[1][r0:r1], conv[2], conv[3])
                self._mr_gemv(x[r0:r1], w, out[r0:r1], cv)
            return out
        bp, bn, bk = tiles or self._tiles(rows, k)
        work = triton.cdiv(n, bn) * triton.cdiv(rows, bp)
        grid = min(work, grid_per_cu * self._cus(x.device))
        if conv is None:
            state, slots, cw, conv_n, kc = out, out, out, 0, 2
            stride_slot = 0
        else:
            state, slots, cw, conv_n = conv
            kc = state.shape[-1]
            stride_slot = state.stride(0)
        _gemv_kernel[(grid,)](
            x, w, out, rows, n, out.stride(0), state, slots, cw, stride_slot, conv_n,
            K=k, BP=bp, BN=bn, BK=bk, CONV=conv is not None, KC=kc, num_warps=4,
        )
        return out

    def _mr_gemv(self, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor,
                 conv: tuple | None) -> None:
        """``out = bf16(x @ w.T)`` through ``_mr_gemv_kernel`` for at most 8 rows."""
        rows, k = x.shape
        n = w.shape[0]
        bn, bk, warps, ksplit, stages, grids = GEMV_TILES.get((n, k), GEMV_DEFAULT)
        bk = min(bk, _pow2(k))
        if k % (ksplit * bk) != 0:  # a shape off the table: one K part, a chunk that divides K
            ksplit = 1
            while k % bk:
                bk //= 2
        assert conv is None or ksplit == 1, "the conv epilogue needs one K part"
        ntiles = triton.cdiv(n, bn) * ksplit
        per_cu = grids.get(min(_pow2(rows), GEMV_MAX_ROWS), 4)
        grid = ntiles if per_cu == 0 else min(ntiles, per_cu * self._cus(x.device))
        if conv is None:
            state, slots, cw, conv_n, kc, stride_slot = out, out, out, 0, 2, 0
        else:
            state, slots, cw, conv_n = conv
            kc, stride_slot = state.shape[-1], state.stride(0)
        part = out
        if ksplit > 1:
            part = torch.empty((ksplit * rows * n,), dtype=torch.float32, device=x.device)
        _mr_gemv_kernel[(grid,)](
            x, w, out, part, n, ntiles, out.stride(0), state, slots, cw, stride_slot, conv_n,
            K=k, NR=rows, BN=bn, BK=bk, KSPLIT=ksplit, CONV=conv is not None, KC=kc,
            num_warps=warps, num_stages=stages,
        )
        if ksplit > 1:
            m = rows * n
            _mr_ksum_kernel[(triton.cdiv(m, 1024),)](part, out, n, out.stride(0), m,
                                                     KSPLIT=ksplit, BLOCK=1024, num_warps=4)

    def gemv(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``bf16(x @ w.T)`` of a decode step's rows through the fused GEMV (the MTP head's fc
        and lm_head), on ``x``'s device."""
        with kernels.on_device(x):
            return self._gemv(x.contiguous(), w)

    def _moe(self, i: int, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x + MoE(h) for layer ``i``."""
        fast = self.fast
        mlp = self._layer(i).mlp
        rows, hidden = h.shape
        w3 = self.w_moe[i]
        r = self._gemv(h, w3)
        ex = mlp.experts
        gu, dn = ex.gate_up_proj, ex.down_proj
        experts, two_i, _ = gu.shape
        inter = two_i // 2
        sd = mlp.shared_expert.down_proj.weight
        shared = sd.shape[1]
        top_k = fast.top_k
        bn = TILES["moe_bn"]
        bk = min(1024, _pow2(hidden))
        cus = 2 * self._cus(h.device)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=h.device)
        wt = torch.empty((rows, top_k), dtype=torch.float32, device=h.device)
        _topk_kernel[(rows,)](r, idx, wt, w3.shape[0], E=experts, TOPK=top_k,
                              TOPKP=_pow2(top_k), num_warps=1)
        act = torch.empty((rows * top_k, inter), dtype=h.dtype, device=h.device)
        items = rows * top_k * triton.cdiv(inter, bn)
        _moe_up_kernel[(min(items, cus),)](
            h, idx, gu, act, rows, gu.stride(0),
            H=hidden, I=inter, TOPK=top_k, BN=bn, BK=bk, num_warps=4,
        )
        out = torch.empty_like(x)
        items = rows * triton.cdiv(hidden, bn)
        _moe_down_kernel[(min(items, cus),)](
            act, r, idx, wt, dn, sd, x, out, rows, dn.stride(0), w3.shape[0],
            H=hidden, I=inter, IS=shared, E=experts, TOPK=top_k, BN=bn,
            BKI=min(1024, _pow2(inter)), BKS=min(1024, _pow2(shared)), num_warps=4,
        )
        return out

    def _gdn(self, i: int, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """``slots`` are the rows' Gated DeltaNet state rows."""
        fast = self.fast
        m = fast.layers[i].linear_attn
        conv_state, rec_state = fast.gdn_state[i]
        conv_dim = 2 * m.key_dim + m.value_dim
        proj = self._gemv(h, self.w_in[i], conv=(conv_state, slots, m.conv1d.weight, conv_dim))
        core = torch.empty((h.shape[0], m.num_v_heads * m.head_v_dim), dtype=h.dtype, device=h.device)
        self._gdn_step(m, rec_state, slots, proj, core)
        return self._gemv(core, m.out_proj.weight)

    def _gdn_step(self, m, rec_state: torch.Tensor, slots: torch.Tensor, proj: torch.Tensor,
                  core: torch.Tensor) -> None:
        """The delta-rule step and gated norm of ``proj``'s rows (conv already applied to their
        q|k|v columns) on state rows ``slots``, into ``core``."""
        rows = proj.shape[0]
        heads, dk, dv = m.num_v_heads, m.head_k_dim, m.head_v_dim
        z_off = 2 * m.key_dim + m.value_dim
        b_off = z_off + m.in_proj_z.weight.shape[0]
        a_off = b_off + m.in_proj_b.weight.shape[0]
        _gdn_step_kernel[(rows, heads)](
            rec_state, slots, proj, m.A_log, m.dt_bias, m.norm.weight, core,
            proj.stride(0), heads, rec_state.stride(0), rec_state.stride(1),
            m.norm.variance_epsilon, dk ** -0.5, z_off, b_off, a_off,
            RATIO=heads // m.num_k_heads, KEY_DIM=m.key_dim, DK=dk, DV=dv, num_warps=8,
        )

    def _gdn_verify(self, i: int, h: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """A verify step's GDN layer: rows ``[:n]`` are each slot's first token on its current
        state row, rows ``[n:]`` its draft on the other bank's row (``FastModel._gdn_verify``).

        The projection GEMV and ``out_proj`` run once over all rows (weights read once); the
        short convolution (``kernels.gdn_conv_``, the same math as the GEMV's conv epilogue)
        and the delta-rule step run on the first half, the first half's states are copied to
        the second half's rows, then the second half runs there, so each draft row starts from
        the state after its slot's first token."""
        fast = self.fast
        m = fast.layers[i].linear_attn
        conv_state, rec_state = fast.gdn_state[i]
        conv_dim = 2 * m.key_dim + m.value_dim
        proj = self._gemv(h, self.w_in[i])  # raw projections; the conv runs per half below
        core = torch.empty((h.shape[0], m.num_v_heads * m.head_v_dim), dtype=h.dtype, device=h.device)
        n = h.shape[0] // 2
        halves = ((slice(0, n), states[:n]), (slice(n, 2 * n), states[n:]))
        for index, (rows, slots) in enumerate(halves):
            if index:
                first = states[:n]
                conv_state.index_copy_(0, slots, conv_state.index_select(0, first))
                rec_state.index_copy_(0, slots, rec_state.index_select(0, first))
            part = proj[rows]
            part[:, :conv_dim] = kernels.gdn_conv_(conv_state, slots, part[:, :conv_dim], m.conv1d.weight)
            self._gdn_step(m, rec_state, slots, part, core[rows])
        return self._gemv(core, m.out_proj.weight)

    def _attn(self, i: int, h: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor,
              meta) -> torch.Tensor:
        fast = self.fast
        m = self._layer(i).self_attn
        qkv = self._gemv(h, self.w_in[i])
        rows = h.shape[0]
        k_pool, v_pool = fast.kv[i]
        kv_heads, page, dim = k_pool.shape[1], k_pool.shape[2], k_pool.shape[3]
        q_heads = m.o_proj.weight.shape[1] // dim
        cos, sin = self.rope[h.device]
        q = torch.empty((rows, q_heads, dim), dtype=h.dtype, device=h.device)
        lengths = torch.empty((rows,), dtype=torch.int32, device=h.device)
        _attn_prep_kernel[(rows, q_heads + 2 * kv_heads)](
            qkv, q, k_pool, v_pool, meta.page_table, slots, positions, lengths, cos, sin,
            m.q_norm.weight, m.k_norm.weight, m.q_norm.eps, m.k_norm.eps,
            qkv.stride(0), meta.page_table.stride(0), k_pool.stride(0), k_pool.stride(1),
            k_pool.stride(2), QH=q_heads, KVH=kv_heads, D=dim, HALF=cos.shape[1], PAGE=page,
            num_warps=4,
        )
        splits = kernels.ATTN_SPLITS
        acc = torch.empty((rows, q_heads, splits, dim), dtype=torch.float32, device=h.device)
        ml = torch.empty((rows, q_heads, splits, 2), dtype=torch.float32, device=h.device)
        group = q_heads // kv_heads
        kernels._paged_attn_kernel[(rows, kv_heads, splits)](
            q, k_pool, v_pool, meta.page_table, slots, lengths, acc, ml, m.scaling, q_heads,
            meta.page_table.stride(0), k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            GROUP=group, GROUP_PAD=max(16, _pow2(group)), D=dim, PAGE=page, SPLITS=splits,
            DOT_F32=kernels.INTERPRET,
        )
        gated = torch.empty((rows, q_heads * dim), dtype=h.dtype, device=h.device)
        _attn_combine_kernel[(rows, q_heads)](
            acc, ml, qkv, gated, qkv.stride(0), QH=q_heads, D=dim, SPLITS=splits, num_warps=4,
        )
        return self._gemv(gated, m.o_proj.weight)

    # ---------------------------------------------------------------- one layer

    def layer(self, i: int, x: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor,
              meta, states: torch.Tensor | None = None, verify: bool = False) -> torch.Tensor:
        """Layer ``i`` (``FastModel.mtp_index``: the MTP head's) for one token per row.
        ``states`` are the rows' GDN state rows (default ``slots``); with ``verify`` the rows
        are a verify step's two halves (see ``_gdn_verify``)."""
        fast = self.fast
        layer = self._layer(i)
        states = slots if states is None else states
        with kernels.on_device(x):
            _, h = kernels.add_rms(x, None, layer.input_layernorm.weight, fast.eps)
            if self._is_gdn(i):
                h = self._gdn_verify(i, h, states) if verify else self._gdn(i, h, states)
            else:
                h = self._attn(i, h, slots, positions, meta)
            x, h = kernels.add_rms(x, h, layer.post_attention_layernorm.weight, fast.eps)
            return self._moe(i, x, h)
