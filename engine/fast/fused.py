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
# Tiles: GEMV outputs per tile, K chunk for up to 4 rows and for 5-8 rows; expert rows per tile.
# From a sweep on one gfx90a GCD (j71): 4 x 2048 beat 8 x 1024 and 16 x 512 at 4 and 8 rows.
TILES = {"bn": 4, "bk4": 2048, "bk8": 1024, "moe_bn": 8}


def enabled() -> bool:
    return os.environ.get("ENGINE_FUSED_DECODE", "1") == "1"


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
        bs = tl.arange(0, BP)
        bok = bs < R
        for t in range(pid, tl.cdiv(N, BN), nprog):
            rows = t * BN + tl.arange(0, BN)
            rok = rows < N
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
        for i, layer in enumerate(fast.layers):
            if fast.layer_types[i] == "linear_attention":
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

    # ---------------------------------------------------------------- pieces

    def _gemv(self, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None,
              conv: tuple | None = None) -> torch.Tensor:
        rows, k = x.shape
        n = w.shape[0]
        assert w.shape[1] == k and x.stride(1) == 1 and x.stride(0) == k
        if out is None:
            out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
        bp = max(2, _pow2(rows))
        bk = min(TILES["bk4"] if bp <= 4 else TILES["bk8"], _pow2(k))
        bn = TILES["bn"]
        grid = min(triton.cdiv(n, bn), 2 * self.cus[x.device])
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

    def _moe(self, i: int, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x + MoE(h) for layer ``i``."""
        fast = self.fast
        mlp = fast.layers[i].mlp
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
        cus = 2 * self.cus[h.device]
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
        fast = self.fast
        m = fast.layers[i].linear_attn
        conv_state, rec_state = fast.gdn_state[i]
        w = self.w_in[i]
        conv_dim = 2 * m.key_dim + m.value_dim
        proj = self._gemv(h, w, conv=(conv_state, slots, m.conv1d.weight, conv_dim))
        rows = h.shape[0]
        heads, dk, dv = m.num_v_heads, m.head_k_dim, m.head_v_dim
        core = torch.empty((rows, heads * dv), dtype=h.dtype, device=h.device)
        z_off = conv_dim
        b_off = z_off + m.in_proj_z.weight.shape[0]
        a_off = b_off + m.in_proj_b.weight.shape[0]
        _gdn_step_kernel[(rows, heads)](
            rec_state, slots, proj, m.A_log, m.dt_bias, m.norm.weight, core,
            proj.stride(0), heads, rec_state.stride(0), rec_state.stride(1),
            m.norm.variance_epsilon, dk ** -0.5, z_off, b_off, a_off,
            RATIO=heads // m.num_k_heads, KEY_DIM=m.key_dim, DK=dk, DV=dv, num_warps=8,
        )
        return self._gemv(core, m.out_proj.weight)

    def _attn(self, i: int, h: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor,
              meta) -> torch.Tensor:
        fast = self.fast
        m = fast.layers[i].self_attn
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
              meta) -> torch.Tensor:
        fast = self.fast
        layer = fast.layers[i]
        with kernels.on_device(x):
            _, h = kernels.add_rms(x, None, layer.input_layernorm.weight, fast.eps)
            if fast.layer_types[i] == "linear_attention":
                h = self._gdn(i, h, slots)
            else:
                h = self._attn(i, h, slots, positions, meta)
            x, h = kernels.add_rms(x, h, layer.post_attention_layernorm.weight, fast.eps)
            return self._moe(i, x, h)
