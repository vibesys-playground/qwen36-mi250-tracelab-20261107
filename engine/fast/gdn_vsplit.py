"""Plan 48 (step 0, not on the served path): the chained verify's Gated DeltaNet recurrence
split over value columns.

Today (``mfma.gdn_chain``) a chained verify step's GDN layer runs two kernels: the short
convolution of every slot's G rows (``_gdn_conv_chain_kernel``, in place on the projection)
and the delta rule, one program per (slot, v-head) holding the 128 x 128 fp32 state and
walking the G rows in turn (``_gdn_chain_step_kernel``): 32 programs at 2 slots.

In the delta rule every value column evolves alone (s[:, c] *= exp(g); kv = k . s[:, c];
s[:, c] += k (v_c - kv) beta; o_c = q . s[:, c]); only the gated RMSNorm over DV couples the
columns. ``_vsplit_kernel`` runs program (slot, v-head, column block of BV) on a 128 x BV
state slice, with every row's inputs loaded and convolved (kernel-4 conv, the same per-channel
math and order as ``_gdn_conv_chain_kernel``) before the first state update:

- its own BV value channels' conv windows are written to the conv ring here;
- the q/k channels are read by every program of a k-head, and ring entry 0 is the row they
  read (``slots[r]``), so their ring rows are written by the next kernel in stream order
  (``_qk_ring``: one program per (slot, 128-channel block));
- the norm: ``NORM_IN`` (only BV == DV) runs today's tail in the kernel; otherwise the
  pre-norm o (bf16-rounded, stored as fp32) goes to a [rows, heads, DV] buffer and either
  ``_norm_qk_kernel`` (N1: today's tail per (row, head), plus the q/k ring) or the out_proj
  GEMV's x-load (N2: ``_gemv_gnorm_kernel``, each 128-wide K chunk is one head) applies it.

``_stream_kernel`` is the state-streaming ceiling at a launch shape: the state slice read once
and written to the G ring rows, no math. Used by ``tools/gdn_vsplit_bench.py`` only.
"""

from __future__ import annotations

import torch

from engine.fast import kernels, mfma

triton, tl = kernels.triton, kernels.tl

if triton is not None:

    @triton.jit
    def _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, OFF: tl.constexpr):
        """[G, C] values E[j + OFF] of the sequence E = (window s1, s2, s3 of conv state row
        ``first``, then the raw projection rows x_0, x_1, ... of slot r): row j's conv window
        is E[j], E[j+1], E[j+2], E[j+3]."""
        t = jj[:, None] + OFF + 0 * ch[None, :]
        st = t < 3
        sv = tl.load(conv_ptr + first * stride_cslot + ch[None, :] * 4 + (t + 1), mask=st, other=0.0)
        xv = tl.load(in_ptr + ((t - 3) * N + r) * stride_in + ch[None, :], mask=t >= 3, other=0.0)
        return tl.where(st, sv, xv)

    @triton.jit
    def _ring_store(conv_ptr, dst, stride_cslot, ch, e0, e1, e2, x):
        b = conv_ptr + dst[:, None] * stride_cslot + ch[None, :] * 4
        tl.store(b, e0)
        tl.store(b + 1, e1)
        tl.store(b + 2, e2)
        tl.store(b + 3, x)

    @triton.jit
    def _conv_rows(conv_ptr, in_ptr, cw_ptr, first, stride_cslot, r, N, stride_in, ch, jj,
                   G: tl.constexpr, CW: tl.constexpr):
        """The G rows' conv outputs (bf16, [G, CW]) of channels ``ch``, with
        ``_gdn_conv_chain_kernel``'s per-channel math and order, and the windows."""
        e0 = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 0)
        e1 = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 1)
        e2 = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 2)
        x = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 3)
        w0 = tl.load(cw_ptr + ch * 4).to(tl.float32)
        w1 = tl.load(cw_ptr + ch * 4 + 1).to(tl.float32)
        w2 = tl.load(cw_ptr + ch * 4 + 2).to(tl.float32)
        w3 = tl.load(cw_ptr + ch * 4 + 3).to(tl.float32)
        acc = tl.zeros([G, CW], dtype=tl.float32)
        acc += e0.to(tl.float32) * w0[None, :]
        acc += e1.to(tl.float32) * w1[None, :]
        acc += e2.to(tl.float32) * w2[None, :]
        acc += x.to(tl.float32) * w3[None, :]
        y = acc.to(tl.bfloat16).to(tl.float32)
        y = (y * tl.sigmoid(y)).to(tl.bfloat16)
        return y, e0, e1, e2, x

    @triton.jit
    def _pick(x, jj, j):
        """Row j of a [G, C] tensor (exact: the other rows add +0)."""
        return tl.sum(tl.where((jj == j)[:, None], x, 0.0), axis=0)

    @triton.jit
    def _vsplit_kernel(
        conv_ptr, rec_ptr, slot_ptr, in_ptr, cw_ptr, alog_ptr, dtb_ptr, nw_ptr, out_ptr,
        N, stride_in, stride_cslot, stride_slot, stride_head, V_HEADS, eps, scale,
        Z_OFF, B_OFF, A_OFF,
        RATIO: tl.constexpr, KEY_DIM: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
        BV: tl.constexpr, G: tl.constexpr, NORM_IN: tl.constexpr,
    ):
        r = tl.program_id(0)
        head = tl.program_id(1)
        vb = tl.program_id(2)
        k_head = head // RATIO
        jj = tl.arange(0, G)
        kk = tl.arange(0, DK)
        cc = vb * BV + tl.arange(0, BV)
        first = tl.load(slot_ptr + r).to(tl.int64)
        dst = tl.load(slot_ptr + jj * N + r).to(tl.int64)
        rows = jj * N + r
        tile = head * stride_head + kk[:, None] * DV + cc[None, :]
        s = tl.load(rec_ptr + first * stride_slot + tile)
        yq, _q0, _q1, _q2, _q3 = _conv_rows(conv_ptr, in_ptr, cw_ptr, first, stride_cslot, r, N,
                                            stride_in, k_head * DK + kk, jj, G, DK)
        yk, _k0, _k1, _k2, _k3 = _conv_rows(conv_ptr, in_ptr, cw_ptr, first, stride_cslot, r, N,
                                            stride_in, KEY_DIM + k_head * DK + kk, jj, G, DK)
        vch = 2 * KEY_DIM + head * DV + cc
        yv, e0, e1, e2, xv = _conv_rows(conv_ptr, in_ptr, cw_ptr, first, stride_cslot, r, N,
                                        stride_in, vch, jj, G, BV)
        bb = tl.load(in_ptr + rows * stride_in + B_OFF + head).to(tl.float32)
        aa = tl.load(in_ptr + rows * stride_in + A_OFF + head).to(tl.float32)
        alog = tl.load(alog_ptr + head).to(tl.float32)
        dtb = tl.load(dtb_ptr + head).to(tl.float32)
        if NORM_IN:
            zall = tl.load(in_ptr + rows[:, None] * stride_in + Z_OFF + head * DV + cc[None, :]).to(tl.float32)
            nw = tl.load(nw_ptr + cc).to(tl.float32)
        else:
            zall = tl.zeros([G, BV], dtype=tl.float32)
            nw = tl.zeros([BV], dtype=tl.float32)
        tl.debug_barrier()  # every load of conv state row ``first`` before the ring stores
        _ring_store(conv_ptr, dst, stride_cslot, vch, e0, e1, e2, xv)
        yqf = yq.to(tl.float32)
        ykf = yk.to(tl.float32)
        yvf = yv.to(tl.float32)
        for j in tl.static_range(G):
            q = _pick(yqf, jj, j)
            k = _pick(ykf, jj, j)
            v = _pick(yvf, jj, j)
            q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * scale
            k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
            beta = tl.sigmoid(tl.sum(tl.where(jj == j, bb, 0.0), axis=0)).to(tl.bfloat16).to(tl.float32)
            a = tl.sum(tl.where(jj == j, aa, 0.0), axis=0) + dtb
            softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
            g = -tl.exp(alog) * softplus
            s = s * tl.exp(g)
            kv_mem = tl.sum(s * k[:, None], axis=0)
            delta = (v - kv_mem) * beta
            s = s + k[:, None] * delta[None, :]
            o = tl.sum(s * q[:, None], axis=0)
            o = o.to(tl.bfloat16).to(tl.float32)
            d = tl.sum(tl.where(jj == j, dst, 0), axis=0)
            tl.store(rec_ptr + d * stride_slot + tile, s)
            row = j * N + r
            if NORM_IN:
                n = (o * tl.rsqrt(tl.sum(o * o, axis=0) / DV + eps)).to(tl.bfloat16).to(tl.float32)
                y = (nw * n).to(tl.bfloat16).to(tl.float32)
                zz = _pick(zall, jj, j)
                tl.store(out_ptr + (row * V_HEADS + head) * DV + cc, (y * zz * tl.sigmoid(zz)).to(tl.bfloat16))
            else:
                tl.store(out_ptr + (row * V_HEADS + head) * DV + cc, o)

    @triton.jit
    def _qk_ring(conv_ptr, in_ptr, slot_ptr, r, blk, N, stride_in, stride_cslot,
                 G: tl.constexpr, CB: tl.constexpr):
        """Conv ring rows of slot r's G rows for the q|k channels ``blk * CB ...``."""
        jj = tl.arange(0, G)
        ch = blk * CB + tl.arange(0, CB)
        first = tl.load(slot_ptr + r).to(tl.int64)
        dst = tl.load(slot_ptr + jj * N + r).to(tl.int64)
        e0 = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 0)
        e1 = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 1)
        e2 = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 2)
        x = _ext(conv_ptr, in_ptr, first, stride_cslot, r, N, stride_in, ch, jj, 3)
        tl.debug_barrier()
        _ring_store(conv_ptr, dst, stride_cslot, ch, e0, e1, e2, x)

    @triton.jit
    def _norm_qk_kernel(
        obuf_ptr, in_ptr, nw_ptr, core_ptr, conv_ptr, slot_ptr, N, stride_in, stride_cslot,
        V_HEADS, eps, Z_OFF, DK: tl.constexpr, DV: tl.constexpr, G: tl.constexpr,
        CB: tl.constexpr, DO_NORM: tl.constexpr, EXACT: tl.constexpr,
    ):
        """N1: program (row, head) applies today's gated RMSNorm tail to the pre-norm o; the
        programs of rows < N (slot r = row) also write the q|k conv ring of block ``head``."""
        p = tl.program_id(0)
        head = tl.program_id(1)
        if DO_NORM:
            vv = tl.arange(0, DV)
            base = obuf_ptr + (p * V_HEADS + head) * DV
            if EXACT:  # o in the layout of a [DK, DV] axis-0 sum, as in the step kernel
                kk = tl.arange(0, DK)
                t = tl.load(base + kk[:, None] * 0 + vv[None, :])
                o = tl.sum(tl.where(kk[:, None] == 0, t, 0.0), axis=0)
            else:
                o = tl.load(base + vv)
            n = (o * tl.rsqrt(tl.sum(o * o, axis=0) / DV + eps)).to(tl.bfloat16).to(tl.float32)
            y = (tl.load(nw_ptr + vv).to(tl.float32) * n).to(tl.bfloat16).to(tl.float32)
            zz = tl.load(in_ptr + p * stride_in + Z_OFF + head * DV + vv).to(tl.float32)
            tl.store(core_ptr + (p * V_HEADS + head) * DV + vv, (y * zz * tl.sigmoid(zz)).to(tl.bfloat16))
        if p < N:
            _qk_ring(conv_ptr, in_ptr, slot_ptr, p, head, N, stride_in, stride_cslot, G, CB)

    @triton.jit(do_not_specialize=["R", "N"])
    def _gemv_gnorm_kernel(
        o_ptr, in_ptr, nw_ptr, w_ptr, out_ptr, part_ptr, R, N, stride_out, stride_in, Z_OFF, eps,
        conv_ptr, slot_ptr, NS, stride_cslot, V_HEADS,
        K: tl.constexpr, DV: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        KSPLIT: tl.constexpr, G: tl.constexpr, CB: tl.constexpr,
    ):
        """N2: ``mfma._mfma_gemv_kernel`` with the gated RMSNorm in its x load (BK == DV: a K
        chunk is one head), x the pre-norm o (fp32); NS * V_HEADS extra programs write the q|k
        conv ring (``_qk_ring``)."""
        pid = tl.program_id(0)
        nmain = tl.cdiv(N, BN) * KSPLIT
        KP: tl.constexpr = K // KSPLIT
        if pid < nmain:
            nb = pid // KSPLIT
            kp = pid % KSPLIT
            rm = tl.arange(0, BM)
            mok = rm < R
            rn = nb * BN + tl.arange(0, BN)
            nok = rn < N
            vv = tl.arange(0, BK)
            nw = tl.load(nw_ptr + vv).to(tl.float32)
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k0 in range(kp * KP, kp * KP + KP, BK):
                rk = k0 + vv
                o = tl.load(o_ptr + rm[:, None] * K + rk[None, :], mask=mok[:, None], other=0.0)
                n = (o * tl.rsqrt(tl.sum(o * o, axis=1) / DV + eps)[:, None]).to(tl.bfloat16).to(tl.float32)
                y = (nw[None, :] * n).to(tl.bfloat16).to(tl.float32)
                zz = tl.load(in_ptr + rm[:, None] * stride_in + Z_OFF + rk[None, :], mask=mok[:, None],
                             other=0.0).to(tl.float32)
                a = (y * zz * tl.sigmoid(zz)).to(tl.bfloat16)
                b = tl.load(w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None], mask=nok[None, :], other=0.0)
                acc = tl.dot(a, b, acc)
            m = mok[:, None] & nok[None, :]
            if KSPLIT == 1:
                tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], acc.to(tl.bfloat16), mask=m)
            else:
                tl.store(part_ptr + (kp * BM + rm[:, None]) * N + rn[None, :], acc, mask=m)
        else:
            q = pid - nmain
            _qk_ring(conv_ptr, in_ptr, slot_ptr, q // V_HEADS, q % V_HEADS, NS, stride_in,
                     stride_cslot, G, CB)

    @triton.jit
    def _stream_kernel(rec_ptr, slot_ptr, N, stride_slot, stride_head, DK: tl.constexpr,
                       DV: tl.constexpr, BV: tl.constexpr, G: tl.constexpr):
        """Ceiling: the state slice read once and written to the G ring rows, no math."""
        r = tl.program_id(0)
        head = tl.program_id(1)
        vb = tl.program_id(2)
        kk = tl.arange(0, DK)
        cc = vb * BV + tl.arange(0, BV)
        tile = head * stride_head + kk[:, None] * DV + cc[None, :]
        first = tl.load(slot_ptr + r).to(tl.int64)
        s = tl.load(rec_ptr + first * stride_slot + tile)
        for j in tl.static_range(G):
            d = tl.load(slot_ptr + j * N + r).to(tl.int64)
            tl.store(rec_ptr + d * stride_slot + tile, s)

    @triton.jit
    def _empty_kernel(x_ptr):
        pass


def _dims(L):
    heads, dk, dv = L.num_v_heads, L.head_k_dim, L.head_v_dim
    z_off = 2 * L.key_dim + L.value_dim
    b_off = z_off + L.value_dim
    a_off = b_off + heads
    return heads, dk, dv, z_off, b_off, a_off


def today(L, proj, states, core, n: int, groups: int) -> torch.Tensor:
    """``mfma.gdn_chain``'s two kernels (conv in place on ``proj``, then the step)."""
    conv_dim = 2 * L.key_dim + L.value_dim
    block = 256
    mfma._gdn_conv_chain_kernel[(n, triton.cdiv(conv_dim, block))](
        L.conv, states, proj, L.conv_w, n, conv_dim, proj.stride(0), L.conv.stride(0),
        G=groups, BLOCK=block,
    )
    heads, dk, dv, z_off, b_off, a_off = _dims(L)
    mfma._gdn_chain_step_kernel[(n, heads)](
        L.rec, states, proj, L.A_log, L.dt_bias, L.norm_w, core, n,
        proj.stride(0), heads, L.rec.stride(0), L.rec.stride(1),
        L.eps, dk ** -0.5, z_off, b_off, a_off,
        RATIO=heads // L.num_k_heads, KEY_DIM=L.key_dim, DK=dk, DV=dv, G=groups, num_warps=8,
    )
    return core


def vsplit(L, proj, states, out, n: int, groups: int, bv: int, warps: int, norm_in: bool) -> None:
    """``_vsplit_kernel``: ``out`` is the bf16 core when ``norm_in`` (bv == DV), else the fp32
    pre-norm o [rows, heads, DV]. The q/k conv ring is left to ``norm_qk`` / ``out_gnorm``."""
    heads, dk, dv, z_off, b_off, a_off = _dims(L)
    assert dv % bv == 0 and (not norm_in or bv == dv)
    _vsplit_kernel[(n, heads, dv // bv)](
        L.conv, L.rec, states, proj, L.conv_w, L.A_log, L.dt_bias, L.norm_w, out,
        n, proj.stride(0), L.conv.stride(0), L.rec.stride(0), L.rec.stride(1), heads,
        L.eps, dk ** -0.5, z_off, b_off, a_off,
        RATIO=heads // L.num_k_heads, KEY_DIM=L.key_dim, DK=dk, DV=dv, BV=bv, G=groups,
        NORM_IN=norm_in, num_warps=warps,
    )


def norm_qk(L, proj, states, obuf, core, n: int, groups: int, do_norm: bool, exact: bool = True,
            warps: int = 8) -> None:
    heads, dk, dv, z_off, _, _ = _dims(L)
    cb = 2 * L.key_dim // heads
    _norm_qk_kernel[(groups * n if do_norm else n, heads)](
        obuf, proj, L.norm_w, core, L.conv, states, n, proj.stride(0), L.conv.stride(0),
        heads, L.eps, z_off, DK=dk, DV=dv, G=groups, CB=cb, DO_NORM=do_norm, EXACT=exact,
        num_warps=warps,
    )


def out_gnorm(L, obuf, proj, states, w: torch.Tensor, n: int, groups: int) -> torch.Tensor:
    """N2: out_proj GEMV (``mfma.gemv``'s tiles, BK = DV) with the norm in its x load, plus
    the q/k conv ring; returns ``bf16(norm(o) @ w.T)``."""
    heads, dk, dv, z_off, _, _ = _dims(L)
    rows = obuf.shape[0]
    k = heads * dv
    nout = w.shape[0]
    assert rows <= mfma.BM and w.shape[1] == k
    bn, bk, warps, ksplit = mfma.tiles_for(nout, k)
    assert bk == dv
    out = torch.empty((rows, nout), dtype=w.dtype, device=w.device)
    part = out
    if ksplit > 1:
        part = torch.empty((ksplit * mfma.BM * nout,), dtype=torch.float32, device=w.device)
    nmain = triton.cdiv(nout, bn) * ksplit
    _gemv_gnorm_kernel[(nmain + n * heads,)](
        obuf, proj, L.norm_w, w, out, part, rows, nout, out.stride(0), proj.stride(0), z_off, L.eps,
        L.conv, states, n, L.conv.stride(0), heads,
        K=k, DV=dv, BM=mfma.BM, BN=bn, BK=bk, KSPLIT=ksplit, G=groups, CB=2 * L.key_dim // heads,
        num_warps=warps,
    )
    if ksplit > 1:
        m = rows * nout
        mfma._ksum_kernel[(triton.cdiv(m, 1024),)](part, out, rows, nout, out.stride(0),
                                                   KSPLIT=ksplit, BM=mfma.BM, BLOCK=1024, num_warps=4)
    return out


def stream(L, states, n: int, groups: int, bv: int, warps: int) -> None:
    heads, dk, dv = L.num_v_heads, L.head_k_dim, L.head_v_dim
    _stream_kernel[(n, heads, dv // bv)](L.rec, states, n, L.rec.stride(0), L.rec.stride(1),
                                         DK=dk, DV=dv, BV=bv, G=groups, num_warps=warps)


def empty(x: torch.Tensor) -> None:
    _empty_kernel[(1,)](x, num_warps=1)
