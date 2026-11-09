"""Fused Gated DeltaNet prefill chain (plan 40 part B, ``ENGINE_GDN_CHUNK``).

One sequence of T rows of one layer on one tensor-parallel rank (HV value heads, H key heads,
HV % H == 0, key and value head dims K = V = 128 here), from the raw projection ``qkv``
[T, 2*H*K + HV*V] (q | k | v, a column block per head; rows may be strided), the slot's conv
state and the raw gate inputs ``a``, ``b`` [T, HV] to the core output [T, HV, V] and the slot's
new conv and recurrent states. After the short conv (``prefill_kernels.gdn_conv_``, unchanged),
two launches replace fla's chain behind ``torch_chunk_gated_delta_rule`` (q/k head repeat, two
l2norms, the decay cumsum, kkt + solve, w/u, the state pass, the output pass) and the gate,
padding and state-row gather/scatter ops around it:

- ``gdn_chunk_wy_kernel`` (grid: chunks x HV, all independent): per 64-row chunk and value
  head, the decay ``g = -exp(A_log) * softplus(a + dt_bias)`` (fp32, as ``gdn_gates``) and
  its in-chunk cumulative sum (log2 units, fla's ``RCP_LN2``), ``beta = sigmoid(b)`` rounded
  to BF16, the l2-normalized q and k rows (fp32, eps 1e-6, rounded to BF16, as fla's l2norm;
  key heads shared by HV / H value heads, no repeat), fla's fused kkt + solve on 16-row
  sub-blocks ((I + A)^-1 with A = beta * k k^T * exp(g_i - g_j), strictly lower; BF16 as fla
  stores it) and fla's ``recompute_w_u``: w = Ai (k * beta * exp(g)), u = Ai (v * beta); and
  the chunk's causal score block q k^T * exp(g_i - g_j) (fla's ``chunk_fwd_o``'s A, BF16).
- ``gdn_chunk_state_kernel`` (grid: V / BV x HV, sequential over chunks): fla's
  ``chunk_fwd_h`` and ``chunk_fwd_o`` in one pass that keeps the fp32 state in registers:
  per chunk, v_new = u - w h, o = scale * (q h * exp(g) + A_qk v_new), then
  h = h * exp(g_last) + k^T (v_new * exp(g_last - g)). The state row is read from and
  written back to the layer's state tensor in place (the row index may live on the device).

Rows at or past the real count (``meta``: padding of a captured bucket) get k = v = 0,
g = 0 and beta = 0, so they leave the state unchanged, as the padded fla call does; their
outputs are zeros. The algorithm and its roundings are fla's (BF16 dot operands, fp32
accumulation and state); results differ from fla only in accumulation order.
"""

from __future__ import annotations

import os

import torch

from engine.fast.kernels import INTERPRET, on_device, tl, triton
from engine.fast.prefill_kernels import gdn_conv_

_WIDEN = INTERPRET  # the interpreter's BF16 dots: fp32 operands holding BF16 values
RCP_LN2 = 1.4426950408889634
BT = 64  # chunk rows (fla's chunk_size)
BC = 16  # sub-block rows of the solve


def enabled() -> bool:
    """``ENGINE_GDN_CHUNK`` (default on; 0 restores the conv kernel and fla's chunk chain)."""
    return os.environ.get("ENGINE_GDN_CHUNK", "1") != "0"


def supported(m: torch.nn.Module, conv_state: torch.Tensor) -> bool:
    kc = conv_state.shape[-1]
    return (
        triton is not None
        and m.head_k_dim == 128
        and m.head_v_dim == 128
        and m.num_v_heads % m.num_k_heads == 0
        and conv_state.dtype == torch.bfloat16
        and kc & (kc - 1) == 0
    )


WY_DEFAULT = "split:2:4:128:2"
NORM_WARPS = 8  # the fused kernel's (head default) warps: its l2norm row sums' order
SPLIT_NT = 16  # chunks above which the split form's solve kernel takes its second warp count


def wy_variant() -> tuple[str, int, int, int, int]:
    """The WY stage's form (plan 54), ``ENGINE_GDN_WY``, fields split by ``:`` (or ``,``):
    ``mono[:W]`` the one fused kernel (plan 40 part B) at W warps (default: the
    ``ENGINE_GDN_CHUNK_OPTS`` setting); ``stack[:W]`` the same with the four diagonal
    substitutions stacked into one chain; ``split:S:U:BW[:S2]`` / ``split1:...`` the q / k
    l2norms in ``gdn_chunk_norm_kernel`` (at the fused kernel's 8 warps, so bitwise its
    rounding), the stacked (``split1``: four chained) substitution and the block merges in
    ``gdn_chunk_solve_kernel`` at S warps (S2 when the pass has more than ``SPLIT_NT``
    chunks; 0: S), then the w / u / score dots in ``gdn_chunk_wu_kernel`` at U warps over
    (chunks, HV, 1 + K/BW + V/BW) programs. Returns (kind, S or W (0: default), U, BW, S2)."""
    s = os.environ.get("ENGINE_GDN_WY", WY_DEFAULT).replace(",", ":").split(":")
    kind = s[0]
    assert kind in ("mono", "stack", "split", "split1"), s
    if kind in ("mono", "stack"):
        return kind, int(s[1]) if len(s) > 1 else 0, 0, 0, 0
    d = [2, 4, 128, 0]
    f = [int(x) for x in s[1:]] + d[len(s) - 1:]
    return kind, f[0], f[1], f[2], f[3]


def _opts() -> tuple[int, int, int, int]:
    """(BV, wy warps, state warps, state stages); ENGINE_GDN_CHUNK_OPTS="bv,w1,w2,stages"."""
    o = [int(x) for x in os.environ.get("ENGINE_GDN_CHUNK_OPTS", "16,8,4,1").split(",")]
    return o[0], o[1], o[2], o[3]


if triton is not None:
    from engine.fast.moe_prefill import _bf16

    @triton.jit
    def _rb(x, WIDEN: tl.constexpr):
        """fp32 ``x`` rounded to BF16: a BF16 tensor (an fp32 one holding it under WIDEN)."""
        if WIDEN:
            return _bf16(x, WIDEN)
        else:
            return x.to(tl.bfloat16)

    @triton.jit
    def _op(x, WIDEN: tl.constexpr):
        """A loaded BF16 dot operand (fp32 under WIDEN)."""
        if WIDEN:
            return x.to(tl.float32)
        else:
            return x

    @triton.jit
    def _ld(mixed, rows, col0, T, SM, D: tl.constexpr):
        """Rows ``rows`` (< T) of the D columns from col0 of the convolved projection, fp32."""
        c = col0 + tl.arange(0, D)
        return tl.load(mixed + rows.to(tl.int64)[:, None] * SM + c[None, :], mask=(rows < T)[:, None],
                       other=0.0).to(tl.float32)

    @triton.jit
    def _norm(x, ok, WIDEN: tl.constexpr):
        """Rows of ``x`` l2-normalized (fp32, eps 1e-6) and rounded to BF16; 0 where not ok."""
        x = tl.where(ok[:, None], x, 0.0)
        return _rb(x * (1.0 / tl.sqrt(tl.sum(x * x, axis=1) + 1e-6))[:, None], WIDEN)

    @triton.jit
    def _gates(a_ptr, b_ptr, alog, dtb, rows, ok, SA, SB, i_h, WIDEN: tl.constexpr):
        """(g fp32, beta fp32 of the BF16 value) at ``rows``, 0 where not ok."""
        a = tl.load(a_ptr + rows.to(tl.int64) * SA + i_h, mask=ok, other=0.0).to(tl.float32) + dtb
        sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        g = tl.where(ok, -tl.exp(alog) * sp, 0.0)
        b = tl.load(b_ptr + rows.to(tl.int64) * SB + i_h, mask=ok, other=0.0).to(tl.float32)
        beta = tl.where(ok, _bf16(tl.sigmoid(b), WIDEN), 0.0)
        return g, beta

    @triton.jit
    def _subst(A, o_i, BC: tl.constexpr):
        """(I + A)^-1 of a strictly lower [BC, BC] block (fla's forward substitution)."""
        Ai = -A
        for i in range(2, BC):
            a = tl.sum(tl.where((o_i == i)[:, None], -A, 0.0), 0)
            a = tl.where(o_i < i, a, 0.0)
            a = a + tl.sum(a[:, None] * Ai, 0)
            Ai = tl.where((o_i == i)[:, None], a, Ai)
        return Ai + (o_i[:, None] == o_i[None, :]).to(tl.float32)

    @triton.jit
    def _subst4(A00, A11, A22, A33, o_i, BC: tl.constexpr):
        """``_subst`` of four blocks at once (plan 54): the blocks stacked on a trailing axis
        [BC, BC, 4], so one chain of BC - 2 dependent steps serves all four instead of four
        chains in a row. Per element the same operations as ``_subst``."""
        A = tl.reshape(tl.join(tl.join(A00, A11), tl.join(A22, A33)), (BC, BC, 4))
        o3 = o_i[:, None, None]
        Ai = -A
        for i in range(2, BC):
            a = tl.sum(tl.where(o3 == i, -A, 0.0), 0)  # [BC, 4]: row i of each block
            a = tl.where((o_i < i)[:, None], a, 0.0)
            a = a + tl.sum(a[:, None, :] * Ai, 0)
            Ai = tl.where(o3 == i, a[None, :, :], Ai)
        Ai = Ai + (o_i[:, None] == o_i[None, :]).to(tl.float32)[:, :, None]
        P, Q = tl.split(tl.reshape(Ai, (BC, BC, 2, 2)))
        I00, I11 = tl.split(P)
        I22, I33 = tl.split(Q)
        return I00, I11, I22, I33

    @triton.jit
    def _inv(A00, A10, A11, A20, A21, A22, A30, A31, A32, A33, o_i, BC: tl.constexpr, STACK: tl.constexpr):
        """(I + A)^-1 of the strictly lower [4 BC, 4 BC] block matrix A as its 10 lower blocks
        (fp32): the diagonal blocks by forward substitution, then fla's block merges."""
        if STACK:
            I00, I11, I22, I33 = _subst4(A00, A11, A22, A33, o_i, BC)
        else:
            I00 = _subst(A00, o_i, BC)
            I11 = _subst(A11, o_i, BC)
            I22 = _subst(A22, o_i, BC)
            I33 = _subst(A33, o_i, BC)
        I10 = -tl.dot(tl.dot(I11, A10, input_precision="ieee"), I00, input_precision="ieee")
        I21 = -tl.dot(tl.dot(I22, A21, input_precision="ieee"), I11, input_precision="ieee")
        I32 = -tl.dot(tl.dot(I33, A32, input_precision="ieee"), I22, input_precision="ieee")
        I20 = -tl.dot(I22, tl.dot(A20, I00, input_precision="ieee") + tl.dot(A21, I10, input_precision="ieee"),
                      input_precision="ieee")
        I31 = -tl.dot(I33, tl.dot(A31, I11, input_precision="ieee") + tl.dot(A32, I21, input_precision="ieee"),
                      input_precision="ieee")
        I30 = -tl.dot(
            I33,
            tl.dot(A30, I00, input_precision="ieee") + tl.dot(A31, I10, input_precision="ieee")
            + tl.dot(A32, I20, input_precision="ieee"),
            input_precision="ieee",
        )
        return I00, I10, I11, I20, I21, I22, I30, I31, I32, I33

    @triton.jit
    def _kkt(ka, kb, ca, cb, ba):
        """beta_a * ka kb^T * exp2(c_a - c_b) (an off-diagonal block, a below b)."""
        return tl.dot(ka, tl.trans(kb)) * tl.exp2(ca[:, None] - cb[None, :]) * ba[:, None]

    @triton.jit
    def _qk(qa, kb, ca, cb):
        """qa kb^T * exp2(c_a - c_b) (an off-diagonal causal score block, a below b)."""
        return tl.dot(qa, tl.trans(kb)) * tl.exp2(ca[:, None] - cb[None, :])

    @triton.jit(do_not_specialize=["T"])
    def gdn_chunk_wy_kernel(
        mixed, a_ptr, b_ptr, alog_ptr, dtb_ptr, meta_ptr, qn, kn, w, u, gc, aqk, T, SM, SA, SB,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BC: tl.constexpr,
        HAS_META: tl.constexpr, WIDEN: tl.constexpr, STACK: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        i_kh = i_h // (HV // H)
        if HAS_META:
            n = tl.load(meta_ptr).to(tl.int32)
        else:
            n = T
        o_i = tl.arange(0, BC)
        r0 = i_t * (4 * BC) + o_i
        r1 = r0 + BC
        r2 = r1 + BC
        r3 = r2 + BC
        m0 = r0 < n
        m1 = r1 < n
        m2 = r2 < n
        m3 = r3 < n
        t0 = r0 < T
        t1 = r1 < T
        t2 = r2 < T
        t3 = r3 < T
        alog = tl.load(alog_ptr + i_h).to(tl.float32)
        dtb = tl.load(dtb_ptr + i_h).to(tl.float32)
        g0, b0 = _gates(a_ptr, b_ptr, alog, dtb, r0, m0, SA, SB, i_h, WIDEN)
        g1, b1 = _gates(a_ptr, b_ptr, alog, dtb, r1, m1, SA, SB, i_h, WIDEN)
        g2, b2 = _gates(a_ptr, b_ptr, alog, dtb, r2, m2, SA, SB, i_h, WIDEN)
        g3, b3 = _gates(a_ptr, b_ptr, alog, dtb, r3, m3, SA, SB, i_h, WIDEN)
        # In-chunk cumulative decay in log2 units.
        s0 = tl.sum(g0, axis=0)
        s1 = s0 + tl.sum(g1, axis=0)
        s2 = s1 + tl.sum(g2, axis=0)
        c0 = tl.cumsum(g0, axis=0) * 1.4426950408889634
        c1 = (tl.cumsum(g1, axis=0) + s0) * 1.4426950408889634
        c2 = (tl.cumsum(g2, axis=0) + s1) * 1.4426950408889634
        c3 = (tl.cumsum(g3, axis=0) + s2) * 1.4426950408889634
        tl.store(gc + r0.to(tl.int64) * HV + i_h, c0, mask=t0)
        tl.store(gc + r1.to(tl.int64) * HV + i_h, c1, mask=t1)
        tl.store(gc + r2.to(tl.int64) * HV + i_h, c2, mask=t2)
        tl.store(gc + r3.to(tl.int64) * HV + i_h, c3, mask=t3)

        # l2norm of the chunk's k rows (zero past the real count) and q rows.
        kc0 = H * K + i_kh * K
        k0 = _norm(_ld(mixed, r0, kc0, T, SM, K), m0, WIDEN)
        k1 = _norm(_ld(mixed, r1, kc0, T, SM, K), m1, WIDEN)
        k2 = _norm(_ld(mixed, r2, kc0, T, SM, K), m2, WIDEN)
        k3 = _norm(_ld(mixed, r3, kc0, T, SM, K), m3, WIDEN)
        qc0 = i_kh * K
        q0 = _norm(_ld(mixed, r0, qc0, T, SM, K), t0, WIDEN)
        q1 = _norm(_ld(mixed, r1, qc0, T, SM, K), t1, WIDEN)
        q2 = _norm(_ld(mixed, r2, qc0, T, SM, K), t2, WIDEN)
        q3 = _norm(_ld(mixed, r3, qc0, T, SM, K), t3, WIDEN)
        d = tl.arange(0, K)
        if i_h % (HV // H) == 0:
            pk = kn + i_kh * K + d[None, :]
            pq = qn + i_kh * K + d[None, :]
            tl.store(pk + r0.to(tl.int64)[:, None] * (H * K), k0.to(tl.bfloat16), mask=t0[:, None])
            tl.store(pk + r1.to(tl.int64)[:, None] * (H * K), k1.to(tl.bfloat16), mask=t1[:, None])
            tl.store(pk + r2.to(tl.int64)[:, None] * (H * K), k2.to(tl.bfloat16), mask=t2[:, None])
            tl.store(pk + r3.to(tl.int64)[:, None] * (H * K), k3.to(tl.bfloat16), mask=t3[:, None])
            tl.store(pq + r0.to(tl.int64)[:, None] * (H * K), q0.to(tl.bfloat16), mask=t0[:, None])
            tl.store(pq + r1.to(tl.int64)[:, None] * (H * K), q1.to(tl.bfloat16), mask=t1[:, None])
            tl.store(pq + r2.to(tl.int64)[:, None] * (H * K), q2.to(tl.bfloat16), mask=t2[:, None])
            tl.store(pq + r3.to(tl.int64)[:, None] * (H * K), q3.to(tl.bfloat16), mask=t3[:, None])

        # The causal score block q k^T * exp2(c_i - c_j) (i >= j), BF16 as fla's chunk_fwd_o.
        m_c = o_i[:, None] >= o_i[None, :]
        pa = aqk + i_h * (4 * BC) + o_i[None, :]
        z = tl.zeros([BC, BC], dtype=tl.float32)
        Q00 = tl.where(m_c, tl.dot(q0, tl.trans(k0)) * tl.exp2(c0[:, None] - c0[None, :]), 0.0)
        Q11 = tl.where(m_c, tl.dot(q1, tl.trans(k1)) * tl.exp2(c1[:, None] - c1[None, :]), 0.0)
        Q22 = tl.where(m_c, tl.dot(q2, tl.trans(k2)) * tl.exp2(c2[:, None] - c2[None, :]), 0.0)
        Q33 = tl.where(m_c, tl.dot(q3, tl.trans(k3)) * tl.exp2(c3[:, None] - c3[None, :]), 0.0)
        R0 = pa + r0.to(tl.int64)[:, None] * (HV * 4 * BC)
        R1 = pa + r1.to(tl.int64)[:, None] * (HV * 4 * BC)
        R2 = pa + r2.to(tl.int64)[:, None] * (HV * 4 * BC)
        R3 = pa + r3.to(tl.int64)[:, None] * (HV * 4 * BC)
        tl.store(R0, Q00.to(tl.bfloat16), mask=t0[:, None])
        tl.store(R0 + BC, z.to(tl.bfloat16), mask=t0[:, None])
        tl.store(R0 + 2 * BC, z.to(tl.bfloat16), mask=t0[:, None])
        tl.store(R0 + 3 * BC, z.to(tl.bfloat16), mask=t0[:, None])
        tl.store(R1, _qk(q1, k0, c1, c0).to(tl.bfloat16), mask=t1[:, None])
        tl.store(R1 + BC, Q11.to(tl.bfloat16), mask=t1[:, None])
        tl.store(R1 + 2 * BC, z.to(tl.bfloat16), mask=t1[:, None])
        tl.store(R1 + 3 * BC, z.to(tl.bfloat16), mask=t1[:, None])
        tl.store(R2, _qk(q2, k0, c2, c0).to(tl.bfloat16), mask=t2[:, None])
        tl.store(R2 + BC, _qk(q2, k1, c2, c1).to(tl.bfloat16), mask=t2[:, None])
        tl.store(R2 + 2 * BC, Q22.to(tl.bfloat16), mask=t2[:, None])
        tl.store(R2 + 3 * BC, z.to(tl.bfloat16), mask=t2[:, None])
        tl.store(R3, _qk(q3, k0, c3, c0).to(tl.bfloat16), mask=t3[:, None])
        tl.store(R3 + BC, _qk(q3, k1, c3, c1).to(tl.bfloat16), mask=t3[:, None])
        tl.store(R3 + 2 * BC, _qk(q3, k2, c3, c2).to(tl.bfloat16), mask=t3[:, None])
        tl.store(R3 + 3 * BC, Q33.to(tl.bfloat16), mask=t3[:, None])

        # beta * k k^T * exp2(c_i - c_j), strictly lower, as 10 [BC, BC] blocks; (I + A)^-1.
        m_d = o_i[:, None] > o_i[None, :]
        A00 = tl.where(m_d, tl.dot(k0, tl.trans(k0)) * tl.exp2(c0[:, None] - c0[None, :]), 0.0) * b0[:, None]
        A11 = tl.where(m_d, tl.dot(k1, tl.trans(k1)) * tl.exp2(c1[:, None] - c1[None, :]), 0.0) * b1[:, None]
        A22 = tl.where(m_d, tl.dot(k2, tl.trans(k2)) * tl.exp2(c2[:, None] - c2[None, :]), 0.0) * b2[:, None]
        A33 = tl.where(m_d, tl.dot(k3, tl.trans(k3)) * tl.exp2(c3[:, None] - c3[None, :]), 0.0) * b3[:, None]
        A10 = _kkt(k1, k0, c1, c0, b1)
        A20 = _kkt(k2, k0, c2, c0, b2)
        A21 = _kkt(k2, k1, c2, c1, b2)
        A30 = _kkt(k3, k0, c3, c0, b3)
        A31 = _kkt(k3, k1, c3, c1, b3)
        A32 = _kkt(k3, k2, c3, c2, b3)
        I00, I10, I11, I20, I21, I22, I30, I31, I32, I33 = _inv(
            A00, A10, A11, A20, A21, A22, A30, A31, A32, A33, o_i, BC, STACK)
        # fla stores (I + A)^-1 in BF16 and reads it back for w and u.
        I00 = _rb(I00, WIDEN)
        I10 = _rb(I10, WIDEN)
        I11 = _rb(I11, WIDEN)
        I20 = _rb(I20, WIDEN)
        I21 = _rb(I21, WIDEN)
        I22 = _rb(I22, WIDEN)
        I30 = _rb(I30, WIDEN)
        I31 = _rb(I31, WIDEN)
        I32 = _rb(I32, WIDEN)
        I33 = _rb(I33, WIDEN)

        # w = Ai (k * beta * exp2(c)) [64, K]
        e0 = _rb(k0.to(tl.float32) * (b0 * tl.exp2(c0))[:, None], WIDEN)
        e1 = _rb(k1.to(tl.float32) * (b1 * tl.exp2(c1))[:, None], WIDEN)
        e2 = _rb(k2.to(tl.float32) * (b2 * tl.exp2(c2))[:, None], WIDEN)
        e3 = _rb(k3.to(tl.float32) * (b3 * tl.exp2(c3))[:, None], WIDEN)
        pw = w + i_h * K + d[None, :]
        tl.store(pw + r0.to(tl.int64)[:, None] * (HV * K), tl.dot(I00, e0).to(tl.bfloat16), mask=t0[:, None])
        tl.store(pw + r1.to(tl.int64)[:, None] * (HV * K),
                 (tl.dot(I10, e0) + tl.dot(I11, e1)).to(tl.bfloat16), mask=t1[:, None])
        tl.store(pw + r2.to(tl.int64)[:, None] * (HV * K),
                 (tl.dot(I20, e0) + tl.dot(I21, e1) + tl.dot(I22, e2)).to(tl.bfloat16), mask=t2[:, None])
        tl.store(pw + r3.to(tl.int64)[:, None] * (HV * K),
                 (tl.dot(I30, e0) + tl.dot(I31, e1) + tl.dot(I32, e2) + tl.dot(I33, e3)).to(tl.bfloat16),
                 mask=t3[:, None])

        # u = Ai (v * beta) [64, V]
        vc0 = 2 * H * K + i_h * V
        v0 = _rb(_ld(mixed, r0, vc0, T, SM, V) * b0[:, None], WIDEN)
        v1 = _rb(_ld(mixed, r1, vc0, T, SM, V) * b1[:, None], WIDEN)
        v2 = _rb(_ld(mixed, r2, vc0, T, SM, V) * b2[:, None], WIDEN)
        v3 = _rb(_ld(mixed, r3, vc0, T, SM, V) * b3[:, None], WIDEN)
        dv = tl.arange(0, V)
        pu = u + i_h * V + dv[None, :]
        tl.store(pu + r0.to(tl.int64)[:, None] * (HV * V), tl.dot(I00, v0).to(tl.bfloat16), mask=t0[:, None])
        tl.store(pu + r1.to(tl.int64)[:, None] * (HV * V),
                 (tl.dot(I10, v0) + tl.dot(I11, v1)).to(tl.bfloat16), mask=t1[:, None])
        tl.store(pu + r2.to(tl.int64)[:, None] * (HV * V),
                 (tl.dot(I20, v0) + tl.dot(I21, v1) + tl.dot(I22, v2)).to(tl.bfloat16), mask=t2[:, None])
        tl.store(pu + r3.to(tl.int64)[:, None] * (HV * V),
                 (tl.dot(I30, v0) + tl.dot(I31, v1) + tl.dot(I32, v2) + tl.dot(I33, v3)).to(tl.bfloat16),
                 mask=t3[:, None])

    @triton.jit(do_not_specialize=["T"])
    def gdn_chunk_norm_kernel(mixed, meta_ptr, qn, kn, T, SM, H: tl.constexpr, K: tl.constexpr,
                              BC: tl.constexpr, HAS_META: tl.constexpr, WIDEN: tl.constexpr):
        """The l2-normalized q and k rows of one key head and 64-row chunk (grid chunks x H),
        BF16 into ``qn``, ``kn``: the fused kernel's ``_norm`` on the same [BC, K] tiles, run
        at its warp count so the row sums reduce in the same order (bitwise the same)."""
        i_t = tl.program_id(0)
        i_kh = tl.program_id(1)
        if HAS_META:
            n = tl.load(meta_ptr).to(tl.int32)
        else:
            n = T
        o_i = tl.arange(0, BC)
        d = tl.arange(0, K)
        for j in tl.static_range(4):
            r = i_t * (4 * BC) + j * BC + o_i
            t = r < T
            k = _norm(_ld(mixed, r, H * K + i_kh * K, T, SM, K), r < n, WIDEN)
            q = _norm(_ld(mixed, r, i_kh * K, T, SM, K), t, WIDEN)
            tl.store(kn + r.to(tl.int64)[:, None] * (H * K) + i_kh * K + d[None, :], k.to(tl.bfloat16),
                     mask=t[:, None])
            tl.store(qn + r.to(tl.int64)[:, None] * (H * K) + i_kh * K + d[None, :], q.to(tl.bfloat16),
                     mask=t[:, None])

    @triton.jit(do_not_specialize=["T"])
    def gdn_chunk_solve_kernel(
        mixed, a_ptr, b_ptr, alog_ptr, dtb_ptr, meta_ptr, kn, gc, ai, T, SM, SA, SB,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BC: tl.constexpr, HAS_META: tl.constexpr,
        WIDEN: tl.constexpr, STACK: tl.constexpr,
    ):
        """The serial half of ``gdn_chunk_wy_kernel`` (plan 54 split; grid chunks x HV, few
        warps): decay cumsum (``gc``) and, from the normalized k rows ``kn``,
        (I + A)^-1 rounded to BF16 as fla stores it, into ``ai`` [T, HV, 4 BC] (row-major
        64-row block per chunk, zero above the diagonal). Same operations as the fused kernel."""
        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        i_kh = i_h // (HV // H)
        if HAS_META:
            n = tl.load(meta_ptr).to(tl.int32)
        else:
            n = T
        o_i = tl.arange(0, BC)
        r0 = i_t * (4 * BC) + o_i
        r1 = r0 + BC
        r2 = r1 + BC
        r3 = r2 + BC
        m0 = r0 < n
        m1 = r1 < n
        m2 = r2 < n
        m3 = r3 < n
        t0 = r0 < T
        t1 = r1 < T
        t2 = r2 < T
        t3 = r3 < T
        alog = tl.load(alog_ptr + i_h).to(tl.float32)
        dtb = tl.load(dtb_ptr + i_h).to(tl.float32)
        g0, b0 = _gates(a_ptr, b_ptr, alog, dtb, r0, m0, SA, SB, i_h, WIDEN)
        g1, b1 = _gates(a_ptr, b_ptr, alog, dtb, r1, m1, SA, SB, i_h, WIDEN)
        g2, b2 = _gates(a_ptr, b_ptr, alog, dtb, r2, m2, SA, SB, i_h, WIDEN)
        g3, b3 = _gates(a_ptr, b_ptr, alog, dtb, r3, m3, SA, SB, i_h, WIDEN)
        s0 = tl.sum(g0, axis=0)
        s1 = s0 + tl.sum(g1, axis=0)
        s2 = s1 + tl.sum(g2, axis=0)
        c0 = tl.cumsum(g0, axis=0) * 1.4426950408889634
        c1 = (tl.cumsum(g1, axis=0) + s0) * 1.4426950408889634
        c2 = (tl.cumsum(g2, axis=0) + s1) * 1.4426950408889634
        c3 = (tl.cumsum(g3, axis=0) + s2) * 1.4426950408889634
        tl.store(gc + r0.to(tl.int64) * HV + i_h, c0, mask=t0)
        tl.store(gc + r1.to(tl.int64) * HV + i_h, c1, mask=t1)
        tl.store(gc + r2.to(tl.int64) * HV + i_h, c2, mask=t2)
        tl.store(gc + r3.to(tl.int64) * HV + i_h, c3, mask=t3)

        # The l2-normalized k rows, written by ``gdn_chunk_norm_kernel``.
        k0 = _ldb(kn, r0, i_kh * K, H * K, T, K, WIDEN)
        k1 = _ldb(kn, r1, i_kh * K, H * K, T, K, WIDEN)
        k2 = _ldb(kn, r2, i_kh * K, H * K, T, K, WIDEN)
        k3 = _ldb(kn, r3, i_kh * K, H * K, T, K, WIDEN)

        m_d = o_i[:, None] > o_i[None, :]
        A00 = tl.where(m_d, tl.dot(k0, tl.trans(k0)) * tl.exp2(c0[:, None] - c0[None, :]), 0.0) * b0[:, None]
        A11 = tl.where(m_d, tl.dot(k1, tl.trans(k1)) * tl.exp2(c1[:, None] - c1[None, :]), 0.0) * b1[:, None]
        A22 = tl.where(m_d, tl.dot(k2, tl.trans(k2)) * tl.exp2(c2[:, None] - c2[None, :]), 0.0) * b2[:, None]
        A33 = tl.where(m_d, tl.dot(k3, tl.trans(k3)) * tl.exp2(c3[:, None] - c3[None, :]), 0.0) * b3[:, None]
        A10 = _kkt(k1, k0, c1, c0, b1)
        A20 = _kkt(k2, k0, c2, c0, b2)
        A21 = _kkt(k2, k1, c2, c1, b2)
        A30 = _kkt(k3, k0, c3, c0, b3)
        A31 = _kkt(k3, k1, c3, c1, b3)
        A32 = _kkt(k3, k2, c3, c2, b3)
        I00, I10, I11, I20, I21, I22, I30, I31, I32, I33 = _inv(
            A00, A10, A11, A20, A21, A22, A30, A31, A32, A33, o_i, BC, STACK)
        z = tl.zeros([BC, BC], dtype=tl.bfloat16)
        pa = ai + i_h * (4 * BC) + o_i[None, :]
        R0 = pa + r0.to(tl.int64)[:, None] * (HV * 4 * BC)
        R1 = pa + r1.to(tl.int64)[:, None] * (HV * 4 * BC)
        R2 = pa + r2.to(tl.int64)[:, None] * (HV * 4 * BC)
        R3 = pa + r3.to(tl.int64)[:, None] * (HV * 4 * BC)
        tl.store(R0, I00.to(tl.bfloat16), mask=t0[:, None])
        tl.store(R0 + BC, z, mask=t0[:, None])
        tl.store(R0 + 2 * BC, z, mask=t0[:, None])
        tl.store(R0 + 3 * BC, z, mask=t0[:, None])
        tl.store(R1, I10.to(tl.bfloat16), mask=t1[:, None])
        tl.store(R1 + BC, I11.to(tl.bfloat16), mask=t1[:, None])
        tl.store(R1 + 2 * BC, z, mask=t1[:, None])
        tl.store(R1 + 3 * BC, z, mask=t1[:, None])
        tl.store(R2, I20.to(tl.bfloat16), mask=t2[:, None])
        tl.store(R2 + BC, I21.to(tl.bfloat16), mask=t2[:, None])
        tl.store(R2 + 2 * BC, I22.to(tl.bfloat16), mask=t2[:, None])
        tl.store(R2 + 3 * BC, z, mask=t2[:, None])
        tl.store(R3, I30.to(tl.bfloat16), mask=t3[:, None])
        tl.store(R3 + BC, I31.to(tl.bfloat16), mask=t3[:, None])
        tl.store(R3 + 2 * BC, I32.to(tl.bfloat16), mask=t3[:, None])
        tl.store(R3 + 3 * BC, I33.to(tl.bfloat16), mask=t3[:, None])

    @triton.jit
    def _ldb(p, rows, col0, S, T, D: tl.constexpr, WIDEN: tl.constexpr):
        """Rows ``rows`` (< T, else 0) of D BF16 columns from col0 of a row-major [T, S] tensor,
        as a dot operand."""
        c = col0 + tl.arange(0, D)
        return _op(tl.load(p + rows.to(tl.int64)[:, None] * S + c[None, :], mask=(rows < T)[:, None], other=0.0),
                   WIDEN)

    @triton.jit
    def _wu_store(p0, p1, p2, p3, e0, e1, e2, e3, I00, I10, I11, I20, I21, I22, I30, I31, I32, I33,
                  t0, t1, t2, t3):
        """``gdn_chunk_wu_kernel``'s w / u block rows: Ai (rows) times the scaled operands, the
        block dots in the fused kernel's order. Called inside each role's branch, so no pointer
        leaves an ``if`` (Triton 3.6's TritonAMDGPUCanonicalizePointers fails on one that does)."""
        tl.store(p0, tl.dot(I00, e0).to(tl.bfloat16), mask=t0[:, None])
        tl.store(p1, (tl.dot(I10, e0) + tl.dot(I11, e1)).to(tl.bfloat16), mask=t1[:, None])
        tl.store(p2, (tl.dot(I20, e0) + tl.dot(I21, e1) + tl.dot(I22, e2)).to(tl.bfloat16), mask=t2[:, None])
        tl.store(p3,
                 (tl.dot(I30, e0) + tl.dot(I31, e1) + tl.dot(I32, e2) + tl.dot(I33, e3)).to(tl.bfloat16),
                 mask=t3[:, None])

    @triton.jit(do_not_specialize=["T"])
    def gdn_chunk_wu_kernel(
        mixed, b_ptr, meta_ptr, ai, qn, kn, w, u, gc, aqk, T, SM, SB,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BC: tl.constexpr,
        BW: tl.constexpr, HAS_META: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """The dot half of ``gdn_chunk_wy_kernel`` (plan 54 split; grid chunks x HV x
        (1 + K / BW + V / BW)) after ``gdn_chunk_solve_kernel``: role 0 the causal score block ``aqk``
        (from ``qn``, ``kn``); then w = Ai (k * beta * exp2(c)) and
        u = Ai (v * beta), BW columns per program. The same block dots, in the same order, as
        the fused kernel."""
        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        role = tl.program_id(2)
        i_kh = i_h // (HV // H)
        if HAS_META:
            n = tl.load(meta_ptr).to(tl.int32)
        else:
            n = T
        o_i = tl.arange(0, BC)
        r0 = i_t * (4 * BC) + o_i
        r1 = r0 + BC
        r2 = r1 + BC
        r3 = r2 + BC
        t0 = r0 < T
        t1 = r1 < T
        t2 = r2 < T
        t3 = r3 < T
        c0 = tl.load(gc + r0.to(tl.int64) * HV + i_h, mask=t0, other=0.0)
        c1 = tl.load(gc + r1.to(tl.int64) * HV + i_h, mask=t1, other=0.0)
        c2 = tl.load(gc + r2.to(tl.int64) * HV + i_h, mask=t2, other=0.0)
        c3 = tl.load(gc + r3.to(tl.int64) * HV + i_h, mask=t3, other=0.0)
        NW: tl.constexpr = K // BW
        if role == 0:
            q0 = _ldb(qn, r0, i_kh * K, H * K, T, K, WIDEN)
            q1 = _ldb(qn, r1, i_kh * K, H * K, T, K, WIDEN)
            q2 = _ldb(qn, r2, i_kh * K, H * K, T, K, WIDEN)
            q3 = _ldb(qn, r3, i_kh * K, H * K, T, K, WIDEN)
            k0 = _ldb(kn, r0, i_kh * K, H * K, T, K, WIDEN)
            k1 = _ldb(kn, r1, i_kh * K, H * K, T, K, WIDEN)
            k2 = _ldb(kn, r2, i_kh * K, H * K, T, K, WIDEN)
            k3 = _ldb(kn, r3, i_kh * K, H * K, T, K, WIDEN)
            m_c = o_i[:, None] >= o_i[None, :]
            pa = aqk + i_h * (4 * BC) + o_i[None, :]
            z = tl.zeros([BC, BC], dtype=tl.float32)
            Q00 = tl.where(m_c, tl.dot(q0, tl.trans(k0)) * tl.exp2(c0[:, None] - c0[None, :]), 0.0)
            Q11 = tl.where(m_c, tl.dot(q1, tl.trans(k1)) * tl.exp2(c1[:, None] - c1[None, :]), 0.0)
            Q22 = tl.where(m_c, tl.dot(q2, tl.trans(k2)) * tl.exp2(c2[:, None] - c2[None, :]), 0.0)
            Q33 = tl.where(m_c, tl.dot(q3, tl.trans(k3)) * tl.exp2(c3[:, None] - c3[None, :]), 0.0)
            R0 = pa + r0.to(tl.int64)[:, None] * (HV * 4 * BC)
            R1 = pa + r1.to(tl.int64)[:, None] * (HV * 4 * BC)
            R2 = pa + r2.to(tl.int64)[:, None] * (HV * 4 * BC)
            R3 = pa + r3.to(tl.int64)[:, None] * (HV * 4 * BC)
            tl.store(R0, Q00.to(tl.bfloat16), mask=t0[:, None])
            tl.store(R0 + BC, z.to(tl.bfloat16), mask=t0[:, None])
            tl.store(R0 + 2 * BC, z.to(tl.bfloat16), mask=t0[:, None])
            tl.store(R0 + 3 * BC, z.to(tl.bfloat16), mask=t0[:, None])
            tl.store(R1, _qk(q1, k0, c1, c0).to(tl.bfloat16), mask=t1[:, None])
            tl.store(R1 + BC, Q11.to(tl.bfloat16), mask=t1[:, None])
            tl.store(R1 + 2 * BC, z.to(tl.bfloat16), mask=t1[:, None])
            tl.store(R1 + 3 * BC, z.to(tl.bfloat16), mask=t1[:, None])
            tl.store(R2, _qk(q2, k0, c2, c0).to(tl.bfloat16), mask=t2[:, None])
            tl.store(R2 + BC, _qk(q2, k1, c2, c1).to(tl.bfloat16), mask=t2[:, None])
            tl.store(R2 + 2 * BC, Q22.to(tl.bfloat16), mask=t2[:, None])
            tl.store(R2 + 3 * BC, z.to(tl.bfloat16), mask=t2[:, None])
            tl.store(R3, _qk(q3, k0, c3, c0).to(tl.bfloat16), mask=t3[:, None])
            tl.store(R3 + BC, _qk(q3, k1, c3, c1).to(tl.bfloat16), mask=t3[:, None])
            tl.store(R3 + 2 * BC, _qk(q3, k2, c3, c2).to(tl.bfloat16), mask=t3[:, None])
            tl.store(R3 + 3 * BC, Q33.to(tl.bfloat16), mask=t3[:, None])
        else:
            if HAS_META:
                m0 = r0 < n
                m1 = r1 < n
                m2 = r2 < n
                m3 = r3 < n
            else:
                m0 = t0
                m1 = t1
                m2 = t2
                m3 = t3
            # beta as the solve kernel's (``_gates``): BF16 sigmoid(b), 0 past the real count.
            b0 = tl.where(m0, _bf16(tl.sigmoid(tl.load(b_ptr + r0.to(tl.int64) * SB + i_h, mask=m0, other=0.0)
                                               .to(tl.float32)), WIDEN), 0.0)
            b1 = tl.where(m1, _bf16(tl.sigmoid(tl.load(b_ptr + r1.to(tl.int64) * SB + i_h, mask=m1, other=0.0)
                                               .to(tl.float32)), WIDEN), 0.0)
            b2 = tl.where(m2, _bf16(tl.sigmoid(tl.load(b_ptr + r2.to(tl.int64) * SB + i_h, mask=m2, other=0.0)
                                               .to(tl.float32)), WIDEN), 0.0)
            b3 = tl.where(m3, _bf16(tl.sigmoid(tl.load(b_ptr + r3.to(tl.int64) * SB + i_h, mask=m3, other=0.0)
                                               .to(tl.float32)), WIDEN), 0.0)
            S_AI: tl.constexpr = HV * 4 * BC
            ac = i_h * (4 * BC)
            I00 = _ldb(ai, r0, ac, S_AI, T, BC, WIDEN)
            I10 = _ldb(ai, r1, ac, S_AI, T, BC, WIDEN)
            I11 = _ldb(ai, r1, ac + BC, S_AI, T, BC, WIDEN)
            I20 = _ldb(ai, r2, ac, S_AI, T, BC, WIDEN)
            I21 = _ldb(ai, r2, ac + BC, S_AI, T, BC, WIDEN)
            I22 = _ldb(ai, r2, ac + 2 * BC, S_AI, T, BC, WIDEN)
            I30 = _ldb(ai, r3, ac, S_AI, T, BC, WIDEN)
            I31 = _ldb(ai, r3, ac + BC, S_AI, T, BC, WIDEN)
            I32 = _ldb(ai, r3, ac + 2 * BC, S_AI, T, BC, WIDEN)
            I33 = _ldb(ai, r3, ac + 3 * BC, S_AI, T, BC, WIDEN)
            dw = tl.arange(0, BW)
            if role <= NW:
                col = (role - 1) * BW
                # w = Ai (k * beta * exp2(c)), columns col .. col + BW
                e0 = _rb(_ldb(kn, r0, i_kh * K + col, H * K, T, BW, WIDEN).to(tl.float32)
                         * (b0 * tl.exp2(c0))[:, None], WIDEN)
                e1 = _rb(_ldb(kn, r1, i_kh * K + col, H * K, T, BW, WIDEN).to(tl.float32)
                         * (b1 * tl.exp2(c1))[:, None], WIDEN)
                e2 = _rb(_ldb(kn, r2, i_kh * K + col, H * K, T, BW, WIDEN).to(tl.float32)
                         * (b2 * tl.exp2(c2))[:, None], WIDEN)
                e3 = _rb(_ldb(kn, r3, i_kh * K + col, H * K, T, BW, WIDEN).to(tl.float32)
                         * (b3 * tl.exp2(c3))[:, None], WIDEN)
                po = w + i_h * K + col + dw[None, :]
                p0 = po + r0.to(tl.int64)[:, None] * (HV * K)
                p1 = po + r1.to(tl.int64)[:, None] * (HV * K)
                p2 = po + r2.to(tl.int64)[:, None] * (HV * K)
                p3 = po + r3.to(tl.int64)[:, None] * (HV * K)
                _wu_store(p0, p1, p2, p3, e0, e1, e2, e3, I00, I10, I11, I20, I21, I22, I30, I31, I32, I33,
                          t0, t1, t2, t3)
            else:
                col = (role - 1 - NW) * BW
                # u = Ai (v * beta), columns col .. col + BW
                vc0 = 2 * H * K + i_h * V + col
                e0 = _rb(_ld(mixed, r0, vc0, T, SM, BW) * b0[:, None], WIDEN)
                e1 = _rb(_ld(mixed, r1, vc0, T, SM, BW) * b1[:, None], WIDEN)
                e2 = _rb(_ld(mixed, r2, vc0, T, SM, BW) * b2[:, None], WIDEN)
                e3 = _rb(_ld(mixed, r3, vc0, T, SM, BW) * b3[:, None], WIDEN)
                po = u + i_h * V + col + dw[None, :]
                p0 = po + r0.to(tl.int64)[:, None] * (HV * V)
                p1 = po + r1.to(tl.int64)[:, None] * (HV * V)
                p2 = po + r2.to(tl.int64)[:, None] * (HV * V)
                p3 = po + r3.to(tl.int64)[:, None] * (HV * V)
                _wu_store(p0, p1, p2, p3, e0, e1, e2, e3, I00, I10, I11, I20, I21, I22, I30, I31, I32, I33,
                          t0, t1, t2, t3)

    @triton.jit(do_not_specialize=["T"])
    def gdn_chunk_state_kernel(
        qn, kn, w, u, gc, aqk, o, state, row_ptr, meta_ptr, T, scale,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
        BV: tl.constexpr, HAS_ROW: tl.constexpr, HAS_META: tl.constexpr, STAGES: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)
        i_kh = i_h // (HV // H)
        if HAS_META:
            n = tl.load(meta_ptr).to(tl.int32)
        else:
            n = T
        if HAS_ROW:
            state += tl.load(row_ptr).to(tl.int64) * (HV * K * V)
        state += i_h * (K * V)
        dk = tl.arange(0, K)
        o_v = i_v * BV + tl.arange(0, BV)
        p_s = state + dk[:, None] * V + o_v[None, :]
        h = tl.load(p_s)  # [K, BV] fp32
        o_t0 = tl.arange(0, BT)
        causal = o_t0[:, None] >= o_t0[None, :]
        nt = tl.cdiv(n, BT)
        for i_t in tl.range(0, nt, num_stages=STAGES):
            o_t = i_t * BT + o_t0
            m_t = o_t < T
            rt = o_t.to(tl.int64)
            q = _op(tl.load(qn + rt[:, None] * (H * K) + i_kh * K + dk[None, :], mask=m_t[:, None], other=0.0), WIDEN)
            kt = _op(tl.load(kn + rt[None, :] * (H * K) + i_kh * K + dk[:, None], mask=m_t[None, :], other=0.0), WIDEN)
            ww = _op(tl.load(w + rt[:, None] * (HV * K) + i_h * K + dk[None, :], mask=m_t[:, None], other=0.0), WIDEN)
            uu = tl.load(u + rt[:, None] * (HV * V) + i_h * V + o_v[None, :], mask=m_t[:, None], other=0.0)
            aq = tl.load(aqk + rt[:, None] * (HV * BT) + i_h * BT + o_t0[None, :], mask=m_t[:, None], other=0.0)
            aq = _op(tl.where(causal, aq, 0.0), WIDEN)
            g = tl.load(gc + rt * HV + i_h, mask=m_t, other=0.0)
            last = tl.minimum((i_t + 1) * BT, T) - 1
            g_last = tl.load(gc + last.to(tl.int64) * HV + i_h)
            hb = _rb(h, WIDEN)
            vn = uu.to(tl.float32) - tl.dot(ww, hb)
            oo = tl.dot(q, hb) * tl.exp2(g)[:, None] * scale + tl.dot(aq, _rb(vn, WIDEN)) * scale
            oo = tl.where((o_t < n)[:, None], oo, 0.0)  # padding rows: 0
            tl.store(o + rt[:, None] * (HV * V) + i_h * V + o_v[None, :], oo.to(tl.bfloat16), mask=m_t[:, None])
            vg = _rb(vn * tl.where(m_t, tl.exp2(g_last - g), 0.0)[:, None], WIDEN)
            h = h * tl.exp2(g_last) + tl.dot(kt, vg)
        tl.store(p_s, h)
        # Chunks of padding rows only: zeros.
        zero = tl.zeros([BT, BV], dtype=tl.bfloat16)
        for i_t in range(nt, tl.cdiv(T, BT)):
            o_t = i_t * BT + o_t0
            rt = o_t.to(tl.int64)
            tl.store(o + rt[:, None] * (HV * V) + i_h * V + o_v[None, :], zero, mask=(o_t < T)[:, None])


def wy(mixed, a, b, a_log, dt_bias, meta, heads, qn, kn, w, u, gc, aqk, warps=None, variant=None):
    """The WY stage into ``qn``, ``kn``, ``w``, ``u``, ``gc``, ``aqk`` (see ``chunk``), in the
    form ``variant`` (default: ``wy_variant()``); ``warps``: the fused kernel's (default: the
    variant's, else the ``_opts`` setting)."""
    kind, ws, ww, bw, s2 = (wy_variant() if variant is None else tuple(variant) + (0,))[:5]
    tokens = mixed.shape[0]
    hv = a.shape[-1]
    kd = vd = 128
    nt = triton.cdiv(tokens, BT)
    mp = meta if meta is not None else gc
    if kind not in ("mono", "stack") and _SPLIT_FAILED and variant is None:
        kind, ws = "mono", 8
    if kind in ("mono", "stack"):
        gdn_chunk_wy_kernel[(nt, hv)](
            mixed, a, b, a_log, dt_bias, mp, qn, kn, w, u, gc, aqk,
            tokens, mixed.stride(0), a.stride(0), b.stride(0), H=heads, HV=hv, K=kd, V=vd, BC=BC,
            HAS_META=meta is not None, WIDEN=_WIDEN, STACK=kind == "stack",
            num_warps=warps if warps is not None else ws if ws else _opts()[1],
        )
        return
    try:
        _wy_split(mixed, a, b, a_log, dt_bias, meta, mp, heads, hv, kd, vd, nt, tokens, qn, kn, w, u, gc,
                  aqk, kind, ws, ww, bw, s2)
    except Exception as exc:  # noqa: BLE001 - a compiler that rejects the split form: the fused one
        if _SPLIT_FAILED or variant is not None:  # an explicit form (the tests) must build
            raise
        _SPLIT_FAILED.append(f"{type(exc).__name__}: {str(exc)[:300]}")
        print(f"engine: GDN WY split form failed ({_SPLIT_FAILED[0]}); using mono:8", flush=True)
        wy(mixed, a, b, a_log, dt_bias, meta, heads, qn, kn, w, u, gc, aqk, warps=warps, variant=("mono", 8))


_SPLIT_FAILED: list[str] = []  # set once when the split form's kernels fail to build or launch


def _wy_split(mixed, a, b, a_log, dt_bias, meta, mp, heads, hv, kd, vd, nt, tokens, qn, kn, w, u, gc,
              aqk, kind, ws, ww, bw, s2):
    ai = torch.empty((tokens, hv, BT), dtype=torch.bfloat16, device=mixed.device)
    gdn_chunk_norm_kernel[(nt, heads)](
        mixed, mp, qn, kn, tokens, mixed.stride(0), H=heads, K=kd, BC=BC, HAS_META=meta is not None,
        WIDEN=_WIDEN, num_warps=NORM_WARPS,
    )
    if s2 and nt > SPLIT_NT:
        ws = s2
    gdn_chunk_solve_kernel[(nt, hv)](
        mixed, a, b, a_log, dt_bias, mp, kn, gc, ai, tokens, mixed.stride(0), a.stride(0), b.stride(0),
        H=heads, HV=hv, K=kd, BC=BC, HAS_META=meta is not None, WIDEN=_WIDEN, STACK=kind == "split",
        num_warps=ws,
    )
    gdn_chunk_wu_kernel[(nt, hv, 1 + kd // bw + vd // bw)](
        mixed, b, mp, ai, qn, kn, w, u, gc, aqk, tokens, mixed.stride(0), b.stride(0),
        H=heads, HV=hv, K=kd, V=vd, BC=BC, BW=bw, HAS_META=meta is not None, WIDEN=_WIDEN, num_warps=ww,
    )


def chunk(
    qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    heads: int,
    row: torch.Tensor | None = None,
    meta: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Core output [T, HV, V] (BF16) of the short conv + gated delta rule over the raw
    projection ``qkv`` [T, 2*H*K + HV*V] (rows may be strided; columns dense), after the
    slot's conv state ``conv_state`` [C, KC] (BF16, updated in place to its last KC real
    inputs; weight [C, 1, KC]), with in-kernel q/k l2norm, decay and beta from ``a``, ``b``
    [T, HV] (strided rows). ``state``: [HV, K, V] fp32 (one row, updated in place) or, with
    ``row`` (int64 device tensor), the [R, HV, K, V] state tensor and the row to read and
    update. ``meta``: int64 device tensor whose first entry is the real row count (rows past
    it are padding); None: all T rows are real."""
    tokens, width = qkv.shape
    hv = a.shape[-1]
    kd = vd = 128
    kc = conv_state.shape[-1]
    assert width == 2 * heads * kd + hv * vd and qkv.stride(-1) == 1
    assert conv_state.is_contiguous() and conv_state.shape[0] == width
    assert a.stride(-1) == 1 and b.stride(-1) == 1 and state.dtype == torch.float32 and state.is_contiguous()
    scale = kd ** -0.5 if scale is None else scale
    dev = qkv.device
    qn = torch.empty((tokens, heads, kd), dtype=torch.bfloat16, device=dev)
    kn = torch.empty_like(qn)
    w = torch.empty((tokens, hv, kd), dtype=torch.bfloat16, device=dev)
    u = torch.empty((tokens, hv, vd), dtype=torch.bfloat16, device=dev)
    gc = torch.empty((tokens, hv), dtype=torch.float32, device=dev)
    aqk = torch.empty((tokens, hv, BT), dtype=torch.bfloat16, device=dev)
    o = torch.empty((tokens, hv, vd), dtype=torch.bfloat16, device=dev)
    bv, w1, w2, stages = _opts()
    # The short conv + silu and the conv state update (``prefill_kernels.gdn_conv_``).
    mixed = gdn_conv_(qkv, conv_state, conv_weight, meta=meta)
    with on_device(qkv):
        wy(mixed, a, b, a_log, dt_bias, meta, heads, qn, kn, w, u, gc, aqk)
        gdn_chunk_state_kernel[(vd // bv, hv)](
            qn, kn, w, u, gc, aqk, o, state, row if row is not None else gc, meta if meta is not None else gc,
            tokens, scale, H=heads, HV=hv, K=kd, V=vd, BT=BT, BV=bv, HAS_ROW=row is not None,
            HAS_META=meta is not None, STAGES=stages, WIDEN=_WIDEN, num_warps=w2,
        )
    return o


def reference(qkv, conv_state, conv_weight, a, b, a_log, dt_bias, state, heads, count=None, scale=None):
    """The conv (``prefill_kernels.gdn_conv_reference``) then the token-by-token gated delta
    rule in fp32 (the math ``chunk`` computes): (core [T, HV, V] fp32, final state [HV, K, V]
    fp32, new conv state); rows at or past ``count`` are padding."""
    import torch.nn.functional as F

    from engine.fast.prefill_kernels import gdn_conv_reference

    tokens = qkv.shape[0]
    hv = a.shape[-1]
    kd = vd = 128
    count = tokens if count is None else count
    scale = kd ** -0.5 if scale is None else scale
    mixed, _ = gdn_conv_reference(qkv, conv_state, conv_weight)
    _, new_conv = gdn_conv_reference(qkv[:count], conv_state, conv_weight)
    x = mixed.float()
    q = x[:, : heads * kd].reshape(tokens, heads, kd)
    k = x[:, heads * kd : 2 * heads * kd].reshape(tokens, heads, kd)
    v = x[:, 2 * heads * kd :].reshape(tokens, hv, vd)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    r = hv // heads
    q = q.repeat_interleave(r, dim=1)
    k = k.repeat_interleave(r, dim=1)
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    beta = b.float().sigmoid().bfloat16().float()
    s = state.float().clone()
    out = torch.zeros(tokens, hv, vd)
    for t in range(count):
        s = s * g[t].exp()[:, None, None]
        mem = torch.einsum("hk,hkv->hv", k[t], s)
        delta = (v[t] - mem) * beta[t][:, None]
        s = s + torch.einsum("hk,hv->hkv", k[t], delta)
        out[t] = torch.einsum("hk,hkv->hv", q[t] * scale, s)
    return out, s, new_conv
