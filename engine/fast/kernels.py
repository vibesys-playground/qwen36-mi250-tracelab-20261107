"""Decode kernels of the engine's own Qwen3.6 forward, each with a PyTorch reference.

Three operations dominate a decode step once the model's state lives in fixed
slots, and each gets one Triton kernel here, used on GPUs:

- ``moe_decode``: the routed experts for a few tokens. Each (token, expert)
  pair reads its expert's ``gate_up_proj`` and ``down_proj`` rows in place,
  where Transformers' ``batched_mm`` copies every selected expert's weights
  into a new tensor first. The down kernel also applies the routing weights and
  sums a token's experts, so nothing but activations is written.
- ``gdn_recurrent_``: one gated delta rule step that updates the fp32
  recurrent state of each sequence's slot in place.
- ``paged_decode_attention``: one query token per sequence against its paged
  keys and values, split over the context (flash-decoding), its length read
  from device memory so one captured graph serves every length.

The PyTorch versions compute the same thing with ordinary tensor ops; they run
on the CPU (and under ``use_triton=False``) and are the reference the kernels
are tested against. Shapes follow the checkpoint's config; nothing is fixed to
one model size.
"""

from __future__ import annotations

import contextlib
import os

import torch
from torch.nn import functional as F

try:  # Triton ships with the ROCm and CUDA builds of torch, not with the CPU build.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the torch build
    triton = None
    tl = None

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def on_device(t: torch.Tensor):
    """Make ``t``'s GPU current: Triton launches on the current device, not on its tensors'.

    Without this a kernel for a layer on another GPU runs on the current one with foreign
    pointers (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on a multi-GPU layer split).
    """
    return torch.cuda.device(t.device) if t.device.type == "cuda" else contextlib.nullcontext()


def triton_usable(device: torch.device) -> bool:
    """True when the Triton kernels can run on tensors of ``device``."""
    if triton is None:
        return False
    return device.type == "cuda" or INTERPRET


# --------------------------------------------------------------------------------------------
# Routed experts for decode-sized batches
# --------------------------------------------------------------------------------------------


def moe_decode_reference(
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Transformers' ``batched_mm`` experts math: gather the selected experts, two matmuls."""
    tokens, top_k = top_k_index.shape
    ids = top_k_index.reshape(-1)
    xs = x.repeat_interleave(top_k, dim=0)
    proj = torch.bmm(gate_up[ids], xs.unsqueeze(-1)).squeeze(-1)
    gate, up = proj.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    out = torch.bmm(down[ids], hidden.unsqueeze(-1)).squeeze(-1)
    out = out * top_k_weights.reshape(-1, 1)
    return out.view(tokens, top_k, -1).sum(dim=1).to(x.dtype)


if triton is not None:

    @triton.jit
    def _moe_gate_up_kernel(
        x_ptr,
        w_ptr,
        idx_ptr,
        out_ptr,
        H: tl.constexpr,
        I: tl.constexpr,
        stride_we,
        stride_wn,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out[p, n] = silu(W[e_p, n] . x_t) * (W[e_p, I + n] . x_t), for pair p of token t."""
        pair = tl.program_id(0)
        block = tl.program_id(1)
        token = pair // TOP_K
        expert = tl.load(idx_ptr + pair).to(tl.int64)
        rows = block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ok = rows < I
        base = w_ptr + expert * stride_we
        acc_gate = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_up = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k0 in range(0, H, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            col_ok = cols < H
            xv = tl.load(x_ptr + token * H + cols, mask=col_ok, other=0.0).to(tl.float32)
            mask = row_ok[:, None] & col_ok[None, :]
            wg = tl.load(base + rows[:, None] * stride_wn + cols[None, :], mask=mask, other=0.0)
            wu = tl.load(base + (rows[:, None] + I) * stride_wn + cols[None, :], mask=mask, other=0.0)
            acc_gate += tl.sum(wg.to(tl.float32) * xv[None, :], axis=1)
            acc_up += tl.sum(wu.to(tl.float32) * xv[None, :], axis=1)
        # Round where the BF16 matmuls of the reference round: projections, activation, product.
        gate = acc_gate.to(tl.bfloat16).to(tl.float32)
        up = acc_up.to(tl.bfloat16).to(tl.float32)
        act = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + pair * I + rows, (act * up).to(tl.bfloat16), mask=row_ok)

    @triton.jit
    def _moe_down_kernel(
        h_ptr,
        w_ptr,
        idx_ptr,
        wt_ptr,
        out_ptr,
        H: tl.constexpr,
        I: tl.constexpr,
        stride_we,
        stride_wn,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out[t, n] = sum_j w[t, j] * (W[e_tj, n] . h[t, j]): down projection and combine."""
        token = tl.program_id(0)
        block = tl.program_id(1)
        rows = block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ok = rows < H
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for j in tl.static_range(TOP_K):
            pair = token * TOP_K + j
            expert = tl.load(idx_ptr + pair).to(tl.int64)
            weight = tl.load(wt_ptr + pair).to(tl.float32)
            base = w_ptr + expert * stride_we
            part = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k0 in range(0, I, BLOCK_K):
                cols = k0 + tl.arange(0, BLOCK_K)
                col_ok = cols < I
                hv = tl.load(h_ptr + pair * I + cols, mask=col_ok, other=0.0).to(tl.float32)
                w = tl.load(
                    base + rows[:, None] * stride_wn + cols[None, :],
                    mask=row_ok[:, None] & col_ok[None, :],
                    other=0.0,
                )
                part += tl.sum(w.to(tl.float32) * hv[None, :], axis=1)
            acc += (part.to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + token * H + rows, acc.to(tl.bfloat16), mask=row_ok)


MOE_BLOCK_N = int(os.environ.get("ENGINE_MOE_BLOCK_N", "16"))
MOE_BLOCK_K = int(os.environ.get("ENGINE_MOE_BLOCK_K", "256"))


def moe_decode(
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    use_triton: bool,
) -> torch.Tensor:
    """Routed-expert output for ``x`` [tokens, hidden]; experts' weights are read in place."""
    if not use_triton:
        return moe_decode_reference(x, top_k_index, top_k_weights, gate_up, down)
    tokens, top_k = top_k_index.shape
    experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    assert gate_up.stride(-1) == 1 and down.stride(-1) == 1, "expert weights must be row-major"
    x = x.contiguous()
    ids = top_k_index.reshape(-1).contiguous()
    weights = top_k_weights.reshape(-1).contiguous()
    act = torch.empty((tokens * top_k, inter), dtype=torch.bfloat16, device=x.device)
    with on_device(x):
        _moe_gate_up_kernel[(tokens * top_k, triton.cdiv(inter, MOE_BLOCK_N))](
            x,
            gate_up,
            ids,
            act,
            hidden,
            inter,
            gate_up.stride(0),
            gate_up.stride(1),
            TOP_K=top_k,
            BLOCK_N=MOE_BLOCK_N,
            BLOCK_K=MOE_BLOCK_K,
        )
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    with on_device(act):
        _moe_down_kernel[(tokens, triton.cdiv(hidden, MOE_BLOCK_N))](
            act,
            down,
            ids,
            weights,
            out,
            hidden,
            inter,
            down.stride(0),
            down.stride(1),
            TOP_K=top_k,
            BLOCK_N=MOE_BLOCK_N,
            BLOCK_K=min(MOE_BLOCK_K, triton.next_power_of_2(inter)),
        )
    return out


# --------------------------------------------------------------------------------------------
# Gated DeltaNet: one recurrent step per sequence, state updated in its slot
# --------------------------------------------------------------------------------------------


def gdn_recurrent_reference(
    state: torch.Tensor,
    slots: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Transformers' ``torch_recurrent_gated_delta_rule`` for one token, on ``state[slots]``.

    ``q`` (already L2-normalized and scaled) and ``k`` (L2-normalized) are
    [batch, v_heads, k_dim] fp32, ``v`` [batch, v_heads, v_dim], ``g`` and
    ``beta`` [batch, v_heads]. Returns the output [batch, v_heads, v_dim].
    """
    s = state.index_select(0, slots)
    s = s * g.exp()[..., None, None]
    kv_mem = (s * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    s = s + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (s * q.unsqueeze(-1)).sum(dim=-2)
    state.index_copy_(0, slots, s)
    return out


if triton is not None:

    @triton.jit
    def _gdn_step_kernel(
        state_ptr,
        slot_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        out_ptr,
        V_HEADS,
        stride_slot,
        stride_head,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BV: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        vblock = tl.program_id(2)
        slot = tl.load(slot_ptr + row).to(tl.int64)
        kk = tl.arange(0, DK)
        vv = vblock * BV + tl.arange(0, BV)
        base = state_ptr + slot * stride_slot + head * stride_head
        s = tl.load(base + kk[:, None] * DV + vv[None, :])
        rh = row * V_HEADS + head
        qv = tl.load(q_ptr + rh * DK + kk)
        kv = tl.load(k_ptr + rh * DK + kk)
        vvec = tl.load(v_ptr + rh * DV + vv)
        decay = tl.exp(tl.load(g_ptr + rh))
        beta = tl.load(beta_ptr + rh)
        s = s * decay
        kv_mem = tl.sum(s * kv[:, None], axis=0)
        delta = (vvec - kv_mem) * beta
        s = s + kv[:, None] * delta[None, :]
        out = tl.sum(s * qv[:, None], axis=0)
        tl.store(base + kk[:, None] * DV + vv[None, :], s)
        tl.store(out_ptr + rh * DV + vv, out)


def gdn_recurrent_(
    state: torch.Tensor,
    slots: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_triton: bool,
) -> torch.Tensor:
    """One gated delta rule step; updates ``state[slots]`` in place and returns the output."""
    if not use_triton:
        return gdn_recurrent_reference(state, slots, q, k, v, g, beta)
    batch, heads, dk = q.shape
    dv = v.shape[-1]
    assert state.is_contiguous() and state.dtype == torch.float32
    out = torch.empty((batch, heads, dv), dtype=torch.float32, device=q.device)
    bv = min(32, dv)
    with on_device(state):
        _gdn_step_kernel[(batch, heads, triton.cdiv(dv, bv))](
            state,
            slots,
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            out,
            heads,
            state.stride(0),
            state.stride(1),
            DK=dk,
            DV=dv,
            BV=bv,
        )
    return out


# --------------------------------------------------------------------------------------------
# Paged decode attention (grouped-query), lengths read from device memory
# --------------------------------------------------------------------------------------------


def paged_decode_attention_reference(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attention of one query per row over its first ``lengths[row]`` cached tokens.

    ``k_pool``/``v_pool`` are [pages, kv_heads, page, head_dim]; row ``r`` reads
    the pages ``page_table[slots[r]]``. Synchronizes with the host (for the
    longest length), so this is for the CPU and for tests.
    """
    batch, q_heads, dim = q.shape
    kv_heads, page = k_pool.shape[1], k_pool.shape[2]
    longest = int(lengths.max())
    pages = -(-longest // page)
    table = page_table[slots, :pages].long()
    keys = k_pool[table].permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, pages * page, dim)
    values = v_pool[table].permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, pages * page, dim)
    keys, values = keys[:, :, :longest], values[:, :, :longest]
    mask = torch.arange(longest, device=q.device)[None, :] < lengths[:, None]
    out = F.scaled_dot_product_attention(
        q.unsqueeze(2),
        keys,
        values,
        attn_mask=mask[:, None, None, :],
        scale=scale,
        enable_gqa=q_heads != kv_heads,
    )
    return out.squeeze(2)


if triton is not None:

    @triton.jit
    def _paged_attn_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        table_ptr,
        slot_ptr,
        len_ptr,
        acc_ptr,
        ml_ptr,
        scale,
        Q_HEADS,
        stride_table,
        stride_page,
        stride_head,
        stride_tok,
        GROUP: tl.constexpr,
        GROUP_PAD: tl.constexpr,
        D: tl.constexpr,
        PAGE: tl.constexpr,
        SPLITS: tl.constexpr,
        DOT_F32: tl.constexpr,
    ):
        row = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)
        slot = tl.load(slot_ptr + row).to(tl.int64)
        length = tl.load(len_ptr + row)
        pages = tl.cdiv(length, PAGE)
        per_split = tl.cdiv(pages, SPLITS)
        first = split * per_split
        last = tl.minimum(first + per_split, pages)
        g = tl.arange(0, GROUP_PAD)
        g_ok = g < GROUP
        d = tl.arange(0, D)
        heads = kv_head * GROUP + g
        q = tl.load(q_ptr + (row * Q_HEADS + heads)[:, None] * D + d[None, :], mask=g_ok[:, None], other=0.0)
        if DOT_F32:  # Triton's CPU interpreter multiplies BF16 dot operands wrongly
            q = q.to(tl.float32)
        m = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
        l = tl.zeros([GROUP_PAD], dtype=tl.float32)
        acc = tl.zeros([GROUP_PAD, D], dtype=tl.float32)
        t = tl.arange(0, PAGE)
        p = first
        while p < last:  # a while loop: its bounds come from device memory
            page = tl.load(table_ptr + slot * stride_table + p).to(tl.int64)
            t_ok = (p * PAGE + t) < length
            offs = page * stride_page + kv_head * stride_head + t[:, None] * stride_tok + d[None, :]
            keys = tl.load(k_ptr + offs, mask=t_ok[:, None], other=0.0)
            if DOT_F32:
                keys = keys.to(tl.float32)
            s = tl.dot(q, tl.trans(keys)) * scale
            s = tl.where(t_ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp(m - m_new)
            prob = tl.exp(s - m_new[:, None])
            l = l * alpha + tl.sum(prob, axis=1)
            values = tl.load(v_ptr + offs, mask=t_ok[:, None], other=0.0)
            if DOT_F32:
                values = values.to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(prob.to(values.dtype), values)
            m = m_new
            p += 1
        out_rows = (row * Q_HEADS + heads) * SPLITS + split
        tl.store(acc_ptr + out_rows[:, None] * D + d[None, :], acc, mask=g_ok[:, None])
        tl.store(ml_ptr + out_rows * 2, m, mask=g_ok)
        tl.store(ml_ptr + out_rows * 2 + 1, l, mask=g_ok)


ATTN_SPLITS = int(os.environ.get("ENGINE_ATTN_SPLITS", "32"))


def paged_decode_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    *,
    use_triton: bool,
) -> torch.Tensor:
    """[batch, q_heads, head_dim] attention output; no host synchronization on the Triton path."""
    if not use_triton:
        return paged_decode_attention_reference(q, k_pool, v_pool, page_table, slots, lengths, scale)
    batch, q_heads, dim = q.shape
    kv_heads, page = k_pool.shape[1], k_pool.shape[2]
    group = q_heads // kv_heads
    splits = ATTN_SPLITS
    acc = torch.empty((batch, q_heads, splits, dim), dtype=torch.float32, device=q.device)
    ml = torch.empty((batch, q_heads, splits, 2), dtype=torch.float32, device=q.device)
    assert k_pool.stride() == v_pool.stride() and k_pool.stride(-1) == 1
    with on_device(q):
        _paged_attn_kernel[(batch, kv_heads, splits)](
            q.contiguous(),
            k_pool,
            v_pool,
            page_table,
            slots,
            lengths,
            acc,
            ml,
            scale,
            q_heads,
            page_table.stride(0),
            k_pool.stride(0),
            k_pool.stride(1),
            k_pool.stride(2),
            GROUP=group,
            GROUP_PAD=max(16, triton.next_power_of_2(group)),
            D=dim,
            PAGE=page,
            SPLITS=splits,
            DOT_F32=INTERPRET,
        )
    m, l = ml[..., 0], ml[..., 1]
    top = m.max(dim=-1, keepdim=True).values
    w = torch.exp(m - top)
    out = (acc * w[..., None]).sum(dim=2) / (l * w).sum(dim=-1)[..., None]
    return out.to(q.dtype)


# --------------------------------------------------------------------------------------------
# Fused decode kernels: fewer launches per layer (a captured step is launch-bound otherwise)
# --------------------------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _add_rms_kernel(
        x_ptr, h_ptr, w_ptr, sum_ptr, out_ptr, eps, N: tl.constexpr, BLOCK: tl.constexpr,
        HAS_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        ok = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=ok, other=0.0).to(tl.float32)
        if HAS_H:  # the residual add, rounded to BF16 as the BF16 tensor add rounds
            h = tl.load(h_ptr + row * N + cols, mask=ok, other=0.0).to(tl.float32)
            x = (x + h).to(tl.bfloat16).to(tl.float32)
            tl.store(sum_ptr + row * N + cols, x.to(tl.bfloat16), mask=ok)
        var = tl.sum(x * x, axis=0) / N
        w = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
        y = x * tl.rsqrt(var + eps) * (1.0 + w)
        tl.store(out_ptr + row * N + cols, y.to(tl.bfloat16), mask=ok)

    @triton.jit
    def _gdn_conv_kernel(
        state_ptr, slot_ptr, x_ptr, w_ptr, out_ptr, C, stride_slot, K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Short causal convolution step: shift the slot's window, append x, silu(conv)."""
        row = tl.program_id(0)
        block = tl.program_id(1)
        slot = tl.load(slot_ptr + row).to(tl.int64)
        c = block * BLOCK + tl.arange(0, BLOCK)
        ok = c < C
        base = state_ptr + slot * stride_slot + c * K
        xn = tl.load(x_ptr + row * C + c, mask=ok, other=0.0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(K - 1):
            sj = tl.load(base + j + 1, mask=ok, other=0.0)
            acc += sj.to(tl.float32) * tl.load(w_ptr + c * K + j, mask=ok, other=0.0).to(tl.float32)
            tl.store(base + j, sj, mask=ok)
        acc += xn.to(tl.float32) * tl.load(w_ptr + c * K + K - 1, mask=ok, other=0.0).to(tl.float32)
        tl.store(base + K - 1, xn, mask=ok)
        y = acc.to(tl.bfloat16).to(tl.float32)
        y = (y * tl.sigmoid(y)).to(tl.bfloat16)
        tl.store(out_ptr + row * C + c, y, mask=ok)

    @triton.jit
    def _gdn_fused_kernel(
        state_ptr, slot_ptr, mixed_ptr, a_ptr, b_ptr, z_ptr, alog_ptr, dtb_ptr, nw_ptr, out_ptr,
        C, V_HEADS, stride_slot, stride_head, eps, scale, RATIO: tl.constexpr,
        KEY_DIM: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    ):
        """One v-head of one sequence: q/k L2 norm, decay and beta, the delta rule on the slot's
        fp32 state (in place), and the gated RMSNorm of the output with z."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        k_head = head // RATIO
        slot = tl.load(slot_ptr + row).to(tl.int64)
        kk = tl.arange(0, DK)
        vv = tl.arange(0, DV)
        base_in = mixed_ptr + row * C
        q = tl.load(base_in + k_head * DK + kk).to(tl.float32)
        k = tl.load(base_in + KEY_DIM + k_head * DK + kk).to(tl.float32)
        v = tl.load(base_in + 2 * KEY_DIM + head * DV + vv).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        rh = row * V_HEADS + head
        beta = tl.sigmoid(tl.load(b_ptr + rh).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        a = tl.load(a_ptr + rh).to(tl.float32) + tl.load(dtb_ptr + head).to(tl.float32)
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
        zz = tl.load(z_ptr + rh * DV + vv).to(tl.float32)
        tl.store(out_ptr + rh * DV + vv, (y * zz * tl.sigmoid(zz)).to(tl.bfloat16))


def add_rms(
    x: torch.Tensor, h: torch.Tensor | None, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(x + h, RMSNorm(x + h) * (1 + weight)) in one launch; ``h`` None skips the add."""
    rows, n = x.shape
    out = torch.empty_like(x)
    total = torch.empty_like(x) if h is not None else x
    with on_device(x):
        _add_rms_kernel[(rows,)](
            x, h if h is not None else x, weight, total, out, eps, N=n,
            BLOCK=triton.next_power_of_2(n), HAS_H=h is not None,
        )
    return total, out


def gdn_conv_(
    state: torch.Tensor, slots: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Convolution step for [batch, conv_dim] ``x`` on ``state[slots]`` (updated in place)."""
    batch, channels = x.shape
    kernel = state.shape[-1]
    out = torch.empty_like(x)
    block = 256
    with on_device(state):
        _gdn_conv_kernel[(batch, triton.cdiv(channels, block))](
            state, slots, x.contiguous(), weight.contiguous(), out, channels, state.stride(0),
            K=kernel, BLOCK=block,
        )
    return out


def gdn_fused_(
    state: torch.Tensor,
    slots: torch.Tensor,
    mixed: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    m: torch.nn.Module,
) -> torch.Tensor:
    """Gated DeltaNet decode after the convolution, for module ``m``: [batch, v_heads * v_dim]."""
    batch = mixed.shape[0]
    heads, dk, dv = m.num_v_heads, m.head_k_dim, m.head_v_dim
    out = torch.empty((batch, heads * dv), dtype=mixed.dtype, device=mixed.device)
    with on_device(state):
        _gdn_fused_kernel[(batch, heads)](
            state, slots, mixed.contiguous(), a.contiguous(), b.contiguous(), z.contiguous(),
            m.A_log, m.dt_bias, m.norm.weight, out, mixed.shape[1], heads, state.stride(0),
            state.stride(1), m.norm.variance_epsilon, dk ** -0.5,
            RATIO=heads // m.num_k_heads, KEY_DIM=m.key_dim, DK=dk, DV=dv, num_warps=8,
        )
    return out
