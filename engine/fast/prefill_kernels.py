"""Fused prefill kernels: fewer launches and no fp32 round trips around the weight GEMMs.

A prefill pass of the eager forward launches ~5,000 kernels per GPU, most of them small
element-wise ops around the weight GEMMs, and runs attention through SDPA's math path
(fp32 matmuls over gathered keys with an explicit mask). ``ENGINE_PREFILL_FUSED`` (default
1; 0 restores the eager ops) replaces, in ``FastModel``'s prefill passes only:

- ``prefill_attention``: causal attention of a segment's queries over its paged KV cache,
  read in place through the slot's page list (no gather, no mask tensor, no GQA expansion),
  flash-style (BF16 dot operands, fp32 scores, softmax and accumulation, the decode
  kernel's numerics), with the output gate ``* sigmoid(gate)`` in the epilogue.
- ``reduce_rms``: the residual add of a layer's output (summed over the tensor-parallel
  ranks first, in fp32 rounded to BF16 as ``tp.Group.allreduce``), then the next RMSNorm
  (``_rms``: fp32 normalize, ``* (1 + w)``, one rounding), one launch per call.
- ``moe``: the MoE block around the grouped expert kernels of ``moe_prefill``: the router
  softmax (fp32), top-k and renormalization with the per-expert counts in one kernel, the
  expert bookkeeping (offsets, tile counts, the pairs' order by expert) in two more, the
  shared expert's ``silu(gate) * up`` in one, and the combine (the top-k rows' fp32 sum
  rounded once, plus ``sigmoid(shared_gate) * shared``, each rounding as the eager ops) in
  one. Every routed expert is computed as before, by the same kernels.

Each has a PyTorch reference (the eager ops it replaces) for the CPU tests.
"""

from __future__ import annotations

import os

import torch
from torch.nn import functional as F

from engine.fast.kernels import INTERPRET, on_device, tl, triton
from engine.fast import flash_prefill, moe_prefill

_WIDEN = INTERPRET  # see ``moe_prefill``: the interpreter's BF16 dot and rounding


def enabled() -> bool:
    """``ENGINE_PREFILL_FUSED`` (default on)."""
    return os.environ.get("ENGINE_PREFILL_FUSED", "1") != "0"


def flash_enabled() -> bool:
    """``ENGINE_PREFILL_FLASH`` (default on): ``prefill_attention`` runs the MFMA-tiled flash
    kernel of ``engine.fast.flash_prefill`` (plan 35); 0 restores ``_prefill_attn_kernel``."""
    return os.environ.get("ENGINE_PREFILL_FLASH", "1") != "0"


_FLASH = flash_enabled()


if triton is not None:
    _bf16 = moe_prefill._bf16

    # ------------------------------------------------------------------------------ attention

    @triton.jit(do_not_specialize=["T", "start", "NR"])
    def _prefill_attn_kernel(
        q_ptr, k_ptr, v_ptr, table_ptr, gate_ptr, out_ptr, T, start, scale,
        stride_page, stride_head, stride_tok, meta_ptr, NR,
        QH: tl.constexpr, GROUP: tl.constexpr, BLOCK_R: tl.constexpr, D: tl.constexpr,
        PAGE: tl.constexpr, WIDEN: tl.constexpr, HAS_META: tl.constexpr,
    ):
        """Rows ``[tile * BLOCK_R, +BLOCK_R)`` x the GROUP query heads of one KV head.

        ``NR`` rows are computed and stored. With ``HAS_META`` the real row count and the
        start position are read from ``meta_ptr`` ([count, start], int64, on the device; see
        ``prefill_attention``), and rows ``>= count`` (padding) attend as row ``count - 1``:
        finite, and never read by a real row."""
        tile = tl.program_id(0)
        kvh = tl.program_id(1)
        if HAS_META:
            n_real = tl.load(meta_ptr).to(tl.int32)
            q0 = tl.load(meta_ptr + 1).to(tl.int32)
        else:
            n_real = T
            q0 = start
        M: tl.constexpr = BLOCK_R * GROUP
        mi = tl.arange(0, M)
        r = tile * BLOCK_R + mi // GROUP
        head = kvh * GROUP + mi % GROUP
        r_ok = r < NR
        d = tl.arange(0, D)
        qo = (r.to(tl.int64) * QH + head)[:, None] * D + d[None, :]
        q = tl.load(q_ptr + qo, mask=r_ok[:, None], other=0.0)
        if WIDEN:
            q = q.to(tl.float32)
        q_pos = q0 + tl.minimum(r, n_real - 1)
        end = q0 + tl.minimum(tile * BLOCK_R + BLOCK_R, n_real)  # keys [0, end) for this tile
        m = tl.full([M], float("-inf"), dtype=tl.float32)
        l = tl.zeros([M], dtype=tl.float32)
        acc = tl.zeros([M, D], dtype=tl.float32)
        t = tl.arange(0, PAGE)
        pages = tl.cdiv(end, PAGE)
        for p in range(0, pages):
            page = tl.load(table_ptr + p).to(tl.int64)
            k_pos = p * PAGE + t
            k_ok = k_pos < end
            offs = page * stride_page + kvh * stride_head + t[:, None] * stride_tok + d[None, :]
            keys = tl.load(k_ptr + offs, mask=k_ok[:, None], other=0.0)
            if WIDEN:
                keys = keys.to(tl.float32)
            s = tl.dot(q, tl.trans(keys)) * scale
            s = tl.where(k_pos[None, :] <= q_pos[:, None], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp(m - m_new)
            prob = tl.exp(s - m_new[:, None])
            l = l * alpha + tl.sum(prob, axis=1)
            values = tl.load(v_ptr + offs, mask=k_ok[:, None], other=0.0)
            if WIDEN:
                acc = acc * alpha[:, None] + tl.dot(_bf16(prob, WIDEN), values.to(tl.float32))
            else:
                acc = acc * alpha[:, None] + tl.dot(prob.to(values.dtype), values)
            m = m_new
        o = _bf16(acc / l[:, None], WIDEN)  # SDPA's BF16 output
        g = tl.load(gate_ptr + qo, mask=r_ok[:, None], other=0.0).to(tl.float32)
        sig = _bf16(tl.sigmoid(g), WIDEN)  # torch.sigmoid on BF16
        tl.store(out_ptr + qo, _bf16(o * sig, WIDEN).to(tl.bfloat16), mask=r_ok[:, None])

    # ------------------------------------------------------------------------ residual + norm

    @triton.jit
    def _reduce_rms_kernel(
        res_ptr, h_ptr, peer_ptr, w_ptr, sum_ptr, out_ptr, eps,
        N: tl.constexpr, BLOCK: tl.constexpr, HAS_H: tl.constexpr, HAS_PEER: tl.constexpr,
        HAS_W: tl.constexpr, WIDEN: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < N
        x = tl.load(res_ptr + row * N + cols, mask=ok, other=0.0).to(tl.float32)
        if HAS_H:
            h = tl.load(h_ptr + row * N + cols, mask=ok, other=0.0).to(tl.float32)
            if HAS_PEER:  # the all-reduce: (mine + theirs) in fp32, rounded to BF16
                h = _bf16(h + tl.load(peer_ptr + row * N + cols, mask=ok, other=0.0).to(tl.float32), WIDEN)
            x = _bf16(x + h, WIDEN)  # the BF16 residual add
            tl.store(sum_ptr + row * N + cols, x.to(tl.bfloat16), mask=ok)
        if HAS_W:
            var = tl.sum(x * x, axis=0) / N
            w = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
            y = x * tl.rsqrt(var + eps) * (1.0 + w)
            tl.store(out_ptr + row * N + cols, _bf16(y, WIDEN).to(tl.bfloat16), mask=ok)

    # ------------------------------------------------------------------------------ MoE block

    @triton.jit
    def _route_kernel(
        logit_ptr, w_ptr, i_ptr, counts_ptr, E: tl.constexpr, K: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        """softmax (fp32) over the row's E logits, top K (ties to the lower expert), the
        weights renormalized to sum 1 and rounded to BF16; per-expert pair counts."""
        row = tl.program_id(0).to(tl.int64)
        e = tl.arange(0, E)
        z = tl.load(logit_ptr + row * E + e).to(tl.float32)
        z = tl.exp(z - tl.max(z, axis=0))
        probs = z / tl.sum(z, axis=0)
        kk = tl.arange(0, K)
        top_w = tl.zeros([K], dtype=tl.float32)
        top_i = tl.zeros([K], dtype=tl.int32)
        work = probs
        for j in tl.static_range(K):
            best = tl.max(work, axis=0)
            idx = tl.min(tl.where(work == best, e, E), axis=0)
            top_w = tl.where(kk == j, best, top_w)
            top_i = tl.where(kk == j, idx, top_i)
            work = tl.where(e == idx, -1.0, work)
        total = tl.sum(top_w, axis=0)
        tl.store(w_ptr + row * K + kk, _bf16(top_w / total, WIDEN).to(tl.bfloat16))
        tl.store(i_ptr + row * K + kk, top_i.to(tl.int64))
        tl.atomic_add(counts_ptr + top_i, tl.full([K], 1, tl.int32))

    @triton.jit
    def _expert_offsets_kernel(
        counts_ptr, starts_ptr, cursor_ptr, tile_ends_ptr, E: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        e = tl.arange(0, E)
        c = tl.load(counts_ptr + e)
        ends = tl.cumsum(c, axis=0)
        tl.store(starts_ptr + e, ends - c)
        tl.store(cursor_ptr + e, ends - c)
        tl.store(tile_ends_ptr + e, tl.cumsum((c + BLOCK_M - 1) // BLOCK_M, axis=0))

    @triton.jit(do_not_specialize=["P"])
    def _expert_order_kernel(ids_ptr, cursor_ptr, order_ptr, P, BLOCK: tl.constexpr):
        """Each pair takes the next row of its expert's run. The order within an expert's
        run depends on scheduling; no result does: a pair's expert rows are computed by
        itself (row independent GEMM tiles) and written at the pair's own index."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = p < P
        ids = tl.load(ids_ptr + p, mask=ok, other=0)
        at = tl.atomic_add(cursor_ptr + ids, tl.full([BLOCK], 1, tl.int32), mask=ok)
        tl.store(order_ptr + at, p.to(tl.int32), mask=ok)

    @triton.jit(do_not_specialize=["NUMEL"])
    def _silu_mul_kernel(g_ptr, u_ptr, out_ptr, NUMEL, BLOCK: tl.constexpr, WIDEN: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = i < NUMEL
        g = tl.load(g_ptr + i, mask=ok, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + i, mask=ok, other=0.0).to(tl.float32)
        a = _bf16(g * tl.sigmoid(g), WIDEN)  # F.silu on BF16
        tl.store(out_ptr + i, _bf16(a * u, WIDEN).to(tl.bfloat16), mask=ok)

    @triton.jit
    def _moe_combine_kernel(
        pairs_ptr, sg_ptr, shared_ptr, out_ptr, H: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """bf16(sum_k pairs[t, k]) + bf16(bf16(sigmoid(sg[t])) * shared[t]), rounded as eager."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = cols < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(K):
            acc += tl.load(pairs_ptr + (row * K + k) * H + cols, mask=ok, other=0.0)
        routed = _bf16(acc, WIDEN)
        gate = _bf16(tl.sigmoid(tl.load(sg_ptr + row).to(tl.float32)), WIDEN)
        shared = tl.load(shared_ptr + row * H + cols, mask=ok, other=0.0).to(tl.float32)
        y = _bf16(routed + _bf16(gate * shared, WIDEN), WIDEN)
        tl.store(out_ptr + row * H + cols, y.to(tl.bfloat16), mask=ok)


if triton is not None:

    # ------------------------------------------------------------------------- Gated DeltaNet

    @triton.jit(do_not_specialize=["T"])
    def _gdn_conv_kernel(
        qkv_ptr, state_ptr, w_ptr, out_ptr, T, C, K: tl.constexpr, BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """silu(causal depthwise conv) of [T, C] inputs after the slot's last K inputs (its
        conv state [C, K]): token t reads inputs t-K+1..t. The state is not changed here."""
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        ok = (t < T)[:, None] & (c < C)[None, :]
        acc = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
        for j in tl.static_range(K):
            src = t + 1 + j - K  # the input row; < 0: the state's column src + K
            from_x = (src >= 0)[:, None]
            xv = tl.load(qkv_ptr + src.to(tl.int64)[:, None] * C + c[None, :], mask=ok & from_x, other=0.0)
            sv = tl.load(state_ptr + c[None, :] * K + (src + K)[:, None], mask=ok & (~from_x), other=0.0)
            w = tl.load(w_ptr + c * K + j, mask=c < C, other=0.0).to(tl.float32)
            acc += tl.where(from_x, xv, sv).to(tl.float32) * w[None, :]
        y = _bf16(acc, WIDEN)  # the BF16 convolution output
        y = _bf16(y * tl.sigmoid(y), WIDEN)  # F.silu on BF16
        tl.store(out_ptr + t.to(tl.int64)[:, None] * C + c[None, :], y.to(tl.bfloat16), mask=ok)

    @triton.jit(do_not_specialize=["T"])
    def _gdn_conv_state_kernel(qkv_ptr, state_ptr, T, C, meta_ptr, K: tl.constexpr, BLOCK_C: tl.constexpr,
                               HAS_META: tl.constexpr):
        """The conv state after T inputs: the last K of (state ++ inputs). With ``HAS_META``,
        T is the real row count read from ``meta_ptr`` (rows past it are padding)."""
        if HAS_META:
            n = tl.load(meta_ptr).to(tl.int32)
        else:
            n = T
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        i = tl.arange(0, K)
        src = n - K + i  # input row of new column i; < 0: old column n + i
        c_ok = (c < C)[:, None]
        from_x = (src >= 0)[None, :]
        xv = tl.load(qkv_ptr + src.to(tl.int64)[None, :] * C + c[:, None], mask=c_ok & from_x, other=0.0)
        sv = tl.load(state_ptr + c[:, None] * K + (n + i)[None, :], mask=c_ok & (~from_x), other=0.0)
        tl.store(state_ptr + c[:, None] * K + i[None, :], tl.where(from_x, xv, sv), mask=c_ok)

    @triton.jit(do_not_specialize=["NUMEL"])
    def _gdn_gates_kernel(
        a_ptr, b_ptr, alog_ptr, dtb_ptr, g_ptr, beta_ptr, NUMEL, HV: tl.constexpr,
        BLOCK: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """g = -exp(A_log) * softplus(a + dt_bias) (fp32), beta = sigmoid(b) (BF16)."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < NUMEL
        h = i % HV
        a = tl.load(a_ptr + i, mask=ok, other=0.0).to(tl.float32) + tl.load(dtb_ptr + h, mask=ok, other=0.0).to(tl.float32)
        softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        g = -tl.exp(tl.load(alog_ptr + h, mask=ok, other=0.0).to(tl.float32)) * softplus
        tl.store(g_ptr + i, g, mask=ok)
        b = tl.load(b_ptr + i, mask=ok, other=0.0).to(tl.float32)
        tl.store(beta_ptr + i, _bf16(tl.sigmoid(b), WIDEN).to(tl.bfloat16), mask=ok)

    @triton.jit(do_not_specialize=["ROWS"])
    def _gated_rms_kernel(
        x_ptr, z_ptr, w_ptr, out_ptr, ROWS, eps, D: tl.constexpr, BLOCK_R: tl.constexpr,
        ROUND_Y: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """``_gated_rms`` per D-wide row: normalize (fp32, rounded), * weight, * silu(z)."""
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        d = tl.arange(0, D)
        ok = (r < ROWS)[:, None]
        offs = r.to(tl.int64)[:, None] * D + d[None, :]
        x = tl.load(x_ptr + offs, mask=ok, other=0.0).to(tl.float32)
        n = _bf16(x * tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)[:, None], WIDEN)
        y = tl.load(w_ptr + d).to(tl.float32)[None, :] * n
        if ROUND_Y:
            y = _bf16(y, WIDEN)
        z = tl.load(z_ptr + offs, mask=ok, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, _bf16(y * (z * tl.sigmoid(z)), WIDEN).to(tl.bfloat16), mask=ok)


# ================================================================================ attention


def prefill_attention_reference(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, table: torch.Tensor,
    start: int, gate: torch.Tensor, scale: float,
) -> torch.Tensor:
    """The eager path (``FastModel._attn_prefill``): gathered keys, masked SDPA, gate."""
    count, q_heads, dim = q.shape
    heads = k_pool.shape[1]
    total = start + count
    used = table[: -(-total // k_pool.shape[2])].long()
    keys = k_pool[used].permute(1, 0, 2, 3).reshape(heads, -1, dim)[None, :, :total]
    values = v_pool[used].permute(1, 0, 2, 3).reshape(heads, -1, dim)[None, :, :total]
    positions = torch.arange(start, total, device=q.device)
    mask = torch.arange(total, device=q.device)[None, :] <= positions[:, None]
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), keys, values, attn_mask=mask[None, None], scale=scale,
        enable_gqa=q_heads != heads,
    )
    out = out[0].transpose(0, 1).reshape(count, -1)
    return out * torch.sigmoid(gate)


def attention_supported(q: torch.Tensor, k_pool: torch.Tensor) -> bool:
    q_heads, dim = q.shape[1], q.shape[2]
    heads, page = k_pool.shape[1], k_pool.shape[2]
    group = q_heads // heads
    pow2 = lambda n: n > 0 and n & (n - 1) == 0  # noqa: E731
    return (
        triton is not None
        and q.dtype == torch.bfloat16
        and k_pool.dtype == torch.bfloat16
        and q_heads % heads == 0
        and pow2(group) and pow2(dim) and dim >= 16 and pow2(page) and page >= 16
        and k_pool.stride(-1) == 1
    )


def prefill_attention(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, table: torch.Tensor,
    start: int, gate: torch.Tensor, scale: float, out: torch.Tensor | None = None,
    meta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gated causal attention of ``q`` [T, q_heads, dim] at positions ``start``... over the
    first ``start + T`` cached tokens, whose pages are ``table`` (the slot's page list, its
    KV already written): [T, q_heads * dim] BF16, as ``prefill_attention_reference``.

    Device-side count and start (plan 33, for a captured pass of a fixed padded size): with
    ``meta``, an int64 tensor ``[count, start]`` on ``q``'s device, ``start`` is ignored and
    ``q``'s T rows are a bucket whose first ``count`` rows are real, at positions
    ``start .. start + count - 1``. Real rows give the same bits as a call over those rows
    alone with the same kernel and launch knobs. Padding rows (``>= count``) are computed as if
    at position ``start + count - 1`` (finite values, to be discarded); keys past
    ``start + count`` are never read.

    With ``ENGINE_PREFILL_FLASH`` (default on) this is the flash kernel of
    ``engine.fast.flash_prefill`` (plan 35), whose tile and key split come from T alone (the
    bucket size under ``meta``); its per-row bits depend on that choice (the summation order
    over keys), so a captured pass and an eager one of the same rows agree bitwise only when
    their T choose the same knobs, as the GEMMs at M = size and M = count."""
    if _FLASH:
        return flash_prefill.flash_prefill_attention(
            q, k_pool, v_pool, table, start, gate, scale, out, meta=meta
        )
    return prefill_attention_rows(q, k_pool, v_pool, table, start, gate, scale, out, meta)


def prefill_attention_rows(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, table: torch.Tensor,
    start: int, gate: torch.Tensor, scale: float, out: torch.Tensor | None = None,
    meta: torch.Tensor | None = None,
) -> torch.Tensor:
    """``_prefill_attn_kernel`` (the path before plan 35, ``ENGINE_PREFILL_FLASH=0``); its
    per-row results do not depend on T."""
    count, q_heads, dim = q.shape
    heads, page = k_pool.shape[1], k_pool.shape[2]
    group = q_heads // heads
    # Rows per program: enough programs for the GPU at a few rows, M = rows x group >= 16.
    # (Per-row results do not depend on it: a tile's extra keys are masked to exact zeros.)
    rows = 2 if count < 256 else 4
    rows = max(rows, 16 // group)
    if out is None:
        out = torch.empty((count, q_heads * dim), dtype=q.dtype, device=q.device)
    assert k_pool.stride() == v_pool.stride()
    q = q.contiguous()
    gate = gate.contiguous()
    with on_device(q):
        _prefill_attn_kernel[(triton.cdiv(count, rows), heads)](
            q, k_pool, v_pool, table, gate, out, count, start, scale,
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            meta if meta is not None else table, count,
            QH=q_heads, GROUP=group, BLOCK_R=rows, D=dim, PAGE=page, WIDEN=_WIDEN,
            HAS_META=meta is not None, num_warps=4,
        )
    return out


# ============================================================================ residual + norm


def reduce_rms_reference(
    residual: torch.Tensor, h: torch.Tensor | None, peer: torch.Tensor | None,
    weight: torch.Tensor | None, eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = residual
    if h is not None:
        if peer is not None:
            h = (h.float() + peer.float()).to(h.dtype)
        x = residual + h
    if weight is None:
        return x, None
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return x, (xf * (1.0 + weight.float())).to(x.dtype)


def reduce_rms(
    residual: torch.Tensor, h: torch.Tensor | None, peer: torch.Tensor | None,
    weight: torch.Tensor | None, eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``x = residual [+ (h [+ peer])]`` and ``RMSNorm(x) * (1 + weight)`` (None without a
    weight), as ``reduce_rms_reference``, one launch over [rows, hidden] BF16 rows."""
    residual = residual.contiguous()
    rows, n = residual.shape
    total = torch.empty_like(residual) if h is not None else residual
    out = torch.empty_like(residual) if weight is not None else None
    h_ = h.contiguous() if h is not None else residual
    p_ = peer.contiguous() if peer is not None else h_
    with on_device(residual):
        _reduce_rms_kernel[(rows,)](
            residual, h_, p_, weight if weight is not None else residual, total,
            out if out is not None else total, eps,
            N=n, BLOCK=triton.next_power_of_2(n), HAS_H=h is not None, HAS_PEER=peer is not None,
            HAS_W=weight is not None, WIDEN=_WIDEN, num_warps=8 if n >= 1024 else 4,
        )
    return total, out


# ================================================================================= MoE block


def moe_reference(x: torch.Tensor, mlp: torch.nn.Module, top_k: int) -> torch.Tensor:
    """``FastModel._moe``'s prefill math (the routed experts by ``moe_prefill_reference``)."""
    shared = mlp.shared_expert
    shared_out = F.linear(
        F.silu(F.linear(x, shared.gate_proj.weight)) * F.linear(x, shared.up_proj.weight),
        shared.down_proj.weight,
    )
    router = F.linear(x, mlp.gate.weight)
    probs = torch.softmax(router, dim=-1, dtype=torch.float)
    top_w, top_i = torch.topk(probs, top_k, dim=-1)
    top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(router.dtype)
    experts = mlp.experts
    routed = moe_prefill.moe_prefill_reference(x, top_i, top_w, experts.gate_up_proj, experts.down_proj)
    return routed + torch.sigmoid(F.linear(x, mlp.shared_expert_gate.weight)) * shared_out


def route(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(top_w BF16 [T, k], top_i int64 [T, k], per-expert pair counts int32 [E])."""
    tokens, experts = logits.shape
    top_w = torch.empty((tokens, top_k), dtype=logits.dtype, device=logits.device)
    top_i = torch.empty((tokens, top_k), dtype=torch.long, device=logits.device)
    counts = torch.zeros(experts, dtype=torch.int32, device=logits.device)
    with on_device(logits):
        _route_kernel[(tokens,)](
            logits.contiguous(), top_w, top_i, counts, E=experts, K=top_k, WIDEN=_WIDEN,
            num_warps=4,
        )
    return top_w, top_i, counts


def moe_supported(x: torch.Tensor, mlp: torch.nn.Module, top_k: int) -> bool:
    experts = mlp.experts
    e = mlp.gate.weight.shape[0]
    return (
        moe_prefill.supported(x, experts.gate_up_proj, experts.down_proj)
        and e & (e - 1) == 0
        and top_k & (top_k - 1) == 0
    )


def moe(x: torch.Tensor, mlp: torch.nn.Module, top_k: int) -> torch.Tensor:
    """The MoE block of ``FastModel._moe`` for a prefill chunk [T, hidden] (T > 16)."""
    tokens, hidden = x.shape
    dev = x.device
    x = x.contiguous()
    shared = mlp.shared_expert
    g = F.linear(x, shared.gate_proj.weight)
    u = F.linear(x, shared.up_proj.weight)
    act = torch.empty_like(g)
    numel = g.numel()
    with on_device(x):
        _silu_mul_kernel[(triton.cdiv(numel, 1024),)](g, u, act, numel, BLOCK=1024, WIDEN=_WIDEN)
    shared_out = F.linear(act, shared.down_proj.weight)
    sg = F.linear(x, mlp.shared_expert_gate.weight)  # [T, 1]
    top_w, top_i, counts = route(F.linear(x, mlp.gate.weight), top_k)
    # The grouped experts (``moe_prefill``) with the bookkeeping from the counts above.
    gate_up, down = mlp.experts.gate_up_proj, mlp.experts.down_proj
    experts, two_i, _ = gate_up.shape
    inter = two_i // 2
    pairs = tokens * top_k
    bm, bn1, bk1, bn2, bk2, warps, stages = moe_prefill.config(pairs, experts)
    bn1, bk2 = min(bn1, inter), min(bk2, inter)
    bk1, bn2 = min(bk1, hidden), min(bn2, hidden)
    meta = torch.empty((3, experts), dtype=torch.int32, device=dev)
    starts, cursor, tile_ends = meta[0], meta[1], meta[2]
    order = torch.empty(pairs, dtype=torch.int32, device=dev)
    act_r = torch.empty((pairs, inter), dtype=torch.bfloat16, device=dev)
    out_r = torch.empty((pairs, hidden), dtype=torch.float32, device=dev)
    max_tiles = triton.cdiv(pairs, bm) + experts
    log_e = max(1, (experts - 1).bit_length())
    y = torch.empty_like(x)
    block = min(1024, triton.next_power_of_2(hidden))
    with on_device(x):
        _expert_offsets_kernel[(1,)](counts, starts, cursor, tile_ends, E=experts, BLOCK_M=bm)
        _expert_order_kernel[(triton.cdiv(pairs, 1024),)](top_i, cursor, order, pairs, BLOCK=1024)
        moe_prefill._gate_up_kernel[(max_tiles, triton.cdiv(inter, bn1))](
            x, gate_up, order, tile_ends, counts, starts, act_r,
            gate_up.stride(0), gate_up.stride(1),
            H=hidden, I=inter, E=experts, LOG_E=log_e, TOP_K=top_k,
            BLOCK_M=bm, BLOCK_N=bn1, BLOCK_K=bk1, WIDEN=_WIDEN, num_warps=warps, num_stages=stages,
        )
        moe_prefill._down_kernel[(max_tiles, triton.cdiv(hidden, bn2))](
            act_r, down, top_w.reshape(-1), order, tile_ends, counts, starts, out_r,
            down.stride(0), down.stride(1),
            H=hidden, I=inter, E=experts, LOG_E=log_e,
            BLOCK_M=bm, BLOCK_N=bn2, BLOCK_K=bk2, WIDEN=_WIDEN, num_warps=warps, num_stages=stages,
        )
        _moe_combine_kernel[(tokens, triton.cdiv(hidden, block))](
            out_r, sg, shared_out.contiguous(), y, H=hidden, K=top_k, BLOCK=block, WIDEN=_WIDEN,
        )
    return y


# ============================================================================ Gated DeltaNet


def gdn_conv_reference(qkv: torch.Tensor, state: torch.Tensor, weight: torch.Tensor):
    """``FastModel._gdn_prefill``'s convolution: (silu(conv) [T, C], new state [C, K])."""
    from engine.fast.model import prefill_conv

    count = qkv.shape[0]
    window = torch.cat((state, qkv.transpose(0, 1)), dim=-1)
    mixed = prefill_conv(window, weight, count)
    return F.silu(mixed.to(qkv.dtype)).transpose(0, 1), window[:, -state.shape[-1]:].clone()


def gdn_conv_(qkv: torch.Tensor, state: torch.Tensor, weight: torch.Tensor,
              meta: torch.Tensor | None = None) -> torch.Tensor:
    """silu(causal conv) of ``qkv`` [T, C] after ``state`` [C, K] (a slot's conv state, updated
    in place to its last K inputs); weight [C, 1, K]. [T, C] BF16, two launches.

    With ``meta`` (int64 ``[count, ...]`` on the device, as ``prefill_attention``) only the
    first ``count`` rows are real: the state becomes the last K of (state ++ real rows), and
    the real rows' outputs are the same bits (a row reads only earlier rows)."""
    count, channels = qkv.shape
    kernel = state.shape[-1]
    assert state.stride(-1) == 1 and state.stride(-2) == kernel and kernel & (kernel - 1) == 0
    qkv = qkv.contiguous()
    out = torch.empty_like(qkv)
    with on_device(qkv):
        _gdn_conv_kernel[(triton.cdiv(count, 16), triton.cdiv(channels, 128))](
            qkv, state, weight.contiguous(), out, count, channels, K=kernel, BLOCK_T=16,
            BLOCK_C=128, WIDEN=_WIDEN,
        )
        _gdn_conv_state_kernel[(triton.cdiv(channels, 256),)](
            qkv, state, count, channels, meta if meta is not None else state, K=kernel, BLOCK_C=256,
            HAS_META=meta is not None,
        )
    return out


def gdn_gates_reference(a, b, a_log, dt_bias):
    beta = b.sigmoid()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
    return g, beta


def gdn_gates(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor):
    """(g fp32 [T, HV], beta [T, HV] in ``b``'s dtype), as ``gdn_gates_reference``, one launch."""
    a, b = a.contiguous(), b.contiguous()
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty_like(b)
    numel = a.numel()
    with on_device(a):
        _gdn_gates_kernel[(triton.cdiv(numel, 1024),)](
            a, b, a_log, dt_bias, g, beta, numel, HV=a.shape[-1], BLOCK=1024, WIDEN=_WIDEN,
        )
    return g, beta


def gated_rms(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``_gated_rms`` over rows of ``weight``'s width: x [T, heads, D] (or any [.., D] rows),
    z [T, heads * D]; [T, heads * D] in ``z``'s dtype, one launch."""
    dim = weight.shape[0]
    tokens = z.shape[0]
    x = x.reshape(-1, dim).contiguous()
    z = z.reshape(-1, dim).contiguous()
    rows = x.shape[0]
    out = torch.empty_like(z)
    with on_device(x):
        _gated_rms_kernel[(triton.cdiv(rows, 16),)](
            x, z, weight, out, rows, eps, D=dim, BLOCK_R=16,
            ROUND_Y=weight.dtype == torch.bfloat16, WIDEN=_WIDEN,
        )
    return out.reshape(tokens, -1)


def gdn_supported(m: torch.nn.Module, state: torch.Tensor) -> bool:
    k = state.shape[-1]
    d = m.head_v_dim
    return (
        triton is not None
        and state.dtype == torch.bfloat16
        and k & (k - 1) == 0
        and d & (d - 1) == 0
        and m.norm.weight.shape[0] == d
    )
