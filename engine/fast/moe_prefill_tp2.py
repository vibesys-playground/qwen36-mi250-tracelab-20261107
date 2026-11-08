"""Plan 38 candidate (not wired): grouped prefill experts with a per-tile expert table and a
BF16 pair buffer. Only ``tools/moe_prefill_tp2_sweep.py`` uses it.

Against ``moe_prefill`` / ``prefill_kernels.moe``:

* ``_tile_table_kernel`` (one program per expert) writes, for each of the expert's
  ``BLOCK_M``-row tiles, (expert, first row, row end) into a ``[3, max_tiles]`` table, so the
  GEMM programs read their tile with three independent loads instead of a dependent binary
  search over ``tile_ends``. Same tiles, same rows: the same results.
* ``_down_kernel_t`` stores ``bf16(acc)`` (BF16) instead of ``bf16(acc) * w`` (fp32);
  ``_combine_kernel_bf16`` multiplies by the routing weight in fp32 before the fp32 sum. The
  product of two BF16 values is exact in fp32, so every summand, and the output, is bitwise
  what today's path computes, with half the pair-buffer bytes.

Kernel launch parameters (tile shape, warps, stages, AMD knobs) are free; rounding points are
those of ``moe_prefill``: projections, SiLU and its product with ``up``, and the down
projection are rounded to BF16; routing weight and ``top_k`` sum in fp32.
"""

from __future__ import annotations

from engine.fast.kernels import tl, triton

if triton is not None:
    from engine.fast.moe_prefill import _bf16

    @triton.jit
    def _tile_table_kernel(counts_ptr, starts_ptr, tile_ends_ptr, table_ptr, MAX_TILES, BLOCK_M: tl.constexpr):
        """table[0, t] = expert, table[1, t] = first row, table[2, t] = row end, per tile t."""
        e = tl.program_id(0)
        count = tl.load(counts_ptr + e)
        start = tl.load(starts_ptr + e)
        n = (count + BLOCK_M - 1) // BLOCK_M
        first = tl.load(tile_ends_ptr + e) - n
        for j in range(0, n):
            t = first + j
            tl.store(table_ptr + t, e)
            tl.store(table_ptr + MAX_TILES + t, start + j * BLOCK_M)
            tl.store(table_ptr + 2 * MAX_TILES + t, start + count)

    @triton.jit
    def _gate_up_kernel_t(
        x_ptr, w_ptr, order_ptr, table_ptr, total_ptr, act_ptr, stride_we, stride_wn, MAX_TILES,
        H: tl.constexpr, I: tl.constexpr, TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WIDEN: tl.constexpr,
    ):
        t = tl.program_id(0)
        nb = tl.program_id(1)
        if t >= tl.load(total_ptr):
            return
        e = tl.load(table_ptr + t)
        row0 = tl.load(table_ptr + MAX_TILES + t)
        row_end = tl.load(table_ptr + 2 * MAX_TILES + t)
        rows = row0 + tl.arange(0, BLOCK_M)
        row_ok = rows < row_end
        pairs = tl.load(order_ptr + rows, mask=row_ok, other=0)
        tokens = (pairs // TOP_K).to(tl.int64)
        cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        a_ptrs = x_ptr + tokens[:, None] * H + ks[None, :]
        w_base = w_ptr + e.to(tl.int64) * stride_we
        g_ptrs = w_base + cols[None, :] * stride_wn + ks[:, None]
        u_ptrs = g_ptrs + I * stride_wn
        acc_g = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        acc_u = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _ in range(0, H, BLOCK_K):
            a = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
            wg = tl.load(g_ptrs)
            wu = tl.load(u_ptrs)
            if WIDEN:
                a, wg, wu = a.to(tl.float32), wg.to(tl.float32), wu.to(tl.float32)
            acc_g = tl.dot(a, wg, acc_g)
            acc_u = tl.dot(a, wu, acc_u)
            a_ptrs += BLOCK_K
            g_ptrs += BLOCK_K
            u_ptrs += BLOCK_K
        gate = _bf16(acc_g, WIDEN)
        up = _bf16(acc_u, WIDEN)
        act = _bf16(gate * tl.sigmoid(gate), WIDEN)
        out = act_ptr + rows.to(tl.int64)[:, None] * I + cols[None, :]
        tl.store(out, _bf16(act * up, WIDEN).to(tl.bfloat16), mask=row_ok[:, None])

    @triton.jit
    def _down_kernel_t(
        act_ptr, w_ptr, order_ptr, table_ptr, total_ptr, out_ptr, stride_we, stride_wn, MAX_TILES,
        H: tl.constexpr, I: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """out[p, n] = bf16(act[r] . Wd[e, n]) (BF16) at pair p = order[r]."""
        t = tl.program_id(0)
        nb = tl.program_id(1)
        if t >= tl.load(total_ptr):
            return
        e = tl.load(table_ptr + t)
        row0 = tl.load(table_ptr + MAX_TILES + t)
        row_end = tl.load(table_ptr + 2 * MAX_TILES + t)
        rows = row0 + tl.arange(0, BLOCK_M)
        row_ok = rows < row_end
        cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        a_ptrs = act_ptr + rows.to(tl.int64)[:, None] * I + ks[None, :]
        w_ptrs = w_ptr + e.to(tl.int64) * stride_we + cols[None, :] * stride_wn + ks[:, None]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _ in range(0, I, BLOCK_K):
            a = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
            wd = tl.load(w_ptrs)
            if WIDEN:
                a, wd = a.to(tl.float32), wd.to(tl.float32)
            acc = tl.dot(a, wd, acc)
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        pairs = tl.load(order_ptr + rows, mask=row_ok, other=0)
        out = out_ptr + pairs.to(tl.int64)[:, None] * H + cols[None, :]
        tl.store(out, _bf16(acc, WIDEN).to(tl.bfloat16), mask=row_ok[:, None])

    @triton.jit
    def _combine_kernel_bf16(
        pairs_ptr, wt_ptr, sg_ptr, shared_ptr, out_ptr, H: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """``prefill_kernels._moe_combine_kernel`` over BF16 pair rows: w (BF16) applied here,
        in fp32 (exact: a product of two BF16 values), then the same fp32 sum and roundings."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = cols < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(K):
            w = tl.load(wt_ptr + row * K + k).to(tl.float32)
            acc += tl.load(pairs_ptr + (row * K + k) * H + cols, mask=ok, other=0.0).to(tl.float32) * w
        routed = _bf16(acc, WIDEN)
        gate = _bf16(tl.sigmoid(tl.load(sg_ptr + row).to(tl.float32)), WIDEN)
        shared = tl.load(shared_ptr + row * H + cols, mask=ok, other=0.0).to(tl.float32)
        y = _bf16(routed + _bf16(gate * shared, WIDEN), WIDEN)
        tl.store(out_ptr + row * H + cols, y.to(tl.bfloat16), mask=ok)
