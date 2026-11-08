"""Routed experts for prefill-sized inputs: a sync-free grouped GEMM in Triton.

The experts implementation the engine registered with Transformers
(``engine.experts``) sorts the (token, expert) pairs by expert, then reads the
largest group size on the host (one synchronization per MoE layer, 40 per
pass) to pad every expert to it and run one batched matmul per projection, or,
for skewed routing, loops over the experts (hundreds of launches and host
copies per layer). This module computes only the real rows and never reads
expert ids or counts on the host:

1. Bookkeeping on the device: ``argsort`` of the pairs' expert ids, per-expert
   counts by ``scatter_add_`` into a fixed ``[experts]`` buffer, their
   exclusive cumulative sum (each expert's first sorted row) and the cumulative
   number of ``BLOCK_M``-row tiles per expert.
2. ``_gate_up_kernel``: one program per (tile, block of ``moe_intermediate``
   columns). The grid holds the most tiles any routing can need
   (``ceil(pairs / BLOCK_M) + experts``, known from the token count alone); a
   program finds its expert by a binary search over the cumulative tile counts
   and returns at once when its tile is past the last one. It gathers its rows
   of ``x`` by token, multiplies them by the expert's gate and up rows in place
   and writes ``silu(gate) * up`` in sorted order.
3. ``_down_kernel``: the same tiling over ``hidden`` columns; it multiplies by
   the expert's ``down_proj`` and writes each row, times its routing weight in
   fp32, at the pair's own (unsorted) index, so a token's ``top_k`` rows are
   adjacent and one ``sum`` combines them. No atomics: the result does not
   depend on scheduling, so greedy decoding stays repeatable.

Rounding follows the padded batched matmul it replaces: projections, the SiLU
and its product with ``up`` and the down projection are rounded to BF16, the
routing weight is applied and the ``top_k`` rows are summed in fp32, then
rounded to BF16 once.

``moe_prefill_reference`` is the same math in PyTorch (the per-expert loop),
for CPU tests. Specialized to what Qwen3.6-35B-A3B needs (BF16 weights laid out
``[experts, 2 * inter, hidden]`` and ``[experts, hidden, inter]``, dimensions
powers of two or multiples of 128); ``supported`` says when it applies.
"""

from __future__ import annotations

import os

import torch
from torch.nn import functional as F

from engine.fast.kernels import INTERPRET, on_device, tl, triton

# The Triton interpreter's ``tl.dot`` mishandles BF16 operands (numpy has no bfloat16) and it
# rounds fp32 to BF16 by truncation; there the tiles are widened to fp32 first, which multiplies
# the same values exactly, and rounding to nearest even is done on the bits (``_bf16``).
_WIDEN = INTERPRET


def moe_prefill_reference(
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Per-expert loop (what ``engine.experts._sorted_experts_forward`` computes)."""
    tokens, top_k = top_k_index.shape
    ids = top_k_index.reshape(-1)
    inter = gate_up.shape[1] // 2
    out = torch.zeros((tokens * top_k, x.shape[-1]), dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        pairs = (ids == expert).nonzero().squeeze(-1)
        rows = x[pairs // top_k]
        proj = F.linear(rows, gate_up[expert])
        gate, up = proj[:, :inter], proj[:, inter:]
        out[pairs] = F.linear(F.silu(gate) * up, down[expert]).float()
    out = out * top_k_weights.reshape(-1, 1).float()
    return out.view(tokens, top_k, -1).sum(dim=1).to(x.dtype)


if triton is not None:

    @triton.jit
    def _bf16(x, WIDEN: tl.constexpr):
        """``x`` (fp32) rounded to BF16, to nearest even, as fp32."""
        if WIDEN:
            bits = x.to(tl.uint32, bitcast=True)
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            return bits.to(tl.float32, bitcast=True)
        else:
            return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _find_tile(t, tile_ends_ptr, counts_ptr, starts_ptr, E: tl.constexpr, LOG_E: tl.constexpr, BLOCK_M: tl.constexpr):
        """Expert of tile ``t`` (smallest e with tile_ends[e] > t), its first row and row end."""
        lo = t * 0
        hi = lo + E
        for _ in tl.static_range(LOG_E + 1):
            mid = (lo + hi) // 2
            v = tl.load(tile_ends_ptr + tl.minimum(mid, E - 1))
            go_left = v > t
            hi = tl.where(go_left, mid, hi)
            lo = tl.where(go_left, lo, mid + 1)
        e = tl.minimum(lo, E - 1)
        count = tl.load(counts_ptr + e)
        start = tl.load(starts_ptr + e)
        first_tile = tl.load(tile_ends_ptr + e) - (count + BLOCK_M - 1) // BLOCK_M
        row0 = start + (t - first_tile) * BLOCK_M
        return e, row0, start + count

    @triton.jit
    def _gate_up_kernel(
        x_ptr,
        w_ptr,
        order_ptr,
        tile_ends_ptr,
        counts_ptr,
        starts_ptr,
        act_ptr,
        stride_we,
        stride_wn,
        H: tl.constexpr,
        I: tl.constexpr,
        E: tl.constexpr,
        LOG_E: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        """act[r, n] = silu(x_t . Wg[e, n]) * (x_t . Wu[e, n]) for sorted row r = pair of token t."""
        t = tl.program_id(0)
        nb = tl.program_id(1)
        if t >= tl.load(tile_ends_ptr + E - 1):
            return
        e, row0, row_end = _find_tile(t, tile_ends_ptr, counts_ptr, starts_ptr, E, LOG_E, BLOCK_M)
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
        # Round where the BF16 matmuls of the padded path round: projections, SiLU, product.
        gate = _bf16(acc_g, WIDEN)
        up = _bf16(acc_u, WIDEN)
        act = _bf16(gate * tl.sigmoid(gate), WIDEN)
        out = act_ptr + rows.to(tl.int64)[:, None] * I + cols[None, :]
        tl.store(out, _bf16(act * up, WIDEN).to(tl.bfloat16), mask=row_ok[:, None])

    @triton.jit
    def _down_kernel(
        act_ptr,
        w_ptr,
        wt_ptr,
        order_ptr,
        tile_ends_ptr,
        counts_ptr,
        starts_ptr,
        out_ptr,
        stride_we,
        stride_wn,
        H: tl.constexpr,
        I: tl.constexpr,
        E: tl.constexpr,
        LOG_E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        """out[p, n] = w_p * bf16(act[r] . Wd[e, n]) in fp32, at pair p = order[r]."""
        t = tl.program_id(0)
        nb = tl.program_id(1)
        if t >= tl.load(tile_ends_ptr + E - 1):
            return
        e, row0, row_end = _find_tile(t, tile_ends_ptr, counts_ptr, starts_ptr, E, LOG_E, BLOCK_M)
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
        weight = tl.load(wt_ptr + pairs, mask=row_ok, other=0.0).to(tl.float32)
        val = _bf16(acc, WIDEN) * weight[:, None]
        out = out_ptr + pairs.to(tl.int64)[:, None] * H + cols[None, :]
        tl.store(out, val, mask=row_ok[:, None])


# Tile shape: BLOCK_M rows, gate_up BLOCK_N/BLOCK_K, down BLOCK_N/BLOCK_K, warps, stages.
# Measured on an MI210 (gfx90a, job j76) over the real routing of four Qwen3.6 layers and a
# skewed one, 64 to 4096 tokens, against eight other shapes (BLOCK_M 16-128, BLOCK_N 64/128,
# BLOCK_K 64/128, 4/8 warps): fastest summed over the layers with real routing at every size,
# within 6% of the fastest with skewed routing from 700 tokens up (0.29 ms behind over four
# layers at 64 skewed tokens). ``ENGINE_MOE_GROUPED_CFG``
# overrides it: "BM,BN1,BK1,BN2,BK2,warps,stages".
DEFAULT_CONFIG = (64, 64, 128, 64, 128, 4, 2)


def config(pairs: int, experts: int) -> tuple[int, int, int, int, int, int, int]:
    """Tile shape for ``pairs`` (token, expert) rows over ``experts``; from the shape alone."""
    override = os.environ.get("ENGINE_MOE_GROUPED_CFG")
    if override:
        values = tuple(int(v) for v in override.split(","))
        if len(values) != 7:
            raise ValueError("ENGINE_MOE_GROUPED_CFG needs BM,BN1,BK1,BN2,BK2,warps,stages")
        return values  # type: ignore[return-value]
    return DEFAULT_CONFIG


def _aligned(dim: int) -> bool:
    """Every block size (a power of two up to 128, clamped to ``dim``) divides ``dim``."""
    return dim >= 16 and (dim % 128 == 0 or dim & (dim - 1) == 0)


def supported(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> bool:
    """Shapes and layouts the kernels handle: BF16, row-major experts, block-aligned dims."""
    if triton is None or x.dtype != torch.bfloat16:
        return False
    experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    return (
        gate_up.dtype == torch.bfloat16
        and down.dtype == torch.bfloat16
        and down.shape == (experts, hidden, inter)
        and gate_up.stride(-1) == 1
        and down.stride(-1) == 1
        and _aligned(hidden)
        and _aligned(inter)
    )


def moe_prefill(
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Routed-expert output for ``x`` [tokens, hidden]; no host synchronization."""
    tokens, top_k = top_k_index.shape
    experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    pairs = tokens * top_k
    bm, bn1, bk1, bn2, bk2, warps, stages = config(pairs, experts)
    bn1, bk2 = min(bn1, inter), min(bk2, inter)
    bk1, bn2 = min(bk1, hidden), min(bn2, hidden)
    dev = x.device
    x = x.contiguous()
    ids = top_k_index.reshape(-1)
    order = torch.argsort(ids, stable=True).to(torch.int32)
    counts = torch.zeros(experts, dtype=torch.int32, device=dev)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    ends = torch.cumsum(counts, 0, dtype=torch.int32)
    starts = ends - counts
    tile_ends = torch.cumsum((counts + (bm - 1)) // bm, 0, dtype=torch.int32)
    weights = top_k_weights.reshape(-1).contiguous()
    max_tiles = triton.cdiv(pairs, bm) + experts
    log_e = max(1, (experts - 1).bit_length())
    act = torch.empty((pairs, inter), dtype=torch.bfloat16, device=dev)
    out = torch.empty((pairs, hidden), dtype=torch.float32, device=dev)
    with on_device(x):
        _gate_up_kernel[(max_tiles, triton.cdiv(inter, bn1))](
            x, gate_up, order, tile_ends, counts, starts, act,
            gate_up.stride(0), gate_up.stride(1),
            H=hidden, I=inter, E=experts, LOG_E=log_e, TOP_K=top_k,
            BLOCK_M=bm, BLOCK_N=bn1, BLOCK_K=bk1, WIDEN=_WIDEN, num_warps=warps, num_stages=stages,
        )
        _down_kernel[(max_tiles, triton.cdiv(hidden, bn2))](
            act, down, weights, order, tile_ends, counts, starts, out,
            down.stride(0), down.stride(1),
            H=hidden, I=inter, E=experts, LOG_E=log_e,
            BLOCK_M=bm, BLOCK_N=bn2, BLOCK_K=bk2, WIDEN=_WIDEN, num_warps=warps, num_stages=stages,
        )
    return out.view(tokens, top_k, hidden).sum(dim=1).to(x.dtype)


def enabled() -> bool:
    """``ENGINE_MOE_GROUPED`` (default on): prefill experts through ``moe_prefill``."""
    return os.environ.get("ENGINE_MOE_GROUPED", "1") != "0"
