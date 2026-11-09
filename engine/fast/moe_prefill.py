"""Routed experts for prefill-sized inputs: a sync-free grouped GEMM in Triton.

The experts implementation the engine registered with Transformers
(``engine.experts``) sorts the (token, expert) pairs by expert, then reads the
largest group size on the host (one synchronization per MoE layer, 40 per
pass) to pad every expert to it and run one batched matmul per projection, or,
for skewed routing, loops over the experts (hundreds of launches and host
copies per layer). This module computes only the real rows and never reads
expert ids or counts on the host:

1. Bookkeeping on the device: ``argsort`` of the pairs' expert ids, per-expert
   counts by ``scatter_add_`` into a fixed ``[experts]`` buffer, then
   ``_tiles_kernel``: each expert's first sorted row and a table
   of the ``BLOCK_M``-row tiles (expert, first row, row end) and their number.
2. ``_gate_up_kernel``: one program per (tile, block of ``moe_intermediate``
   columns). The grid holds the most tiles any routing can need
   (``ceil(pairs / BLOCK_M) + experts``, known from the token count alone); a
   program reads its tile from the table and returns at once when its tile is
   past the last one. It gathers its rows of ``x`` by token, multiplies them by
   the expert's gate and up rows in place and writes ``silu(gate) * up`` in
   sorted order.
3. ``_down_kernel``: the same tiling over ``hidden`` columns; it multiplies by
   the expert's ``down_proj`` and writes each row (BF16) at the pair's own
   (unsorted) index, so a token's ``top_k`` rows are adjacent; the combine
   applies the routing weights in fp32 and sums them. No atomics: the result
   does not depend on scheduling, so greedy decoding stays repeatable.

The tile shapes come from ``config``: a few buckets of the mean rows per
expert, i.e. of the token count alone (graph-capturable per bucket).

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
from typing import NamedTuple

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

    @triton.jit(do_not_specialize=["MAX_TILES"])
    def _tiles_kernel(
        counts_ptr, starts_ptr, cursor_ptr, table_ptr, MAX_TILES, E: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Grid ``cdiv(MAX_TILES, BLOCK_T)``; each program recomputes the per-expert prefix sums
        (E values) and fills BLOCK_T entries of the tile table the GEMM kernels read: for tile
        t, table[t] = expert, table[MAX_TILES + t] = first row, table[2 * MAX_TILES + t] = row
        end; table[3 * MAX_TILES] = the number of tiles. Program 0 also writes each expert's
        first sorted row (``starts``, and a copy in ``cursor`` for ``_expert_order_kernel``).
        A GEMM program finds its tile with three independent loads instead of a binary search;
        the tiles are the same."""
        pid = tl.program_id(0)
        e = tl.arange(0, E)
        c = tl.load(counts_ptr + e)
        st = tl.cumsum(c, axis=0) - c
        n = (c + BLOCK_M - 1) // BLOCK_M
        tend = tl.cumsum(n, axis=0)
        tstart = tend - n
        total = tl.sum(n, axis=0)
        if pid == 0:
            tl.store(starts_ptr + e, st)
            tl.store(cursor_ptr + e, st)
            tl.store(table_ptr + 3 * MAX_TILES, total)
        t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        # The tile's expert: the first e with tend[e] > t (tend is non-decreasing).
        ex = tl.sum((tend[None, :] <= t[:, None]).to(tl.int32), axis=1)
        sel = e[None, :] == ex[:, None]
        row_start = tl.sum(tl.where(sel, st[None, :], 0), axis=1)
        first = tl.sum(tl.where(sel, tstart[None, :], 0), axis=1)
        count = tl.sum(tl.where(sel, c[None, :], 0), axis=1)
        ok = t < total
        tl.store(table_ptr + t, ex, mask=ok)
        tl.store(table_ptr + MAX_TILES + t, row_start + (t - first) * BLOCK_M, mask=ok)
        tl.store(table_ptr + 2 * MAX_TILES + t, row_start + count, mask=ok)

    @triton.jit(do_not_specialize=["MAX_TILES"])
    def _tiles_one_kernel(
        counts_ptr, starts_ptr, cursor_ptr, table_ptr, MAX_TILES, E: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        """``_tiles_kernel``'s table from one program (plan 56): the prefix sums once, then
        each expert writes its own tiles' entries (tile tstart[e] + j, j < n[e]), in max(n)
        steps of E-wide stores, with no [tiles, E] compare. The entries [0, total) and the
        count are the same integers; entries past the count are not written by either."""
        e = tl.arange(0, E)
        c = tl.load(counts_ptr + e)
        st = tl.cumsum(c, axis=0) - c
        n = (c + BLOCK_M - 1) // BLOCK_M
        tend = tl.cumsum(n, axis=0)
        tstart = tend - n
        tl.store(starts_ptr + e, st)
        tl.store(cursor_ptr + e, st)
        tl.store(table_ptr + 3 * MAX_TILES, tl.sum(n, axis=0))
        end = st + c
        steps = tl.max(n, axis=0)
        j = 0
        while j < steps:
            ok = j < n
            t = tstart + j
            tl.store(table_ptr + t, e, mask=ok)
            tl.store(table_ptr + MAX_TILES + t, st + j * BLOCK_M, mask=ok)
            tl.store(table_ptr + 2 * MAX_TILES + t, end, mask=ok)
            j += 1

    @triton.jit(do_not_specialize=["MAX_TILES"])
    def _gate_up_kernel(
        x_ptr,
        w_ptr,
        order_ptr,
        table_ptr,
        act_ptr,
        stride_we,
        stride_wn,
        MAX_TILES,
        H: tl.constexpr,
        I: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        """act[r, n] = silu(x_t . Wg[e, n]) * (x_t . Wu[e, n]) for sorted row r = pair of token t."""
        t = tl.program_id(0)
        nb = tl.program_id(1)
        if t >= tl.load(table_ptr + 3 * MAX_TILES):
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
        # Round where the BF16 matmuls of the padded path round: projections, SiLU, product.
        gate = _bf16(acc_g, WIDEN)
        up = _bf16(acc_u, WIDEN)
        act = _bf16(gate * tl.sigmoid(gate), WIDEN)
        out = act_ptr + rows.to(tl.int64)[:, None] * I + cols[None, :]
        tl.store(out, _bf16(act * up, WIDEN).to(tl.bfloat16), mask=row_ok[:, None])

    @triton.jit(do_not_specialize=["MAX_TILES"])
    def _down_kernel(
        act_ptr,
        w_ptr,
        order_ptr,
        table_ptr,
        out_ptr,
        stride_we,
        stride_wn,
        MAX_TILES,
        H: tl.constexpr,
        I: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        """out[p, n] = bf16(act[r] . Wd[e, n]) (BF16) at pair p = order[r]. The routing weight is
        applied by the combine, in fp32: w * bf16 is exact in fp32, so the summands are the
        ones ``bf16(acc) * w`` stored in fp32 would give, with half the bytes."""
        t = tl.program_id(0)
        nb = tl.program_id(1)
        if t >= tl.load(table_ptr + 3 * MAX_TILES):
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

    @triton.jit(do_not_specialize=["MAX_TILES"])
    def _down_stream_kernel(
        act_ptr,
        w_ptr,
        order_ptr,
        table_ptr,
        out_ptr,
        stride_we,
        stride_wn,
        MAX_TILES,
        H: tl.constexpr,
        I: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        """``_down_kernel`` with each program streaming N (plan 60): grid (tiles,
        H / (GROUP * BLOCK_N)). A program loads its act tile [BLOCK_M, I] once (I = BLOCK_K or
        2 * BLOCK_K, as two halves) and its rows' pair indices once, then loops over GROUP
        consecutive BLOCK_N column blocks of the expert's W_down (one contiguous stream, which
        the compiler can software-pipeline). Each block is today's math: the same BLOCK_K dots
        in the same order from a zero accumulator, rounded to BF16 and stored at the pairs'
        rows, so the output is bitwise ``_down_kernel``'s."""
        t = tl.program_id(0)
        nb = tl.program_id(1)
        if t >= tl.load(table_ptr + 3 * MAX_TILES):
            return
        e = tl.load(table_ptr + t)
        row0 = tl.load(table_ptr + MAX_TILES + t)
        row_end = tl.load(table_ptr + 2 * MAX_TILES + t)
        rows = row0 + tl.arange(0, BLOCK_M)
        row_ok = rows < row_end
        ks = tl.arange(0, BLOCK_K)
        a_ptrs = act_ptr + rows.to(tl.int64)[:, None] * I + ks[None, :]
        a0 = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
        if I > BLOCK_K:
            a1 = tl.load(a_ptrs + BLOCK_K, mask=row_ok[:, None], other=0.0)
        else:
            a1 = a0
        if WIDEN:
            a0, a1 = a0.to(tl.float32), a1.to(tl.float32)
        pairs = tl.load(order_ptr + rows, mask=row_ok, other=0)
        out_rows = out_ptr + pairs.to(tl.int64)[:, None] * H
        ns = tl.arange(0, BLOCK_N)
        w_base = w_ptr + e.to(tl.int64) * stride_we + ks[:, None]
        col0 = nb * (GROUP * BLOCK_N)
        for g in range(GROUP):
            cols = col0 + g * BLOCK_N + ns
            w_ptrs = w_base + cols[None, :] * stride_wn
            w0 = tl.load(w_ptrs)
            if WIDEN:
                w0 = w0.to(tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            acc = tl.dot(a0, w0, acc)
            if I > BLOCK_K:
                w1 = tl.load(w_ptrs + BLOCK_K)
                if WIDEN:
                    w1 = w1.to(tl.float32)
                acc = tl.dot(a1, w1, acc)
            tl.store(out_rows + cols[None, :], _bf16(acc, WIDEN).to(tl.bfloat16), mask=row_ok[:, None])


class KernelConfig(NamedTuple):
    """One GEMM kernel's launch: BLOCK_N, BLOCK_K, warps, stages, and the AMD knobs
    ``matrix_instr_nonkdim`` and ``waves_per_eu`` (0: the compiler's choice)."""

    bn: int
    bk: int
    warps: int
    stages: int
    nonkdim: int = 0
    waves: int = 0


class Config(NamedTuple):
    """BLOCK_M (shared: one tile table) and the two GEMM kernels' launches."""

    bm: int
    gate_up: KernelConfig
    down: KernelConfig


# Tile shapes by mean routed rows per expert (pairs / experts), from the token count alone so a
# captured graph per token bucket replays the same launches. Tuned in plan 38 (job j180) on one
# MI250 GCD with the TP2 shard (gate_up [256, 512, 2048], down [256, 2048, 256]) and the real
# routing of all 40 layers (replay-like and natural text, 61/561/1,650/2,300 new tokens): the
# fastest per bucket, every output bitwise equal to the earlier DEFAULT_CONFIG
# (64, 64,128, 64,128, 4 warps, 2 stages) since BLOCK_K stays 128. Upper bound (rows per
# expert, inclusive) -> config; the last bucket has no bound.
BUCKETS: tuple[tuple[float, Config], ...] = (
    (4.0, Config(16, KernelConfig(64, 128, 4, 2), KernelConfig(256, 128, 4, 1, 16, 2))),  # <= 128 tokens at 8 of 256
    (32.0, Config(64, KernelConfig(64, 128, 4, 1), KernelConfig(256, 128, 8, 1))),  # <= 1,024 tokens
    (float("inf"), Config(64, KernelConfig(128, 128, 4, 1, 16, 2), KernelConfig(128, 128, 4, 1, 16, 0))),
)


def config(pairs: int, experts: int) -> Config:
    """Tile shapes for ``pairs`` (token, expert) rows over ``experts``; from the shape alone.
    ``ENGINE_MOE_GROUPED_CFG`` = "BM,BN1,BK1,BN2,BK2,warps,stages" overrides it (both kernels,
    no AMD knobs)."""
    override = os.environ.get("ENGINE_MOE_GROUPED_CFG")
    if override:
        values = tuple(int(v) for v in override.split(","))
        if len(values) != 7:
            raise ValueError("ENGINE_MOE_GROUPED_CFG needs BM,BN1,BK1,BN2,BK2,warps,stages")
        bm, bn1, bk1, bn2, bk2, warps, stages = values
        return Config(bm, KernelConfig(bn1, bk1, warps, stages), KernelConfig(bn2, bk2, warps, stages))
    rows = pairs / max(1, experts)
    for bound, cfg in BUCKETS:
        if rows <= bound:
            return cfg
    return BUCKETS[-1][1]


def bucket_tokens(top_k: int, experts: int, limit: int) -> list[int]:
    """One token count (at most ``limit``) in each bucket that ``config`` can pick: what to warm."""
    out = []
    lo = 0.0
    for bound, _ in BUCKETS:
        tokens = int(lo * experts) // top_k + 1
        if tokens <= limit:
            out.append(tokens)
        lo = bound
    return out


def launch_options(k: KernelConfig) -> dict:
    """``num_warps`` / ``num_stages`` and, on ROCm only, the AMD backend's knobs."""
    opts = {"num_warps": k.warps, "num_stages": k.stages}
    if torch.version.hip and not INTERPRET:
        if k.nonkdim:
            opts["matrix_instr_nonkdim"] = k.nonkdim
        if k.waves:
            opts["waves_per_eu"] = k.waves
    return opts


class StreamConfig(NamedTuple):
    """``_down_stream_kernel``'s launch: BLOCK_N, GROUP (column blocks per program), warps,
    stages and the AMD knobs (0: the compiler's choice). BLOCK_K is the bucket's down BK."""

    bn: int
    group: int
    warps: int
    stages: int
    nonkdim: int = 0
    waves: int = 0


# Plan 60: the N-streaming down projection per ``BUCKETS`` entry (same bounds, same BLOCK_M and
# BLOCK_K, so one tile table and the same dots). The fastest per bucket in the step 0 sweep (job
# j282, GCD 0 of a TP2 pair, real routing of all 41 MoE layers, replay-like text; every config
# bitwise equal): 61 tokens 0.978x today's down, 561 0.892x, 1,650 0.761x, 2,300 0.752x. The plan's
# kill rule (<= 0.85x at 561 and <= 0.75x at 1,650) failed for all sizes, so by default the stream
# kernel runs only on passes above ``STREAM_MIN_TOKENS`` tokens (0.76x there) and today's
# ``_down_kernel`` below.
STREAM_BUCKETS: tuple[StreamConfig, ...] = (
    StreamConfig(256, 1, 4, 1, 0, 2),
    StreamConfig(128, 4, 8, 1),
    StreamConfig(128, 8, 8, 2, 16, 0),
)


STREAM_MIN_TOKENS = 1024  # the default turns the stream kernel on for passes above this


def down_stream_mode() -> bool | None:
    """``ENGINE_MOE_DOWN_STREAM`` (plan 60): ``0`` today's ``_down_kernel`` always, ``1`` the
    prefill down projection through ``_down_stream_kernel`` always (bitwise the same output),
    unset or ``auto`` (None): the stream kernel only for passes above ``STREAM_MIN_TOKENS``."""
    v = os.environ.get("ENGINE_MOE_DOWN_STREAM", "auto").strip().lower()
    if v in ("", "auto"):
        return None
    return v != "0"


_DOWN_STREAM = down_stream_mode()


def use_down_stream(tokens: int) -> bool:
    """Whether a pass of ``tokens`` tokens runs the down projection through the stream kernel."""
    if _DOWN_STREAM is None:
        return tokens > STREAM_MIN_TOKENS
    return bool(_DOWN_STREAM)


def stream_config(pairs: int, experts: int) -> StreamConfig:
    """The bucket's ``StreamConfig`` (as ``config``, from the shape alone).
    ``ENGINE_MOE_DOWN_STREAM_CFG`` = "BN,GROUP,warps,stages[,nonkdim,waves]" overrides it."""
    override = os.environ.get("ENGINE_MOE_DOWN_STREAM_CFG")
    if override:
        values = tuple(int(v) for v in override.split(","))
        if len(values) not in (4, 6):
            raise ValueError("ENGINE_MOE_DOWN_STREAM_CFG needs BN,GROUP,warps,stages[,nonkdim,waves]")
        return StreamConfig(*values)
    rows = pairs / max(1, experts)
    for (bound, _), cfg in zip(BUCKETS, STREAM_BUCKETS):
        if rows <= bound:
            return cfg
    return STREAM_BUCKETS[-1]


def launch_down(act: torch.Tensor, down: torch.Tensor, order: torch.Tensor, table: torch.Tensor,
                out: torch.Tensor, max_tiles: int, cfg: Config, pairs: int,
                stream: bool | None = None, tokens: int | None = None) -> None:
    """The down projection of the grouped experts into ``out`` [pairs, hidden] (BF16, at each
    pair's row): ``_down_stream_kernel`` when ``stream`` (default: ``use_down_stream(tokens)``,
    the pass's token count) and the shapes fit it (inter = BK or 2 * BK, hidden a multiple of
    GROUP * BN), else ``_down_kernel``. Launched on the current device; the same output either
    way."""
    experts, hidden, inter = down.shape
    bm = cfg.bm
    bk = min(cfg.down.bk, inter)
    if stream is None:
        if tokens is None:
            raise ValueError("launch_down needs stream or the pass's tokens")
        stream = use_down_stream(tokens)
    if stream:
        s = stream_config(pairs, experts)
        bn = min(s.bn, hidden)
        group = max(1, min(s.group, hidden // bn))
        if inter in (bk, 2 * bk) and hidden % (group * bn) == 0:
            opts = launch_options(KernelConfig(bn, bk, s.warps, s.stages, s.nonkdim, s.waves))
            _down_stream_kernel[(max_tiles, hidden // (group * bn))](
                act, down, order, table, out, down.stride(0), down.stride(1), max_tiles,
                H=hidden, I=inter, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP=group, WIDEN=_WIDEN,
                **opts,
            )
            return
    bn = min(cfg.down.bn, hidden)
    _down_kernel[(max_tiles, triton.cdiv(hidden, bn))](
        act, down, order, table, out, down.stride(0), down.stride(1), max_tiles,
        H=hidden, I=inter, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, WIDEN=_WIDEN,
        **launch_options(cfg.down),
    )


TILE_BLOCK = 32  # tiles per step of ``_tiles_kernel``


def tiles_one_enabled() -> bool:
    """``ENGINE_MOE_TILES_ONE`` (plan 56, default on; 0 for ``_tiles_kernel``): the tile table
    from one program, ``_tiles_one_kernel`` (the same table)."""
    return os.environ.get("ENGINE_MOE_TILES_ONE", "1") != "0"


_TILES_ONE = tiles_one_enabled()


def build_tiles(counts: torch.Tensor, starts: torch.Tensor, cursor: torch.Tensor, table: torch.Tensor,
                max_tiles: int, bm: int, one: bool | None = None) -> None:
    """The tile table ``table`` [3 * max_tiles + 1] and each expert's first sorted row
    (``starts`` and its copy ``cursor``) from the per-expert pair ``counts`` (see
    ``_tiles_kernel``); launched on the current device."""
    experts = counts.shape[0]
    if _TILES_ONE if one is None else one:
        _tiles_one_kernel[(1,)](counts, starts, cursor, table, max_tiles, E=experts, BLOCK_M=bm,
                                num_warps=4)
    else:
        _tiles_kernel[(triton.cdiv(max_tiles, TILE_BLOCK),)](
            counts, starts, cursor, table, max_tiles, E=experts, BLOCK_M=bm, BLOCK_T=TILE_BLOCK,
        )


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
    cfg = config(pairs, experts)
    bm = cfg.bm
    bn1, bk1 = min(cfg.gate_up.bn, inter), min(cfg.gate_up.bk, hidden)
    dev = x.device
    x = x.contiguous()
    ids = top_k_index.reshape(-1)
    order = torch.argsort(ids, stable=True).to(torch.int32)
    counts = torch.zeros(experts, dtype=torch.int32, device=dev)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    meta = torch.empty((2, experts), dtype=torch.int32, device=dev)
    max_tiles = triton.cdiv(pairs, bm) + experts
    table = torch.empty(3 * max_tiles + 1, dtype=torch.int32, device=dev)
    act = torch.empty((pairs, inter), dtype=torch.bfloat16, device=dev)
    out = torch.empty((pairs, hidden), dtype=torch.bfloat16, device=dev)
    with on_device(x):
        build_tiles(counts, meta[0], meta[1], table, max_tiles, bm)
        _gate_up_kernel[(max_tiles, triton.cdiv(inter, bn1))](
            x, gate_up, order, table, act, gate_up.stride(0), gate_up.stride(1), max_tiles,
            H=hidden, I=inter, TOP_K=top_k, BLOCK_M=bm, BLOCK_N=bn1, BLOCK_K=bk1, WIDEN=_WIDEN,
            **launch_options(cfg.gate_up),
        )
        launch_down(act, down, order, table, out, max_tiles, cfg, pairs, tokens=tokens)
    # The routing weight in fp32 (exact on two BF16 values), the top_k rows summed in fp32.
    weighted = out.float() * top_k_weights.reshape(-1, 1).float()
    return weighted.view(tokens, top_k, hidden).sum(dim=1).to(x.dtype)


def enabled() -> bool:
    """``ENGINE_MOE_GROUPED`` (default on): prefill experts through ``moe_prefill``."""
    return os.environ.get("ENGINE_MOE_GROUPED", "1") != "0"
