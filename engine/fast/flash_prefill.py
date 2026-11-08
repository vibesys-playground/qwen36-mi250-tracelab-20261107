"""Plan 35: MFMA-tiled flash prefill attention over the paged KV cache.

``prefill_kernels.prefill_attention`` calls it under ``ENGINE_PREFILL_FUSED`` (and
``ENGINE_PREFILL_FLASH``, default on; 0 restores ``_prefill_attn_kernel``).

Same interface, inputs and outputs as ``prefill_kernels.prefill_attention``: gated causal
attention of ``q`` [T, QH, D] at positions ``start``... over the first ``start + T`` cached
tokens of one slot (pages ``table``), BF16 dot operands, fp32 scores, softmax and
accumulation, P.V with BF16 probabilities, the BF16-rounded output times
``bf16(sigmoid(gate))`` rounded once more (the eager ops' roundings).

What differs from ``_prefill_attn_kernel``:

- More query rows per program (BLOCK_R rows x GROUP heads of one KV head, M = 64 or 128) so
  each K/V block feeds a larger MFMA tile; BLOCK_N keys per step (32, 64 or 128; a block
  wider than a page gathers its pages per key).
- The keys below the tile's first query position run without the causal mask; only the
  last blocks of the tile apply it (the loop is split in two).
- Softmax in base 2 (scores times scale * log2(e), ``exp2``).
- K loaded as [D, BLOCK_N] (``KT``) or [BLOCK_N, D] and transposed in registers.
- Optional split over the keys (``SPLIT`` > 1, flash-decoding) for short segments: partial
  (m, l, acc) per split in an fp32 workspace, then a combine kernel sums them in split order
  (deterministic, no atomics). The split count is a launch parameter chosen from shapes only.

Launch parameters depend on (T, start) only through the grid (T) and scalar arguments: no
host sync, graph-capturable.
"""

from __future__ import annotations

import math

import torch

from engine.fast.kernels import INTERPRET, on_device, tl, triton

_WIDEN = INTERPRET
LOG2E = 1.4426950408889634

if triton is not None:
    from engine.fast.moe_prefill import _bf16

    @triton.jit
    def _blocks(
        q, m, l, acc, k_ptr, v_ptr, table_ptr, kvh, q_pos, end, qk_scale, lo, hi,
        stride_page, stride_head, stride_tok,
        D: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr, MASKED: tl.constexpr,
        KT: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """Key blocks [lo, hi) of BLOCK_N keys into the running (m, l, acc)."""
        tn = tl.arange(0, BLOCK_N)
        d = tl.arange(0, D)
        for b in range(lo, hi):
            k_pos = b * BLOCK_N + tn
            if BLOCK_N <= PAGE:
                page = tl.load(table_ptr + (b * BLOCK_N) // PAGE).to(tl.int64)
                koff = page * stride_page + kvh * stride_head + ((b * BLOCK_N) % PAGE + tn) * stride_tok
            else:
                if MASKED:
                    pg = tl.load(table_ptr + k_pos // PAGE, mask=k_pos < end, other=0).to(tl.int64)
                else:
                    pg = tl.load(table_ptr + k_pos // PAGE).to(tl.int64)
                koff = pg * stride_page + kvh * stride_head + (k_pos % PAGE) * stride_tok
            if MASKED:
                k_ok = k_pos < end
                if KT:
                    kt = tl.load(k_ptr + koff[None, :] + d[:, None], mask=k_ok[None, :], other=0.0)
                else:
                    kt = tl.trans(tl.load(k_ptr + koff[:, None] + d[None, :], mask=k_ok[:, None], other=0.0))
                values = tl.load(v_ptr + koff[:, None] + d[None, :], mask=k_ok[:, None], other=0.0)
            else:
                if KT:
                    kt = tl.load(k_ptr + koff[None, :] + d[:, None])
                else:
                    kt = tl.trans(tl.load(k_ptr + koff[:, None] + d[None, :]))
                values = tl.load(v_ptr + koff[:, None] + d[None, :])
            if WIDEN:
                kt = kt.to(tl.float32)
            s = tl.dot(q, kt) * qk_scale
            if MASKED:
                s = tl.where(k_pos[None, :] <= q_pos[:, None], s, float("-inf"))
                m_new = tl.maximum(m, tl.max(s, axis=1))
                m_use = tl.where(m_new == float("-inf"), 0.0, m_new)  # a row with no key yet
            else:
                m_new = tl.maximum(m, tl.max(s, axis=1))
                m_use = m_new
            alpha = tl.exp2(m - m_use)
            prob = tl.exp2(s - m_use[:, None])
            l = l * alpha + tl.sum(prob, axis=1)
            if WIDEN:
                acc = acc * alpha[:, None] + tl.dot(_bf16(prob, WIDEN), values.to(tl.float32))
            else:
                acc = acc * alpha[:, None] + tl.dot(prob.to(values.dtype), values)
            m = m_new
        return m, l, acc

    @triton.jit(do_not_specialize=["T", "start", "nsplit"])
    def _flash_prefill_kernel(
        q_ptr, k_ptr, v_ptr, table_ptr, gate_ptr, out_ptr, ws_acc, ws_ml, T, start, qk_scale,
        stride_page, stride_head, stride_tok, nsplit, meta_ptr,
        QH: tl.constexpr, GROUP: tl.constexpr, BLOCK_R: tl.constexpr, D: tl.constexpr,
        PAGE: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr, KT: tl.constexpr,
        WIDEN: tl.constexpr, HAS_META: tl.constexpr,
    ):
        """SPLIT (constexpr) says whether the keys are split; ``nsplit`` (runtime, so a new
        split count compiles nothing) how many ways. T rows are computed and stored; with
        HAS_META the real row count and start come from ``meta_ptr`` ([count, start], int64,
        on the device: plan 33's captured prefill) and rows >= count attend as row count - 1."""
        tile = tl.program_id(0)
        kvh = tl.program_id(1)
        split = tl.program_id(2)
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
        r_ok = r < T
        d = tl.arange(0, D)
        qo = (r.to(tl.int64) * QH + head)[:, None] * D + d[None, :]
        q = tl.load(q_ptr + qo, mask=r_ok[:, None], other=0.0)
        if WIDEN:
            q = q.to(tl.float32)
        q_lo = q0 + tl.minimum(tile * BLOCK_R, n_real - 1)  # the tile's first query position
        q_pos = q0 + tl.minimum(r, n_real - 1)
        end = q0 + tl.minimum(tile * BLOCK_R + BLOCK_R, n_real)  # keys [0, end) for this tile
        n_full = q_lo // BLOCK_N  # blocks entirely below the tile's first query: no mask
        n_all = tl.cdiv(end, BLOCK_N)
        m = tl.full([M], float("-inf"), dtype=tl.float32)
        l = tl.zeros([M], dtype=tl.float32)
        acc = tl.zeros([M, D], dtype=tl.float32)
        if not SPLIT:
            lo = n_all * 0
            hi = n_all
        else:
            per = tl.cdiv(n_all, nsplit)
            lo = tl.minimum(split * per, n_all)
            hi = tl.minimum(lo + per, n_all)
        m, l, acc = _blocks(
            q, m, l, acc, k_ptr, v_ptr, table_ptr, kvh, q_pos, end, qk_scale,
            lo, tl.minimum(hi, n_full), stride_page, stride_head, stride_tok,
            D, PAGE, BLOCK_N, False, KT, WIDEN,
        )
        m, l, acc = _blocks(
            q, m, l, acc, k_ptr, v_ptr, table_ptr, kvh, q_pos, end, qk_scale,
            tl.maximum(lo, n_full), hi, stride_page, stride_head, stride_tok,
            D, PAGE, BLOCK_N, True, KT, WIDEN,
        )
        if not SPLIT:
            o = _bf16(acc / l[:, None], WIDEN)  # SDPA's BF16 output
            g = tl.load(gate_ptr + qo, mask=r_ok[:, None], other=0.0).to(tl.float32)
            sig = _bf16(tl.sigmoid(g), WIDEN)  # torch.sigmoid on BF16
            tl.store(out_ptr + qo, _bf16(o * sig, WIDEN).to(tl.bfloat16), mask=r_ok[:, None])
        else:
            row = r.to(tl.int64) * QH + head  # [M]
            base = split.to(tl.int64) * T * QH
            tl.store(ws_acc + (base + row)[:, None] * D + d[None, :], acc, mask=r_ok[:, None])
            tl.store(ws_ml + (base + row) * 2, m, mask=r_ok)
            tl.store(ws_ml + (base + row) * 2 + 1, l, mask=r_ok)

    @triton.jit(do_not_specialize=["T", "nsplit"])
    def _flash_combine_kernel(
        ws_acc, ws_ml, gate_ptr, out_ptr, T, nsplit,
        QH: tl.constexpr, D: tl.constexpr, WIDEN: tl.constexpr,
    ):
        """One (row, head): the SPLIT partials summed in split order, then the epilogue."""
        row = tl.program_id(0).to(tl.int64)
        d = tl.arange(0, D)
        mx = tl.load(ws_ml + row * 2)  # split 0 always holds key 0: finite
        for s in range(1, nsplit):
            mx = tl.maximum(mx, tl.load(ws_ml + (s * T * QH + row) * 2))
        l = mx * 0.0
        acc = tl.zeros([D], dtype=tl.float32)
        for s in range(nsplit):
            ms = tl.load(ws_ml + (s * T * QH + row) * 2)
            w = tl.exp2(ms - mx)  # 0 for an empty split (ms = -inf)
            l += w * tl.load(ws_ml + (s * T * QH + row) * 2 + 1)
            acc += w * tl.load(ws_acc + (s * T * QH + row) * D + d)
        o = _bf16(acc / l, WIDEN)
        g = tl.load(gate_ptr + row * D + d).to(tl.float32)
        sig = _bf16(tl.sigmoid(g), WIDEN)
        tl.store(out_ptr + row * D + d, _bf16(o * sig, WIDEN).to(tl.bfloat16))


# Tile and split choice from shapes alone (plan 35): the fastest of the tune sweep in job j168
# (tools/flash_prefill_probe.py --tune, one GCD, 8 query heads x 1 KV head, D=256, on a
# 3,600-token context, and 2,300 after 5,800), per range of new tokens T. Knobs per row:
# (rows per program, keys per block, waves_per_eu or None, key splits). The choice depends on T
# only (a captured pass knows its start only on the device); a split past the keys is empty.
_TABLE = (
    # T below   rows  keys  wpe   splits    measured (us/layer)
    (96,        8,    64,   None, 13),    # T=61: 75.5
    (192,       8,    64,   None, 6),     # T=128: 117.4
    (330,       8,    64,   None, 3),     # T=256: 198.6
    (480,       16,   128,  1,    4),     # T=400: 239.4
    (780,       16,   64,   None, 2),     # T=561: 327.3, 700: 339.7
    (1100,      16,   128,  1,    3),     # T=850: 471.0, 1000: 479.7
    (1300,      16,   128,  1,    4),     # T=1200: 569.3
    (None,      16,   64,   None, 2),     # T=1400: 612.8, 1650: 661.7, 2300 (after 5,800): 1875.4
)
SPLIT_MIN_BLOCKS = 4  # split_for (the sweeps' explicit tiles)

_CUS: dict = {}


def _cus(device: torch.device) -> int:
    if device.type != "cuda":
        return 104
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _CUS:
        _CUS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _CUS[idx]


def choose(count: int, start: int | None, heads: int, group: int, cus: int = 104) -> dict:
    """Launch knobs for a segment (or a captured bucket) of ``count`` query rows; from
    ``count`` alone (``start`` is not used)."""
    for below, rows, keys, wpe, splits in _TABLE:
        if below is None or count < below:
            break
    cfg = {"block_r": max(rows, 16 // group), "block_n": keys, "num_warps": 4, "num_stages": 1}
    if wpe is not None:
        cfg["waves_per_eu"] = wpe
    cfg["split"] = splits
    return cfg


def split_for(count: int, start: int, heads: int, rows: int, block_n: int, cus: int = 104) -> int:
    """Splits over the keys from shapes alone: enough programs for the CUs on short segments."""
    tiles = -(-count // rows) * heads
    if tiles >= cus:
        return 1
    blocks = -(-(start + count) // block_n)
    return max(1, min(math.ceil(2.5 * cus / tiles), blocks // SPLIT_MIN_BLOCKS, 16))


def flash_prefill_attention(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, table: torch.Tensor,
    start: int, gate: torch.Tensor, scale: float, out: torch.Tensor | None = None, *,
    block_r: int | None = None, block_n: int = 64, split: int | None = None, kt: bool = True,
    num_warps: int = 4, num_stages: int = 1, meta: torch.Tensor | None = None, **launch,
) -> torch.Tensor:
    """As ``prefill_kernels.prefill_attention`` (``meta``: its device-side [count, start]).
    Without ``block_r`` every knob comes from ``choose`` (the engine's path); the knobs are
    for the plan 35 sweeps."""
    count, q_heads, dim = q.shape
    heads, page = k_pool.shape[1], k_pool.shape[2]
    group = q_heads // heads
    if block_r is None:
        cfg = choose(count, start, heads, group, _cus(q.device))
        block_r, block_n, split = cfg["block_r"], cfg["block_n"], cfg["split"]
        num_warps, num_stages = cfg["num_warps"], cfg["num_stages"]
        launch = {k: cfg[k] for k in ("waves_per_eu",) if k in cfg}
    elif split is None:
        split = split_for(count, 0 if meta is not None else start, heads, block_r, block_n, _cus(q.device))
    if meta is not None:
        start = 0
    if INTERPRET:
        launch = {}  # AMD-only launch options
    if out is None:
        out = torch.empty((count, q_heads * dim), dtype=q.dtype, device=q.device)
    assert k_pool.stride() == v_pool.stride()
    q = q.contiguous()
    gate = gate.contiguous()
    if split > 1:
        ws_acc = torch.empty((split, count, q_heads, dim), dtype=torch.float32, device=q.device)
        ws_ml = torch.empty((split, count, q_heads, 2), dtype=torch.float32, device=q.device)
    else:
        ws_acc = ws_ml = out  # unused
    with on_device(q):
        _flash_prefill_kernel[(triton.cdiv(count, block_r), heads, split)](
            q, k_pool, v_pool, table, gate, out, ws_acc, ws_ml, count, start, scale * LOG2E,
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), split,
            meta if meta is not None else table,
            QH=q_heads, GROUP=group, BLOCK_R=block_r, D=dim, PAGE=page, BLOCK_N=block_n,
            SPLIT=split > 1, KT=kt, WIDEN=_WIDEN, HAS_META=meta is not None,
            num_warps=num_warps, num_stages=num_stages,
            **launch,
        )
        if split > 1:
            _flash_combine_kernel[(count * q_heads,)](
                ws_acc, ws_ml, gate, out, count, split, QH=q_heads, D=dim, WIDEN=_WIDEN,
                num_warps=4,
            )
    return out


def warm(k_pool: torch.Tensor, v_pool: torch.Tensor, q_heads: int) -> None:
    """Compile every kernel ``choose`` can launch on this pool's device (each tile of the table,
    split, with and without a device-side ``meta``, and the combine), before serving: reads
    page 0 of the pool, writes nothing to it."""
    heads, page, dim = k_pool.shape[1], k_pool.shape[2], k_pool.shape[3]
    dev = k_pool.device
    count, start = 16, 8 * page
    q = torch.zeros((count, q_heads, dim), dtype=k_pool.dtype, device=dev)
    gate = torch.zeros((count, q_heads * dim), dtype=k_pool.dtype, device=dev)
    table = torch.zeros(-(-(start + count) // page), dtype=torch.long, device=dev)
    meta = torch.tensor([count - 3, start], dtype=torch.long, device=dev)
    seen = set()
    for _, rows, keys, wpe, _ in _TABLE:
        if (rows, keys, wpe) in seen:
            continue
        seen.add((rows, keys, wpe))
        launch = {} if wpe is None else {"waves_per_eu": wpe}
        for m in (None, meta):
            flash_prefill_attention(
                q, k_pool, v_pool, table, start, gate, 1.0, meta=m,
                block_r=max(rows, 16 // (q_heads // heads)), block_n=keys, split=2, **launch,
            )
