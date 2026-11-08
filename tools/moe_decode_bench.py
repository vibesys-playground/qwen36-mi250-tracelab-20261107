"""Plan 22 step 0: work order and tiles of the fused MoE decode kernels on one gfx90a GCD.

    python tools/moe_decode_bench.py [--devices 6,7] [--layers 4] [--samples 8] [--quick]
    TRITON_INTERPRET=1 python tools/moe_decode_bench.py --smoke     # CPU check of the kernels

1. Routing and weights. One 2-GCD replica is loaded as the server loads it (``Generator.load``).
   ``--sessions`` replay-like prompts (Request Factory's synthetic coding lines, as
   ``tools/spec_step_bench.py`` builds them; not the benchmark's sessions) are prefilled and
   greedy-decoded together, eagerly (graphs off), for ``--steps`` steps; the MoE input ``h``
   and residual ``x`` of every row are recorded at ``--layers`` MoE layers held by the first
   GCD. Row sets mirror a verify step (each slot's token, then the next token of the same
   session as its draft): R=1 one row, R=2 one slot, R=4 two slots, R=8 four slots, at
   ``--samples`` different (step, sessions). Each set's router logits, top-8 and weights come
   from today's kernels (``FusedDecode._gemv``, ``_topk_kernel``) on the real layer weights,
   which stay where the engine loaded them (no copy).
2. Kernels, timed on cuda:0 only, as CUDA graphs of back-to-back launches cycling over every
   (layer, sample) set of one R (each launch reads a different layer's experts than the one
   before; the 8 MB L2 holds none of a launch's 70-280 MB):
   - ``cur``: ``fused._moe_up_kernel`` / ``fused._moe_down_kernel`` exactly as ``_moe`` calls
     them (BN 8, BK 1024, 2 programs per CU, 4 warps);
   - ``up-o{order}-g{grid}-n{BN}-k{BK}-w{warps}``: a copy of the up kernel with a work order
     (0: today's pair-major, tile inner; 1: tile-outer, pairs sorted by expert id, the sort
     done in-kernel from idx; 2: tile-outer, pairs unsorted, a control), programs per CU
     (0 = one program per work item), BN, BK and warps;
   - ``dn-o{order}-g{grid}-n{BN}-w{warps}``: a copy of the down kernel (0: today's (row,
     tile) order; 1: (tile, row): all rows of an output tile on adjacent programs), BKI = I.
   The per-item arithmetic of a copy is today's line for line; only the item -> (pair, tile)
   map and the constexpr tiles differ.
   (a) is order 1 at today's tiles, (b) the sweep at order 0, (c) both.
   Bitwise: every config's ``act`` (up) and ``out`` (down, fed today's act) against today's
   kernels on every set of every R (torch.equal).
   Bytes: unique = distinct experts of the set (up 2*I*H*2, down H*I*2) + shared down once;
   total = every (row, k) pair (and the shared down per row).
   L2 dedup check: rocprofv3 is not in the candidate image, so FETCH_SIZE is reported only if
   it is on PATH. The proxy instead: the same sets with every duplicate expert replaced by an
   unused one (``distinct`` control, same pair count). If a work order lets rows share expert
   reads through L2, it runs faster on the real sets than on their distinct control.
3. The pick: per R the fastest up and down configs (any, and bitwise-identical), their
   up+down time per layer (kill rule: <= 150 us at R=4), and one fixed (up, down) config
   timed at every R.

Prints ``RESULT`` lines. A hint for the plan, not the task's metric.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine.fast import fused  # noqa: E402
from engine.fast import kernels as _kernels  # noqa: E402

triton, tl = _kernels.triton, _kernels.tl

ROWS = (1, 2, 4, 8)

if triton is not None:

    @triton.jit
    def _moe_up_x(
        h_ptr, idx_ptr, w_ptr, act_ptr, R, stride_we,
        H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr,
        BK: tl.constexpr, ORDER: tl.constexpr, PP: tl.constexpr,
    ):
        """``fused._moe_up_kernel`` with a work order: ORDER 0 item t -> (pair t // ntile,
        tile t % ntile) as today; 1 -> (tile t // P, the (t % P)-th pair sorted by expert id,
        ties by pair); 2 -> (tile t // P, pair t % P)."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ntile = tl.cdiv(I, BN)
        P = R * TOPK
        pp = tl.arange(0, PP)
        rank = pp
        if ORDER == 1:
            e = tl.load(idx_ptr + pp, mask=pp < P, other=2147483647)
            before = (e[None, :] < e[:, None]) | ((e[None, :] == e[:, None]) & (pp[None, :] < pp[:, None]))
            rank = tl.sum(before.to(tl.int32), axis=1)
        for t in range(pid, P * ntile, nprog):
            if ORDER == 0:
                pair = t // ntile
                tile = t % ntile
            else:
                tile = t // P
                if ORDER == 1:
                    pair = tl.sum(tl.where(rank == t % P, pp, 0), axis=0)
                else:
                    pair = t % P
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
    def _moe_down_x(
        act_ptr, r_ptr, idx_ptr, wt_ptr, w_ptr, sd_ptr, res_ptr, out_ptr, R, stride_we, N3,
        H: tl.constexpr, I: tl.constexpr, IS: tl.constexpr, E: tl.constexpr,
        TOPK: tl.constexpr, BN: tl.constexpr, BKI: tl.constexpr, BKS: tl.constexpr,
        ORDER: tl.constexpr,
    ):
        """``fused._moe_down_kernel`` with a work order: ORDER 0 item t -> (row t // ntile,
        tile t % ntile) as today; 1 -> (tile t // R, row t % R)."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        ntile = tl.cdiv(H, BN)
        for t in range(pid, R * ntile, nprog):
            if ORDER == 0:
                row = t // ntile
                tile = t % ntile
            else:
                row = t % R
                tile = t // R
            rows = tile * BN + tl.arange(0, BN)
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


def out(msg: str) -> None:
    print(msg, flush=True)


def p2(n: int) -> int:
    return triton.next_power_of_2(n)


# ============================================================================ sets


class Layer:
    """One MoE layer's weights (views of the engine's tensors, or random in --smoke)."""

    def __init__(self, name, gu, dn, sd, w3=None, gemv=None):
        self.name, self.gu, self.dn, self.sd, self.w3, self.gemv = name, gu, dn, sd, w3, gemv
        self.E, two_i, self.H = gu.shape
        self.I = two_i // 2
        self.IS = sd.shape[1]
        self.N3 = self.E + 2 * self.IS + 1


class Set:
    """One routing set: R rows of one layer, today's routing and today's outputs."""

    def __init__(self, layer: Layer, h, x, r, topk):
        self.layer, self.h, self.x, self.r = layer, h, x, r
        self.R = h.shape[0]
        self.topk = topk
        dev = h.device
        self.idx = torch.empty((self.R, topk), dtype=torch.int32, device=dev)
        self.wt = torch.empty((self.R, topk), dtype=torch.float32, device=dev)
        fused._topk_kernel[(self.R,)](r, self.idx, self.wt, layer.N3, E=layer.E, TOPK=topk,
                                      TOPKP=p2(topk), num_warps=1)
        self.act = torch.empty((self.R * topk, layer.I), dtype=h.dtype, device=dev)
        self.out = torch.empty_like(x)
        self.ref_act = None
        self.ref_out = None
        self.nu = 0

    def distinct(self) -> "Set":
        """The same set with every duplicate expert replaced by an unused one."""
        c = Set.__new__(Set)
        c.__dict__.update(self.__dict__)
        flat = self.idx.flatten().tolist()
        used = set(flat)
        spare = [e for e in range(self.layer.E) if e not in used]
        seen, new = set(), []
        for e in flat:
            if e in seen:
                e = spare.pop()
            seen.add(e)
            new.append(e)
        c.idx = torch.tensor(new, dtype=torch.int32, device=self.idx.device).view_as(self.idx)
        c.act = torch.empty_like(self.act)
        c.out = torch.empty_like(self.out)
        c.nu = len(new)
        return c


# ============================================================================ launchers

CUR_UP = ("cur",)


def up_cfg_name(cfg) -> str:
    if cfg == CUR_UP:
        return "up-cur"
    o, g, n, k, w = cfg
    return f"up-o{o}-g{g}-n{n}-k{k}-w{w}"


def dn_cfg_name(cfg) -> str:
    if cfg == CUR_UP:
        return "dn-cur"
    o, g, n, w = cfg
    return f"dn-o{o}-g{g}-n{n}-w{w}"


def make_launchers(cus: int, max_rows: int):
    def up(s: Set, cfg, act=None):
        L = s.layer
        act = s.act if act is None else act
        if cfg == CUR_UP:  # exactly as FusedDecode._moe
            bn = fused.TILES["moe_bn"]
            items = s.R * s.topk * triton.cdiv(L.I, bn)
            fused._moe_up_kernel[(min(items, 2 * cus),)](
                s.h, s.idx, L.gu, act, s.R, L.gu.stride(0),
                H=L.H, I=L.I, TOPK=s.topk, BN=bn, BK=min(1024, p2(L.H)), num_warps=4,
            )
            return act
        o, g, bn, bk, w = cfg
        items = s.R * s.topk * triton.cdiv(L.I, bn)
        grid = items if g == 0 else min(items, g * cus)
        _moe_up_x[(grid,)](
            s.h, s.idx, L.gu, act, s.R, L.gu.stride(0),
            H=L.H, I=L.I, TOPK=s.topk, BN=bn, BK=min(bk, p2(L.H)), ORDER=o,
            PP=p2(max_rows * s.topk), num_warps=w,
        )
        return act

    def down(s: Set, cfg, act=None, outt=None):
        L = s.layer
        act = s.act if act is None else act
        outt = s.out if outt is None else outt
        if cfg == CUR_UP:
            bn = fused.TILES["moe_bn"]
            items = s.R * triton.cdiv(L.H, bn)
            fused._moe_down_kernel[(min(items, 2 * cus),)](
                act, s.r, s.idx, s.wt, L.dn, L.sd, s.x, outt, s.R, L.dn.stride(0), L.N3,
                H=L.H, I=L.I, IS=L.IS, E=L.E, TOPK=s.topk, BN=bn,
                BKI=min(1024, p2(L.I)), BKS=min(1024, p2(L.IS)), num_warps=4,
            )
            return outt
        o, g, bn, w = cfg
        items = s.R * triton.cdiv(L.H, bn)
        grid = items if g == 0 else min(items, g * cus)
        _moe_down_x[(grid,)](
            act, s.r, s.idx, s.wt, L.dn, L.sd, s.x, outt, s.R, L.dn.stride(0), L.N3,
            H=L.H, I=L.I, IS=L.IS, E=L.E, TOPK=s.topk, BN=bn,
            BKI=min(1024, p2(L.I)), BKS=min(1024, p2(L.IS)), ORDER=o, num_warps=w,
        )
        return outt

    return up, down


def up_configs(quick: bool, H: int):
    orders, grids, bns, bks, warps = ((0, 1, 2), (2, 4, 8, 0), (8, 16, 32), (1024, 2048), (4, 8))
    if quick:
        orders, grids, bns, bks, warps = (0, 1, 2), (2, 4), (8, 16), (1024,), (4,)
    cfgs = []
    for o in orders:
        for g in grids:
            for n in bns:
                for k in bks:
                    k = min(k, p2(H))
                    if n * k > 32768:
                        continue
                    for w in warps:
                        cfgs.append((o, g, n, k, w))
    return list(dict.fromkeys(cfgs))


def dn_configs(quick: bool):
    orders, grids, bns, warps = (0, 1), (2, 4, 8, 0), (8, 16, 32), (4, 8)
    if quick:
        grids, bns, warps = (2, 4), (8, 16), (4,)
    return [(o, g, n, w) for o in orders for g in grids for n in bns for w in warps]


# ============================================================================ timing


def time_us(fns, min_launches: int = 32, reps: int = 5) -> float:
    """us per call of a list of launch closures, captured back to back as one CUDA graph."""
    seq = list(fns) * max(1, math.ceil(min_launches / len(fns)))
    for f in fns:
        f()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fns[0]()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for f in seq:
            f()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(seq))
    del graph
    return statistics.median(times)


# ============================================================================ the bench


def bench(sets_by_r: dict[int, list[Set]], cus: int, max_rows: int, args, timing: bool,
          extra=None) -> None:
    up, down = make_launchers(cus, max_rows)
    any_set = next(iter(sets_by_r.values()))[0]
    L0 = any_set.layer
    H, I, IS, topk = L0.H, L0.I, L0.IS, any_set.topk
    up_b = 2 * I * H * 2
    dn_b = H * I * 2
    sd_b = H * IS * 2

    # references and routing stats
    for R, sets in sets_by_r.items():
        nus = []
        for s in sets:
            s.ref_act = up(s, CUR_UP).clone()
            s.ref_out = down(s, CUR_UP).clone()
            s.nu = len(set(s.idx.flatten().tolist()))
            nus.append(s.nu)
        mean_nu = statistics.mean(nus)
        out(f"RESULT routing R={R} sets={len(sets)} pairs={R * topk} distinct_mean={mean_nu:.2f} "
            f"distinct_min={min(nus)} distinct_max={max(nus)} dup_share={1 - mean_nu / (R * topk):.3f}")
    if extra is not None:
        extra(up, down)

    def bytes_of(sets, kind):
        uq = statistics.mean(s.nu for s in sets)
        R = sets[0].R
        if kind == "up":
            return uq * up_b, R * topk * up_b
        if kind == "dn":
            return uq * dn_b + sd_b, R * topk * dn_b + R * sd_b
        return uq * (up_b + dn_b) + sd_b, R * topk * (up_b + dn_b) + R * sd_b

    def report(kind, R, label, us, sets, extra_s=""):
        ub, tb = bytes_of(sets, kind)
        out(f"RESULT t kind={kind} R={R} cfg={label} us={us:.1f} GBps_unique={ub / us / 1e3:.0f} "
            f"GBps_total={tb / us / 1e3:.0f}{extra_s}")

    # bitwise check of every config on every set
    ucfgs = up_configs(args.quick, H)
    dcfgs = dn_configs(args.quick)
    bit_up: dict = {}
    bit_dn: dict = {}
    for cfg in ucfgs:
        ok = True
        try:
            for sets in sets_by_r.values():
                for s in sets:
                    s.act.fill_(float("nan"))
                    a = up(s, cfg)
                    if not torch.equal(a, s.ref_act):
                        ok = False
            bit_up[cfg] = ok
            out(f"RESULT bitwise kind=up cfg={up_cfg_name(cfg)} identical={ok}")
        except Exception as exc:  # noqa: BLE001
            bit_up[cfg] = None
            out(f"RESULT error cfg={up_cfg_name(cfg)} {type(exc).__name__}: {str(exc)[:300]}")
    for cfg in dcfgs:
        ok = True
        try:
            for sets in sets_by_r.values():
                for s in sets:
                    s.out.fill_(float("nan"))
                    o = down(s, cfg, act=s.ref_act)
                    if not torch.equal(o, s.ref_out):
                        ok = False
            bit_dn[cfg] = ok
            out(f"RESULT bitwise kind=dn cfg={dn_cfg_name(cfg)} identical={ok}")
        except Exception as exc:  # noqa: BLE001
            bit_dn[cfg] = None
            out(f"RESULT error cfg={dn_cfg_name(cfg)} {type(exc).__name__}: {str(exc)[:300]}")
    out(f"RESULT bitwise_summary up_identical={sum(1 for v in bit_up.values() if v)}/{len(bit_up)} "
        f"dn_identical={sum(1 for v in bit_dn.values() if v)}/{len(bit_dn)}")
    if not timing:
        return

    # sweep
    tu: dict = {}
    td: dict = {}
    for R, sets in sets_by_r.items():
        for cfg in [CUR_UP] + [c for c in ucfgs if bit_up.get(c) is not None]:
            try:
                us = time_us([lambda s=s: up(s, cfg) for s in sets])
                tu[(R, cfg)] = us
                report("up", R, up_cfg_name(cfg), us, sets, f" bitwise={bit_up.get(cfg, True)}")
            except Exception as exc:  # noqa: BLE001
                out(f"RESULT error R={R} cfg={up_cfg_name(cfg)} {type(exc).__name__}: {str(exc)[:200]}")
        for cfg in [CUR_UP] + [c for c in dcfgs if bit_dn.get(c) is not None]:
            try:
                us = time_us([lambda s=s: down(s, cfg, act=s.ref_act) for s in sets])
                td[(R, cfg)] = us
                report("dn", R, dn_cfg_name(cfg), us, sets, f" bitwise={bit_dn.get(cfg, True)}")
            except Exception as exc:  # noqa: BLE001
                out(f"RESULT error R={R} cfg={dn_cfg_name(cfg)} {type(exc).__name__}: {str(exc)[:200]}")

    def best(table, R, pred):
        c = [(us, cfg) for (r, cfg), us in table.items() if r == R and pred(cfg)]
        return min(c) if c else (float("nan"), None)

    a_up = (1, 2, fused.TILES["moe_bn"], min(1024, p2(H)), 4)
    a_dn = (1, 2, fused.TILES["moe_bn"], 4)
    t_up = (0, 2, fused.TILES["moe_bn"], min(1024, p2(H)), 4)
    t_dn = (0, 2, fused.TILES["moe_bn"], 4)
    is_ord0 = lambda c: c != CUR_UP and c[0] == 0  # noqa: E731
    bit_u = lambda c: c != CUR_UP and bit_up.get(c)  # noqa: E731
    bit_d = lambda c: c != CUR_UP and bit_dn.get(c)  # noqa: E731
    anyc = lambda c: c != CUR_UP  # noqa: E731

    def combined(R, ucfg, dcfg) -> float:
        sets = sets_by_r[R]
        fns = []
        for s in sets:
            fns.append(lambda s=s: (up(s, ucfg), down(s, dcfg)))
        return time_us(fns)

    pairs_all = set()
    for R in sets_by_r:
        rows = {
            "today": (CUR_UP, CUR_UP),
            "today-copy": (t_up, t_dn),
            "a-reorder": (a_up, a_dn),
            "b-sweep-order0": (best(tu, R, is_ord0)[1], best(td, R, is_ord0)[1]),
            "c-best-bitwise": (best(tu, R, bit_u)[1], best(td, R, bit_d)[1]),
            "c-best-any": (best(tu, R, anyc)[1], best(td, R, anyc)[1]),
        }
        for label, (uc, dc) in rows.items():
            if uc is None or dc is None:
                continue
            pairs_all.add((uc, dc))
            us = combined(R, uc, dc)
            ok = (uc == CUR_UP or bit_up.get(uc)) and (dc == CUR_UP or bit_dn.get(dc))
            report("updn", R, f"{label}:{up_cfg_name(uc)}+{dn_cfg_name(dc)}", us, sets_by_r[R],
                   f" bitwise={bool(ok)} step_ms_41={us * 41 / 1e3:.2f}")

    # one fixed config across every R
    fixed = {}
    for uc, dc in sorted(pairs_all, key=str):
        ts = {R: combined(R, uc, dc) for R in sets_by_r}
        fixed[(uc, dc)] = ts
        ok = (uc == CUR_UP or bit_up.get(uc)) and (dc == CUR_UP or bit_dn.get(dc))
        out(f"RESULT fixed cfg={up_cfg_name(uc)}+{dn_cfg_name(dc)} bitwise={bool(ok)} "
            + " ".join(f"R{R}_us={t:.1f}" for R, t in ts.items()))
    r4 = 4 if 4 in sets_by_r else max(sets_by_r)
    for want_bit in (True, False):
        cands = [(ts[r4], sum(ts.values()), k) for k, ts in fixed.items()
                 if not want_bit or ((k[0] == CUR_UP or bit_up.get(k[0])) and (k[1] == CUR_UP or bit_dn.get(k[1])))]
        if not cands:
            continue
        t4, _, (uc, dc) = min(cands)
        base4 = fixed.get((CUR_UP, CUR_UP), {}).get(r4, float("nan"))
        verdict = "PASS" if t4 <= 150.0 else "KILL"
        out(f"RESULT pick bitwise_only={want_bit} cfg={up_cfg_name(uc)}+{dn_cfg_name(dc)} "
            f"R{r4}_us={t4:.1f} today_R{r4}_us={base4:.1f} cut={1 - t4 / base4:.3f} "
            f"step_moe_ms_41={t4 * 41 / 1e3:.2f} kill_rule(<=150us)={verdict}")

    # L2 dedup proxy: the same sets with duplicate experts made distinct
    for R in (4, 8):
        if R not in sets_by_r:
            continue
        dsets = [s.distinct() for s in sets_by_r[R]]
        cands = {"today": (CUR_UP, CUR_UP), "a-reorder": (a_up, a_dn)}
        bu, bd = best(tu, R, anyc)[1], best(td, R, anyc)[1]
        if bu and bd:
            cands["c-best-any"] = (bu, bd)
        for label, (uc, dc) in cands.items():
            for kind, fn in (("up", lambda s: up(s, uc)), ("dn", lambda s: down(s, dc, act=s.ref_act))):
                real = time_us([lambda s=s: fn(s) for s in sets_by_r[R]])
                dist = time_us([lambda s=s: fn(s) for s in dsets])
                ub_real = bytes_of(sets_by_r[R], kind)[0]
                ub_dist = bytes_of(dsets, kind)[0]
                out(f"RESULT dedup kind={kind} R={R} cfg={label} real_us={real:.1f} distinct_us={dist:.1f} "
                    f"real/distinct={real / dist:.3f} unique_bytes_ratio={ub_real / ub_dist:.3f}")


# ============================================================================ modes


def smoke(args) -> None:
    """CPU (TRITON_INTERPRET=1): small shapes, random weights, the bitwise checks only."""
    torch.manual_seed(0)
    E, H, I, IS, topk = 32, 128, 32, 32, 8
    layers = []
    for li in range(1):
        gu = (torch.randn(E, 2 * I, H) * H ** -0.5).to(torch.bfloat16)
        dn = (torch.randn(E, H, I) * I ** -0.5).to(torch.bfloat16)
        sd = (torch.randn(H, IS) * IS ** -0.5).to(torch.bfloat16)
        layers.append(Layer(f"L{li}", gu, dn, sd))
    sets_by_r = {}
    for R in ROWS:
        sets = []
        for L in layers:
            for k in range(2):
                h = torch.randn(R, H).to(torch.bfloat16)
                x = torch.randn(R, H).to(torch.bfloat16)
                r = torch.randn(R, L.N3).to(torch.bfloat16)
                if R > 1:  # force shared experts
                    r[1::2, :E] = r[0, :E]
                sets.append(Set(L, h, x, r, topk))
        sets_by_r[R] = sets
    args.quick = True
    bench(sets_by_r, cus=4, max_rows=8, args=args, timing=False)


def real(args) -> None:
    os.environ.setdefault("ENGINE_MTP", "1")
    from engine.generator import CHECKPOINT, Generator

    started = time.perf_counter()
    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    tok = g.tokenizer
    if fast.fused is None:
        raise SystemExit(f"needs the fused layers: {getattr(fast, 'fused_error', None)}")
    dev0 = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev0)
    cus = props.multi_processor_count
    out(f"RESULT env load_s={time.perf_counter() - started:.0f} device={props.name} "
        f"gcn={getattr(props, 'gcnArchName', '?')} cus={cus} torch={torch.__version__} "
        f"triton={triton.__version__} visible={os.environ.get('HIP_VISIBLE_DEVICES')} "
        f"gpus={torch.cuda.device_count()} max_rows={fast.fused.max_rows} tiles={fused.TILES} "
        f"rocprofv3={shutil.which('rocprofv3')}")
    if not shutil.which("rocprofv3"):
        out("RESULT fetch_size unavailable: rocprofv3 is not in the candidate image "
            "(Dockerfile.rocm installs hsa-rocr and rocminfo only); see the dedup proxy lines")

    # layers on cuda:0, spread over depth
    on0 = [i for i in range(len(fast.layers)) if fast.layers[i].mlp.experts.gate_up_proj.device == dev0]
    pick = [on0[round(j * (len(on0) - 1) / max(1, args.layers - 1))] for j in range(args.layers)]
    pick = list(dict.fromkeys(pick))
    out(f"RESULT layers on_gcd0={len(on0)} picked={pick}")

    # prompts and an eager decode that records each picked layer's MoE inputs
    def pool(limit: int, nonce: str = "5f0c2a9e1b7d4c3a8e6f0b2d4a6c8e0f") -> list[int]:
        ids: list[int] = []
        i = 0
        while len(ids) < limit:
            for line in (f"def case_{i}(items):", f"    return sorted(set(items))  # {i} {nonce}"):
                ids.extend(tok.encode(line, add_special_tokens=False))
            i += 1
        return ids[:limit]

    S = args.sessions
    corpus = pool(9973 * S + args.prompt_len + 10)
    prompts = [corpus[9973 * k : 9973 * k + args.prompt_len] for k in range(S)]

    def prefill(slot: int, prompt: list[int]) -> torch.Tensor:
        fast.reset(slot)
        body = len(prompt) - 1
        position = 0
        while position < body:
            end = min((position // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[position:end], position, prompt[end])
            position = end
        return fast.prefill(slot, prompt[body:], body)

    rec: dict[int, list] = {i: [] for i in pick}
    state = {"on": False}
    moe0 = fast.fused._moe

    def moe_rec(i, x, h):
        if state["on"] and i in rec:
            rec[i].append((h.detach().clone(), x.detach().clone()))
        return moe0(i, x, h)

    fast.fused._moe = moe_rec
    t0 = time.perf_counter()
    with torch.inference_mode():
        slots = list(range(S))
        lasts = [int(torch.argmax(prefill(s, p))) for s, p in zip(slots, prompts, strict=True)]
        fast.use_graphs = False
        gen = [[t] for t in lasts]
        for step in range(args.steps):
            state["on"] = True
            logits = fast.decode(lasts, slots, [args.prompt_len + step] * S)
            state["on"] = False
            lasts = logits.argmax(-1).tolist()
            for s in range(S):
                gen[s].append(lasts[s])
        for s in slots:
            fast.reset(s)
    fast.fused._moe = moe0
    out(f"RESULT decode sessions={S} steps={args.steps} s={time.perf_counter() - t0:.1f} "
        f"sample={tok.decode(gen[0][:24])!r}")

    # routing sets mirroring a verify step: slot tokens, then their next tokens (the drafts)
    T = args.steps
    fd = fast.fused
    layers = {}
    for i in pick:
        mlp = fast.layers[i].mlp
        layers[i] = Layer(f"L{i}", mlp.experts.gate_up_proj, mlp.experts.down_proj,
                          mlp.shared_expert.down_proj.weight, w3=fd.w_moe[i])
    sets_by_r: dict[int, list[Set]] = {}
    with torch.inference_mode():
        for R in ROWS:
            nslot = max(1, R // 2)
            sets = []
            for i in pick:
                hs = rec[i]
                for k in range(args.samples):
                    t = (k * 5 + 3) % (T - 1)
                    sess = [(k * nslot + j) % S for j in range(nslot)]
                    if R == 1:
                        spec = [(t, sess[0])]
                    else:
                        spec = [(t, s) for s in sess] + [(t + 1, s) for s in sess]
                    h = torch.stack([hs[tt][0][s] for tt, s in spec]).contiguous()
                    x = torch.stack([hs[tt][1][s] for tt, s in spec]).contiguous()
                    r = fd._gemv(h, layers[i].w3)
                    sets.append(Set(layers[i], h, x, r, fast.top_k))
            sets_by_r[R] = sets
        torch.cuda.synchronize()

        def extra(up, down):
            # sanity: the cur launchers reproduce FusedDecode._moe bitwise; today's full MoE time
            for R, sets in sets_by_r.items():
                same = all(torch.equal(fd._moe(int(s.layer.name[1:]), s.x, s.h), s.ref_out) for s in sets)
                us = time_us([lambda s=s: fd._moe(int(s.layer.name[1:]), s.x, s.h) for s in sets])
                out(f"RESULT today_moe_full R={R} us={us:.1f} (router GEMV + top-k + up + down) "
                    f"matches_cur_launchers={same} step_ms_41={us * 41 / 1e3:.2f}")
            big = torch.empty(int(256e6), dtype=torch.uint8, device=dev0)
            big2 = torch.empty_like(big)
            us = time_us([lambda: big2.copy_(big)], min_launches=8)
            out(f"RESULT copy_ref bytes_rw=5.12e8 us={us:.1f} GBps={5.12e8 / us / 1e3:.0f}")
            del big, big2

        torch.cuda.set_device(dev0)
        bench(sets_by_r, cus=cus, max_rows=fast.fused.max_rows, args=args, timing=True, extra=extra)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="6,7")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--prompt-len", type=int, default=1800)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="CPU, TRITON_INTERPRET=1, bitwise only")
    args = parser.parse_args()
    if not args.smoke:
        os.environ.setdefault("HIP_VISIBLE_DEVICES", args.devices)
    try:
        if args.smoke:
            smoke(args)
        else:
            real(args)
    except Exception:  # noqa: BLE001
        out("RESULT fatal " + traceback.format_exc().replace("\n", " | "))
        raise


if __name__ == "__main__":
    main()
