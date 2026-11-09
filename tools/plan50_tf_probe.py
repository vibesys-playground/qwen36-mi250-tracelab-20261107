"""Plan 50: where do the HIP GDN path (ENGINE_GDN_HIP=1) and today's Triton GDN kernels part under
teacher forcing, and which one is closer to a float64 GDN?

    ENGINE_TP=2 ENGINE_MTP=0 python tools/plan50_tf_probe.py --ref /tmp/p50_off.pt

One TP2 replica (``Generator.load``, decode steps eager so the GDN layer can be switched per
call). ``--ref`` is a tools/tp_parity.py reference (prompts, the fed tokens, picks, margins).
For each prompt, three slots are prefilled identically, then fed the reference's tokens one
decode step at a time, each slot with its own GDN implementation:

  T  today's Triton kernels (``_gemv`` conv epilogue + ``_gdn_step_kernel``)
  H  the HIP kernel (``gdn_hip.layer_core``)
  F  float64 GDN: the same bf16 rounding points (conv output, silu, beta, o, norm, y, out), the
     conv, l2 norms, gates and delta rule in float64 with a float64 state kept per slot (the
     fp32 state row gets its rounded copy). Every other op as served.

Every step: each slot's greedy pick and top-2 margin; per GDN layer (both ranks) the state
error of T and of H against F's float64 state (max |err| / max |F|), the H - T state
difference, and the max |diff| of the layer's core output (pre out_proj) H vs T, T vs F, H vs F;
the logits' max |diff|. Prints every step where two of {ref, T, H, F} pick differently with a
margin > ``--margin`` (details per layer), and per prompt the agreement of T and H with F.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="/tmp/p50_off.pt")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--prompts", default="0,1,2,3,4,5,6,7")
    args = p.parse_args()
    os.environ.setdefault("ENGINE_MTP", "0")
    os.environ.setdefault("ENGINE_TP", "2")

    import torch

    from engine.fast import fused, gdn_hip
    from engine.generator import CHECKPOINT, Generator

    g = Generator.load(CHECKPOINT["model"], CHECKPOINT["revision"], max_model_len=262144,
                       chunk_tokens=args.chunk, prefix_cache_tokens=0, prefix_cache_entries=0)
    fast = g.fast
    ranks = getattr(fast, "ranks", [fast])
    for r in ranks:
        r.use_graphs = False  # eager decode: the GDN layer below is chosen per call
    saved = torch.load(args.ref)
    prompts = saved["prompts"]
    print(f"RESULT probe ranks={len(ranks)} hip_layers="
          f"{sum(getattr(r.layers[i].linear_attn, '_gdn_hip', None) is not None for r in ranks for i in r.gdn_state)}",
          flush=True)

    state = {"mode": "T"}
    cores: dict = {}
    f64: dict = {}  # (rank id, layer, row) -> float64 state

    def bf(t):
        return t.to(torch.bfloat16).double()

    def f64_step(rank_id, i, m, conv_state, rec_state, proj, slots):
        rows = proj.shape[0]
        C = 2 * m.key_dim + m.value_dim
        H, DK, DV, NK = m.num_v_heads, m.head_k_dim, m.head_v_dim, m.num_k_heads
        core = torch.empty((rows, H * DV), dtype=proj.dtype, device=proj.device)
        w = m.conv1d.weight.reshape(C, 4).double()
        z_off = C
        b_off, a_off = z_off + m.value_dim, z_off + m.value_dim + H
        for r_i, row in enumerate(slots[:rows].tolist()):
            x = proj[r_i]
            win = conv_state[row, :, 1:]
            full = torch.cat([win, x[:C, None]], -1)
            y = bf((full.double() * w).sum(-1))
            y = bf(y * torch.sigmoid(y))
            conv_state[row] = full
            q = y[: m.key_dim].view(NK, DK)
            k = y[m.key_dim: 2 * m.key_dim].view(NK, DK)
            v = y[2 * m.key_dim:].view(H, DV)
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            q = q.repeat_interleave(H // NK, 0)
            k = k.repeat_interleave(H // NK, 0)
            b = x[b_off: b_off + H].double()
            a = x[a_off: a_off + H].double() + m.dt_bias.double()
            beta = bf(torch.sigmoid(b))
            sp = torch.where(a > 20, a, torch.log1p(torch.exp(a)))
            dec = torch.exp(-torch.exp(m.A_log.double()) * sp)
            key = (rank_id, i, row)
            S = f64.get(key)
            if S is None:
                S = rec_state[row].double()
            S = S * dec[:, None, None]
            kv = torch.einsum("hk,hkv->hv", k, S)
            S = S + k[..., None] * ((v - kv) * beta[:, None])[:, None, :]
            o = bf(torch.einsum("hk,hkv->hv", q, S))
            nrm = bf(o / torch.sqrt((o * o).mean(-1, keepdim=True) + m.norm.variance_epsilon))
            yv = bf(m.norm.weight.double() * nrm)
            z = x[z_off: z_off + m.value_dim].double().view(H, DV)
            core[r_i] = (yv * z * torch.sigmoid(z)).reshape(-1).to(proj.dtype)
            f64[key] = S
            rec_state[row] = S.float()
        return core

    def patched(self, i, h, slots, defer=False):
        mode = state["mode"]
        fastm = self.fast
        m = fastm.layers[i].linear_attn
        conv_state, rec_state = fastm.gdn_state[i]
        conv_dim = 2 * m.key_dim + m.value_dim
        if mode == "H":
            proj = self._gemv(h, self.w_in[i])
            core = gdn_hip.layer_core(m, conv_state, rec_state, proj, slots, 1)
        elif mode == "T":
            proj = self._gemv(h, self.w_in[i], conv=(conv_state, slots, m.conv1d.weight, conv_dim))
            core = torch.empty((h.shape[0], m.num_v_heads * m.head_v_dim), dtype=h.dtype, device=h.device)
            self._gdn_step(m, rec_state, slots, proj, core)
        else:
            proj = self._gemv(h, self.w_in[i])
            core = f64_step(id(fastm), i, m, conv_state, rec_state, proj, slots)
        cores[(mode, id(fastm), i)] = core.float().clone()
        return self._gemv(core, m.out_proj.weight, defer=defer)

    fused.FusedDecode._gdn = patched

    def prefill(slot, prompt):
        fast.reset(slot)
        body = len(prompt) - 1
        pos = 0
        while pos < body:
            end = min((pos // args.chunk + 1) * args.chunk, body)
            fast.prefill(slot, prompt[pos:end], pos, prompt[end])
            pos = end
        return fast.prefill(slot, prompt[body:], body)

    modes = (("T", 0), ("H", 1), ("F", 2))
    layers = sorted(ranks[0].gdn_state)
    tot = {"T_vs_F": [0, 0], "H_vs_F": [0, 0], "T_vs_ref": [0, 0], "H_vs_T": [0, 0]}
    big = {"T_vs_F": [], "H_vs_F": [], "T_vs_ref": [], "H_vs_T": []}
    for k in [int(v) for v in args.prompts.split(",")]:
        f64.clear()
        prompt = prompts[k]
        firsts = [prefill(s, prompt).float() for _, s in modes]
        same0 = all(torch.equal(firsts[0], f) for f in firsts[1:])
        st0 = all(torch.equal(r.gdn_state[i][1][r._g(0)], r.gdn_state[i][1][r._g(s)])
                  for r in ranks for i in layers for s in (1, 2))
        print(f"RESULT prompt={k} len={len(prompt)} prefill_logits_equal={same0} prefill_states_equal={st0}",
              flush=True)
        feed = saved["inputs"][k]
        for step, tok in enumerate(feed):
            outs, picks, margins = {}, {}, {}
            for mode, s in modes:
                state["mode"] = mode
                out = fast.decode([tok], [s], [len(prompt) + step]).float()[0]
                outs[mode] = out
                top = torch.topk(out, 2)
                picks[mode] = int(top.indices[0])
                margins[mode] = float(top.values[0] - top.values[1])
            ref_pick, ref_margin = saved["picks"][k][step + 1], saved["margins"][k][step + 1]
            pairs = {"T_vs_F": ("T", "F"), "H_vs_F": ("H", "F"), "H_vs_T": ("H", "T")}
            flips = []
            for name, (a, b) in pairs.items():
                tot[name][1] += 1
                if picks[a] == picks[b]:
                    tot[name][0] += 1
                else:
                    mg = margins[b]
                    flips.append((name, mg))
                    if mg > 2:
                        big[name].append((k, step, round(mg, 3)))
            tot["T_vs_ref"][1] += 1
            if picks["T"] == ref_pick:
                tot["T_vs_ref"][0] += 1
            else:
                flips.append(("T_vs_ref", ref_margin))
                if ref_margin > 2:
                    big["T_vs_ref"].append((k, step, round(ref_margin, 3)))
            if picks["H"] != ref_pick:
                flips.append(("H_vs_ref", ref_margin))
            if any(mg > args.margin for _, mg in flips):
                rows = []
                for r_id, r in enumerate(ranks):
                    for i in layers:
                        rec = r.gdn_state[i][1]
                        T, Hh = rec[r._g(0)].double(), rec[r._g(1)].double()
                        F = f64.get((id(r), i, r._g(2)))
                        fm = float(F.abs().max())
                        cT, cH, cF = (cores[(mm, id(r), i)] for mm in ("T", "H", "F"))
                        rows.append((r_id, i, float((T - F).abs().max()) / fm, float((Hh - F).abs().max()) / fm,
                                     float((Hh - T).abs().max()) / float(T.abs().max()),
                                     float((cH - cT).abs().max()), float((cT - cF).abs().max()),
                                     float((cH - cF).abs().max())))
                print(f"RESULT flip prompt={k} step={step} ref_pick={ref_pick} ref_margin={ref_margin:.4f} "
                      f"picks={picks} margins={ {m_: round(v, 4) for m_, v in margins.items()} } flips={flips} "
                      f"logits_maxdiff H-T={float((outs['H'] - outs['T']).abs().max()):.4f} "
                      f"T-F={float((outs['T'] - outs['F']).abs().max()):.4f} "
                      f"H-F={float((outs['H'] - outs['F']).abs().max()):.4f}", flush=True)
                for mode in ("T", "H", "F"):
                    tv = torch.topk(outs[mode], 5)
                    print(f"RESULT flip_top5 prompt={k} step={step} {mode} ids={tv.indices.tolist()} "
                          f"logits={[round(v, 3) for v in tv.values.tolist()]} "
                          f"ref_pick_logit={float(outs[mode][ref_pick]):.3f}", flush=True)
                worst = max(rows, key=lambda x: max(x[2], x[3]))
                print(f"RESULT flip_state prompt={k} step={step} max_over_layers "
                      f"T_vs_f64={max(x[2] for x in rows):.3e} H_vs_f64={max(x[3] for x in rows):.3e} "
                      f"H_vs_T={max(x[4] for x in rows):.3e} core_maxabs H-T={max(x[5] for x in rows):.4f} "
                      f"T-F={max(x[6] for x in rows):.4f} H-F={max(x[7] for x in rows):.4f} worst_layer={worst[:2]}",
                      flush=True)
                for x in rows:
                    print(f"RESULT flip_layer prompt={k} step={step} rank={x[0]} layer={x[1]} "
                          f"state_relerr T={x[2]:.3e} H={x[3]:.3e} H-T={x[4]:.3e} "
                          f"core_maxabs H-T={x[5]:.4f} T-F={x[6]:.4f} H-F={x[7]:.4f}", flush=True)
        # end-of-prompt state error summary
        eT = eH = 0.0
        for r in ranks:
            for i in layers:
                rec = r.gdn_state[i][1]
                F = f64[(id(r), i, r._g(2))]
                fm = float(F.abs().max())
                eT = max(eT, float((rec[r._g(0)].double() - F).abs().max()) / fm)
                eH = max(eH, float((rec[r._g(1)].double() - F).abs().max()) / fm)
        print(f"RESULT prompt_end prompt={k} steps={len(feed)} state_relerr_vs_f64 max T={eT:.3e} H={eH:.3e}",
              flush=True)
        for _, s in modes:
            fast.reset(s)
    for name, (a, n) in tot.items():
        print(f"RESULT agree {name} {a}/{n} ({100.0 * a / max(n, 1):.2f}%) flips_with_margin_gt2={big[name]}",
              flush=True)


if __name__ == "__main__":
    main()
