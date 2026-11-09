"""Plan 50: the HIP GDN step (``engine.fast.gdn_hip.layer_core``) on one GPU, at the TP2 rank
shapes of Qwen3.6-35B-A3B (16 v-heads, 8 k-heads, DK = DV = 128) with random weights:

- a chained step of G rows equals G single-row (decode) steps, bit for bit (states, conv
  ring, output), and the same at every slot count (the column split changes no bits);
- close to today's Triton kernels (``mfma.gdn_chain``'s two kernels);
- a captured graph replays the eager bits.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _layer(dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    nk, nv, dk, dv = 8, 16, 128, 128
    kd, vd = nk * dk, nv * dv
    conv_dim = 2 * kd + vd

    def bf(t):
        return t.to(torch.bfloat16).to(dev)

    return NS(
        conv1d=NS(weight=bf(torch.randn(conv_dim, 1, 4, generator=g) * 0.3)),
        in_proj_z=NS(weight=torch.empty(vd, 1)), in_proj_b=NS(weight=torch.empty(nv, 1)),
        in_proj_a=NS(weight=torch.empty(nv, 1)),
        A_log=torch.log(torch.empty(nv).uniform_(1, 16, generator=g)).to(dev),
        dt_bias=(torch.randn(nv, generator=g) * 0.5 + 1).to(dev),
        norm=NS(weight=bf(1 + 0.1 * torch.randn(dv, generator=g)), variance_epsilon=1e-6),
        num_k_heads=nk, num_v_heads=nv, head_k_dim=dk, head_v_dim=dv, key_dim=kd, value_dim=vd,
        conv_dim=conv_dim,
    )


def _states(m, rows, dev, seed=1):
    g = torch.Generator().manual_seed(seed)
    conv = (torch.randn(rows, m.conv_dim, 4, generator=g)).to(torch.bfloat16).to(dev)
    rec = (torch.randn(rows, m.num_v_heads, 128, 128, generator=g) * 0.05).to(dev)
    return conv, rec


def _proj(m, rows, dev, seed=2):
    g = torch.Generator().manual_seed(seed)
    width = m.conv_dim + m.value_dim + 2 * m.num_v_heads
    return torch.randn(rows, width, generator=g).to(torch.bfloat16).to(dev)


SLOTS = 9  # ring rows of slot s, entry j: s + j * SLOTS


def _ring(n, groups, dev):
    return torch.tensor([s + j * SLOTS for j in range(groups) for s in range(n)], dtype=torch.long, device=dev)


@cuda
@pytest.mark.parametrize("n,groups", [(1, 4), (2, 4), (2, 2), (4, 3)])
def test_chain_equals_decode_steps(n, groups):
    from engine.fast import gdn_hip

    dev = torch.device("cuda", 0)
    m = _layer(dev)
    assert gdn_hip.supported(m)
    gdn_hip.prepare(m)
    conv, rec = _states(m, SLOTS * groups, dev)
    proj = _proj(m, n * groups, dev)
    states = _ring(n, groups, dev)
    c1, r1 = conv.clone(), rec.clone()
    core = gdn_hip.layer_core(m, c1, r1, proj, states, groups)
    # the same rows as G decode steps: state row of entry j-1 copied to entry j, then G = 1
    c2, r2 = conv.clone(), rec.clone()
    outs = []
    for j in range(groups):
        rows = states[j * n:(j + 1) * n]
        if j:
            prev = states[(j - 1) * n:j * n]
            c2[rows] = c2[prev]
            r2[rows] = r2[prev]
        outs.append(gdn_hip.layer_core(m, c2, r2, proj[j * n:(j + 1) * n].contiguous(), rows, 1))
    torch.cuda.synchronize()
    assert torch.equal(core, torch.cat(outs))
    assert torch.equal(r1[states], r2[states])
    assert torch.equal(c1[states], c2[states])
    # one slot alone (another column split) gives the same bits as inside the batch
    if n > 1:
        c3, r3 = conv.clone(), rec.clone()
        s0 = states.view(groups, n)[:, :1].reshape(-1).contiguous()
        p0 = proj.view(groups, n, -1)[:, 0].contiguous()
        core0 = gdn_hip.layer_core(m, c3, r3, p0, s0, groups)
        torch.cuda.synchronize()
        assert torch.equal(core0, core.view(groups, n, -1)[:, 0])
        assert torch.equal(r3[s0], r1[s0])


@cuda
def test_close_to_triton():
    from engine.fast import gdn_hip, mfma

    dev = torch.device("cuda", 0)
    m = _layer(dev, 3)
    gdn_hip.prepare(m)
    n, groups = 2, 4
    conv, rec = _states(m, SLOTS * groups, dev, 4)
    proj = _proj(m, n * groups, dev, 5)
    states = _ring(n, groups, dev)
    c1, r1 = conv.clone(), rec.clone()
    core = gdn_hip.layer_core(m, c1, r1, proj, states, groups)
    c2, r2 = conv.clone(), rec.clone()
    p2 = proj.clone()
    ref = torch.empty_like(core)
    heads = m.num_v_heads
    z_off = 2 * m.key_dim + m.value_dim
    mfma._gdn_conv_chain_kernel[(n, (m.conv_dim + 255) // 256)](
        c2, states, p2, m.conv1d.weight, n, m.conv_dim, p2.stride(0), c2.stride(0), G=groups, BLOCK=256)
    mfma._gdn_chain_step_kernel[(n, heads)](
        r2, states, p2, m.A_log, m.dt_bias, m.norm.weight, ref, n, p2.stride(0), heads,
        r2.stride(0), r2.stride(1), 1e-6, 128 ** -0.5, z_off, z_off + m.value_dim, z_off + m.value_dim + heads,
        RATIO=2, KEY_DIM=m.key_dim, DK=128, DV=128, G=groups, num_warps=8)
    torch.cuda.synchronize()
    assert torch.equal(c1[states], c2[states])
    a, b = r1[states], r2[states]
    assert float((a - b).abs().max()) <= 1e-5 * float(b.abs().max())
    diff = (core.float() - ref.float()).abs()
    assert float(diff.max()) <= 0.02 * float(ref.float().abs().max())
    assert float((core != ref).float().mean()) < 0.02


@cuda
def test_graph_replay_equal():
    from engine.fast import gdn_hip

    dev = torch.device("cuda", 0)
    m = _layer(dev, 6)
    gdn_hip.prepare(m)
    n, groups = 1, 4
    conv, rec = _states(m, SLOTS * groups, dev, 7)
    proj = _proj(m, n * groups, dev, 8)
    states = _ring(n, groups, dev)
    ce, re_ = conv.clone(), rec.clone()
    cg, rg = conv.clone(), rec.clone()
    stream = torch.cuda.Stream(dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out_g = gdn_hip.layer_core(m, cg, rg, proj, states, groups)
    for _ in range(3):
        out_e = gdn_hip.layer_core(m, ce, re_, proj, states, groups)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_e, out_g)
        assert torch.equal(re_, rg) and torch.equal(ce, cg)
