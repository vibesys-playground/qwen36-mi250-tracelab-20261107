"""Fused prefill kernels (``engine.fast.prefill_kernels``) against the eager ops they replace.

The Triton kernels run on a GPU or under Triton's CPU interpreter (``TRITON_INTERPRET=1`` set
before Triton is imported); skipped otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vibesys" / "shared" / "cpu_check"))

from engine.fast import kernels  # noqa: E402
from engine.fast import prefill_kernels as pk  # noqa: E402


def _device():
    if torch.cuda.is_available() and kernels.triton is not None:
        return torch.device("cuda")
    if kernels.INTERPRET and kernels.triton is not None:
        return torch.device("cpu")
    return None


pytestmark = pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")


def _close(got, ref, tol=2e-2):
    got, ref = got.float(), ref.float()
    scale = max(1.0, float(ref.abs().max()))
    err = float((got - ref).abs().max())
    assert err <= tol * scale, f"max error {err} vs scale {scale}"


@pytest.mark.parametrize("q_heads,kv_heads,dim,page", [(4, 2, 32, 64), (8, 1, 64, 16)])
@pytest.mark.parametrize("count,start", [(1, 0), (5, 0), (37, 0), (40, 100), (300, 70)])
def test_attention_matches_masked_sdpa(q_heads, kv_heads, dim, page, count, start):
    dev = _device()
    g = torch.Generator().manual_seed(count * 7 + start)
    total = start + count
    used = -(-total // page)
    pages = used + 5
    k_pool = torch.randn(pages, kv_heads, page, dim, generator=g).bfloat16()
    v_pool = torch.randn(pages, kv_heads, page, dim, generator=g).bfloat16()
    # Positions past the sequence hold garbage (NaN) that must never reach the output.
    perm = torch.randperm(pages, generator=g)[:used]
    for j, p in enumerate(perm.tolist()):
        lo = j * page
        bad = max(0, min(page, total - lo))
        k_pool[p, :, bad:] = float("nan")
        v_pool[p, :, bad:] = float("nan")
    q = torch.randn(count, q_heads, dim, generator=g).bfloat16()
    gate = torch.randn(count, q_heads * dim, generator=g).bfloat16()
    table = perm.long()
    k_pool, v_pool, q, gate, table = (t.to(dev) for t in (k_pool, v_pool, q, gate, table))
    scale = dim ** -0.5
    ref = pk.prefill_attention_reference(
        q, k_pool.nan_to_num(), v_pool.nan_to_num(), table, start, gate, scale
    )
    got = pk.prefill_attention(q, k_pool, v_pool, table, start, gate, scale)
    assert got.shape == ref.shape and got.dtype == torch.bfloat16
    assert not got.isnan().any()
    _close(got, ref)


@pytest.mark.parametrize("has_h,has_peer,has_w", [(False, False, True), (True, False, True), (True, True, True), (True, True, False)])
def test_reduce_rms_matches_eager(has_h, has_peer, has_w):
    dev = _device()
    g = torch.Generator().manual_seed(1)
    rows, n = 19, 96
    res = torch.randn(rows, n, generator=g).bfloat16().to(dev)
    h = torch.randn(rows, n, generator=g).bfloat16().to(dev) if has_h else None
    peer = torch.randn(rows, n, generator=g).bfloat16().to(dev) if has_peer else None
    w = (torch.randn(n, generator=g) * 0.1).bfloat16().to(dev) if has_w else None
    ref_x, ref_y = pk.reduce_rms_reference(res, h, peer, w, 1e-6)
    x, y = pk.reduce_rms(res, h, peer, w, 1e-6)
    assert torch.equal(x, ref_x)  # the adds round exactly as the eager adds
    if has_w:
        _close(y, ref_y, 1e-2)
    else:
        assert y is None


@pytest.mark.parametrize("experts,k,tokens", [(8, 2, 33), (64, 8, 50)])
def test_route_matches_softmax_topk(experts, k, tokens):
    dev = _device()
    g = torch.Generator().manual_seed(experts)
    logits = (torch.randn(tokens, experts, generator=g) * 2).bfloat16().to(dev)
    w, i, counts = pk.route(logits, k)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    rw, ri = torch.topk(probs, k, dim=-1)
    rw = (rw / rw.sum(-1, keepdim=True)).bfloat16()
    for row in range(tokens):  # the same experts, up to ties (equal BF16 logits)
        if set(i[row].tolist()) != set(ri[row].tolist()):
            assert torch.equal(probs[row, i[row]].sort().values, probs[row, ri[row]].sort().values)
    _close(w.sort(-1).values, rw.sort(-1).values, 1e-2)
    assert torch.equal(counts.long().cpu(), torch.bincount(i.reshape(-1).cpu(), minlength=experts))
    assert i.dtype == torch.long and w.dtype == torch.bfloat16


def _mlp(experts, hidden, inter, shared_inter, seed):
    g = torch.Generator().manual_seed(seed)

    def lin(o, i_):
        return NS(weight=(torch.randn(o, i_, generator=g) * 0.1).bfloat16())

    return NS(
        shared_expert=NS(
            gate_proj=lin(shared_inter, hidden), up_proj=lin(shared_inter, hidden),
            down_proj=lin(hidden, shared_inter),
        ),
        gate=lin(experts, hidden),
        shared_expert_gate=lin(1, hidden),
        experts=NS(
            gate_up_proj=(torch.randn(experts, 2 * inter, hidden, generator=g) * 0.1).bfloat16(),
            down_proj=(torch.randn(experts, hidden, inter, generator=g) * 0.1).bfloat16(),
        ),
    )


def _to(mlp, dev):
    for part in (mlp.shared_expert.gate_proj, mlp.shared_expert.up_proj, mlp.shared_expert.down_proj,
                 mlp.gate, mlp.shared_expert_gate):
        part.weight = part.weight.to(dev)
    mlp.experts.gate_up_proj = mlp.experts.gate_up_proj.to(dev)
    mlp.experts.down_proj = mlp.experts.down_proj.to(dev)
    return mlp


@pytest.mark.parametrize("tokens", [17, 90])
def test_moe_block_matches_eager(tokens):
    dev = _device()
    experts, hidden, inter, k = 16, 64, 32, 4
    mlp = _to(_mlp(experts, hidden, inter, 48, tokens), dev)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(tokens, hidden, generator=g).bfloat16().to(dev)
    assert pk.moe_supported(x, mlp, k)
    ref = pk.moe_reference(x, mlp, k)
    got = pk.moe(x, mlp, k)
    _close(got, ref)
    again = pk.moe(x, mlp, k)
    assert torch.equal(got, again)  # deterministic despite the atomic expert order


# ------------------------------------------------------------------------- the whole prefill


@pytest.fixture(scope="module")
def fast(tmp_path_factory: pytest.TempPathFactory):
    if not kernels.INTERPRET:
        pytest.skip("the tiny model test runs on the CPU interpreter")
    from tiny_checkpoint import write_checkpoint

    from engine.fast.model import FastModel
    from engine.generator import Generator

    path = tmp_path_factory.mktemp("tiny")
    write_checkpoint(path)
    gen = Generator.load(str(path), None, max_model_len=1024, chunk_tokens=8,
                         prefix_cache_tokens=0, prefix_cache_entries=0)
    return FastModel(gen.model, slots=8, max_len=1024, use_triton=True, use_graphs=False)


def test_prefill_fused_matches_eager(fast):
    prompt = [(7 * j + 3) % 200 + 1 for j in range(90)]
    out = {}
    for flag in (False, True):
        fast.triton_kernels["prefill_fused"] = flag
        with torch.inference_mode():
            fast.reset(0)
            a = fast.prefill(0, prompt[:40], 0).float()  # fresh
            da = None if fast.mtp is None else fast.last_drafts.clone()
            b = fast.prefill(0, prompt[40:], 40).float()  # on top of a state, with cached KV
            db = None if fast.mtp is None else fast.last_drafts.clone()
            nxt = int(b.argmax())
            c = fast.decode([nxt], [0], [len(prompt)]).float()[0]
        out[flag] = (a, b, c, da, db)
    fast.triton_kernels["prefill_fused"] = True
    for x, y in zip(out[False][:3], out[True][:3]):
        assert int(x.argmax()) == int(y.argmax())
        assert float((x - y).abs().max()) <= 0.05 * float(x.abs().max()) + 1e-3
    if fast.mtp is not None:
        assert torch.equal(out[False][3], out[True][3]) and torch.equal(out[False][4], out[True][4])


def test_packed_prefill_fused_matches_eager(fast):
    p1 = [(5 * j + 1) % 200 + 1 for j in range(30)]
    p2 = [(11 * j + 2) % 200 + 1 for j in range(50)]
    out = {}
    for flag in (False, True):
        fast.triton_kernels["prefill_fused"] = flag
        with torch.inference_mode():
            fast.reset(0)
            fast.reset(1)
            fast.prefill(0, p1[:20], 0)
            out[flag] = fast.prefill_packed([(0, p1[20:], 20), (1, p2, 0)]).float()
    fast.triton_kernels["prefill_fused"] = True
    a, b = out[False], out[True]
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    assert float((a - b).abs().max()) <= 0.05 * float(a.abs().max()) + 1e-3
