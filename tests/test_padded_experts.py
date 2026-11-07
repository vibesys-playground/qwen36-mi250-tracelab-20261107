"""CPU equivalence of the padded-bmm prefill experts path against the per-expert loop."""

import pytest
import torch

from engine import experts as ex


class _Experts(torch.nn.Module):
    def __init__(self, num_experts, hidden, inter, dtype, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_experts = num_experts
        self.gate_up_proj = torch.randn(num_experts, 2 * inter, hidden, generator=g).mul(0.1).to(dtype)
        self.down_proj = torch.randn(num_experts, hidden, inter, generator=g).mul(0.1).to(dtype)
        self.act_fn = torch.nn.functional.silu


def _routing(n, e, k, mode, g):
    if mode == "uniform":
        logits = torch.randn(n, e, generator=g)
    elif mode == "skewed":  # most tokens pick the same few experts
        logits = torch.randn(n, e, generator=g)
        logits[:, :3] += 6.0
    else:  # "sparse": only 3 experts ever get tokens, the rest get zero
        logits = torch.full((n, e), -1e9)
        logits[:, :3] = torch.randn(n, 3, generator=g)
    w, idx = torch.topk(logits, k, dim=-1)
    return idx, torch.softmax(w, dim=-1)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 0.0), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("n", [17, 64, 300, 1000])
@pytest.mark.parametrize("mode", ["uniform", "skewed", "sparse"])
def test_padded_matches_loop(dtype, atol, n, mode):
    e, h, i, k = 16, 32, 16, 3
    m = _Experts(e, h, i, dtype)
    g = torch.Generator().manual_seed(n)
    x = torch.randn(n, h, generator=g).to(dtype)
    idx, w = _routing(n, e, k, mode, g)
    ref = ex._sorted_experts_forward(m, x, idx, w)
    got = ex._padded_experts_forward(m, x, idx, w)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    if dtype == torch.float32:
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    else:
        assert (got.float() - ref.float()).abs().max() <= atol


def test_fallback_on_extreme_skew(monkeypatch):
    e, h, i, k, n = 8, 16, 8, 2, 100
    m = _Experts(e, h, i, torch.float32)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, h, generator=g)
    idx, w = _routing(n, e, k, "sparse", g)
    called = []
    orig = ex._sorted_experts_forward
    monkeypatch.setattr(ex, "_sorted_experts_forward", lambda *a: called.append(1) or orig(*a))
    monkeypatch.setattr(ex, "PADDED_MAX_ROWS", 10)
    got = ex._padded_experts_forward(m, x, idx, w)
    assert called
    torch.testing.assert_close(got, orig(m, x, idx, w))


def test_dispatch(monkeypatch):
    m = _Experts(4, 8, 4, torch.float32)
    x = torch.randn(40, 8)
    idx, w = _routing(40, 4, 2, "uniform", torch.Generator().manual_seed(0))
    torch.testing.assert_close(ex.experts_forward(m, x, idx, w), ex._sorted_experts_forward(m, x, idx, w), rtol=1e-5, atol=1e-5)
