"""The sync-free grouped prefill experts (``engine.fast.moe_prefill``) against the current paths.

The Triton kernels run under the Triton interpreter on the CPU
(``TRITON_INTERPRET=1`` set before Triton is imported) or on a GPU; without
Triton only the reference test runs.
"""

import pytest
import torch

from engine import experts as ex
from engine.fast import kernels, moe_prefill as mp


class _Experts(torch.nn.Module):
    def __init__(self, num_experts, hidden, inter, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_experts = num_experts
        self.gate_up_proj = torch.randn(num_experts, 2 * inter, hidden, generator=g).mul(0.1).bfloat16()
        self.down_proj = torch.randn(num_experts, hidden, inter, generator=g).mul(0.1).bfloat16()
        self.act_fn = torch.nn.functional.silu


def _routing(n, e, k, mode, g):
    logits = torch.randn(n, e, generator=g)
    if mode == "skewed":  # most tokens pick the same few experts
        logits[:, :3] += 6.0
    elif mode == "sparse":  # only k experts ever get tokens
        logits[:, k:] = -1e9
    w, idx = torch.topk(torch.softmax(logits, -1), k, dim=-1)
    return idx, (w / w.sum(-1, keepdim=True)).bfloat16()


TOL = 2e-2


def _device():
    if torch.cuda.is_available() and kernels.triton is not None:
        return torch.device("cuda")
    if kernels.INTERPRET and kernels.triton is not None:
        return torch.device("cpu")
    return None


@pytest.mark.parametrize("mode", ["uniform", "skewed", "sparse"])
def test_reference_matches_loop(mode):
    e, h, i, k, n = 16, 64, 64, 4, 50
    m = _Experts(e, h, i)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, h, generator=g).bfloat16()
    idx, w = _routing(n, e, k, mode, g)
    ref = ex._sorted_experts_forward(m, x, idx, w)
    got = mp.moe_prefill_reference(x, idx, w, m.gate_up_proj, m.down_proj)
    assert (got.float() - ref.float()).abs().max() <= TOL * max(1.0, float(ref.float().abs().max()))


@pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")
@pytest.mark.parametrize("n", [17, 40, 300])
@pytest.mark.parametrize("mode", ["uniform", "skewed", "sparse"])
@pytest.mark.parametrize("cfg", [None, "16,64,64,64,64,4,1", "64,64,64,128,64,4,1"])
def test_grouped_matches_padded_and_loop(monkeypatch, n, mode, cfg):
    if cfg:
        monkeypatch.setenv("ENGINE_MOE_GROUPED_CFG", cfg)
    dev = _device()
    e, h, i, k = 16, 128, 64, 4
    m = _Experts(e, h, i).to(dev)
    g = torch.Generator().manual_seed(n)
    x = torch.randn(n, h, generator=g).bfloat16().to(dev)
    idx, w = _routing(n, e, k, mode, g)
    idx, w = idx.to(dev), w.to(dev)
    assert mp.supported(x, m.gate_up_proj, m.down_proj)
    got = mp.moe_prefill(x, idx, w, m.gate_up_proj, m.down_proj)
    loop = ex._sorted_experts_forward(m, x, idx, w)
    padded = ex._padded_experts_forward(m, x, idx, w)
    assert got.dtype == torch.bfloat16 and got.shape == loop.shape
    scale = loop.float().abs().max()
    # BF16 rounding of the same products; reduction order inside a dot may differ.
    assert (got.float() - loop.float()).abs().max() <= TOL * max(1.0, float(scale))
    assert (got.float() - padded.float()).abs().max() <= TOL * max(1.0, float(scale))
    # Deterministic: a second run gives the same bits.
    assert torch.equal(got, mp.moe_prefill(x, idx, w, m.gate_up_proj, m.down_proj))


@pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")
def test_grouped_unused_experts_and_all_on_one():
    """Every pair on one expert (one long group, many tiles), most experts empty."""
    dev = _device()
    e, h, i, k, n = 8, 64, 64, 1, 70
    m = _Experts(e, h, i, seed=2).to(dev)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(n, h, generator=g).bfloat16().to(dev)
    idx = torch.full((n, k), 5, dtype=torch.long, device=dev)
    w = torch.ones((n, k), dtype=torch.bfloat16, device=dev)
    got = mp.moe_prefill(x, idx, w, m.gate_up_proj, m.down_proj)
    ref = mp.moe_prefill_reference(x, idx, w, m.gate_up_proj, m.down_proj)
    assert (got.float() - ref.float()).abs().max() <= TOL * max(1.0, float(ref.float().abs().max()))
