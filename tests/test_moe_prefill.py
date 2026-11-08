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
@pytest.mark.parametrize("n", [10, 17, 40, 300])
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


def test_config_buckets_by_token_count(monkeypatch):
    """Tile shapes depend on the token count only; every bucket is reachable and warmed."""
    monkeypatch.delenv("ENGINE_MOE_GROUPED_CFG", raising=False)
    e, k = 256, 8
    seen = {mp.config(t * k, e) for t in range(17, 4097)}
    assert seen == {cfg for _, cfg in mp.BUCKETS}
    for t in mp.bucket_tokens(k, e, 4096):
        assert mp.config(t * k, e) in seen
    assert {mp.config(t * k, e) for t in mp.bucket_tokens(k, e, 4096)} == seen
    assert mp.config(61 * k, e).bm == 16 and mp.config(561 * k, e).bm == 64
    # Every tuned shape keeps BLOCK_K at 128 (the summation order of today's tiles).
    assert all(c.gate_up.bk == 128 and c.down.bk == 128 for _, c in mp.BUCKETS)
    monkeypatch.setenv("ENGINE_MOE_GROUPED_CFG", "32,64,128,128,128,8,1")
    c = mp.config(561 * k, e)
    assert c.bm == 32 and c.gate_up == mp.KernelConfig(64, 128, 8, 1) and c.down == mp.KernelConfig(128, 128, 8, 1)


@pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")
@pytest.mark.parametrize("bm", [16, 64])
@pytest.mark.parametrize("mode", ["uniform", "skewed", "sparse"])
def test_tile_table(bm, mode):
    dev = _device()
    e, k, n = 16, 4, 75
    g = torch.Generator().manual_seed(bm)
    idx, _ = _routing(n, e, k, mode, g)
    counts = torch.bincount(idx.reshape(-1), minlength=e).to(torch.int32)
    pairs = n * k
    max_tiles = -(-pairs // bm) + e
    meta = torch.full((2, e), -1, dtype=torch.int32, device=dev)
    table = torch.full((3 * max_tiles + 1,), -7, dtype=torch.int32, device=dev)
    with kernels.on_device(meta):
        mp._tiles_kernel[(-(-max_tiles // mp.TILE_BLOCK),)](
            counts.to(dev), meta[0], meta[1], table, max_tiles, E=e, BLOCK_M=bm, BLOCK_T=mp.TILE_BLOCK,
        )
    starts = torch.cumsum(counts, 0) - counts
    want = []
    for ex_, (c, s) in enumerate(zip(counts.tolist(), starts.tolist())):
        for j in range(-(-c // bm)):
            want.append((ex_, s + j * bm, s + c))
    table = table.cpu()
    assert torch.equal(meta[0].cpu(), starts.int()) and torch.equal(meta[1].cpu(), starts.int())
    assert int(table[3 * max_tiles]) == len(want)
    got = [(int(table[t]), int(table[max_tiles + t]), int(table[2 * max_tiles + t])) for t in range(len(want))]
    assert got == want
