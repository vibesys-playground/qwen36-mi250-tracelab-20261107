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

    def to(self, device):  # plain tensor attributes: nn.Module.to would leave them on the CPU
        self.gate_up_proj = self.gate_up_proj.to(device)
        self.down_proj = self.down_proj.to(device)
        return self


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
@pytest.mark.parametrize("one", [False, True])
def test_tile_table(bm, mode, one):
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
        mp.build_tiles(counts.to(dev), meta[0], meta[1], table, max_tiles, bm, one=one)
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


@pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")
@pytest.mark.parametrize("tokens,bm", [(61, 16), (561, 64), (1650, 64)])
@pytest.mark.parametrize("mode", ["uniform", "skewed", "sparse"])
def test_tile_table_one_program_bitwise(tokens, bm, mode):
    """Plan 56: ``_tiles_one_kernel`` writes the same buffers as ``_tiles_kernel`` (Qwen3.6:
    256 experts, top 8), bit for bit, including the untouched entries past the tile count."""
    dev = _device()
    e, k = 256, 8
    g = torch.Generator().manual_seed(tokens)
    idx, _ = _routing(tokens, e, k, mode, g)
    counts = torch.bincount(idx.reshape(-1), minlength=e).to(torch.int32).to(dev)
    max_tiles = -(-(tokens * k) // bm) + e
    out = []
    for one in (False, True):
        meta = torch.full((2, e), -1, dtype=torch.int32, device=dev)
        table = torch.full((3 * max_tiles + 1,), -7, dtype=torch.int32, device=dev)
        with kernels.on_device(meta):
            mp.build_tiles(counts, meta[0], meta[1], table, max_tiles, bm, one=one)
        out.append((meta.cpu(), table.cpu()))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])


def _pair_buffers(x, idx, gate_up, down, cfg):
    """The grouped path's bookkeeping and act (``moe_prefill``'s steps), for the down tests."""
    tokens, k = idx.shape
    experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    pairs = tokens * k
    dev = x.device
    ids = idx.reshape(-1)
    order = torch.argsort(ids, stable=True).to(torch.int32)
    counts = torch.zeros(experts, dtype=torch.int32, device=dev)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    meta = torch.empty((2, experts), dtype=torch.int32, device=dev)
    max_tiles = -(-pairs // cfg.bm) + experts
    table = torch.empty(3 * max_tiles + 1, dtype=torch.int32, device=dev)
    act = torch.empty((pairs, inter), dtype=torch.bfloat16, device=dev)
    bn1, bk1 = min(cfg.gate_up.bn, inter), min(cfg.gate_up.bk, hidden)
    with kernels.on_device(x):
        mp.build_tiles(counts, meta[0], meta[1], table, max_tiles, cfg.bm)
        mp._gate_up_kernel[(max_tiles, -(-inter // bn1))](
            x, gate_up, order, table, act, gate_up.stride(0), gate_up.stride(1), max_tiles,
            H=hidden, I=inter, TOP_K=k, BLOCK_M=cfg.bm, BLOCK_N=bn1, BLOCK_K=bk1, WIDEN=mp._WIDEN,
            **mp.launch_options(cfg.gate_up),
        )
    return act, order, table, max_tiles


@pytest.mark.skipif(_device() is None, reason="needs Triton (GPU or TRITON_INTERPRET=1)")
@pytest.mark.parametrize("inter", [128, 256])
@pytest.mark.parametrize("n", [5, 17, 300])
@pytest.mark.parametrize("mode", ["uniform", "skewed", "sparse"])
@pytest.mark.parametrize("scfg", [None, "64,2,4,1", "128,4,4,2", "256,2,8,1,16,2", "128,8,4,2,16,0"])
def test_down_stream_bitwise(monkeypatch, inter, n, mode, scfg):
    """Plan 60: ``_down_stream_kernel`` writes the pair buffer of ``_down_kernel`` bit for
    bit (the same BLOCK_K dots in the same order), with empty experts (sparse), one expert
    over 256 rows (skewed, n=300) and every bucket's BLOCK_M (n=5: 16, else 64)."""
    if scfg:
        monkeypatch.setenv("ENGINE_MOE_DOWN_STREAM_CFG", scfg)
    else:
        monkeypatch.delenv("ENGINE_MOE_DOWN_STREAM_CFG", raising=False)
    dev = _device()
    e, h, k = 8, 1024, 4
    m = _Experts(e, h, inter, seed=n).to(dev)
    g = torch.Generator().manual_seed(n + inter)
    x = torch.randn(n, h, generator=g).bfloat16().to(dev)
    idx, w = _routing(n, e, k, mode, g)
    idx, w = idx.to(dev), w.to(dev)
    pairs = n * k
    cfg = mp.config(pairs, e)
    act, order, table, max_tiles = _pair_buffers(x, idx, m.gate_up_proj, m.down_proj, cfg)
    outs = []
    for stream in (False, True, True):
        out = torch.full((pairs, h), float("nan"), dtype=torch.bfloat16, device=dev)
        with kernels.on_device(x):
            mp.launch_down(act, m.down_proj, order, table, out, max_tiles, cfg, pairs, stream=stream)
        outs.append(out.cpu())
    assert not torch.isnan(outs[0].float()).any()
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
    if mode == "skewed" and n == 300:
        assert int(torch.bincount(idx.reshape(-1).cpu()).max()) > 256
    # The whole routed output through ``moe_prefill`` with the stream kernel on and off.
    monkeypatch.setattr(mp, "_DOWN_STREAM", False)
    y0 = mp.moe_prefill(x, idx, w, m.gate_up_proj, m.down_proj)
    monkeypatch.setattr(mp, "_DOWN_STREAM", True)
    y1 = mp.moe_prefill(x, idx, w, m.gate_up_proj, m.down_proj)
    assert torch.equal(y0, y1)


def test_stream_config_buckets(monkeypatch):
    """One ``StreamConfig`` per ``BUCKETS`` entry, picked from the token count alone."""
    monkeypatch.delenv("ENGINE_MOE_DOWN_STREAM_CFG", raising=False)
    e, k = 256, 8
    assert len(mp.STREAM_BUCKETS) == len(mp.BUCKETS)
    for t in mp.bucket_tokens(k, e, 4096):
        i = [c for _, c in mp.BUCKETS].index(mp.config(t * k, e))
        assert mp.stream_config(t * k, e) == mp.STREAM_BUCKETS[i]
    monkeypatch.setenv("ENGINE_MOE_DOWN_STREAM_CFG", "128,4,8,2")
    assert mp.stream_config(561 * k, e) == mp.StreamConfig(128, 4, 8, 2)
