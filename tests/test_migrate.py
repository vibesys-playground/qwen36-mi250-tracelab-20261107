"""Plan 49: a session tail exported from one replica and imported into another is the same
state, byte for byte, and the importing replica's prefix cache hands it to the next round.

CPU only: a stand-in for one rank of ``FastModel`` (its KV pools, GDN state banks and page
pool) is enough, since ``engine.migrate`` touches nothing else.
"""

from __future__ import annotations

import pytest
import torch
from engine import migrate
from engine.fast.model import PAGE, FastModel, PagePool, Snapshot
from engine.generator import FastPrefixCache


class _Rank:
    """The state ``engine.migrate`` reads and writes, shaped like one rank of the model."""

    def __init__(self, pages: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.pool = PagePool(pages)
        self.kv = {
            i: tuple(torch.randn((pages, 1, PAGE, 16), generator=g).to(torch.bfloat16) for _ in range(2))
            for i in (3, 7, 40)
        }
        self.gdn_state = {
            i: (
                torch.randn((6, 24, 4), generator=g).to(torch.bfloat16),
                torch.randn((6, 2, 8, 8), generator=g),
            )
            for i in (0, 1, 2, 4)
        }

    def release(self, snap: Snapshot) -> None:
        FastModel.release(self, snap)

    snapshot_bytes = staticmethod(FastModel.snapshot_bytes)


def _snapshot(model: _Rank, length: int, row: int) -> Snapshot:
    pages = [model.pool.alloc() for _ in range(-(-length // PAGE))]
    gdn = [(c[row].clone(), r[row].clone()) for c, r in model.gdn_state.values()]
    return Snapshot(length=length, gdn=gdn, pages=pages)


@pytest.mark.parametrize("length", [1, PAGE, PAGE + 1, 5 * PAGE - 3])
def test_round_trip_is_bitwise(length: int) -> None:
    src, dst = _Rank(16, 0), _Rank(16, 1)
    for _ in range(3):  # the target's free pages are not the source's
        dst.pool.alloc()
    snap = _snapshot(src, length, row=2)
    bufs = migrate.staging(src, length)
    manifest = migrate.export_into(src, snap, bufs)
    assert manifest.pages == len(snap.pages)
    assert manifest.rank_bytes[0] <= bufs[0].numel()
    # Through the host channel, as between two processes.
    channel = migrate.HostChannel(2 * manifest.rank_bytes[0] + 4096, ranks=1)
    channel.put(manifest, bufs)
    landed = migrate.staging(dst, length)
    channel.get(manifest, landed)
    got = migrate.import_from(dst, manifest, landed)
    assert got.length == length and len(got.pages) == len(snap.pages)
    assert set(got.pages).isdisjoint({0, 1, 2})
    for (sk, sv), (dk, dv) in zip(src.kv.values(), dst.kv.values(), strict=True):
        assert torch.equal(sk[snap.pages], dk[got.pages])
        assert torch.equal(sv[snap.pages], dv[got.pages])
    for (sc, sr), (dc, dr) in zip(snap.gdn, got.gdn, strict=True):
        assert torch.equal(sc, dc) and torch.equal(sr, dr)


def test_layout_mismatch_is_refused() -> None:
    src, dst = _Rank(8, 0), _Rank(8, 1)
    dst.kv.pop(40)
    snap = _snapshot(src, 70, row=0)
    bufs = migrate.staging(src, 70)
    manifest = migrate.export_into(src, snap, bufs)
    free = dst.pool.free
    with pytest.raises(ValueError):
        migrate.import_from(dst, manifest, bufs)
    assert dst.pool.free == free


def test_imported_tail_is_handed_to_the_next_round() -> None:
    src, dst = _Rank(16, 0), _Rank(16, 1)
    key = tuple(range(100, 230))
    src_cache = FastPrefixCache(src, chunk_tokens=64, max_tokens=10_000, max_entries=8)
    dst_cache = FastPrefixCache(dst, chunk_tokens=64, max_tokens=10_000, max_entries=8)
    src_cache._insert(key, migrate_entry(src, _snapshot(src, len(key), row=1)))
    snap = src_cache.tail(key, keep=True)
    assert snap is not None and len(src_cache) == 1  # kept on the source
    bufs = migrate.staging(src, len(key))
    manifest = migrate.export_into(src, snap, bufs)
    assert dst_cache.put_tail(key, migrate.import_from(dst, manifest, bufs))
    reused, got, tail = dst_cache.lookup(key + (5, 6, 7))
    assert reused == len(key) and tail and got.length == len(key)
    assert src_cache.tail(key, keep=False) is snap and len(src_cache) == 0
    assert src_cache.tail(key, keep=False) is None


def test_put_tail_refuses_a_wrong_length() -> None:
    dst = _Rank(8, 1)
    cache = FastPrefixCache(dst, chunk_tokens=64, max_tokens=10_000, max_entries=8)
    snap = _snapshot(dst, 70, row=0)
    free = dst.pool.free
    assert not cache.put_tail(tuple(range(71)), snap)
    assert snap.released and dst.pool.free == free + 2


def migrate_entry(model: _Rank, snap: Snapshot):
    from engine.generator import _FastEntry

    return _FastEntry(snap, model.snapshot_bytes(snap), tail=True)
