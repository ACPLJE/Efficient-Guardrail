"""SPEC 6.1: hit@k[d], hit@k[union], per-site any/all, payload rank and the random baseline."""
import numpy as np
import pytest

from ag.data.types import ChunkingConfig, LedgerRecord
from ag.data import mock_lc_inject
from ag.metrics.quality import hit_rate, random_baseline
from ag.metrics.score import hits_for_record, selector_variants


def _rec(T, NB, hot_rows):
    """block mass: rows in hot_rows[block] point at that block, others are flat."""
    bm = np.full((T, 2, NB), 1.0 / NB)
    for block, rows in hot_rows.items():
        for t in rows:
            bm[t, :, :] = 0.0
            bm[t, :, block] = 1.0
    return LedgerRecord("r", "MT", "mock", 1, "h", "fp32", "include", NB * 128, 128, ("all",),
                        np.zeros(T, np.int32), np.zeros(T, np.int64), np.arange(2, dtype=np.int16), bm,
                        np.ones(NB, np.uint8))


VARIANTS = selector_variants({"selector": {"k": [1, 2], "layers": [[0, 1]], "first_token": "include"},
                              "chunking": {"first": 8, "next": 32}})
REQ = mock_lc_inject.generate(n_pairs=1, length_classes=["512"])[0]


def _get(rows, cd, k, site="single"):
    return next(r for r in rows if r["chunk_def"] == cd and r["k"] == k and r["site_rule"] == site)


def test_first8_vs_union():
    # payload window 7 (blocks 7, 8); attention reaches it only in tokens 40..47 (serving chunk 2)
    rec = _rec(48, 10, {8: range(40, 48), 2: range(0, 40)})
    rows = hits_for_record(rec, REQ, [7], [], ChunkingConfig(), VARIANTS)
    assert not _get(rows, "first8", 1)["hit"]
    assert _get(rows, "union", 1)["hit"]
    assert _get(rows, "union", 1)["payload_rank"] == 1
    assert _get(rows, "all", 2)["m"] == 2


def test_per_site_any_all():
    rec = _rec(48, 10, {8: range(40, 48), 2: range(0, 40)})
    rows = hits_for_record(rec, REQ, [1, 7], [[7], [4]], ChunkingConfig(), VARIANTS)
    assert _get(rows, "union", 1, "any")["hit"] and not _get(rows, "union", 1, "all")["hit"]


def test_hit_rate_and_random_baseline():
    rows = [{"request_id": f"r{i}", "variant_id": "v", "chunk_def": "union", "site_rule": "single", "k": 8,
             "hit": i < 3, "n_eligible": 255, "n_payload": 1, "m": 8, "source": "s"} for i in range(4)]
    out = hit_rate(rows, chunk_def="union", k=8, units={"r0", "r1", "r2", "r3"})
    assert (out["num"], out["den"], out["value"]) == (3, 4, 0.75)
    assert out["ci_low"] <= 0.75 <= out["ci_high"]
    assert random_baseline(rows, chunk_def="union", k=8, units={"r0"}) == pytest.approx(8 / 255)
    assert hit_rate(rows, chunk_def="union", k=8, units=set())["value"] is None
