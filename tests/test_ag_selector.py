"""SPEC 4.5: layer selection (D-031, D-037), aggregations, selection rules."""
import numpy as np
import pytest

from ag.data.types import ChunkInfo, LedgerRecord
from ag.selector.layers import resolve_layers
from ag.selector.select import (SelectorConfig, chunk_block_mass, payload_rank, random_hit_prob,
                                record_layer_index, select, union_select)


def test_layer_specs():
    assert resolve_layers("upper_half", 8) == (4, 5, 6, 7)
    assert resolve_layers("tracked8", 36) == (18, 20, 23, 25, 28, 30, 33, 35)
    assert resolve_layers("tracked8", 32) == (16, 18, 20, 22, 25, 27, 29, 31)
    assert resolve_layers("last", 32) == (31,)
    kinds = ["sliding" if i % 2 == 0 else "full" for i in range(24)]       # T4-like: 12 full layers
    assert resolve_layers("upper_half", 24, kinds) == (13, 15, 17, 19, 21, 23)
    assert resolve_layers("last", 24, kinds) == (23,)


def _record(T=10, L=4, NB=5, hot_block=3, hot_layer=3):
    bm = np.full((T, L, NB), 0.1)
    bm[:, hot_layer, hot_block] = 0.6
    bm[0] = 0.0
    bm[0, :, 0] = 1.0                                 # row 0 points elsewhere (first token)
    return LedgerRecord("r", "MT", "mock", 1, "h", "fp32", "include", NB * 128, 128, ("all",),
                        np.arange(T, dtype=np.int32), np.zeros(T, np.int64), np.arange(L, dtype=np.int16), bm,
                        np.ones(NB, np.uint8))


def test_chunk_block_mass_aggregates():
    rec = _record()
    c = ChunkInfo(0, 0, 8)
    m_c = chunk_block_mass(rec, c, SelectorConfig(layers="tracked8"), n_target_layers=4)   # upper half [2, 3]
    assert np.argmax(m_c) == 3
    m_last = chunk_block_mass(rec, c, SelectorConfig(layers="last"), n_target_layers=4)
    assert m_last[3] == pytest.approx(0.6)
    with_first = chunk_block_mass(rec, c, SelectorConfig(layers=(0,), first_token="include"))
    assert with_first[0] > chunk_block_mass(rec, c, SelectorConfig(layers=(0,)))[0]


def test_unimplemented_and_validation():
    rec = _record()
    with pytest.raises(NotImplementedError):
        chunk_block_mass(rec, ChunkInfo(0, 0, 8), SelectorConfig(head_weighting="learned"))
    with pytest.raises(ValueError):
        SelectorConfig(head_weighting="learned", first_token="include").validate()
    with pytest.raises(ValueError):
        chunk_block_mass(rec, ChunkInfo(0, 0, 8), SelectorConfig(layers=(9,)))


def test_named_layers_need_target_info_on_partial_records():
    import dataclasses
    rec = dataclasses.replace(_record(L=4), layers=np.array([4, 5, 6, 7], dtype=np.int16))   # tracked8 of 8
    c = ChunkInfo(0, 0, 8)
    with pytest.raises(ValueError, match="needs n_target_layers"):
        chunk_block_mass(rec, c, SelectorConfig(layers="tracked8"))
    lead = dataclasses.replace(_record(L=4), layers=np.array([0, 1, 2, 3], dtype=np.int16))   # leading 4 of 8
    with pytest.raises(ValueError, match="needs n_target_layers"):
        chunk_block_mass(lead, c, SelectorConfig(layers="tracked8"))
    with pytest.raises(ValueError, match="not in the Record"):
        chunk_block_mass(lead, c, SelectorConfig(layers="tracked8"), n_target_layers=8)
    kinds = ["sliding" if i % 2 == 0 else "full" for i in range(8)]
    full8 = dataclasses.replace(_record(L=8), layers=np.arange(8, dtype=np.int16))
    idx = record_layer_index(full8, SelectorConfig(layers="upper_half"), n_target_layers=8, attn_kind=kinds)
    assert idx.tolist() == [5, 7]                                      # sliding layers excluded (D-037)
    m = chunk_block_mass(rec, c, SelectorConfig(layers="tracked8"), n_target_layers=8)
    assert np.allclose(m, chunk_block_mass(rec, c, SelectorConfig(layers=(4, 5, 6, 7))))
    with pytest.raises(ValueError, match="not in the Record"):
        chunk_block_mass(rec, c, SelectorConfig(layers="all"), n_target_layers=8)


def test_select_rules():
    s = np.array([0.2, -np.inf, 0.5, 0.5, 0.1])
    assert select(s, SelectorConfig(k=2)) == (2, 3)                   # tie -> lower id first
    assert select(s, SelectorConfig(rule="all")) == (0, 2, 3, 4)
    assert select(s, SelectorConfig(rule="oracle"), oracle=[4]) == (4,)
    r1 = select(s, SelectorConfig(rule="random", k=2, seed=1), key="req")
    assert r1 == select(s, SelectorConfig(rule="random", k=2, seed=1), key="req") and 1 not in r1
    assert select(s, SelectorConfig(rule="mass_threshold", mass_tau=0.5)) == (2, 3)
    assert union_select([s, np.array([0.9, 0, 0, 0, 0.0])], SelectorConfig(k=1)) == (0, 2)
    assert payload_rank(s, [4, 0]) == 3 and payload_rank(s, [1]) is None


def test_mass_threshold_full_mass_is_minimal():
    s = np.array([0.86, 0.66, 0.41, 0.23, 0.11, 0.06, 0.05, 0.01, 0.0])
    assert len(select(s, SelectorConfig(rule="mass_threshold", mass_tau=1.0))) == 8    # the zero window is not added
    assert len(select(s, SelectorConfig(rule="mass_threshold", mass_tau=0.5))) == 2


def test_random_baseline_formula():
    assert random_hit_prob(255, 1, 8) == pytest.approx(8 / 255)        # build plan: N1 8/255 = 3.1 %
    assert random_hit_prob(10, 2, 9) == 1.0
    assert random_hit_prob(10, 0, 3) == 0.0
