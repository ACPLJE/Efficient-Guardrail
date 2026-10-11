"""SPEC 5: Record v1 npz + index.jsonl round trip, key, filters, schema check."""
import numpy as np
import pytest

from ag.data import mock_lc_inject
from ag.reference.record import RecordStore, load_record, record_key, record_to_bytes
from ag.targets.base import RecordSpec
from ag.targets.mock import MockTarget


def _rec(spec=RecordSpec(), seed=0, idx=0):
    t = MockTarget(seed=seed)
    req = mock_lc_inject.generate(n_pairs=1, length_classes=["1k"], seed=idx)[0]
    rp = t.render(req)
    return t.generate(rp, 40, record=spec).record


def test_roundtrip_all_arrays(tmp_path):
    rec = _rec(RecordSpec(per_token_heads=False))
    st = RecordStore(tmp_path / "records", run_id="r1")
    name = st.put(rec)
    assert name == record_key(rec.request_id, rec.target_id, rec.record_spec_hash, "mock") + ".npz"
    back = st.get(rec.request_id, rec.target_id)
    for k in ("output_ids", "layers", "block_mass", "block_untrusted", "gen_mass", "chunk_head_mass"):
        np.testing.assert_array_equal(getattr(back, k), getattr(rec, k))
    assert back.token_head_mass is None and back.sink_mass is None
    assert back.chunk_defs == ("first8", "first32", "all") and back.first_token_policy == "include"
    assert back.block_mass.dtype == np.float16
    row = st.index()[0]
    assert row["run_id"] == "r1" and row["T"] == 40 and row["n_blocks"] == rec.n_blocks


def test_mass_rows_sum_to_one_for_every_request():
    t = MockTarget(seed=0)
    for req in mock_lc_inject.generate(n_pairs=30, length_classes=["1k"], seed=0):     # benign incl. last block
        rec = t.generate(t.render(req), 24, record=RecordSpec(precision="fp32")).record
        tot = rec.block_mass.astype(np.float64).sum(-1) + rec.gen_mass.astype(np.float64).sum(-1)
        np.testing.assert_allclose(tot, 1.0, atol=1e-3, err_msg=req.request_id)   # gen_mass is fp16


def test_split_tokens_must_match_and_forced_text():
    t = MockTarget(seed=0)
    rp = t.render(mock_lc_inject.generate(n_pairs=1, length_classes=["1k"])[0])
    with pytest.raises(ValueError, match="split_tokens"):
        t.generate(rp, 8, record=RecordSpec(split_tokens=64))
    t64 = MockTarget(seed=0, split_tokens=64)
    rp64 = t64.render(mock_lc_inject.generate(n_pairs=1, length_classes=["1k"])[0])
    assert t64.generate(rp64, 8, record=RecordSpec(split_tokens=64)).record.n_blocks == 2 * rp.n_blocks
    free = t.generate(rp, 12, record=None)
    forced = t.generate(rp, 12, record=None, forced_output=free.output_ids[:5])
    assert forced.output_ids == free.output_ids[:5] and forced.text.split() == free.text.split()[:5]


def test_filters_and_ambiguity(tmp_path):
    st = RecordStore(tmp_path)
    a = _rec(RecordSpec(precision="fp16"))
    b = _rec(RecordSpec(precision="fp32"))
    st.put(a)
    st.put(b)
    with pytest.raises(KeyError):
        st.get(a.request_id, a.target_id)
    got = st.get(a.request_id, a.target_id, record_spec_hash=b.record_spec_hash)
    assert got.block_mass.dtype == np.float32
    assert len(list(st.iter(precision="fp16"))) == 1
    st.put(a)                                       # re-put is idempotent on the file
    assert len(list(st.iter())) == 2


def test_schema_version_checked(tmp_path):
    import dataclasses
    rec = dataclasses.replace(_rec(), schema_version=99)
    p = tmp_path / "x.npz"
    p.write_bytes(record_to_bytes(rec))
    with pytest.raises(ValueError, match="schema_version"):
        load_record(p)
