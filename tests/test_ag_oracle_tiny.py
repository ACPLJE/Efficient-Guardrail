"""Reference ledger on the tiny Qwen3 (SPEC 4.5, 5.2): row sums, prefill row, teacher forcing,
Record round trip. CPU; skipped when torch / transformers are missing."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ag.data import mock_lc_inject  # noqa: E402
from ag.reference.ledger_oracle import decode_with_ledger  # noqa: E402
from ag.reference.record import RecordStore  # noqa: E402
from ag.targets.base import RecordSpec  # noqa: E402
from ag.targets.mock import MockTarget  # noqa: E402


@pytest.fixture(scope="module")
def model():
    from lgs1.lgcore.model import tiny_qwen3
    return tiny_qwen3(seed=0, attn="eager")


def test_rows_sum_to_one_and_prefill_row(model):
    ids = list(range(5, 5 + 300))
    res = decode_with_ledger(model, ids, 6, split_tokens=128, spec=RecordSpec(layers="all"))
    assert res.head_block_mass.shape == (6, 4, 4, 3)
    tot = res.block_mass.sum(-1) + res.gen_mass.sum(-1)
    np.testing.assert_allclose(tot, 1.0, atol=1e-6)
    assert np.all(np.triu(res.gen_mass.transpose(1, 0, 2)[0], k=0) == 0)          # gen_mass is strictly lower
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_attentions=True)
    last = out.attentions[2][0, :, -1, :].double().numpy().mean(0)
    np.testing.assert_allclose(res.block_mass[0, 2], [last[:128].sum(), last[128:256].sum(), last[256:].sum()],
                               atol=1e-9)


def test_teacher_forcing_reproduces(model):
    ids = list(range(10, 10 + 200))
    a = decode_with_ledger(model, ids, 5, spec=RecordSpec())
    b = decode_with_ledger(model, ids, 0, spec=RecordSpec(), forced_output=a.output_ids)
    assert b.output_ids == a.output_ids
    np.testing.assert_allclose(a.block_mass, b.block_mass, atol=1e-12)


def test_tiny_target_record_roundtrip(tmp_path):
    t = MockTarget("tiny", seed=0)
    req = mock_lc_inject.generate(n_pairs=1, length_classes=["512"])[0]
    rp = t.render(req)
    out = t.generate(rp, 12, record=RecordSpec(precision="fp32"))
    rec = out.record
    assert rec.producer == "reference" and rec.block_mass.shape == (12, 4, rp.n_blocks)
    assert rec.chunk_head_mass.shape == (3, 4, 4, rp.n_blocks)
    st = RecordStore(tmp_path)
    st.put(rec)
    back = st.get(rec.request_id, "MT", producer="reference")
    np.testing.assert_array_equal(back.block_mass, rec.block_mass)
