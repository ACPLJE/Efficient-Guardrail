"""E0-3: (split, subset, id) uniqueness, strict matched-FPR with achieved FPR, domain hard-fail."""
import numpy as np
import pytest

from lgs1 import metrics as M


# ---- 1. uniqueness ------------------------------------------------------------------------
def test_unique_ids_pass_and_fail():
    rows = [{"split": "test", "subset": "main", "id": i} for i in range(10)]
    assert M.assert_unique_ids(rows) == 10
    rows.append({"split": "test", "subset": "main", "id": 3})
    with pytest.raises(M.DuplicateIdError):
        M.assert_unique_ids(rows)
    # same id in another subset is fine
    rows[-1]["subset"] = "decoy"
    assert M.assert_unique_ids(rows) == 11
    with pytest.raises(M.DuplicateIdError):
        M.assert_unique_ids([{"split": "test", "id": 1}])  # missing subset


# ---- 2. strict matched-FPR ---------------------------------------------------------------
def test_strict_threshold_never_exceeds_target_with_ties():
    benign = [0.9] * 10 + [0.1] * 90          # 10% of benign tied at the top
    th = M.strict_matched_fpr(benign, 0.05)
    assert th.tau == 0.9                      # the tie value itself; strict '>' excludes it
    assert th.achieved_fpr == 0.0             # a '>=' quantile rule would have given 10% > 5%
    assert th.achieved_fpr <= th.target_fpr


def test_strict_threshold_achieves_target_without_ties():
    benign = np.linspace(0, 1, 200)
    th = M.strict_matched_fpr(benign, 0.05)
    assert abs(th.achieved_fpr - 0.05) < 1e-9
    assert th.n_benign == 200


@pytest.mark.parametrize("seed", range(20))
def test_strict_threshold_property_random_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    benign = rng.integers(0, 6, size=n) / 5.0   # heavy ties
    for t in (0.0, 0.01, 0.05, 0.2, 0.5, 1.0):
        th = M.strict_matched_fpr(benign, t)
        assert th.achieved_fpr <= t + 1e-12
        # maximality: lowering tau to the next distinct value below would break the target
        lower = benign[benign < th.tau] if np.isfinite(th.tau) else np.array([])
        if lower.size:
            nxt = lower.max()
            assert np.mean(benign > nxt) > t or np.isclose(np.mean(benign > nxt), t)


def test_recall_and_fpr_use_strict_rule():
    scores = [0.9, 0.9, 0.5, 0.1]
    labels = [1, 0, 1, 0]
    assert M.recall_at(scores, labels, 0.9) == 0.0      # score == tau is NOT positive
    assert M.recall_at(scores, labels, 0.4) == 1.0
    assert M.fpr_at(scores, labels, 0.5) == 0.5


def test_summary_reports_achieved_fpr_columns():
    calib = [{"method": "LG", "L_target": 1024, "score": s, "label": 0} for s in np.linspace(0, 1, 100)]
    test = ([{"method": "LG", "L_target": 1024, "score": 0.99, "label": 1} for _ in range(10)] +
            [{"method": "LG", "L_target": 1024, "score": 0.2, "label": 0} for _ in range(10)])
    out = M.summarize_matched(test, calib)
    assert len(out) == 1
    r = out[0]
    for col in ("tau@5pct", "recall@5pct", "calib_fpr@5pct", "test_fpr@5pct", "tau@1pct", "auprc"):
        assert col in r
    assert r["calib_fpr@5pct"] <= 0.05 and r["recall@5pct"] == 1.0


# ---- 3. domain hard-fail -----------------------------------------------------------------
def test_missing_domain_hard_fails():
    req = ["govreport", "pg19", "wildchat", "stack", "arxiv"]
    rows = [{"split": "test", "L_target": 4096, "domain": d} for d in req]
    rows += [{"split": "test", "L_target": 8192, "domain": d} for d in req if d != "stack"]
    with pytest.raises(M.MissingDomainError) as ei:
        M.check_domains(rows, req)
    assert "stack" in str(ei.value) and "8192" in str(ei.value)
    rows.append({"split": "test", "L_target": 8192, "domain": "stack"})
    groups = M.check_domains(rows, req)
    assert len(groups) == 2


def test_f1_and_auprc_basics():
    m = M.f1_binary([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and abs(m["f1"] - 0.5) < 1e-9
    assert abs(M.auprc([0.9, 0.8, 0.1], [1, 1, 0]) - 1.0) < 1e-9
