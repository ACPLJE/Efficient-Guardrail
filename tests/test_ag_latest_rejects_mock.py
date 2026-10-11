"""SPEC 7.1 latest.txt eligibility; plots refuse mock / dirty runs by default."""
import subprocess
import sys

import pytest

from _ag_util import ROOT, run_dirs, small_exp
from ag.harness.run import main as run_main
from ag.harness.runner import latest_eligible


@pytest.mark.parametrize("lin,engine,ok,want", [
    ({"clock": "monotonic", "code_dirty": False}, "sglang-0.5.21", True, True),
    ({"clock": "virtual", "code_dirty": False}, "sglang-0.5.21", True, False),
    ({"clock": "monotonic", "code_dirty": True}, "sglang-0.5.21", True, False),
    ({"clock": "monotonic", "code_dirty": False}, "mock", True, False),
    ({"clock": "monotonic", "code_dirty": False}, "sglang-0.5.21", False, False),
])
def test_eligibility(lin, engine, ok, want):
    assert latest_eligible(lin, engine, ok) is want


def test_mock_monotonic_run_still_not_latest(tmp_path):
    exp = small_exp(tmp_path, systems=["mock_b0"], clock="monotonic", data={"length_classes": ["512"]},
                    fairness={"max_new_tokens": 4}, record=None)
    out = tmp_path / "out"
    assert run_main(["--config", str(exp), "--out", str(out), "--limit", "1"]) == 0
    assert not (out / "raw" / "mock" / "latest.txt").exists()
    d = run_dirs(out)["B0"]
    for args, msg in ((["--run-id", "latest"], "latest.txt"), (["--run-id", d.name], "mock / virtual-clock run")):
        r = subprocess.run([sys.executable, str(ROOT / "plots" / "fig09.py"), "--out", str(out), "--exp", "mock",
                            *args], capture_output=True, text=True)
        assert r.returncode == 1 and msg in r.stderr, r.stderr


def _lineage(tmp_path, name, **kw):
    import json
    d = tmp_path / name
    d.mkdir()
    lin = {"run_id": name, "clock": "monotonic", "producer": "sglang-shim", "engine_version": "sglang-0.5.21",
           "code_commit": "abc", "code_dirty": False, "data_manifest_sha": "d1", **kw}
    (d / "lineage.json").write_text(json.dumps(lin))
    (d / "checks.json").write_text(json.dumps([{"name": "x", "ok": True, "detail": ""}]))
    return d


def test_check_run_rejects_dirty_and_failed_checks(tmp_path):
    import json
    sys.path.insert(0, str(ROOT))
    from plots import _common as C
    assert C.check_run(_lineage(tmp_path, "ok"), allow_mock=False)["run_id"] == "ok"
    with pytest.raises(C.RunRejected, match="dirty"):
        C.check_run(_lineage(tmp_path, "dirty", code_dirty=True, code_commit="abc-dirty"), allow_mock=False)
    bad = _lineage(tmp_path, "bad")
    (bad / "checks.json").write_text(json.dumps([{"name": "x", "ok": False, "detail": ""}]))
    with pytest.raises(C.RunRejected, match="checks"):
        C.check_run(bad, allow_mock=False)


def test_check_consistent_code_and_data():
    sys.path.insert(0, str(ROOT))
    from plots import _common as C
    a = {"code_commit": "c1", "data_manifest_sha": "d1"}
    with pytest.raises(C.RunRejected, match="data_manifest_sha"):
        C.check_consistent([a, {**a, "data_manifest_sha": "d2"}], allow_mixed=False)
    C.check_consistent([a, {**a, "data_manifest_sha": "d2"}], allow_mixed=False, keys=("code_commit",))
    with pytest.raises(C.RunRejected, match="code_commit"):     # --allow-mixed-data still compares code
        C.check_consistent([a, {**a, "code_commit": "c2"}], allow_mixed=False, keys=("code_commit",))
