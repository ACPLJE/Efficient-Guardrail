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
    for args in (["--run-id", "latest"], ["--run-id", d.name]):
        r = subprocess.run([sys.executable, str(ROOT / "plots" / "fig09.py"), "--out", str(out), "--exp", "mock",
                            *args], capture_output=True, text=True)
        assert r.returncode == 1 and ("mock" in r.stderr or "latest" in r.stderr)
