"""plots/fig09.py on the committed mock fixture (no GPU, no harness run)."""
import csv
import shutil
import subprocess
import sys

from _ag_util import ROOT

FIX = ROOT / "tests" / "fixtures" / "results_mock"


def _run_ids():
    return sorted(d.name for d in (FIX / "raw" / "mock").iterdir() if d.is_dir())


def _csv(out, run_id):
    with open(out / "processed" / "fig09" / run_id / "hit_at_k.csv") as f:
        return [r for r in csv.DictReader(f)]


def test_fig09_on_fixture(tmp_path):
    out = tmp_path / "results"
    shutil.copytree(FIX, out)
    ids = _run_ids()
    assert len(ids) == 2
    args = [sys.executable, str(ROOT / "plots" / "fig09.py"), "--out", str(out), "--exp", "mock", "--allow-mock"]
    for rid in ids:
        args += ["--run-id", rid]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "figures" / "fig09.pdf").exists()
    curves = {}
    for rid in ids:
        rows = [x for x in _csv(out, rid) if x["chunk_def"] == "union"]
        curves[rid] = [float(x["hit"]) for x in sorted(rows, key=lambda x: int(x["k"]))]
    hi = max(curves.values(), key=lambda c: c[0])
    lo = min(curves.values(), key=lambda c: c[0])
    assert hi[0] >= 0.9                                   # concentrated ledger: hit@1 ~ 1
    assert lo[0] < hi[0] and all(a <= b for a, b in zip(lo, lo[1:]))   # noisy ledger: grows with k


def test_fig09_rejects_fixture_without_allow_mock(tmp_path):
    out = tmp_path / "results"
    shutil.copytree(FIX, out)
    r = subprocess.run([sys.executable, str(ROOT / "plots" / "fig09.py"), "--out", str(out), "--exp", "mock",
                        "--run-id", _run_ids()[0]], capture_output=True, text=True)
    assert r.returncode == 1 and "mock" in r.stderr


def test_test_manifests_classify_every_module():
    sys.path.insert(0, str(ROOT / "tools"))
    import check_test_classification as c
    assert c.problems() == []
