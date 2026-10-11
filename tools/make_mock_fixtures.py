#!/usr/bin/env python3
"""Regenerate tests/fixtures/results_mock/: two mock B0 runs (concentrated ledger, noisy ledger).

Only what the figure scripts read is kept (lineage, summary, checks, resolved, hits, outcomes);
row files are renamed *.jsonl -> *.jsonl.txt so the *.jsonl ignore rule (CLAUDE.md 3.2) does not
hide them.  python tools/make_mock_fixtures.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ag.harness.run import main as run_main  # noqa: E402

DEST = REPO / "tests" / "fixtures" / "results_mock"
KEEP = ("lineage.json", "summary.json", "checks.json", "resolved.yaml", "hits.jsonl", "outcomes.jsonl")
TS = "20261011T000000"


def main() -> int:
    exp = yaml.safe_load((REPO / "configs" / "exp" / "skeleton.yaml").read_text())
    exp["data"].update(length_classes=["8k"], n_attack=6, n_benign=2)
    exp["selector"]["layers"] = ["tracked8"]
    exp["notes"] = "fixture for tests/test_ag_plots_on_fixture.py"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sysdir = td / "systems"
        sysdir.mkdir()
        shutil.copy(REPO / "configs" / "systems" / "mock_b0.yaml", sysdir / "mock_b0.yaml")
        (sysdir / "mock_b0_noise.yaml").write_text(yaml.safe_dump({
            "base": "mock_b0.yaml", "system_id": "B0",
            "override": {"mock": {"target": {"concentration": 0.03}}}}))
        out = td / "out"
        for name in ("mock_b0.yaml", "mock_b0_noise.yaml"):
            e = dict(exp, systems=[str(sysdir / name)])
            p = td / f"exp_{name}"
            p.write_text(yaml.safe_dump(e))
            assert run_main(["--config", str(p), "--out", str(out), "--ts", TS]) == 0
        if DEST.exists():
            shutil.rmtree(DEST)
        for d in sorted((out / "raw" / "mock").iterdir()):
            if d.name == "jobs":
                continue
            dst = DEST / "raw" / "mock" / d.name
            dst.mkdir(parents=True)
            for f in KEEP:
                src = d / f
                shutil.copy(src, dst / (f + ".txt" if f.endswith(".jsonl") else f))
        shutil.copy(out / "raw" / "index.jsonl", DEST / "raw" / "index.jsonl.txt")
    print(DEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
