"""SPEC 7.2: AG lineage fields, resume keys, mismatch -> exit 2 (build plan: patch_sha-only resume)."""
import json

import pytest

from _ag_util import run_dirs, small_exp
from ag.harness.lineage import (RESUME_KEYS, LineageMismatch, build_lineage, close_lineage, env_hash,
                                open_lineage)
from ag.harness.run import main as run_main


def _lin(**kw):
    base = dict(clock="virtual", seed=0, producer="mock", model_sha={"target": "m"}, data_manifest_sha="d")
    base.update(kw)
    return build_lineage("run1", "mock", "cfg", **base)


def test_fields_present(tmp_path):
    lin = _lin()
    d = lin.to_json()
    for k in ("code_commit", "code_dirty", "config_hash", "env_hash", "gpu_name", "gpu_count", "hostname",
              "slurm_job_id", "started_unix_ns", "finished_unix_ns", "model_sha", "data_manifest_sha", "clock",
              "parent_run_ids", "schema_version"):
        assert k in d
    assert set(env_hash()) == {"pip_freeze_sha", "env_vars_sha"}
    assert lin.code_dirty == lin.code_commit.endswith("-dirty")
    open_lineage(tmp_path, lin)
    done = close_lineage(tmp_path, lin)
    assert json.loads((tmp_path / "lineage.json").read_text())["finished_unix_ns"] == done.finished_unix_ns


def test_resume_same_and_mismatch(tmp_path):
    open_lineage(tmp_path, _lin())
    _, resumed = open_lineage(tmp_path, _lin())
    assert resumed
    with pytest.raises(LineageMismatch) as ei:
        open_lineage(tmp_path, _lin(patch_sha="abc"))
    assert set(ei.value.diffs) == {"patch_sha"}
    assert "hostname" not in RESUME_KEYS


def _ok(d):
    return all(c["ok"] for c in json.loads((d / "checks.json").read_text()))


def test_resume_after_partial_write(tmp_path):
    # a kill mid-write leaves half a line in events.jsonl; resume must cut it and finish cleanly
    exp = small_exp(tmp_path, systems=["mock_b0"])
    out = tmp_path / "out"
    args = ["--config", str(exp), "--out", str(out), "--ts", "20261011T000001"]
    assert run_main(args) == 0
    d = run_dirs(out)["B0"]
    lines = (d / "events.jsonl").read_text().splitlines(keepends=True)
    (d / "events.jsonl").write_text("".join(lines[:40]) + lines[40][: len(lines[40]) // 2])
    assert run_main(args) == 0
    assert _ok(d)
    rows = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    assert max(r["attempt"] for r in rows) == 1


def test_resume_reruns_request_without_outcome_row(tmp_path):
    exp = small_exp(tmp_path, systems=["mock_b0"])
    out = tmp_path / "out"
    args = ["--config", str(exp), "--out", str(out), "--ts", "20261011T000002"]
    assert run_main(args) == 0
    d = run_dirs(out)["B0"]
    rows = (d / "outcomes_run.jsonl").read_text().splitlines(keepends=True)
    (d / "outcomes_run.jsonl").write_text("".join(rows[:-1]))        # killed between FINISHED and the row
    assert run_main(args) == 0
    assert _ok(d)
    outcomes = [json.loads(l) for l in (d / "outcomes.jsonl").read_text().splitlines()]
    assert len(outcomes) == len(rows)


def test_cli_resume_patch_sha_only_exits_2(tmp_path):
    exp = small_exp(tmp_path, systems=["mock_b0"])
    out = tmp_path / "out"
    args = ["--config", str(exp), "--out", str(out), "--ts", "20261011T000000"]
    assert run_main(args) == 0
    d = run_dirs(out)["B0"]
    n_events = len((d / "events.jsonl").read_text().splitlines())
    assert run_main(args) == 0                                   # same lineage: every request is skipped
    assert len((d / "events.jsonl").read_text().splitlines()) == n_events
    lin = json.loads((d / "lineage.json").read_text())
    lin["patch_sha"] = "deadbeef"
    (d / "lineage.json").write_text(json.dumps(lin))
    with pytest.raises(SystemExit) as ei:
        run_main(args)
    assert ei.value.code == 2
