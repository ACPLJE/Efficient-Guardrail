"""agctl run --sbatch renders CLAUDE.md-compliant scripts and always runs sbatch --test-only."""
import os
import stat
import sys

import pytest

from _ag_util import ROOT

sys.path.insert(0, str(ROOT))
import agctl  # noqa: E402


@pytest.fixture()
def fake_sbatch(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sbatch.log"
    script = bindir / "sbatch"
    script.write_text(f"""#!/bin/sh
echo "$@" >> {log}
case "$1" in
  --test-only) echo "sbatch: Job 4242 to start at 2026-10-11T00:00:00 using 8 processors" >&2 ;;
  *) echo "Submitted batch job 4242" ;;
esac
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return log


def test_render_header():
    text = agctl.render_sbatch(name="x", config="c.yaml", ts="t", out="/o", n_jobs=3, offset=2)
    assert "#SBATCH --gres=gpu:1" in text and "#SBATCH --qos=normal" in text and "#SBATCH -p RTX6000ADA" in text
    assert "#SBATCH --array=0-2\n" in text and "%" not in text.split("--array=")[1].split("\n")[0]
    assert "slurm/ag_common.sh" in text and "--job-index $(( ${SLURM_ARRAY_TASK_ID:-0} + 2 ))" in text
    with pytest.raises(ValueError):
        agctl.render_sbatch(name="x", config="c", ts="t", out="/o", n_jobs=1, offset=0, gres="shard:1")


def test_sbatch_test_only_then_submit(tmp_path, fake_sbatch):
    out = tmp_path / "out"
    cfg = str(ROOT / "configs" / "exp" / "skeleton.yaml")
    assert agctl.main(["run", "--config", cfg, "--out", str(out), "--sbatch"]) == 0
    calls = fake_sbatch.read_text().splitlines()
    assert len(calls) == 2 and all(c.startswith("--test-only ") for c in calls)    # wave 0 (B0), wave 1 (AG)
    scripts = sorted((out / "raw" / "mock" / "jobs").glob("*.sbatch"))
    assert len(scripts) == 2 and "--array" not in scripts[0].read_text()
    assert (out / "logs").is_dir()                                   # Slurm needs the -o directory to exist
    assert f'export AG_OUT="{out.resolve()}"' in scripts[0].read_text()
    fake_sbatch.write_text("")
    assert agctl.main(["run", "--config", cfg, "--out", str(out), "--sbatch", "--submit"]) == 0
    calls = fake_sbatch.read_text().splitlines()
    assert [c.split()[0] for c in calls] == ["--test-only", "--test-only", calls[2].split()[0], "--dependency=afterok:4242"]


def test_clock_passed_to_array_tasks(tmp_path, fake_sbatch):
    import re
    from ag.harness.run import main as run_main
    out = tmp_path / "out"
    cfg = str(ROOT / "configs" / "exp" / "skeleton.yaml")
    assert agctl.main(["run", "--config", cfg, "--out", str(out), "--sbatch", "--clock", "monotonic"]) == 0
    script = sorted((out / "raw" / "mock" / "jobs").glob("*.sbatch"))[0].read_text()
    assert "--clock monotonic" in script
    ts = re.search(r"--ts (\S+)", script).group(1)
    # an array task that re-expands with a different clock would make other run_ids: refused
    assert run_main(["--config", cfg, "--out", str(out), "--ts", ts, "--clock", "virtual", "--job-index", "0"]) == 2


def test_doctor_runs():
    assert agctl.main(["doctor"]) == 0
