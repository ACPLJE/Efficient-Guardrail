"""s2ctl: sbatch scripts are generated per tier without an array throttle; status runs on an empty root."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(tmp_path):
    e = dict(os.environ)
    e.update({"LG_ENV_FILE_NO_OVERRIDE": "1", "LG_DATA_V2": str(tmp_path / "v2"), "LG_SOURCES": str(tmp_path / "v2" / "sources"),
              "LG_OUT": str(tmp_path / "out"), "HF_HOME": str(tmp_path / "hf"), "LG_PARTITIONS_OVERRIDE": "A6000,L40S"})
    return e


def test_screen_array_script_has_no_throttle_and_task_file(tmp_path):
    r = subprocess.run([sys.executable, "s2ctl.py", "screen", "--sbatch", "--dry-run-sbatch", "--domains", "govreport,stack"],
                       cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "#SBATCH --array=0-3\n" in r.stdout and "%" not in r.stdout.split("--array=")[1].split("\n")[0]
    assert "--partition=A6000,L40S" in r.stdout and "--gres=gpu:1" in r.stdout
    files = list((tmp_path / "out" / "slurm").glob("screen_tasks_*.txt"))
    assert files and files[0].read_text().splitlines() == ["govreport llamaguard3_8b", "govreport qwen3guard_gen_8b",
                                                            "stack llamaguard3_8b", "stack qwen3guard_gen_8b"]


def test_status_on_empty_root(tmp_path):
    r = subprocess.run([sys.executable, "s2ctl.py", "status"], cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0 and "D4a haystack" in r.stdout and "PENDING" in r.stdout
