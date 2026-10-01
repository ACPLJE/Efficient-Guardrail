"""s3ctl: task lists, tiers and the b1-start plan (dry-run sbatch, no cluster)."""
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


def test_b1_start_plan_and_tiers(tmp_path):
    r = subprocess.run([sys.executable, "s3ctl.py", "b1-start"], cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    assert len(lines) == 13 + 2 + 12                                   # v2main (nemotron head+tail) + CD 2 + SafetyNIAH 12
    assert "nemotron_safety_guard_v3 v2main trunc_keep_head" in r.stdout and "gpt_oss_safeguard_20b safetyniah native" in r.stdout
    r = subprocess.run([sys.executable, "s3ctl.py", "longbench", "--guards", "qwen3guard_gen_8b,gpt_oss_safeguard_20b", "--bench", "v2main",
                        "--policy", "native", "--L", "1024,28672", "--sbatch", "--dry-run-sbatch"], cwd=ROOT, env=_env(tmp_path),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "tier 24 GB: 1 tasks" in r.stdout and "tier 48 GB: 1 tasks" in r.stdout      # gpt-oss needs the 48 GB tier
    assert "%" not in [l for l in r.stdout.splitlines() if "--array=" in l][0]
    # two tiers submitted within the same second keep SEPARATE task lists (9/20: a shared stamp let the 40 GB list overwrite the 24 GB one)
    lists = sorted((tmp_path / "out" / "slurm").glob("longbench_*.txt"))
    assert len(lists) == 2 and {f.name.rsplit("_", 1)[-1] for f in lists} == {"24gb.txt", "48gb.txt"}
    assert any("qwen3guard_gen_8b v2main native" in f.read_text() for f in lists if f.name.endswith("24gb.txt"))
    assert any("gpt_oss_safeguard_20b v2main native" in f.read_text() for f in lists if f.name.endswith("48gb.txt"))
    sb = [l for l in r.stdout.splitlines() if "sed -n" in l]
    assert sb and all("longbench_" in l and (".txt" in l) for l in sb)
    # longbench array tasks self-requeue at the time limit (9/25): SIGTERM 15 min before, flush, rc=3, scontrol requeue
    assert "#SBATCH --signal=B:TERM@900" in r.stdout and "#SBATCH --requeue" in r.stdout and "scontrol requeue" in r.stdout
    r = subprocess.run([sys.executable, "s3ctl.py", "longbench", "--guards", "llamaguard3_8b", "--bench", "v2main", "--policy", "native",
                        "--L", "65536,131072", "--max-pairs", "128", "--sbatch", "--dry-run-sbatch"], cwd=ROOT, env=_env(tmp_path),
                       capture_output=True, text=True)
    assert "tier 48 GB: 1 tasks" in r.stdout                                              # >= 64k tokens -> 48 GB
    files = sorted((tmp_path / "out" / "slurm").glob("longbench_*.txt"))
    assert any("--max-pairs_128" in f.read_text() for f in files)


def test_status_and_help(tmp_path):
    for c in ["synth", "variants", "tokspans", "gates", "datacard", "pilot", "longbench", "b1-start", "status", "all"]:
        r = subprocess.run([sys.executable, "s3ctl.py", c, "--help"], cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True)
        assert r.returncode == 0, (c, r.stderr)
    r = subprocess.run([sys.executable, "s3ctl.py", "status"], cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0 and "D6 synthesis" in r.stdout and "E0-4 gates" in r.stdout


def test_finish_and_cpu_partitions(tmp_path):
    """`finish` = gates + datacard + pilot in one job; CPU jobs go to LG_CPU_PARTITIONS without --gres when it is set,
    and every script prints `[s1] done rc=N` even when the body fails (set -e used to skip the line)."""
    env = _env(tmp_path)
    r = subprocess.run([sys.executable, "s3ctl.py", "finish", "--sbatch", "--dry-run-sbatch", "--dep", "12345"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "lgs3.gates" in r.stdout and "lgs3.datacard" in r.stdout and "lgs3.pilot" in r.stdout
    assert "--dependency=afterok:12345" in r.stdout and "--gres=gpu:1" in r.stdout and "--partition=A6000,L40S" in r.stdout
    env2 = dict(env, LG_CPU_PARTITIONS="cpu-max24,cpu-max16", LG_CPU_QOS="")
    r = subprocess.run([sys.executable, "s3ctl.py", "synth", "--splits", "calib,train", "--sbatch", "--dry-run-sbatch"], cwd=ROOT, env=env2,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--partition=cpu-max24,cpu-max16" in r.stdout and "--gres" not in r.stdout and "--qos" not in r.stdout
    assert "--split calib --set calib" in r.stdout and "--split train --set train,train16k --native" in r.stdout
    r = subprocess.run([sys.executable, "s3ctl.py", "variants", "--sbatch", "--dry-run-sbatch"], cwd=ROOT, env=env2, capture_output=True, text=True)
    assert "--gres=gpu:1" in r.stdout and "--partition=A6000,L40S" in r.stdout            # GPU job unchanged
    script = sorted((tmp_path / "out" / "slurm").glob("s3_synth_train_*.sbatch"))[-1].read_text()
    assert 'echo "[s1] done rc=$rc"' in script and "exit $rc" in script
    fake = tmp_path / "fake.sh"
    fake.write_text(script.replace("source " + str(ROOT) + "/slurm/common.sh", "PY=false").replace("${PY} -m lgs2.synth", "false #"))
    r = subprocess.run(["bash", str(fake)], capture_output=True, text=True)
    assert r.returncode == 1 and "[s1] done rc=1" in r.stdout and "requeue" not in r.stdout        # synth: no requeue
