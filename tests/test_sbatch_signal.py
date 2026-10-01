"""The requeue-able sbatch script forwards Slurm's end-of-time signal (--signal=B:TERM@N -> batch shell) to the trainer,
which checkpoints and exits 3; the script then asks for a requeue (2026-09-20).  Simulated with a fake trainer."""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(tmp_path):
    e = dict(os.environ)
    e.update({"LG_ENV_FILE_NO_OVERRIDE": "1", "LG_DATA_V2": str(tmp_path / "v2"), "LG_SOURCES": str(tmp_path / "v2" / "sources"),
              "LG_OUT": str(tmp_path / "out"), "HF_HOME": str(tmp_path / "hf"), "LG_PARTITIONS_OVERRIDE": "A6000"})
    return e


def test_forwarded_sigterm_reaches_the_trainer_and_requeues(tmp_path, monkeypatch):
    import s1ctl
    sys.argv = ["s1ctl.py"]
    for k, v in _env(tmp_path).items():
        monkeypatch.setenv(k, v)
    marker = tmp_path / "ckpt.txt"
    fake = tmp_path / "trainer.py"
    fake.write_text(textwrap.dedent(f"""
        import signal, sys, time
        stop = False
        def h(*_):
            global stop
            stop = True
        signal.signal(signal.SIGTERM, h)
        for _ in range(600):
            time.sleep(0.05)
            if stop:
                open({str(marker)!r}, "w").write("saved")
                sys.exit(3)
        sys.exit(0)
    """))
    from lgs1.paths import S1Paths
    S1Paths.load().ensure_out()
    body = f"{sys.executable} {fake}"
    jid = s1ctl.write_sbatch("t_sig", body, "00:10:00", dry_run=True, min_gpu_gb=8, signal="TERM@1500", requeue=True)
    assert jid is None
    script = sorted((tmp_path / "out" / "slurm").glob("t_sig_*.sbatch"))[-1].read_text()
    assert "#SBATCH --signal=B:TERM@1500" in script and "#SBATCH --requeue" in script and "trap '' TERM USR1" in script
    runnable = script.replace(f"source {ROOT}/slurm/common.sh", "true")
    sh = tmp_path / "run.sh"
    sh.write_text(runnable)
    (tmp_path / "scontrol").write_text("#!/bin/bash\necho \"[fake scontrol] $@\"\n")     # a fake scontrol on PATH
    os.chmod(tmp_path / "scontrol", 0o755)
    proc = subprocess.Popen(["bash", str(sh)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, SLURM_JOB_ID="12345", LG_REQUEUE_SLEEP="1",
                                     PATH=str(tmp_path) + os.pathsep + os.environ["PATH"]))
    time.sleep(1.5)
    proc.send_signal(signal.SIGTERM)                       # what Slurm does to the batch shell at the time limit (B:)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 3, out
    assert marker.read_text() == "saved"
    assert "[s1] done rc=3" in out and "scontrol requeue 12345" in out and "[fake scontrol] requeue 12345" in out, out
    # plain jobs: no forwarding block, subshell semantics unchanged
    s1ctl.write_sbatch("t_plain", "true", "00:10:00", dry_run=True, min_gpu_gb=8)
    plain = sorted((tmp_path / "out" / "slurm").glob("t_plain_*.sbatch"))[-1].read_text()
    assert "trap" not in plain and "--signal" not in plain and "--requeue" not in plain
