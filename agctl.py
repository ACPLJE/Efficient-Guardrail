#!/usr/bin/env python3
"""agctl: minimal control CLI for AG experiments (build plan 4.1, item 11).

  python agctl.py run --config configs/exp/skeleton.yaml [--out DIR]       # run locally
  python agctl.py run --config ... --sbatch [--submit]                     # render sbatch, sbatch --test-only
  python agctl.py doctor                                                   # environment report

--sbatch always runs `sbatch --test-only` on every rendered script; --submit then submits them.
Defended runs depend on their B0 run (S comes from it), so jobs are split into two waves and the
second wave is submitted with --dependency=afterok:<wave-0 job id>. No %N throttle (CLAUDE.md 4).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SBATCH_HEADER = """#!/bin/bash
#SBATCH -J {name}
#SBATCH -p {partition}
#SBATCH --qos=normal
#SBATCH --gres={gres}
#SBATCH -c {cpus}
#SBATCH -t {time}
#SBATCH -o {log_dir}/%x.%A_%a.out
{array_line}export AG_OUT="{out}"
source "{repo}/slurm/ag_common.sh"
"""


def render_sbatch(*, name: str, config: str, ts: str, out: str, n_jobs: int, offset: int, gres: str = "gpu:1",
                  partition: str = "RTX6000ADA", cpus: int = 8, time_limit: str = "24:00:00",
                  log_dir: str | None = None, clock: str | None = None) -> str:
    """One sbatch script for jobs [offset, offset + n_jobs) of an expand. The log directory must exist
    before submission (Slurm does not create the -o directory); cmd_run creates it."""
    if not re.fullmatch(r"gpu:\d+", gres):
        raise ValueError(f"--gres must be gpu:N (QOSMinGRES), got {gres!r}")
    array_line = f"#SBATCH --array=0-{n_jobs - 1}\n" if n_jobs > 1 else ""
    idx = "${SLURM_ARRAY_TASK_ID:-0}"
    body = SBATCH_HEADER.format(name=name, partition=partition, gres=gres, cpus=cpus, time=time_limit,
                                log_dir=log_dir or f"{out}/logs", array_line=array_line, repo=REPO, out=out)
    clock_arg = f"--clock {clock} " if clock else ""
    body += (f'"$PY" -m ag.harness.run --config "{config}" --out "{out}" --ts {ts} {clock_arg}'
             f'--job-index $(( {idx} + {offset} ))\n')
    if "%" in array_line:
        raise AssertionError("array throttle (%N) is forbidden")
    return body


def _sbatch(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", *args], capture_output=True, text=True)


def cmd_run(a: argparse.Namespace) -> int:
    from ag.config.schema import load_exp
    from ag.harness.run import main as run_main
    from ag.sweep.expand import expand, fresh_ts, write_jobs

    out = a.out or os.environ.get("AG_OUT", "results")
    if not a.sbatch:
        argv = ["--config", a.config, "--out", out] + (["--clock", a.clock] if a.clock else [])
        return run_main(argv)

    exp = load_exp(a.config)
    ts = fresh_ts(Path(out) / "raw" / exp["exp_id"] / "jobs")
    jobs = expand(exp, ts=ts, clock=a.clock)
    write_jobs(out, jobs, ts)
    slurm = exp.get("slurm") or {}
    waves = [[j for j in jobs if not j.parent_run_ids], [j for j in jobs if j.parent_run_ids]]
    script_dir = Path(out) / "raw" / exp["exp_id"] / "jobs"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(out).resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)       # Slurm cannot open -o in a missing directory
    scripts: list[Path] = []
    offset = 0
    for wi, wave in enumerate(waves):
        if not wave:
            continue
        text = render_sbatch(name=f"ag-{exp['exp_id']}-w{wi}", config=str(Path(a.config).resolve()), ts=ts,
                             out=str(Path(out).resolve()), n_jobs=len(wave), offset=offset,
                             gres=slurm.get("gres", "gpu:1"), partition=slurm.get("partition", "RTX6000ADA"),
                             time_limit=slurm.get("time", "24:00:00"), log_dir=str(log_dir), clock=a.clock)
        p = script_dir / f"{ts}.w{wi}.sbatch"
        p.write_text(text)
        scripts.append(p)
        offset += len(wave)
    if shutil.which("sbatch") is None:
        print(f"sbatch not found; rendered {[str(s) for s in scripts]} (sbatch --test-only skipped)")
        return 0
    ok = True
    for s in scripts:
        r = _sbatch(["--test-only", str(s)])
        print(f"[test-only] {s.name}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
        ok &= r.returncode == 0
    if not ok:
        return 1
    if a.submit:
        dep = None
        for s in scripts:
            r = _sbatch(([f"--dependency=afterok:{dep}"] if dep else []) + [str(s)])
            print(r.stdout.strip() or r.stderr.strip())
            if r.returncode:
                return 1
            m = re.search(r"Submitted batch job (\d+)", r.stdout)
            dep = m.group(1) if m else dep
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    import importlib.metadata as md
    import platform

    rc = 0
    py = platform.python_version()
    print(f"python      {py} ({sys.executable})")
    if sys.version_info < (3, 11):
        print("  ERROR: AG needs Python >= 3.11 (D-001)")
        rc = 1
    try:
        import ag
        from ag.lineage_util import code_commit
        print(f"ag          {ag.__version__} at {Path(ag.__file__).parent}")
        here = code_commit(REPO)
        print(f"code        {here}")
        if Path(ag.__file__).resolve().parent.parent != REPO:
            print("  WARN: the imported ag package is not this checkout (re-run pip install -e .)")
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"ag          import failed: {e!r}")
        rc = 1
    for k in ("AG_OUT", "HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "AG_VENV"):
        print(f"{k:<11} {os.environ.get(k)}")
    for pkg in ("numpy", "pyyaml", "matplotlib", "torch", "transformers", "sglang", "flashinfer-python"):
        try:
            v = md.version(pkg)
        except md.PackageNotFoundError:
            v = "-"
        print(f"{pkg:<11} {v}")
    print(f"sbatch      {shutil.which('sbatch') or '-'}")
    print(f"nvidia-smi  {shutil.which('nvidia-smi') or '-'}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agctl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", default=None)
    r.add_argument("--clock", choices=["monotonic", "virtual"], default=None)
    r.add_argument("--sbatch", action="store_true", help="render sbatch scripts and run sbatch --test-only")
    r.add_argument("--submit", action="store_true", help="with --sbatch: submit after --test-only passes")
    r.set_defaults(fn=cmd_run)
    d = sub.add_parser("doctor")
    d.set_defaults(fn=cmd_doctor)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
