"""python -m ag.harness.run --config configs/exp/skeleton.yaml [--out DIR] [--clock virtual]

Expands the experiment into jobs, writes raw/<E>/jobs/<ts>.jsonl and runs every job (or one
--job-index, the Slurm array index). Exit code 2 on a lineage mismatch (SPEC 7.1).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from ag.config.schema import load_exp
from ag.harness.lineage import LineageMismatch, fail_on_mismatch
from ag.harness.runner import run_job
from ag.sweep.expand import expand, write_jobs


def default_out() -> Path:
    return Path(os.environ.get("AG_OUT", "results"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ag.harness.run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="results root ($AG_OUT, default ./results)")
    ap.add_argument("--clock", choices=["monotonic", "virtual"], default=None)
    ap.add_argument("--job-index", type=int, default=None, help="run one job (Slurm array index)")
    ap.add_argument("--limit", type=int, default=None, help="first N requests per job (smoke runs)")
    ap.add_argument("--ts", default=None, help="expand timestamp (reuse to resume the same run_ids)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(args.out) if args.out else default_out()
    exp = load_exp(args.config)
    ts = args.ts or time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    jobs = expand(exp, ts=ts, clock=args.clock)
    path = write_jobs(out, jobs, ts)
    print(f"expanded {len(jobs)} jobs -> {path}")
    if args.job_index is not None:
        jobs = [jobs[args.job_index]]
    cmd = " ".join([sys.executable, "-m", "ag.harness.run"] + (argv if argv is not None else sys.argv[1:]))
    try:
        for job in jobs:
            run_dir = run_job(job, out, cmd=cmd, limit=args.limit)
            print(f"run {job.run_id} -> {run_dir}")
    except LineageMismatch as e:
        fail_on_mismatch(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
