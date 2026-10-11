"""Shared plotting helpers (CLAUDE.md principle 7, SPEC 7.1).

Plots read only the results tree ($AG_OUT/raw, $AG_OUT/processed). Run selection:
  --run-id latest   raw/<E>/latest.txt (only eligible runs are ever written there, SPEC 7.1)
  --run-id <id>     an explicit run; mock / virtual-clock / dirty runs are rejected unless --allow-mock
Runs combined into one figure must share code_commit and data_manifest_sha unless --allow-mixed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ag.harness.writer import atomic_write_json, read_json, read_rows  # noqa: E402


class RunRejected(RuntimeError):
    pass


def add_common_args(ap: argparse.ArgumentParser, *, default_exp: str) -> None:
    ap.add_argument("--out", default=os.environ.get("AG_OUT", "results"), help="results root ($AG_OUT)")
    ap.add_argument("--exp", default=default_exp, help="experiment id used with --run-id latest")
    ap.add_argument("--run-id", action="append", default=None, help="run id or 'latest' (repeatable)")
    ap.add_argument("--allow-mock", action="store_true", help="accept mock / virtual-clock / dirty runs")
    ap.add_argument("--allow-mixed", action="store_true", help="accept runs with different code or data")
    ap.add_argument("--fig-dir", default=None, help="output dir for the PDF (default $AG_OUT/figures)")


def run_dir_of(out_root: Path, run_id: str, exp: str) -> Path:
    if run_id == "latest":
        p = out_root / "raw" / exp / "latest.txt"
        if not p.exists():
            raise RunRejected(f"{p} does not exist: no eligible run for {exp} (mock runs never qualify)")
        run_id = p.read_text().strip()
    for row in read_rows(out_root / "raw" / "index.jsonl"):
        if row.get("run_id") == run_id:
            return out_root / row["path"]
    hits = list((out_root / "raw").glob(f"*/{run_id}"))
    if len(hits) == 1:
        return hits[0]
    raise RunRejected(f"run {run_id} not found under {out_root / 'raw'}")


def is_mock(lin: Mapping[str, Any]) -> bool:
    return lin.get("clock") == "virtual" or lin.get("producer") == "mock" or lin.get("engine_version") == "mock"


def check_run(run_dir: Path, *, allow_mock: bool) -> dict[str, Any]:
    lin = read_json(run_dir / "lineage.json")
    if not allow_mock:
        if is_mock(lin):
            raise RunRejected(f"{lin['run_id']}: mock / virtual-clock run (pass --allow-mock for tests)")
        if lin.get("code_dirty") or str(lin.get("code_commit", "")).endswith("-dirty"):
            raise RunRejected(f"{lin['run_id']}: produced from a dirty tree")
        checks = read_json(run_dir / "checks.json") if (run_dir / "checks.json").exists() else []
        if not all(c["ok"] for c in checks):
            raise RunRejected(f"{lin['run_id']}: checks.json has failures")
    return lin


def check_consistent(lineages: Sequence[Mapping[str, Any]], *, allow_mixed: bool,
                     keys: Iterable[str] = ("code_commit", "data_manifest_sha")) -> None:
    if allow_mixed or len(lineages) < 2:
        return
    for k in keys:
        vals = {str(l.get(k)) for l in lineages}
        if len(vals) > 1:
            raise RunRejected(f"runs differ in {k}: {sorted(vals)} (pass --allow-mixed to combine)")


def load_runs(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
    out_root = Path(args.out)
    ids = args.run_id or ["latest"]
    runs = []
    for rid in ids:
        d = run_dir_of(out_root, rid, args.exp)
        runs.append((d, check_run(d, allow_mock=args.allow_mock)))
    check_consistent([l for _, l in runs], allow_mixed=args.allow_mixed)
    return runs


def rows(run_dir: Path, name: str) -> list[dict]:
    """Rows of <name>.jsonl, or <name>.jsonl.txt in test fixtures."""
    return read_rows(run_dir / name)


def write_csv(out_root: Path, fig: str, run_id: str, name: str, data: Sequence[Mapping[str, Any]]) -> Path:
    d = out_root / "processed" / fig / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    cols = sorted({k for r in data for k in r})
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(data)
    return p


def save_figure(fig, out_root: Path, name: str, run_ids: Sequence[str], fig_dir: str | None = None) -> Path:
    d = Path(fig_dir) if fig_dir else out_root / "figures"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.pdf"
    fig.savefig(p, bbox_inches="tight")
    man_path = d / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man[name] = {"file": p.name, "run_ids": list(run_ids)}
    atomic_write_json(man_path, man)
    return p
