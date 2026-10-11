#!/usr/bin/env python3
"""Export the W0-grader label sheet of a run (plan 5.3 #7, 8.2 W0; build plan S2 step 0).

  python scripts/export_label_sheet.py --run-dir $AG_OUT/raw/W0-grader/<run_id> [--out FILE.csv] [--sample N --seed S]

Joins outcomes_run.jsonl with the run's data manifest (raw/data/<lineage.data_manifest_sha>/requests.jsonl,
out root = run_dir.parents[2]) and writes label_sheet.csv (utf-8-sig, so spreadsheet apps detect UTF-8)
plus label_sheet.jsonl next to the run, or FILE.csv + FILE.jsonl with --out. Rows are ordered by
request_id; --sample picks a seeded subset (same seed -> same rows, still in request_id order).
Requests that ended in a fatal error (no generation) cannot be labelled: they are left out and counted.

marker_hint = re.search(task_meta.marker, output_text) on attack requests (empty on benign ones). It is a
HINT for the labeller only, never an attack-success verdict (some markers fire on neutral text).
label_attack_success, label_utility, labeler and notes are left empty for the labellers.

CSV only: a text cell that starts with = + - @ TAB or CR gets a leading apostrophe, so a spreadsheet shows it as
text instead of evaluating it as a formula (a markdown bullet list "- ...", "+1-555-0100"). label_sheet.jsonl
keeps the exact text and is the authoritative copy. An existing sheet with filled label columns is never
overwritten (pass --force to replace it).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

COLUMNS = ("request_id", "pair_key", "is_attack", "attack_type", "position", "goal_id", "injected_instruction",
           "goal", "task_text", "output_text", "n_output_tokens", "marker_hint", "label_attack_success",
           "label_utility", "labeler", "notes")
LABEL_COLUMNS = ("label_attack_success", "label_utility", "labeler", "notes")


def build_rows(run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(rows in request_id order, info {run_id, n_skipped_errors, skipped})."""
    from ag.data.lc_inject.payloads import marker_hint
    from ag.data.manifest import load_requests
    from ag.harness.writer import read_json, read_rows

    run_dir = Path(os.path.abspath(run_dir))   # parents[2] needs an absolute path; lexical, so symlinked trees work
    lin = read_json(run_dir / "lineage.json")
    data_dir = run_dir.parents[2] / "raw" / "data" / lin["data_manifest_sha"]
    reqs = {r.request_id: r for r in load_requests(data_dir)}
    if not reqs:
        raise FileNotFoundError(f"no requests in {data_dir} (data_manifest_sha of {run_dir.name})")
    live: dict[str, Mapping[str, Any]] = {}
    for row in read_rows(run_dir / "outcomes_run.jsonl"):
        live[row["request_id"]] = row              # a resumed request keeps its latest row
    unknown = sorted(set(live) - set(reqs))
    if unknown:
        raise ValueError(f"{len(unknown)} outcomes_run rows are not in {data_dir}: {unknown[:3]}")
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for rid in sorted(live):
        gen = live[rid].get("generation")
        if gen is None:
            skipped.append(rid)
            continue
        req = reqs[rid]
        meta = dict(req.task_meta or {})
        att = req.attack
        text = gen.get("text_delivered", "")
        marker = meta.get("marker")
        rows.append({
            "request_id": rid, "pair_key": req.pair_key, "is_attack": req.is_attack,
            "attack_type": att.attack_type if att else None, "position": att.position if att else None,
            "goal_id": meta.get("goal_id"), "injected_instruction": meta.get("injected_instruction"),
            "goal": att.goal if att else None, "task_text": meta.get("task_text", req.task),
            "output_text": text, "n_output_tokens": len(gen.get("output_ids") or ()),
            "marker_hint": marker_hint(marker, text) if (att and marker) else None,
            **{k: None for k in LABEL_COLUMNS},
        })
    return rows, {"run_id": lin["run_id"], "n_skipped_errors": len(skipped), "skipped": skipped}


def sample_rows(rows: list[dict[str, Any]], n: int | None, seed: int = 0) -> list[dict[str, Any]]:
    if n is None or n >= len(rows):
        return list(rows)
    import numpy as np
    pick = sorted(int(i) for i in np.random.default_rng(seed).choice(len(rows), size=n, replace=False))
    return [rows[i] for i in pick]


FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str) and v.startswith(FORMULA_LEAD):
        return "'" + v                         # shown as text, never evaluated (the JSONL keeps the exact text)
    return str(v)


def has_labels(csv_path: Path) -> bool:
    """True if an existing sheet may hold labeller work: any filled label column, or a file this exporter cannot
    read back as its own sheet (re-saved as cp949 or with another delimiter, renamed headers). Fails closed."""
    if not csv_path.exists():
        return False
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            if not set(LABEL_COLUMNS) <= set(rd.fieldnames or ()):
                return True
            return any(any((row.get(c) or "").strip() for c in LABEL_COLUMNS) for row in rd)
    except (UnicodeDecodeError, csv.Error):
        return True


def csv_text(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([_cell(r[c]) for c in COLUMNS])
    return buf.getvalue()


def write_sheet(rows: list[dict[str, Any]], csv_path: Path, run_id: str) -> tuple[Path, Path]:
    from ag.harness.writer import atomic_write_text, write_rows
    jsonl_path = csv_path.with_suffix(".jsonl")
    if jsonl_path == csv_path:
        raise ValueError(f"--out must not end in .jsonl: {csv_path}")
    atomic_write_text(csv_path, "﻿" + csv_text(rows))         # utf-8 + BOM == utf-8-sig
    write_rows(jsonl_path, run_id, [{c: r[c] for c in COLUMNS} for r in rows])
    return csv_path, jsonl_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_label_sheet", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="raw/<E>/<run_id> of a finished run")
    ap.add_argument("--out", default=None, help="CSV path (the JSONL goes next to it); default <run-dir>/label_sheet.csv")
    ap.add_argument("--sample", type=int, default=None, help="export a seeded random subset of N requests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite a sheet whose label columns are filled")
    a = ap.parse_args(argv)
    run_dir = Path(os.path.abspath(a.run_dir))
    target = Path(a.out) if a.out else run_dir / "label_sheet.csv"
    if not a.force and has_labels(target):
        print(f"error: {target} already has labels; not overwriting (use --out for a new file or --force)",
              file=sys.stderr)
        return 2
    rows, info = build_rows(run_dir)
    rows = sample_rows(rows, a.sample, a.seed)
    csv_path, jsonl_path = write_sheet(rows, target, info["run_id"])
    n_att = sum(1 for r in rows if r["is_attack"])
    print(f"wrote {len(rows)} rows ({n_att} attack, {len(rows) - n_att} benign) -> {csv_path} and {jsonl_path}")
    if info["n_skipped_errors"]:
        print(f"left out {info['n_skipped_errors']} requests without a generation (fatal errors): "
              f"{info['skipped'][:3]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
