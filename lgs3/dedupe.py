"""Drop exact duplicate rows (same id) from one synthesised subset and keep every derived file consistent.

    python -m lgs3.dedupe --split train --subset main

2026-09-19: gate 3 found 6 duplicate (split, subset, id) keys in train/main -- three (pair, L) combinations drew the
same random depth twice (4-decimal rounding), so the h/b rows of that depth were generated twice with identical text.
Dropping the second copy loses nothing.  Rewrites rows.jsonl.gz (first occurrence kept), the subset manifest (new
sha256 / row count / counts) and tok/<model>/<split>/<subset>/spans.jsonl.gz for every tokenised model (duplicate
span records dropped); the per-model report.json keeps its statistics (computed over a superset of identical rows).
Refuses to touch a sealed lg_data_v2."""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from lgs2.config import RECIPE_V2, S2Paths
from lgs2.synth import write_manifest


def _dedupe_jsonl_gz(path: Path) -> Tuple[int, int, List[Dict]]:
    """Rewrite `path` keeping the first record per id.  Returns (n_in, n_dropped, kept records without text)."""
    seen = set()
    kept: List[Dict] = []
    n_in = n_drop = 0
    tmp = path.with_suffix(".tmp.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f, gzip.open(tmp, "wt", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            r = json.loads(line)
            if r["id"] in seen:
                n_drop += 1
                continue
            seen.add(r["id"])
            g.write(line if line.endswith("\n") else line + "\n")
            r.pop("text", None)
            kept.append(r)
    if n_drop:
        os.replace(tmp, path)
    else:
        tmp.unlink()
    return n_in, n_drop, kept


def run(P: S2Paths, split: str, subset: str, recipe=None) -> Dict:
    recipe = recipe if recipe is not None else RECIPE_V2
    if (P.root / "SEALED").exists():
        raise SystemExit(f"{P.root}/SEALED exists: lg_data_v2 is sealed, refusing to rewrite rows")
    rows_path = P.synth / split / subset / "rows.jsonl.gz"
    if not rows_path.exists():
        raise SystemExit(f"missing {rows_path}")
    n_in, n_drop, kept = _dedupe_jsonl_gz(rows_path)
    out = {"split": split, "subset": subset, "rows_in": n_in, "dropped": n_drop, "rows": n_in - n_drop, "spans": {}}
    if n_drop:
        counts: Counter = Counter()
        for r in kept:
            counts[(r["subset"], r["L_target"], r["label"])] += 1
        info = {"rows": n_in - n_drop, "n_len_bad": 0,
                "counts": {f"{s}/L{L}/{lab}": c for (s, L, lab), c in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2]))}}
        # variant subsets carry a different recipe hash; only main-set subsets are expected here
        write_manifest(P, split, subset, rows_path, info, recipe, dry_run=False)
    for sp in sorted(glob.glob(str(P.root / "tok" / "*" / split / subset / "spans.jsonl.gz"))):
        m_in, m_drop, _ = _dedupe_jsonl_gz(Path(sp))
        out["spans"][Path(sp).parents[2].name] = {"in": m_in, "dropped": m_drop}
    print(f"[dedupe] {split}/{subset}: rows {n_in:,} -> {n_in - n_drop:,} ({n_drop} duplicate ids dropped); "
          f"spans: " + ", ".join(f"{m} -{v['dropped']}" for m, v in out["spans"].items()) + (" (manifest rewritten)" if n_drop else " (nothing to do)"),
          flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="drop duplicate-id rows from one subset (+ manifest + token spans)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", required=True)
    a = ap.parse_args(argv)
    run(S2Paths.load().ensure(), a.split, a.subset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
