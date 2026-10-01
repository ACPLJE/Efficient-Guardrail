"""E0-4 — the seven leakage/consistency gates, the root manifest, and the seal (report §5.6; plan S3 item 4).

    python -m lgs3.gates                 # evaluate -> splits/audit.json["gates_e04"], manifests/lg_data_v2.manifest.json
    python -m lgs3.gates --seal          # additionally chmod -R a-w the data directories when all gates pass

Gates (all must pass before any training job reads lg_data_v2):
  1 needle GROUP and background PIECE sets are disjoint between train / val / calib / test rows
  2 no train/val/calib row uses a needle that the audit removed (external / public-test overlap) or a test-source needle
  3 (split, subset, id) unique; every main harmful row has an existing benign twin; every row has recipe_hash and a
    needle_tok_span (tok/qwen3)
  4 main-set needles <= 128 tokens (100 %); long needles only in the long_needle subset (native rows are exempt:
    they carry the pool prompt as-is for the E1 moderation task -- 16,961 of them exceed 128 tokens, 2026-09-19)
  5 label balance 50:50 per (split, L) in the main sets; identical row counts per position
  6 screening complete for all 10 domains (pass rates recorded) and the manual-review file exists
  7 per-model re-tokenisation: qwen3 rows within ±10 % of L_target, and the 100-row reconstruction check passed for
    every model that was tokenised
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import stat
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from lgs1.lineage import code_commit, sha256_file
from lgs2 import config as C
from lgs2.config import NEEDLE_MAX_TOK, RECIPE_V2, S2Paths
from lgs2.needles import load_pool

MAIN_SUBSETS = ("main", "main16k")
SPLITS = ("train", "val", "calib", "test")


def iter_rows(path: Path) -> Iterable[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r.pop("text", None)            # gates never need the text
                yield r


def rows_files(root: Path) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(str(root / "*" / "*" / "rows.jsonl.gz")))


def evaluate(P: S2Paths, root: Optional[Path] = None, tok_root: Optional[Path] = None) -> Dict:
    root = root or P.synth
    tok_root = tok_root or (P.root / "tok")
    files = rows_files(root)
    if not files:
        raise SystemExit(f"no rows files under {root}")
    pool = {r["id"]: r for r in load_pool(P)}
    audit = json.loads((P.splits / "audit.json").read_text(encoding="utf-8"))
    removed = set(audit.get("removed", {}))
    test_source_ids = {i for i, r in pool.items() if r["pool"] == "test"}
    groups_by_split: Dict[str, Set[str]] = defaultdict(set)
    docs_by_split: Dict[str, Set[str]] = defaultdict(set)
    seen_ids: Counter = Counter()
    twin_targets: Dict[str, str] = {}
    all_ids: Set[str] = set()
    need_span: Set[str] = set()
    n_rows = n_missing_recipe = n_needle_long_main = n_long_outside = n_removed_used = n_testsrc_in_train = n_native_long = 0
    balance: Counter = Counter()
    per_pos: Counter = Counter()
    per_file: Dict[str, int] = {}
    for f in files:
        split, subset = f.parent.parent.name, f.parent.name
        n_file = 0
        for r in iter_rows(f):
            n_rows += 1
            n_file += 1
            key = (r["split"], r["subset"], r["id"])
            seen_ids[key] += 1
            all_ids.add(r["id"])
            if not r.get("recipe_hash"):
                n_missing_recipe += 1
            if r.get("needle_id"):
                need_span.add(r["id"])
            for nid in (r.get("needle_ids") or ([r["needle_id"]] if r.get("needle_id") else [])):
                if nid in pool:
                    groups_by_split[split].add(pool[nid]["group_id"])
                if split != "test":
                    if nid in removed:
                        n_removed_used += 1
                    if nid in test_source_ids:
                        n_testsrc_in_train += 1
            for d in r.get("haystack_doc_ids") or []:
                docs_by_split[split].add(d)
            if subset in MAIN_SUBSETS:
                if r.get("n_tok_needle", 0) > NEEDLE_MAX_TOK:
                    n_needle_long_main += 1
                balance[(split, subset, r["L_target"], r["label"])] += 1
                per_pos[(split, subset, r["L_target"], r["depth"])] += 1
                if r["label"] == "harmful":
                    twin_targets[r["id"]] = r.get("twin_id")
            elif subset == "native":
                if r.get("n_tok_needle", 0) > NEEDLE_MAX_TOK:
                    n_native_long += 1          # the moderation prompt IS the row (E1); the 128-token rule is for inserted needles
            elif subset != "long_needle" and r.get("n_tok_needle", 0) > NEEDLE_MAX_TOK:
                n_long_outside += 1
        per_file[f"{split}/{subset}"] = n_file
    # gate 1
    inter = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            inter[f"{a}&{b}"] = {"needle_groups": len(groups_by_split[a] & groups_by_split[b]),
                                 "haystack_docs": len(docs_by_split[a] & docs_by_split[b])}
    g1 = all(v["needle_groups"] == 0 and v["haystack_docs"] == 0 for v in inter.values())
    # gate 2
    g2 = n_removed_used == 0 and n_testsrc_in_train == 0
    # gate 3
    dup = sum(1 for k, c in seen_ids.items() if c > 1)
    twin_missing = sum(1 for h, t in twin_targets.items() if not t or t not in all_ids)
    span_missing = None
    qwen_dir = tok_root / "qwen3"
    if qwen_dir.exists():
        have = set()
        for sp in glob.glob(str(qwen_dir / "*" / "*" / "spans.jsonl.gz")):
            with gzip.open(sp, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("needle_tok_span"):
                            have.add(rec["id"])
        span_missing = len(need_span - have)
    g3 = dup == 0 and twin_missing == 0 and n_missing_recipe == 0 and (span_missing == 0)
    # gate 4
    g4 = n_needle_long_main == 0 and n_long_outside == 0
    # gate 5
    imbalance = []
    for (split, subset, L, lab), n in balance.items():
        other = balance.get((split, subset, L, "benign" if lab == "harmful" else "harmful"), 0)
        if other != n:
            imbalance.append(f"{split}/{subset}/L{L}: {lab}={n} other={other}")
    pos_counts = defaultdict(set)
    for (split, subset, L, depth), n in per_pos.items():
        pos_counts[(split, subset, L)].add(n)
    pos_uneven = [f"{k[0]}/{k[1]}/L{k[2]}: {sorted(v)}" for k, v in pos_counts.items() if len(v) > 1 and k[0] != "train"]  # train uses random depths
    g5 = not imbalance and not pos_uneven
    # gate 6
    ss = P.screen / "screen_summary.json"
    screen_ok, pass_rates = False, {}
    if ss.exists():
        s = json.loads(ss.read_text(encoding="utf-8"))
        pass_rates = {d: v.get("pass_rate") for d, v in s["domains"].items()}
        screen_ok = all(v.get("status") == "complete" for v in s["domains"].values()) and len(s["domains"]) == len(C.DOMAINS)
    review = P.screen / "manual_review_50.md"
    review_lines = review.read_text(encoding="utf-8").count("verdict:") if review.exists() else 0
    g6 = screen_ok and review.exists()
    # gate 7
    tok_reports = {}
    for rp in glob.glob(str(tok_root / "*" / "*" / "report.json")):
        rep = json.loads(Path(rp).read_text(encoding="utf-8"))
        tok_reports[f"{rep['model']}/{rep['split']}"] = rep["subsets"]
    len_fail = [k for k, subs in tok_reports.items() if k.startswith("qwen3/") for s2, v in subs.items()
                if v["len_within_tol"] is not None and v["len_within_tol"] < 1.0]
    recon_fail = [f"{k}/{s2}" for k, subs in tok_reports.items() for s2, v in subs.items() if not v["recon_pass"]]
    g7 = bool(tok_reports) and any(k.startswith("qwen3/") for k in tok_reports) and not len_fail and not recon_fail
    gates = {
        "1_split_disjointness": {"pass": g1, "intersections": inter},
        "2_no_removed_or_test_source_needles_in_train": {"pass": g2, "removed_used": n_removed_used, "test_source_in_train": n_testsrc_in_train,
                                                         "audit_removed_total": len(removed)},
        "3_ids_twins_recipe_spans": {"pass": g3, "duplicate_ids": dup, "twin_missing": twin_missing, "missing_recipe_hash": n_missing_recipe,
                                     "rows_without_qwen3_span": span_missing},
        "4_needle_length": {"pass": g4, "main_rows_needle_gt128": n_needle_long_main, "long_needles_outside_subset": n_long_outside,
                            "native_rows_gt128_exempt": n_native_long},
        "5_balance_positions": {"pass": g5, "imbalance": imbalance[:10], "uneven_positions": pos_uneven[:10]},
        "6_screening": {"pass": g6, "complete": screen_ok, "pass_rates": pass_rates, "manual_review_entries": review_lines},
        "7_retokenisation": {"pass": g7, "qwen3_length_failures": len_fail, "reconstruction_failures": recon_fail,
                             "models_tokenised": sorted({k.split("/")[0] for k in tok_reports})},
    }
    result = {"pass": all(v["pass"] for v in gates.values()), "gates": gates, "n_rows": n_rows, "rows_per_file": per_file,
              "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    audit["gates_e04"] = result
    (P.splits / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def root_manifest(P: S2Paths) -> Path:
    """manifests/lg_data_v2.manifest.json: every rows manifest + sha256 of the manifests themselves."""
    entries = []
    for mp in sorted(P.manifests.glob("*.manifest.json")):
        if mp.name == "lg_data_v2.manifest.json":
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        entries.append({"manifest": mp.name, "manifest_sha256": sha256_file(mp), "path": m["path"], "sha256": m["sha256"],
                        "rows": m["rows"], "recipe_hash": m["recipe_hash"]})
    root = {"name": "lg_data_v2", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_commit": code_commit(Path(__file__).resolve().parents[1]),
            "recipe_version": RECIPE_V2["version"], "n_files": len(entries), "n_rows": sum(e["rows"] for e in entries), "files": entries}
    out = P.manifests / "lg_data_v2.manifest.json"
    out.write_text(json.dumps(root, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def seal(P: S2Paths) -> List[str]:
    """chmod -R a-w on the frozen directories (never synth_dryrun / logs)."""
    sealed = []
    for d in ("synth", "tok", "needles", "splits", "haystack", "screen", "related", "manifests"):
        p = P.root / d
        if not p.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            for name in filenames:
                fp = Path(dirpath) / name
                fp.chmod(fp.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            for name in dirnames:
                dp = Path(dirpath) / name
                dp.chmod(dp.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        sealed.append(str(p))
    (P.root / "SEALED").write_text(f"sealed_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n" + "\n".join(sealed) + "\n", encoding="utf-8")
    return sealed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E0-4 gates + manifest + seal")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="evaluate synth_dryrun/ (no seal)")
    a = ap.parse_args(argv)
    P = S2Paths.load()
    root = P.synth_dryrun if a.dry_run else P.synth
    tok_root = (P.synth_dryrun / "tok") if a.dry_run else (P.root / "tok")
    res = evaluate(P, root, tok_root)
    for k, v in res["gates"].items():
        print(f"[gate] {k:48s} {'PASS' if v['pass'] else 'FAIL'}  " + ", ".join(f"{a2}={b2}" for a2, b2 in v.items() if a2 not in ("pass", "intersections", "pass_rates")))
    print(f"[gates] {'ALL PASS' if res['pass'] else 'FAIL'}  rows={res['n_rows']:,}")
    if not a.dry_run:
        mp = root_manifest(P)
        print(f"[manifest] {mp}")
        if a.seal:
            if not res["pass"]:
                print("[seal] refused: gates failed", file=sys.stderr)
                return 1
            for d in seal(P):
                print(f"[seal] read-only: {d}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
