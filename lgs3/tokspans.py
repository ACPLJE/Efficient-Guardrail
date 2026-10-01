"""D8 — per-model re-tokenisation of the SAME strings (report §5.1, §5.8; plan S3 D8).

    python -m lgs3.tokspans --model llama3 --split test          # -> tok/llama3/test/<subset>/spans.jsonl.gz + report.json
    python -m lgs3.tokspans --model all --split all

Per row: n_tok, needle_tok_span (token span covering the needle chars), needle_tok_spans (all needles / pieces),
window_ids_containing_needle (2b=256-token windows, stride b=128), fully_contained (Prop 1 premise: the needle sits
wholly inside at least one window).  The report checks (a) |n_tok - L_target| <= 10 % for the REFERENCE tokenizer
(qwen3; other models only record the ratio) and (b) 100 random rows: the decoded token span contains the needle.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from lgs1.lgcore.windows import make_windows, windows_containing
from lgs2.config import RECIPE_V2, S2Paths
from lgs2.tok import HFTok, WhitespaceTok, token_span_for_chars

TOK_MODELS: Dict[str, str] = {
    "qwen3": "Qwen/Qwen3Guard-Gen-0.6B",            # reference (length grid)
    "llama3": "meta-llama/Llama-Guard-3-8B",
    "mistral": "allenai/wildguard",                 # Mistral-7B-v0.3 vocabulary (WildGuard)
    "gemma": "google/shieldgemma-9b",
    "gptoss": "openai/gpt-oss-safeguard-20b",       # o200k_harmony
}
B = RECIPE_V2["window_b"]


def load_tok(name: str):
    if name in ("whitespace", "ws"):
        return WhitespaceTok()
    return HFTok(TOK_MODELS.get(name, name), name)


def iter_rows(path: Path) -> Iterable[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def span_record(tok, row: Dict) -> Dict:
    ids, offs = tok.encode_with_offsets(row["text"])
    n = len(ids)
    wins = make_windows(n, B)
    spans = row.get("needle_char_spans") or ([row["needle_char_span"]] if row.get("needle_char_span") else [])
    tspans, wids, contained = [], [], []
    for sp in spans:
        a, b = token_span_for_chars(offs, (sp[0], sp[1]))
        tspans.append([a, b])
        w = windows_containing((a, b), wins)
        wids.append(w)
        contained.append(bool(w))
    return {"id": row["id"], "n_tok": n, "needle_tok_span": tspans[0] if tspans else None, "needle_tok_spans": tspans,
            "window_ids_containing_needle": wids[0] if wids else [], "window_ids_per_span": wids,
            "fully_contained": all(contained) if contained else None, "n_windows": len(wins),
            "L_target": row.get("L_target"), "n_tok_qwen3_ref": row.get("n_tok_qwen3"), "label": row.get("label")}


def run(P: S2Paths, model: str, split: str, subsets: Optional[List[str]] = None, tok=None, root: Optional[Path] = None,
        n_check: int = 100, seed: int = 0) -> Dict:
    tok = tok or load_tok(model)
    root = root or P.synth
    split_dir = root / split
    if not split_dir.exists():
        raise SystemExit(f"{split_dir} does not exist (run synthesis first)")
    out_root = (P.root / "tok" if root == P.synth else P.synth_dryrun / "tok") / model / split
    report: Dict = {"model": model, "repo": TOK_MODELS.get(model, model), "split": split, "subsets": {}, "b": B,
                    "length_tol": RECIPE_V2["length_tol"], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rng = random.Random(seed)
    for sub_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        sub = sub_dir.name
        if subsets and sub not in subsets:
            continue
        rows_path = sub_dir / "rows.jsonl.gz"
        if not rows_path.exists():
            continue
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n = n_len_ok = n_with_target = n_contained = n_needle = 0
        ratios: List[float] = []
        check_pool: List[Tuple[Dict, Dict]] = []
        with gzip.open(out_dir / "spans.jsonl.gz", "wt", encoding="utf-8") as f:
            for row in iter_rows(rows_path):
                rec = span_record(tok, row)
                f.write(json.dumps(rec) + "\n")
                n += 1
                if row.get("L_target"):
                    n_with_target += 1
                    if abs(rec["n_tok"] - row["L_target"]) <= RECIPE_V2["length_tol"] * row["L_target"]:
                        n_len_ok += 1
                if row.get("n_tok_qwen3"):
                    ratios.append(rec["n_tok"] / max(1, row["n_tok_qwen3"]))
                if rec["fully_contained"] is not None:
                    n_needle += 1
                    n_contained += int(rec["fully_contained"])
                # reservoir sample for the reconstruction check
                if rec["needle_tok_span"] and row.get("needle_id"):
                    if len(check_pool) < n_check:
                        check_pool.append((row, rec))
                    else:
                        j = rng.randint(0, n - 1)
                        if j < n_check:
                            check_pool[j] = (row, rec)
        # reconstruction: decoded covering token span must contain the needle characters
        n_recon_ok = 0
        for row, rec in check_pool:
            ids, offs = tok.encode_with_offsets(row["text"])
            a, b = rec["needle_tok_span"]
            dec = tok.decode_span(row["text"], offs, a, b)
            s, e = row["needle_char_span"]
            if row["text"][s:e] in dec:
                n_recon_ok += 1
        report["subsets"][sub] = {
            "n_rows": n, "seconds": round(time.time() - t0, 1),
            "len_within_tol": (n_len_ok / n_with_target) if n_with_target else None, "n_with_target": n_with_target,
            "ratio_to_qwen3_mean": (sum(ratios) / len(ratios)) if ratios else None,
            "ratio_to_qwen3_min": min(ratios) if ratios else None, "ratio_to_qwen3_max": max(ratios) if ratios else None,
            "needle_fully_contained": (n_contained / n_needle) if n_needle else None,
            "recon_checked": len(check_pool), "recon_ok": n_recon_ok, "recon_pass": n_recon_ok == len(check_pool),
        }
        print(f"[tokspans] {model:8s} {split}/{sub:14s} rows={n:6d} len_ok={report['subsets'][sub]['len_within_tol']} "
              f"ratio={report['subsets'][sub]['ratio_to_qwen3_mean']} contained={report['subsets'][sub]['needle_fully_contained']} "
              f"recon={n_recon_ok}/{len(check_pool)}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D8 per-model token spans")
    ap.add_argument("--model", default="all", help="qwen3|llama3|mistral|gemma|gptoss|all (or a repo id)")
    ap.add_argument("--split", default="all")
    ap.add_argument("--subsets", default="", help="comma list (default: every subset of the split)")
    ap.add_argument("--dry-run", action="store_true", help="read synth_dryrun/ instead of synth/")
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    models = list(TOK_MODELS) if a.model == "all" else [a.model]
    root = P.synth_dryrun if a.dry_run else P.synth
    splits = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "manifests" and p.name != "tok") if a.split == "all" else [a.split]
    rc = 0
    for m in models:
        try:
            tok = load_tok(m)
        except Exception as e:
            print(f"[tokspans] {m}: tokenizer unavailable ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
            rc = 1
            continue
        for sp in splits:
            rep = run(P, m, sp, [x for x in a.subsets.split(",") if x] or None, tok=tok, root=root)
            bad = [k for k, v in rep["subsets"].items() if not v["recon_pass"] or (m == "qwen3" and v["len_within_tol"] not in (None, 1.0))]
            if bad:
                rc = 1
                print(f"[tokspans] {m}/{sp}: CHECK FAILED for {bad}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
