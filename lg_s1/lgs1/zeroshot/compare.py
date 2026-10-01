"""E0-6 roll-up: collect every <guard>/<bench>.summary.json and compare with published.yaml.

    python -m lgs1.zeroshot.compare            -> $LG_OUT/reports/e06_reproduction.md (+ .json)

Pass rule: |ours(mode) - published| <= tolerance_pt (1.0).  Guards without a published number are
listed with "n/a" (they still need a number before entering Table 3).  Parse-failure rate and mean
decode tokens are printed next to each row (Table 3 / E4 inputs).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..paths import S1Paths
from . import guards as G

PUB = Path(__file__).with_name("published.yaml")


def collect(out_root: Path) -> List[Dict]:
    rows = []
    for f in sorted(out_root.glob("*/*.summary.json")):
        try:
            rows.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[skip] {f}: {e}", file=sys.stderr)
    return rows


def compare(rows: List[Dict], pub: Dict) -> List[Dict]:
    tol = float(pub.get("tolerance_pt", 1.0))
    out = []
    for s in rows:
        g, b = s["guard"], s["bench"]
        p = (pub.get("guards", {}).get(g, {}) or {}).get(b, {}) or {}
        mode = p.get("mode", "loose")
        ours = 100.0 * (s["f1_strict"] if mode == "strict" else s["f1_loose"])
        other = 100.0 * (s["f1_loose"] if mode == "strict" else s["f1_strict"])
        ref = p.get("f1")
        delta = (ours - ref) if ref is not None else None
        status = "n/a" if ref is None else ("PASS" if abs(delta) <= tol else "FAIL")
        exp_n = (pub.get("expected_n") or {}).get(b)
        if exp_n and s["n"] != exp_n:
            status = f"RERUN(n={s['n']}!={exp_n})"
        out.append(dict(guard=g, bench=b, n=s["n"], mode=mode, ours=round(ours, 2), other_mode=round(other, 2),
                        published=ref, delta=None if delta is None else round(delta, 2), status=status,
                        parse_fail_rate=round(100 * s["parse_fail_rate"], 2), mean_decode=round(s["mean_decode_tokens"], 1),
                        auprc=(round(100 * s["auprc"], 2) if s.get("auprc") is not None else None),
                        template_status=s.get("template_status", G.GUARDS.get(g).template_status if g in G.GUARDS else "?"),
                        source=p.get("source", ""), seconds=s.get("seconds")))
    return out


def to_markdown(cmp: List[Dict], tol: float) -> str:
    lines = [f"# E0-6 zero-shot reproduction ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Pass = |ours − published| ≤ {tol} pt (F1, harmful class, %). `other` = the other controversial mode.", "",
             "| guard | bench | n | mode | ours | other | published | Δ | status | parse fail % | decode tok | AUPRC | template | source |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in cmp:
        lines.append(f"| {r['guard']} | {r['bench']} | {r['n']} | {r['mode']} | {r['ours']} | {r['other_mode']} | "
                     f"{'' if r['published'] is None else r['published']} | {'' if r['delta'] is None else r['delta']} | "
                     f"{r['status']} | {r['parse_fail_rate']} | {r['mean_decode']} | {'' if r['auprc'] is None else r['auprc']} | "
                     f"{r['template_status']} | {r['source']} |")
    n_pass = sum(1 for r in cmp if r["status"] == "PASS")
    n_fail = sum(1 for r in cmp if r["status"] == "FAIL")
    n_na = sum(1 for r in cmp if r["status"] == "n/a")
    lines += ["", f"PASS {n_pass} · FAIL {n_fail} · n/a {n_na}", ""]
    n_rerun = sum(1 for r in cmp if str(r["status"]).startswith("RERUN"))
    if n_rerun:
        lines += ["## RERUN rows — wrong benchmark size (old loader); delete and resubmit", "",
                  "    python s1ctl.py zeroshot --redo-mismatch --sbatch", ""]
    if n_fail:
        lines += ["## FAIL rows — next action", ""]
        for r in cmp:
            if r["status"] == "FAIL":
                lines.append(f"- {r['guard']}/{r['bench']}: Δ={r['delta']} → 템플릿({r['template_status']})·파서·max_new_tokens·"
                             f"benchmark 버전(ToxicChat 0124, WGTest prompt split) 확인 후 재실행; parse fail {r['parse_fail_rate']}%")
    if n_na:
        lines += ["", "## n/a rows — fill published.yaml from the paper/model card before Table 3", ""]
        for r in cmp:
            if r["status"] == "n/a":
                lines.append(f"- {r['guard']}/{r['bench']}: ours={r['ours']} ({r['mode']}); {r['source']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="zeroshot output root (default $LG_OUT/zeroshot)")
    a = ap.parse_args(argv)
    P = S1Paths.load()
    P.ensure_out()
    root = Path(a.out) if a.out else P.out / "zeroshot"
    pub = yaml.safe_load(PUB.read_text(encoding="utf-8"))
    rows = collect(root)
    cmp = compare(rows, pub)
    md = to_markdown(cmp, float(pub.get("tolerance_pt", 1.0)))
    (P.out / "reports" / "e06_reproduction.md").write_text(md, encoding="utf-8")
    (P.out / "reports" / "e06_reproduction.json").write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print(f"[written] {P.out / 'reports' / 'e06_reproduction.md'}")
    return 0 if not any(r["status"] == "FAIL" for r in cmp) else 1


if __name__ == "__main__":
    sys.exit(main())
