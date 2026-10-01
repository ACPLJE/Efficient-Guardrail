"""D10 — DATACARD.md for lg_data_v2 (report §5.4 'presence label', §5.2 licences, Table A1 draft).

    python -m lgs3.datacard        # -> $LG_DATA_V2/DATACARD.md (+ reports/table_A1_draft.md)

Everything numeric is read from the files the pipeline wrote (pool_stats, audit, screen_summary, manifests,
tok reports); nothing is typed in by hand except the licence table below.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from lgs2 import config as C
from lgs2.config import NEEDLE_SOURCES, RECIPE_V2, S2Paths

LICENCES = {
    "wildguardmix": "ODC-BY 1.0 (allenai/wildguardmix)", "aegis2": "CC-BY-4.0 (nvidia/Aegis-AI-Content-Safety-Dataset-2.0)",
    "aegis1": "CC-BY-4.0", "toxicchat": "CC-BY-NC-4.0 (lmsys/toxic-chat)", "orbench": "CC-BY-4.0 (bench-llm/or-bench)",
    "openai_moderation": "MIT (mmathys/openai-moderation-api-evaluation)", "simplesafetytests": "CC-BY-4.0",
    "harmbench": "MIT (walledai/HarmBench mirror)", "jbb": "MIT (JailbreakBench)", "xstest": "CC-BY-4.0 (Paul/XSTest)",
    "govreport": "CC-BY-4.0 (ccdv/govreport-summarization)", "pg19": "Apache-2.0 (deepmind/pg19 mirror)",
    "wildchat": "AI2 ImpACT low-risk (allenai/WildChat-1M)", "stack": "permissive-only subset (bigcode/the-stack-smol)",
    "arxiv": "arXiv non-exclusive (ccdv/arxiv-summarization)", "wiki_en": "CC-BY-SA-4.0", "wiki_zh": "CC-BY-SA-4.0",
    "pubmed": "PubMed terms (MedRAG/pubmed)", "agentdojo": "MIT (ethz-spylab/agentdojo)", "ledgar": "CC-BY-4.0 (LexGLUE)",
    "safetyniah": "see caskcsg/SafetyNIAH", "longsafetybench": "see LutherXD/LongSafetyBench",
}


def _load(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build(P: S2Paths) -> str:
    pool = _load(P.needles / "pool_stats.json")
    audit = _load(P.splits / "audit.json")
    screen = _load(P.screen / "screen_summary.json")
    manifests = [json.loads(Path(m).read_text(encoding="utf-8")) for m in sorted(glob.glob(str(P.manifests / "*.manifest.json")))
                 if not m.endswith("lg_data_v2.manifest.json")]
    tok_reports = {Path(r).parent.parent.name + "/" + Path(r).parent.name: json.loads(Path(r).read_text(encoding="utf-8"))
                   for r in glob.glob(str(P.root / "tok" / "*" / "*" / "report.json"))}
    L = ["# lg_data_v2 — data card", "", f"generated {time.strftime('%Y-%m-%d %H:%M')} from the pipeline outputs under `{P.root}`", "",
         "## 1. What a label means", "",
         "Every synthesized row carries a **presence label**: the label of the needle inserted into the background "
         "(`harmful` or `benign`). It is *not* a human moderation verdict on the whole document. Backgrounds are public "
         "documents that passed a two-model screening (Llama Guard 3-8B + Qwen3Guard-Gen-8B strict, 512-token chunks); "
         "residual sensitive content in a background is therefore possible and is measured, not assumed away "
         "(screening pass rates below, manual check of 50 rejected pieces in `screen/manual_review_50.md`). Every harmful row "
         "has a benign twin with the same background, depth and length (±20 %), so the label depends only on the needle.", "",
         "## 2. Sources and licences", "", "| source | role | licence |", "| --- | --- | --- |"]
    for s in NEEDLE_SOURCES:
        L.append(f"| {s.name} ({s.hf_repo}) | needle {s.pool} | {LICENCES.get(s.registry, '?')} |")
    for d in C.DOMAINS:
        L.append(f"| {d.name} | background ({d.role}) | {LICENCES.get(d.source, '?')} |")
    L += ["", "## 3. Needle pool", ""]
    if pool:
        L += [f"* {pool['n_total']:,} prompts ({pool['n_train_pool']:,} train pool, {pool['n_test_pool']:,} test pool); "
              f"{pool['n_usable_main']:,} with ≤128 Qwen3 tokens (main sets), {pool['n_long_needle']:,} long needles (129–512), "
              f"{pool['n_too_long']:,} excluded (>512)",
              f"* groups: {pool['groups']['n_groups']:,} (exact + MinHash 5-gram Jaccard ≥ 0.8; {pool['groups']['n_near_dup_merged']:,} near-duplicates merged)"]
        L += ["", "| source | status | rows read | unique | harmful |", "| --- | --- | --- | --- | --- |"]
        for k, v in pool.get("sources", {}).items():
            L.append(f"| {k} | {v.get('status')} | {v.get('n_rows_read', '')} | {v.get('n_unique', '')} | {v.get('n_harmful', '')} |")
    L += ["", "## 4. Splits and leakage audit", ""]
    if audit:
        ov = audit.get("overlap", {})
        L += [f"* split keys: needle group + background piece id; public test sources are test-only",
              f"* removed from the train pool: {ov.get('n_removed', 0):,} needles "
              f"(exact {ov.get('n_removed_exact', 0):,}, near-duplicate {ov.get('n_removed_near_dup', 0):,}) against "
              f"{', '.join(f'{k}={v:,}' for k, v in ov.get('external_sizes', {}).items())} external needles and the public test sets",
              f"* gates (S2): {'PASS' if audit.get('gates', {}).get('pass') else 'FAIL'}; "
              f"gates E0-4 (S3): {'PASS' if audit.get('gates_e04', {}).get('pass') else 'not evaluated / FAIL'}"]
        L += ["", "| split | source | harmful | benign |", "| --- | --- | --- | --- |"]
        for sp, srcs in audit.get("needle_counts", {}).items():
            for src, labs in srcs.items():
                L.append(f"| {sp} | {src} | {labs.get('harmful', 0):,} | {labs.get('benign', 0):,} |")
    L += ["", "## 5. Background screening", ""]
    if screen:
        L += ["| domain | role | pieces | kept | pass rate | kept tokens | budget met |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for d in C.DOMAINS:
            v = screen.get("domains", {}).get(d.name, {})
            pr = v.get("pass_rate")
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (d.name, d.role, v.get("n_docs", ""), v.get("n_kept", ""),
                                                            "" if pr is None else "%.3f" % pr, f"{v.get('kept_tokens', 0):,}", v.get("budget_met", "")))
    L += ["", "## 6. Synthesized sets (Table A1 draft)", "", "| split/subset | rows | recipe hash | sha256 |", "| --- | --- | --- | --- |"]
    for m in manifests:
        L.append(f"| {m['split']}/{m['subset']} | {m['rows']:,} | `{m['recipe_hash'][:12]}` | `{m['sha256'][:12]}` |")
    L += ["", f"Recipe {RECIPE_V2['version']}: length grid (Qwen3 tokens) {RECIPE_V2['sets']['test']['L']}, test positions "
          f"{RECIPE_V2['sets']['test']['positions']}, {RECIPE_V2['sets']['test']['pairs']} test pairs, needle ≤ {RECIPE_V2['needle_max_tok']} tokens, "
          f"twin length ±{int(RECIPE_V2['twin_len_tol'] * 100)} %, length tolerance ±{int(RECIPE_V2['length_tol'] * 100)} %, "
          f"insertion at the nearest paragraph boundary (code: blank lines)."]
    L += ["", "## 7. Per-model tokenisation", ""]
    if tok_reports:
        L += ["| model/split | subset | rows | within ±10 % (qwen3) | tokens / qwen3 | needle fully in a window | reconstruction |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for k, rep in sorted(tok_reports.items()):
            for sub, v in rep.get("subsets", {}).items():
                f3 = lambda x: "" if x is None else "%.3f" % x
                L.append("| %s | %s | %s | %s | %s | %s | %s/%s |" % (k, sub, f"{v['n_rows']:,}", f3(v["len_within_tol"]),
                                                                  f3(v["ratio_to_qwen3_mean"]), f3(v["needle_fully_contained"]),
                                                                  v["recon_ok"], v["recon_checked"]))
    L += ["", "## 8. Known limitations", "",
          "* Backgrounds are screened, not annotated: a benign row can contain sensitive background text the two guards missed.",
          "* AgentDojo tool outputs are rendered from the public suite environments; the domain is smaller than the others.",
          "* Fragmented needles are generated on a 256-token gap grid; the disjoint-window ablation must use the same rows.",
          "* Long-needle rows (129–512 tokens) violate the whole-containment premise by construction and are reported separately.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    md = build(P)
    (P.root / "DATACARD.md").write_text(md, encoding="utf-8")
    (P.out / "reports" / "table_A1_draft.md").write_text(md, encoding="utf-8")
    print(f"[datacard] {P.root / 'DATACARD.md'}  ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
