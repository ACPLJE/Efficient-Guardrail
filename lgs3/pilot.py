"""E0-5 — tiny-backbone pilot: every LG runner path on the ~200 dry-run rows, schema + lineage checks.

    python -m lgs3.pilot                     # rows under $LG_OUT/pilot/, report reports/e05_pilot.json
    python -m lgs3.pilot --tok <tiny tokenizer dir>   (tests)

Runner paths exercised (each writes lineage-stamped rows):
    E1  native short prompts through the LG scorer            pilot/e1_native.rows.jsonl
    E2  main set (needle rows): score, routed windows, hit@k   pilot/e2_main.rows.jsonl
    E3  benign-long rows: FPR side                             pilot/e3_benign_long.rows.jsonl
    B5  sliding scan (k = all) on the main rows                pilot/b5_sliding.rows.jsonl
    E7  variant rows present in synth_dryrun/test/* (if any)   pilot/e7_<variant>.rows.jsonl
    E7-7 router evasion (random-token prefix baseline; GCG replaces the proposal in S10)  pilot/e77_evasion.rows.jsonl
    matched-FPR summary with lgs1.metrics (benign-long rows stand in for the calibration split in the pilot)
The backbone is a randomly initialised 4-layer Qwen3 with the real Qwen3 vocabulary; numbers mean nothing —
only the plumbing is under test (report §E0-5: 'error 0, output schema identical').
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from lgs1.lgcore.attack import random_search, run_attack, to_row
from lgs1.lgcore.model import LGConfig, tiny_qwen3
from lgs1.lineage import LINEAGE_KEYS, RowsWriter
from lgs2.config import S2Paths
from lgs2.synth import read_rows
from lgs2.tok import HFTok, token_span_for_chars
from lgs3.lgeval import ROW_KEYS, LGRunner, lineage_for, read_eval_rows, run_rows, summarize


def build_tiny_runner(tok, cfg: Optional[LGConfig] = None, seed: int = 0, method: str = "lg_tiny", device: str = "cpu",
                      sliding: bool = False) -> LGRunner:
    vocab = len(tok.tok)
    model = tiny_qwen3(seed=seed, n_layers=4, hidden=64, vocab=vocab).to(device)
    cfg = cfg or LGConfig(b=128, ell=2, k=8, pool="max", tau=0.5, pad_id=tok.tok.pad_token_id or 0)
    return LGRunner.from_modules(model, tok, cfg, method=method, device=device, sliding=sliding)


def _check_rows(rows: List[Dict], need_hit: bool) -> List[str]:
    problems = []
    if not rows:
        return ["no rows"]
    for k in ROW_KEYS + list(LINEAGE_KEYS):
        if any(k not in r for r in rows):
            problems.append(f"missing key {k}")
    if any(r["score"] is None or (isinstance(r["score"], float) and math.isnan(r["score"])) for r in rows):
        problems.append("NaN/None scores")
    if need_hit and any(r["label"] == 1 and r["hit"] is None for r in rows):
        problems.append("harmful row without hit@k (needle span not inside any window?)")
    ids = [(r["split"], r["subset"], r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    return problems


def run(P: S2Paths, tok, rows_root: Optional[Path] = None, device: str = "cpu", n_attack: int = 3, attack_steps: int = 5) -> Dict:
    rows_root = rows_root or P.synth_dryrun
    out_dir = P.out / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = build_tiny_runner(tok, device=device)
    lin = lineage_for(None, None, runner.lg.cfg, runner.method, recipe_hash="dryrun")
    report: Dict = {"device": device, "cfg": runner.lg.cfg.to_dict(), "paths": {}, "checks": {}, "timing_ms": {}}

    def stage(name: str, rows: List[Dict], r: LGRunner, need_hit: bool):
        path = out_dir / f"{name}.rows.jsonl"
        if path.exists():
            path.unlink()
        t0 = time.time()
        n = run_rows(r, rows, path, lin, resume=False, log_every=1000)
        out_rows = read_eval_rows(path)
        report["paths"][name] = str(path)
        report["checks"][name] = {"n": n, "problems": _check_rows(out_rows, need_hit)}
        report["timing_ms"][name] = round(1000 * (time.time() - t0) / max(1, n), 1)
        print(f"[pilot] {name:16s} rows={n:4d}  {report['timing_ms'][name]} ms/row  problems={report['checks'][name]['problems'] or 'none'}", flush=True)
        return out_rows

    main_rows = read_rows(rows_root / "test" / "main" / "rows.jsonl.gz")
    bl_rows = read_rows(rows_root / "test" / "benign_long" / "rows.jsonl.gz") if (rows_root / "test" / "benign_long" / "rows.jsonl.gz").exists() else []
    nat_rows = read_rows(rows_root / "train" / "native" / "rows.jsonl.gz") if (rows_root / "train" / "native" / "rows.jsonl.gz").exists() else []
    e2 = stage("e2_main", main_rows, runner, need_hit=True)
    e3 = stage("e3_benign_long", bl_rows, runner, need_hit=False) if bl_rows else []
    if nat_rows:
        stage("e1_native", nat_rows, runner, need_hit=False)
    slide = LGRunner(runner.lg, tok, "sliding_tiny", device, sliding=True)
    stage("b5_sliding", main_rows[: min(len(main_rows), 40)], slide, need_hit=True)
    for vf in sorted(glob.glob(str(rows_root / "test" / "*" / "rows.jsonl.gz"))):
        name = Path(vf).parent.name
        if name in ("main", "benign_long"):
            continue
        stage(f"e7_{name}", read_rows(Path(vf))[:40], runner, need_hit=False)
    # E7-7: random-token prefix evasion on a few harmful rows (end-to-end verdict, theta on the route() scale)
    att_path = out_dir / "e77_evasion.rows.jsonl"
    if att_path.exists():
        att_path.unlink()
    n_att = 0
    with RowsWriter(att_path, lin) as w:
        for r in [x for x in main_rows if x["label"] == "harmful"][:n_attack]:
            ids, offs = tok.encode_with_offsets(r["text"])
            span = token_span_for_chars(offs, tuple(r["needle_char_span"]))
            doc = torch.tensor(ids, dtype=torch.long, device=device)
            try:
                adv, after = random_search(runner.lg, doc, span, n_adv=8, steps=attack_steps, slot="prefix", seed=0)
                before, _ = run_attack(runner.lg, doc, span, adv, "prefix")
                w.write({"id": r["id"], "split": r["split"], "subset": "e77", **to_row(before, after, {"slot": "prefix", "steps": attack_steps,
                         "theta_scale": "route()", "L_target": r["L_target"], "label": 1})})
                n_att += 1
            except ValueError as e:      # needle not wholly inside a window (never for the main set)
                w.write({"id": r["id"], "split": r["split"], "subset": "e77", "error": str(e), "label": 1})
    report["paths"]["e77_evasion"] = str(att_path)
    report["checks"]["e77_evasion"] = {"n": n_att, "problems": [] if n_att == min(n_attack, sum(1 for x in main_rows if x["label"] == "harmful")) else ["attack rows missing"]}
    print(f"[pilot] e77_evasion      rows={n_att}", flush=True)
    # summary (matched-FPR): the pilot has no calibration split -> benign-long rows stand in
    summ = summarize(e2, e3 or [r for r in e2 if r["label"] == 0])
    report["summary_rows"] = len(summ["table"])
    report["summary_example"] = summ["table"][0] if summ["table"] else None
    report["pass"] = all(not v["problems"] for v in report["checks"].values())
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (P.out / "reports").mkdir(parents=True, exist_ok=True)
    (P.out / "reports" / "e05_pilot.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E0-5 tiny pilot")
    ap.add_argument("--tok", default="qwen3", help="qwen3 | a tokenizer directory")
    ap.add_argument("--rows-root", default=None, help="default: synth_dryrun/")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--attack-steps", type=int, default=5)
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    from lgs2.config import TOKENIZERS
    tok = HFTok(TOKENIZERS.get(a.tok, a.tok), a.tok)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    rep = run(P, tok, Path(a.rows_root) if a.rows_root else None, dev, attack_steps=a.attack_steps)
    print(f"[pilot] {'PASS' if rep['pass'] else 'FAIL'} -> {P.out / 'reports' / 'e05_pilot.json'}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
