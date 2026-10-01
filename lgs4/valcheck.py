"""T1/T3 checkpoint check — router hit@k on val (all L) and calib @28k, matched-FPR 5 % recall @28k (plan S4 item 1).

    python -m lgs4.valcheck --backbone qwen3guard_0.6b --seed 0

Reads runs/v2/<bb>_s<seed>/{head.pt, router.pt} and the prepped val/main, calib/main (L=28672), calib/benign_long rows.
    hit@{8,16,32} per L        val/main harmful rows: a needle window among the top-k router windows (route() only)
    hit@{8,16,32} @28k         calib/main harmful rows at L=28672
    recall@5pct / @1pct @28k   full LG forward (k=32) on calib/main 28k rows; thresholds from calib/benign_long 28k rows
                               (lgs1.metrics.summarize_matched: strict `score > tau`, achieved FPR printed)
Pass (T1): hit@8 @28k >= 0.95 (round-3 reference 0.96-1.00).  Output val_report.json in the run dir + reports/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from lgs1.lgcore.model import LGConfig, LandmarkGuard, pool_scores
from lgs1.lgcore.windows import windows_containing
from lgs1.metrics import summarize_matched
from lgs4 import backbone as BB
from lgs4.common import device_auto, run_meta
from lgs4.config import CALIB_L, CALIB_SETS, HIT_PASS, LG_K_DEFAULT, LG_K_REPORT, VAL_SETS, S4Paths, backbone
from lgs4.prep import PrepSet
from lgs4.config import WINDOW_B
from lgs4.router_train import row_windows, trunk_features

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def hits_on(model, router, ps: PrepSet, rows: List[int], ell, pad_id: int, device: str, ks=LG_K_REPORT, pool: str = "max",
            b: int = WINDOW_B) -> Dict:
    """hit@k of the router alone; `ell` may be the list of depths the router reads (S8 E6-3), `pool` / `b` its cfg."""
    per_L: Dict = {}
    for r in rows:
        m = ps.meta[r]
        ids, valid, wins = row_windows(ps, r, pad_id, b)
        h = trunk_features(model, ids.to(device), valid.to(device), ell)
        sc = pool_scores(router(h.float()), valid.to(device), pool)
        order = torch.argsort(sc, descending=True).tolist()
        nw = windows_containing((m["needle_tok_span"][0], m["needle_tok_span"][1]), wins)
        rec = per_L.setdefault(m["L_target"], {f"hit@{k}": 0 for k in ks} | {"n": 0})
        rec["n"] += 1
        for k in ks:
            rec[f"hit@{k}"] += int(any(w in order[:k] for w in nw))
    out = {str(L): {f"hit@{k}": v[f"hit@{k}"] / v["n"] for k in ks} | {"n": v["n"]} for L, v in sorted(per_L.items(), key=lambda kv: kv[0] or 0)}
    n = sum(v["n"] for v in per_L.values())
    out["all"] = {f"hit@{k}": sum(v[f"hit@{k}"] for v in per_L.values()) / max(1, n) for k in ks} | {"n": n}
    return out


@torch.no_grad()
def score_rows(lg: LandmarkGuard, ps: PrepSet, rows: List[int], device: str, method: str) -> List[Dict]:
    out = []
    for r in rows:
        m = ps.meta[r]
        ids = torch.from_numpy(ps.row_ids(r)).to(device)
        o = lg(ids)
        sel = [int(i) for i in o.selected.tolist()]
        nw = windows_containing((m["needle_tok_span"][0], m["needle_tok_span"][1]), o.windows) if m.get("needle_tok_span") else []
        out.append({"id": m["id"], "split": ps.split, "subset": ps.subset, "L_target": m["L_target"], "label": m["label"], "method": method,
                    "score": o.score(), "verdict": bool(o.verdict_unsafe), "hit": (any(w in sel for w in nw) if nw else None)})
    return out


def run(P: S4Paths, bb_name: str, seed: int, device: str = "auto", k: int = LG_K_DEFAULT, max_val_rows: int = 2000,
        max_calib_rows: Optional[int] = None, repo_override: Optional[str] = None, calib_L: int = CALIB_L,
        dtype: torch.dtype = torch.bfloat16, variant: str = "") -> Dict:
    device = device_auto(device)
    run_dir = P.run_dir(bb_name, seed, variant)
    model, tok, head, router, cfg, info = BB.load_lg(run_dir, device=device, k=k, repo_override=repo_override, dtype=dtype)
    if router is None:
        raise SystemExit(f"{run_dir}/router.pt missing (train the router first)")
    lg = LandmarkGuard(model, cfg, router, head).to(device).eval()
    rng = random.Random(seed)
    t0 = time.time()
    rep: Dict = {"backbone": bb_name, "seed": seed, "k": k, "cfg": cfg.to_dict(), "run_dir": str(run_dir)}
    # ---- val hit@k (all L) ----
    val = PrepSet(P, *VAL_SETS[0])
    vrows = [i for i, m in enumerate(val.meta) if m["label"] == 1 and m.get("needle_tok_span")]
    rng.shuffle(vrows)
    rep["val_hit"] = hits_on(model, router, val, sorted(vrows[:max_val_rows]), cfg.router_depths, cfg.pad_id, device, pool=cfg.pool, b=cfg.b)
    print(f"[valcheck] val hit@k: {rep['val_hit']['all']}  ({time.time() - t0:.0f}s)", flush=True)
    # ---- calib @28k ----
    try:
        cm, cb = PrepSet(P, *CALIB_SETS[0]), PrepSet(P, *CALIB_SETS[1])
    except SystemExit as e:
        print(f"[valcheck] {e} -> calib @{calib_L} skipped")
        cm = cb = None
    if cm is not None:
        crows = [i for i, m in enumerate(cm.meta) if m["L_target"] == calib_L]
        if max_calib_rows:
            rng.shuffle(crows)
            crows = sorted(crows[:max_calib_rows])
        harm = [i for i in crows if cm.meta[i]["label"] == 1 and cm.meta[i].get("needle_tok_span")]
        rep["calib_hit"] = hits_on(model, router, cm, harm, cfg.router_depths, cfg.pad_id, device, pool=cfg.pool, b=cfg.b)
        print(f"[valcheck] calib @{calib_L} hit@k: {rep['calib_hit']['all']}  ({time.time() - t0:.0f}s)", flush=True)
        brows = [i for i, m in enumerate(cb.meta) if m["L_target"] == calib_L]
        if max_calib_rows:
            brows = brows[:max(50, max_calib_rows // 4)]
        test_rows = score_rows(lg, cm, crows, device, "lg")
        calib_rows = score_rows(lg, cb, brows, device, "lg")
        tab = summarize_matched(test_rows, calib_rows, group_keys=("method", "L_target"))
        rep["matched"] = tab[0] if tab else None
        rep["calib_fpr_default"] = sum(1 for r in calib_rows if r["verdict"]) / max(1, len(calib_rows))
        print(f"[valcheck] matched-FPR @{calib_L}: {rep['matched']}  ({time.time() - t0:.0f}s)", flush=True)
    h8 = (rep.get("calib_hit") or rep["val_hit"])["all"].get("hit@8")
    rep["hit8_at_28k"] = h8
    rep["pass"] = bool(h8 is not None and h8 >= HIT_PASS)
    rep["seconds"] = round(time.time() - t0, 1)
    rep["meta"] = run_meta(ROOT)
    (run_dir / "val_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / f"s4_valcheck_{run_dir.name}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"[valcheck] {'PASS' if rep['pass'] else 'FAIL'} hit@8@{calib_L}={h8} -> {run_dir / 'val_report.json'}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S4 checkpoint check (hit@k, matched-FPR @28k)")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=LG_K_DEFAULT)
    ap.add_argument("--max-val-rows", type=int, default=2000)
    ap.add_argument("--variant", default="", help="run-dir suffix (see head_train --variant)")
    ap.add_argument("--max-calib-rows", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args(argv)
    P = S4Paths.load().ensure().for_backbone(a.backbone)      # prep arrays of the backbone's tokenizer family (T5: llama3)
    rep = run(P, a.backbone, a.seed, a.device, a.k, a.max_val_rows, a.max_calib_rows, a.repo, variant=a.variant)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
