"""Table 4 — the E6 ablation table (plan S8-S9), Fig 10 (k sweep) data.

    python -m lgs8.table4                 # reads eval/v2/e2/*.rows.jsonl (baseline restricted to the ablation subset + arms)
                                          # -> reports/s8/table4_ablation.{json,md}, fig10_k_sweep.json

Every arm and the baseline are measured on the SAME rows (lgs8.config.subset_row): test/main at the 5 ablation depths
(0, .2, .5, .8, 1 -- grid values) x 200 pairs x 8 L, test benign-long, thresholds from calib benign-long (matched-FPR 5 %, E0-3).
An arm evaluated on part of the subset so far is compared with the baseline on the common rows (`partial n/N`).  Columns: recall@28k, recall@128k
(matched, position mean), position-min@128k, FPR of the 1k threshold at 128k on benign-long (length inflation), hit@k@128k,
eps-hat, ms@128k, and the deltas to the baseline.  k = 4/8/16 and the router-gated aggregation are derived from the baseline
rows (lgs5.tables), so the k axis needs only the k=64 and k=all passes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from lgs1.lineage import iter_rows, sha256_file, dedupe_rows
from lgs5.config import S5Paths, split_lg_method
from lgs5.tables import _f, derive_agg, derive_k, table2

from .arms import ARMS, AXES, BASE_BACKBONE, BASE_SEED, Arm, arms_of
from .config import ABL_DEPTHS, ABL_L, ABL_MAX_PAIRS, S8Paths, subset_row

BASE_METHOD = "lg_k32"
DERIVED_K = (4, 8, 16)


def load_subset_rows(P5: S5Paths, models: Optional[Sequence[str]] = None) -> Tuple[List[Dict], Dict[str, str]]:
    """e2 rows restricted to the ablation subset (optionally to some model names -- the base run and the arms)."""
    rows, srcs = [], {}
    want = set(models) if models else None
    for f in sorted(glob.glob(str(P5.eval / "e2" / "*.rows.jsonl"))):
        name = Path(f).name[: -len(".rows.jsonl")]
        model = name.split("__", 1)[1] if "__" in name else ""
        if want is not None and model not in want:
            continue
        rs = [r for r in dedupe_rows(iter_rows(Path(f), tolerant=True)) if subset_row(r)]
        if not rs:
            continue
        srcs[str(f)] = sha256_file(f)
        rows += rs
    return rows, srcs


def _entry(t2_rows: List[Dict], method: str, model: str) -> Optional[Dict]:
    return next((r for r in t2_rows if r["method"] == method and r["model"] == model), None)


def _mid_L(Ls: Sequence[int]) -> int:
    """The 'R@28k' column: 28,672 when it is in the grid, else the middle length (tiny tests)."""
    return 28672 if 28672 in Ls else Ls[len(Ls) // 2]


G2A_MARGIN = 0.05          # lgs6.tables.g2_check: G2 (a) = position-min recall@128k >= position-mean recall@1k - 0.05


def _metrics(e: Optional[Dict], Ls: Sequence[int]) -> Dict:
    if not e:
        return {}
    p1 = e["per_L"].get(str(Ls[0]), {})
    p28 = e["per_L"].get(str(_mid_L(Ls)), {})
    pm = e["per_L"].get(str(Ls[-1]), {})
    r1, pmin = p1.get("recall@5pct_pos_mean"), pm.get("recall@5pct_pos_min")
    return {"R@1k": r1, "R@28k": p28.get("recall@5pct_pos_mean"), "R@128k": pm.get("recall@5pct_pos_mean"),
            "pos_min@128k": pmin, "g2a": (pmin >= r1 - G2A_MARGIN) if (r1 is not None and pmin is not None) else None,
            "fpr@tau1k_bl@128k": pm.get("fpr@tau1k_bl"),
            "fpr_bl@128k": pm.get("fpr_default_bl"), "hit@k@28k": p28.get("hit_at_k"), "hit@k@128k": pm.get("hit_at_k"),
            # the matched threshold itself at 128k: when it sits at the head's ceiling (tau -> 1), recall there counts the
            # needles the head saturates on and different routers tie exactly (E6-3 ell12 / ell16 / ell6_12, 2026-09-28)
            "tau@128k": pm.get("tau@5pct"), "calib_fpr@128k": pm.get("calib_fpr@5pct"),
            # benign twins (same background, benign needle): the default-verdict FPR at 28k and the matched-threshold test FPR
            # (twins + benign-long) at 128k -- a head trained without twins (E6-9 notwin) buys recall with these
            "twin_fpr@28k": p28.get("fpr_default_twin"), "test_fpr@5pct@128k": pm.get("test_fpr@5pct"),
            "eps_hat": e.get("epsilon_hat"), "ms@128k": pm.get("ms_mean"), "ms@28k": p28.get("ms_mean"),
            "n_rows": e.get("n_rows"), "n_128k": pm.get("n"), "n_calib_128k": pm.get("n_calib_benign")}


def _delta(m: Dict, base: Dict) -> Dict:
    out = {}
    for k in ("R@28k", "R@128k", "pos_min@128k", "fpr@tau1k_bl@128k"):
        if m.get(k) is not None and base.get(k) is not None:
            out[f"d_{k}"] = m[k] - base[k]
    return out


def _main_ids(rows: List[Dict]) -> set:
    return {r["id"] for r in rows if r.get("split") == "test" and r.get("subset") == "main" and r.get("id")}


def _of(rows: List[Dict], method: str, model: str) -> List[Dict]:
    return [r for r in rows if r.get("method") == method and r.get("model") == model]


def arm_metrics(rows: List[Dict], t2_rows: List[Dict], method: str, model: str, base_method: str, base_model: str,
                base: Dict, Ls: Sequence[int]) -> Tuple[Dict, Dict, str]:
    """(metrics, deltas, status) of one arm.  An arm whose test/main rows are a strict subset of the baseline's (still being
    evaluated, or evaluated on fewer positions -- the 2026-09-27 arms before their 0.2 / 0.8 rows) is compared with the
    baseline on the COMMON rows: both sides are re-measured on them (thresholds from the shared calib benign-long), and the
    status says how much of the subset the arm covers."""
    arm_rows = _of(rows, method, model)
    if not arm_rows:
        return {}, {}, "no rows"
    ids, base_ids = _main_ids(arm_rows), _main_ids(_of(rows, base_method, base_model))
    if not ids or not base_ids or ids == base_ids or not ids < base_ids:
        m = _metrics(_entry(t2_rows, method, model), Ls)
        st = "ok" if ids == base_ids or not base_ids else f"rows differ from the baseline ({len(ids):,} vs {len(base_ids):,} test/main)"
        return m, _delta(m, base), st
    common = [r for r in _of(rows, base_method, base_model) + arm_rows if r.get("subset") == "benign_long" or r.get("id") in ids]
    t2c = table2(common, Ls)["rows"]
    m = _metrics(_entry(t2c, method, model), Ls)
    bc = _metrics(_entry(t2c, base_method, base_model), Ls)
    return m, _delta(m, bc), f"partial {len(ids):,}/{len(base_ids):,} test/main rows (d vs the baseline on the same rows)"


SEEDS = (0, 1, 2)


def build(P8: S8Paths, Ls: Sequence[int] = ABL_L, backbone: str = BASE_BACKBONE, seed: int = BASE_SEED,
          arms: Optional[Sequence[Arm]] = None, base_k: int = 32, derived_ks: Sequence[int] = DERIVED_K,
          seeds: Sequence[int] = SEEDS) -> Dict:
    P5 = P8.s5
    arms = list(arms) if arms is not None else list(ARMS.values())
    base_run = f"{backbone}_s{seed}"
    base_method = f"lg_k{base_k}"
    seed_runs = [f"{backbone}_s{s}" for s in seeds if s != seed]         # the recipe re-trained: the noise floor of d
    models = {base_run} | {a.run_name(backbone, seed) for a in arms} | set(seed_runs)
    rows, srcs = load_subset_rows(P5, models)
    if not rows:
        return {"rows": [], "sources": srcs}
    dks = [k for k in derived_ks if k < base_k]
    allrows = rows + derive_k(rows, dks) + derive_agg(rows, k=min(base_k, 32))
    t2 = table2(allrows, Ls)["rows"]
    base = _metrics(_entry(t2, base_method, base_run), Ls)
    out_rows: List[Dict] = [{"axis": "baseline", "arm": "base", "method": base_method, "model": base_run, "kind": "base",
                             "desc": f"LG {backbone} s{seed}, k={base_k} (the ablation subset)", **base}]
    for run_s in seed_runs:
        m, d, st = arm_metrics(allrows, t2, base_method, run_s, base_method, base_run, base, Ls)
        if m:
            out_rows.append({"axis": "baseline", "arm": run_s.rsplit("_", 1)[1], "method": base_method, "model": run_s, "kind": "seed",
                             "desc": f"the same recipe trained with seed {run_s.rsplit('_s', 1)[1]} (noise floor of d)", "status": st, **m, **d})
    # derived rows of the baseline (k, aggregation)
    for k in dks:
        m = _metrics(_entry(t2, f"lg_k{k}", base_run), Ls)
        if m:
            out_rows.append({"axis": "E6-1 k", "arm": f"k{k}", "method": f"lg_k{k}", "model": base_run, "kind": "derived",
                             "desc": f"top-{k} routing (from the k={base_k} rows)", **m, **_delta(m, base)})
    rg3_method = f"lg_k{min(base_k, 32)}_rg3"
    m = _metrics(_entry(t2, rg3_method, base_run), Ls)
    if m:
        out_rows.append({"axis": "E6-12 aggregation", "arm": "rg3", "method": rg3_method, "model": base_run, "kind": "derived",
                         "desc": f"router-gated-3 aggregation (from the k={base_k} rows)", **m, **_delta(m, base)})
    seen = {(r["method"], r["model"]) for r in out_rows}
    combo_k = 8 if 8 in dks else (dks[-1] if dks else None)
    for a in arms:
        sp = a.spec(backbone, seed)
        if (sp.method, sp.name) in seen:                     # e.g. a k arm the derivation already covers
            continue
        seen.add((sp.method, sp.name))
        m, d, st = arm_metrics(allrows, t2, sp.method, sp.name, base_method, base_run, base, Ls)
        out_rows.append({"axis": a.axis, "arm": a.name, "method": sp.method, "model": sp.name, "kind": a.kind, "desc": a.desc,
                         "status": st, **m, **d})
        # the arm's own rows re-read with the router-gated aggregation / a smaller k (the G2 (a) candidates combine a lever
        # with the aggregation: tail pad + rg3): derived rows of an lg_k<base_k><options> pass
        km = split_lg_method(sp.method)
        if not m or km is None or km[0] != base_k:
            continue
        opts = km[1]
        for tag, meth, desc in ((f"+rg3", f"lg_k{base_k}{opts}_rg3", "router-gated-3 aggregation of this arm's rows"),
                                (f"+k{combo_k}", f"lg_k{combo_k}{opts}", f"top-{combo_k} of this arm's routed windows")):
            if combo_k is None or (meth, sp.name) in seen:
                continue
            mc, dc, stc = arm_metrics(allrows, t2, meth, sp.name, base_method, base_run, base, Ls)
            if mc:
                seen.add((meth, sp.name))
                out_rows.append({"axis": a.axis, "arm": f"{a.name}{tag}", "method": meth, "model": sp.name, "kind": "derived",
                                 "desc": desc, "status": stc, **mc, **dc})
    # Fig 10 data: recall vs k (+ latency per k from the S7 k sweep when measured)
    fig10 = {"recall": {}, "latency_ms": {}}
    for k, meth in [(k, f"lg_k{k}") for k in sorted(set(list(dks) + [base_k, 64]))] + [("all", "sliding")]:
        e = _entry(t2, meth, base_run)
        if e:
            fig10["recall"][str(k)] = {str(L): e["per_L"].get(str(L), {}).get("recall@5pct_pos_mean") for L in (_mid_L(Ls), Ls[-1])}
    for f in glob.glob(str(P5.out / "cost" / "*" / f"ksweep__lg__{base_run}_k{base_k}*.rows.jsonl")):
        for r in iter_rows(Path(f), tolerant=True):
            if r.get("p50_ms") is not None:
                fig10["latency_ms"].setdefault(Path(f).parent.name, {}).setdefault(str(r.get("k")), {})[str(r.get("L_target"))] = r["p50_ms"]
    res = {"rows": out_rows, "Ls": list(Ls), "base_run": base_run, "sources": srcs, "fig10": fig10,
           "subset": {"depths": list(ABL_DEPTHS), "max_pairs": ABL_MAX_PAIRS, "n_subset_rows": len(rows)}}
    P8.reports.mkdir(parents=True, exist_ok=True)
    (P8.reports / "table4_ablation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (P8.reports / "table4_ablation.md").write_text(table4_md(res), encoding="utf-8")
    (P8.reports / "fig10_k_sweep.json").write_text(json.dumps(fig10, indent=2), encoding="utf-8")
    return res


def _pct(v, nd=3):
    return _f(v, nd)


def _d(v):
    if v is None:
        return ""
    return f"{v:+.3f}"


def table4_md(res: Dict) -> str:
    Ls = res.get("Ls", ABL_L)
    lines = [f"# Table 4 — ablations (E6), LG 0.6B s0 on the ablation subset  ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Subset: test/main at depths {res.get('subset', {}).get('depths')} x {res.get('subset', {}).get('max_pairs')} pairs x {len(Ls)} L "
             f"+ test benign-long; thresholds = matched-FPR 5 % on calib benign-long per (method, L).  R = position-mean recall at the "
             f"matched threshold; pos-min = the worst depth at {Ls[-1] // 1024}k; infl = FPR of the 1k threshold at {Ls[-1] // 1024}k on "
             f"benign-long (length inflation); d = arm - baseline.  G2(a) = pos-min@{Ls[-1] // 1024}k >= R@{Ls[0] // 1024}k - {G2A_MARGIN} "
             f"(lgs6 g2_check).  `no rows` = the arm has not been evaluated yet; `partial n/N` = the arm covers n of the baseline's N "
             f"test/main rows so far and its d is measured against the baseline re-measured on those same rows (so it is not the "
             f"difference to the baseline row above).  `<arm>+rg3` / `+k8` = the arm's own rows re-read with the router-gated aggregation "
             f"/ a smaller k.  `seed` rows = the baseline recipe re-trained with another seed (the noise floor of d).  "
             f"tau@128k = the matched threshold at 128k: near 1 it sits at the head's ceiling and R@128k counts the needles the "
             f"head saturates on (different routers then tie exactly).  twinFPR@28k = default-verdict FPR on the benign twins "
             f"(a head trained without twins buys recall with it).  ms = mean eval-row time on whatever GPU the job ran on "
             f"(indicative only; Table 5 is the cost reference).", "",
             f"| axis | arm | kind | R@{Ls[0] // 1024}k | R@28k | R@128k | dR@128k | pos-min@128k | d pos-min | G2(a) | infl@128k | tau@128k | hit@k@128k | twinFPR@28k | eps-hat | ms@128k | n | note |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["rows"]:
        g2 = "" if r.get("g2a") is None else ("PASS" if r["g2a"] else "FAIL")
        st = r.get("status", "")
        note = (f"{st}; " if st and st != "ok" else "") + r.get("desc", "")
        lines.append(f"| {r['axis']} | {r['arm']} | {r['kind']} | {_pct(r.get('R@1k'))} | {_pct(r.get('R@28k'))} | {_pct(r.get('R@128k'))} | "
                     f"{_d(r.get('d_R@128k'))} | {_pct(r.get('pos_min@128k'))} | {_d(r.get('d_pos_min@128k'))} | {g2} | "
                     f"{_pct(r.get('fpr@tau1k_bl@128k'))} | {_f(r.get('tau@128k'), 4)} | {_pct(r.get('hit@k@128k'))} | {_pct(r.get('twin_fpr@28k'))} | "
                     f"{_pct(r.get('eps_hat'))} | {_f(r.get('ms@128k'), 0)} | {r.get('n_rows') or ''} | {note} |")
    lines += ["", "Sources (sha256):"] + [f"- `{Path(k).name}` {v[:12]}" for k, v in res.get("sources", {}).items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S8 Table 4 (ablations)")
    ap.add_argument("--backbone", default=BASE_BACKBONE)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--L", default=None, help="comma list (default: the 8 lengths)")
    a = ap.parse_args(argv)
    P8 = S8Paths.load().ensure()
    Ls = [int(x) for x in a.L.split(",")] if a.L else ABL_L
    res = build(P8, Ls, a.backbone, a.seed, arms_of([x for x in a.arms.split(",") if x]))
    print(f"[table4] {len(res['rows'])} rows -> {P8.reports / 'table4_ablation.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
