"""Fig 1 — accuracy / cost Pareto (E4-6) and $/1M documents (plan S7 item 9).

    python -m lgs8.fig1                        # reads reports/s5/table2_allL.json + reports/s7/table5_cost.json
    python -m lgs8.fig1 --gpu rtx6000adageneration --usd-per-hour 1.2 --png

x = p50 ms per document (batch 1) at 28k, y = matched-FPR 5 % recall at 28k (position mean), marker size = benign-long FPR at 128k
(default verdict), inset = the same at 128k; $/1M docs = ms/doc x GPU hour price (an assumption, printed with the figure).
Accuracy comes from Table 2 (eval rows), cost from Table 5 (E4 rows); the two name the methods differently, so `match_cost`
maps a Table 2 (method, model) onto the Table 5 entry measured for it -- the LG main config (c16+h8 when measured), the CD
`chunk_or_W_o` policies onto `cdWoO`, and a 32k-native guard's `native` rows at <= 28k onto its `yarn` latency (the same forward).
Outputs reports/s7/fig1_pareto.{json,md} (+ .png with matplotlib).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lgs5.config import S5Paths, split_lg_method

DEFAULT_USD_PER_HOUR = 1.2          # RTX 6000 Ada class, on-demand cloud list price order of magnitude (assumption, stated in the caption)
L_MAIN, L_INSET = 28672, 131072


def _load_json(p: Path) -> Optional[Dict]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _t2_to_t5_candidates(method: str, model: str) -> List[Tuple[str, str]]:
    """Table 5 (method, model) keys to try, most specific first."""
    km = split_lg_method(method)
    if km is not None:
        k, opts = km
        base = f"lg_k{k}" + re.sub(r"_rg\d+$", "", opts)      # the aggregation is free: lg_k32_tp64_rg3 costs what lg_k32_tp64 costs
        return [(f"{base}+c16+h8", model), (f"{base}+c16+h16", model), (f"{base}+c16", model), (base, model)]
    if method == "sliding":
        return [("sliding+c16", model), ("sliding", model)]
    if method.startswith("dense_ft"):
        return [("dense_ft_yarn", model), (method, model)]
    m = re.fullmatch(r"(.+)/chunk_or_(\d+)_(\d+)", method)
    if m:
        return [(f"{m.group(1)}/cd{m.group(2)}o{m.group(3)}", m.group(1))]
    m = re.fullmatch(r"(.+)/(native|yarn|trunc_keep_head|trunc_keep_tail)", method)
    if m:
        g, pol = m.group(1), m.group(2)
        cands = [(f"{g}/{pol}", g)]
        if pol in ("native", "trunc_keep_head", "trunc_keep_tail"):
            cands.append((f"{g}/yarn", g))            # <= 32k: the yarn forward is the native forward with another rope scale
            cands.append((f"{g}/native", g))
        return cands
    return [(method, model)]


def match_cost(method: str, model: str, t5: List[Dict]) -> Optional[Dict]:
    idx = {(e["method"], e["model"]): e for e in t5}
    for key in _t2_to_t5_candidates(method, model):
        if key in idx:
            return idx[key]
    return None


def build(P5: S5Paths, gpu: Optional[str] = None, usd_per_hour: float = DEFAULT_USD_PER_HOUR, png: bool = False) -> Dict:
    t2 = _load_json(P5.reports / "table2_allL.json") or _load_json(P5.reports / "table2_le28k.json")
    t5all = _load_json(P5.out / "reports" / "s7" / "table5_cost.json")
    if not t2:
        raise SystemExit("reports/s5/table2_*.json missing (python s6ctl.py tables)")
    if not t5all:
        raise SystemExit("reports/s7/table5_cost.json missing (python s7ctl.py tables)")
    gpu = gpu or (sorted(t5all, key=lambda g: -len(t5all[g]))[0])
    t5 = t5all.get(gpu) or []
    points: List[Dict] = []
    for r in t2["rows"]:
        e28, e128 = r["per_L"].get(str(L_MAIN), {}), r["per_L"].get(str(L_INSET), {})
        rec28 = e28.get("recall@5pct_pos_mean", e28.get("recall_default_pos_mean"))
        rec128 = e128.get("recall@5pct_pos_mean", e128.get("recall_default_pos_mean"))
        if rec28 is None:
            continue
        c = match_cost(r["method"], r["model"], t5)
        ms28 = c.get(f"p50_ms@{L_MAIN}") if c else e28.get("ms_mean")
        ms128 = c.get(f"p50_ms@{L_INSET}") if c else e128.get("ms_mean")
        if c and r["method"] in ("dense_ft", "dense_ft_tail") and c["method"] == "dense_ft_yarn":
            ms128 = None                      # the truncation policies process 32k tokens at 128k: the yarn cost does not apply there
        if ms28 is None:
            continue
        pt = {"method": r["method"], "model": r["model"], "cost_method": c["method"] if c else None, "cost_source": "E4 p50" if c else "eval ms_mean",
              "ms@28k": ms28, "recall@28k": rec28, "matched": "recall@5pct_pos_mean" in e28, "fpr_bl@128k": e128.get("fpr_default_bl"),
              "fpr_bl@28k": e28.get("fpr_default_bl"), "ms@128k": ms128, "recall@128k": rec128,
              "usd_per_1M_docs@28k": ms28 / 1000.0 / 3600.0 * usd_per_hour * 1e6,
              "usd_per_1M_docs@128k": (ms128 / 1000.0 / 3600.0 * usd_per_hour * 1e6) if ms128 else None,
              "peak_mb@128k": c.get(f"peak_mb@{L_INSET}") if c else None}
        points.append(pt)
    # Pareto front at 28k: no other point is both cheaper and more accurate
    for p in points:
        p["pareto@28k"] = not any((q["ms@28k"] <= p["ms@28k"] and q["recall@28k"] > p["recall@28k"]) or
                                  (q["ms@28k"] < p["ms@28k"] and q["recall@28k"] >= p["recall@28k"]) for q in points if q is not p)
    points.sort(key=lambda p: p["ms@28k"])
    res = {"gpu": gpu, "usd_per_hour": usd_per_hour, "L_main": L_MAIN, "L_inset": L_INSET, "points": points,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rep = P5.out / "reports" / "s7"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "fig1_pareto.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (rep / "fig1_pareto.md").write_text(fig1_md(res), encoding="utf-8")
    if png:
        try:
            draw_png(res, rep / "fig1_pareto.png")
            res["png"] = str(rep / "fig1_pareto.png")
        except Exception as e:                                       # noqa: BLE001  (matplotlib absent on the cluster)
            res["png_error"] = str(e)
    return res


def _num(v, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.{nd}f}"


def fig1_md(res: Dict) -> str:
    lines = [f"# Fig 1 — recall vs cost at 28k (inset 128k), {res['gpu']}, batch 1  ({res['generated']})", "",
             f"$/1M docs = p50 ms/doc x GPU price ({res['usd_per_hour']} $/h assumed).  * = default-verdict recall (no calibration scores).  "
             "pareto = on the 28k accuracy/cost frontier.", "",
             "| method | model | ms@28k | R@28k | $/1M docs@28k | ms@128k | R@128k | FPR bl@128k | peak MB@128k | cost row | pareto |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in res["points"]:
        r28 = _num(p["recall@28k"], 3) + ("" if p["matched"] else "*")
        lines.append(f"| {p['method']} | {p['model']} | {_num(p['ms@28k'])} | {r28} | {_num(p['usd_per_1M_docs@28k'], 1)} | "
                     f"{_num(p.get('ms@128k'))} | {_num(p.get('recall@128k'), 3)} | {_num(p.get('fpr_bl@128k'), 3)} | "
                     f"{_num(p.get('peak_mb@128k'))} | {p.get('cost_method') or p['cost_source']} | {'yes' if p.get('pareto@28k') else ''} |")
    return "\n".join(lines) + "\n"


def draw_png(res: Dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = res["points"]
    for p in pts:
        s = 30 + 400 * float(p.get("fpr_bl@128k") or 0.0)
        is_lg = str(p["method"]).startswith("lg_k")
        ax.scatter(p["ms@28k"], p["recall@28k"], s=s, alpha=0.75, marker="o" if is_lg else "s",
                   edgecolor="black" if p.get("pareto@28k") else "none", linewidth=0.8)
        ax.annotate(f"{p['method']}\n{p['model']}", (p["ms@28k"], p["recall@28k"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(f"p50 ms / document @ {res['L_main'] // 1024}k ({res['gpu']}, batch 1)")
    ax.set_ylabel("matched-FPR 5 % recall @ 28k")
    ax.grid(True, which="both", alpha=0.3)
    ins = ax.inset_axes([0.58, 0.08, 0.4, 0.4])
    for p in pts:
        if p.get("ms@128k") and p.get("recall@128k") is not None:
            ins.scatter(p["ms@128k"], p["recall@128k"], s=15, marker="o" if str(p["method"]).startswith("lg_k") else "s", alpha=0.75)
    ins.set_xscale("log")
    ins.set_title("128k", fontsize=7)
    ins.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fig 1 Pareto + $/1M docs")
    ap.add_argument("--gpu", default=None, help="Table 5 GPU block (default: the one with most rows)")
    ap.add_argument("--usd-per-hour", type=float, default=DEFAULT_USD_PER_HOUR)
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)
    res = build(S5Paths.load().ensure(), a.gpu, a.usd_per_hour, a.png)
    print(f"[fig1] {len(res['points'])} points ({res['gpu']}) -> reports/s7/fig1_pareto.md" + (f" + {res['png']}" if res.get("png") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
