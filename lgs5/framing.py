"""Framing diagnostic (2026-09-20) -- where in its window does the needle sit, and does the head care?  No GPU.

The head is a probe on the LAST valid token of a 256-token window.  The oracle rows keep, for every window that wholly
contains the needle, the head probability (`deep_probs`, aligned with `needle_windows`) and the needle's token span, so
we can bin the windows by the number of background tokens AFTER the needle (window end - needle end) and see whether the
probability falls as the needle moves away from the probe position.  If it does, the framing loss (E2-7) is a pooling
problem -- a token-level head (score every position, max inside the window, the router's own pooling) fixes it -- rather
than a context problem.

    python -m lgs5.framing            # -> reports/s5/framing_by_offset_le28k.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Dict, List, Sequence

from lgs1.lgcore.windows import make_windows
from lgs4.config import WINDOW_B

from .config import E2_L_LE28K, S5Paths, parse_Ls

BINS = [(0, 16, "0-15"), (16, 64, "16-63"), (64, 128, "64-127"), (128, 10 ** 9, "128+")]


def _bin(tail: int) -> str:
    for lo, hi, name in BINS:
        if lo <= tail < hi:
            return name
    return "128+"


def by_offset(rows: List[Dict], Ls: Sequence[int], tau: float = 0.5) -> Dict:
    """{run: {L: {bin: {n, mean_prob, frac_gt_tau}}, 'all': {...}, 'rows': {bin: {n, share, recall}}}} -- window level over
    the harmful oracle windows, and row level: each harmful row binned by the LARGEST tail among its containing windows
    (the oracle takes the max over them), recall = share of rows whose max window probability > tau."""
    out: Dict = {}
    rowlevel: Dict = {}
    twins: Dict = {}
    for r in rows:
        if r.get("method") != "oracle" or not r.get("deep_probs") or not r.get("needle_tok_span"):
            continue
        if r.get("split") != "test" or r.get("L_target") not in Ls:
            continue
        run = r["model"]
        wins = make_windows(int(r["n_tok"]), WINDOW_B)
        s, e = r["needle_tok_span"]
        if r.get("label") == 0:                    # benign twin windows: does the score also rise with trailing context?
            for wi, p in zip(r.get("needle_windows") or [], r["deep_probs"]):
                if wi >= len(wins):
                    continue
                d = twins.setdefault(run, {}).setdefault(_bin(int(wins[wi][1] - e)), {"n": 0, "sum_p": 0.0, "n_gt": 0})
                d["n"] += 1
                d["sum_p"] += float(p)
                d["n_gt"] += int(float(p) > tau)
            continue
        tails = []
        for wi, p in zip(r.get("needle_windows") or [], r["deep_probs"]):
            if wi >= len(wins):
                continue
            ws, we = wins[wi]
            tail = int(we - e)                     # background tokens after the needle inside this window
            tails.append(tail)
            b = _bin(tail)
            for key in (str(r["L_target"]), "all"):
                d = out.setdefault(run, {}).setdefault(key, {}).setdefault(b, {"n": 0, "sum_p": 0.0, "n_gt": 0})
                d["n"] += 1
                d["sum_p"] += float(p)
                d["n_gt"] += int(float(p) > tau)
        if tails:
            b = _bin(max(tails))
            d = rowlevel.setdefault(run, {}).setdefault(b, {"n": 0, "n_gt": 0, "n_len_gt128": 0})
            d["n"] += 1
            d["n_gt"] += int(max(float(x) for x in r["deep_probs"]) > tau)
            d["n_len_gt128"] += int(e - s > 128)
    for run in out:
        for key in out[run]:
            for b, d in out[run][key].items():
                d["mean_prob"] = d["sum_p"] / d["n"]
                d["frac_gt_tau"] = d["n_gt"] / d["n"]
                del d["sum_p"], d["n_gt"]
        rl = rowlevel.get(run, {})
        tot = sum(d["n"] for d in rl.values()) or 1
        out[run]["rows"] = {b: {"n": d["n"], "share": d["n"] / tot, "recall": d["n_gt"] / d["n"], "frac_needle_gt128": d["n_len_gt128"] / d["n"]}
                            for b, d in rl.items()}
        out[run]["twins"] = {b: {"n": d["n"], "mean_prob": d["sum_p"] / d["n"], "frac_gt_tau": d["n_gt"] / d["n"]}
                             for b, d in twins.get(run, {}).items()}
    return out


def md(d: Dict, tau: float = 0.5) -> str:
    lines = [f"# Framing diagnostic -- head probability of the needle window vs background tokens after the needle  ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Harmful test rows, every window that wholly contains the needle (oracle rows).  tail = window end - needle end (tokens).  "
             f"frac>tau = share of windows with head probability > {tau}.  9/20 finding: the 0-15 bin is the WORST (0.72 vs 0.85) -- "
             "the probe wants 16+ tokens after the needle; trailing background does not hurt.", "",
             "| run | L | tail 0-15: n / mean p / frac>tau | 16-63 | 64-127 | 128+ |", "| --- | --- | --- | --- | --- | --- |"]
    for run, per in d.items():
        keys = sorted([k for k in per if k not in ("all", "rows", "twins")], key=int) + ["all"]
        for key in keys:
            cells = []
            for _, _, name in BINS:
                x = per[key].get(name)
                cells.append(f"{x['n']} / {x['mean_prob']:.3f} / {x['frac_gt_tau']:.3f}" if x else "")
            lines.append(f"| {run} | {key} | " + " | ".join(cells) + " |")
    lines += ["", "Row level: each harmful row by the LARGEST tail among its containing windows (what the oracle max sees).  "
              "n / share / recall@tau / share of needles longer than 128 tokens.  A row stuck in the 0-15 bin has NO containing window "
              "with trailing context -- a geometry (stride) effect, not a pooling one.", "",
              "| run | max-tail 0-15 | 16-63 | 64-127 | 128+ |", "| --- | --- | --- | --- | --- |"]
    for run, per in d.items():
        cells = []
        for _, _, name in BINS:
            x = per.get("rows", {}).get(name)
            cells.append(f"{x['n']} / {x['share']:.3f} / {x['recall']:.3f} / {x['frac_needle_gt128']:.2f}" if x else "")
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    lines += ["", "Benign twin windows (label 0) by the same tail: n / mean p / frac>tau.  If the twin score also rises with trailing "
              "context, the E2-7 'benign side' is the same probe-position effect (needle-only inputs are all tail 0), not the document "
              "making benign text look harmful.", "",
              "| run | twin tail 0-15 | 16-63 | 64-127 | 128+ |", "| --- | --- | --- | --- | --- |"]
    for run, per in d.items():
        cells = []
        for _, _, name in BINS:
            x = per.get("twins", {}).get(name)
            cells.append(f"{x['n']} / {x['mean_prob']:.3f} / {x['frac_gt_tau']:.3f}" if x else "")
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build(P: S5Paths, Ls: Sequence[int]) -> str | None:
    from .tables import load_eval
    rows, _ = load_eval(P, "e2")
    d = by_offset(rows, Ls)
    if not d:
        return None
    tag = "le28k" if max(Ls) <= 28672 else "allL"
    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / f"framing_by_offset_{tag}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    out = P.reports / f"framing_by_offset_{tag}.md"
    out.write_text(md(d), encoding="utf-8")
    return str(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="framing diagnostic from the oracle rows")
    ap.add_argument("--L", default="le28k")
    a = ap.parse_args(argv)
    P = S5Paths.load().ensure()
    out = build(P, parse_Ls(a.L) or E2_L_LE28K)
    print(f"[framing] {out or 'no oracle rows'}")
    if out:
        print(open(out, encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
