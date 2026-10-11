"""Fig. 9(a): hit@k versus k for successful attacks (X = S), per chunk definition, with the
random baseline (SPEC 6.1). Source: hits.jsonl of the B0 run that defines S (E0.1).

python plots/fig09.py --run-id latest            # $AG_OUT/raw/E0.1/latest.txt
python plots/fig09.py --run-id <mock run> --allow-mock --exp mock
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import _common as C  # noqa: E402

from ag.harness.writer import read_json  # noqa: E402
from ag.metrics.quality import random_baseline  # noqa: E402

CHUNK_DEFS = ("first8", "first32", "all", "union")
STYLE = {"first8": "o-", "first32": "s-", "all": "^-", "union": "D-"}


def hit_curve(hit_rows, S: set[str], variant_id: str, chunk_def: str) -> list[dict]:
    out = []
    ks = sorted({int(r["k"]) for r in hit_rows if r["variant_id"] == variant_id})
    for k in ks:
        sel = [r for r in hit_rows if r["variant_id"] == variant_id and r["chunk_def"] == chunk_def
               and int(r["k"]) == k and r["site_rule"] == "single" and r["request_id"] in S]
        num = sum(1 for r in sel if r["hit"])
        rb = random_baseline(hit_rows, chunk_def=chunk_def, k=k, units=S, variant_id=variant_id)
        out.append({"chunk_def": chunk_def, "k": k, "num": num, "den": len(sel),
                    "hit": num / len(sel) if sel else float("nan"), "random": rb if rb is not None else float("nan")})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fig09")
    C.add_common_args(ap, default_exp="E0.1")
    ap.add_argument("--variant", default=None, help="variant_id (default: the run's aggregate-C variant)")
    args = ap.parse_args(argv)
    runs = C.load_runs(args)
    out_root = Path(args.out)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.2), squeeze=False)
    for ax, (run_dir, lin) in zip(axes[0], runs):
        hits = C.rows(run_dir, "hits.jsonl")
        if not hits:
            raise C.RunRejected(f"{lin['run_id']}: no hits.jsonl (Fig. 9 needs the B0 run with Records)")
        outcomes = C.rows(run_dir, "outcomes.jsonl")
        S = {o["unit_id"] for o in outcomes if o.get("attack_success_b0") and o.get("error_kind") is None}
        summary = read_json(run_dir / "summary.json")
        vid = args.variant or summary["_meta"]["default_variant_id"]
        table = []
        for cd in CHUNK_DEFS:
            cur = hit_curve(hits, S, vid, cd)
            table += cur
            ks = [r["k"] for r in cur]
            ax.plot(ks, [r["hit"] for r in cur], STYLE[cd], label=cd, ms=4)
        rnd = [r for r in table if r["chunk_def"] == "union"]
        ax.plot([r["k"] for r in rnd], [r["random"] for r in rnd], "k:", label="random (union)")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("k (windows judged per chunk)")
        ax.set_ylabel("hit@k (successful attacks)")
        parts = lin["run_id"].split("__")
        ax.set_title(f"{parts[0]} {parts[1]} {parts[2]} {parts[4]} |S|={len(S)}", fontsize=9)
        ax.axhline(0.9, color="grey", lw=0.6, ls="--")
        ax.grid(alpha=0.3)
        C.write_csv(out_root, "fig09", lin["run_id"], "hit_at_k.csv", table)
        if np.isnan([r["hit"] for r in table]).all():
            raise C.RunRejected(f"{lin['run_id']}: S is empty, nothing to plot")
    axes[0][0].legend(fontsize=7, loc="lower right")
    p = C.save_figure(fig, out_root, "fig09", [l["run_id"] for _, l in runs], args.fig_dir)
    print(p)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except C.RunRejected as e:
        print(f"fig09: {e}", file=sys.stderr)
        raise SystemExit(1)
