"""Layer curves, l95 per contrast, cluster-bootstrap CIs, the E0.3 gate, figures and report.md.

Contrast = (task, negative set), e.g. (harm, all) or (inj, instr).  Positives are the task's positive test windows;
negatives are the test windows of the negative set's classes plus, when scored, the big negative pool.
Every metric is computed per seed and averaged; l95 is read off the seed-mean curve.

Bootstrap: groups (snippet templates / near-duplicate clusters / source documents) are resampled with replacement over
the whole test set (+ big pool), the same replicate weights apply to every contrast, layer and seed, so the l95 ratio of
two contrasts gets a paired CI.

Gate (plan §8.2 E0.3): l95(inj vs instruction-heavy negatives) >= 1.5 x l95(harm vs all negatives), on the primary
metric TPR at window FPR 2e-4 (PRIMARY_FPR, plan §8.2 / §9.1 / SPEC 6.5).  When the negatives cannot resolve 2e-4
(< 5 expected false positives) the verdict is INCONCLUSIVE -- the metric is never switched silently; --primary-fpr
overrides it explicitly and the report says so.  The ratio uses max(l95_harm, 1) so a harm curve that is already
saturated at the embeddings does not divide by zero (DECISIONS D-041); the relative-depth gap (l95_inj - l95_harm) / L
and the compute ratio l95 / L (SPEC 6.5) are reported next to it.

    python -m agdepth.analyze --run NAME [--primary-fpr 2e-4] [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import COMPARISONS, FPR_TARGETS, GATE, L95_FRAC, PRIMARY_FPR, TASKS, AGPaths
from . import metrics as MX

CONTRAST_STYLE = [   # (task, negset, label) in fixed categorical order -> colours never depend on what is present
    ("harm", "all", "harmful vs all benign"),
    ("inj", "instr", "injection vs instruction-heavy"),
    ("inj", "plain", "injection vs plain"),
    ("inj", "notinject", "injection vs NotInject"),
]


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
def discover(R) -> List[Tuple[str, str, str]]:
    out = []
    for f in sorted((R.root / "probes").glob("*/*/*.test_scores.npz")):
        out.append((f.parent.parent.name, f.parent.name, f.name.split(".")[0]))
    return out


def _bigneg_ready(R, bb, framing, pool, dsha: str) -> Tuple[bool, str]:
    """(usable, reason) for the big-pool scores of one combo."""
    D = R.probes(bb, framing)
    prog_p = D / f"{pool}.bigneg_progress.json"
    if not prog_p.exists():
        return False, "not scored (run score / pipeline)"
    prog = json.loads(prog_p.read_text())
    sha = hashlib.sha1((D / f"{pool}.probes.npz").read_bytes()).hexdigest()[:16]
    if prog.get("data_manifest_sha") != dsha:
        return False, "scored on another dataset"
    if prog.get("probes_sha") != sha:
        return False, "stale (probes re-fit after scoring)"
    if not prog.get("complete"):
        return False, f"incomplete ({len(prog.get('done', []))} shards done)"
    return True, "ok"


def _bigneg_scores(R, bb, framing, pool, n_tasks, n_seeds, L1):
    D = R.probes(bb, framing)
    ids = json.loads((D / "bigneg_ids.json").read_text())
    mm = np.memmap(D / f"{pool}.bigneg_scores.f32", dtype=np.float32, mode="r", shape=(len(ids), n_tasks, n_seeds, L1))
    return mm, ids


class Contrast:
    """Positive / negative score tensors [S, n, L] of one (task, negset) for one combo, plus group indices."""

    def __init__(self, task: str, negset: str, Sp, Sn, gp, gn, n_neg_big: int):
        self.task, self.negset = task, negset
        self.Sp, self.Sn, self.gp, self.gn, self.n_neg_big = Sp, Sn, gp, gn, n_neg_big

    @property
    def key(self) -> str:
        return f"{self.task}/{self.negset}"


def build_contrasts(test_scores: np.ndarray, tasks: Sequence[str], te_cls, te_gidx, big=None, big_cls=None,
                    big_gidx=None, te_heldout=None) -> Dict[str, Contrast]:
    """test_scores [T, S, n_test, L]; big [n_big, T, S, L] or None.  te_heldout (bool mask, group split mode only):
    test windows of a test_only source (LLMail-Inject).  They are kept out of the regular contrasts -- the gate compares
    in-distribution curves on both sides -- and get their own `<task>@heldout/<negset>` contrasts (plan E6.3)."""
    out = {}
    for ti, t in enumerate(tasks):
        task = TASKS[t]
        pos_all = np.isin(te_cls, list(task.pos))
        ho = te_heldout if te_heldout is not None else np.zeros(len(te_cls), bool)
        for tag, pos in ((t, pos_all & ~ho), (f"{t}@heldout", pos_all & ho)):
            for ns, classes in task.eval_neg.items():
                neg = np.isin(te_cls, list(classes)) & ~ho
                if not pos.any() or not neg.any():
                    continue
                Sp = test_scores[ti][:, pos, :]
                Sn = test_scores[ti][:, neg, :]
                gn = te_gidx[neg]
                nb = 0
                if big is not None:
                    bm = np.where(np.isin(big_cls, list(classes)))[0]
                    if bm.size:
                        Sb = np.asarray(big[bm, ti]).transpose(1, 0, 2)          # [S, nb, L]
                        Sn = np.concatenate([Sn, Sb], axis=1)
                        gn = np.concatenate([gn, big_gidx[bm]])
                        nb = bm.size
                out[f"{tag}/{ns}"] = Contrast(tag, ns, Sp, Sn, te_gidx[pos], gn, nb)
    return out


# ----------------------------------------------------------------------------------------------
# per-combo analysis
# ----------------------------------------------------------------------------------------------
PEAK_FLOOR = {"tpr": 0.2, "auroc": 0.7}     # below this peak the curve is too flat for its l95 to mean anything


def l95_of(metric: str, curve) -> float:
    """l95 on the metric's informative scale: TPR as is, AUROC chance-corrected (2*AUC - 1, floored at 0) -- otherwise
    the 0.5 floor makes '95 % of the maximum' trivially reachable."""
    c = np.asarray(curve, np.float64)
    if metric.startswith("auroc"):
        c = np.clip(2.0 * c - 1.0, 0.0, None)
    return MX.ell95(c, L95_FRAC)


def curves_of(c: Contrast, targets: Sequence[float]) -> Dict[str, np.ndarray]:
    per_seed = [MX.curves(c.Sp[s], c.Sn[s], targets) for s in range(c.Sp.shape[0])]
    out = {k: np.mean([p[k] for p in per_seed], 0) for k in per_seed[0]}
    out["_per_seed_l95"] = {k: [l95_of(k, p[k]) for p in per_seed] for k in per_seed[0] if not k.startswith("fpr")}
    out["_per_seed"] = per_seed
    return out


def gate_n_neg(te_cls, big_cls, with_big: bool) -> int:
    """Smallest negative count among the two gate contrasts (test windows + big pool when it is scored)."""
    ns = []
    for t, negset in (GATE["shallow"], GATE["deep"]):
        classes = list(TASKS[t].eval_neg[negset])
        n = int(np.isin(te_cls, classes).sum())
        if with_big and big_cls is not None:
            n += int(np.isin(big_cls, classes).sum())
        ns.append(n)
    return min(ns)


def choose_primary(n_min: int, explicit: Optional[float]) -> Tuple[float, str]:
    """The plan's TPR @ 2e-4 unless overridden; never switched silently (an unresolvable target -> INCONCLUSIVE)."""
    t = explicit or PRIMARY_FPR
    src = "set on the command line (plan: 2e-4)" if explicit and explicit != PRIMARY_FPR else "plan §8.2 E0.3 / SPEC 6.5"
    if MX.resolvable(n_min, t):
        return t, f"{src}; resolvable (min n_neg in the gate contrasts = {n_min:,})"
    need = int(np.ceil(5 / t))
    return t, (f"{src}; NOT resolvable with min n_neg = {n_min:,} (needs >= {need:,} for 5 expected false positives) "
               f"-> gate INCONCLUSIVE; build with a big pool (--n-bigneg) or report a looser FPR explicitly")


def fpr_key(t: float) -> str:
    """SPEC 7.3 key of a window FPR, exact: 2e-4 -> '2e-4', 1.6e-4 -> '1.6e-4', 0.015 -> '1.5e-2'."""
    m, e = f"{t:.6e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


NEGSET_SPEC = {"plain": "general", "instr": "instr_heavy", "notinject": "notinject"}   # SPEC 2.6 negative_set names


def bootstrap(contrasts: Dict[str, Contrast], n_groups: int, primary: float, n_boot: int, L: int,
              seed: int = 0) -> Dict[str, Dict]:
    """Per contrast: replicate curves (primary TPR, AUROC) -> per-layer bands and l95 samples; plus comparison ratios."""
    rng = np.random.default_rng(seed)
    boots = {k: MX.CurveBoot(c.Sp, c.Sn, primary, seed=seed) for k, c in contrasts.items()}
    res = {k: {"tpr": [], "auroc": []} for k in contrasts}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups).astype(np.float64)
        for k, c in contrasts.items():
            wp, wn = counts[c.gp], counts[c.gn]
            if wp.sum() == 0 or wn.sum() == 0:              # keep replicate indices aligned across contrasts
                res[k]["tpr"].append(np.full(L, np.nan))
                res[k]["auroc"].append(np.full(L, np.nan))
                continue
            res[k]["tpr"].append(boots[k].tpr(wp, wn))
            res[k]["auroc"].append(boots[k].auroc(wp, wn))
    out = {}
    for k, r in res.items():
        o = {}
        for m in ("tpr", "auroc"):
            arr = np.array(r[m])                                   # [B, L1]
            if arr.size == 0:
                continue
            o[m] = {"lo": np.nanpercentile(arr, 2.5, axis=0), "hi": np.nanpercentile(arr, 97.5, axis=0),
                    "l95": np.array([l95_of(m, x) for x in arr])}
        out[k] = o
    return out


def compare(name: str, a: Tuple[str, str], b: Tuple[str, str], cur: Dict, boot: Dict, L: int, metric_key: str,
            boot_key: str, n_neg: Optional[Dict[str, int]] = None) -> Optional[Dict]:
    ka, kb = f"{a[0]}/{a[1]}", f"{b[0]}/{b[1]}"
    if ka not in cur or kb not in cur:
        return None
    la, lb = l95_of(metric_key, cur[ka][metric_key]), l95_of(metric_key, cur[kb][metric_key])
    rec = {"comparison": name, "shallow": ka, "deep": kb, "metric": metric_key, "n_layers": L,
           "l95_shallow": la, "l95_deep": lb, "rel_shallow": la / L, "rel_deep": lb / L,
           "compute_ratio_shallow": la / L, "compute_ratio_deep": lb / L,
           "ratio": lb / max(la, 1.0) if np.isfinite(la) and np.isfinite(lb) else float("nan"),
           "rel_gap": (lb - la) / L if np.isfinite(la) and np.isfinite(lb) else float("nan"),
           "max_shallow": float(np.nanmax(cur[ka][metric_key])), "max_deep": float(np.nanmax(cur[kb][metric_key]))}
    if boot and ka in boot and kb in boot and boot_key in boot[ka] and boot_key in boot[kb]:
        sa, sb = boot[ka][boot_key]["l95"], boot[kb][boot_key]["l95"]
        ok = np.isfinite(sa) & np.isfinite(sb)              # same replicate index = same resample (paired)
        sa, sb = sa[ok], sb[ok]
        ratio = sb / np.maximum(sa, 1.0)
        gap = (sb - sa) / L
        rec.update(l95_shallow_ci=MX.percentile_ci(sa), l95_deep_ci=MX.percentile_ci(sb), ratio_ci=MX.percentile_ci(ratio),
                   rel_gap_ci=MX.percentile_ci(gap),
                   p_ratio_ge_min=float(np.mean(ratio >= GATE["min_ratio"])) if ratio.size else float("nan"),
                   p_deeper=float(np.mean(sb > sa)) if sa.size else float("nan"), n_boot=int(ok.sum()))
    floor = PEAK_FLOOR["auroc" if metric_key.startswith("auroc") else "tpr"]
    under = metric_key.startswith("tpr@") and n_neg is not None and \
        not all(MX.resolvable(n_neg.get(k, 0), float(metric_key[4:])) for k in (ka, kb))
    if not (np.isfinite(la) and np.isfinite(lb)) or min(rec["max_shallow"], rec["max_deep"]) < floor:
        rec["verdict"] = f"INCONCLUSIVE (peak < {floor})"
    elif under:
        rec["verdict"] = "INCONCLUSIVE (too few negatives for this FPR)"
    else:
        rec["verdict"] = "PASS" if rec["ratio"] >= GATE["min_ratio"] else "FAIL"
    return rec


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df",
         "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]}
STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.edgecolor": LIGHT["text2"], "axes.labelcolor": LIGHT["text"],
                             "xtick.color": LIGHT["text2"], "ytick.color": LIGHT["text2"], "axes.titlesize": 10,
                             "figure.facecolor": LIGHT["surface"], "axes.facecolor": LIGHT["surface"],
                             "savefig.facecolor": LIGHT["surface"], "legend.frameon": False})
        return plt
    except Exception as e:      # matplotlib not installed: the numbers are still written
        print(f"[analyze] figures skipped ({type(e).__name__}: {e}); pip install matplotlib", file=sys.stderr)
        return None


def fig_curves(results: Dict, framing: str, pool: str, metric: str, ylabel: str, path: Path) -> Optional[Path]:
    plt = _mpl()
    combos = [k for k in results if k[1] == framing and k[2] == pool]
    if plt is None or not combos:
        return None
    n = len(combos)
    cols = min(4, n)
    rows_ = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(max(7.0, 3.3 * cols), 3.0 * rows_ + 0.6), squeeze=False, sharey=True)
    for ax, combo in zip(axes.flat, combos):
        r = results[combo]
        L = r["n_layers"]
        x = np.arange(L + 1) / L
        for si, (t, ns, label) in enumerate(CONTRAST_STYLE):
            k = f"{t}/{ns}"
            if k not in r["curves"]:
                continue
            y = r["curves"][k][metric]
            col = LIGHT["series"][si]
            bk = "tpr" if metric.startswith("tpr") else "auroc"
            band = r["boot"].get(k, {}).get(bk)
            if band is not None:
                ax.fill_between(x, band["lo"], band["hi"], color=col, alpha=0.15, linewidth=0)
            ax.plot(x, y, STYLES[si], color=col, linewidth=2, label=label)
            l95 = l95_of(metric, y)
            if np.isfinite(l95):
                ax.plot([l95 / L], [y[int(l95)]], MARKERS[si], color=col, markersize=8,
                        markeredgecolor=LIGHT["surface"], markeredgewidth=2, zorder=5)
        ax.set_title(f"{combo[0]}", color=LIGHT["text"])
        ax.set_xlabel("relative depth (layer / L)")
        ax.grid(True, color=LIGHT["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-0.02, 1.02)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0][0].set_ylabel(ylabel)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2 if cols < 3 else 4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{ylabel} by layer -- {framing} framing, {pool} pooling\n"
                 f"dots = l95 (first layer at 95 % of the peak), bands = 95 % bootstrap CI", color=LIGHT["text"], fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_gap(gates: List[Dict], path: Path) -> Optional[Path]:
    plt = _mpl()
    rows_ = [g for g in gates if g.get("comparison") == "gate" and g["metric"].startswith("tpr")]
    if plt is None or not rows_:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows_) + 1.3))
    ys = np.arange(len(rows_))[::-1]
    for y, g in zip(ys, rows_):
        L = g["n_layers"]
        a, b = g["rel_shallow"], g["rel_deep"]
        ax.plot([a, b], [y, y], color=LIGHT["grid"], linewidth=2, zorder=1)
        for val, ci, col, mk in ((a, g.get("l95_shallow_ci"), LIGHT["series"][0], "o"),
                                 (b, g.get("l95_deep_ci"), LIGHT["series"][1], "s")):
            if ci and all(np.isfinite(ci)):
                ax.plot([ci[0] / L, ci[1] / L], [y, y], color=col, linewidth=2, alpha=0.45, zorder=2)
            ax.plot([val], [y], mk, color=col, markersize=8, markeredgecolor=LIGHT["surface"], markeredgewidth=2, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([g["combo"] for g in rows_])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"relative depth of l95 ({rows_[0]['metric']})")
    ax.grid(True, axis="x", color=LIGHT["grid"], linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot([], [], "o", color=LIGHT["series"][0], label="harmful vs all benign")
    ax.plot([], [], "s", color=LIGHT["series"][1], label="injection vs instruction-heavy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _ci(ci, scale=1.0, nd=2):
    if not ci or not all(np.isfinite(ci)):
        return ""
    return f" [{ci[0] * scale:.{nd}f}, {ci[1] * scale:.{nd}f}]"


def analyze(run: str, primary_fpr: Optional[float] = None, n_boot: Optional[int] = None, P: Optional[AGPaths] = None,
            only: Optional[Sequence[str]] = None) -> Dict:
    from ag.harness.writer import atomic_write_json, write_rows
    from . import lineage as LN
    from .fit import stored_sha
    from .windows import exp_of, load_card, read_jsonl_gz
    P = P or AGPaths.load()
    R = P.run(run)
    card = load_card(R)
    dsha = card["data_manifest_sha"]
    n_boot = card["recipe"]["profile"]["n_boot"] if n_boot is None else n_boot
    rows = read_jsonl_gz(R.windows)
    test = [r for r in rows if r["split"] == "test"]
    big = read_jsonl_gz(R.bigneg) if R.bigneg.exists() else []
    groups = sorted({r["group"] for r in test} | {r["group"] for r in big})
    gi = {g: i for i, g in enumerate(groups)}
    te_ids = [r["id"] for r in test]
    te_cls = np.array([r["cls"] for r in test])
    te_g = np.array([gi[r["group"]] for r in test])
    te_ho = np.array([bool(r.get("test_only")) for r in test])
    te_ho = te_ho if card["recipe"]["split_mode"] == "group" and te_ho.any() else None   # source mode: all held out
    big_ids = [r["id"] for r in big]
    big_cls = np.array([r["cls"] for r in big]) if big else None
    big_g = np.array([gi[r["group"]] for r in big]) if big else None
    combos = [c for c in discover(R) if not only or c[0] in only]
    stale = [c for c in combos if stored_sha(R.probes(c[0], c[1]) / f"{c[2]}.test_scores.npz") != dsha]
    for c in stale:
        print(f"[analyze] {'/'.join(c)}: probes were fit on another dataset than {R.windows} -> skipped "
              f"(re-run the pipeline for this backbone)", file=sys.stderr)
    combos = [c for c in combos if c not in stale]
    if not combos:
        raise SystemExit(f"[analyze] no current probe scores under {R.root / 'probes'}: run extract + fit first")
    R.report.mkdir(parents=True, exist_ok=True)
    big_ok = {}
    for c in combos:
        ok, why = _bigneg_ready(R, *c, dsha) if big else (False, "no big pool in this dataset")
        big_ok[c] = ok
        if big and not ok:
            print(f"[analyze] {'/'.join(c)}: big pool {why} -> test negatives only for this combo", file=sys.stderr)
    n_by = {c: gate_n_neg(te_cls[te_ho == False] if te_ho is not None else te_cls, big_cls, big_ok[c])  # noqa: E712
            for c in combos}
    primary, _ = choose_primary(max(n_by.values()), primary_fpr)
    unres = ["/".join(c) for c in combos if not MX.resolvable(n_by[c], primary)]
    src = "set on the command line (plan: 2e-4)" if primary_fpr and primary_fpr != PRIMARY_FPR else "plan §8.2 E0.3 / SPEC 6.5"
    primary_note = (f"{src}; resolvable in every combo (min n_neg in the gate contrasts = {min(n_by.values()):,})"
                    if not unres else
                    f"{src}; NOT resolvable for {unres} (needs >= {int(np.ceil(5 / primary)):,} negatives for 5 expected "
                    f"false positives) -> those combos are INCONCLUSIVE; the others are judged normally")
    targets = sorted(set(FPR_TARGETS) | {primary}, reverse=True)
    if len({fpr_key(t) for t in targets}) != len(targets):
        raise SystemExit(f"[analyze] FPR targets {targets} collide as result keys")
    pm = f"tpr@{primary:g}"
    results: Dict = {}
    table: List[Dict] = []
    gates: List[Dict] = []
    probe_rows: List[Dict] = []
    exp_id = exp_of_card(card)
    parents = [card.get("run_id", "")] + [_bb_run_id(R, b) for b in sorted({c[0] for c in combos})]
    model_sha = {}
    for b in sorted({c[0] for c in combos}):                 # the backbone run's lineage is the source of truth
        lp = R.bb_run(b) / "lineage.json"
        if lp.exists():
            model_sha[b] = (json.loads(lp.read_text()).get("model_sha") or {}).get("backbone", "")
    acfg = {"data_manifest_sha": dsha, "primary_fpr": primary, "n_boot": n_boot, "combos": ["/".join(c) for c in combos],
            "gate": GATE, "l95_frac": L95_FRAC, "fpr_targets": list(FPR_TARGETS)}
    chash = LN.config_hash(acfg)
    run_id = LN.make_run_id(exp_id, "analyze", "all", f"depthwin-{card['recipe']['profile']['window']}", chash,
                            int(card["recipe"].get("seed", 0)))
    lin = LN.start(R.report, run_id, exp_id, chash, seed=int(card["recipe"].get("seed", 0)),
                   producer="mock" if card.get("smoke") else "reference", model_sha=model_sha, data_manifest_sha=dsha,
                   parents=[p for p in parents if p], fresh=True)
    for combo in combos:
        bb, framing, pool = combo
        z = np.load(R.probes(bb, framing) / f"{pool}.test_scores.npz")
        tasks = [str(t) for t in z["tasks"]]
        ids = [str(x) for x in z["test_ids"]]
        if ids != te_ids:
            raise SystemExit(f"[analyze] {combo}: test ids differ from {R.windows} (re-run extract/fit)")
        sc = z["scores"]                                           # [T, S, n_test, L1]
        T, S, _, L1 = sc.shape
        L = L1 - 1
        bmm, bids = _bigneg_scores(R, bb, framing, pool, T, S, L1) if big_ok[combo] else (None, None)
        if bmm is not None and bids != big_ids:
            print(f"[analyze] {combo}: big-pool ids differ from {R.bigneg} -> ignored", file=sys.stderr)
            bmm = None
        C = build_contrasts(sc, tasks, te_cls, te_g, bmm, big_cls, big_g, te_ho)
        cur = {k: curves_of(c, targets) for k, c in C.items()}
        t0 = time.time()
        boot = bootstrap(C, len(groups), primary, n_boot, L1) if n_boot else {}
        for k, c in C.items():
            for l in range(L1):
                rec = {"backbone": bb, "framing": framing, "pooling": pool, "contrast": k, "layer": l, "rel_depth": l / L,
                       "n_pos": c.Sp.shape[1], "n_neg": c.Sn.shape[1], "n_neg_bigpool": c.n_neg_big,
                       "auroc": cur[k]["auroc"][l]}
                for t in targets:
                    rec[f"tpr@{t:g}"] = cur[k][f"tpr@{t:g}"][l]
                    rec[f"fpr@{t:g}"] = cur[k][f"fpr@{t:g}"][l]
                    rec[f"resolvable@{t:g}"] = MX.resolvable(c.Sn.shape[1], t)
                if k in boot and "tpr" in boot[k]:
                    rec[f"{pm}_lo"], rec[f"{pm}_hi"] = boot[k]["tpr"]["lo"][l], boot[k]["tpr"]["hi"][l]
                    rec["auroc_lo"], rec["auroc_hi"] = boot[k]["auroc"]["lo"][l], boot[k]["auroc"]["hi"][l]
                table.append(rec)
        summ = {}
        for k, c in C.items():
            e = {}
            for m in ["auroc"] + [f"tpr@{t:g}" for t in targets]:
                l95 = l95_of(m, cur[k][m])
                e[m] = {"l95": l95, "compute_ratio": l95 / L, "max": float(np.nanmax(cur[k][m])),
                        "at_last": float(cur[k][m][-1]), "per_seed_l95": cur[k]["_per_seed_l95"].get(m),
                        "resolvable": MX.resolvable(c.Sn.shape[1], float(m[4:])) if m.startswith("tpr@") else True}
            for m, bk in ((pm, "tpr"), ("auroc", "auroc")):
                if k in boot and bk in boot[k]:
                    e[m]["l95_ci"] = MX.percentile_ci(boot[k][bk]["l95"])
            summ[k] = {"n_pos": c.Sp.shape[1], "n_neg": c.Sn.shape[1], "n_neg_bigpool": c.n_neg_big, "metrics": e}
        cmp_ = []
        for name, a, b in COMPARISONS:
            for mk, bk in ((pm, "tpr"), ("auroc", "auroc")):
                g = compare(name, a, b, cur, boot, L, mk, bk, {k: c.Sn.shape[1] for k, c in C.items()})
                if g:
                    g["combo"] = f"{bb} / {framing} / {pool}"
                    cmp_.append(g)
        gates += cmp_
        seeds = [int(x) for x in z["seeds"]]
        for k, c in C.items():                              # SPEC 7.3-style rows, one per seed (not seed means)
            t_name, ns = k.split("/")
            for si, per in enumerate(cur[k]["_per_seed"]):
                for l in range(L1):
                    probe_rows.append({
                        "backbone": bb, "framing": framing, "pooling": pool, "layer": l, "rel_depth": l / L,
                        "task": t_name, "negative_set": NEGSET_SPEC.get(ns, ns), "negative_set_poc": ns,
                        "seed": seeds[si], "auroc": float(per["auroc"][l]),
                        "tpr_at_fpr": {fpr_key(t): float(per[f"tpr@{t:g}"][l]) for t in targets},
                        "achieved_fpr": {fpr_key(t): float(per[f"fpr@{t:g}"][l]) for t in targets},
                        "n_pos": int(c.Sp.shape[1]), "n_neg": int(c.Sn.shape[1])})
        fitj = R.probes(bb, framing) / f"{pool}.fit.json"
        controls = json.loads(fitj.read_text()).get("controls_shuffled_auroc") if fitj.exists() else None
        results[combo] = {"n_layers": L, "curves": cur, "boot": boot, "summary": summ, "comparisons": cmp_,
                          "primary": primary, "controls": controls, "bootstrap_seconds": round(time.time() - t0, 1)}
        g0 = next((g for g in cmp_ if g["comparison"] == "gate" and g["metric"] == pm), None)
        if g0:
            print(f"[analyze] {bb}/{framing}/{pool}: l95 harm={_fmt(g0['l95_shallow'], 0)} inj|instr={_fmt(g0['l95_deep'], 0)} "
                  f"/ L={L}  ratio={_fmt(g0['ratio'], 2)}{_ci(g0.get('ratio_ci'))}  {g0['verdict']}", flush=True)
    lex = analyze_lex(R, te_ids, te_cls, te_g, big_ids, big_cls, big_g, primary, dsha, te_ho)
    write_rows(R.report / "probe_results.jsonl", run_id, probe_rows)
    e11 = {}
    for fr in sorted({c[1] for c in combos}):
        for po in sorted({c[2] for c in combos}):
            for gname in ("gate", "gate_plan"):
                gs = [g for g in gates if g["comparison"] == gname and g["metric"] == pm
                      and g["combo"].endswith(f" / {fr} / {po}")]
                if gs:
                    n_pass = sum(g["verdict"] == "PASS" for g in gs)
                    e11[f"{gname}:{fr}/{po}"] = {"n_backbones": len(gs), "n_pass": n_pass,
                                                 "e11_criterion_2plus": n_pass >= 2,
                                                 "verdicts": {g["combo"].split(" / ")[0]: g["verdict"] for g in gs}}
    # ---- outputs
    keys = sorted({k for r in table for k in r}, key=lambda k: (k not in ("backbone", "framing", "pooling", "contrast",
                                                                          "layer", "rel_depth"), k))
    with open(R.report / "layer_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    figs = []
    for fr in sorted({c[1] for c in results}):
        for po in sorted({c[2] for c in results}):
            p1 = fig_curves(results, fr, po, pm, f"TPR @ FPR {primary:g}", R.report / f"fig_depth_{fr}_{po}_tpr.png")
            p2 = fig_curves(results, fr, po, "auroc", "AUROC", R.report / f"fig_depth_{fr}_{po}_auroc.png")
            figs += [p for p in (p1, p2) if p]
    pg = fig_gap(gates, R.report / "fig_l95_gap.png")
    if pg:
        figs.append(pg)
    summary = {"run": run, "run_id": run_id, "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "recipe_hash": card["recipe_hash"], "data_manifest_sha": dsha, "e11_backbones": e11,
               "profile": card["recipe"]["profile"]["name"], "split_mode": card["recipe"]["split_mode"],
               "primary_metric": pm, "primary_note": primary_note,
               "gate": {"rule": f"l95({GATE['deep'][0]}/{GATE['deep'][1]}) >= {GATE['min_ratio']} x "
                                f"l95({GATE['shallow'][0]}/{GATE['shallow'][1]})", "l95_frac": L95_FRAC},
               "n_boot": n_boot, "comparisons": gates, "lexical": lex,
               "per_combo": {"|".join(c): {"summary": v["summary"], "controls_shuffled_auroc": v["controls"],
                                           "n_layers": v["n_layers"]} for c, v in results.items()},
               "figures": [str(p) for p in figs]}
    atomic_write_json(R.report / "summary.json", summary)
    md = report_md(summary, card, results)
    (R.report / "report.md").write_text(md, encoding="utf-8")
    LN.finish(R.report, lin)
    print(f"[analyze] -> {R.report / 'report.md'}")
    return summary


def exp_of_card(card: Dict) -> str:
    from .config import EXP_OF_PROFILE
    return EXP_OF_PROFILE.get(str(card["recipe"]["profile"]["name"]).split("+")[0], "E0.3")


def _bb_run_id(R, bb: str) -> str:
    f = R.bb_run(bb) / "run_id.txt"
    return f.read_text().strip() if f.exists() else ""


def analyze_lex(R, te_ids, te_cls, te_g, big_ids, big_cls, big_g, primary: float, dsha: str, te_ho=None) -> Dict:
    from .fit import stored_sha
    f = R.lex / "lex.test_scores.npz"
    if not f.exists():
        return {}
    if stored_sha(f) != dsha:
        print("[analyze] lexical baseline was fit on another dataset -> omitted (re-run fit)", file=sys.stderr)
        return {"error": "lexical baseline is stale (another dataset)"}
    z = np.load(f)
    sc = z["scores"]
    tasks = [str(t) for t in z["tasks"]]
    bigf = R.lex / "lex.bigneg_scores.npz"
    bmm = None
    if big_ids:
        if bigf.exists() and stored_sha(bigf) == dsha:
            bmm = np.load(bigf)["scores"]
        else:
            print("[analyze] lexical big-pool scores missing or stale -> lexical baseline on test negatives only",
                  file=sys.stderr)
    C = build_contrasts(sc, tasks, te_cls, te_g, bmm, big_cls, big_g, te_ho)
    out = {}
    for k, c in C.items():
        cv = curves_of(c, [primary])
        out[k] = {"auroc": float(cv["auroc"][0]), f"tpr@{primary:g}": float(cv[f"tpr@{primary:g}"][0]),
                  "n_neg": c.Sn.shape[1]}
    return out


def report_md(s: Dict, card: Dict, results: Dict) -> str:
    pm = s["primary_metric"]
    L = []
    L.append(f"# Depth asymmetry PoC -- run `{s['run']}`\n")
    L.append(f"Profile `{s['profile']}`, split mode `{s['split_mode']}`, recipe `{s['recipe_hash']}`, analyzed "
             f"{s['analyzed_at']}. Bootstrap: {s['n_boot']} replicates over snippet groups.\n")
    if card.get("smoke"):
        L.append("> **Smoke run**: placeholder classes and a tiny random model. These numbers check the plumbing only.\n")
    L.append("## Claim and gate\n")
    L.append("Same backbone, same probe: harmful windows separate from benign windows at shallow layers, while injection "
             "windows separate from instruction-heavy benign windows only at deeper layers. "
             f"Gate (plan E0.3): {s['gate']['rule']}, l95 = first layer reaching {s['gate']['l95_frac']:.0%} of the curve's "
             f"maximum. Primary metric `{pm}` ({s['primary_note']}).\n")
    L.append("`gate` = PoC proposal (harm vs plain + harm_hard); `gate_plan` = the plan's negatives on the harm side "
             "(harm vs plain). Which one feeds G0 is DECISIONS D-041 (e).\n")
    L.append("| backbone / framing / pooling | comparison | metric | l95 harm (rel) | l95 inj/instr (rel) | ratio [95% CI] | "
             "P(ratio >= 1.5) | rel. gap [95% CI] | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for g in s["comparisons"]:
        if g["comparison"] not in ("gate", "gate_plan"):
            continue
        L.append(f"| {g['combo']} | {g['comparison']} ({g['shallow']}) | {g['metric']} | "
                 f"{_fmt(g['l95_shallow'], 0)} ({_fmt(g['rel_shallow'], 2)}) | "
                 f"{_fmt(g['l95_deep'], 0)} ({_fmt(g['rel_deep'], 2)}) | {_fmt(g['ratio'], 2)}{_ci(g.get('ratio_ci'))} | "
                 f"{_fmt(g.get('p_ratio_ge_min'), 2)} | {_fmt(g['rel_gap'], 2)}{_ci(g.get('rel_gap_ci'))} | {g['verdict']} |")
    L.append("\nSecondary comparisons (same metric on both sides):\n")
    L.append("| backbone / framing / pooling | comparison | metric | l95 shallow side | l95 deep side | ratio [95% CI] |")
    L.append("|---|---|---|---|---|---|")
    for g in s["comparisons"]:
        if g["comparison"] in ("gate", "gate_plan"):
            continue
        L.append(f"| {g['combo']} | {g['shallow']} vs {g['deep']} | {g['metric']} | {_fmt(g['l95_shallow'], 0)} | "
                 f"{_fmt(g['l95_deep'], 0)} | {_fmt(g['ratio'], 2)}{_ci(g.get('ratio_ci'))} |")
    L.append("\n## l95 and peak value per contrast\n")
    L.append("`<task>@heldout` rows use the test_only source (LLMail-Inject) as positives and are not part of any gate.\n")
    L.append(f"| backbone / framing / pooling | contrast | n_pos | n_neg | l95 AUROC | max AUROC | l95 {pm} [95% CI] | "
             f"compute ratio l95/L | max {pm} | {pm} at last layer |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, v in s["per_combo"].items():
        for k, e in v["summary"].items():
            m = e["metrics"]
            L.append(f"| {key.replace('|', ' / ')} | {k} | {e['n_pos']} | {e['n_neg']:,} | {_fmt(m['auroc']['l95'], 0)} | "
                     f"{_fmt(m['auroc']['max'])} | {_fmt(m[pm]['l95'], 0)}{_ci(m[pm].get('l95_ci'), nd=0)} | "
                     f"{_fmt(m[pm]['compute_ratio'], 2)} | {_fmt(m[pm]['max'])} | {_fmt(m[pm]['at_last'])} |")
    if s.get("lexical"):
        L.append("\n## Lexical baseline (hashed n-grams, same probe, no model)\n")
        L.append(f"| contrast | AUROC | {pm} |")
        L.append("|---|---|---|")
        for k, e in s["lexical"].items():
            if isinstance(e, dict):
                L.append(f"| {k} | {_fmt(e.get('auroc'))} | {_fmt(e.get(pm))} |")
    L.append("\n## Controls\n")
    L.append("Shuffled training labels (first seed), test AUROC against the task's training negatives; ~0.5 expected:\n")
    for key, v in s["per_combo"].items():
        if v.get("controls_shuffled_auroc"):
            L.append(f"- {key.replace('|', ' / ')}: " + "; ".join(
                f"{t}: " + ", ".join(f"L{l}={a}" for l, a in d.items()) for t, d in v["controls_shuffled_auroc"].items()))
    L.append("\n## Data\n")
    w = card["windows"]
    classes = sorted({c for d in w.values() for c in d})
    L.append("| split | " + " | ".join(classes) + " |")
    L.append("|---|" + "---|" * len(classes))
    for sp in ("train", "val", "test"):
        L.append(f"| {sp} | " + " | ".join(str(w.get(sp, {}).get(c, 0)) for c in classes) + " |")
    if card.get("bigneg"):
        L.append(f"| big pool | " + " | ".join(str(card['bigneg'].get(c, 0)) for c in classes) + " |")
    L.append("\nSnippet tokens on test (mean / p50): " + "; ".join(
        f"{c} {card['snippet_tokens'][c]['test'].get('mean', 'n/a')} / {card['snippet_tokens'][c]['test'].get('p50', 'n/a')}"
        for c in classes if card['snippet_tokens'].get(c, {}).get('test')))
    if card.get("snippet_tokens_heldout"):
        L.append("Held-out (test_only) snippet tokens (mean / p50): " + "; ".join(
            f"{c} {v.get('mean', 'n/a')} / {v.get('p50', 'n/a')}" for c, v in card["snippet_tokens_heldout"].items()))
    L.append("\nSources per class (windows): " + "; ".join(
        f"{c}: " + ", ".join(f"{s_}={n}" for s_, n in d.items()) for c, d in card["windows_by_source"].items()))
    if card.get("length_warning"):
        L.append(f"\n> Length: {card['length_warning']}\n")
    L.append(f"\nSplit audit: {card['audit']}\n")
    if card.get("shortfall"):
        L.append(f"Shortfall (fewer distinct snippets than requested): {card['shortfall']}\n")
    if s["figures"]:
        L.append("## Figures\n")
        for f in s["figures"]:
            L.append(f"![{Path(f).stem}]({Path(f).name})")
    L.append("\n## Reading the result\n")
    L.append("- A PASS needs both curves to be well estimated: check that `max` of the primary metric is not tiny "
             "(an l95 read off a flat, near-zero curve is noise) and that the gate contrasts are `resolvable` at the "
             "primary FPR in layer_metrics.csv.\n"
             "- If the lexical baseline already reaches the peak value of a contrast, that contrast is solved by surface "
             "features; its l95 then says little about depth.\n"
             "- `raw` framing = what the LG trunk sees (window alone); `task` framing = what a serving target sees "
             "(window inside its chat template after a user task). The plan's claim is about the backbone, so a "
             "consistent gap in both framings is the strong form of the result.\n")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="layer curves, l95, gate, report")
    ap.add_argument("--run", required=True)
    ap.add_argument("--primary-fpr", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--backbones", default="", help="comma list (default: every fitted backbone)")
    a = ap.parse_args(argv)
    analyze(a.run, a.primary_fpr, a.n_boot, only=[b for b in a.backbones.split(",") if b] or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
