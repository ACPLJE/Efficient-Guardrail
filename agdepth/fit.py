"""Per-layer probes for every task (harm, inj) x seed, scored on the test windows of EVERY class.

For each (backbone, framing, pooling) and each layer l (0 = embeddings):
    standardize with mean/std of all train windows at that layer
    for task in TASKS, seed in seeds:
        train = train windows of task.pos + task.train_neg, 85 % of their groups (seeded), balanced weights
        lam   = argmax val AUROC over LAMBDAS (warm-started path)
        score every test window (the analysis picks the negative set per contrast)
Controls (first seed): the same probe on shuffled train labels at layers {0, L/2, L} -> test AUROC should be ~0.5.
Lexical baseline: the same probe on hashed n-gram features of the window text (no model), written once per run.

Outputs in <run>/probes/<bb>/<framing>/:
    <pool>.probes.npz       W [T,S,L+1,d] f16, b, lam, val_auc [T,S,L+1], mu, sd [L+1,d]
    <pool>.test_scores.npz  scores [T,S,n_test,L+1] f32, test_ids
    <pool>.fit.json         val AUROC curves, chosen lambdas, controls, timing

    python -m agdepth.fit --run NAME --backbone qwen3guard_0.6b [--framings raw] [--poolings mean,last]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DEFAULT_POOLINGS, LAMBDAS, TASKS, TRAIN_SUBSAMPLE, AGPaths, backbone
from .metrics import auroc
from .probe import balanced_weights, lex_features, select_and_fit, standardize_stats


def _sub(group: str, seed: int) -> bool:
    return int(hashlib.sha1(f"sub:{seed}:{group}".encode()).hexdigest()[:8], 16) / 16 ** 8 < TRAIN_SUBSAMPLE


def _arrays(rows: Sequence[Dict]):
    split = np.array([r["split"] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    group = np.array([r["group"] for r in rows])
    return split, cls, group


def _device(device: Optional[str]) -> str:
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def fit_matrix(get_layer, n_layers1: int, rows: Sequence[Dict], seeds: Sequence[int], lambdas: Sequence[float],
               device: str, controls: bool = True, log_prefix: str = "") -> Dict:
    """Core loop shared by the model features and the lexical baseline.  get_layer(l) -> float32 array [N, d]."""
    import torch
    split, cls, group = _arrays(rows)
    tr_all, va_all, te_all = split == "train", split == "val", split == "test"
    te_idx = np.where(te_all)[0]
    tasks = list(TASKS)
    S = len(seeds)
    out: Dict = {"tasks": tasks, "seeds": list(seeds), "test_ids": [rows[i]["id"] for i in te_idx]}
    d = None
    scores = np.zeros((len(tasks), S, te_idx.size, n_layers1), np.float32)
    lam_sel = np.zeros((len(tasks), S, n_layers1), np.float32)
    val_auc = np.zeros((len(tasks), S, n_layers1), np.float32)
    ctrl_layers = sorted({0, n_layers1 // 2, n_layers1 - 1})
    controls_out: Dict[str, Dict[str, float]] = {}
    W = B = MU = SD = None
    sub_masks = {s: np.array([_sub(g, s) for g in group]) for s in seeds}
    t0 = time.time()
    for l in range(n_layers1):
        X = torch.from_numpy(np.ascontiguousarray(get_layer(l), dtype=np.float32)).to(device)
        if d is None:
            d = X.shape[1]
            W = np.zeros((len(tasks), S, n_layers1, d), np.float16)
            B = np.zeros((len(tasks), S, n_layers1), np.float32)
            MU = np.zeros((n_layers1, d), np.float32)
            SD = np.zeros((n_layers1, d), np.float32)
        tr_t = torch.from_numpy(np.where(tr_all)[0]).to(device)
        mu, sd = standardize_stats(X[tr_t])
        X = (X - mu) / sd
        MU[l], SD[l] = mu.cpu().numpy(), sd.cpu().numpy()
        Xte = X[torch.from_numpy(te_idx).to(device)]
        for ti, tname in enumerate(tasks):
            task = TASKS[tname]
            in_task = np.isin(cls, list(task.pos + task.train_neg))
            va = np.where(va_all & in_task)[0]
            yva = np.isin(cls[va], list(task.pos)).astype(int)
            Xva = X[torch.from_numpy(va).to(device)]
            for si, s in enumerate(seeds):
                tr = np.where(tr_all & in_task & sub_masks[s])[0]
                y = torch.from_numpy(np.isin(cls[tr], list(task.pos)).astype(np.float32)).to(device)
                sw = torch.from_numpy(balanced_weights(cls[tr], task.pos, task.train_neg).astype(np.float32)).to(device)
                Xtr = X[torch.from_numpy(tr).to(device)]
                w, b, lam, val = select_and_fit(Xtr, y, sw, Xva, yva, lambdas)
                scores[ti, si, :, l] = (Xte @ w + b).float().cpu().numpy()
                W[ti, si, l] = w.float().cpu().numpy().astype(np.float16)
                B[ti, si, l] = float(b.item())
                lam_sel[ti, si, l] = lam
                val_auc[ti, si, l] = val[lam]
                if controls and si == 0 and l in ctrl_layers:
                    # labels permuted on train AND val, weights balanced on the permuted labels: nothing about the
                    # true classes reaches the probe, so test AUROC must sit near 0.5
                    r = np.random.default_rng(1000 + l)
                    yp_np = r.permutation(np.isin(cls[tr], list(task.pos)).astype(np.float32))
                    swp = np.where(yp_np == 1, 0.5 / max(1, yp_np.sum()), 0.5 / max(1, (1 - yp_np).sum()))
                    wc, bc, _, _ = select_and_fit(Xtr, torch.from_numpy(yp_np).to(device),
                                                  torch.from_numpy(swp.astype(np.float32)).to(device), Xva,
                                                  r.permutation(yva), lambdas)
                    sc = (Xte @ wc + bc).float().cpu().numpy()
                    te_cls = cls[te_idx]
                    neg_all = np.isin(te_cls, list(task.train_neg))
                    pos = np.isin(te_cls, list(task.pos))
                    controls_out.setdefault(tname, {})[str(l)] = round(auroc(sc[pos], sc[neg_all]), 4)
        print(f"[fit] {log_prefix} layer {l}/{n_layers1 - 1}  val AUROC " +
              "  ".join(f"{t}={val_auc[i, :, l].mean():.4f}" for i, t in enumerate(tasks)) +
              f"  ({time.time() - t0:.0f}s)", flush=True)
        del X, Xte
    out.update(scores=scores, W=W, b=B, lam=lam_sel, val_auc=val_auc, mu=MU, sd=SD, controls=controls_out,
               seconds=round(time.time() - t0, 1))
    return out


def savez_atomic(path: Path, **arrays) -> None:
    """np.savez through a temp file + os.replace (a killed job never leaves a half-written npz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def stored_sha(path: Path) -> str:
    """data_manifest_sha recorded in a probes / scores npz ('' if absent or unreadable)."""
    try:
        with np.load(path) as z:
            return str(z["data_manifest_sha"]) if "data_manifest_sha" in z.files else ""
    except Exception:
        return ""


def probes_current(R, bb: str, framing: str, pool: str, dsha: str) -> bool:
    D = R.probes(bb, framing)
    return all(stored_sha(D / f"{pool}.{k}.npz") == dsha for k in ("probes", "test_scores"))


def _save(D: Path, pool: str, res: Dict, extra: Dict, dsha: str) -> None:
    from ag.harness.writer import atomic_write_json
    D.mkdir(parents=True, exist_ok=True)
    savez_atomic(D / f"{pool}.probes.npz", W=res["W"], b=res["b"], lam=res["lam"], val_auc=res["val_auc"], mu=res["mu"],
                 sd=res["sd"], tasks=np.array(res["tasks"]), seeds=np.array(res["seeds"]), data_manifest_sha=np.array(dsha))
    savez_atomic(D / f"{pool}.test_scores.npz", scores=res["scores"], test_ids=np.array(res["test_ids"]),
                 tasks=np.array(res["tasks"]), seeds=np.array(res["seeds"]), data_manifest_sha=np.array(dsha))
    summ = {"tasks": res["tasks"], "seeds": res["seeds"], "seconds": res["seconds"], "controls_shuffled_auroc": res["controls"],
            "val_auroc_mean": {t: [round(float(x), 4) for x in res["val_auc"][i].mean(0)] for i, t in enumerate(res["tasks"])},
            "lambda_counts": {t: {f"{v:g}": int((res["lam"][i] == v).sum()) for v in np.unique(res["lam"][i])}
                              for i, t in enumerate(res["tasks"])}, "data_manifest_sha": dsha, **extra}
    atomic_write_json(D / f"{pool}.fit.json", summ)


def fit(run: str, bb_key: str, framings: Sequence[str] = ("raw",), poolings: Sequence[str] = DEFAULT_POOLINGS,
        seeds: Optional[Sequence[int]] = None, lambdas: Sequence[float] = LAMBDAS, device: Optional[str] = None,
        controls: bool = True, P: Optional[AGPaths] = None) -> None:
    from .extract import load_feats
    from .windows import load_card, read_jsonl_gz
    P = P or AGPaths.load()
    R = P.run(run)
    card = load_card(R)
    dsha = card["data_manifest_sha"]
    seeds = list(seeds) if seeds else list(card["recipe"]["profile"]["seeds"])
    rows = read_jsonl_gz(R.windows)
    dev = _device(device)
    bb = backbone(bb_key)
    for framing in framings:
        for pool in poolings:
            mm, ids, meta = load_feats(R, bb.key, framing, pool)
            if meta.get("data_manifest_sha") != dsha or ids != [r["id"] for r in rows]:
                raise SystemExit(f"{R.feats(bb.key, framing)}: features were extracted from another dataset "
                                 f"({meta.get('data_manifest_sha')} != {dsha}); re-run extract")
            res = fit_matrix(lambda l: mm[l], meta["layers"], rows, seeds, lambdas, dev, controls,
                             f"{bb.key}/{framing}/{pool}")
            _save(R.probes(bb.key, framing), pool, res, {"backbone": bb.key, "framing": framing, "pooling": pool,
                                                         "layers": meta["layers"], "d": meta["d"], "device": dev}, dsha)
            print(f"[fit] {bb.key}/{framing}/{pool} done in {res['seconds']}s -> {R.probes(bb.key, framing)}")
    fit_lex(run, seeds, lambdas, dev, P)


def fit_lex(run: str, seeds: Sequence[int], lambdas: Sequence[float] = LAMBDAS, device: Optional[str] = None,
            P: Optional[AGPaths] = None, force: bool = False) -> None:
    from .windows import load_card, read_jsonl_gz
    P = P or AGPaths.load()
    R = P.run(run)
    dsha = load_card(R)["data_manifest_sha"]
    big_ok = not R.bigneg.exists() or stored_sha(R.lex / "lex.bigneg_scores.npz") == dsha
    if stored_sha(R.lex / "lex.test_scores.npz") == dsha and big_ok and not force:
        return                                                # same dataset: the baseline is current
    rows = read_jsonl_gz(R.windows)
    X = lex_features([r["text"] for r in rows])
    res = fit_matrix(lambda l: X, 1, rows, seeds, lambdas, _device(device), False, "lexical")
    _save(R.lex, "lex", res, {"backbone": "lexical", "features": "signed hashed unigram+bigram, 8192 dims"}, dsha)
    (R.lex / "lex.bigneg_scores.npz").unlink(missing_ok=True)
    if R.bigneg.exists():
        big = read_jsonl_gz(R.bigneg)
        Xb = (lex_features([r["text"] for r in big]) - res["mu"][0]) / res["sd"][0]
        sc = np.einsum("nd,tsd->nts", Xb, res["W"][:, :, 0].astype(np.float32)) + res["b"][None, :, :, 0]
        savez_atomic(R.lex / "lex.bigneg_scores.npz", scores=sc[..., None].astype(np.float32),
                     ids=np.array([r["id"] for r in big]), data_manifest_sha=np.array(dsha))
    print(f"[fit] lexical baseline -> {R.lex}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="per-layer linear probes")
    ap.add_argument("--run", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--framings", default="raw")
    ap.add_argument("--poolings", default=",".join(DEFAULT_POOLINGS))
    ap.add_argument("--seeds", default="", help="comma list (default: the profile's seeds)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args(argv)
    fit(a.run, a.backbone, [f for f in a.framings.split(",") if f], [p for p in a.poolings.split(",") if p],
        [int(s) for s in a.seeds.split(",") if s] or None, LAMBDAS, a.device, not a.no_controls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
