"""Linear probes: L2-regularised logistic regression on standardized features, full-batch L-BFGS (convex, so the
solution does not depend on initialisation; seeds vary the training subsample instead).

Loss = sum_i sw_i * BCE(x_i . w + b, y_i) + lam/2 * ||w||^2 with sample weights sw that give the positives half the
mass and split the other half equally across the negative classes present (so 'plain' cannot drown the hard
negatives).  lam is chosen on validation AUROC along a warm-started path from the largest value down.

Also here: the lexical baseline (signed hashed unigram+bigram counts, log1p, L2-normalised) -- the same probe on
features that carry no model computation at all.
"""
from __future__ import annotations

import re
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import auroc


def balanced_weights(cls: np.ndarray, pos_cls: Sequence[str], neg_cls: Sequence[str]) -> np.ndarray:
    """Positives share 0.5; the negative classes present share the other 0.5 equally."""
    cls = np.asarray(cls)
    w = np.zeros(len(cls), np.float64)
    pos = np.isin(cls, list(pos_cls))
    if pos.any():
        w[pos] = 0.5 / pos.sum()
    negs = [c for c in dict.fromkeys(neg_cls) if (cls == c).any()]
    for c in negs:
        m = cls == c
        w[m] = 0.5 / len(negs) / m.sum()
    return w


def fit_logreg(X, y, sw, lam: float, init: Optional[Tuple] = None, max_iter: int = 100):
    """X [n, d] float32 torch (standardized), y [n] float {0,1}, sw [n] float (sums to 1).  Returns (w [d], b)."""
    import torch
    d = X.shape[1]
    w = torch.zeros(d, device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(1, device=X.device, dtype=X.dtype, requires_grad=True)
    if init is not None:
        with torch.no_grad():
            w.copy_(init[0])
            b.copy_(init[1])
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-9)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = (bce(z, y, reduction="none") * sw).sum() + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def select_and_fit(Xtr, ytr, swtr, Xva, yva_bin: np.ndarray, lambdas: Sequence[float]):
    """Warm-started path over lambdas (largest first); returns (w, b, lam, {lam: val_auroc})."""
    import torch
    best = None
    init = None
    val = {}
    for lam in sorted(lambdas, reverse=True):
        w, b = fit_logreg(Xtr, ytr, swtr, lam, init)
        init = (w, b)
        s = (Xva @ w + b).float().cpu().numpy()
        a = auroc(s[yva_bin == 1], s[yva_bin == 0])
        val[lam] = a
        if best is None or (np.isfinite(a) and a > best[3] + 1e-6):
            best = (w, b, lam, a)
    return best[0], best[1], best[2], val


def standardize_stats(X, eps: float = 1e-6):
    mu = X.mean(0)
    sd = X.std(0).clamp_min(eps)
    return mu, sd


# ----------------------------------------------------------------------------------------------
# lexical baseline
# ----------------------------------------------------------------------------------------------
_TOK = re.compile(r"\w+|[^\w\s]")


def lex_features(texts: Sequence[str], dim: int = 1 << 13) -> np.ndarray:
    X = np.zeros((len(texts), dim), np.float32)
    for i, t in enumerate(texts):
        toks = _TOK.findall(t.lower())
        grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = zlib.crc32(g.encode("utf-8"))
            X[i, h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    X = np.sign(X) * np.log1p(np.abs(X))
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-6)
