"""Layer-curve metrics, l95, and the cluster bootstrap.

Decision rule (same as lgs1.metrics.strict_matched_fpr): positive iff score > tau, tau = the (allowed+1)-th largest
negative score with allowed = floor(target * n_neg); ties at tau count as negative, so achieved FPR <= target.
With bootstrap weights w (multiplicities), allowed = floor(target * sum(w)) and tau is the first score in descending
order whose cumulative weight exceeds `allowed` -- identical to the unweighted rule when every w = 1.

l95(curve) = first layer whose value reaches L95_FRAC x the curve's maximum (plan §8.2 E0.3).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, "left")
    hi = np.searchsorted(ns, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (pos.size * neg.size))


def auroc_w(pos: np.ndarray, wp: np.ndarray, neg_sorted: np.ndarray, wn_sorted_cum: np.ndarray) -> float:
    """Weighted AUROC; neg_sorted ascending with the cumulative weights of that order (cum[-1] = total)."""
    W_n = wn_sorted_cum[-1]
    W_p = wp.sum()
    if W_n <= 0 or W_p <= 0:
        return float("nan")
    lo = np.searchsorted(neg_sorted, pos, "left")
    hi = np.searchsorted(neg_sorted, pos, "right")
    c = np.concatenate([[0.0], wn_sorted_cum])
    below, ties = c[lo], c[hi] - c[lo]
    return float((wp * (below + 0.5 * ties)).sum() / (W_p * W_n))


def tau_at(neg: np.ndarray, target: float) -> float:
    s = np.sort(np.asarray(neg, np.float64))[::-1]
    allowed = int(math.floor(target * s.size + 1e-12))
    return -math.inf if allowed >= s.size else float(s[allowed])


def tpr_at(pos: np.ndarray, neg: np.ndarray, target: float) -> Tuple[float, float, float]:
    """(TPR, achieved FPR, tau)."""
    tau = tau_at(neg, target)
    return float(np.mean(np.asarray(pos) > tau)), float(np.mean(np.asarray(neg) > tau)), tau


def curves(Sp: np.ndarray, Sn: np.ndarray, targets: Sequence[float]) -> Dict[str, np.ndarray]:
    """Sp [n_pos, L], Sn [n_neg, L] -> {'auroc': [L], 'tpr@t': [L], 'fpr@t': [L]}."""
    L = Sp.shape[1]
    out = {"auroc": np.array([auroc(Sp[:, l], Sn[:, l]) for l in range(L)])}
    for t in targets:
        r = [tpr_at(Sp[:, l], Sn[:, l], t) for l in range(L)]
        out[f"tpr@{t:g}"] = np.array([x[0] for x in r])
        out[f"fpr@{t:g}"] = np.array([x[1] for x in r])
    return out


def ell95(curve: Sequence[float], frac: float = 0.95) -> float:
    c = np.asarray(curve, np.float64)
    if c.size == 0 or not np.isfinite(c).any():
        return float("nan")
    m = np.nanmax(c)
    if m <= 0:
        return float("nan")
    hit = np.where(np.nan_to_num(c, nan=-np.inf) >= frac * m)[0]
    return float(hit[0]) if hit.size else float("nan")


def resolvable(n_neg: int, target: float, min_fp: float = 5.0) -> bool:
    """Enough negatives that the threshold sits on >= min_fp expected false positives."""
    return n_neg * target >= min_fp


# ----------------------------------------------------------------------------------------------
# cluster bootstrap
# ----------------------------------------------------------------------------------------------
def group_weights(group_index: np.ndarray, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    """Per-sample multiplicities from resampling GROUPS with replacement (paired across layers / seeds / contrasts)."""
    counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups).astype(np.float64)
    return counts[group_index]


class CurveBoot:
    """Bootstrap curves of one contrast for many weight vectors.  Scores Sp [seeds, n_pos, L], Sn [seeds, n_neg, L].
    For TPR at a low target only the top-K negatives can set tau, so each (seed, layer) keeps its K largest negatives
    (K covers allowed + a wide margin); a replicate whose allowance falls past K recomputes exactly."""

    def __init__(self, Sp: np.ndarray, Sn: np.ndarray, target: Optional[float], auroc_max_neg: int = 5000,
                 seed: int = 0):
        self.Sp, self.Sn, self.target = np.asarray(Sp, np.float32), np.asarray(Sn, np.float32), target
        S, n_n, L = Sn.shape
        self.S, self.L = S, L
        self.pos_order = np.argsort(self.Sp, axis=1)                 # ascending per (seed, layer)
        self.pos_sorted = np.take_along_axis(self.Sp, self.pos_order, 1)
        if target is not None:
            allowed = target * n_n
            self.K = int(min(n_n, max(200, math.ceil(3 * allowed + 10 * math.sqrt(allowed + 1) + 50))))
            part = np.argpartition(-self.Sn, self.K - 1, axis=1)[:, :self.K, :] if self.K < n_n else \
                np.broadcast_to(np.arange(n_n)[None, :, None], (S, n_n, L)).copy()
            vals = np.take_along_axis(self.Sn, part, 1)
            o = np.argsort(-vals, axis=1)
            self.top_idx = np.take_along_axis(part, o, 1)               # [S, K, L], descending scores
            self.top_val = np.take_along_axis(vals, o, 1)
        rng = np.random.default_rng(seed)
        self.auc_idx = np.arange(n_n) if n_n <= auroc_max_neg else np.sort(rng.choice(n_n, auroc_max_neg, replace=False))
        sub = self.Sn[:, self.auc_idx, :]
        self.auc_order = np.argsort(sub, axis=1)
        self.auc_sorted = np.take_along_axis(sub, self.auc_order, 1)

    def tpr(self, wp: np.ndarray, wn: np.ndarray) -> np.ndarray:
        """Seed-mean TPR curve [L] under weights."""
        Wn, Wp = wn.sum(), wp.sum()
        allowed = math.floor(self.target * Wn + 1e-12)
        out = np.zeros((self.S, self.L))
        for s in range(self.S):
            for l in range(self.L):
                cw = np.cumsum(wn[self.top_idx[s, :, l]])
                k = int(np.searchsorted(cw, allowed, "right"))
                if k < cw.size:
                    tau = self.top_val[s, k, l]
                elif cw[-1] >= Wn:                                    # every negative fits in the allowance
                    tau = -math.inf
                else:                                                 # allowance past the top-K: exact path
                    order = np.argsort(-self.Sn[s, :, l])
                    cw2 = np.cumsum(wn[order])
                    k2 = int(np.searchsorted(cw2, allowed, "right"))
                    tau = self.Sn[s, order[k2], l] if k2 < cw2.size else -math.inf
                ps = self.pos_sorted[s, :, l]
                pw = np.cumsum(wp[self.pos_order[s, :, l]])
                j = int(np.searchsorted(ps, tau, "right"))            # first positive strictly above tau
                above = Wp - (pw[j - 1] if j > 0 else 0.0)
                out[s, l] = above / Wp if Wp > 0 else np.nan
        return out.mean(0)

    def auroc(self, wp: np.ndarray, wn: np.ndarray) -> np.ndarray:
        wn_sub = wn[self.auc_idx]
        out = np.zeros((self.S, self.L))
        for s in range(self.S):
            for l in range(self.L):
                cum = np.cumsum(wn_sub[self.auc_order[s, :, l]])
                out[s, l] = auroc_w(self.Sp[s, :, l], wp, self.auc_sorted[s, :, l], cum)
        return out.mean(0)


def percentile_ci(x: Sequence[float], level: float = 0.95) -> Tuple[float, float]:
    a = np.asarray([v for v in x if np.isfinite(v)], np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    q = (1 - level) / 2
    return float(np.quantile(a, q)), float(np.quantile(a, 1 - q))
