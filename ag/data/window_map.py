"""Blocks, windows, eligibility and payload windows (SPEC 1.1-1.3).

Block j covers prompt tokens [j*B, min((j+1)*B, n)). Window w covers blocks {w, w+1}.
NW = NB - 1 for NB >= 2. A prompt shorter than two blocks has one window (block 0) so that
every request has at least one window to rank.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Span = tuple[int, int]


def n_blocks(n_tokens: int, split_tokens: int = 128) -> int:
    return max(1, math.ceil(n_tokens / split_tokens))


def n_windows(n_tokens: int, split_tokens: int = 128) -> int:
    return max(1, n_blocks(n_tokens, split_tokens) - 1)


def window_token_span(w: int, n_tokens: int, split_tokens: int = 128) -> Span:
    return w * split_tokens, min((w + 2) * split_tokens, n_tokens)


def spans_to_mask(spans: Sequence[Span], n_tokens: int) -> np.ndarray:
    m = np.zeros(n_tokens, dtype=bool)
    for a, b in spans:
        m[max(0, a):min(n_tokens, b)] = True
    return m


def block_untrusted(untrusted: np.ndarray, split_tokens: int = 128) -> np.ndarray:
    """A block is untrusted if any of its tokens is untrusted (D-019)."""
    nb = n_blocks(len(untrusted), split_tokens)
    out = np.zeros(nb, dtype=np.uint8)
    for j in range(nb):
        out[j] = 1 if untrusted[j * split_tokens:(j + 1) * split_tokens].any() else 0
    return out


def eligible_windows(block_untr: np.ndarray) -> np.ndarray:
    """Window w is eligible iff block w or block w+1 is untrusted (D-019). Shape [NW]."""
    b = np.asarray(block_untr).astype(bool)
    if len(b) < 2:
        return b[:1].copy()
    return b[:-1] | b[1:]


def window_counts(mask: np.ndarray, split_tokens: int = 128) -> np.ndarray:
    """Number of masked tokens in each window. Shape [NW]."""
    n = len(mask)
    nb = n_blocks(n, split_tokens)
    per_block = np.array([mask[j * split_tokens:(j + 1) * split_tokens].sum() for j in range(nb)], dtype=np.int64)
    if nb < 2:
        return per_block[:1]
    return per_block[:-1] + per_block[1:]


def payload_windows_for(spans: Sequence[Span], n_tokens: int, split_tokens: int = 128) -> tuple[tuple[int, ...], str]:
    """P(r) for one payload (SPEC 1.3, D-026): windows holding >= half the payload tokens,
    else the window with the largest overlap. Returns (windows, rule)."""
    mask = spans_to_mask(spans, n_tokens)
    p = int(mask.sum())
    if p == 0:
        return (), "none"
    counts = window_counts(mask, split_tokens)
    half = tuple(int(w) for w in np.flatnonzero(counts * 2 >= p))
    if half:
        return half, "half"
    return (int(np.argmax(counts)),), "max_overlap"


def payload_windows(spans: Sequence[Span], n_tokens: int, split_tokens: int = 128,
                    n_sites: int = 1) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], str]:
    """Returns (P(r), per-site windows, payload_rule). With n_sites > 1 each span is a site."""
    if not spans:
        return (), (), "none"
    if n_sites > 1:
        per_site = tuple(payload_windows_for([s], n_tokens, split_tokens)[0] for s in spans)
        union = tuple(sorted({w for site in per_site for w in site}))
        return union, per_site, "per_site"
    wins, rule = payload_windows_for(spans, n_tokens, split_tokens)
    return wins, (), rule


def window_scores_from_blocks(block_mass: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Window score = sum of its two blocks; ineligible windows get -inf (SPEC 4.5)."""
    bm = np.asarray(block_mass, dtype=np.float64)
    s = bm[:1].copy() if len(bm) < 2 else bm[:-1] + bm[1:]
    s[~np.asarray(eligible, dtype=bool)] = -np.inf
    return s
