"""Text utilities for data v2: normalisation, exact / near-duplicate grouping (MinHash + LSH, numpy only),
paragraph segmentation and insertion-point selection.  No network, no third-party dependencies beyond numpy.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


# ----------------------------------------------------------------------------------------------
# normalisation and hashing
# ----------------------------------------------------------------------------------------------
def norm_text(s: str) -> str:
    """Lower-case, NFKC, punctuation removed, whitespace collapsed — the key for EXACT duplicate groups."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def exact_key(s: str) -> str:
    return sha1_hex(norm_text(s))


def word_shingles(s: str, n: int = 5) -> Set[str]:
    """Word n-gram shingles of the normalised text; texts shorter than n words yield one shingle (the text)."""
    words = norm_text(s).split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ----------------------------------------------------------------------------------------------
# MinHash + LSH (banding) + union-find -> near-duplicate clusters
# ----------------------------------------------------------------------------------------------
_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 20260917):
        rng = np.random.RandomState(seed)
        self.num_perm = num_perm
        # 32-bit coefficients keep a*h + b below 2^64 (no silent wrap-around inside uint64 arithmetic)
        self.a = rng.randint(1, _MAX_HASH, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.b = rng.randint(0, _MAX_HASH, size=num_perm, dtype=np.int64).astype(np.uint64)

    def _h32(self, shingles: Iterable[str]) -> np.ndarray:
        return np.array([int.from_bytes(hashlib.blake2b(x.encode("utf-8"), digest_size=4).digest(), "little")
                         for x in shingles], dtype=np.uint64)

    def signature(self, shingles: Set[str]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        h = self._h32(shingles)                                        # (n,)
        # (a*h + b) mod p, then mod 2^32 ; vectorised over permutations
        prod = (np.outer(h, self.a) + self.b) % _MERSENNE                # (n, perm)  uint64 arithmetic mod p
        return (prod & _MAX_HASH).min(axis=0).astype(np.uint64)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_dup_clusters(texts: Sequence[str], n: int = 5, num_perm: int = 128, bands: int = 32,
                      threshold: float = 0.8, seed: int = 20260917,
                      shingle_cache: Optional[List[Set[str]]] = None) -> List[int]:
    """Cluster texts whose word-5-gram Jaccard >= threshold (verified exactly on candidate pairs from LSH).

    Returns cluster root index per text (root = smallest member index).  O(N * num_perm) hashing plus
    candidate-pair verification; fine for ~200k short needles."""
    N = len(texts)
    if N == 0:
        return []
    mh = MinHasher(num_perm, seed)
    shingles = shingle_cache if shingle_cache is not None else [word_shingles(t, n) for t in texts]
    sigs = np.stack([mh.signature(s) for s in shingles])                 # (N, perm)
    rows = num_perm // bands
    uf = UnionFind(N)
    seen_pairs: Set[Tuple[int, int]] = set()
    for b in range(bands):
        band = sigs[:, b * rows:(b + 1) * rows]
        buckets: Dict[bytes, List[int]] = {}
        for i in range(N):
            buckets.setdefault(band[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            base = members[0]
            for j in members[1:]:
                pair = (base, j)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if jaccard(shingles[base], shingles[j]) >= threshold:
                    uf.union(base, j)
    return [uf.find(i) for i in range(N)]


def group_ids(texts: Sequence[str], **kw) -> Tuple[List[str], Dict[str, int]]:
    """Exact groups (normalised hash) merged with near-duplicate clusters -> stable group id per text.

    Returns (group_id per text, stats).  group id = 'g' + sha1 of the smallest exact key in the cluster, so it
    is stable across runs and independent of input order."""
    keys = [exact_key(t) for t in texts]
    uniq: Dict[str, int] = {}
    rep_idx: List[int] = []
    for i, k in enumerate(keys):
        if k not in uniq:
            uniq[k] = len(rep_idx)
            rep_idx.append(i)
    rep_texts = [texts[i] for i in rep_idx]
    roots = near_dup_clusters(rep_texts, **kw)
    root_key: Dict[int, str] = {}
    for r_i, root in enumerate(roots):
        k = keys[rep_idx[r_i]]
        root_key[root] = min(root_key.get(root, k), k)
    gids: List[str] = []
    for i, k in enumerate(keys):
        root = roots[uniq[k]]
        gids.append("g" + sha1_hex(root_key[root])[:16])
    n_groups = len(set(gids))
    stats = {"n": len(texts), "n_exact_unique": len(rep_idx), "n_groups": n_groups,
             "n_near_dup_merged": len(rep_idx) - n_groups}
    return gids, stats


# ----------------------------------------------------------------------------------------------
# paragraphs and insertion points
# ----------------------------------------------------------------------------------------------
_BLANK = re.compile(r"\n[ \t\r]*\n+")
_SENT = re.compile(r"(?<=[.!?。！？])\s+")


def split_paragraphs(text: str, style: str = "blank") -> List[str]:
    """Paragraph units used for both piece cutting and needle insertion.

    blank: split at blank lines (fallback: single newlines, then sentences)
    line:  split at single newlines (chat logs, tool outputs)
    code:  split at blank lines only (a code file without blank lines is ONE paragraph)"""
    text = (text or "").replace("\r\n", "\n").strip("\n")
    if not text.strip():
        return []
    if style == "code":
        parts = [p for p in _BLANK.split(text) if p.strip()]
        return parts or [text]
    if style == "line":
        parts = [p for p in text.split("\n") if p.strip()]
        return parts or [text]
    parts = [p for p in _BLANK.split(text) if p.strip()]
    if len(parts) <= 1:
        lines = [p for p in text.split("\n") if p.strip()]
        if len(lines) > 1:
            return lines
        sents = [p for p in _SENT.split(text) if p.strip()]
        return sents if len(sents) > 1 else [text]
    return parts


def paragraph_boundaries(paragraphs: Sequence[str], sep: str = "\n\n") -> List[int]:
    """Character offsets of every paragraph start plus the end of the text (len(paragraphs)+1 boundaries)."""
    bounds = [0]
    pos = 0
    for i, p in enumerate(paragraphs):
        pos += len(p)
        if i < len(paragraphs) - 1:
            pos += len(sep)
        bounds.append(pos)
    return bounds


def nearest_boundary(bounds: Sequence[int], target_char: int) -> Tuple[int, int]:
    """(index into bounds, char offset) of the paragraph boundary closest to target_char."""
    best = min(range(len(bounds)), key=lambda i: (abs(bounds[i] - target_char), i))
    return best, bounds[best]


def insert_at_depth(paragraphs: Sequence[str], needle: str, depth: float, sep: str = "\n\n") -> Tuple[str, Tuple[int, int], int]:
    """Insert `needle` as its own paragraph at the boundary nearest to depth*len(background).

    Returns (text, (char_start, char_end) of the needle, boundary index).  depth 0 -> before the first
    paragraph, depth 1 -> after the last one.  Never cuts inside a paragraph (report §5.4)."""
    if not paragraphs:
        return needle, (0, len(needle)), 0
    background = sep.join(paragraphs)
    bounds = paragraph_boundaries(paragraphs, sep)
    target = int(round(min(max(depth, 0.0), 1.0) * len(background)))
    idx, _ = nearest_boundary(bounds, target)
    before = paragraphs[:idx]
    after = paragraphs[idx:]
    parts = list(before) + [needle] + list(after)
    text = sep.join(parts)
    start = len(sep.join(before)) + (len(sep) if before else 0)
    return text, (start, start + len(needle)), idx


def char_ratio_to_tokens(text: str, span: Tuple[int, int], offsets: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    """Token span [a, b) whose characters cover the char span (from a tokenizer offset mapping)."""
    a = next((i for i, (s, e) in enumerate(offsets) if e > span[0]), len(offsets))
    b = next((i for i, (s, e) in enumerate(offsets) if s >= span[1]), len(offsets))
    return a, max(b, a + 1 if a < len(offsets) else a)


def insert_many(paragraphs: Sequence[str], items: Sequence[Tuple[int, str]], sep: str = "\n\n") -> Tuple[str, List[Tuple[int, int]]]:
    """Insert several texts, each as its own paragraph, at paragraph-boundary INDICES (0..len(paragraphs)).

    `items` = [(boundary_index, text), ...]; several items may share a boundary (kept in the given order).
    Returns (text, char spans aligned with `items`)."""
    by_b: Dict[int, List[int]] = {}
    for j, (b, _) in enumerate(items):
        by_b.setdefault(max(0, min(int(b), len(paragraphs))), []).append(j)
    parts: List[Tuple[str, Optional[int]]] = []            # (text, item index or None)
    for i in range(len(paragraphs) + 1):
        for j in by_b.get(i, []):
            parts.append((items[j][1], j))
        if i < len(paragraphs):
            parts.append((paragraphs[i], None))
    spans: List[Tuple[int, int]] = [(0, 0)] * len(items)
    pos = 0
    for n, (t, j) in enumerate(parts):
        if j is not None:
            spans[j] = (pos, pos + len(t))
        pos += len(t) + (len(sep) if n < len(parts) - 1 else 0)
    return sep.join(t for t, _ in parts), spans


def boundary_for_depth(paragraphs: Sequence[str], depth: float, sep: str = "\n\n") -> int:
    """Boundary index (0..len) nearest to depth * len(background) — the single-needle rule of insert_at_depth."""
    if not paragraphs:
        return 0
    bounds = paragraph_boundaries(paragraphs, sep)
    target = int(round(min(max(depth, 0.0), 1.0) * len(sep.join(paragraphs))))
    idx, _ = nearest_boundary(bounds, target)
    return idx
