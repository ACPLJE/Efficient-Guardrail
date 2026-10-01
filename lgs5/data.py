"""Row sources for S5 (all from the sealed lg_data_v2 and the needle pool).

Every yielded row has the lgs2 synth keys the runners need: id, split, subset, L_target, depth, label ('harmful'|'benign'),
text, needle_char_span (None for benign-long / native rows), needle_id, pair_id, haystack_domain, meta.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from lgs2.config import S2Paths
from lgs2.needles import load_pool

from .config import E1_SOURCES


def iter_synth(P: S2Paths, split: str, subset: str) -> Iterator[Dict]:
    f = P.synth / split / subset / "rows.jsonl.gz"
    if not f.exists():
        raise SystemExit(f"{f} missing (S3 synthesis)")
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def e2_rows(P: S2Paths, Ls: Sequence[int], split: str = "test", subsets: Sequence[str] = ("main", "benign_long"),
            depths: Optional[Sequence[float]] = None, max_pairs: Optional[int] = None) -> Iterator[Dict]:
    """test/main + test/benign_long rows at the given lengths (E2-1), optionally a positions/pairs subset."""
    want = set(int(L) for L in Ls)
    dset = set(round(float(d), 4) for d in depths) if depths else None
    for sub in subsets:
        for r in iter_synth(P, split, sub):
            if r.get("L_target") not in want:
                continue
            if dset is not None and r.get("depth") is not None and round(float(r["depth"]), 4) not in dset:
                continue
            if max_pairs is not None and r.get("pair_id"):
                try:
                    if int(r["pair_id"].rsplit("-p", 1)[1]) >= max_pairs:
                        continue
                except (ValueError, IndexError):
                    pass
            yield r


def calib_rows(P: S2Paths, Ls: Sequence[int], subsets: Sequence[str] = ("benign_long",)) -> Iterator[Dict]:
    """calib benign-long rows per length: the matched-FPR thresholds (E0-3) come from these."""
    yield from e2_rows(P, Ls, split="calib", subsets=subsets)


# keys of a variant row (lgs3.variants) that the eval rows carry through for the S6 tables
VARIANT_KEYS = ("variant", "needle_ids", "needle_char_spans", "n_decoys", "n_pieces", "piece_gap_tokens", "frame", "frame_requested",
                "multi_config", "related_m", "n_repeats", "decoy_source", "needle_share", "n_tok_needle", "twin_len_ratio")


def variant_rows(P: S2Paths, subsets: Sequence[str], Ls: Optional[Sequence[int]] = None, max_rows_per_subset: Optional[int] = None) -> List[Dict]:
    """test/<variant> rows (S6 E2-4 / E3-2 / E3-3 and the S8-S9 robustness subsets), in file order, optionally restricted
    to a length list.  Missing subsets raise (D7 must have run)."""
    out: List[Dict] = []
    want = set(int(L) for L in Ls) if Ls else None
    for sub in subsets:
        n = 0
        for r in iter_synth(P, "test", sub):
            if want is not None and r.get("L_target") not in want:
                continue
            out.append(r)
            n += 1
            if max_rows_per_subset and n >= max_rows_per_subset:
                break
    return out


def e1_rows(P: S2Paths, sources: Sequence[str] = E1_SOURCES) -> List[Dict]:
    """Short-prompt benches = every unique row of the test-pool sources (the needle pool keeps them all with their
    labels and source meta; `usable` only marks the <=128-token ones used as needles)."""
    rows = []
    for r in load_pool(P):
        if r.get("pool") != "test" or r.get("source") not in sources:
            continue
        rows.append({"id": r["id"], "split": "native", "subset": r["source"], "L_target": None, "depth": None,
                     "label": r["label"], "text": r["text"], "needle_char_span": None, "needle_id": r["id"], "pair_id": None,
                     "haystack_domain": None, "meta": {"category": r.get("category"), "source_split": r.get("source_split"),
                                                       **(r.get("meta") or {})}})
    return rows


def needle_texts(P: S2Paths, needle_ids: Iterable[str]) -> Dict[str, Dict]:
    want = set(needle_ids)
    return {r["id"]: r for r in load_pool(P) if r["id"] in want}


def e27_needle_rows(P: S2Paths, Ls: Sequence[int]) -> List[Dict]:
    """E2-7 (i): each needle of the test/main pairs alone (harmful + benign twin), one row per needle id."""
    seen: Dict[str, Dict] = {}
    for r in e2_rows(P, Ls, subsets=("main",)):
        nid = r.get("needle_id")
        if nid and nid not in seen:
            seen[nid] = r
    texts = needle_texts(P, seen)
    out = []
    for nid, r in seen.items():
        nd = texts.get(nid)
        text = nd["text"] if nd else r["text"][r["needle_char_span"][0]:r["needle_char_span"][1]]
        out.append({"id": f"needle-{nid}", "split": "test", "subset": "needle_only", "L_target": None, "depth": None,
                    "label": r["label"], "text": text, "needle_char_span": [0, len(text)], "needle_id": nid, "pair_id": r.get("pair_id"),
                    "haystack_domain": None, "meta": {}})
    return out
