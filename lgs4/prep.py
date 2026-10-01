"""S4 data prep — tokenise the sealed v2 rows ONCE into packed token arrays (train/val/calib subsets).

    python -m lgs4.prep                          # every subset S4 needs, Qwen3 tokenizer
    python -m lgs4.prep --tok llama3             # T5 (Llama Guard 3-8B backbone): same rows, Llama 3 ids -> s4/prep_llama3/
    python -m lgs4.prep --sets train/main,val/main --tok <tokenizer dir>   (tests)

Output per subset  $LG_OUT/s4/prep[_<family>]/<split>/<subset>/
    ids.npy        uint32, all rows concatenated
    offsets.npy    int64 [n_rows + 1]; row i = ids[offsets[i]:offsets[i+1]]
    meta.jsonl.gz  one line per row: id, label (1 harmful / 0 benign), L_target, depth, needle_tok_span, pair_id,
                   needle_id, twin_id, haystack_domain, subset, split, n_tok
    prep.json      counts, tokenizer, data-manifest sha, span-check result
The needle token span is recomputed here from the character span and cross-checked against the D8 file
tok/<family>/<split>/<subset>/spans.jsonl.gz when it exists (E0-2: the training grid must equal the sealed grid).
Rows are read in file order; nothing is shuffled here (seeds act in the trainers).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lgs1.lineage import sha256_file
from lgs2.config import TOKENIZERS
from lgs2.tok import HFTok, WhitespaceTok, token_span_for_chars
from lgs4.config import CALIB_L, CALIB_SETS, NATIVE_SETS, PRESENCE_SETS, S4Paths, VAL_SETS

ALL_SETS = PRESENCE_SETS + NATIVE_SETS + VAL_SETS + CALIB_SETS
META_KEYS = ["id", "label", "L_target", "depth", "needle_tok_span", "pair_id", "needle_id", "twin_id", "haystack_domain",
             "subset", "split", "n_tok"]


def load_tok(name_or_dir: str):
    if name_or_dir in ("whitespace", "ws"):
        return WhitespaceTok()
    return HFTok(TOKENIZERS.get(name_or_dir, name_or_dir), name_or_dir if name_or_dir in TOKENIZERS else "custom")


def iter_rows(path: Path) -> Iterable[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_ref_spans(P: S4Paths, split: str, subset: str, family: str = "qwen3") -> Dict[str, Dict]:
    f = P.data.root / "tok" / family / split / subset / "spans.jsonl.gz"
    if not f.exists():
        return {}
    return {r["id"]: r for r in iter_rows(f)}


def prep_subset(P: S4Paths, split: str, subset: str, tok, L_filter: Optional[Sequence[int]] = None, limit: Optional[int] = None,
                check_ref: bool = True) -> Dict:
    src = P.data.synth / split / subset / "rows.jsonl.gz"
    if not src.exists():
        raise SystemExit(f"{src} missing (S3 synthesis not finished?)")
    out = P.prep / split / subset
    out.mkdir(parents=True, exist_ok=True)
    family = getattr(tok, "name", "qwen3")
    ref = load_ref_spans(P, split, subset, family if family in TOKENIZERS else "qwen3") if check_ref else {}
    ids_chunks: List[np.ndarray] = []
    offsets = [0]
    metas: List[Dict] = []
    mism = {"n_tok": 0, "span": 0, "checked": 0}
    t0 = time.time()
    n = 0
    for r in iter_rows(src):
        if L_filter and r.get("L_target") not in L_filter:
            continue
        ids, offs = tok.encode_with_offsets(r["text"])
        span = None
        if r.get("needle_char_span"):
            a, b = token_span_for_chars(offs, (r["needle_char_span"][0], r["needle_char_span"][1]))
            span = [int(a), int(b)]
        if ref and r["id"] in ref:
            rr = ref[r["id"]]
            mism["checked"] += 1
            if int(rr["n_tok"]) != len(ids):
                mism["n_tok"] += 1
            if (rr.get("needle_tok_span") or None) != span:
                mism["span"] += 1
        ids_chunks.append(np.asarray(ids, dtype=np.uint32))
        offsets.append(offsets[-1] + len(ids))
        metas.append({"id": r["id"], "label": 1 if r["label"] == "harmful" else 0, "L_target": r.get("L_target"),
                      "depth": r.get("depth"), "needle_tok_span": span, "pair_id": r.get("pair_id"), "needle_id": r.get("needle_id"),
                      "twin_id": r.get("twin_id"), "haystack_domain": r.get("haystack_domain"), "subset": subset, "split": split,
                      "n_tok": len(ids)})
        n += 1
        if n % 5000 == 0:
            print(f"   {split}/{subset}: {n:,} rows  {time.time() - t0:.0f}s", flush=True)
        if limit and n >= limit:
            break
    flat = np.concatenate(ids_chunks) if ids_chunks else np.zeros((0,), dtype=np.uint32)
    np.save(out / "ids.npy", flat)
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    with gzip.open(out / "meta.jsonl.gz", "wt", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    man = P.data.manifests / f"{split}.{subset}.manifest.json"
    info = {"split": split, "subset": subset, "rows": n, "tokens": int(flat.shape[0]), "tokenizer": getattr(tok, "name", "?"),
            "tokenizer_repo": getattr(tok, "repo_id", None), "L_filter": list(L_filter) if L_filter else None,
            "data_manifest_sha": sha256_file(man) if man.exists() else None, "source_rows": str(src),
            "span_check": mism, "seconds": round(time.time() - t0, 1), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    bad = mism["n_tok"] + mism["span"]
    info["span_check"]["pass"] = (bad == 0) if mism["checked"] else None
    (out / "prep.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[prep] {split}/{subset}: rows={n:,} tokens={flat.shape[0]:,} span_check={mism} {info['seconds']}s", flush=True)
    if mism["checked"] and bad > max(1, int(0.001 * mism["checked"])):
        raise SystemExit(f"[prep] {split}/{subset}: {bad} rows disagree with tok/{family} spans (tokenizer mismatch?) -> refusing")
    return info


class PrepSet:
    """Memory-mapped access to one prepped subset."""

    def __init__(self, P: S4Paths, split: str, subset: str):
        d = P.prep / split / subset
        if not (d / "ids.npy").exists():
            raise SystemExit(f"{d} not prepared (python s4ctl.py prep)")
        self.ids = np.load(d / "ids.npy", mmap_mode="r")
        self.offsets = np.load(d / "offsets.npy")
        self.meta: List[Dict] = list(iter_rows(d / "meta.jsonl.gz"))
        self.info = json.loads((d / "prep.json").read_text(encoding="utf-8"))
        self.split, self.subset = split, subset

    def __len__(self) -> int:
        return len(self.meta)

    def row_ids(self, i: int) -> np.ndarray:
        return np.asarray(self.ids[self.offsets[i]:self.offsets[i + 1]], dtype=np.int64)

    def rows(self, L: Optional[int] = None, label: Optional[int] = None) -> List[int]:
        return [i for i, m in enumerate(self.meta) if (L is None or m["L_target"] == L) and (label is None or m["label"] == label)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S4 prep: tokenise v2 rows into packed arrays")
    ap.add_argument("--sets", default="all", help="comma list split/subset (default: all S4 sets)")
    ap.add_argument("--tok", default="qwen3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--calib-L", default=str(CALIB_L), help="calib rows kept (comma list of L); 0 = all")
    ap.add_argument("--no-ref-check", action="store_true")
    a = ap.parse_args(argv)
    P = S4Paths.load().ensure().for_tok(a.tok if a.tok in TOKENIZERS else "qwen3")
    P.prep.mkdir(parents=True, exist_ok=True)
    tok = load_tok(a.tok)
    sets = ALL_SETS if a.sets == "all" else [tuple(s.split("/")) for s in a.sets.split(",") if s]
    calib_L = [int(x) for x in a.calib_L.split(",") if x and int(x) > 0]
    for split, subset in sets:
        prep_subset(P, split, subset, tok, L_filter=calib_L if (split == "calib" and calib_L) else None, limit=a.limit,
                    check_ref=not a.no_ref_check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
