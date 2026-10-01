"""D5a — the needle pool (report §5.2).

    python -m lgs2.needles                    # -> needles/pool.jsonl.gz + needles/pool_stats.json

One row per prompt from the 13 public sources, all read from the offline HF cache (S1 D3):
    id, text, label (harmful|benign), source, source_split, pool (train|test), category, meta,
    n_tok_qwen3, n_tok_llama3, bucket (le32|33_64|65_128|129_512|gt512), long_needle, usable,
    group_id (normalised-text hash + MinHash 5-gram Jaccard >= 0.8 clusters, computed over the whole pool)
Public test sources are `pool=test` and never enter train/val/calib (splits.py).  Sizes are compared with the
published counts (warning only: e.g. WildGuardTest 1,725 prompt-labelled rows of 1,756).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from lgs1.zeroshot.datasets import _external_dir, _find_files, _hub_cache_dir, _snapshot_dir
from lgs1.zeroshot.datasets import _read_any as _read_any_s1
from .config import (LENGTH_BUCKETS, LONG_NEEDLE_MAX_TOK, MINHASH, NEEDLE_MAX_TOK, NEEDLE_SOURCES, ORBENCH_80K_SAMPLE,
                     NeedleSource, S2Paths)
from .textutil import exact_key, group_ids, norm_text
from .tok import get_tok

HARM = {"harmful", "unsafe", "toxic", "1", "true", "yes"}
SAFE = {"unharmful", "benign", "safe", "0", "false", "no"}


def _read_any(path: Path) -> List[Dict]:
    """parquet / csv / jsonl(.gz) via the S1 reader; plain .json may be an array, {"data": [...]}, or JSON lines."""
    p = str(path)
    if p.endswith(".json"):
        raw = Path(p).read_text(encoding="utf-8")
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for k in ("data", "rows", "examples"):
                    if isinstance(obj.get(k), list):
                        return obj[k]
                return [obj]
        except json.JSONDecodeError:
            return [json.loads(l) for l in raw.splitlines() if l.strip()]
    return _read_any_s1(path)


def _label(v) -> Optional[str]:
    s = str(v).strip().lower()
    if s in HARM:
        return "harmful"
    if s in SAFE:
        return "benign"
    return None


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _flag(v) -> Optional[bool]:
    """bool or None (pandas NaN / empty -> None; never bool(nan) == True)."""
    if v is None or (isinstance(v, float) and v != v) or str(v).strip() == "":
        return None
    if isinstance(v, bool):
        return v
    return _truthy(v)


def _split_of(path: Path) -> str:
    """Split name from the file name, else from the parent directory name (HF cache layout)."""
    for part in (Path(path).name.lower(), Path(path).parent.name.lower()):
        for k in ("validation", "valid", "dev", "train", "test"):
            if k in part:
                return "validation" if k in ("validation", "valid", "dev") else k
    return "unknown"


# ----------------------------------------------------------------------------------------------
# loaders: files -> [{source_id, text, label, category, meta, source_split}]
# ----------------------------------------------------------------------------------------------
def load_wildguard(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            lab = _label(r.get("prompt_harm_label"))
            text = r.get("prompt")
            if not text or lab is None:
                continue
            out.append({"source_id": f"{f.stem}-{i}", "text": str(text), "label": lab, "category": r.get("subcategory"),
                        "source_split": _split_of(f),
                        "meta": {"adversarial": _flag(r.get("adversarial")),
                                 "response_present": bool(r.get("response")) and str(r.get("response")) != "nan"}})
    return out


def load_aegis2(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            cats = str(r.get("violated_categories") or "")
            lab = _label(r.get("prompt_label"))
            text = r.get("prompt")
            if not text or lab is None or "needs caution" in cats.lower():
                continue
            out.append({"source_id": str(r.get("id") or f"{f.stem}-{i}"), "text": str(text), "label": lab,
                        "category": cats or None, "source_split": _split_of(f),
                        "meta": {"prompt_label_source": r.get("prompt_label_source")}})
    return out


def load_toxicchat(files: List[Path], human_only: Optional[bool] = None) -> List[Dict]:
    out = []
    for f in files:
        split = _split_of(f)
        ho = human_only if human_only is not None else (split == "test")
        for i, r in enumerate(_read_any(f)):
            text = r.get("user_input")
            lab = _label(r.get("toxicity"))
            if not text or lab is None:
                continue
            ha = r.get("human_annotation")
            if ho and ha not in (None, "") and not _truthy(ha):
                continue
            out.append({"source_id": str(r.get("conv_id") or f"{f.stem}-{i}"), "text": str(text), "label": lab,
                        "category": None, "source_split": split,
                        "meta": {"jailbreaking": int(r["jailbreaking"]) if r.get("jailbreaking") not in (None, "") else None,
                                 "human_annotation": _truthy(ha) if ha not in (None, "") else None}})
    return out


def load_orbench_80k(files: List[Path]) -> List[Dict]:
    rows = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            if r.get("prompt"):
                rows.append({"source_id": f"{f.stem}-{i}", "text": str(r["prompt"]), "label": "benign",
                             "category": r.get("category"), "source_split": "train", "meta": {"hard_benign": True}})
    # deterministic 10,000-row sample (hash of the text, not file order)
    rows.sort(key=lambda r: exact_key(r["text"]))
    rng = random.Random(MINHASH["seed"])
    rng.shuffle(rows)
    return rows[:ORBENCH_80K_SAMPLE]


def load_orbench_hard(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            if r.get("prompt"):
                out.append({"source_id": f"{f.stem}-{i}", "text": str(r["prompt"]), "label": "benign",
                            "category": r.get("category"), "source_split": "test", "meta": {"over_refusal": True}})
    return out


def load_harmbench(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        cfg = next((k for k in ("standard", "contextual", "copyright") if k in str(f).lower()), "unknown")
        if cfg == "copyright":
            continue
        for i, r in enumerate(_read_any(f)):
            text = r.get("prompt") or r.get("behavior") or r.get("Behavior")
            if not text:
                continue
            out.append({"source_id": f"{cfg}-{i}", "text": str(text), "label": "harmful",
                        "category": r.get("category") or r.get("SemanticCategory"), "source_split": "test",
                        "meta": {"config": cfg, "context_present": bool(r.get("context"))}})
    return out


def load_sst(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            if r.get("prompt"):
                out.append({"source_id": str(r.get("id") or f"sst-{i}"), "text": str(r["prompt"]), "label": "harmful",
                            "category": r.get("harm_area"), "source_split": "test", "meta": {"sst_category": r.get("category")}})
    return out


def load_jbb(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        name = str(f).lower()
        if "judge" in name:
            continue
        lab = "harmful" if "harmful" in name else ("benign" if "benign" in name else None)
        for i, r in enumerate(_read_any(f)):
            text = r.get("Goal") or r.get("goal")
            l2 = lab or _label(r.get("label"))
            if not text or l2 is None:
                continue
            out.append({"source_id": f"{'harmful' if l2 == 'harmful' else 'benign'}-{r.get('Index', i)}", "text": str(text),
                        "label": l2, "category": r.get("Category"), "source_split": "test",
                        "meta": {"behavior": r.get("Behavior"), "jbb_source": r.get("Source")}})
    return out


def load_openai_mod(files: List[Path]) -> List[Dict]:
    flags = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]
    out = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            text = r.get("prompt")
            if not text:
                continue
            on = [k for k in flags if k in r and _truthy(r[k])]
            out.append({"source_id": f"oaimod-{i}", "text": str(text), "label": "harmful" if on else "benign",
                        "category": ",".join(on) or None, "source_split": "test", "meta": {}})
    return out


def load_xstest(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            text, lab = r.get("prompt"), _label(r.get("label"))
            if not text or lab is None:
                continue
            out.append({"source_id": str(r.get("id") or f"xstest-{i}"), "text": str(text), "label": lab,
                        "category": r.get("type"), "source_split": "test", "meta": {"focus": r.get("focus"), "note": r.get("note")}})
    return out


def load_aegis1(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        for i, r in enumerate(_read_any(f)):
            if str(r.get("text_type") or "").lower() not in ("user_message", "user message", "prompt"):
                continue
            labels = [str(r.get(f"labels_{k}") or "").strip() for k in range(5)]
            labels = [x for x in labels if x and x.lower() != "none"]
            if not labels:
                continue
            c = Counter("safe" if x.lower() == "safe" else ("caution" if "caution" in x.lower() else "unsafe") for x in labels)
            top, n_top = c.most_common(1)[0]
            if top == "caution" or list(c.values()).count(n_top) > 1:
                continue
            out.append({"source_id": str(r.get("id") or f"aegis1-{i}"), "text": str(r.get("text") or "").strip(),
                        "label": "harmful" if top == "unsafe" else "benign",
                        "category": ";".join(sorted({x for x in labels if x.lower() not in ("safe",)})) or None,
                        "source_split": _split_of(f), "meta": {"n_annotations": r.get("num_annotations")}})
    return [o for o in out if o["text"]]


LOADERS: Dict[str, Callable[[List[Path]], List[Dict]]] = {
    "load_wildguard": load_wildguard, "load_aegis2": load_aegis2, "load_toxicchat": load_toxicchat,
    "load_orbench_80k": load_orbench_80k, "load_orbench_hard": load_orbench_hard, "load_harmbench": load_harmbench,
    "load_sst": load_sst, "load_jbb": load_jbb, "load_openai_mod": load_openai_mod, "load_xstest": load_xstest,
    "load_aegis1": load_aegis1,
}


# ----------------------------------------------------------------------------------------------
# file resolution (offline): sources/<registry>/revision.json -> HF hub cache -> LG_EXTERNAL_DATA/audited
# ----------------------------------------------------------------------------------------------
def source_files(src: NeedleSource, P: S2Paths) -> List[Path]:
    roots: List[Path] = []
    snap = _snapshot_dir(src.registry) if src.registry else None
    if snap:
        roots.append(snap)
    hub = _hub_cache_dir(src.hf_repo)
    if hub and hub not in roots:
        roots.append(hub)
    ext = _external_dir(src.registry or src.name)
    if ext:
        roots.append(ext)
    for root in roots:
        files = _find_files(root, src.file_globs)
        if files:
            return files
    return []


def bucket_of(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return name
    return "gt512"


def build_pool(P: S2Paths, tok_q=None, tok_l=None, sources: Optional[List[str]] = None, limit_per_source: Optional[int] = None) -> Dict:
    P.ensure()
    tok_q = tok_q or get_tok("qwen3")
    if tok_l is None:
        try:
            tok_l = get_tok("llama3")
        except Exception as e:  # gated tokenizer not cached: record None, do not fail
            print(f"[needles] Llama 3 tokenizer unavailable ({type(e).__name__}); n_tok_llama3 = null", file=sys.stderr)
            tok_l = None
    rows: List[Dict] = []
    report: Dict[str, Dict] = {}
    for src in NEEDLE_SOURCES:
        if sources and src.name not in sources:
            continue
        files = source_files(src, P)
        if not files:
            report[src.name] = {"status": "missing", "expected_n": src.expected_n, "hint": f"S1 download --only {src.registry}"}
            print(f"[needles] {src.name:16s} MISSING (no files for {src.hf_repo}; bash slurm/s1_download_login.sh --only {src.registry})", file=sys.stderr)
            continue
        try:
            items = LOADERS[src.loader](files)
        except Exception as e:
            report[src.name] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}", "files": [str(f) for f in files][:5]}
            print(f"[needles] {src.name:16s} ERROR {type(e).__name__}: {e}", file=sys.stderr)
            continue
        # within-source exact dedup (response pairs repeat the prompt); keep the first occurrence
        seen = set()
        uniq = []
        for it in items:
            k = exact_key(it["text"])
            if k in seen or not norm_text(it["text"]):
                continue
            seen.add(k)
            uniq.append(it)
        if limit_per_source:
            uniq = uniq[:limit_per_source]
        for it in uniq:
            rows.append({"id": f"{src.name}:{it['source_id']}", "text": it["text"], "label": it["label"], "source": src.name,
                         "source_split": it.get("source_split", "unknown"), "pool": src.pool, "category": it.get("category"),
                         "meta": it.get("meta", {})})
        report[src.name] = {"status": "ok", "n_rows_read": len(items), "n_unique": len(uniq), "expected_n": src.expected_n,
                            "files": [str(f) for f in files][:6],
                            "n_harmful": sum(1 for it in uniq if it["label"] == "harmful")}
        flag = ""
        if src.expected_n and abs(len(uniq) - src.expected_n) > max(50, 0.05 * src.expected_n):
            flag = f"  (published {src.expected_n:,}; check the loader/files)"
        print(f"[needles] {src.name:16s} {len(uniq):7,d} unique ({report[src.name]['n_harmful']:,} harmful){flag}", flush=True)
    if not rows:
        raise SystemExit("no needle source could be read")
    # tokens
    texts = [r["text"] for r in rows]
    nq = tok_q.count(texts)
    nl = tok_l.count(texts) if tok_l else [None] * len(rows)
    for r, a, b in zip(rows, nq, nl):
        r["n_tok_qwen3"], r["n_tok_llama3"] = int(a), (None if b is None else int(b))
        r["bucket"] = bucket_of(int(a))
        r["long_needle"] = NEEDLE_MAX_TOK < int(a) <= LONG_NEEDLE_MAX_TOK
        r["usable"] = int(a) <= LONG_NEEDLE_MAX_TOK
    # groups over the whole pool
    gids, gstats = group_ids(texts, n=MINHASH["shingle_words"], num_perm=MINHASH["num_perm"], bands=MINHASH["bands"],
                             threshold=MINHASH["jaccard"], seed=MINHASH["seed"])
    for r, g in zip(rows, gids):
        r["group_id"] = g
    out = P.needles / "pool.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = pool_stats(rows)
    stats.update(sources=report, groups=gstats, tokenizer_qwen3=getattr(tok_q, "name", "?"),
                 tokenizer_llama3=getattr(tok_l, "name", None), built_at=time.strftime("%Y-%m-%dT%H:%M:%S"), path=str(out))
    (P.needles / "pool_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def pool_stats(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    by = Counter((r["pool"], r["source"], r["label"], r["bucket"]) for r in rows)
    table: Dict[str, Dict] = {}
    for (pool, src, lab, b), n in sorted(by.items()):
        table.setdefault(f"{pool}/{src}", {}).setdefault(lab, {})[b] = n
    return {"n_total": len(rows), "n_train_pool": sum(1 for r in rows if r["pool"] == "train"),
            "n_test_pool": sum(1 for r in rows if r["pool"] == "test"),
            "n_usable_main": sum(1 for r in rows if r["n_tok_qwen3"] <= NEEDLE_MAX_TOK),
            "n_long_needle": sum(1 for r in rows if r["long_needle"]), "n_too_long": sum(1 for r in rows if not r["usable"]),
            "by_pool_source_label_bucket": table}


def load_pool(P: S2Paths) -> List[Dict]:
    f = P.needles / "pool.jsonl.gz"
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D5a needle pool")
    ap.add_argument("--tok", default="qwen3", help="qwen3 | whitespace (tests)")
    ap.add_argument("--sources", default="", help="comma list (default: all 14)")
    ap.add_argument("--limit-per-source", type=int, default=None)
    a = ap.parse_args(argv)
    P = S2Paths.load()
    tq = get_tok(a.tok)
    tl = None if a.tok == "qwen3" else tq          # qwen3 run: Llama 3 tokenizer is loaded (or skipped) inside
    st = build_pool(P, tq, tl, [x for x in a.sources.split(",") if x] or None, a.limit_per_source)
    print(f"[needles] total={st['n_total']:,} train={st['n_train_pool']:,} test={st['n_test_pool']:,} "
          f"usable(<=128)={st['n_usable_main']:,} long(129-512)={st['n_long_needle']:,} groups={st['groups']['n_groups']:,} "
          f"near-dup merged={st['groups']['n_near_dup_merged']:,}")
    print(f"[written] {st['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
