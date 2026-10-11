"""Snippet pools and background documents, read offline (compute nodes have no network).

Snippet record: {sid, text, cls, source, group, holdout, split?, meta}
    sid      "<source>:<source_id>" (unique)
    group    split unit: near-duplicates / same template / same source document never straddle splits
    holdout  split_mode=source puts the snippet in test only (LG public test sources, LLMail-Inject)
    split    preset split (mined paragraphs inherit the split of their document); None = assigned by group hash

Resolution of a registry source (same order as the S1/S2 loaders):
    $LG_SOURCES/<registry>/revision.json local_path  ->  $LG_SOURCES/<registry>/ itself (url downloads)
    ->  HF hub cache snapshot of hf_repo
"""
from __future__ import annotations

import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from lgs1.zeroshot.datasets import _find_files, _hub_cache_dir
from lgs2.textutil import exact_key, group_ids, norm_text, split_paragraphs

from . import synth
from .config import AGPaths, SNIP_SOURCES, SnipSource


def _log(msg: str) -> None:
    print(f"[sources] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------
# file helpers
# ----------------------------------------------------------------------------------------------
def _read_rows(path: Path) -> List[Dict]:
    p = str(path)
    if p.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(p).to_dict("records")
    if p.endswith(".csv"):
        import csv
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if p.endswith(".jsonl") or p.endswith(".jsonl.gz"):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if p.endswith(".json"):
        raw = Path(p).read_text(encoding="utf-8")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return [json.loads(l) for l in raw.splitlines() if l.strip()]
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("data", "rows", "examples", "submissions"):
                if isinstance(obj.get(k), list):
                    return obj[k]
            return [obj]
    raise ValueError(f"unsupported file {p}")


def registry_files(src: SnipSource, P: AGPaths) -> List[Path]:
    roots: List[Path] = []
    rev = P.sources / src.registry / "revision.json"
    if rev.exists():
        try:
            lp = json.loads(rev.read_text()).get("local_path")
            if lp and Path(lp).exists():
                roots.append(Path(lp))
        except Exception:
            pass
    if (P.sources / src.registry).is_dir():
        roots.append(P.sources / src.registry)
    if src.hf_repo:
        hub = _hub_cache_dir(src.hf_repo)
        if hub:
            roots.append(hub)
    for root in roots:
        files = [root] if root.is_file() else _find_files(root, list(src.file_globs))
        files = [f for f in files if f.name != "revision.json"]
        if files:
            return files
    return []


def _first(r: Dict, keys: Sequence[str]):
    for k in keys:
        v = r.get(k)
        if v is not None and not (isinstance(v, float) and v != v) and str(v).strip():
            return v
    return None


# ----------------------------------------------------------------------------------------------
# loaders: files -> [{source_id, text, group?, holdout?, meta}]
# ----------------------------------------------------------------------------------------------
def load_bipia(files: List[Path]) -> List[Dict]:
    """BIPIA benchmark/{text,code}_attack_{train,test}.json: {category: [5 attack instructions]}."""
    out = []
    for f in files:
        obj = json.loads(Path(f).read_text(encoding="utf-8"))
        split = "test" if "test" in f.name else "train"
        kind = "code" if "code" in f.name else "text"
        for cat, items in obj.items():
            for i, t in enumerate(items):
                out.append({"source_id": f"{kind}-{split}-{cat}-{i}", "text": str(t), "group": f"bipia:{kind}:{cat}",
                            "meta": {"category": cat, "bipia_split": split, "kind": kind}})
    return out


def load_text_label(files: List[Path]) -> List[Dict]:
    """Generic (text, label) tables; rows with a label keep label == 1 only, unlabeled tables keep every row."""
    out = []
    for f in files:
        for i, r in enumerate(_read_rows(f)):
            text = _first(r, ("text", "prompt", "Text", "input", "question"))
            if text is None:
                continue
            lab = _first(r, ("label", "Label", "is_injection", "injection"))
            if lab is not None and str(lab).strip().lower() not in ("1", "true", "yes", "injection", "jailbreak"):
                continue
            out.append({"source_id": f"{f.stem}-{i}", "text": str(text), "meta": {"file": f.name}})
    return out


def _llmail_success(obj) -> Optional[bool]:
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    if isinstance(obj, dict) and obj:
        return all(bool(v) for v in obj.values())
    return None


def load_llmail(files: List[Path]) -> List[Dict]:
    """LLMail-Inject submissions (subject + body of the attacker's email; schema read defensively)."""
    out = []
    for f in files:
        try:
            rows = _read_rows(f)
        except Exception as e:
            _log(f"llmail: skip {f.name} ({type(e).__name__})")
            continue
        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                continue
            body = _first(r, ("body", "Body", "email_body", "text"))
            if body is None:
                continue
            subj = _first(r, ("subject", "Subject", "email_subject"))
            text = f"Subject: {subj}\n\n{body}" if subj else str(body)
            out.append({"source_id": f"{f.stem}-{i}", "text": text,
                        "meta": {"scenario": _first(r, ("scenario", "level", "job_id")),
                                 "success": _llmail_success(r.get("objectives"))}})
    return out


def load_notinject(files: List[Path]) -> List[Dict]:
    out = []
    for f in files:
        for i, r in enumerate(_read_rows(f)):
            text = _first(r, ("prompt", "text"))
            if text is None:
                continue
            words = r.get("word_list")
            out.append({"source_id": f"{f.stem}-{i}", "text": str(text),
                        "meta": {"category": r.get("category"),
                                 "trigger_words": list(words) if isinstance(words, (list, tuple)) else words}})
    return out


LOADERS = {"load_bipia": load_bipia, "load_text_label": load_text_label, "load_llmail": load_llmail,
           "load_notinject": load_notinject}


# ----------------------------------------------------------------------------------------------
# LG v2 needle pool (harm / harm_hard)
# ----------------------------------------------------------------------------------------------
def load_lg_pool(P: AGPaths) -> Tuple[List[Dict], str]:
    """Rows {id, text, label, source, pool, group_id} from $LG_DATA_V2/needles/pool.jsonl.gz, else read the raw needle
    sources with the S2 loaders in memory (nothing is written to the LG data directory)."""
    f = P.data_v2 / "needles" / "pool.jsonl.gz"
    if f.exists():
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()], str(f)
    from lgs2 import needles as N
    from lgs2.config import NEEDLE_SOURCES, S2Paths
    S2 = S2Paths.load()
    rows: List[Dict] = []
    for src in NEEDLE_SOURCES:
        files = N.source_files(src, S2)
        if not files:
            continue
        try:
            items = N.LOADERS[src.loader](files)
        except Exception as e:
            _log(f"lg needle source {src.name}: {type(e).__name__}: {e}")
            continue
        for it in items:
            rows.append({"id": f"{src.name}:{it['source_id']}", "text": it["text"], "label": it["label"],
                         "source": src.name, "pool": src.pool, "group_id": None})
    return rows, "raw S2 needle sources (pool.jsonl.gz not built)"


# ----------------------------------------------------------------------------------------------
# background documents
# ----------------------------------------------------------------------------------------------
def _iter_jsonl_gz(path: Path) -> Iterator[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_backgrounds(P: AGPaths, domains: Sequence[str], max_docs: int = 4000, max_chars: int = 60000,
                     seed: int = 0) -> Tuple[List[Dict], Dict[str, str]]:
    """[{id, domain, text}] per domain: LG v2 haystack pieces first, then S1 streamed docs.  Domains without data are
    skipped (reported in the second return value)."""
    rng = random.Random(seed)
    docs: List[Dict] = []
    where: Dict[str, str] = {}
    for dom in domains:
        cands = [(P.data_v2 / "haystack" / dom / "docs.jsonl.gz", "haystack"), (P.sources / dom / "docs.jsonl.gz", "stream")]
        got = None
        for path, kind in cands:
            if path.exists():
                got = (path, kind)
                break
        if got is None:
            where[dom] = "missing"
            continue
        path, kind = got
        recs = []
        for i, r in enumerate(_iter_jsonl_gz(path)):
            if kind == "haystack":
                text = (r.get("sep") or "\n\n").join(r.get("pars") or [])
                did = r.get("id") or f"{dom}-{i}"
            else:
                text = r.get("text") or ""
                did = r.get("id") or f"{dom}-{i}"
            if len(text) < 2000:
                continue
            recs.append({"id": f"{dom}:{did}", "domain": dom, "text": text[:max_chars]})
            if len(recs) >= max_docs * 3:
                break
        rng.shuffle(recs)
        docs += recs[:max_docs]
        where[dom] = f"{kind}:{path} ({min(len(recs), max_docs)} docs)"
    return docs, where


# ----------------------------------------------------------------------------------------------
# paragraph mining: instruction-heavy (instr) and instruction-free (plain) paragraphs
# ----------------------------------------------------------------------------------------------
IMPERATIVE = set("""add apply arrange ask attach avoid bake be bring call change check choose chop clean click close
collect complete configure confirm connect contact copy create cut delete describe disable do don't download drain
drop enable ensure enter fill find fix follow go hand include insert install keep leave let list load log make mark
mix move never note open pack place please plug pour prepare press print put read register remember remove replace
report reset restart return review run save see select send set sign start stay stir stop store submit switch take
tap tell try turn type unplug update upload use verify visit wait wash wear write preheat whisk fold boil simmer
season slice serve heat cook combine beat rinse soak cover reduce sprinkle bake grease attach tighten loosen insert
fill answer show underline cite escort display document notify""".split())
_SENT = re.compile(r"(?<=[.!?])\s+")
_LEAD = re.compile(r"^[\s>*#\-•\d.)(]+")


def sentences(par: str) -> List[str]:
    """Lines with their list markers removed, then split at sentence ends."""
    out: List[str] = []
    for line in par.split("\n"):
        line = _LEAD.sub("", line).strip()
        if line:
            out += [s for s in _SENT.split(line) if s.strip()]
    return out


def imperative_count(par: str) -> Tuple[int, int]:
    """(imperative sentences, sentences).  A sentence is imperative when its first word (after list markers) is a
    base-form verb from IMPERATIVE (or 'please')."""
    sents = sentences(par)
    n_imp = 0
    for s in sents:
        w = s.split()
        if w and w[0].lower().strip(",:;") in IMPERATIVE:
            n_imp += 1
    return n_imp, len(sents)


def _texty(par: str) -> bool:
    if len(par) < 60:
        return False
    letters = sum(c.isalpha() for c in par)
    return letters / max(1, len(par)) > 0.70 and par.count("{") + par.count(";") < 4


def mine_paragraphs(docs: Sequence[Dict], doc_split: Dict[str, str], max_per_doc: int = 4,
                    seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """(instr, plain) snippet candidates; each inherits the split of its source document (group = document)."""
    rng = random.Random(seed)
    instr, plain = [], []
    for d in docs:
        pars = [p.strip() for p in split_paragraphs(d["text"], "blank") if p.strip()]
        rng.shuffle(pars)
        ni = npl = 0
        for k, p in enumerate(pars):
            if not _texty(p):
                continue
            n_imp, n_s = imperative_count(p)
            rec = {"source_id": f"{d['id']}#{k}", "text": p, "group": f"doc:{d['id']}", "split": doc_split.get(d["id"]),
                   "meta": {"domain": d["domain"], "n_imperative": n_imp, "n_sentences": n_s}}
            if n_imp >= 2 and n_imp / max(1, n_s) >= 0.4 and ni < max_per_doc:
                instr.append(rec)
                ni += 1
            elif n_imp == 0 and n_s >= 2 and "?" not in p and npl < max_per_doc:
                plain.append(rec)
                npl += 1
    return instr, plain


# ----------------------------------------------------------------------------------------------
# assemble every class
# ----------------------------------------------------------------------------------------------
def _dedup_group(items: List[Dict], source: str, cls: str, holdout: bool) -> List[Dict]:
    seen, uniq = set(), []
    for it in items:
        k = exact_key(it["text"])
        if not norm_text(it["text"]) or k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    need = [i for i, it in enumerate(uniq) if not it.get("group")]
    if need:
        gids, _ = group_ids([uniq[i]["text"] for i in need])
        for i, g in zip(need, gids):
            uniq[i]["group"] = f"{source}:{g}"
    return [{"sid": f"{source}:{it['source_id']}", "text": it["text"], "cls": cls, "source": source,
             "group": it["group"], "holdout": bool(it.get("holdout", holdout)), "split": it.get("split"),
             "meta": it.get("meta", {})} for it in uniq]


def collect_snippets(P: AGPaths, docs: Sequence[Dict], doc_split: Dict[str, str], smoke: bool = False,
                     seed: int = 0, only: Optional[Iterable[str]] = None) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """{cls: [snippet]} and a per-source report.  smoke=True uses placeholder classes (no files read)."""
    by_cls: Dict[str, List[Dict]] = defaultdict(list)
    report: Dict[str, Dict] = {}
    if smoke:
        for cls in ("harm", "harm_hard", "inj", "notinject"):
            by_cls[cls] += _dedup_group(synth.smoke_snippets(cls, 80, seed), f"smoke_{cls}", cls, False)
        by_cls["instr"] += _dedup_group(synth.gen_instructions(4, seed), "gen_instr", "instr", False)
        mi, mp = mine_paragraphs(docs, doc_split, seed=seed)
        by_cls["plain"] += _dedup_group(mp, "bg_plain", "plain", False)
        for c, v in by_cls.items():
            report[c] = {"status": "smoke", "n": len(v)}
        return dict(by_cls), report
    want = set(only) if only else None
    mined = None
    lg_cache = None
    for src in SNIP_SOURCES:
        if want and src.name not in want:
            continue
        try:
            if src.kind == "lg_pool":
                lg_cache = lg_cache or load_lg_pool(P)
                rows, where = lg_cache
                lab = "harmful" if src.cls == "harm" else "benign"
                items = [{"source_id": r["id"], "text": r["text"], "group": f"lg:{r['group_id']}" if r.get("group_id") else None,
                          "holdout": r.get("pool") == "test", "meta": {"lg_source": r.get("source"), "pool": r.get("pool")}}
                         for r in rows if r.get("label") == lab]
            elif src.kind == "builtin":
                items, where = synth.gen_instructions(12, seed), "agdepth.synth"
            elif src.kind == "mined":
                if mined is None:
                    mined = mine_paragraphs(docs, doc_split, seed=seed)
                items, where = (mined[0] if src.cls == "instr" else mined[1]), "background paragraphs"
            else:
                files = registry_files(src, P)
                if not files:
                    report[src.name] = {"status": "missing", "hint": f"python depthctl.py download --only {src.registry}"}
                    _log(f"{src.name:12s} MISSING ({src.registry}: python depthctl.py download --only {src.registry})")
                    continue
                items, where = LOADERS[src.loader](files), f"{len(files)} files under {files[0].parent}"
            if src.max_items and len(items) > src.max_items:
                rng = random.Random(f"{seed}-{src.name}")
                items = rng.sample(items, src.max_items)
            recs = _dedup_group(items, src.name, src.cls, src.holdout)
            by_cls[src.cls] += recs
            report[src.name] = {"status": "ok" if recs else "empty", "cls": src.cls, "n": len(recs), "where": where,
                                "n_groups": len({r["group"] for r in recs})}
            _log(f"{src.name:12s} {len(recs):7,d} snippets ({src.cls}) <- {where}")
        except Exception as e:
            report[src.name] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            _log(f"{src.name:12s} ERROR {type(e).__name__}: {e}")
    return dict(by_cls), report


def class_summary(by_cls: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    return {c: {"n": len(v), "by_source": dict(Counter(r["source"] for r in v))} for c, v in sorted(by_cls.items())}
