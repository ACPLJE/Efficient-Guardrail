"""E0-6 benchmark loaders: WildGuardTest (1,725 prompt-labelled rows) and ToxicChat 0124 test, human-annotated (2,853 / 362 toxic).

Row schema returned everywhere:  {"id", "prompt", "label" (1 = harmful/toxic), "meta": {...}}

Resolution order for each bench:
  1. --path given (parquet / jsonl / csv with the known columns, or a directory to search)
  2. $LG_SOURCES/<name>/revision.json -> local_path (D3 snapshot; offline-safe)
  3. $HF_HUB_CACHE/datasets--<org>--<name>/snapshots/*  (already-cached hub files, offline)
  4. $LG_EXTERNAL_DATA/audited/*<name>*  (lg_external_v01 prepared files)
  5. Hugging Face online (login node only)

Label conventions:
  WildGuardTest: prompt_harm_label == "harmful"  -> 1 ; adversarial kept in meta (E1-3 split)
  ToxicChat    : toxicity == 1                    -> 1 ; jailbreaking kept in meta; ONLY data/0124/..._test.csv and
                 human_annotation == True (2,853 rows) — the published protocol; the repo's 1123 copy is ignored
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..paths import S1Paths

BENCHES = ("wgtest", "toxicchat")


def _read_any(path: Path) -> List[Dict]:
    p = str(path)
    if p.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(p).to_dict("records")
    if p.endswith(".csv"):
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if p.endswith(".jsonl") or p.endswith(".jsonl.gz"):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if p.endswith(".json"):
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        return obj if isinstance(obj, list) else obj.get("data", [])
    raise ValueError(f"unsupported file {p}")


def _find_files(root: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out += [Path(x) for x in glob.glob(str(root / "**" / pat), recursive=True)]
    return sorted(set(out))


def _snapshot_dir(name: str) -> Optional[Path]:
    P = S1Paths.load()
    rev = P.sources / name / "revision.json"
    if rev.exists():
        try:
            lp = json.loads(rev.read_text()).get("local_path")
            if lp and Path(lp).exists():
                return Path(lp)
        except Exception:
            return None
    return None


def _hub_cache_dir(repo_id: str) -> Optional[Path]:
    """Newest snapshot of a dataset repo in the local HF hub cache (works with HF_HUB_OFFLINE=1)."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"))
    d = hub / f"datasets--{repo_id.replace('/', '--')}" / "snapshots"
    if not d.exists():
        return None
    snaps = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
    return snaps[-1] if snaps else None


def _external_dir(keyword: str) -> Optional[Path]:
    P = S1Paths.load()
    if P.external_data and (P.external_data / "audited").exists():
        for d in (P.external_data / "audited").iterdir():
            if keyword in d.name.lower():
                return d
    return None


# --------------------------------------------------------------------------------------------
# normalisers
# --------------------------------------------------------------------------------------------
def _norm_wgtest(rows: Iterable[Dict]) -> List[Dict]:
    out = []
    for i, r in enumerate(rows):
        prompt = r.get("prompt") or r.get("text") or r.get("user_input")
        lab = r.get("prompt_harm_label", r.get("label"))
        if prompt is None or lab is None:
            continue
        y = 1 if str(lab).lower() in ("harmful", "1", "true", "unsafe", "toxic") else 0
        adv = r.get("adversarial")
        out.append({"id": f"wgtest-{i}", "prompt": str(prompt), "label": y,
                    "meta": {"adversarial": bool(adv) if adv is not None else None,
                             "subcategory": r.get("subcategory"), "response_present": bool(r.get("response"))}})
    return out


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _norm_toxicchat(rows: Iterable[Dict], human_only: bool = True) -> List[Dict]:
    """ToxicChat protocol of WildGuard/GuardReasoner/Qwen3Guard: 0124 test split, human-annotated rows only
    (2,853 prompts, 362 toxic; the 2,230 non-human-annotated rows are all benign)."""
    out = []
    for i, r in enumerate(rows):
        prompt = r.get("user_input") or r.get("prompt") or r.get("text")
        lab = r.get("toxicity", r.get("label"))
        if prompt is None or lab is None:
            continue
        ha = r.get("human_annotation")
        if human_only and ha not in (None, "") and not _truthy(ha):
            continue
        y = 1 if str(lab).lower() in ("1", "true", "toxic", "harmful", "unsafe") else 0
        jb = r.get("jailbreaking")
        out.append({"id": f"toxicchat-{r.get('conv_id', i)}", "prompt": str(prompt), "label": y,
                    "meta": {"jailbreaking": int(jb) if jb not in (None, "") else None,
                             "human_annotation": (_truthy(ha) if ha not in (None, "") else None)}})
    return out


def _pick_toxicchat_file(cand: List[Path]) -> List[Path]:
    """One file only: prefer data/0124/*test*.csv (the repo also ships the 1123 version)."""
    if not cand:
        return cand
    pref = [c for c in cand if "0124" in str(c)]
    return [pref[0]] if pref else [cand[0]]


# --------------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------------
def load_wgtest(path: Optional[str] = None, allow_online: bool = False) -> List[Dict]:
    cand: List[Path] = []
    if path:
        p = Path(path)
        cand = [p] if p.is_file() else _find_files(p, ["*wildguardtest*.parquet", "*test*.parquet", "*.jsonl", "*.jsonl.gz", "*.csv"])
    if not cand:
        snap = _snapshot_dir("wildguardmix")
        if snap:
            cand = _find_files(snap, ["*wildguardtest*.parquet", "*wildguardtest*/*.parquet", "*test*.parquet"])
    if not cand:
        hub = _hub_cache_dir("allenai/wildguardmix")
        if hub:
            cand = _find_files(hub, ["*wildguardtest*.parquet", "*wildguardtest*/*.parquet", "*test*.parquet"])
    if not cand:
        ext = _external_dir("wildguard")
        if ext:
            cand = _find_files(ext, ["*.parquet", "*.jsonl", "*.jsonl.gz", "*.csv"])
    if cand:
        rows: List[Dict] = []
        for f in cand:
            rows += _read_any(f)
        norm = _norm_wgtest(rows)
        if norm:
            return norm
    if allow_online:
        from datasets import load_dataset
        ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        return _norm_wgtest(ds)
    raise FileNotFoundError("WildGuardTest not found: pass --wgtest PATH, run D3 (allenai/wildguardmix, gated), or set LG_EXTERNAL_DATA")


def load_toxicchat(path: Optional[str] = None, allow_online: bool = False) -> List[Dict]:
    cand: List[Path] = []
    if path:
        p = Path(path)
        cand = [p] if p.is_file() else _find_files(p, ["*test*.csv", "*test*.parquet", "*.jsonl", "*.jsonl.gz"])
    if not cand:
        snap = _snapshot_dir("toxicchat")
        if snap:
            cand = _find_files(snap, ["*0124*test*.csv", "*0124*test*.parquet", "*test*.csv", "*test*.parquet"])
    if not cand:
        hub = _hub_cache_dir("lmsys/toxic-chat")
        if hub:
            cand = _find_files(hub, ["*0124*test*.csv", "*0124*test*.parquet", "*test*.csv", "*test*.parquet"])
    if not cand:
        ext = _external_dir("toxic")
        if ext:
            cand = _find_files(ext, ["*.parquet", "*.jsonl", "*.jsonl.gz", "*.csv"])
    cand = _pick_toxicchat_file(cand)
    if cand:
        rows: List[Dict] = []
        for f in cand:
            rows += _read_any(f)
        norm = _norm_toxicchat(rows)
        if norm:
            return norm
    if allow_online:
        from datasets import load_dataset
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        return _norm_toxicchat(ds)
    raise FileNotFoundError("ToxicChat test not found: pass --toxicchat PATH or run D3 (lmsys/toxic-chat)")


def load_bench(name: str, path: Optional[str] = None, allow_online: bool = False, limit: Optional[int] = None) -> List[Dict]:
    if name == "wgtest":
        rows = load_wgtest(path, allow_online)
    elif name == "toxicchat":
        rows = load_toxicchat(path, allow_online)
    else:
        raise KeyError(name)
    if limit:
        rows = rows[:limit]
    return rows
