"""Login-node downloader for the AG registry (build plan 4.2 S0; `ag/data/registry_ag.py`).

Reuses the LG D3 machinery in `lgs1/download.py` by import (fetchers, refs/main repair, status
roll-up); only the AG paths, the stage filter, the schema probe and the dated snapshot are new.

Writes under $AG_SOURCES (env/ag.env):
  <name>/revision.json     LG record shape + stage, ag_role (+ probe / probe_error)
  <name>/probe.json        github / hf_dataset / hf_stream: file list + key/type samples of data files
  download_status.json     roll-up of every <name>/revision.json (lgs1.download.rebuild_status)
  ag_assets_<MMDD>.json    dated snapshot: status, sha, stage and gated state of every registry entry

Rules (CLAUDE.md 4): login node only. Refuses inside sbatch / without network unless --force-online.
HF_HUB_OFFLINE / HF_DATASETS_OFFLINE are unset for this process only (restored when run() returns).

  python -m ag.data.download_ag --list
  python -m ag.data.download_ag --stage pilot          # T2 + bg_wiki_en (S2 step 0)
  python -m ag.data.download_ag                        # no --stage/--only: pilot + s2; --stage all = everything
  python -m ag.data.download_ag --only bg_wiki_en --no-skip
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import socket
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lgs1 import download as _lgd
from lgs1.download import _online_ok, _write_rev, fix_refs, rebuild_status   # _online_ok is patched in tests

from ag.data.registry_ag import BASELINES, DATASETS, MODELS, STAGES, AGEntry, all_entries, validate

REPO = Path(__file__).resolve().parents[2]
ENV_FILE = REPO / "env" / "ag.env"
LOGIN_HOST = "gsai-login-3"

# kind -> fetch(entry, dst, token) -> {"sha", "local_path", "n_files", "bytes", ...}; a copy so tests can patch it
FETCHERS: dict[str, Any] = dict(_lgd.FETCHERS)
PROBE_KINDS = ("github", "hf_dataset", "hf_stream")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")

# ----------------------------------------------------------------------------------------------
# schema probe
# ----------------------------------------------------------------------------------------------
DATA_SUFFIXES = (".jsonl.gz", ".json.gz", ".csv.gz", ".tsv.gz", ".jsonl", ".json", ".csv", ".tsv", ".parquet",
                 ".arrow", ".yaml", ".yml", ".txt", ".pkl", ".pickle", ".npy", ".npz", ".zip", ".tar", ".tgz",
                 ".tar.gz", ".gz", ".xlsx", ".h5")
SAMPLE_SUFFIXES = (".jsonl.gz", ".jsonl", ".json", ".csv", ".tsv", ".parquet", ".yaml", ".yml")
SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "venv", "build", "dist"}
SKIP_NAMES = {"package.json", "package-lock.json", "tsconfig.json", "requirements.txt", "requirements-dev.txt",
              "cmakelists.txt", "license.txt", "dataset_infos.json"}
DATA_DIR_HINTS = {"data", "dataset", "datasets", "benchmark", "benchmarks", "trace", "traces", "eval", "evaluation"}
MAX_WHOLE_FILE = 64 << 20        # json / yaml parsed whole only below this size
TRUNC = 200


def _suffix(name: str, suffixes: Sequence[str]) -> str | None:
    low = name.lower()
    for s in sorted(suffixes, key=len, reverse=True):
        if low.endswith(s):
            return s
    return None


def _data_files(root: Path) -> list[Path]:
    """Data-like files under root (relative paths, sorted); hidden and tooling directories pruned."""
    if root.is_file():
        return [Path(root.name)] if _suffix(root.name, DATA_SUFFIXES) else []
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
        for fn in filenames:
            if fn.startswith(".") or fn.lower() in SKIP_NAMES or not _suffix(fn, DATA_SUFFIXES):
                continue
            out.append((Path(dirpath) / fn).relative_to(root))
    return sorted(out, key=lambda p: p.as_posix())


def _trunc(v: Any) -> str:
    r = repr(v)
    return r if len(r) <= TRUNC else r[:TRUNC - 3] + "..."


def _infer(s: str) -> Any:
    if s == "":
        return None
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return s


def _describe_rows(rows: list[Any]) -> dict:
    """keys (first-seen order), types (name|name over the rows read), example (first row, reprs truncated)."""
    keys: list[str] = []
    types: dict[str, set[str]] = {}
    for r in rows:
        items = r.items() if isinstance(r, dict) else [("_row", r)]
        for k, v in items:
            k = str(k)
            if k not in types:
                keys.append(k)
                types[k] = set()
            types[k].add(type(v).__name__)
    first = rows[0] if rows else None
    example = ({str(k): _trunc(v) for k, v in first.items()} if isinstance(first, dict)
               else ({"_row": _trunc(first)} if rows else {}))
    return {"n_rows_read": len(rows), "keys": keys, "types": {k: "|".join(sorted(t)) for k, t in types.items()},
            "example": example}


def _open_text(p: Path):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.name.lower().endswith(".gz") \
        else open(p, encoding="utf-8", errors="replace")


def _jsonl_rows(p: Path, max_rows: int) -> list[Any]:
    rows: list[Any] = []
    with _open_text(p) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as ex:
                raise ValueError(f"line {i}: {ex}") from None
            if len(rows) >= max_rows:
                break
    return rows


def _structured(obj: Any, max_rows: int) -> tuple[list[Any], dict]:
    extra: dict[str, Any] = {"top_level": type(obj).__name__}
    if isinstance(obj, list):
        extra["n_items"] = len(obj)
        return obj[:max_rows], extra
    if isinstance(obj, dict):
        lists = {str(k): len(v) for k, v in obj.items() if isinstance(v, list)}
        if lists:
            extra["list_lengths"] = dict(list(lists.items())[:50])
        first = next(iter(obj.values()), None)
        if isinstance(first, dict):
            extra["value_keys"] = [str(k) for k in first][:100]
        return [obj], extra
    return [obj], extra


def sample_file(path: Path, rel: str, *, max_rows: int = 50) -> dict:
    """Keys / types / one example row of a data file. Errors are recorded, never raised."""
    fmt = _suffix(path.name, SAMPLE_SUFFIXES) or "?"
    out: dict[str, Any] = {"path": rel, "format": fmt.lstrip(".")}
    try:
        size = path.stat().st_size
        if fmt in (".jsonl", ".jsonl.gz"):
            out.update(_describe_rows(_jsonl_rows(path, max_rows)))
        elif fmt == ".json":
            if size > MAX_WHOLE_FILE:
                try:
                    rows = _jsonl_rows(path, max_rows)          # large .json files are often JSON lines
                    out.update(_describe_rows(rows), parsed_as="jsonl")
                except ValueError:
                    with _open_text(path) as f:
                        out.update(skipped=f"{size} bytes > {MAX_WHOLE_FILE}: not parsed", head=f.read(TRUNC))
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    out.update(_describe_rows(_jsonl_rows(path, max_rows)), parsed_as="jsonl")
                else:
                    rows, extra = _structured(obj, max_rows)
                    out.update(_describe_rows(rows), **extra)
        elif fmt in (".yaml", ".yml"):
            if size > MAX_WHOLE_FILE:
                out.update(skipped=f"{size} bytes > {MAX_WHOLE_FILE}: not parsed")
            else:
                import yaml
                rows, extra = _structured(yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")),
                                          max_rows)
                out.update(_describe_rows(rows), **extra)
        elif fmt in (".csv", ".tsv"):
            csv.field_size_limit(2**31 - 1)
            with _open_text(path) as f:
                rd = csv.DictReader(f, delimiter="\t" if fmt == ".tsv" else ",")
                rows = []
                for r in rd:
                    rows.append({k: _infer(v) if isinstance(v, str) else v for k, v in r.items()})
                    if len(rows) >= max_rows:
                        break
                out.update(_describe_rows(rows), columns=list(rd.fieldnames or []))
        elif fmt == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                out["error"] = "pyarrow not importable"
            else:
                pf = pq.ParquetFile(path)
                batch = next(pf.iter_batches(batch_size=max_rows), None)
                rows = batch.to_pylist() if batch is not None else []
                out.update(_describe_rows(rows), n_rows_total=pf.metadata.num_rows,
                           schema={f.name: str(f.type) for f in pf.schema_arrow})
        else:
            out["error"] = f"no reader for {path.name}"
    except Exception as ex:  # probing is advisory: never break the download
        out["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
    return out


def _sample_order(files: list[Path]) -> list[Path]:
    """Files under data-like directories first; inside each group one file per (directory, suffix)
    before the rest, so six samples cover more of the layout."""
    cands = [p for p in files if _suffix(p.name, SAMPLE_SUFFIXES)]
    hinted = [p for p in cands if any(part.lower() in DATA_DIR_HINTS for part in p.parts[:-1])]
    hinted_set = set(hinted)
    order: list[Path] = []
    for group in (hinted, [p for p in cands if p not in hinted_set]):
        seen: set[tuple[str, str | None]] = set()
        first, later = [], []
        for p in group:
            key = (p.parent.as_posix(), _suffix(p.name, SAMPLE_SUFFIXES))
            (later if key in seen else first).append(p)
            seen.add(key)
        order += first + later
    return order


def probe_path(root: str | Path, *, max_files: int = 300, max_samples: int = 6, max_rows: int = 50) -> dict:
    """{"root", "n_data_files", "files": [{path, bytes}] (first max_files, sorted), "samples": [...]}."""
    root = Path(root)
    files = _data_files(root)
    base = root.parent if root.is_file() else root

    def size(p: Path) -> int:
        try:
            return (base / p).stat().st_size
        except OSError:
            return -1

    samples = [sample_file(base / p, p.as_posix(), max_rows=max_rows) for p in _sample_order(files)[:max_samples]]
    return {"root": str(root), "n_data_files": len(files),
            "files": [{"path": p.as_posix(), "bytes": size(p)} for p in files[:max_files]], "samples": samples}


def _write_json(path: Path, obj: Any) -> None:
    from ag.harness.writer import atomic_write_text
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


# ----------------------------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------------------------
def _hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser() / "hub"


def _failure_message(e: AGEntry, ex: BaseException) -> str:
    msg = f"{type(ex).__name__}: {str(ex)[:400]}"
    if "scripts are no longer supported" in msg:
        msg += " (script-based dataset: use a parquet mirror from `alternatives`)"
    if e.gated and ("401" in msg or "403" in msg or "gated" in msg.lower()):
        msg += " (gated: accept the licence on the HF page and set HF_TOKEN)"
    return msg


def _count_weights(local: str | None) -> int:
    if not local or not Path(local).is_dir():
        return 0
    return sum(1 for p in Path(local).rglob("*") if p.name.lower().endswith(WEIGHT_SUFFIXES) and p.exists())


def _record(e: AGEntry, host: str) -> dict:
    return {"name": e.name, "kind": e.kind, "repo_id": e.repo_id, "config": e.config, "split": e.split,
            "role": e.role, "ag_role": e.ag_role, "stage": e.stage, "verify": e.verify, "gated": e.gated,
            "max_docs": e.max_docs, "url": e.url, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": host}


def _read_rev(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _probe_into(rec: dict, dst: Path) -> None:
    try:
        _write_json(dst / "probe.json", probe_path(rec["local_path"]))
        rec["probe"] = str(dst / "probe.json")
        rec.pop("probe_error", None)
    except Exception as ex:  # never fail a finished download because of the probe
        rec["probe_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        print(f"   probe failed: {rec['probe_error']}", file=sys.stderr)


def _select(only: Sequence[str] | None, stages: Sequence[str] | None) -> tuple[list[AGEntry], str | None]:
    entries = MODELS + DATASETS + BASELINES
    if stages:
        bad = sorted(set(stages) - set(STAGES))
        if bad:
            return [], f"unknown stage(s) {bad}; known: {list(STAGES)}"
        entries = [e for e in entries if e.stage in stages]
    if only:
        known = all_entries()
        bad = sorted(set(only) - set(known))
        if bad:
            return [], f"unknown names {bad} (python -m ag.data.download_ag --list)"
        want = set(only)
        dropped = sorted(n for n in want if n not in {e.name for e in entries})
        if dropped:
            print(f"[download_ag] not in stage {list(stages or [])}: {dropped}", file=sys.stderr)
        entries = [e for e in entries if e.name in want]
    return entries, None


def snapshot(sources: Path, status: dict[str, dict], *, host: str, run_info: dict) -> dict:
    """ag_assets_<MMDD>.json: every registry entry (status "missing" if never fetched) + extra status rows."""
    reg = all_entries()
    entries: dict[str, dict] = {}
    for name in list(reg) + sorted(set(status) - set(reg)):
        e, r = reg.get(name), status.get(name, {})
        entries[name] = {
            "status": r.get("status", "missing"), "sha": r.get("sha"),
            "stage": e.stage if e else r.get("stage"), "gated": e.gated if e else bool(r.get("gated")),
            "verify": e.verify if e else bool(r.get("verify")), "kind": e.kind if e else r.get("kind"),
            "id": (e.repo_id or e.url) if e else (r.get("repo_id") or r.get("url")),
            "downloaded_at": r.get("downloaded_at"), "local_path": r.get("local_path"), "error": r.get("error"),
            "in_registry": e is not None,
        }
    by_stage: dict[str, dict[str, int]] = {}
    for v in entries.values():
        by_stage.setdefault(str(v["stage"]), Counter())[v["status"]] += 1
    from ag.lineage_util import canonical_hash
    snap = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": host, "sources": str(sources),
        "registry_sha": canonical_hash([asdict(e) for e in reg.values()])[:16], "run": run_info,
        "entries": entries,
        "gated": {n: v["status"] for n, v in entries.items() if v["gated"]},
        "counts": {"total": len(entries), **Counter(v["status"] for v in entries.values()),
                   "by_stage": {k: dict(c) for k, c in sorted(by_stage.items())}},
    }
    path = sources / f"ag_assets_{time.strftime('%m%d')}.json"
    _write_json(path, snap)
    snap["path"] = str(path)
    return snap


def run(only: Sequence[str] | None = None, stage: str | Sequence[str] | None = None, skip_existing: bool = True,
        token: str | None = None, force_online: bool = False, sources: str | Path | None = None,
        probe: bool = True) -> dict:
    """Fetch the selected registry entries into `sources` (default $AG_SOURCES). Returns
    {"ok", "failed", "manual", "skipped", "attempted", "failed_total", "status", "snapshot"} or {"error"}."""
    src = sources or os.environ.get("AG_SOURCES")
    if not src:
        print("[download_ag] AG_SOURCES is not set (env/ag.env) and no sources given", file=sys.stderr)
        return {"error": "no_sources"}
    src = Path(src).expanduser()
    probs = validate()
    if probs:
        print("[download_ag] registry problems:\n  " + "\n  ".join(probs), file=sys.stderr)
        return {"error": "registry"}
    stages = [stage] if isinstance(stage, str) else (list(stage) if stage else None)
    entries, err = _select(only, stages)
    if err:
        print(f"[download_ag] {err}", file=sys.stderr)
        return {"error": err}
    host = socket.gethostname()
    if not force_online and (os.environ.get("SLURM_JOB_ID") or not _online_ok()):
        print(f"[download_ag] no network from {host} (or inside a Slurm job). Run this on the login node "
              f"({LOGIN_HOST}), never inside sbatch.", file=sys.stderr)
        return {"error": "offline"}
    saved = {k: os.environ.pop(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE") if k in os.environ}
    hub_mod = sys.modules.get("huggingface_hub.constants")
    if hub_mod is not None and getattr(hub_mod, "HF_HUB_OFFLINE", False):
        print("[download_ag] WARN: huggingface_hub was imported in offline mode before run(); "
              "use the CLI (python -m ag.data.download_ag) in a fresh process", file=sys.stderr)
    token = token or os.environ.get("HF_TOKEN")
    res: dict[str, list[str]] = {"ok": [], "failed": [], "manual": [], "skipped": [], "attempted": []}
    try:
        src.mkdir(parents=True, exist_ok=True)
        for e in entries:
            dst = src / e.name
            old = _read_rev(dst / "revision.json")
            if skip_existing and old.get("status") == "ok":
                print(f"[skip] {e.name} (revision.json present)")
                res["skipped"].append(e.name)
                if probe and e.kind in PROBE_KINDS and not (dst / "probe.json").exists() \
                        and old.get("local_path") and Path(old["local_path"]).exists():
                    _probe_into(old, dst)
                    _write_rev(dst, old)
                continue
            res["attempted"].append(e.name)
            rec = _record(e, host)
            if e.kind == "manual" or not (e.repo_id or e.url):
                rec.update(status="manual", error=None, note=e.note, alternatives=e.alternatives)
                print(f"[manual] {e.name}: {e.note or e.url}")
                res["manual"].append(e.name)
            else:
                t0 = time.time()
                dst.mkdir(parents=True, exist_ok=True)   # hf_stream writes docs.jsonl.gz before revision.json
                try:
                    print(f"[fetch] {e.name} <- {e.repo_id or e.url} ({e.kind}, stage {e.stage}) ...", flush=True)
                    got = FETCHERS[e.kind](e, dst, token)
                    if e.kind == "hf_model" and _count_weights(got.get("local_path")) == 0:
                        raise RuntimeError(f"snapshot {got.get('local_path')} has no weight files "
                                           f"({', '.join(WEIGHT_SUFFIXES)}): check allow/ignore patterns")
                    rec.update(got, status="ok", error=None, seconds=round(time.time() - t0, 1))
                    print(f"   ok  sha={str(got.get('sha'))[:12]} files={got.get('n_files')} bytes={got.get('bytes')}")
                    if probe and e.kind in PROBE_KINDS:
                        _probe_into(rec, dst)
                    res["ok"].append(e.name)
                except Exception as ex:  # keep going; record the failure with the alternatives
                    msg = _failure_message(e, ex)
                    print(f"   FAILED {msg}", file=sys.stderr)
                    res["failed"].append(e.name)
                    if old.get("status") == "ok":     # --no-skip re-fetch failed: the earlier copy is still there
                        rec = {**old, "refetch_error": msg, "refetch_at": rec["downloaded_at"]}
                        print(f"   kept the previous ok record (sha={str(old.get('sha'))[:12]})", file=sys.stderr)
                    else:
                        rec.update(status="failed", error=msg, alternatives=e.alternatives,
                                   seconds=round(time.time() - t0, 1))
            _write_rev(dst, rec)
            rebuild_status(src)
        hub = _hub_dir()
        fixed = fix_refs(hub) if hub.exists() else 0
        if fixed:
            print(f"[refs] wrote refs/main for {fixed} cached repos (needed for offline lookups)")
        status = rebuild_status(src)
        snap = snapshot(src, status, host=host,
                        run_info={"only": list(only or []), "stage": stages, "skip_existing": skip_existing,
                                  **{k: v for k, v in res.items()}})
    finally:
        os.environ.update(saved)
    failed_total = sorted(n for n, r in status.items() if r.get("status") == "failed")
    c = snap["counts"]
    print(f"\n[summary] this run: ok={len(res['ok'])} failed={len(res['failed'])} manual={len(res['manual'])} "
          f"skipped={len(res['skipped'])} | all: ok={c.get('ok', 0)} failed={c.get('failed', 0)} "
          f"manual={c.get('manual', 0)} missing={c.get('missing', 0)}")
    print(f"[summary] {src / 'download_status.json'}\n[summary] {snap['path']}")
    for n in failed_total:
        print(f"  FAILED {n}: {status[n].get('error')}")
        for a in status[n].get("alternatives") or []:
            print(f"     alt: {a}")
    gated = ", ".join(f"{n}={s}" for n, s in snap["gated"].items())
    print(f"[gated] {gated}")
    return {**res, "failed_total": failed_total, "status": str(src / "download_status.json"),
            "snapshot": snap["path"]}


# ----------------------------------------------------------------------------------------------
# status table (agctl assets)
# ----------------------------------------------------------------------------------------------
def asset_rows(sources: str | Path) -> list[dict]:
    """Registry entries merged with $AG_SOURCES/download_status.json (status "missing" if absent)."""
    p = Path(sources) / "download_status.json"
    status = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    reg = all_entries()
    rows = []
    for name in list(reg) + sorted(set(status) - set(reg)):
        e, r = reg.get(name), status.get(name, {})
        rows.append({"name": name, "stage": e.stage if e else r.get("stage", "?"),
                     "status": r.get("status", "missing"), "sha": (r.get("sha") or "")[:12],
                     "gated": bool(e.gated if e else r.get("gated")), "kind": e.kind if e else r.get("kind", "?")})
    return rows


def format_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    head = f"{'name':34s} {'stage':8s} {'status':8s} {'sha':12s} gated"
    lines = [head, "-" * len(head)]
    lines += [f"{r['name']:34s} {r['stage']:8s} {r['status']:8s} {r['sha'] or '-':12s} {'yes' if r['gated'] else ''}"
              for r in rows]
    c = Counter(r["status"] for r in rows)
    lines.append("counts: " + " ".join(f"{k}={c[k]}" for k in sorted(c)))
    return "\n".join(lines)


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_ENV_DEFAULT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """`export KEY=VALUE` lines of env/ag.env with $VAR, ${VAR} and ${VAR:-default} expanded against the
    process environment and earlier lines. Nothing is exported."""
    vals: dict[str, str] = {}
    if not path.exists():
        return vals
    look = {**os.environ}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(raw.split(" #", 1)[0].strip())
        if not m:
            continue
        val = m.group(2).strip()
        if len(val) >= 2 and val[0] in "\"'" and val[-1] == val[0]:
            val = val[1:-1]
        val = _ENV_DEFAULT.sub(lambda d: look.get(d.group(1)) or d.group(2), val)
        val = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda v: look.get(v.group(1) or v.group(2), v.group(0)), val)
        vals[m.group(1)] = look[m.group(1)] = os.path.expanduser(val)
    return vals


def default_sources() -> Path | None:
    """$AG_SOURCES, else AG_SOURCES from env/ag.env, else None."""
    v = os.environ.get("AG_SOURCES") or read_env_file().get("AG_SOURCES")
    return Path(v).expanduser() if v else None


DEFAULT_STAGES = ("pilot", "s2")      # CLI default when neither --stage nor --only is given


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="AG asset downloader (login node only; build plan 4.2 S0)")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--only", default=None, help="comma-separated registry names")
    ap.add_argument("--stage", default=None,
                    help=f"comma-separated stages {','.join(STAGES)}, or all (default without --only: {','.join(DEFAULT_STAGES)})")
    ap.add_argument("--no-skip", action="store_true", help="re-fetch entries whose revision.json says ok")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    ap.add_argument("--force-online", action="store_true", help="skip the network / sbatch check")
    ap.add_argument("--no-probe", action="store_true", help="do not write probe.json")
    ap.add_argument("--sources", default=None, help="override $AG_SOURCES")
    a = ap.parse_args(argv)
    if a.list:
        for e in MODELS + DATASETS + BASELINES:
            flags = " ".join(f for f, on in (("gated", e.gated), ("verify", e.verify)) if on)
            print(f"{e.name:34s} {e.kind:10s} {(e.repo_id or e.url or '-'):58s} {e.stage:8s} {flags:12s} | {e.ag_role}")
        return 0
    if not (a.sources or os.environ.get("AG_SOURCES")):
        for k, v in read_env_file().items():      # direct CLI use: env/ag.env fills what the shell did not set
            os.environ.setdefault(k, v)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # classic HTTP resumes partial blobs (as the wrapper script)
    if a.stage == "all" or (a.only and not a.stage):
        stages = None
    elif a.stage:
        stages = a.stage.split(",")
    else:      # T3, T4 and the later sets are tens of GB: fetch them only when asked
        stages = list(DEFAULT_STAGES)
        print(f"[download_ag] no --stage/--only: fetching stages {stages} (--stage all for everything)")
    r = run(only=a.only.split(",") if a.only else None, stage=stages,
            skip_existing=not a.no_skip, token=a.token, force_online=a.force_online, sources=a.sources,
            probe=not a.no_probe)
    return 0 if not r.get("error") and not r.get("failed") else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)   # skip interpreter teardown: streaming datasets threads can hang at exit (as lgs1.download)
