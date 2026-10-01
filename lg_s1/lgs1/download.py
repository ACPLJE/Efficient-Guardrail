"""D3: fetch every registry entry on the LOGIN node and pin its revision.

Writes, per entry,  $LG_SOURCES/<name>/revision.json:
    {"name", "kind", "repo_id", "config", "sha" (resolved commit), "downloaded_at", "local_path",
     "n_files", "bytes", "max_docs", "status", "error"}
and the roll-up            $LG_SOURCES/download_status.json

Rules
  * refuses to run on a compute node (no network): checks hostname / HF reachability
  * HF_HUB_OFFLINE is unset for this process only; compute-node jobs keep it =1
  * models: safetensors + tokenizer + config only (no original/, no gguf)
  * --only NAME[,NAME]  --skip-existing  --models-only  --datasets-only  --token
"""
from __future__ import annotations

import gzip
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .paths import S1Paths, env
from .registry import DATASETS, MODELS, MODEL_IGNORE, MODEL_PATTERNS, Entry, all_entries


def _online_ok(timeout: float = 8.0) -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def _dir_stats(p: Path) -> Dict[str, int]:
    n, b = 0, 0
    for f in p.rglob("*"):
        if f.is_file() and not f.is_symlink():
            n += 1
            b += f.stat().st_size
        elif f.is_symlink() and f.exists():
            n += 1
            b += f.stat().st_size
    return {"n_files": n, "bytes": b}


def _write_rev(dst: Path, rec: Dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "revision.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_ref_main(local: str, sha: str) -> None:
    """huggingface_hub writes refs/main only when the revision is a branch name; with a pinned sha it
    does not, and then every offline lookup of "main" (from_pretrained, snapshot_download(local_files_only))
    fails on the compute nodes.  Write it ourselves."""
    repo_dir = Path(local).parent.parent          # .../models--org--name/snapshots/<sha> -> repo dir
    refs = repo_dir / "refs" / "main"
    refs.parent.mkdir(parents=True, exist_ok=True)
    if not refs.exists() or refs.read_text().strip() != sha:
        refs.write_text(sha)


def _lfs_sizes(info) -> Dict[str, int]:
    """{sha256: size} for every LFS file of a repo (from *_info(..., files_metadata=True))."""
    sizes: Dict[str, int] = {}
    for s in getattr(info, "siblings", None) or []:
        lfs = getattr(s, "lfs", None)
        if not lfs:
            continue
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        if sha and size:
            sizes[sha] = int(size)
    return sizes


def purge_sparse_incomplete(blobs: Path, sizes: Dict[str, int]) -> List[Path]:
    """Delete `<sha256>[.xxxx].incomplete` blobs whose st_size already equals the full file size.

    The xet backend writes segments at offsets, so a killed download leaves a full-size sparse file
    with holes; the classic HTTP path (HF_HUB_DISABLE_XET=1) would treat it as 'fully resumed' and
    the consistency check (size only) would pass -> corrupt safetensors.  Partial files that are
    genuinely shorter than the target are kept: the classic path resumes them with a Range request."""
    removed: List[Path] = []
    if not blobs.exists():
        return removed
    for inc in blobs.glob("*.incomplete"):
        sha = inc.name.split(".", 1)[0]
        exp = sizes.get(sha)
        try:
            actual = inc.stat().st_size
        except OSError:
            continue
        if exp is not None and actual >= exp:
            inc.unlink(missing_ok=True)
            removed.append(inc)
    return removed


def _hub_blobs_dir(repo_id: str, repo_type: str) -> Path:
    from huggingface_hub import constants as C
    prefix = "models" if repo_type == "model" else f"{repo_type}s"
    return Path(C.HF_HUB_CACHE) / f"{prefix}--{repo_id.replace('/', '--')}" / "blobs"


def fetch_hf_model(e: Entry, dst: Path, token: Optional[str]) -> Dict:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    info = api.model_info(e.repo_id, files_metadata=True)
    sha = info.sha
    for p in purge_sparse_incomplete(_hub_blobs_dir(e.repo_id, "model"), _lfs_sizes(info)):
        print(f"[purge] sparse partial blob removed: {p.name}", flush=True)
    local = snapshot_download(e.repo_id, revision=sha, token=token,
                              allow_patterns=e.allow_patterns or MODEL_PATTERNS,
                              ignore_patterns=e.ignore_patterns or MODEL_IGNORE)
    _write_ref_main(local, sha)
    st = _dir_stats(Path(local))
    return {"sha": sha, "local_path": local, **st}


def fetch_hf_dataset(e: Entry, dst: Path, token: Optional[str]) -> Dict:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    info = api.dataset_info(e.repo_id, files_metadata=True)
    sha = info.sha
    for p in purge_sparse_incomplete(_hub_blobs_dir(e.repo_id, "dataset"), _lfs_sizes(info)):
        print(f"[purge] sparse partial blob removed: {p.name}", flush=True)
    local = snapshot_download(e.repo_id, repo_type="dataset", revision=sha, token=token)
    _write_ref_main(local, sha)
    st = _dir_stats(Path(local))
    return {"sha": sha, "local_path": local, **st}


def rebuild_status(sources: Path) -> Dict[str, Dict]:
    """download_status.json rebuilt from every <name>/revision.json (the per-entry files are the truth,
    e.g. after the sources directory was moved or merged)."""
    status: Dict[str, Dict] = {}
    if not sources.exists():
        return status
    for rev in sorted(sources.glob("*/revision.json")):
        try:
            status[rev.parent.name] = json.loads(rev.read_text(encoding="utf-8"))
        except Exception:
            continue
    (sources / "download_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def fix_refs(hub: Path) -> int:
    """Add refs/main to every cached repo that has exactly one snapshot and no ref (idempotent)."""
    n = 0
    for repo in list(hub.glob("models--*")) + list(hub.glob("datasets--*")):
        refs, snaps = repo / "refs" / "main", repo / "snapshots"
        if refs.exists() or not snaps.exists():
            continue
        shas = [d.name for d in snaps.iterdir() if d.is_dir()]
        if len(shas) == 1:
            refs.parent.mkdir(parents=True, exist_ok=True)
            refs.write_text(shas[0])
            n += 1
            print(f"[refs] {repo.name} -> main = {shas[0][:12]}")
    return n


def fetch_hf_stream(e: Entry, dst: Path, token: Optional[str]) -> Dict:
    """Stream a large dataset and keep at most max_docs documents as docs.jsonl.gz (text + meta)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    sha = api.dataset_info(e.repo_id).sha
    ds = load_dataset(e.repo_id, e.config, split=e.split or "train", streaming=True, token=token)
    out = dst / "docs.jsonl.gz"
    n = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            text = row.get(e.text_field) if e.text_field else None
            if isinstance(text, list):  # e.g. WildChat conversation turns
                text = "\n".join(str(t.get("content", t)) if isinstance(t, dict) else str(t) for t in text)
            if not text or not isinstance(text, str):
                continue
            meta = {k: v for k, v in row.items() if k != e.text_field and isinstance(v, (str, int, float, bool))}
            f.write(json.dumps({"id": f"{e.name}-{i}", "text": text, "meta": meta}, ensure_ascii=False) + "\n")
            n += 1
            if e.max_docs and n >= e.max_docs:
                break
    return {"sha": sha, "local_path": str(out), "n_files": 1, "bytes": out.stat().st_size, "n_docs": n}


def fetch_github(e: Entry, dst: Path, token: Optional[str]) -> Dict:
    repo = dst / "repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", e.url, str(repo)], check=True, timeout=1800)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return {"sha": sha, "local_path": str(repo), **_dir_stats(repo)}


FETCHERS = {"hf_model": fetch_hf_model, "hf_dataset": fetch_hf_dataset, "hf_stream": fetch_hf_stream, "github": fetch_github}


def run(only: Optional[List[str]] = None, skip_existing: bool = True, models_only: bool = False,
        datasets_only: bool = False, token: Optional[str] = None, force_online: bool = False) -> Dict:
    P = S1Paths.load()
    os.environ.pop("HF_HUB_OFFLINE", None)   # this process must be online
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ["HF_HOME"] = str(P.hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(P.hf_home / "hub"))
    host = socket.gethostname()
    login = env("LG_LOGIN_HOST", "")
    if not force_online and not _online_ok():
        print(f"[download] no network from {host}. Run this on the login node ({login or 'gsai-login-3'}), "
              f"never inside sbatch.", file=sys.stderr)
        return {"error": "offline"}
    token = token or os.environ.get("HF_TOKEN")
    entries: List[Entry] = []
    if not datasets_only:
        entries += MODELS
    if not models_only:
        entries += DATASETS
    if only:
        want = set(only)
        entries = [e for e in entries if e.name in want]
        missing = want - {e.name for e in entries}
        if missing:
            print(f"[download] unknown names: {sorted(missing)}", file=sys.stderr)
    P.sources.mkdir(parents=True, exist_ok=True)
    status_path = P.sources / "download_status.json"
    status: Dict[str, Dict] = json.loads(status_path.read_text()) if status_path.exists() else {}
    for e in entries:
        dst = P.sources / e.name
        rev = dst / "revision.json"
        if skip_existing and rev.exists() and json.loads(rev.read_text()).get("status") == "ok":
            print(f"[skip] {e.name} (revision.json present)")
            continue
        rec = {"name": e.name, "kind": e.kind, "repo_id": e.repo_id, "config": e.config, "split": e.split,
               "role": e.role, "verify": e.verify, "gated": e.gated, "max_docs": e.max_docs,
               "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": host}
        if e.kind == "manual" or not (e.repo_id or e.url):
            rec.update(status="manual", error=None, note=e.note, url=e.url, alternatives=e.alternatives)
            print(f"[manual] {e.name}: {e.note or e.url}")
        else:
            t0 = time.time()
            dst.mkdir(parents=True, exist_ok=True)   # hf_stream writes docs.jsonl.gz here BEFORE revision.json exists
            try:
                print(f"[fetch] {e.name} <- {e.repo_id or e.url} ({e.kind}) ...", flush=True)
                got = FETCHERS[e.kind](e, dst, token)
                rec.update(got, status="ok", error=None, seconds=round(time.time() - t0, 1))
                print(f"   ok  sha={got['sha'][:12]} files={got.get('n_files')} bytes={got.get('bytes')}")
            except Exception as ex:  # keep going; record the failure with the alternatives
                msg = f"{type(ex).__name__}: {str(ex)[:400]}"
                if "scripts are no longer supported" in msg:
                    msg += " (script-based dataset: use a parquet mirror from `alternatives`)"
                gated_hint = " (gated: accept the licence on the HF page and set HF_TOKEN)" if e.gated and ("401" in msg or "403" in msg or "gated" in msg.lower()) else ""
                rec.update(status="failed", error=msg + gated_hint, alternatives=e.alternatives, seconds=round(time.time() - t0, 1))
                print(f"   FAILED {msg}{gated_hint}", file=sys.stderr)
        _write_rev(dst, rec)
        status[e.name] = rec
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    fixed = fix_refs(P.hf_home / "hub")
    if fixed:
        print(f"[refs] wrote refs/main for {fixed} cached repos (needed for offline lookups)")
    status = rebuild_status(P.sources)
    # summary
    ok = [n for n, r in status.items() if r.get("status") == "ok"]
    failed = [n for n, r in status.items() if r.get("status") == "failed"]
    manual = [n for n, r in status.items() if r.get("status") == "manual"]
    print(f"\n[summary] ok={len(ok)} failed={len(failed)} manual={len(manual)} -> {status_path}")
    for n in failed:
        print(f"  FAILED {n}: {status[n]['error']}")
        for a in status[n].get("alternatives") or []:
            print(f"     alt: {a}")
    return {"ok": ok, "failed": failed, "manual": manual, "status": str(status_path)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="D3 downloader (login node only)")
    ap.add_argument("--only", default=None, help="comma-separated registry names")
    ap.add_argument("--no-skip", action="store_true", help="re-download entries with revision.json")
    ap.add_argument("--models-only", action="store_true")
    ap.add_argument("--datasets-only", action="store_true")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    ap.add_argument("--force-online", action="store_true", help="skip the connectivity check")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--fix-refs", action="store_true", help="only write missing refs/main in the HF cache and exit")
    a = ap.parse_args(argv)
    if a.fix_refs:
        P = S1Paths.load()
        n = fix_refs(P.hf_home / "hub")
        print(f"[refs] fixed {n} repos under {P.hf_home / 'hub'}")
        return 0
    if a.list:
        for e in MODELS + DATASETS:
            flag = ("gated " if e.gated else "") + ("verify " if e.verify else "")
            print(f"{e.name:28s} {e.kind:10s} {e.repo_id or e.url:55s} {flag}| {e.role}")
        return 0
    r = run(only=a.only.split(",") if a.only else None, skip_existing=not a.no_skip, models_only=a.models_only,
            datasets_only=a.datasets_only, token=a.token, force_online=a.force_online)
    return 0 if not r.get("error") and not r.get("failed") else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)   # skip interpreter teardown: streaming datasets threads can hang at exit
