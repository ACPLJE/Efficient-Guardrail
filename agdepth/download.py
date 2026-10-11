"""Login-node downloads for the PoC (compute nodes are offline).  Same layout as S1 D3:
$LG_SOURCES/<name>/revision.json {name, kind, repo_id|urls, sha, local_path, status, ...}.

    python depthctl.py download                    # BIPIA, NotInject, deepset, LLMail-Inject, Gandalf, 2 backbones
    python depthctl.py download --only bipia,notinject
    python depthctl.py download --with-s1          # + the S1 entries the PoC reads when LG v2 data is not built

Run it on the login node under nohup (large models): `nohup python depthctl.py download > dl.log 2>&1 &`.
xet is turned off (HF_HUB_DISABLE_XET=1) as in slurm/s1_download_login.sh: a stalled xet transfer cannot resume.
"""
from __future__ import annotations

import hashlib
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .config import DOWNLOADS, S1_DEPENDENCIES, AGPaths, Download


def fetch_urls(d: Download, dst: Path) -> Dict:
    """All or nothing: every URL is read into memory first, files are written only after all of them arrived."""
    blobs = []
    for url in d.urls:
        with urllib.request.urlopen(url, timeout=120) as r:
            blobs.append((Path(url).name, r.read()))
    dst.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for name, data in blobs:
        tmp = dst / f".{name}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, dst / name)
        h.update(data)
    return {"sha": h.hexdigest(), "local_path": str(dst), "n_files": len(blobs), "bytes": sum(len(b) for _, b in blobs),
            "urls": list(d.urls)}


def run(only: Optional[List[str]] = None, with_s1: bool = False, token: Optional[str] = None,
        force_online: bool = False, skip_existing: bool = True) -> Dict:
    import json
    from lgs1 import download as D1
    from lgs1.registry import Entry
    P = AGPaths.load()
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ["HF_HOME"] = str(P.hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(P.hf_home / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")         # set before huggingface_hub is imported
    if not force_online and not D1._online_ok():
        print(f"[download] no network from {socket.gethostname()}: run on the login node, never inside sbatch",
              file=sys.stderr)
        return {"error": "offline"}
    token = token or os.environ.get("HF_TOKEN")
    todo = [d for d in DOWNLOADS if not only or d.name in only]
    status = {}
    for d in todo:
        dst = P.sources / d.name
        rev = dst / "revision.json"
        if skip_existing and rev.exists() and json.loads(rev.read_text()).get("status") == "ok":
            print(f"[skip] {d.name} (revision.json present)")
            status[d.name] = "ok"
            continue
        rec = {"name": d.name, "kind": d.kind, "repo_id": d.repo_id, "role": d.role, "license": d.license,
               "verify": d.verify, "gated": d.gated, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "host": socket.gethostname(), "by": "agdepth"}
        t0 = time.time()
        dst.mkdir(parents=True, exist_ok=True)
        D1._write_rev(dst, {**rec, "status": "started"})      # an interrupted fetch is never taken for a finished one
        try:
            print(f"[fetch] {d.name} <- {d.repo_id or ', '.join(d.urls)} ({d.kind})", flush=True)
            if d.kind == "urls":
                got = fetch_urls(d, dst)
            else:
                got = D1.FETCHERS[d.kind](Entry(d.name, d.kind, d.repo_id, gated=d.gated, verify=d.verify), dst, token)
            rec.update(got, status="ok", error=None, seconds=round(time.time() - t0, 1))
            print(f"   ok  sha={got['sha'][:12]} files={got.get('n_files')} bytes={got.get('bytes')}")
        except Exception as ex:
            msg = f"{type(ex).__name__}: {str(ex)[:400]}"
            if d.gated and ("401" in msg or "403" in msg or "gated" in msg.lower()):
                msg += " (gated: accept the licence on the HF page and set HF_TOKEN)"
            rec.update(status="failed", error=msg, seconds=round(time.time() - t0, 1))
            print(f"   FAILED {msg}", file=sys.stderr)
        D1._write_rev(dst, rec)
        status[d.name] = rec["status"]
    if with_s1:
        r1 = D1.run(only=S1_DEPENDENCIES, skip_existing=skip_existing, token=token, force_online=force_online)
        s1 = D1.rebuild_status(P.sources)
        for name in S1_DEPENDENCIES:                           # only the entries this call asked for
            status[f"s1:{name}"] = (s1.get(name) or {}).get("status", "missing")
        if r1.get("error"):
            status["s1"] = r1["error"]
    D1.rebuild_status(P.sources)
    failed = [k for k, v in status.items() if v not in ("ok", "manual")]
    print(f"[download] ok={len(status) - len(failed)} failed={len(failed)} {failed if failed else ''}")
    return {"status": status, "failed": failed}
