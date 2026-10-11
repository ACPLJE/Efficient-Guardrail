"""lineage.json for every PoC step (CLAUDE.md 3.3), written with the AG implementation (ag/harness/lineage.py, D-038).

Three kinds of run, each with its own run directory:
    build     <run>/data/          fresh every time (a rebuild is a new run)
    backbone  <run>/probes/<bb>/   extract + fit + score of one backbone; RESUMABLE: run_id.txt keeps the run id, and
                                   a resume whose resume keys differ (code, env, config, model, data, GPU) exits 2
    analyze   <run>/report/        fresh every time
run_id follows SPEC 1.5 (ag.sweep.expand.make_run_id); system_id is `agdepth-<step>` so a PoC run can never be taken
for a SPEC experiment run.  Every JSONL row the PoC writes carries run_id (ag.harness.writer.write_rows).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from lgs1.lineage import canonical_hash, dir_fingerprint

REPO = Path(__file__).resolve().parent.parent


def now_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def config_hash(config: Mapping) -> str:
    return canonical_hash(dict(config))


def make_run_id(exp_id: str, step: str, target: str, dtag: str, chash: str, seed: int, ts: Optional[str] = None) -> str:
    from ag.sweep.expand import make_run_id as _mk
    safe = lambda x: str(x).replace("__", "_").replace("/", "_") or "none"     # noqa: E731
    return _mk(exp_id, f"agdepth-{step}", safe(target), safe(dtag), chash, 0, 0, seed, ts or now_ts())


def model_sha_of(repo: str, sha: str) -> str:
    """The HF commit sha, else a content fingerprint of a local checkpoint directory."""
    if sha:
        return sha
    p = Path(repo)
    return dir_fingerprint(p) if p.is_dir() else ""


def start(run_dir: Path, run_id: str, exp_id: str, chash: str, *, seed: int, producer: str,
          model_sha: Mapping[str, str], data_manifest_sha: str, parents: Sequence[str] = (), fresh: bool = False):
    """open_lineage; a resume with different resume keys exits with code 2 (CLAUDE.md 3.3)."""
    import os
    from ag.harness.lineage import LineageMismatch, build_lineage, fail_on_mismatch, open_lineage
    # lgs4 sets this default on import (model loading); set it here too so env_hash -- a resume key -- does not depend
    # on whether this process loaded a model (a standalone fit would otherwise always exit 2 after a pipeline run)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        (run_dir / "lineage.json").unlink(missing_ok=True)
    lin = build_lineage(run_id, exp_id, chash, clock="monotonic", seed=seed, producer=producer,
                        model_sha=dict(model_sha), data_manifest_sha=data_manifest_sha,
                        engine_version="hf-transformers" if producer != "mock" else "mock",
                        parent_run_ids=tuple(parents), code_dir=REPO)
    try:
        lin, _ = open_lineage(run_dir, lin)
    except LineageMismatch as e:
        print(f"[lineage] {run_dir}: this run was started under different resume keys; start a new run "
              f"(--fresh, or another --run name) instead of mixing them", flush=True)
        fail_on_mismatch(e)
    return lin


def finish(run_dir: Path, lin) -> None:
    from ag.harness.lineage import close_lineage
    close_lineage(run_dir, lin)


def stable_run_id(run_dir: Path, make) -> Tuple[str, bool]:
    """The run id stored in run_dir/run_id.txt, or a new one from make() (returned flag: True if it was new)."""
    f = Path(run_dir) / "run_id.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text().strip(), False
    rid = make()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(rid + "\n")
    return rid, True
