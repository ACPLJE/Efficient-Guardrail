"""AG lineage (SPEC 7.2, CLAUDE.md 3.3, D-038).

From lgs1.lineage only canonical_hash / code_commit are reused; comparison and writing of the
AG fields live here. Resuming a run_id whose resume keys differ is an error (exit code 2).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ag import SCHEMA_VERSION
from ag.data.serde import JsonMixin
from ag.harness.writer import atomic_write_json, read_json
from ag.lineage_util import canonical_hash, code_commit

ENV_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
            "PYTORCH_CUDA_ALLOC_CONF")
RESUME_KEYS = ("config_hash", "code_commit", "env_hash", "engine_version", "patch_sha", "model_sha",
               "data_manifest_sha", "producer", "gpu_name", "gpu_count", "topology_id", "seed", "clock")
EXIT_LINEAGE_MISMATCH = 2


class LineageMismatch(RuntimeError):
    def __init__(self, run_id: str, diffs: Mapping[str, tuple[Any, Any]]):
        self.run_id = run_id
        self.diffs = dict(diffs)
        lines = [f"  {k}: existing={a!r} new={b!r}" for k, (a, b) in self.diffs.items()]
        super().__init__(f"run {run_id}: resume keys differ\n" + "\n".join(lines))


@dataclass(frozen=True)
class AGLineage(JsonMixin):
    run_id: str
    exp_id: str
    config_hash: str
    code_commit: str
    code_dirty: bool
    env_hash: Mapping[str, str]
    clock: str
    seed: int
    schema_version: int = SCHEMA_VERSION
    python: str = ""
    torch: str | None = None
    transformers: str | None = None
    engine_version: str | None = None
    patch_sha: str | None = None
    shim_active: bool = False
    gpu_name: str = "cpu"
    gpu_count: int = 0
    gpu_driver: str | None = None
    topology_id: str = "1gpu"
    placement: Mapping[str, str] = field(default_factory=dict)
    hostname: str = ""
    slurm_job_id: str | None = None
    slurm_array_job_id: str | None = None
    slurm_array_task_id: str | None = None
    started_monotonic_ns: int = 0
    started_unix_ns: int = 0
    finished_unix_ns: int | None = None
    clock_offset_ns: int = 0
    model_sha: Mapping[str, str] = field(default_factory=dict)
    data_manifest_sha: str = ""
    record_cache_keys: tuple[str, ...] = ()
    threshold_ids: tuple[str, ...] = ()
    producer: str = ""
    parent_run_ids: tuple[str, ...] = ()

    def resume_view(self) -> dict:
        d = self.to_json()
        return {k: d.get(k) for k in RESUME_KEYS}


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def env_hash() -> dict[str, str]:
    """pip_freeze_sha: sorted name==version of the installed distributions (same content as
    `pip freeze`, without spawning pip). env_vars_sha: the fixed variable list + GPU count."""
    dists = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in importlib.metadata.distributions()})
    env = {k: os.environ.get(k) for k in ENV_VARS}
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES_count"] = None if cvd is None else len([x for x in cvd.split(",") if x.strip()])
    return {"pip_freeze_sha": hashlib.sha256("\n".join(dists).encode()).hexdigest()[:16],
            "env_vars_sha": canonical_hash(env)[:16]}


def gpu_info() -> tuple[str, int, str | None]:
    if shutil.which("nvidia-smi") is None:
        return "cpu", 0, None
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=20)
    except (subprocess.SubprocessError, OSError):
        return "cpu", 0, None
    rows = [r.split(", ") for r in out.strip().splitlines() if r.strip()]
    if not rows:
        return "cpu", 0, None
    return rows[0][0], len(rows), rows[0][1] if len(rows[0]) > 1 else None


def build_lineage(run_id: str, exp_id: str, config_hash: str, *, clock: str, seed: int, producer: str,
                  model_sha: Mapping[str, str], data_manifest_sha: str, engine_version: str | None = None,
                  patch_sha: str | None = None, shim_active: bool = False, topology_id: str = "1gpu",
                  placement: Mapping[str, str] | None = None, threshold_ids: Sequence[str] = (),
                  parent_run_ids: Sequence[str] = (), record_cache_keys: Sequence[str] = (),
                  code_dir: str | Path | None = None) -> AGLineage:
    commit = code_commit(code_dir)
    gname, gcount, gdriver = gpu_info()
    return AGLineage(
        run_id=run_id, exp_id=exp_id, config_hash=config_hash, code_commit=commit,
        code_dirty=commit.endswith("-dirty"), env_hash=env_hash(), clock=clock, seed=seed,
        python=platform.python_version(), torch=_pkg_version("torch"), transformers=_pkg_version("transformers"),
        engine_version=engine_version, patch_sha=patch_sha, shim_active=shim_active,
        gpu_name=gname, gpu_count=gcount, gpu_driver=gdriver, topology_id=topology_id,
        placement=dict(placement or {}), hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        started_monotonic_ns=time.monotonic_ns(), started_unix_ns=time.time_ns(),
        model_sha=dict(model_sha), data_manifest_sha=data_manifest_sha,
        record_cache_keys=tuple(record_cache_keys), threshold_ids=tuple(threshold_ids), producer=producer,
        parent_run_ids=tuple(parent_run_ids),
    )


def diff_resume(existing: Mapping[str, Any], new: AGLineage) -> dict[str, tuple[Any, Any]]:
    nv = new.resume_view()
    old = json.loads(json.dumps(dict(existing)))
    return {k: (old.get(k), nv[k]) for k in RESUME_KEYS if old.get(k) != nv[k]}


def open_lineage(run_dir: str | Path, new: AGLineage) -> tuple[AGLineage, bool]:
    """Write lineage.json for a fresh run, or check it for a resume.

    Returns (lineage, resumed). Raises LineageMismatch when the resume keys differ.
    """
    path = Path(run_dir) / "lineage.json"
    if path.exists():
        old = read_json(path)
        diffs = diff_resume(old, new)
        if diffs:
            raise LineageMismatch(new.run_id, diffs)
        kept = AGLineage.from_json(old)
        return replace(kept, finished_unix_ns=None), True
    atomic_write_json(path, new.to_json())
    return new, False


def close_lineage(run_dir: str | Path, lin: AGLineage) -> AGLineage:
    done = replace(lin, finished_unix_ns=time.time_ns())
    atomic_write_json(Path(run_dir) / "lineage.json", done.to_json())
    return done


def fail_on_mismatch(exc: LineageMismatch) -> None:
    print(str(exc), file=sys.stderr)
    raise SystemExit(EXIT_LINEAGE_MISMATCH)
