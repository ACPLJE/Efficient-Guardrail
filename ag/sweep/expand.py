"""Experiment YAML -> jobs (SPEC 1.5, 8.1, 8.3).

Job axes = sweep.job_axes + the implicit axes (systems, targets, seeds, repeats, shard). One job
is one run_id and one Slurm array index. run_id is fixed once per job at expand time and written
to raw/<E>/jobs/<expand_ts>.jsonl; config_hash excludes notes, out_dir, slurm.time,
slurm.partition, shard and repeat.
"""
from __future__ import annotations

import copy
import itertools
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ag.config.schema import del_dotted, get_dotted, load_system, set_dotted
from ag.data.serde import JsonMixin
from ag.harness.writer import JsonlWriter
from ag.lineage_util import canonical_hash

HASH_EXCLUDE = ("notes", "out_dir", "slurm.time", "slurm.partition", "shard", "repeat")


@dataclass(frozen=True)
class Job(JsonMixin):
    run_id: str
    exp_id: str
    system_id: str
    target: str
    data_tag: str
    config_hash: str
    shard: int
    n_shards: int
    repeat: int
    seed: int
    resolved: Mapping[str, Any]
    parent_run_ids: tuple[str, ...] = ()
    axes: Mapping[str, Any] = field(default_factory=dict)


def config_hash(resolved_job: Mapping[str, Any]) -> str:
    d = copy.deepcopy(dict(resolved_job))
    for k in HASH_EXCLUDE:
        del_dotted(d, k)
    return canonical_hash(d)


def _tag(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]", "", str(s)) or "x"


def data_tag(job: Mapping[str, Any]) -> str:
    data = job.get("data") or {}
    wl = job.get("workload")
    if wl:
        return _tag(wl if isinstance(wl, str) else wl.get("id", "workload"))
    lcs = data.get("length_classes") or []
    lcs = lcs if isinstance(lcs, list) else [lcs]
    return "-".join([_tag(data.get("manifest", "data"))] + [_tag(x) for x in lcs])


def make_run_id(exp_id: str, system_id: str, target: str, dtag: str, chash: str, shard: int, repeat: int,
                seed: int, ts: str) -> str:
    parts = [exp_id, system_id, target, dtag, chash[:8], f"s{shard:02d}r{repeat}", str(seed), ts]
    for p in parts:
        if "__" in p or not p:
            raise ValueError(f"run_id component {p!r} is empty or contains '__'")
    return "__".join(parts)


def expand(exp: Mapping[str, Any], *, systems_dir: str | Path | None = None, ts: str | None = None,
           clock: str | None = None) -> list[Job]:
    ts = ts or time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    sweep = exp.get("sweep") or {}
    job_axes = [a for a in (sweep.get("job_axes") or []) if a not in ("systems", "targets")]
    n_shards = int((exp.get("slurm") or {}).get("shards", 1))
    seeds = exp.get("seeds") or [0]
    repeats = range(int(exp.get("repeats", 1)))
    axis_vals = []
    for a in job_axes:
        v = get_dotted(exp, a)
        axis_vals.append(v if isinstance(v, list) else [v])
    systems = {name: load_system(name, systems_dir) for name in exp["systems"]}
    jobs: list[Job] = []
    for sys_name, target, seed, rep, shard, combo in itertools.product(
            exp["systems"], exp["targets"], seeds, repeats, range(n_shards), itertools.product(*axis_vals)):
        r = copy.deepcopy(dict(exp))
        for a, v in zip(job_axes, combo):
            set_dotted(r, a, [v] if isinstance(get_dotted(exp, a), list) else v)
        sysd = systems[sys_name]
        r.pop("systems", None)
        r.pop("targets", None)
        r.pop("seeds", None)
        r.pop("repeats", None)          # the count; the job keeps its own repeat index (SPEC 8.1)
        r.update(system_id=sysd["system_id"], system=sysd, target=target, seed=seed, repeat=rep, shard=shard,
                 n_shards=n_shards, clock=clock or exp.get("clock", "monotonic"))
        chash = config_hash(r)
        dtag = data_tag(r)
        rid = make_run_id(exp["exp_id"], sysd["system_id"], target, dtag, chash, shard, rep, seed, ts)
        jobs.append(Job(rid, exp["exp_id"], sysd["system_id"], target, dtag, chash, shard, n_shards, rep, seed, r,
                        axes=dict(zip(job_axes, combo))))
    return _link_parents(jobs)


def _link_parents(jobs: Sequence[Job]) -> list[Job]:
    """Defended runs take S from the B0 run with the same target, data, seed, repeat and shard."""
    def key(j: Job) -> tuple:
        return (j.target, j.data_tag, j.seed, j.repeat, j.shard, canonical_hash(j.axes))
    b0 = {key(j): j.run_id for j in jobs if (j.resolved["system"]["flags"].get("gate") == "none")}
    out = []
    for j in jobs:
        p = b0.get(key(j))
        if p and p != j.run_id:
            j = Job(**{**j.__dict__, "parent_run_ids": (p,)})
        out.append(j)
    # B0 jobs first so their success/hits exist when defended runs are scored
    return sorted(out, key=lambda j: 0 if not j.parent_run_ids else 1)


def write_jobs(out_root: str | Path, jobs: Sequence[Job], ts: str) -> Path | None:
    if not jobs:
        return None
    path = Path(out_root) / "raw" / jobs[0].exp_id / "jobs" / f"{ts}.jsonl"
    if path.exists():           # written once per expand; array tasks re-expand with the same ts
        return path
    with JsonlWriter(path, run_id="") as w:
        for j in jobs:
            w.write({**j.to_json(), "run_id": j.run_id})
    return path
