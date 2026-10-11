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


def _make_job(exp: Mapping[str, Any], sysd: Mapping[str, Any], target: str, seed: int, rep: int, shard: int,
              n_shards: int, axes: Mapping[str, Any], *, ts: str, clock: str | None) -> Job:
    r = copy.deepcopy(dict(exp))
    for a, v in axes.items():
        set_dotted(r, a, [v] if isinstance(get_dotted(exp, a), list) and not isinstance(v, list) else v)
    r.pop("systems", None)
    r.pop("targets", None)
    r.pop("seeds", None)
    r.pop("repeats", None)          # the count; the job keeps its own repeat index (SPEC 8.1)
    r.update(system_id=sysd["system_id"], system=sysd, target=target, seed=seed, repeat=rep, shard=shard,
             n_shards=n_shards, clock=clock or exp.get("clock", "monotonic"))
    chash = config_hash(r)
    dtag = data_tag(r)
    rid = make_run_id(exp["exp_id"], sysd["system_id"], target, dtag, chash, shard, rep, seed, ts)
    return Job(rid, exp["exp_id"], sysd["system_id"], target, dtag, chash, shard, n_shards, rep, seed, r,
               axes=dict(axes))


def expand(exp: Mapping[str, Any], *, systems_dir: str | Path | None = None, ts: str | None = None,
           clock: str | None = None) -> list[Job]:
    """Jobs of an experiment: the product of the implicit axes (systems, targets, seeds, repeats, shards)
    and sweep.job_axes, plus one extra combination per sweep.include entry (SPEC 8.1)."""
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
        jobs.append(_make_job(exp, systems[sys_name], target, seed, rep, shard, n_shards, dict(zip(job_axes, combo)),
                              ts=ts, clock=clock))
    for inc in sweep.get("include") or []:          # combinations outside the product
        inc = dict(inc)
        inc_systems = inc.pop("systems", exp["systems"])
        inc_targets = inc.pop("targets", exp["targets"])
        inc_systems = inc_systems if isinstance(inc_systems, list) else [inc_systems]
        inc_targets = inc_targets if isinstance(inc_targets, list) else [inc_targets]
        for name in inc_systems:
            systems.setdefault(name, load_system(name, systems_dir))
        for sys_name, target, seed, rep, shard in itertools.product(inc_systems, inc_targets, seeds, repeats,
                                                                     range(n_shards)):
            jobs.append(_make_job(exp, systems[sys_name], target, seed, rep, shard, n_shards, inc, ts=ts, clock=clock))
    if len({j.run_id for j in jobs}) != len(jobs):
        raise ValueError("sweep.include repeats a combination of the product (duplicate run_id)")
    return _link_parents(jobs)


# fairness knobs that every system of one experiment must share (SPEC 8.3)
FAIR_KNOBS = ("max_new_tokens", "temperature", "ignore_eos", "dtype", "kv_capacity_tokens", "max_running_requests",
              "chunked_prefill_size", "attention_backend", "cuda_graph", "target_revision")


def fairness_knobs(job: Job) -> dict[str, Any]:
    """Effective fairness knobs of a job: the system flag when set, else the experiment's fairness value.
    AG-family systems (gate != none) also carry their effective chunking."""
    r = job.resolved
    flags = r["system"]["flags"]
    fair = dict(r.get("fairness") or {})
    knobs: dict[str, Any] = {k: flags.get(k, fair.get(k)) for k in FAIR_KNOBS}
    if flags.get("gate", "ag") != "none":
        knobs["chunking"] = dict(flags.get("chunking") or r.get("chunking") or {})
    return knobs


def fairness_mismatches(jobs: Sequence[Job], exempt: Sequence[str] = ()) -> list[str]:
    """checks.fairness_knobs_equal (SPEC 8.3): within one condition (target, data, seed, repeat, shard,
    job axes) every system must share FAIR_KNOBS, and every AG-family system must use the experiment's
    chunking (hit@k[union] and the offline files are computed with it). Keys in `exempt`
    (exp.fairness_exempt) are experiment axes and are skipped. Returns one message per mismatch."""
    groups: dict[tuple, list[Job]] = {}
    for j in jobs:
        groups.setdefault((j.target, j.data_tag, j.seed, j.repeat, j.shard, canonical_hash(j.axes)), []).append(j)
    out: list[str] = []
    for key, members in groups.items():
        ref = members[0]
        rk = fairness_knobs(ref)
        for j in members[1:]:
            jk = fairness_knobs(j)
            for k in FAIR_KNOBS:
                if k in exempt:
                    continue
                if rk.get(k) != jk.get(k):
                    out.append(f"{j.system_id} vs {ref.system_id} ({key[0]}, {key[1]}, seed {key[2]}): "
                               f"{k} {jk.get(k)!r} != {rk.get(k)!r}")
        if "chunking" in exempt:
            continue
        for j in members:
            k = fairness_knobs(j)
            exp_chunking = dict(j.resolved.get("chunking") or {})
            if "chunking" in k and k["chunking"] != exp_chunking:
                out.append(f"{j.system_id} ({key[0]}, {key[1]}): system chunking {k['chunking']} != experiment "
                           f"chunking {exp_chunking} (set chunking in the experiment file; hit@k[union] uses it)")
    return out


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


def _now_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def fresh_ts(jobs_dir: str | Path) -> str:
    """A new expand timestamp: never one that already has a jobs file (another expand of the same
    exp_id in the same second would otherwise share it). Waits for the next second."""
    jobs_dir = Path(jobs_dir)
    while True:
        ts = _now_ts()
        if not (jobs_dir / f"{ts}.jsonl").exists():
            return ts
        _sleep(1.0)


class JobsFileMismatch(ValueError):
    pass


def write_jobs(out_root: str | Path, jobs: Sequence[Job], ts: str) -> Path | None:
    """Write raw/<E>/jobs/<ts>.jsonl once. Array tasks re-expand with the same ts and find the file;
    run_ids that differ from it (config or flags changed since expand) raise JobsFileMismatch."""
    if not jobs:
        return None
    path = Path(out_root) / "raw" / jobs[0].exp_id / "jobs" / f"{ts}.jsonl"
    if path.exists():
        from ag.harness.writer import read_rows
        want = [r["run_id"] for r in read_rows(path)]
        got = [j.run_id for j in jobs]
        if want != got:
            raise JobsFileMismatch(f"run_ids differ from {path} (config or --clock changed since expand):\n"
                                   f"  expanded: {got}\n  jobs file: {want}")
        return path
    with JsonlWriter(path, run_id="") as w:
        for j in jobs:
            w.write({**j.to_json(), "run_id": j.run_id})
    return path
