"""Closed-loop runner (SPEC 3.3, 4.2, 7.1).

One job = one run directory raw/<E>/<run_id>/. The runner builds the data, sets up the system,
emits ARRIVED / ADMITTED, forwards System.run events to the collector, mirrors CHUNK_RELEASED to
CHUNK_DELIVERED for in-process systems, turns each B0 TOKENS event into one chunk, writes
outcomes_run.jsonl per request, runs the checks, and finally score_outcomes.
"""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ag import SCHEMA_VERSION
from ag.data import mock_lc_inject
from ag.data.manifest import stamp, write_manifest
from ag.data.types import Generation, Request
from ag.harness.checks import all_ok, run_checks
from ag.harness.collector import Collector
from ag.harness.context import RunContext, make_clock
from ag.harness.checks import Check
from ag.harness.events import Event, EventType as E, JobKind, event_from_row
from ag.harness.lineage import build_lineage, close_lineage, open_lineage
from ag.harness.writer import JsonlWriter, atomic_write_json, atomic_write_text, drop_partial_tail, read_rows
from ag.metrics.score import check_record_covers_selector, score_outcomes
from ag.reference.record import RecordStore
from ag.sweep.expand import Job
from ag.systems.base import SystemConfig
from ag.systems.registry import create_system

_INDEX_LOCK = threading.Lock()


def _common_params(job: Mapping[str, Any], data: Mapping[str, Any], *, split: str, source: str,
                   max_new_default: int) -> dict[str, Any]:
    return {"n_pairs": int(data.get("n_attack", 4)), "n_benign": int(data.get("n_benign", data.get("n_attack", 4))),
            "length_classes": list(data.get("length_classes") or ["1k"]), "split": data.get("split", split),
            "source": data.get("source", source), "seed": int(data.get("seed", 0)),
            "max_new_tokens": int((job.get("fairness") or {}).get("max_new_tokens", max_new_default)),
            "sink": data.get("sink", "stream")}


def _gen_mock_lc_inject(job: Mapping[str, Any], data: Mapping[str, Any]) -> tuple[list[Request], dict]:
    params = _common_params(job, data, split="test", source="mock", max_new_default=64)
    return mock_lc_inject.generate(**params), params


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _background_path(p: str) -> Path:
    """Env vars expanded (FileNotFoundError with the download hint, ag.data.lc_inject.background); a
    relative path that is not under the cwd is tried against the repository root (smoke configs)."""
    from ag.config.schema import REPO
    from ag.data.lc_inject.background import resolve_path
    try:
        return resolve_path(p)
    except FileNotFoundError:
        alt = REPO / os.path.expandvars(p)
        if not Path(os.path.expandvars(p)).is_absolute() and alt.is_file():
            return alt
        raise


def _gen_lc_inject_draft(job: Mapping[str, Any], data: Mapping[str, Any]) -> tuple[list[Request], dict]:
    """Draft LC-Inject (S2 step 0): background docs + trusted tasks, length-fitted with the job target's
    tokenizer and chat template (ag.targets.registry.length_fn_for)."""
    from ag.data.lc_inject import draft
    from ag.targets.registry import length_fn_for
    params = _common_params(job, data, split="pilot", source="lc_draft", max_new_default=256)
    for k in ("tolerance", "positions", "attack_types"):
        if k in data:
            params[k] = tuple(data[k]) if isinstance(data[k], list) else data[k]
    given = data.get("background_paths")
    if not given:
        raise ValueError("data generator lc_inject_draft needs data.background_paths")
    given = [given] if isinstance(given, str) else [str(p) for p in given]
    paths = [_background_path(p) for p in given]
    flags = (job.get("system") or {}).get("flags") or {}
    length_fn = length_fn_for(str(job["target"]), model_path=flags.get("model_path"), targets_dir=flags.get("targets_dir"))
    reqs = draft.generate(**params, background_paths=[str(p) for p in paths], length_fn=length_fn)
    # manifest: paths as written in the config (not expanded) + the content hash of each file
    return reqs, {**params, "target": str(job["target"]), "background_paths": given,
                  "background_sha256": {g: _sha256_file(p) for g, p in zip(given, paths)}}


# data.generator -> fn(job, data) -> (requests, manifest params)
GENERATORS = {"mock_lc_inject": _gen_mock_lc_inject, "lc_inject_draft": _gen_lc_inject_draft}

# system flag engine -> lineage producer (Records and S are matched by producer, SPEC 4.9, 6.1)
PRODUCERS = {"mock": "mock", "hf": "reference"}


def build_requests(job: Mapping[str, Any]) -> tuple[list[Request], dict]:
    data = dict(job.get("data") or {})
    gen = data.get("generator", "mock_lc_inject")
    if gen not in GENERATORS:
        raise NotImplementedError(f"data generator {gen!r} is not implemented; known: {sorted(GENERATORS)}")
    reqs, params = GENERATORS[gen](job, data)
    return reqs, {"generator": gen, **params}


def _shard(reqs: Sequence[Request], shard: int, n: int) -> list[Request]:
    return [r for i, r in enumerate(reqs) if i % n == shard]


# job kind -> the block reason of that stage (SPEC 2.4 block_reasons; R_inj in SPEC 6.1)
JOB_REASON = {JobKind.INJ_JUDGE: "inj_window", JobKind.PROBE: "probe", JobKind.SANITIZE: "sanitize",
              JobKind.ATTRIBUTION: "attribution", JobKind.HARM_SCAN: "harm_input",
              JobKind.HARM_DEEP: "harm_input", JobKind.OUTPUT_GUARD: "output_guard"}


def outcome_run_row(req: Request, evs: Sequence[Event]) -> dict[str, Any]:
    fin = next((e for e in evs if e.type == E.FINISHED), None)
    err = next((e for e in evs if e.type == E.ERROR and e.fatal), None)
    gen: Generation | None = fin.generation if fin else None
    # block_reasons = every stage that judged unsafe, blocked or not (e.g. a late verdict under
    # release_before_verdict); block_reason stays the one that actually stopped the request
    reasons = [e.reason for e in evs if e.type in (E.CHUNK_BLOCKED, E.REQUEST_BLOCKED) and e.reason]
    judged: set[int] = set()
    uncached: set[int] = set()
    specs: set[str] = set()
    calls = 0
    selected: dict[int, list[int]] = {}
    for e in evs:
        if e.type == E.JOB_DONE:
            if e.job_kind in JOB_REASON and (any(v.unsafe for v in e.verdicts) or (e.result or {}).get("unsafe")):
                reasons.append(JOB_REASON[e.job_kind])
            for v in e.verdicts:
                specs.add(v.window_spec)
                if v.window_id is not None:
                    judged.add(v.window_id)
                    uncached.add(v.window_id)
                    calls += 1
        elif e.type in (E.CHUNK_GENERATED, E.REQUEST_BLOCKED):
            # cache hits create no job: their windows, unsafe verdicts and spec travel in the payload
            judged.update(e.payload.get("cached_windows", ()))
            if e.payload.get("cached_windows") and e.payload.get("window_spec"):
                specs.add(e.payload["window_spec"])
            if e.payload.get("cached_unsafe_windows"):
                reasons.append(JOB_REASON[JobKind.INJ_JUDGE])
            if "selected" in e.payload:
                selected[e.chunk_id if e.type == E.CHUNK_GENERATED else -1] = list(e.payload["selected"])
    prompt = dict(fin.payload.get("prompt", {})) if fin else {}
    return {
        "request_id": req.request_id, "variant_id": "live", "is_attack": req.is_attack,
        "blocked": bool(gen.blocked) if gen else False, "block_reason": gen.block_reason if gen else None,
        "block_reasons": sorted(set(reasons)),
        "error_kind": (err.reason or "error").split(":")[0] if err else None,
        "judged_windows": len(judged), "judged_windows_uncached": len(uncached), "judge_calls": calls,
        "judged_windows_spec": ",".join(sorted(specs)), "selected": selected,
        "payload_windows": prompt.get("payload_windows", []),
        "payload_windows_per_site": prompt.get("payload_windows_per_site", []),
        "n_windows": prompt.get("n_windows"), "n_eligible": prompt.get("n_eligible"),
        "prompt_len": prompt.get("n_tokens"), "harm_score": None, "doc_scores": {}, "cost": {},
        "generation": gen.to_json() if gen else None,
    }


def drive_request(system, req: Request, *, emit, clock, run_id: str, attempt: int, b0_chunks: bool,
                  t_source: str = "harness") -> Iterator[Event]:
    """Run one request through a system and yield every event after it was emitted.

    Harness-side events: ARRIVED, ADMITTED, CHUNK_DELIVERED (in-process: same time as CHUNK_RELEASED,
    SPEC 3.3) and, for systems without a gate (B0), CHUNK_GENERATED = RELEASED = DELIVERED per TOKENS.
    The released text travels in payload["text"] so a front end can stream it.
    """
    rid = req.request_id

    def h(type_: E, t: int, **kw) -> Event:
        ev = Event(run_id, rid, type_, t, t_source, attempt=attempt, **kw)
        emit(ev)
        return ev

    yield h(E.ARRIVED, clock.now_ns())
    yield h(E.ADMITTED, clock.now_ns())
    n_chunk = 0
    for e in system.run(req):
        emit(e)
        yield e
        if b0_chunks and e.type == E.TOKENS:
            kw = dict(chunk_id=n_chunk, tok_range=e.tok_range)
            yield h(E.CHUNK_GENERATED, e.t_ns, **kw)
            yield h(E.CHUNK_RELEASED, e.t_ns, payload=dict(e.payload), **kw)
            yield h(E.CHUNK_DELIVERED, e.t_ns, payload=dict(e.payload), **kw)
            n_chunk += 1
        elif e.type == E.CHUNK_RELEASED and not b0_chunks:
            yield h(E.CHUNK_DELIVERED, e.t_ns, chunk_id=e.chunk_id, payload=dict(e.payload))


def _done_requests(events_path: Path) -> tuple[set[str], int]:
    """Requests that already have FINISHED or a fatal ERROR, and the next attempt number."""
    done: set[str] = set()
    attempt = -1
    for row in read_rows(events_path):
        e = event_from_row(row)
        attempt = max(attempt, e.attempt)
        if e.type == E.FINISHED or (e.type == E.ERROR and e.fatal):
            done.add(e.request_id)
    return done, attempt + 1


def _append_index(out_root: Path, job: Job, run_dir: Path) -> None:
    """Append this run to raw/index.jsonl. Array tasks on other nodes append to the same file, so the
    dedup read and the append happen under an exclusive file lock and the row is fsynced before the
    lock is released (O_APPEND alone is not safe for concurrent appenders on NFS)."""
    idx = out_root / "raw" / "index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    row = {"path": str(run_dir.relative_to(out_root)), "exp_id": job.exp_id, "system_id": job.system_id,
           "run_id": job.run_id, "schema_version": SCHEMA_VERSION}
    with _INDEX_LOCK, open(idx, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if any(r.get("run_id") == job.run_id for r in read_rows(idx)):
                return
            f.seek(0, os.SEEK_END)
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def latest_eligible(lineage: Mapping[str, Any], engine: str | None, checks_ok: bool,
                    flags: Mapping[str, Any] | None = None) -> bool:
    """latest.txt eligibility, defined only here (SPEC 7.1). evaluation_only runs (B-oracle) use the
    answer key and never qualify (SPEC 9.1)."""
    return (lineage.get("clock") == "monotonic" and not lineage.get("code_dirty", True)
            and engine not in (None, "mock") and checks_ok
            and not bool((flags or {}).get("evaluation_only", False)))


def run_job(job: Job, out_root: str | Path, *, cmd: str | None = None, log: logging.Logger | None = None,
            limit: int | None = None, fairness_detail: str = "not compared (single job)") -> Path:
    out_root = Path(out_root)
    jr = job.resolved
    run_dir = out_root / "raw" / job.exp_id / job.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "server.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ag_log = logging.getLogger("ag")
    if ag_log.level == logging.NOTSET:
        ag_log.setLevel(logging.INFO)
    ag_log.addHandler(handler)
    try:
        return _run_job(job, out_root, run_dir, jr, cmd=cmd, log=log or logging.getLogger("ag.run"), limit=limit,
                        fairness_detail=fairness_detail)
    finally:
        ag_log.removeHandler(handler)
        handler.close()


def _run_job(job: Job, out_root: Path, run_dir: Path, jr: Mapping[str, Any], *, cmd: str | None,
             log: logging.Logger, limit: int | None, fairness_detail: str) -> Path:
    log.info("run %s start (%s)", job.run_id, job.system_id)
    check_record_covers_selector(jr)        # fail before generation, not in score_outcomes afterwards

    reqs, params = build_requests(jr)
    reqs, sha = stamp(reqs)
    write_manifest(out_root, reqs, sha, generator=params["generator"], params=params)
    reqs = _shard(reqs, job.shard, job.n_shards)
    if limit is not None:
        reqs = reqs[:limit]

    clock = make_clock(jr.get("clock", "monotonic"))
    sysd = jr["system"]
    system = create_system(sysd["impl"])
    flags = sysd["flags"]
    engine = str(flags.get("engine", "mock"))
    lin = build_lineage(job.run_id, job.exp_id, job.config_hash, clock=clock.kind, seed=job.seed,
                        producer=PRODUCERS.get(engine, engine),
                        model_sha={}, data_manifest_sha=sha, engine_version=flags.get("engine", "mock"),
                        threshold_ids=[jr.get("operating_point", "F5")], parent_run_ids=job.parent_run_ids)
    events_path = run_dir / "events.jsonl"
    for p in (events_path, run_dir / "outcomes_run.jsonl", run_dir / "records" / "index.jsonl"):
        # repair crash-left tails of this run's row files before deciding what is done
        drop_partial_tail(p)
    done, attempt = _done_requests(events_path) if (run_dir / "lineage.json").exists() else (set(), 0)
    # a request counts as done only with both its terminal event and its outcomes_run row (a kill
    # between the two re-runs it with attempt + 1; by_request keeps the latest attempt)
    done &= {r["request_id"] for r in read_rows(run_dir / "outcomes_run.jsonl")}
    collector = Collector(job.run_id, events_path)
    records = RecordStore(run_dir / "records", job.run_id)
    ctx = RunContext(job.run_id, job.exp_id, run_dir, clock, job.seed, lin, records, collector.emit, log, attempt)
    try:
        system.setup(SystemConfig(sysd["system_id"], sysd["impl"], flags, exp=jr), ctx)
        info = system.info()
        lin = dataclasses.replace(lin, model_sha=dict(info.model_sha), shim_active=info.shim_active,
                                  patch_sha=info.patch_sha,
                                  engine_version=info.flags.get("engine_version") or lin.engine_version,
                                  placement=dict(info.placement) or lin.placement)
        lin, _ = open_lineage(run_dir, lin)
    except Exception:
        collector.close()       # no stray writer thread (and no loaded model) after a refused setup or resume
        try:
            system.teardown()
        except Exception:
            log.exception("teardown after a failed setup")
        raise
    ctx.lineage = lin
    atomic_write_text(run_dir / "resolved.yaml", yaml.safe_dump(dict(jr), sort_keys=True))
    atomic_write_text(run_dir / "cmd.txt", (cmd or " ".join(sys.argv)) + "\n")

    src = "virtual" if clock.kind == "virtual" else "harness"
    b0_chunks = bool(info.flags.get("b0_chunks"))
    out_rows = JsonlWriter(run_dir / "outcomes_run.jsonl", job.run_id)
    try:
        for req in reqs:
            if req.request_id in done:
                continue
            evs = list(drive_request(system, req, emit=collector.emit, clock=clock, run_id=job.run_id,
                                     attempt=attempt, b0_chunks=b0_chunks, t_source=src))
            collector.flush()           # FINISHED / fatal ERROR is on disk before the outcomes_run row (resume order)
            out_rows.write(outcome_run_row(req, evs))
    finally:
        out_rows.close()
        system.teardown()
        collector.close()

    log.info("run %s generated %d requests", job.run_id, len(reqs))
    events = [event_from_row(r) for r in read_rows(events_path)]
    log_text = (run_dir / "server.log").read_text() if (run_dir / "server.log").exists() else ""
    checks = run_checks(events, info.flags, expected_requests=[r.request_id for r in reqs], log_text=log_text)
    have_rows = {r["request_id"] for r in read_rows(run_dir / "outcomes_run.jsonl")}
    missing = sorted({r.request_id for r in reqs} - have_rows)
    checks.append(Check("all_requests_have_outcomes", not missing, ", ".join(missing[:5])))
    checks.append(Check("fairness_knobs_equal", True, fairness_detail))
    atomic_write_json(run_dir / "checks.json", [c.to_json() for c in checks])   # before scoring (it may raise)
    lin = close_lineage(run_dir, lin)
    _append_index(out_root, job, run_dir)
    summary = score_outcomes(run_dir)
    meta = summary.get("_meta", {})
    has_s = bool(info.flags.get("b0_chunks")) or meta.get("s_source_run_id") is not None
    checks.append(Check("s_source_available", has_s,
                        "" if has_s else f"parent B0 run {job.parent_run_ids} not indexed or not scored yet"))
    atomic_write_json(run_dir / "checks.json", [c.to_json() for c in checks])
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        s["_meta"]["system_info"] = info.to_json()
        s["_meta"]["checks_ok"] = all_ok(checks)
        atomic_write_json(summary_path, s)
    if latest_eligible(lin.to_json(), info.engine, all_ok(checks), info.flags):
        atomic_write_text(out_root / "raw" / job.exp_id / "latest.txt", job.run_id + "\n")
    return run_dir
