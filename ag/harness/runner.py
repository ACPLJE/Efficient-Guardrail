"""Closed-loop runner (SPEC 3.3, 4.2, 7.1).

One job = one run directory raw/<E>/<run_id>/. The runner builds the data, sets up the system,
emits ARRIVED / ADMITTED, forwards System.run events to the collector, mirrors CHUNK_RELEASED to
CHUNK_DELIVERED for in-process systems, turns each B0 TOKENS event into one chunk, writes
outcomes_run.jsonl per request, runs the checks, and finally score_outcomes.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ag.data import mock_lc_inject
from ag.data.manifest import stamp, write_manifest
from ag.data.types import Generation, Request
from ag.harness.checks import all_ok, run_checks
from ag.harness.collector import Collector
from ag.harness.context import RunContext, make_clock
from ag.harness.checks import Check
from ag.harness.events import Event, EventType as E, JobKind, event_from_row
from ag.harness.lineage import build_lineage, close_lineage, open_lineage
from ag.harness.writer import JsonlWriter, atomic_write_json, atomic_write_text, read_rows
from ag.metrics.score import score_outcomes
from ag.reference.record import RecordStore
from ag.sweep.expand import Job
from ag.systems.base import SystemConfig
from ag.systems.registry import create_system

_INDEX_LOCK = threading.Lock()


def build_requests(job: Mapping[str, Any]) -> tuple[list[Request], dict]:
    data = dict(job.get("data") or {})
    gen = data.get("generator", "mock_lc_inject")
    if gen != "mock_lc_inject":
        raise NotImplementedError(f"data generator {gen!r} is S2 work (S1 has mock_lc_inject only)")
    params = {"n_pairs": int(data.get("n_attack", 4)), "n_benign": int(data.get("n_benign", data.get("n_attack", 4))),
              "length_classes": list(data.get("length_classes") or ["1k"]), "split": data.get("split", "test"),
              "source": data.get("source", "mock"), "seed": int(data.get("seed", 0)),
              "max_new_tokens": int((job.get("fairness") or {}).get("max_new_tokens", 64)),
              "sink": data.get("sink", "stream")}
    return mock_lc_inject.generate(**params), {"generator": gen, **params}


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
        elif e.type == E.CHUNK_GENERATED:
            judged.update(e.payload.get("cached_windows", ()))
            if "selected" in e.payload:
                selected[e.chunk_id] = list(e.payload["selected"])
        elif e.type == E.REQUEST_BLOCKED:
            judged.update(e.payload.get("cached_windows", ()))
            if "selected" in e.payload:
                selected[-1] = list(e.payload["selected"])
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
    idx = out_root / "raw" / "index.jsonl"
    with _INDEX_LOCK:
        if any(r.get("run_id") == job.run_id for r in read_rows(idx)):
            return
        with JsonlWriter(idx, job.run_id, repair_tail=False) as w:
            w.write({"path": str(run_dir.relative_to(out_root)), "exp_id": job.exp_id, "system_id": job.system_id})


def latest_eligible(lineage: Mapping[str, Any], engine: str | None, checks_ok: bool) -> bool:
    """latest.txt eligibility, defined only here (SPEC 7.1)."""
    return (lineage.get("clock") == "monotonic" and not lineage.get("code_dirty", True)
            and engine not in (None, "mock") and checks_ok)


def run_job(job: Job, out_root: str | Path, *, cmd: str | None = None, log: logging.Logger | None = None,
            limit: int | None = None) -> Path:
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
        return _run_job(job, out_root, run_dir, jr, cmd=cmd, log=log or logging.getLogger("ag.run"), limit=limit)
    finally:
        ag_log.removeHandler(handler)
        handler.close()


def _run_job(job: Job, out_root: Path, run_dir: Path, jr: Mapping[str, Any], *, cmd: str | None,
             log: logging.Logger, limit: int | None) -> Path:
    log.info("run %s start (%s)", job.run_id, job.system_id)

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
    lin = build_lineage(job.run_id, job.exp_id, job.config_hash, clock=clock.kind, seed=job.seed,
                        producer="mock" if flags.get("engine", "mock") == "mock" else str(flags.get("engine")),
                        model_sha={}, data_manifest_sha=sha, engine_version=flags.get("engine", "mock"),
                        threshold_ids=[jr.get("operating_point", "F5")], parent_run_ids=job.parent_run_ids)
    events_path = run_dir / "events.jsonl"
    done, attempt = _done_requests(events_path) if (run_dir / "lineage.json").exists() else (set(), 0)
    # a request counts as done only with both its terminal event and its outcomes_run row (a kill
    # between the two re-runs it with attempt + 1; by_request keeps the latest attempt)
    done &= {r["request_id"] for r in read_rows(run_dir / "outcomes_run.jsonl")}
    collector = Collector(job.run_id, events_path)
    records = RecordStore(run_dir / "records", job.run_id)
    ctx = RunContext(job.run_id, job.exp_id, run_dir, clock, job.seed, lin, records, collector.emit, log, attempt)
    system.setup(SystemConfig(sysd["system_id"], sysd["impl"], flags, exp=jr), ctx)
    info = system.info()
    lin = dataclasses.replace(lin, model_sha=dict(info.model_sha), shim_active=info.shim_active, patch_sha=info.patch_sha)
    try:
        lin, resumed = open_lineage(run_dir, lin)
    except Exception:
        collector.close()
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
    atomic_write_json(run_dir / "checks.json", [c.to_json() for c in checks])
    lin = close_lineage(run_dir, lin)
    _append_index(out_root, job, run_dir)
    score_outcomes(run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        s["_meta"]["system_info"] = info.to_json()
        s["_meta"]["checks_ok"] = all_ok(checks)
        atomic_write_json(summary_path, s)
    if latest_eligible(lin.to_json(), info.engine, all_ok(checks)):
        atomic_write_text(out_root / "raw" / job.exp_id / "latest.txt", job.run_id + "\n")
    return run_dir
