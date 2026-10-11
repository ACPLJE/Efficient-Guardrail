"""Post-run checks (SPEC 3.4 invariants, 7.4 checks.json). Each check is Check(name, ok, detail)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ag.data.serde import JsonMixin
from ag.harness.events import Event, EventType as E, JobKind, by_request

CLIENT_WAIT_BOUND_NS = 1_000_000
DEADLINE_SLACK_NS = 1_000_000      # gate poll granularity allowed after a deadline before the fail-closed block


@dataclass(frozen=True)
class Check(JsonMixin):
    name: str
    ok: bool
    detail: str = ""


def _first(evs: list[Event], t: E) -> Event | None:
    return next((e for e in evs if e.type == t), None)


def _check_request(rid: str, evs: list[Event], flags: Mapping[str, Any]) -> dict[str, list[str]]:
    bad: dict[str, list[str]] = {}

    def fail(name: str, msg: str) -> None:
        bad.setdefault(name, []).append(f"{rid}: {msg}")

    # 1. lifecycle order, no prefill for a request refused before generation
    seq = [_first(evs, t) for t in (E.ARRIVED, E.ADMITTED, E.PREFILL_START, E.PREFILL_DONE)]
    present = [e for e in seq if e is not None]
    if any(a.t_ns > b.t_ns for a, b in zip(present, present[1:])):
        fail("invariants.lifecycle_order", "ARRIVED <= ADMITTED <= PREFILL_START <= PREFILL_DONE violated")
    if _first(evs, E.REQUEST_BLOCKED) and not _first(evs, E.TOKENS):
        if seq[2] or seq[3]:
            fail("invariants.lifecycle_order", "REQUEST_BLOCKED before generation but PREFILL_* present")

    done = {e.job_id: e for e in evs if e.type == E.JOB_DONE}
    submitted = {e.job_id: e for e in evs if e.type == E.JOB_SUBMITTED}
    gen = {e.chunk_id: e for e in evs if e.type == E.CHUNK_GENERATED}
    rel = {e.chunk_id: e for e in evs if e.type == E.CHUNK_RELEASED}
    blk = {e.chunk_id: e for e in evs if e.type == E.CHUNK_BLOCKED}
    dlv = {e.chunk_id: e for e in evs if e.type == E.CHUNK_DELIVERED}
    rbv = bool(flags.get("release_before_verdict", False))

    # 2. release only after generation and all gating verdicts, and only on safe verdicts
    for j, r in rel.items():
        g = gen.get(j)
        if g is None or r.t_ns < g.t_ns:
            fail("invariants.release_after_generated", f"chunk {j} released before CHUNK_GENERATED")
            continue
        if rbv:
            continue
        for jid in g.required_jobs:
            d = done.get(jid)
            if d is None or d.t_ns > r.t_ns:
                fail("no_release_before_verdict", f"chunk {j} released before {jid} finished")
            elif any(v.unsafe for v in d.verdicts):
                fail("no_release_before_verdict", f"chunk {j} released despite an unsafe verdict in {jid}")
            elif d.result and d.result.get("unsafe"):
                fail("no_release_before_verdict", f"chunk {j} released despite unsafe result in {jid}")
        if g.cached_verdict_ns is not None and g.cached_verdict_ns > r.t_ns:
            fail("no_release_before_verdict", f"chunk {j} released before its cached verdict time")

    # one gate decision per chunk (released xor blocked, at most once)
    decisions: dict[int, int] = {}
    for e in evs:
        if e.type in (E.CHUNK_RELEASED, E.CHUNK_BLOCKED):
            decisions[e.chunk_id] = decisions.get(e.chunk_id, 0) + 1
    for j, n in decisions.items():
        if n > 1:
            fail("invariants.one_decision_per_chunk", f"chunk {j} has {n} release/block decisions")

    # 3. in-order release; 4. nothing released after a block
    rel_ids = sorted(rel)
    for a, b in zip(rel_ids, rel_ids[1:]):
        if b != a + 1 or rel[b].t_ns < rel[a].t_ns:
            fail("invariants.in_order_release", f"chunk {b} released out of order after {a}")
    if rel_ids and rel_ids[0] != 0:
        fail("invariants.in_order_release", f"first released chunk is {rel_ids[0]}")
    if blk:
        jb = min(blk)
        if any(j > jb for j in rel):
            fail("invariants.no_release_after_block", f"chunk released after chunk {jb} was blocked")

    # 5. first chunk waits for the harm scan
    scans = [d for d in done.values() if d.job_kind == JobKind.HARM_SCAN]
    if 0 in rel and flags.get("harm_scan", "off") != "off":
        if not scans:
            fail("invariants.harm_scan_before_release", "harm_scan on but no HARM_SCAN job before chunk 0")
        elif any(d.t_ns > rel[0].t_ns for d in scans):
            fail("invariants.harm_scan_before_release", "chunk 0 released before HARM_SCAN finished")

    # 6. fail-closed deadlines: a gating job without a verdict by its deadline -> the chunk is
    # CHUNK_BLOCKED(reason="deadline_miss") at the deadline; it is never released
    if not rbv:
        stop = min(blk) if blk else None
        for j, g in gen.items():
            if stop is not None and j > stop:
                continue
            late = []
            for jid in g.required_jobs:
                s, d = submitted.get(jid), done.get(jid)
                if s is not None and s.deadline_ns is not None and (d is None or d.t_ns > s.deadline_ns):
                    late.append((jid, s.deadline_ns))
            if not late:
                continue
            dl = min(t for _, t in late)
            if j in rel:
                fail("invariants.deadline_fail_closed", f"chunk {j} released on {late[0][0]} after its deadline")
            elif j in blk and (blk[j].reason != "deadline_miss" or blk[j].t_ns > dl + DEADLINE_SLACK_NS):
                fail("invariants.deadline_fail_closed",
                     f"chunk {j} missed its deadline but was blocked with reason={blk[j].reason!r} at {blk[j].t_ns}")

    # 7. exactly one terminal event; 8. FINISHED after all JOB_DONE
    fin = [e for e in evs if e.type == E.FINISHED]
    fatal = [e for e in evs if e.type == E.ERROR and e.fatal]
    if len(fin) + len(fatal) != 1:
        fail("invariants.one_terminal", f"{len(fin)} FINISHED + {len(fatal)} fatal ERROR")
    if fin and done and max(d.t_ns for d in done.values()) > fin[0].t_ns:
        fail("invariants.finished_after_jobs", "JOB_DONE after FINISHED")

    # 9. every chunk is gated when the system judges
    if flags.get("uses_judge", False) and flags.get("inj_path", "on") != "off":
        for j, g in gen.items():
            exempt = flags.get("block_before_generation", False) and j >= 1
            if not exempt and not g.required_jobs and g.cached_verdict_ns is None:
                fail("chunk_without_verdict", f"chunk {j} has no required_jobs and no cached verdict")

    # held chunks: every generated chunk up to the first block is released or blocked
    stop = min(blk) if blk else None
    for j in gen:
        if (stop is None or j <= stop) and j not in rel and j not in blk:
            fail("held_chunks_at_end", f"chunk {j} neither released nor blocked")
    for j, r in rel.items():
        d = dlv.get(j)
        if d is None or d.t_ns < r.t_ns:
            fail("invariants.delivered_after_released", f"chunk {j} not delivered after release")

    a, q = seq[0], seq[1]
    if a is not None and q is not None and q.t_ns - a.t_ns > CLIENT_WAIT_BOUND_NS:
        fail("client_wait_bounded", f"q - a = {q.t_ns - a.t_ns} ns")
    return bad


CHECK_NAMES = (
    "invariants.lifecycle_order", "invariants.release_after_generated", "no_release_before_verdict",
    "invariants.one_decision_per_chunk", "invariants.in_order_release", "invariants.no_release_after_block", "invariants.harm_scan_before_release",
    "invariants.deadline_fail_closed", "invariants.one_terminal", "invariants.finished_after_jobs",
    "chunk_without_verdict", "held_chunks_at_end", "invariants.delivered_after_released", "client_wait_bounded",
)


def run_checks(events: Iterable[Event], flags: Mapping[str, Any], *, expected_requests: Iterable[str] = (),
               log_text: str = "") -> list[Check]:
    per = by_request(events)
    failures: dict[str, list[str]] = {}
    for rid, evs in per.items():
        for k, v in _check_request(rid, evs, flags).items():
            failures.setdefault(k, []).extend(v)
    out = [Check(n, n not in failures, "; ".join(failures.get(n, [])[:5])) for n in CHECK_NAMES]
    missing = sorted(set(expected_requests) - set(per))
    out.append(Check("all_requests_have_events", not missing, ", ".join(missing[:5])))
    out.append(Check("no_traceback_in_logs", "Traceback (most recent call last)" not in log_text, ""))
    return out


def all_ok(checks: Iterable[Check]) -> bool:
    return all(c.ok for c in checks)
