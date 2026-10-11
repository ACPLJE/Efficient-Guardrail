"""Events (SPEC 3). Plugins only emit events; metrics are computed from them by ag/metrics."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

from ag import SCHEMA_VERSION
from ag.data.serde import JsonMixin
from ag.data.types import Generation, Verdict


class EventType(IntEnum):
    ARRIVED = 1
    ADMITTED = 2
    PREFILL_START = 3
    PREFILL_DONE = 4
    TOKENS = 5
    CHUNK_GENERATED = 6
    JOB_SUBMITTED = 7
    JOB_STARTED = 8
    JOB_DONE = 9
    CHUNK_RELEASED = 10
    CHUNK_BLOCKED = 11
    CHUNK_DELIVERED = 12
    REQUEST_BLOCKED = 13
    FINISHED = 14
    ERROR = 15


class JobKind(StrEnum):
    HARM_SCAN = "harm_scan"
    HARM_DEEP = "harm_deep"
    OUTPUT_GUARD = "output_guard"
    INJ_JUDGE = "inj_judge"
    ATTRIBUTION = "attribution"
    PROBE = "probe"
    SANITIZE = "sanitize"
    TOOL_COMMIT = "tool_commit"


class DeadlineKind(StrEnum):
    FIRST_TOKEN = "first_token"
    RELEASE = "release"
    COMMIT = "commit"
    NONE = "none"


TSource = Literal["harness", "engine", "client", "judge_pool", "virtual"]


@dataclass(frozen=True)
class Event(JsonMixin):
    run_id: str
    request_id: str
    type: EventType
    t_ns: int
    t_source: TSource
    seq: int = -1
    attempt: int = 0
    chunk_id: int | None = None
    tok_range: tuple[int, int] | None = None
    job_id: str | None = None
    job_kind: JobKind | None = None
    deadline_kind: DeadlineKind | None = None
    deadline_ns: int | None = None
    window_ids: tuple[int, ...] = ()
    required_jobs: tuple[str, ...] = ()
    cached_verdict_ns: int | None = None
    verdicts: tuple[Verdict, ...] = ()
    result: Mapping[str, Any] | None = None
    generation: Generation | None = None
    fatal: bool = False
    reason: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


def sort_key(e: Event) -> tuple:
    """(t_ns, request_id, type, chunk_id or -1, job_id or "", seq) (SPEC 3.2)."""
    return (e.t_ns, e.request_id, int(e.type), -1 if e.chunk_id is None else e.chunk_id, e.job_id or "", e.seq)


def sorted_events(events: Iterable[Event]) -> list[Event]:
    return sorted(events, key=sort_key)


def event_row(e: Event) -> dict:
    row = e.to_json()
    row["schema_version"] = SCHEMA_VERSION
    return row


def event_from_row(row: Mapping[str, Any]) -> Event:
    v = row.get("schema_version", SCHEMA_VERSION)
    if v != SCHEMA_VERSION:
        raise ValueError(f"event schema_version {v} != {SCHEMA_VERSION}")
    return Event.from_json({k: x for k, x in row.items() if k != "schema_version"})


def by_request(events: Iterable[Event], *, latest_attempt: bool = True) -> dict[str, list[Event]]:
    """Group events per request, sorted by the SPEC key. Earlier attempts are dropped (SPEC 7.1)."""
    out: dict[str, list[Event]] = {}
    for e in events:
        out.setdefault(e.request_id, []).append(e)
    for rid, evs in out.items():
        if latest_attempt:
            a = max(e.attempt for e in evs)
            evs = [e for e in evs if e.attempt == a]
        out[rid] = sorted_events(evs)
    return out
