"""Per-request latency metrics from events (SPEC 6.2, D-004). One formula = one function.

Times: a = ARRIVED, q = ADMITTED, s = PREFILL_START, f = PREFILL_DONE, t_last = last TOKENS,
g_j / l_j / b_j / d_j = CHUNK_GENERATED / RELEASED / BLOCKED / DELIVERED of chunk j,
n_j = tokens of chunk j, J = delivered chunks, V_j = time all gating verdicts of chunk j are in.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ag.harness.events import Event, EventType as E, sorted_events

INF = math.inf


@dataclass
class ReqTimes:
    request_id: str
    a: int | None = None
    q: int | None = None
    s: int | None = None
    f: int | None = None
    t_last: int | None = None
    T: int = 0
    g: dict[int, int] = field(default_factory=dict)
    l: dict[int, int] = field(default_factory=dict)
    b: dict[int, int] = field(default_factory=dict)
    d: dict[int, int] = field(default_factory=dict)
    n: dict[int, int] = field(default_factory=dict)
    V: dict[int, float] = field(default_factory=dict)

    @property
    def J(self) -> list[int]:
        return sorted(self.d)

    @property
    def n_vis(self) -> int:
        return sum(self.n.get(j, 0) for j in self.J)


def times_from_events(events: Iterable[Event]) -> ReqTimes:
    evs = sorted_events(events)
    if not evs:
        raise ValueError("no events")
    rt = ReqTimes(evs[0].request_id)
    done: dict[str, int] = {}
    gens: dict[int, Event] = {}
    for e in evs:
        if e.type == E.ARRIVED and rt.a is None:
            rt.a = e.t_ns
        elif e.type == E.ADMITTED and rt.q is None:
            rt.q = e.t_ns
        elif e.type == E.PREFILL_START and rt.s is None:
            rt.s = e.t_ns
        elif e.type == E.PREFILL_DONE and rt.f is None:
            rt.f = e.t_ns
        elif e.type == E.TOKENS:
            rt.t_last = e.t_ns
            if e.tok_range:
                rt.T = max(rt.T, e.tok_range[1])
        elif e.type == E.CHUNK_GENERATED:
            rt.g[e.chunk_id] = e.t_ns
            gens[e.chunk_id] = e
            if e.tok_range:
                rt.n[e.chunk_id] = e.tok_range[1] - e.tok_range[0]
        elif e.type == E.CHUNK_RELEASED:
            rt.l.setdefault(e.chunk_id, e.t_ns)
        elif e.type == E.CHUNK_BLOCKED:
            rt.b.setdefault(e.chunk_id, e.t_ns)
        elif e.type == E.CHUNK_DELIVERED:
            rt.d.setdefault(e.chunk_id, e.t_ns)
        elif e.type == E.JOB_DONE and e.job_id:
            done[e.job_id] = e.t_ns
    for j, ge in gens.items():
        rt.V[j] = verdict_time(ge, done)
    return rt


def verdict_time(gen_event: Event, job_done: dict[str, int]) -> float:
    """V_j = max(JOB_DONE of required_jobs, cached_verdict_ns); +inf if a job never finished;
    -inf when nothing gates the chunk."""
    ts: list[float] = []
    for jid in gen_event.required_jobs:
        ts.append(job_done.get(jid, INF))
    if gen_event.cached_verdict_ns is not None:
        ts.append(gen_event.cached_verdict_ns)
    return max(ts) if ts else -INF


def _sub(x: int | None, y: int | None) -> int | None:
    return None if x is None or y is None else x - y


def ttft_gen(rt: ReqTimes) -> int | None:
    return _sub(rt.f, rt.a)


def prefill_time(rt: ReqTimes) -> int | None:
    return _sub(rt.f, rt.s)


def tpot_gen(rt: ReqTimes) -> float | None:
    if rt.f is None or rt.t_last is None or rt.T < 2:
        return None
    return (rt.t_last - rt.f) / (rt.T - 1)


def ttfv(rt: ReqTimes) -> int | None:
    return _sub(rt.d.get(0), rt.a)


def ttfv_engine(rt: ReqTimes) -> int | None:
    return _sub(rt.l.get(0), rt.a)


def release_lags(rt: ReqTimes) -> list[int]:
    return [rt.l[j] - rt.g[j] for j in sorted(rt.l) if j in rt.g]


def block_lags(rt: ReqTimes) -> list[int]:
    return [rt.b[j] - rt.g[j] for j in sorted(rt.b) if j in rt.g]


def tpot_vis(rt: ReqTimes) -> float | None:
    J = rt.J
    if len(J) < 2:
        return None
    return (rt.d[J[-1]] - rt.d[J[0]]) / (rt.n_vis - rt.n.get(J[0], 0))


def itl_vis(rt: ReqTimes) -> list[tuple[float, int]]:
    """(value, weight) per delivered chunk j >= 1: (d_j - d_{j-1}) / n_j, counted n_j times."""
    J = rt.J
    out = []
    for prev, j in zip(J, J[1:]):
        nj = rt.n.get(j, 0)
        if nj > 0:
            out.append(((rt.d[j] - rt.d[prev]) / nj, nj))
    return out


def itl_vis_flat(rt: ReqTimes) -> list[float]:
    return [v for v, w in itl_vis(rt) for _ in range(w)]


def exposed_tokens(rt: ReqTimes) -> int:
    return sum(rt.n.get(j, 0) for j in rt.J if rt.d[j] < rt.V.get(j, -INF))


def gate_delays(rt: ReqTimes) -> list[float]:
    out = []
    for j in sorted(rt.l):
        v = rt.V.get(j, -INF)
        if j in rt.g and math.isfinite(v):
            out.append(rt.l[j] - max(rt.g[j], v))
    return out


def e2e_vis(rt: ReqTimes) -> int | None:
    J = rt.J
    return _sub(rt.d[J[-1]], rt.a) if J else None


def client_wait(rt: ReqTimes) -> int | None:
    return _sub(rt.q, rt.a)


def percentiles(values: Sequence[float], ps: Sequence[int] = (50, 90, 95, 99)) -> dict[int, float | None]:
    """numpy linear percentiles; method="higher" when infinities are present (SPEC 6.2)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {p: None for p in ps}
    method = "linear" if np.all(np.isfinite(v)) else "higher"
    return {p: float(np.percentile(v, p, method=method)) for p in ps}
