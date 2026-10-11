"""SPEC 6.2 identities on scripted events, plus V_{r,j} and the B1b / N5 shapes on mock runs."""
import math

import pytest

from ag.data.types import Verdict
from ag.harness.events import Event, EventType as E
from ag.metrics import per_request as pr


def ev(t, typ, **kw):
    return Event("run", "q", typ, t, "virtual", **kw)


def scripted():
    v = Verdict("MJ", "q", 0, 1, "128x2", 0.1, 0.5, "F5", False, 0, 0)
    return [
        ev(0, E.ARRIVED), ev(0, E.ADMITTED), ev(10, E.PREFILL_START), ev(100, E.PREFILL_DONE),
        ev(170, E.TOKENS, tok_range=(0, 8)),
        ev(170, E.CHUNK_GENERATED, chunk_id=0, tok_range=(0, 8), required_jobs=("j0",)),
        ev(190, E.JOB_DONE, job_id="j0", verdicts=(v,)),
        ev(195, E.CHUNK_RELEASED, chunk_id=0), ev(197, E.CHUNK_DELIVERED, chunk_id=0),
        ev(490, E.TOKENS, tok_range=(8, 40)),
        ev(490, E.CHUNK_GENERATED, chunk_id=1, tok_range=(8, 40), cached_verdict_ns=150),
        ev(491, E.CHUNK_RELEASED, chunk_id=1), ev(493, E.CHUNK_DELIVERED, chunk_id=1),
        ev(560, E.TOKENS, tok_range=(40, 48)),
        ev(560, E.CHUNK_GENERATED, chunk_id=2, tok_range=(40, 48), required_jobs=("j2",)),
        ev(600, E.JOB_DONE, job_id="j2", verdicts=(v,)),
        ev(601, E.CHUNK_RELEASED, chunk_id=2), ev(605, E.CHUNK_DELIVERED, chunk_id=2),
    ]


def test_identity_1_ttfv_decomposition():
    rt = pr.times_from_events(scripted())
    ttfv = pr.ttfv(rt)
    assert ttfv == pr.ttft_gen(rt) + (rt.g[0] - rt.f) + (rt.l[0] - rt.g[0]) + (rt.d[0] - rt.l[0]) == 197
    assert pr.ttfv_engine(rt) == 195 and pr.prefill_time(rt) == 90


def test_identity_2_itl_sums():
    rt = pr.times_from_events(scripted())
    itl = pr.itl_vis(rt)
    total = sum(v * w for v, w in itl)
    assert total == pytest.approx(rt.d[2] - rt.d[0])
    assert pr.tpot_vis(rt) * (rt.n_vis - rt.n[0]) == pytest.approx(total)
    assert len(pr.itl_vis_flat(rt)) == 40


def test_v_and_gate_metrics():
    rt = pr.times_from_events(scripted())
    assert rt.V == {0: 190, 1: 150, 2: 600}
    assert pr.release_lags(rt) == [25, 1, 41]
    assert pr.gate_delays(rt) == [5, 1, 1]
    assert pr.exposed_tokens(rt) == 0
    assert pr.tpot_gen(rt) == pytest.approx((560 - 100) / 47)
    assert pr.e2e_vis(rt) == 605


def test_v_infinities():
    g = ev(1, E.CHUNK_GENERATED, chunk_id=0, required_jobs=("never",))
    assert pr.verdict_time(g, {}) == math.inf
    assert pr.verdict_time(ev(1, E.CHUNK_GENERATED, chunk_id=0), {}) == -math.inf


def test_percentiles_inf_uses_higher():
    p = pr.percentiles([1.0, 2.0, math.inf], ps=(50, 99))
    assert p[99] == math.inf and p[50] == 2.0
    assert pr.percentiles([]) == {50: None, 90: None, 95: None, 99: None}
