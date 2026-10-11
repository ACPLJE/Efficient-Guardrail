"""SPEC 3.4 invariants are caught on hand-built violations."""
from ag.data.types import Verdict
from ag.harness.checks import run_checks
from ag.harness.events import Event, EventType as E

FLAGS = {"uses_judge": True, "inj_path": "on"}


def ev(t, typ, rid="q", **kw):
    return Event("run", rid, typ, t, "virtual", **kw)


def good(rid="q", unsafe=False):
    v = Verdict("MJ", rid, 0, 1, "128x2", 0.9 if unsafe else 0.1, 0.5, "F5", unsafe, 30, 5)
    evs = [ev(0, E.ARRIVED, rid), ev(0, E.ADMITTED, rid), ev(1, E.PREFILL_START, rid), ev(5, E.PREFILL_DONE, rid),
           ev(20, E.CHUNK_GENERATED, rid, chunk_id=0, required_jobs=("j",)),
           ev(20, E.JOB_SUBMITTED, rid, job_id="j", deadline_ns=100), ev(30, E.JOB_DONE, rid, job_id="j", verdicts=(v,))]
    if unsafe:
        evs.append(ev(31, E.CHUNK_BLOCKED, rid, chunk_id=0, reason="inj_window"))
    else:
        evs += [ev(31, E.CHUNK_RELEASED, rid, chunk_id=0), ev(31, E.CHUNK_DELIVERED, rid, chunk_id=0)]
    evs.append(ev(40, E.FINISHED, rid))
    return evs


def failed(evs, flags=FLAGS):
    return {c.name for c in run_checks(evs, flags) if not c.ok}


def test_good_runs_pass():
    assert failed(good() + good("b", unsafe=True)) == set()


def test_release_before_verdict_caught():
    evs = [e for e in good() if e.type != E.JOB_DONE] + [ev(35, E.JOB_DONE, job_id="j")]
    assert "no_release_before_verdict" in failed(evs)
    assert "no_release_before_verdict" not in failed(evs, {**FLAGS, "release_before_verdict": True})


def test_unsafe_release_caught():
    v = Verdict("MJ", "q", 0, 1, "128x2", 0.9, 0.5, "F5", True, 30, 5)
    evs = [e if e.type != E.JOB_DONE else ev(30, E.JOB_DONE, job_id="j", verdicts=(v,)) for e in good()]
    assert "no_release_before_verdict" in failed(evs)


def test_missing_terminal_and_held_chunk():
    evs = [e for e in good() if e.type not in (E.FINISHED, E.CHUNK_RELEASED, E.CHUNK_DELIVERED)]
    f = failed(evs)
    assert {"invariants.one_terminal", "held_chunks_at_end"} <= f


def test_chunk_without_verdict_and_order():
    evs = good() + [ev(50, E.CHUNK_GENERATED, chunk_id=2), ev(51, E.CHUNK_RELEASED, chunk_id=2),
                    ev(51, E.CHUNK_DELIVERED, chunk_id=2)]
    f = failed(evs)
    assert {"chunk_without_verdict", "invariants.in_order_release"} <= f


def test_deadline_and_client_wait():
    evs = [e if e.type != E.JOB_SUBMITTED else ev(20, E.JOB_SUBMITTED, job_id="j", deadline_ns=25) for e in good()]
    assert "invariants.deadline_fail_closed" in failed(evs)
    late = [e if e.type != E.ADMITTED else ev(2_000_000, E.ADMITTED) for e in good()]
    assert "client_wait_bounded" in failed(late)


def test_prefill_after_request_blocked():
    evs = [ev(0, E.ARRIVED), ev(0, E.ADMITTED), ev(1, E.PREFILL_START), ev(2, E.REQUEST_BLOCKED, reason="x"),
           ev(3, E.FINISHED)]
    assert "invariants.lifecycle_order" in failed(evs)
