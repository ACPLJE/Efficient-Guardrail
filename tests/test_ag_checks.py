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


def test_late_verdict_blocked_with_wrong_reason():
    # verdict after the deadline, chunk blocked at verdict time as inj_window: must be deadline_miss at 100
    v = Verdict("MJ", "q", 0, 1, "128x2", 0.9, 0.5, "F5", True, 500, 5)
    evs = [ev(0, E.ARRIVED), ev(0, E.ADMITTED), ev(1, E.PREFILL_START), ev(5, E.PREFILL_DONE),
           ev(20, E.CHUNK_GENERATED, chunk_id=0, required_jobs=("j",)),
           ev(20, E.JOB_SUBMITTED, job_id="j", deadline_ns=100), ev(500, E.JOB_DONE, job_id="j", verdicts=(v,)),
           ev(501, E.CHUNK_BLOCKED, chunk_id=0, reason="inj_window"), ev(510, E.FINISHED)]
    assert "invariants.deadline_fail_closed" in failed(evs)
    ok = [e if e.type != E.CHUNK_BLOCKED else ev(100, E.CHUNK_BLOCKED, chunk_id=0, reason="deadline_miss")
          for e in evs]
    assert "invariants.deadline_fail_closed" not in failed(ok)


def test_deadline_slack_and_early_block():
    def chunk_events(block_t, reason, jobs=(("j", 100, 500, True),), cached_ns=None):
        evs = [ev(0, E.ARRIVED), ev(0, E.ADMITTED), ev(1, E.PREFILL_START), ev(5, E.PREFILL_DONE),
               ev(20, E.CHUNK_GENERATED, chunk_id=0, required_jobs=tuple(j for j, *_ in jobs),
                  cached_verdict_ns=cached_ns)]
        for jid, dl, done, unsafe in jobs:
            v = Verdict("MJ", "q", 0, 1, "128x2", 0.9 if unsafe else 0.1, 0.5, "F5", unsafe, done, 5)
            evs += [ev(20, E.JOB_SUBMITTED, job_id=jid, deadline_ns=dl), ev(done, E.JOB_DONE, job_id=jid, verdicts=(v,))]
        end = max(done for *_, done, _ in jobs) + 10
        return evs + [ev(block_t, E.CHUNK_BLOCKED, chunk_id=0, reason=reason), ev(end, E.FINISHED)]
    dl = 10_000_000
    late = (("j", dl, dl + 5_000_000, True),)
    assert "invariants.deadline_fail_closed" in failed(chunk_events(dl + 2_000_000, "deadline_miss", late))
    assert "invariants.deadline_fail_closed" not in failed(chunk_events(dl + 500_000, "deadline_miss", late))
    # blocked as unsafe before the deadline while a sibling job runs late: correct, not a violation
    two = (("j1", 100, 30, True), ("j2", 100, 500, False))
    assert "invariants.deadline_fail_closed" not in failed(chunk_events(31, "inj_window", two))
    assert "invariants.deadline_fail_closed" not in failed(chunk_events(21, "inj_window", (("j", 100, 500, False),),
                                                                       cached_ns=10))


def test_one_decision_per_chunk():
    evs = good() + [ev(32, E.CHUNK_BLOCKED, chunk_id=0, reason="inj_window")]
    assert "invariants.one_decision_per_chunk" in failed(evs)


def test_prefill_after_request_blocked():
    evs = [ev(0, E.ARRIVED), ev(0, E.ADMITTED), ev(1, E.PREFILL_START), ev(2, E.REQUEST_BLOCKED, reason="x"),
           ev(3, E.FINISHED)]
    assert "invariants.lifecycle_order" in failed(evs)
