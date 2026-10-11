"""SPEC 3: 15 event types, sort key, collector seq per request in arrival order."""
import threading

from ag.harness.collector import Collector
from ag.harness.events import Event, EventType, by_request, event_from_row, sort_key
from ag.harness.writer import read_rows


def test_fifteen_types():
    assert len(EventType) == 15 and EventType.ERROR == 15


def test_sort_key_order():
    a = Event("r", "q", EventType.CHUNK_RELEASED, 5, "virtual", chunk_id=1)
    b = Event("r", "q", EventType.CHUNK_GENERATED, 5, "virtual", chunk_id=1)
    c = Event("r", "q", EventType.TOKENS, 4, "virtual")
    assert sorted([a, b, c], key=sort_key) == [c, b, a]


def test_collector_assigns_seq(tmp_path):
    col = Collector("run", tmp_path / "events.jsonl")

    def worker(rid):
        for t in range(50):
            col.emit(Event("run", rid, EventType.TOKENS, t, "virtual", tok_range=(t, t + 1)))

    ths = [threading.Thread(target=worker, args=(f"req{i}",)) for i in range(4)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    col.close()
    rows = read_rows(tmp_path / "events.jsonl")
    assert len(rows) == 200 and all(r["run_id"] == "run" and r["schema_version"] == 1 for r in rows)
    per = by_request(event_from_row(r) for r in rows)
    for rid, evs in per.items():
        assert sorted(e.seq for e in evs) == list(range(50))
        assert [e.tok_range[0] for e in sorted(evs, key=lambda e: e.seq)] == list(range(50))


def test_collector_rejects_foreign_run(tmp_path):
    import pytest
    col = Collector("run", tmp_path / "e.jsonl")
    with pytest.raises(ValueError):
        col.emit(Event("other", "q", EventType.ARRIVED, 0, "virtual"))
    col.close()


def test_latest_attempt_only():
    evs = [Event("r", "q", EventType.ARRIVED, 0, "virtual", attempt=0),
           Event("r", "q", EventType.ARRIVED, 9, "virtual", attempt=1)]
    assert [e.t_ns for e in by_request(evs)["q"]] == [9]
