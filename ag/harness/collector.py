"""The single event collector (SPEC 3.5).

A thread-safe queue and one writer thread. Events from System.run and from RunContext.emit
(judge pool threads, engine spool readers) all go through here. Only the collector assigns seq,
counted from 0 per (run_id, request_id) in arrival order. events.jsonl is written in arrival order.
"""
from __future__ import annotations

import dataclasses
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from ag.harness.events import Event, event_row
from ag.harness.writer import JsonlWriter

_STOP = object()


class Collector:
    def __init__(self, run_id: str, events_path: str | Path, *,
                 listeners: tuple[Callable[[Event], None], ...] = ()):
        self.run_id = run_id
        self._q: queue.Queue = queue.Queue()
        self._writer = JsonlWriter(events_path, run_id)
        self._seq: dict[str, int] = {}
        self._listeners = list(listeners)
        self.events: list[Event] = []
        self.errors: list[BaseException] = []
        self._drained = threading.Condition()
        self._pending = 0
        self._thread = threading.Thread(target=self._loop, name="ag-collector", daemon=True)
        self._thread.start()

    def add_listener(self, fn: Callable[[Event], None]) -> None:
        self._listeners.append(fn)

    def emit(self, e: Event) -> None:
        if e.run_id != self.run_id:
            raise ValueError(f"event run_id {e.run_id!r} != collector run_id {self.run_id!r}")
        with self._drained:
            self._pending += 1
        self._q.put(e)

    def _loop(self) -> None:
        while True:
            e = self._q.get()
            if e is _STOP:
                return
            try:
                seq = self._seq.get(e.request_id, 0)
                self._seq[e.request_id] = seq + 1
                e = dataclasses.replace(e, seq=seq)
                self._writer.write(event_row(e))
                self.events.append(e)
                for fn in self._listeners:
                    fn(e)
            except BaseException as exc:  # surfaced by flush()/close()
                self.errors.append(exc)
            finally:
                with self._drained:
                    self._pending -= 1
                    self._drained.notify_all()

    def flush(self) -> None:
        """Block until every emitted event is written and listeners have run."""
        with self._drained:
            while self._pending:
                self._drained.wait()
        if self.errors:
            raise RuntimeError(f"collector failed: {self.errors[0]!r}") from self.errors[0]

    def close(self) -> None:
        self.flush()
        self._q.put(_STOP)
        self._thread.join()
        self._writer.close()
