"""Run context and clocks (SPEC 1.6, 4.1)."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ag.harness.events import Event
    from ag.harness.lineage import AGLineage
    from ag.reference.record import RecordStore


class Clock(Protocol):
    kind: str

    def now_ns(self) -> int: ...

    def sleep_until(self, t_ns: int) -> None: ...


class MonotonicClock:
    kind = "monotonic"

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def sleep_until(self, t_ns: int) -> None:
        dt = t_ns - time.monotonic_ns()
        if dt > 0:
            time.sleep(dt / 1e9)


class VirtualClock:
    """Discrete-event clock for mocks and tests.

    Threads that take part register themselves. When every registered thread is blocked in
    sleep_until(), now jumps to the earliest wake time. A thread that is not registered (the
    single-thread case) moves now to max(now, t) and returns at once.
    """

    kind = "virtual"

    def __init__(self, start_ns: int = 0):
        self._now = start_ns
        self._cv = threading.Condition()
        self._registered: set[int] = set()
        self._waiting: dict[int, int] = {}

    def now_ns(self) -> int:
        with self._cv:
            return self._now

    def register(self) -> None:
        with self._cv:
            self._registered.add(threading.get_ident())

    def unregister(self) -> None:
        with self._cv:
            self._registered.discard(threading.get_ident())
            self._waiting.pop(threading.get_ident(), None)
            self._maybe_advance()

    def advance(self, dt_ns: int) -> None:
        with self._cv:
            self._now += dt_ns
            self._cv.notify_all()

    def sleep_until(self, t_ns: int) -> None:
        me = threading.get_ident()
        with self._cv:
            if me not in self._registered:
                if t_ns > self._now:
                    self._now = t_ns
                    self._cv.notify_all()
                return
            if t_ns <= self._now:
                return
            self._waiting[me] = t_ns
            self._maybe_advance()
            while self._now < t_ns:
                self._cv.wait()
            self._waiting.pop(me, None)

    def _maybe_advance(self) -> None:
        if self._registered and self._registered <= set(self._waiting):
            nxt = min(self._waiting.values())
            if nxt > self._now:
                self._now = nxt
            self._cv.notify_all()


def make_clock(kind: str) -> MonotonicClock | VirtualClock:
    if kind == "monotonic":
        return MonotonicClock()
    if kind == "virtual":
        return VirtualClock()
    raise ValueError(f"unknown clock {kind!r}")


@dataclass
class RunContext:
    run_id: str
    exp_id: str
    out_dir: Path
    clock: Clock
    seed: int
    lineage: "AGLineage"
    records: "RecordStore | None"
    emit: Callable[["Event"], None]
    log: logging.Logger
    attempt: int = 0
