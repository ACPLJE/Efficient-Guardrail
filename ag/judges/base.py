"""Judge interface and verdict cache (SPEC 4.6). The judge pool and scheduler are S3 work."""
from __future__ import annotations

import abc
import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ag.data.serde import JsonMixin
from ag.data.types import Verdict, WindowSpec


@dataclass(frozen=True)
class JudgeInput(JsonMixin):
    request_id: str
    chunk_id: int | None
    window_id: int | None
    window_spec: str
    task: str
    window_text: str
    window_token_ids: tuple[int, ...]
    chunk_text: str | None
    chunk_token_ids: tuple[int, ...] | None
    threshold_id: str


class Judge(abc.ABC):
    judge_id: str
    needs_task: bool = False
    needs_chunk: bool = False
    native_window: WindowSpec | None = None

    @abc.abstractmethod
    def score(self, batch: Sequence[JudgeInput]) -> list[float]:
        """Higher = more unsafe."""

    def latency_ns(self, batch: Sequence[JudgeInput]) -> int:
        """Modelled latency for virtual-clock runs; real judges are timed by the pool."""
        raise NotImplementedError

    def info(self) -> Mapping[str, Any]:
        return {"judge_id": self.judge_id}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cache_key(judge: Judge, x: JudgeInput) -> tuple:
    """(judge_id, sha256(window ids), sha256(task) if needs_task, sha256(chunk ids) if needs_chunk, threshold_id)."""
    ids = ",".join(map(str, x.window_token_ids)).encode()
    task = _sha(x.task.encode()) if judge.needs_task else None
    chunk = _sha(",".join(map(str, x.chunk_token_ids or ())).encode()) if judge.needs_chunk else None
    return (judge.judge_id, _sha(ids), task, chunk, x.threshold_id)


class VerdictCache:
    """Policy off | per_request | global (D-011). Hits create no job (SPEC 4.6)."""

    def __init__(self, policy: Literal["off", "per_request", "global"] = "per_request"):
        self.policy = policy
        self._d: dict[tuple, Verdict] = {}
        self._lock = threading.Lock()

    def _k(self, request_id: str, key: tuple) -> tuple:
        return (request_id, *key) if self.policy == "per_request" else key

    def get(self, request_id: str, key: tuple) -> Verdict | None:
        if self.policy == "off":
            return None
        with self._lock:
            return self._d.get(self._k(request_id, key))

    def put(self, request_id: str, key: tuple, v: Verdict) -> None:
        if self.policy == "off":
            return
        with self._lock:
            self._d[self._k(request_id, key)] = v

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
