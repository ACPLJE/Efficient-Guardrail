"""Deterministic mock judge MJ (SPEC 4.6).

Windows containing a payload marker word score in [0.9, 1.0). Other windows score in [0, 0.3),
except a configurable share (false_positive_rate) that scores in [0.6, 0.9) so document-level
FPR grows with the number of judged windows (E1.2 shape). Latency = base + per_window * |batch|.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ag.judges.base import Judge, JudgeInput
from ag.targets.mock import stable_hash

PAYLOAD_MARKERS = frozenset({"PWNED", "IGNORE"})


class MockJudge(Judge):
    def __init__(self, judge_id: str = "MJ", *, false_positive_rate: float = 0.01, base_ms: float = 5.0,
                 per_window_ms: float = 2.0, needs_task: bool = False, needs_chunk: bool = False, seed: int = 0):
        self.judge_id = judge_id
        self.false_positive_rate = false_positive_rate
        self.base_ns = int(base_ms * 1e6)
        self.per_window_ns = int(per_window_ms * 1e6)
        self.needs_task = needs_task
        self.needs_chunk = needs_chunk
        self.seed = seed

    def _one(self, x: JudgeInput) -> float:
        h = stable_hash(f"{self.seed}:{self.judge_id}:{','.join(map(str, x.window_token_ids))}", bits=64)
        u = (h & 0xFFFFFFFF) / 2**32          # score jitter
        v = (h >> 32) / 2**32                 # false-positive draw, independent of u
        if PAYLOAD_MARKERS & set(x.window_text.split()):
            return 0.9 + 0.1 * u
        if v < self.false_positive_rate:
            return 0.6 + 0.3 * u
        return 0.3 * u

    def score(self, batch: Sequence[JudgeInput]) -> list[float]:
        return [self._one(x) for x in batch]

    def latency_ns(self, batch: Sequence[JudgeInput]) -> int:
        return self.base_ns + self.per_window_ns * len(batch)

    def info(self) -> Mapping[str, Any]:
        return {"judge_id": self.judge_id, "kind": "mock", "false_positive_rate": self.false_positive_rate}
