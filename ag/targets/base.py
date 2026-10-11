"""Target interface (SPEC 4.3)."""
from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ag.data.serde import JsonMixin
from ag.data.types import LedgerRecord, RenderedPrompt, Request

LayerSpec = tuple[int, ...] | Literal["all", "upper_half", "tracked8"]


@dataclass(frozen=True)
class RecordSpec(JsonMixin):
    layers: LayerSpec = "all"
    per_chunk_heads: bool = True
    per_token_heads: bool = False
    gen_mass: bool = True
    first_token: bool = True
    precision: Literal["fp16", "fp32"] = "fp16"
    chunk_defs: tuple[str, ...] = ("first8", "first32", "all")
    split_tokens: int = 128


@dataclass(frozen=True, eq=False)
class TargetOutput:
    output_ids: tuple[int, ...]
    text: str
    tool_calls: tuple[Mapping[str, Any], ...]
    t_prefill_done_ns: int
    t_token_ns: tuple[int, ...]
    peak_mem_bytes: int
    record: LedgerRecord | None
    logprobs: tuple[float, ...] | None = None


class Target(abc.ABC):
    target_id: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    max_context: int
    layer_attn_kind: tuple[str, ...]

    @abc.abstractmethod
    def render(self, req: Request) -> RenderedPrompt: ...

    @abc.abstractmethod
    def generate(self, rp: RenderedPrompt, max_new_tokens: int, *, record: RecordSpec | None,
                 forced_output: Sequence[int] | None = None,
                 drop_tokens: Sequence[tuple[int, int]] = ()) -> TargetOutput: ...

    def hidden_states(self, token_ids: Sequence[int], layers: Sequence[int],
                      positions: Literal["last", "mean"]) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> Mapping[str, Any]:
        return {"target_id": self.target_id}
