"""Data model (SPEC 2). All classes are frozen dataclasses with to_json / from_json (SPEC 2.0)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ag.data.serde import JsonMixin

Role = Literal["system", "user", "assistant", "tool", "document"]
Threat = Literal["I", "H_context", "H_user", "mixed"]
Split = Literal["calib", "test", "pilot"]


# --------------------------------------------------------------------------------------------
# 2.1 requests (target-independent, text level)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment(JsonMixin):
    role: Role
    text: str
    trusted: bool
    payload_char_spans: tuple[tuple[int, int], ...] = ()
    harm_char_spans: tuple[tuple[int, int], ...] = ()
    tool_name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class AttackMeta(JsonMixin):
    threat: Threat = "I"
    attack_type: str = ""
    position: float | None = None
    goal: str = ""
    template_id: str = ""
    n_sites: int = 1


@dataclass(frozen=True)
class Request(JsonMixin):
    request_id: str
    pair_key: str
    episode_id: str
    turn: int
    source: str
    split: Split
    length_class: str
    background_tier: Literal["low", "mid", "high", "follow"] | None
    task: str
    segments: tuple[Segment, ...]
    sink: Literal["stream", "tool"]
    max_new_tokens: int
    ignore_eos: bool
    attack: AttackMeta | None
    task_meta: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[Mapping[str, Any], ...] = ()
    arrival_ns: int | None = None
    target_len_tokens: int | None = None
    data_manifest_sha: str = ""

    @property
    def is_attack(self) -> bool:
        return self.attack is not None


def make_request_id(source: str, split: str, length_class: str, idx: int, attack: bool, turn: int = 0) -> str:
    """SPEC 1.5: <source>/<split>/<length_class>/<idx:06d>/<a|b>/t<turn:02d>."""
    return f"{source}/{split}/{length_class}/{idx:06d}/{'a' if attack else 'b'}/t{turn:02d}"


def pair_key_of(request_id: str) -> str:
    return request_id.rsplit("/", 2)[0]


def episode_id_of(request_id: str) -> str:
    return request_id.rsplit("/", 1)[0]


# --------------------------------------------------------------------------------------------
# 2.2 rendered prompt (target-specific, token level)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderedPrompt(JsonMixin):
    request_id: str
    target_id: str
    prompt_ids: tuple[int, ...]
    untrusted_mask: bytes
    payload_spans: tuple[tuple[int, int], ...]
    harm_spans: tuple[tuple[int, int], ...]
    split_tokens: int
    n_blocks: int
    block_untrusted: bytes
    payload_windows: tuple[int, ...]
    payload_windows_per_site: tuple[tuple[int, ...], ...]
    payload_rule: Literal["none", "half", "max_overlap", "per_site"]
    eligible_windows: int

    @property
    def n_tokens(self) -> int:
        return len(self.prompt_ids)

    def untrusted_bool(self) -> np.ndarray:
        bits = np.unpackbits(np.frombuffer(self.untrusted_mask, dtype=np.uint8), bitorder="little")
        return bits[: self.n_tokens].astype(bool)

    def block_untrusted_array(self) -> np.ndarray:
        return np.frombuffer(self.block_untrusted, dtype=np.uint8).astype(bool)


# --------------------------------------------------------------------------------------------
# 2.3 generation and verdicts
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkInfo(JsonMixin):
    chunk_id: int
    tok_start: int
    tok_end: int
    is_tool_call: bool = False

    @property
    def n_tokens(self) -> int:
        return self.tok_end - self.tok_start


@dataclass(frozen=True)
class Generation(JsonMixin):
    request_id: str
    output_ids: tuple[int, ...]
    text_generated: str
    text_delivered: str
    tool_calls_delivered: tuple[Mapping[str, Any], ...]
    chunks: tuple[ChunkInfo, ...]
    blocked: bool
    block_reason: str | None
    error: str | None = None


@dataclass(frozen=True)
class Verdict(JsonMixin):
    judge_id: str
    request_id: str
    chunk_id: int | None
    window_id: int | None
    window_spec: str
    score: float
    threshold: float
    threshold_id: str
    unsafe: bool                 # score > threshold (strict, lgs1/metrics convention)
    decided_ns: int
    latency_ns: int
    cached: bool = False


def is_unsafe(score: float, threshold: float) -> bool:
    """The single decision rule (SPEC 2.3): strictly greater than the threshold."""
    return score > threshold


# --------------------------------------------------------------------------------------------
# 2.4 outcome rows
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Outcome(JsonMixin):
    run_id: str
    unit: Literal["request", "episode"]
    unit_id: str
    variant_id: str
    variant: Mapping[str, Any]
    threshold_id: str
    tau: float | None
    system_id: str
    target_id: str
    judge_id: str | None
    labels: Mapping[str, bool]
    is_attack: bool
    attack_success: bool | None
    attack_success_b0: bool | None
    s_source_run_id: str | None
    utility: bool | None
    blocked: bool
    block_reason: str | None
    block_reasons: tuple[str, ...]
    false_block: bool
    error_kind: str | None
    judged_windows: int
    judged_windows_uncached: int
    judge_calls: int
    judged_windows_spec: str
    selected: Mapping[int, tuple[int, ...]]
    payload_windows: tuple[int, ...]
    payload_windows_per_site: tuple[tuple[int, ...], ...]
    payload_rank: Mapping[str, int | None]
    hit: Mapping[str, Mapping[int, bool]]
    hit_live: Mapping[int, bool]
    doc_scores: Mapping[str, float]
    harm_score: float | None
    cost: Mapping[str, float]
    per_call: tuple[Mapping[str, Any], ...] = ()
    success_judge: str = ""


# --------------------------------------------------------------------------------------------
# 2.7 auxiliary types
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowSpec(JsonMixin):
    block_tokens: int = 128
    span_blocks: int = 2

    @staticmethod
    def parse(s: str) -> "WindowSpec":
        a, b = s.lower().split("x")
        return WindowSpec(int(a), int(b))

    def __str__(self) -> str:
        return f"{self.block_tokens}x{self.span_blocks}"


@dataclass(frozen=True)
class ChunkingConfig(JsonMixin):
    first: int = 8
    next: int = 32
    sentence: bool = False


@dataclass(frozen=True)
class SamplingConfig(JsonMixin):
    temperature: float = 0.0
    max_new_tokens: int = 256
    ignore_eos: bool = False
    top_logprobs: int = 0
    seed: int = 0


@dataclass(frozen=True, eq=False)
class LedgerRecord:
    """Record v1 (SPEC 5.2). Arrays are numpy; metadata mirrors the npz scalars.
    Not JSON-serialisable on purpose: Records live in npz files (ag/reference/record.py)."""

    request_id: str
    target_id: str
    producer: str
    schema_version: int
    record_spec_hash: str
    precision: str
    first_token_policy: Literal["include", "skip"]
    prompt_len: int
    split_tokens: int
    chunk_defs: tuple[str, ...]
    output_ids: np.ndarray                 # int32 [T]
    t_ns: np.ndarray                       # int64 [T]
    layers: np.ndarray                     # int16 [L]
    block_mass: np.ndarray                 # [T, L, NB]
    block_untrusted: np.ndarray            # uint8 [NB]
    gen_mass: np.ndarray | None = None     # [T, L, T]
    chunk_head_mass: np.ndarray | None = None   # [C, L, H, NB]
    token_head_mass: np.ndarray | None = None   # [T, L, H, NB]
    sink_mass: np.ndarray | None = None    # [T, L]

    @property
    def n_tokens(self) -> int:
        return int(self.output_ids.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(self.block_mass.shape[-1])
