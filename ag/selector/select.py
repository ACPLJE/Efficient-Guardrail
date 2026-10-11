"""Trace selector (SPEC 4.5). The same functions run on Records (offline) and inside the
engine shim (live); there is no second copy of the formulas.

  chunk_block_mass   per-block mass of one chunk: mean over its tokens, mean over the chosen layers
  window_scores      window score = sum of its two blocks, ineligible windows = -inf
  select             top-k / all / oracle / random / mass_threshold
  union_select       top-k of each scorer, then the union (scorer_combine = union_topk)

Aggregate A = upper_half + uniform, aggregate C = tracked8 + uniform (G0 uses C, D-029).
Aggregate B (learned head weights) and rollout are S2 work and raise NotImplementedError.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ag.data import window_map as wm
from ag.data.serde import JsonMixin
from ag.data.types import ChunkInfo, LedgerRecord, RenderedPrompt
from ag.selector.layers import resolve_layers

LayerSel = tuple[int, ...] | Literal["upper_half", "last", "tracked8", "all"]


@dataclass(frozen=True)
class SelectorConfig(JsonMixin):
    rule: Literal["topk", "mass_threshold", "all", "oracle", "random"] = "topk"
    k: int = 8
    mass_tau: float | None = None
    layers: LayerSel = "tracked8"
    head_weighting: Literal["uniform", "learned"] = "uniform"
    weights_path: str | None = None
    rollout: bool = False
    first_token: Literal["include", "skip"] = "skip"
    ledger_sample_every: int = 1
    seed: int = 0

    def validate(self) -> None:
        if self.head_weighting == "learned" and self.first_token != "skip":
            raise ValueError("head_weighting=learned requires first_token=skip (SPEC 4.5)")
        if self.rule == "mass_threshold" and self.mass_tau is None:
            raise ValueError("rule=mass_threshold needs mass_tau")
        if self.ledger_sample_every < 1:
            raise ValueError("ledger_sample_every must be >= 1")


AGGREGATES: dict[str, dict] = {
    "A": {"layers": "upper_half", "head_weighting": "uniform"},
    "B": {"layers": "upper_half", "head_weighting": "learned"},
    "C": {"layers": "tracked8", "head_weighting": "uniform"},
}


def record_layer_index(rec: LedgerRecord, cfg: SelectorConfig, *, n_target_layers: int | None = None,
                       attn_kind: Sequence[str] | None = None) -> np.ndarray:
    """Indices into rec.layers for cfg.layers.

    Named specs are resolved over the target's full-attention layers. Without n_target_layers the
    recorded layers are taken as the full list, which is right when the Record was written with
    layers="all".
    """
    recorded = [int(x) for x in rec.layers]
    if isinstance(cfg.layers, (tuple, list)):
        want = tuple(int(x) for x in cfg.layers)
    elif n_target_layers is not None:
        want = resolve_layers(cfg.layers, n_target_layers, attn_kind)
    else:
        idx = resolve_layers(cfg.layers, len(recorded))
        want = tuple(recorded[i] for i in idx)
    missing = [l for l in want if l not in recorded]
    if missing:
        raise ValueError(f"{rec.request_id}: layers {missing} not in the Record (has {recorded})")
    return np.asarray([recorded.index(l) for l in want], dtype=np.int64)


def token_rows(rec: LedgerRecord, chunk: ChunkInfo, cfg: SelectorConfig) -> np.ndarray:
    """Output-token rows that enter the chunk mean (first-token policy and sampling applied)."""
    lo = chunk.tok_start
    if cfg.first_token == "skip":
        lo = max(lo, 1)
    elif rec.first_token_policy == "skip" and chunk.tok_start == 0:
        raise ValueError(f"{rec.request_id}: selector wants first_token=include but the Record skipped row 0")
    rows = np.arange(lo, min(chunk.tok_end, rec.n_tokens))
    if cfg.ledger_sample_every > 1:
        rows = rows[rows % cfg.ledger_sample_every == 0]
    return rows


def chunk_block_mass(rec: LedgerRecord, chunk: ChunkInfo, cfg: SelectorConfig, *,
                     n_target_layers: int | None = None, attn_kind: Sequence[str] | None = None) -> np.ndarray:
    """[NB] mean block mass of one chunk. Zeros when the chunk has no rows (e.g. a 1-token chunk 0
    with first_token=skip)."""
    if cfg.head_weighting == "learned":
        raise NotImplementedError("aggregate B (learned head weights) is S2 work")
    if cfg.rollout:
        raise NotImplementedError("rollout is S2 work")
    li = record_layer_index(rec, cfg, n_target_layers=n_target_layers, attn_kind=attn_kind)
    rows = token_rows(rec, chunk, cfg)
    if len(rows) == 0:
        return np.zeros(rec.n_blocks, dtype=np.float64)
    bm = np.asarray(rec.block_mass, dtype=np.float64)[rows][:, li, :]
    return bm.mean(axis=(0, 1))


def window_scores(block_mass: np.ndarray, rp: RenderedPrompt) -> np.ndarray:
    """[NW] window scores; ineligible windows are -inf (SPEC 1.2)."""
    return wm.window_scores_from_blocks(block_mass, wm.eligible_windows(rp.block_untrusted_array()))


def _order(scores: np.ndarray) -> np.ndarray:
    """Eligible window ids by descending score, ties broken by the lower window id."""
    s = np.asarray(scores, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(s))
    return finite[np.lexsort((finite, -s[finite]))]


def _rng_for(cfg: SelectorConfig, key: str) -> np.random.Generator:
    import hashlib
    h = int.from_bytes(hashlib.blake2b(f"{cfg.seed}:{key}".encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(h)


def select(scores: np.ndarray, cfg: SelectorConfig, *, oracle: Sequence[int] | None = None,
           key: str = "") -> tuple[int, ...]:
    """Windows to judge. Returned in rank order (topk, mass_threshold) or id order (others).

    mass_threshold: the shortest prefix of the ranking whose share of the eligible positive mass
    reaches mass_tau (S1 definition; the plan leaves it open, E6.1 compares it with top-k).
    """
    order = _order(scores)
    if cfg.rule == "topk":
        return tuple(int(w) for w in order[: cfg.k])
    if cfg.rule == "all":
        return tuple(sorted(int(w) for w in order))
    if cfg.rule == "oracle":
        if oracle is None:
            raise ValueError("rule=oracle needs the payload windows")
        return tuple(sorted(int(w) for w in oracle))
    if cfg.rule == "random":
        if len(order) <= cfg.k:
            return tuple(sorted(int(w) for w in order))
        pick = _rng_for(cfg, key).choice(np.sort(order), size=cfg.k, replace=False)
        return tuple(sorted(int(w) for w in pick))
    if cfg.rule == "mass_threshold":
        s = np.clip(np.asarray(scores, dtype=np.float64)[order], 0.0, None)
        total = s.sum()
        if total <= 0:
            return tuple(int(w) for w in order[:1])
        n = int(np.searchsorted(np.cumsum(s) / total, float(cfg.mass_tau), side="left")) + 1
        return tuple(int(w) for w in order[: min(n, len(order))])
    raise ValueError(f"unknown selector rule {cfg.rule!r}")


def union_select(per_scorer: Sequence[np.ndarray], cfg: SelectorConfig, *, key: str = "") -> tuple[int, ...]:
    out: set[int] = set()
    for s in per_scorer:
        out.update(select(s, cfg, key=key))
    return tuple(sorted(out))


def payload_rank(scores: np.ndarray, payload: Sequence[int]) -> int | None:
    """Best (1-based) rank of any payload window in the ranking; None if none is eligible."""
    if not payload:
        return None
    order = _order(scores)
    pos = {int(w): i + 1 for i, w in enumerate(order)}
    ranks = [pos[w] for w in payload if w in pos]
    return min(ranks) if ranks else None


def random_hit_prob(n_eligible: int, n_payload: int, m: int) -> float:
    """Chance that m random eligible windows include a payload window: 1 - C(NW_e-|P|, m)/C(NW_e, m)."""
    if n_payload <= 0 or n_eligible <= 0:
        return 0.0
    m = min(m, n_eligible)
    if n_eligible - n_payload < m:
        return 1.0
    return 1.0 - math.comb(n_eligible - n_payload, m) / math.comb(n_eligible, m)
