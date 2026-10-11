"""Layer selection for the ledger (SPEC 4.5, D-031, D-037).

L is the list of layers whose attention kind is "full" (sliding layers are dropped, D-037).
  upper_half = the back half of L
  tracked8   = 8 evenly spaced layers of upper_half: U[round(i * (|U| - 1) / 7)], i = 0..7
  last       = the last layer of L
"""
from __future__ import annotations

from collections.abc import Sequence


def full_layers(n_layers: int, attn_kind: Sequence[str] | None = None) -> list[int]:
    return [l for l in range(n_layers) if attn_kind is None or attn_kind[l] == "full"]


def resolve_layers(spec, n_layers: int, attn_kind: Sequence[str] | None = None) -> tuple[int, ...]:
    """Target layer ids for a layer spec (tuple of ids, "all", "upper_half", "tracked8", "last")."""
    if isinstance(spec, (tuple, list)):
        return tuple(int(x) for x in spec)
    full = full_layers(n_layers, attn_kind)
    if spec == "all":
        return tuple(range(n_layers))
    upper = full[len(full) // 2:]
    if spec == "upper_half":
        return tuple(upper)
    if spec == "last":
        return (full[-1],)
    if spec == "tracked8":
        if len(upper) <= 8:
            return tuple(upper)
        return tuple(upper[round(i * (len(upper) - 1) / 7)] for i in range(8))
    raise ValueError(f"unknown layer spec {spec!r}")
