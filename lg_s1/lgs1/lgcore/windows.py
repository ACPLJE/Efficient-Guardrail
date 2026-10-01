"""Window grid for Landmark Guard.

Definitions (fixed by the paper design, see the experiment report §6.1):
  * b            = half-window (default 128).  A window is 2b tokens long.
  * stride       = distance between window starts.  stride = b  -> overlapping (default),
                   stride = 2b -> disjoint.  The window LENGTH never changes with the stride.
                   (The 2026-09-10 review found the old `no_overlap` path shrank windows to b
                   tokens, which confounded the overlap ablation.  E0-1(e) guards against that.)
  * The grid starts at token 0.  The last window is shifted left so that it ends exactly at
    n_tokens (no token is left uncovered) unless the document is shorter than one window.

Guarantee (Prop. 1 precondition): with stride <= b, every contiguous span of at most b tokens is
wholly contained in at least one window.  `spans_fully_covered()` checks this and is used by the
tests and by the v2 data tokenizer (`window_ids_containing_needle`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Window = Tuple[int, int]  # [start, end) in token indices


@dataclass(frozen=True)
class WindowGrid:
    b: int = 128
    stride: Optional[int] = None  # None -> b (overlap)

    @property
    def length(self) -> int:
        return 2 * self.b

    @property
    def eff_stride(self) -> int:
        return self.b if self.stride is None else int(self.stride)


def make_windows(n_tokens: int, b: int = 128, stride: Optional[int] = None) -> List[Window]:
    """Return the list of [start, end) windows covering a document of n_tokens tokens.

    * window length is always 2b (except when n_tokens < 2b: one window [0, n_tokens))
    * starts are 0, stride, 2*stride, ... while start + 2b <= n_tokens
    * if the last window does not reach n_tokens, an extra window [n_tokens-2b, n_tokens) is added
    """
    if n_tokens <= 0:
        return []
    if b <= 0:
        raise ValueError("b must be positive")
    L = 2 * b
    s = b if stride is None else int(stride)
    if s <= 0:
        raise ValueError("stride must be positive")
    if n_tokens <= L:
        return [(0, n_tokens)]
    starts = list(range(0, n_tokens - L + 1, s))
    wins = [(st, st + L) for st in starts]
    if wins[-1][1] < n_tokens:
        wins.append((n_tokens - L, n_tokens))
    return wins


def windows_containing(span: Tuple[int, int], windows: Sequence[Window]) -> List[int]:
    """Indices of windows that wholly contain span=[s, e)."""
    s, e = span
    return [i for i, (ws, we) in enumerate(windows) if ws <= s and e <= we]


def spans_fully_covered(n_tokens: int, b: int, stride: Optional[int] = None, span_len: Optional[int] = None) -> bool:
    """True iff every span of length span_len (default b) is wholly inside some window."""
    L = span_len or b
    wins = make_windows(n_tokens, b, stride)
    for s in range(0, max(1, n_tokens - L + 1)):
        if not windows_containing((s, min(n_tokens, s + L)), wins):
            return False
    return True


def window_lengths(windows: Sequence[Window]) -> List[int]:
    return [e - s for s, e in windows]
