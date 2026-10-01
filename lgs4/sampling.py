"""Window sampling for the LG head (and row sampling for dense-FT) from the prepped subsets.

A presence row (train/main, train/main16k) contributes, per epoch:
    needle windows (<= 3, the windows wholly containing the needle span)   label = row label (harmful 1 / twin 0)
    `near_neg` windows adjacent to the needle windows that do not contain it  label 0 (hard boundary negatives)
    `hard_per_row` mined hard negatives (S8 E6-6: the non-needle windows the base run's router ranks highest and its
                   head scores highest -- the windows that inflate the top-k max on long documents)   label 0
    `neg_per_row - hard` random windows that do not contain the needle       label 0
A native row (train/native: the needle text alone) contributes every window of its text with the row label
(native texts are short: 1-4 windows).  Per epoch `native_ratio * n_presence` native rows are drawn.
Units are shuffled with the seed, so seed = probe init + data order (plan §3).  `row_filter` (E6-7/8/9 recipes:
training length cap, needle fraction, background domains, twins on/off) restricts the presence rows before sampling.
"""
from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lgs1.lgcore.windows import make_windows, windows_containing
from lgs4.config import WINDOW_B
from lgs4.prep import PrepSet


@dataclass(frozen=True)
class Unit:
    set_idx: int          # index into the PrepSet list
    row: int
    start: int
    end: int
    label: int
    kind: str             # needle | near | hard | neg | native


# ------------------------------------------------------------------------------------------------
# mined hard negatives (lgs4.hardneg writes them; one npz per prepped set)
# ------------------------------------------------------------------------------------------------
class HardNegIndex:
    """Per prepped set: for every row the mined non-needle windows (start, end, head_prob, router_score), sorted by
    head probability (desc).  Loaded from <dir>/<split>.<subset>.npz; rows without a file / entry get no hard negatives."""

    FILES = ("row_ptr", "win_start", "win_end", "head_prob", "router_score")

    def __init__(self, sets: Dict[str, Dict[str, np.ndarray]], meta: Optional[Dict] = None):
        self.sets = sets
        self.meta = meta or {}

    @classmethod
    def load(cls, d: Path, set_names: Sequence[str]) -> "HardNegIndex":
        d = Path(d)
        sets = {}
        for name in set_names:
            f = d / f"{name}.npz"
            if f.exists():
                z = np.load(f)
                sets[name] = {k: z[k] for k in cls.FILES}
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        if not sets:
            raise SystemExit(f"{d}: no hard-negative files for {list(set_names)} (python -m lgs4.hardneg first)")
        return cls(sets, meta)

    def n_rows(self, set_name: str) -> int:
        s = self.sets.get(set_name)
        return int(len(s["row_ptr"]) - 1) if s is not None else 0

    def windows(self, set_name: str, row: int) -> List[Tuple[int, int, float, float]]:
        s = self.sets.get(set_name)
        if s is None or row + 1 >= len(s["row_ptr"]):
            return []
        a, b = int(s["row_ptr"][row]), int(s["row_ptr"][row + 1])
        out = [(int(s["win_start"][i]), int(s["win_end"][i]), float(s["head_prob"][i]), float(s["router_score"][i])) for i in range(a, b)]
        out.sort(key=lambda t: (-t[2], -t[3]))
        return out


def set_name(ps: PrepSet) -> str:
    return f"{ps.split}.{ps.subset}"


def row_units(ps: PrepSet, set_idx: int, row: int, rng: random.Random, neg_per_row: int, near_neg: int, b: int = WINDOW_B,
              hard: Optional[HardNegIndex] = None, hard_per_row: int = 0) -> List[Unit]:
    m = ps.meta[row]
    n = m["n_tok"]
    wins = make_windows(n, b)
    if not wins:
        return []
    span = m.get("needle_tok_span")
    if ps.subset == "native" or span is None:         # native row (needle text alone): every window carries the row label
        return [Unit(set_idx, row, s, e, m["label"], "native") for s, e in wins]
    nw = windows_containing((span[0], span[1]), wins)
    if not nw:                                        # needle longer than a window (never in main sets): windows overlapping it
        nw = [i for i, (s, e) in enumerate(wins) if s < span[1] and span[0] < e]
    nwset = set(nw)
    units = [Unit(set_idx, row, wins[i][0], wins[i][1], m["label"], "needle") for i in nw]
    near = []
    for i in nw:
        for j in (i - 2, i - 1, i + 1, i + 2):
            if 0 <= j < len(wins) and j not in nwset and j not in near:
                near.append(j)
    rng.shuffle(near)
    near_taken = near[:near_neg]
    units += [Unit(set_idx, row, wins[j][0], wins[j][1], 0, "near") for j in near_taken]
    taken = nwset | set(near_taken)
    n_hard = 0
    if hard is not None and hard_per_row > 0:
        by_span = {w: j for j, w in enumerate(wins)}
        for s, e, _p, _sc in hard.windows(set_name(ps), row):
            j = by_span.get((s, e))
            if j is None or j in taken:                # a mined window that became a needle / near window: skip
                continue
            units.append(Unit(set_idx, row, s, e, 0, "hard"))
            taken.add(j)
            n_hard += 1
            if n_hard >= hard_per_row:
                break
    others = [j for j in range(len(wins)) if j not in taken]
    rng.shuffle(others)
    units += [Unit(set_idx, row, wins[j][0], wins[j][1], 0, "neg") for j in others[:max(0, neg_per_row - n_hard)]]
    return units


# ------------------------------------------------------------------------------------------------
# presence-row filters (E6-7 / E6-8 / E6-9 recipes)
# ------------------------------------------------------------------------------------------------
def pair_keep(pair_key: str, frac: float, seed: int) -> bool:
    """Seeded, stable per-pair coin: both rows of a pair (harmful + twin) fall on the same side."""
    if frac >= 1.0:
        return True
    h = zlib.crc32(f"{seed}:{pair_key}".encode("utf-8")) / 2 ** 32
    return h < frac


def make_row_filter(max_L: Optional[int] = None, needle_frac: float = 1.0, domains: Optional[Sequence[str]] = None,
                    twin: bool = True, seed: int = 0) -> Optional[Callable[[Dict], bool]]:
    """Presence-row filter from the recipe knobs; None when nothing is restricted (the default recipe)."""
    doms = set(domains) if domains else None
    if max_L is None and needle_frac >= 1.0 and doms is None and twin:
        return None

    def keep(m: Dict) -> bool:
        if max_L is not None and (m.get("L_target") or 0) > max_L:
            return False
        if doms is not None and m.get("haystack_domain") not in doms:
            return False
        if not twin and int(m.get("label", 1)) == 0 and m.get("needle_tok_span") is not None:
            return False
        if needle_frac < 1.0 and not pair_keep(str(m.get("pair_id") or m.get("needle_id") or m.get("id")), needle_frac, seed):
            return False
        return True
    return keep


def epoch_units(presence: Sequence[PrepSet], native: Sequence[PrepSet], seed: int, epoch: int, neg_per_row: int, near_neg: int,
                native_ratio: float, limit_rows: Optional[int] = None, b: int = WINDOW_B,
                row_filter: Optional[Callable[[Dict], bool]] = None, hard: Optional[HardNegIndex] = None,
                hard_per_row: int = 0) -> List[Unit]:
    """All units of one epoch, shuffled.  `presence`/`native` PrepSets are indexed jointly (presence first)."""
    rng = random.Random(f"{seed}:{epoch}")
    units: List[Unit] = []
    prows = [(si, r) for si, ps in enumerate(presence) for r in range(len(ps)) if row_filter is None or row_filter(ps.meta[r])]
    rng.shuffle(prows)
    if limit_rows:
        prows = prows[:limit_rows]
    for si, r in prows:
        units += row_units(presence[si], si, r, rng, neg_per_row, near_neg, b, hard, hard_per_row)
    n_native = int(round(native_ratio * len(prows)))
    nrows = [(len(presence) + si, r) for si, ps in enumerate(native) for r in range(len(ps))]
    rng.shuffle(nrows)
    sets = list(presence) + list(native)
    for si, r in nrows[:n_native]:
        units += row_units(sets[si], si, r, rng, neg_per_row, near_neg, b)
    rng.shuffle(units)
    return units


def collate(units: Sequence[Unit], sets: Sequence[PrepSet], pad_id: int, b: int = WINDOW_B) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = 2 * b
    ids = torch.full((len(units), L), pad_id, dtype=torch.long)
    valid = torch.zeros((len(units), L), dtype=torch.long)
    y = torch.zeros(len(units), dtype=torch.float32)
    for i, u in enumerate(units):
        row = sets[u.set_idx].row_ids(u.row)[u.start:u.end]
        n = min(len(row), L)
        ids[i, :n] = torch.from_numpy(np.asarray(row[:n], dtype=np.int64))
        valid[i, :n] = 1
        y[i] = float(u.label)
    return ids, valid, y


def batches(units: Sequence[Unit], batch_size: int) -> Iterator[List[Unit]]:
    for i in range(0, len(units), batch_size):
        yield list(units[i:i + batch_size])


# ---- dense-FT rows -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RowUnit:
    set_idx: int
    row: int
    label: int
    n_tok: int


def epoch_rows(presence: Sequence[PrepSet], native: Sequence[PrepSet], seed: int, epoch: int, native_ratio: float,
               max_tokens: int, limit_rows: Optional[int] = None) -> List[RowUnit]:
    rng = random.Random(f"{seed}:{epoch}:dense")
    prows = [(si, r) for si, ps in enumerate(presence) for r in range(len(ps))]
    rng.shuffle(prows)
    if limit_rows:
        prows = prows[:limit_rows]
    sets = list(presence) + list(native)
    out = [RowUnit(si, r, presence[si].meta[r]["label"], min(presence[si].meta[r]["n_tok"], max_tokens)) for si, r in prows]
    n_native = int(round(native_ratio * len(prows)))
    nrows = [(len(presence) + si, r) for si, ps in enumerate(native) for r in range(len(ps))]
    rng.shuffle(nrows)
    out += [RowUnit(si, r, sets[si].meta[r]["label"], min(sets[si].meta[r]["n_tok"], max_tokens)) for si, r in nrows[:n_native]]
    rng.shuffle(out)
    return out


def token_budget_batches(rows: Sequence[RowUnit], micro_tokens: int) -> Iterator[List[RowUnit]]:
    """Greedy micro-batches: padded size (max_len x n) <= micro_tokens; a single row longer than the budget is alone."""
    cur: List[RowUnit] = []
    cur_max = 0
    for r in rows:
        L = r.n_tok
        if cur and max(cur_max, L) * (len(cur) + 1) > micro_tokens:
            yield cur
            cur, cur_max = [], 0
        cur.append(r)
        cur_max = max(cur_max, L)
    if cur:
        yield cur


def collate_rows(rows: Sequence[RowUnit], sets: Sequence[PrepSet], pad_id: int, max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = max(r.n_tok for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    valid = torch.zeros((len(rows), L), dtype=torch.long)
    y = torch.zeros(len(rows), dtype=torch.float32)
    for i, r in enumerate(rows):
        arr = sets[r.set_idx].row_ids(r.row)[:max_tokens]        # keep the head of over-long rows (train rows are <= 16k)
        n = min(len(arr), L)
        ids[i, :n] = torch.from_numpy(np.asarray(arr[:n], dtype=np.int64))
        valid[i, :n] = 1
        y[i] = float(r.label)
    return ids, valid, y
