"""Mooncake FAST '25 trace loader and W0-mooncake statistics (plan 5.2, 5.5, 12.2; SPEC 4.10, 10, 11;
F-11, A-12, D-020, D-022).

Row format (FAST25-release README [literature]; not checked against the real files from the machine
this was written on): one JSON object per line,

    {"timestamp": <ms since trace start, int>, "input_length": <int>, "output_length": <int>,
     "hash_ids": [<int>, ...]}

`hash_ids` are prompt block hashes, one per 512-token block, each hashed together with its prefix
(A-12), so len(hash_ids) should be ceil(input_length / 512). `trace_stats` measures how often that
holds instead of assuming it (F-11); W-MT relies on it.

The registry entry `mooncake` (ag/data/registry_ag.py) clones the repo to
$AG_SOURCES/mooncake/repo; the traces are under FAST25-release/traces/ (conversation_trace.jsonl,
toolagent_trace.jsonl [literature, file names unverified here]).
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = ("timestamp", "input_length", "output_length", "hash_ids")
BLOCK_TOKENS = 512                                   # A-12
INPUT_THRESHOLDS = (8192, 32768, 65536, 131072)      # plan 5.5: 8k-and-above sub-trace, 32k+, 128k
OUTPUT_THRESHOLDS = (3, 8, 32)                       # outputs within the first serving chunk (SPEC 1.4)
LENGTH_BINS = (("short", 0, 4096), ("8k", 4096, 16384), ("32k", 16384, 65536), ("128k", 65536, None))  # D-020
PERCENTILES = (50, 90, 95, 99)
BLOCK_SCAN = (256, 512, 1024)                        # hash_ids unit candidates reported next to block_tokens
MAX_MISMATCH_EXAMPLES = 10
WINDOW_S = 1200.0                                    # E4.4 replays 20-minute slices (D-022)
WINDOW_STEP_S = 60.0


def _as_int(v: Any, name: str) -> int:
    if isinstance(v, bool):
        raise ValueError(f"field {name!r} must be an integer, got {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise ValueError(f"field {name!r} must be an integer, got {v!r}")


@dataclass(frozen=True)
class TraceRow:
    """One trace request. JSON uses the FAST25 field names (`timestamp` is `timestamp_ms` here)."""
    timestamp_ms: int
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hash_ids, tuple):
            object.__setattr__(self, "hash_ids", tuple(self.hash_ids))

    def to_json(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp_ms, "input_length": self.input_length,
                "output_length": self.output_length, "hash_ids": list(self.hash_ids)}

    @classmethod
    def from_json(cls, d: Any) -> TraceRow:
        if not isinstance(d, Mapping):
            raise ValueError(f"row is not a JSON object: {type(d).__name__}")
        missing = [k for k in FIELDS if k not in d]
        if missing:
            raise ValueError(f"missing field(s) {missing}")
        il = _as_int(d["input_length"], "input_length")
        ol = _as_int(d["output_length"], "output_length")
        if il < 0 or ol < 0:
            raise ValueError(f"negative length: input_length={il} output_length={ol}")
        hs = d["hash_ids"]
        if not isinstance(hs, list):
            raise ValueError(f"field 'hash_ids' must be a list, got {type(hs).__name__}")
        return cls(_as_int(d["timestamp"], "timestamp"), il, ol, tuple(_as_int(h, "hash_ids") for h in hs))


def _open_text(path: Path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":                         # gzip, whatever the suffix
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_trace(path: str | os.PathLike) -> Iterator[TraceRow]:
    """Rows in file order; blank lines are skipped. A malformed row raises ValueError("<path>:<line>: ...")."""
    p = Path(path)
    with _open_text(p) as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if lineno == 1 and s.startswith("version https://git-lfs"):
                raise ValueError(f"{p}:1: Git LFS pointer, not trace data (run `git lfs pull` in the clone)")
            try:
                row = TraceRow.from_json(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: malformed JSON: {e.msg} (column {e.colno})") from e
            except ValueError as e:
                raise ValueError(f"{p}:{lineno}: {e}") from e
            yield row


def load_trace(path: str | os.PathLike) -> list[TraceRow]:
    """Read a .jsonl / .jsonl.txt / .jsonl.gz trace without reordering rows."""
    return list(iter_trace(path))


def dump_trace(rows: Sequence[TraceRow], path: str | os.PathLike) -> Path:
    """Write rows in the FAST25 format (gzip when the name ends in .gz, mtime 0 for stable bytes)."""
    p = Path(path)
    text = "".join(json.dumps(r.to_json()) + "\n" for r in rows)
    data = gzip.compress(text.encode("utf-8"), mtime=0) if p.name.endswith(".gz") else text.encode("utf-8")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return p


def expected_blocks(input_length: int, block_tokens: int = BLOCK_TOKENS) -> int:
    """ceil(input_length / block_tokens): the hash_ids length under A-12."""
    return -(-input_length // block_tokens)


def _dist(a: np.ndarray) -> dict[str, Any]:
    ps = np.percentile(a, PERCENTILES)
    total = int(a.sum())
    return {"min": int(a.min()), **{f"p{p}": float(v) for p, v in zip(PERCENTILES, ps)},
            "max": int(a.max()), "mean": total / len(a), "sum": total}


def _per_s(x: float, duration_s: float) -> float | None:
    return x / duration_s if duration_s > 0 else None


def _block_check(inp: np.ndarray, nh: np.ndarray, block_tokens: int) -> dict[str, Any]:
    n = len(inp)
    match = nh == -(-inp // block_tokens)
    bad = np.flatnonzero(~match)
    return {
        "block_tokens": block_tokens, "n_match": int(match.sum()), "n_mismatch": len(bad),
        "frac_match": int(match.sum()) / n,
        "frac_match_floor": int((nh == inp // block_tokens).sum()) / n,
        "mismatch_rows": [{"row": int(i), "input_length": int(inp[i]), "n_hash_ids": int(nh[i]),
                           "expected": expected_blocks(int(inp[i]), block_tokens)}
                          for i in bad[:MAX_MISMATCH_EXAMPLES]],
        "scan": {str(b): int((nh == -(-inp // b)).sum()) / n for b in sorted(set(BLOCK_SCAN) | {block_tokens})},
    }


def _prefix_reuse(rows: Sequence[TraceRow]) -> dict[str, Any]:
    """Reuse against EARLIER rows in file order (ids repeated inside one row do not count)."""
    seen: set[int] = set()
    n_first = n_seen = n_ids = 0
    fracs: list[float] = []
    for r in rows:
        h = r.hash_ids
        if h:
            n_first += h[0] in seen
            k = sum(1 for x in h if x in seen)
            fracs.append(k / len(h))
            n_seen += k
            n_ids += len(h)
        seen.update(h)
    return {
        "n_rows_with_hash_ids": len(fracs), "n_first_hash_seen": n_first, "first_hash_seen_frac": n_first / len(rows),
        "n_unique_hash_ids": len(seen), "n_hash_ids": n_ids,
        "mean_seen_frac": math.fsum(fracs) / len(fracs) if fracs else None,
        "seen_hash_frac_total": n_seen / n_ids if n_ids else None,
    }


def _windows(ts: np.ndarray, inp: np.ndarray, window_s: float, step_s: float) -> dict[str, Any]:
    """Arrival rate over half-open windows [start, start + window_s) stepped by step_s from the first
    arrival, kept while the window ends at or before the last arrival (D-022 slice choice)."""
    if window_s <= 0 or step_s <= 0:
        raise ValueError(f"window_s and step_s must be > 0, got {window_s}, {step_s}")
    order = np.argsort(ts, kind="stable")
    t = ts[order]
    cum = np.concatenate([[0], np.cumsum(inp[order])])
    w_ms, st_ms = window_s * 1000.0, step_s * 1000.0
    span = int(t[-1] - t[0])
    out: dict[str, Any] = {"window_s": window_s, "step_s": step_s}
    if span < w_ms:
        return {**out, "n_windows": 0, "rate_req_s": None, "input_tok_s": None,
                "peak_rate_start_s": None, "peak_input_start_s": None}
    k = int((span - w_ms) // st_ms) + 1
    starts = t[0] + np.arange(k) * st_ms
    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, starts + w_ms, side="left")
    rate = (hi - lo) / window_s
    tok = (cum[hi] - cum[lo]) / window_s

    def mm(a: np.ndarray) -> dict[str, float]:
        return {"min": float(a.min()), "p50": float(np.percentile(a, 50)), "max": float(a.max())}

    return {**out, "n_windows": k, "rate_req_s": mm(rate), "input_tok_s": mm(tok),
            "peak_rate_start_s": float((starts[int(np.argmax(rate))] - t[0]) / 1000.0),
            "peak_input_start_s": float((starts[int(np.argmax(tok))] - t[0]) / 1000.0)}


def trace_stats(rows: Sequence[TraceRow], *, block_tokens: int = BLOCK_TOKENS, window_s: float = WINDOW_S,
                window_step_s: float = WINDOW_STEP_S) -> dict[str, Any]:
    """W0-mooncake statistics of one trace (SPEC 11). Deterministic; rows are taken in file order.

    rate_req_s = n / duration_s with duration_s = (last - first timestamp) / 1000. Percentiles are numpy
    linear. Fractions use n as the denominator except mean_seen_frac (rows with at least one hash id)
    and seen_hash_frac_total (all hash ids).
    """
    if not rows:
        raise ValueError("trace_stats: empty trace")
    if block_tokens <= 0:
        raise ValueError(f"block_tokens must be > 0, got {block_tokens}")
    n = len(rows)
    ts = np.array([r.timestamp_ms for r in rows], dtype=np.int64)
    inp = np.array([r.input_length for r in rows], dtype=np.int64)
    out = np.array([r.output_length for r in rows], dtype=np.int64)
    nh = np.array([len(r.hash_ids) for r in rows], dtype=np.int64)
    t0, t1 = int(ts.min()), int(ts.max())
    duration_s = (t1 - t0) / 1000.0
    n_dec = int((np.diff(ts) < 0).sum())

    def count(mask: np.ndarray) -> dict[str, Any]:
        c = int(mask.sum())
        return {"count": c, "frac": c / n}

    return {
        "n": n, "t_first_ms": t0, "t_last_ms": t1, "duration_s": duration_s,
        "timestamps_nondecreasing": n_dec == 0, "n_timestamp_decreases": n_dec,
        "rate_req_s": _per_s(n, duration_s),
        "input_tok_s": _per_s(int(inp.sum()), duration_s),
        "output_tok_s": _per_s(int(out.sum()), duration_s),
        "input_length": _dist(inp),
        "output_length": _dist(out),
        "input_ge": {str(x): {**count(inp >= x), "rate_req_s": _per_s(int((inp >= x).sum()), duration_s)}
                     for x in INPUT_THRESHOLDS},
        "output_le": {str(x): count(out <= x) for x in OUTPUT_THRESHOLDS},
        "length_bins": {name: {"lo": lo, "hi": hi, **count((inp >= lo) & ((inp < hi) if hi is not None else True))}
                        for name, lo, hi in LENGTH_BINS},
        "hash_ids_block_check": _block_check(inp, nh, block_tokens),
        "prefix_reuse": _prefix_reuse(rows),
        "windows": _windows(ts, inp, window_s, window_step_s),
    }
