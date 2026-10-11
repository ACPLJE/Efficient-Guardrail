"""Result writers (SPEC 7.1): atomic JSON/text files and locked line-append JSONL.

Every JSONL row carries run_id and schema_version.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ag import SCHEMA_VERSION
from ag.data.serde import to_jsonable


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def drop_partial_tail(path: Path) -> int:
    """Cut a partial last line (and a trailing NUL block) left by a crash mid-write, so appended rows
    start on a fresh line. iter_rows already ignores such a tail; the request it belonged to is
    re-run on resume. Returns the number of bytes removed."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "r+b") as f:
        data = f.read()
        keep = data.rstrip(b"\x00")
        if keep.endswith(b"\n"):
            cut = len(keep)
        else:
            start = keep.rfind(b"\n") + 1
            try:                                # a complete row that only lost its newline is kept
                json.loads(keep[start:])
                f.truncate(len(keep))
                f.seek(len(keep))
                f.write(b"\n")
                return 0
            except ValueError:
                cut = start
        if cut != len(data):
            f.truncate(cut)
        return len(data) - cut


class JsonlWriter:
    """Append-only JSONL with one lock; every line is flushed."""

    def __init__(self, path: str | Path, run_id: str, *, repair_tail: bool = True):
        """repair_tail: drop a crash-left partial last line first. Only for files that one run owns;
        shared files (raw/index.jsonl) that other jobs may be appending to pass False."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if repair_tail:
            drop_partial_tail(self.path)
        self.run_id = run_id
        self._lock = threading.Lock()
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        d = dict(to_jsonable(dict(row)))
        d.setdefault("run_id", self.run_id)
        d.setdefault("schema_version", SCHEMA_VERSION)
        line = json.dumps(d, sort_keys=True, allow_nan=False) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_rows(path: str | Path, run_id: str, rows) -> None:
    """Write a whole derived JSONL file atomically (score_outcomes output)."""
    lines = []
    for r in rows:
        d = dict(to_jsonable(dict(r)))
        d.setdefault("run_id", run_id)
        d.setdefault("schema_version", SCHEMA_VERSION)
        lines.append(json.dumps(d, sort_keys=True, allow_nan=False))
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_rows(path: str | Path) -> list[dict]:
    """Rows of a JSONL file; `x.jsonl` falls back to `x.jsonl.txt` (test fixtures, CLAUDE.md 3.2)."""
    p = Path(path)
    if not p.exists() and Path(str(p) + ".txt").exists():
        p = Path(str(p) + ".txt")
    if not p.exists():
        return []
    from ag.lineage_util import iter_rows
    return list(iter_rows(p))
