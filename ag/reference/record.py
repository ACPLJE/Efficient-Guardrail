"""Record v1 store: one npz per (request, target, record spec, producer) + records/index.jsonl
(SPEC 5). Writes are atomic (tmp + os.replace); index appends hold a thread lock and an flock.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import numpy as np

from ag import SCHEMA_VERSION
from ag.data.types import LedgerRecord

_META_STR = ("request_id", "target_id", "producer", "record_spec_hash", "precision", "first_token_policy")
_META_INT = ("schema_version", "prompt_len", "split_tokens")
_ARRAYS = ("output_ids", "t_ns", "layers", "block_mass", "block_untrusted", "gen_mass", "chunk_head_mass",
           "token_head_mass", "sink_mass")
PRODUCERS = ("mock", "reference", "sglang-shim", "replay")


def record_key(request_id: str, target_id: str, record_spec_hash: str, producer: str) -> str:
    raw = "\x1f".join((request_id, target_id, record_spec_hash, producer)).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def record_to_bytes(rec: LedgerRecord) -> bytes:
    arrays = {k: getattr(rec, k) for k in _ARRAYS if getattr(rec, k) is not None}
    for k in _META_STR:
        arrays[k] = np.asarray(getattr(rec, k))
    for k in _META_INT:
        arrays[k] = np.asarray(getattr(rec, k), dtype=np.int64)
    arrays["chunk_defs"] = np.asarray(rec.chunk_defs, dtype=np.str_)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def load_record(path: str | Path) -> LedgerRecord:
    with np.load(path, allow_pickle=False) as z:
        kw = {k: str(z[k]) for k in _META_STR}
        kw.update({k: int(z[k]) for k in _META_INT})
        if kw["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"{path}: Record schema_version {kw['schema_version']} != {SCHEMA_VERSION}")
        kw["chunk_defs"] = tuple(str(x) for x in z["chunk_defs"])
        for k in _ARRAYS:
            kw[k] = z[k] if k in z.files else None
    names = {f.name for f in fields(LedgerRecord)}
    return LedgerRecord(**{k: v for k, v in kw.items() if k in names})


class RecordStore:
    """records/<key>.npz + records/index.jsonl under `root`."""

    def __init__(self, root: str | Path, run_id: str = ""):
        self.root = Path(root)
        self.run_id = run_id
        self._lock = threading.Lock()

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    def put(self, rec: LedgerRecord) -> str:
        if rec.producer not in PRODUCERS:
            raise ValueError(f"unknown producer {rec.producer!r}")
        name = record_key(rec.request_id, rec.target_id, rec.record_spec_hash, rec.producer) + ".npz"
        self.root.mkdir(parents=True, exist_ok=True)
        data = record_to_bytes(rec)
        tmp = self.root / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, self.root / name)
        row = {
            "request_id": rec.request_id, "target_id": rec.target_id, "producer": rec.producer,
            "schema_version": rec.schema_version, "file": name, "sha256": hashlib.sha256(data).hexdigest(),
            "T": rec.n_tokens, "prompt_len": rec.prompt_len, "n_blocks": rec.n_blocks,
            "layers": [int(x) for x in rec.layers], "precision": rec.precision,
            "first_token_policy": rec.first_token_policy, "record_spec_hash": rec.record_spec_hash,
            "run_id": self.run_id, "created_ns": time.time_ns(),
        }
        line = json.dumps(row, sort_keys=True) + "\n"
        with self._lock, open(self.index_path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return name

    def index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        with open(self.index_path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _match(self, row: dict, filters: dict) -> bool:
        return all(v is None or row.get(k) == v for k, v in filters.items())

    def get(self, request_id: str, target_id: str, *, producer: str | None = None,
            record_spec_hash: str | None = None) -> LedgerRecord:
        rows = [r for r in self.index() if self._match(r, {"request_id": request_id, "target_id": target_id,
                                                              "producer": producer,
                                                              "record_spec_hash": record_spec_hash})]
        files = {r["file"] for r in rows}
        if not files:
            raise KeyError(f"no Record for {request_id} / {target_id}")
        if len(files) > 1:
            raise KeyError(f"{len(files)} Records match {request_id} / {target_id}; pass producer and record_spec_hash")
        return load_record(self.root / files.pop())

    def iter(self, **filters) -> Iterator[LedgerRecord]:
        seen: set[str] = set()
        for r in self.index():
            if r["file"] not in seen and self._match(r, filters):
                seen.add(r["file"])
                yield load_record(self.root / r["file"])
