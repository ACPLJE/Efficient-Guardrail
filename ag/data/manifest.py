"""Data manifests (SPEC 7.1): raw/data/<data_manifest_sha>/{manifest.json, requests.jsonl}.

data_manifest_sha = canonical_hash of the request rows (with data_manifest_sha blanked), so the
same requests always land in the same directory.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ag import SCHEMA_VERSION
from ag.data.types import Request
from ag.harness.writer import atomic_write_json, atomic_write_text
from ag.lineage_util import canonical_hash


def manifest_sha(reqs: Sequence[Request]) -> str:
    rows = [dataclasses.replace(r, data_manifest_sha="").to_json() for r in reqs]
    return canonical_hash(rows)[:16]


def stamp(reqs: Sequence[Request]) -> tuple[list[Request], str]:
    sha = manifest_sha(reqs)
    return [dataclasses.replace(r, data_manifest_sha=sha) for r in reqs], sha


def write_manifest(out_root: str | Path, reqs: Sequence[Request], sha: str, *, generator: str,
                   params: dict) -> Path:
    d = Path(out_root) / "raw" / "data" / sha
    if (d / "manifest.json").exists():
        return d
    lines = [json.dumps({**r.to_json(), "schema_version": SCHEMA_VERSION}, sort_keys=True) for r in reqs]
    atomic_write_text(d / "requests.jsonl", "\n".join(lines) + "\n")
    atomic_write_json(d / "manifest.json", {
        "data_manifest_sha": sha, "schema_version": SCHEMA_VERSION, "generator": generator, "params": params,
        "n_requests": len(reqs), "n_attack": sum(r.is_attack for r in reqs),
        "by_length_class": dict(Counter(r.length_class for r in reqs)),
        "by_source": dict(Counter(r.source for r in reqs)),
    })
    return d


def load_requests(data_dir: str | Path) -> list[Request]:
    from ag.harness.writer import read_rows
    return [Request.from_json({k: v for k, v in row.items() if k not in ("schema_version", "run_id")})
            for row in read_rows(Path(data_dir) / "requests.jsonl")]
