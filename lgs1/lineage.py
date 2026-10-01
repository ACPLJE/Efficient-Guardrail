"""E0-2: lineage hashes on every result row + resume/rescore guard.

Every rows file written by any runner carries, on EVERY row, the six lineage fields:

    model_sha          fingerprint of the backbone checkpoint directory (weights + config)
    router_sha         fingerprint of the router file (or "none")
    cfg_hash           canonical hash of the run configuration dict (b, stride, ell, k, pool, tau, ...)
    recipe_hash        canonical hash of the data recipe (synthesis rules) or the recipe file
    data_manifest_sha  sha256 of the dataset manifest file (lg_data_v2/manifests/*.manifest.json)
    code_commit        git HEAD of the code base (+ "-dirty" when the tree has changes)

`assert_resume_compatible()` compares the lineage of an existing rows file with the lineage of the
run that wants to resume or rescore it.  A mismatch in model_sha / router_sha / cfg_hash /
recipe_hash / data_manifest_sha raises `LineageMismatch`; the CLI maps that to exit code 2 (the
2026-09-10 review: resume/rescore only checked ids and window counts).  `code_commit` is stamped on
every row but does not block a resume: an unzipped package has no git, so code_commit is a hash of
every *.py file and changes with any harness fix — a resumed file then records both code versions
row by row (a warning is printed).  `keys=LINEAGE_KEYS` (CLI `--strict`) makes it blocking again.

Usage in a runner:

    lin = Lineage.build(model_dir, router_path, cfg_dict, recipe, manifest_path, code_dir)
    assert_resume_compatible(rows_path, lin)         # raises on mismatch (exit 2 in CLI)
    with RowsWriter(rows_path, lin, resume=True) as w:
        for row in ...:
            w.write(row)                              # lineage fields added automatically
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

LINEAGE_KEYS = ("model_sha", "router_sha", "cfg_hash", "recipe_hash", "data_manifest_sha", "code_commit")
RESUME_KEYS = ("model_sha", "router_sha", "cfg_hash", "recipe_hash", "data_manifest_sha")   # code_commit: warn only
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "config.json", "adapter_config.json",
                   "generation_config.json", "tokenizer.json", "tokenizer_config.json")


class LineageMismatch(RuntimeError):
    """Raised when an existing rows file was produced under a different lineage."""


# --------------------------------------------------------------------------------------------
# hashing primitives
# --------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: os.PathLike | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    """sha256 of the canonical JSON encoding (sorted keys, no whitespace, NaN forbidden)."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)
    return sha256_bytes(s.encode("utf-8"))


def _cache_path(d: Path) -> Path:
    return d / ".lineage_cache.json"


def dir_fingerprint(path: os.PathLike | str, patterns: Sequence[str] = WEIGHT_PATTERNS, use_cache: bool = True) -> str:
    """Fingerprint of a checkpoint directory: sha256 over sorted (relpath, size, sha256(file)).

    File hashes are cached in `<dir>/.lineage_cache.json` keyed by (size, mtime_ns) so that a
    16 GB checkpoint is read once.  A single file path is fingerprinted directly.
    """
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if not p.is_dir():
        raise FileNotFoundError(path)
    files = sorted({f for pat in patterns for f in p.rglob(pat) if f.is_file()})
    cache: Dict[str, Any] = {}
    cp = _cache_path(p)
    if use_cache and cp.exists():
        try:
            cache = json.loads(cp.read_text())
        except Exception:
            cache = {}
    entries = []
    dirty = False
    for f in files:
        rel = str(f.relative_to(p))
        st = f.stat()
        key = f"{st.st_size}:{st.st_mtime_ns}"
        ent = cache.get(rel)
        if ent and ent.get("key") == key:
            digest = ent["sha256"]
        else:
            digest = sha256_file(f)
            cache[rel] = {"key": key, "sha256": digest}
            dirty = True
        entries.append((rel, st.st_size, digest))
    if use_cache and dirty:
        try:
            cp.write_text(json.dumps(cache, indent=0, sort_keys=True))
        except OSError:
            pass  # read-only checkpoint dirs are fine
    return canonical_hash(entries)


def code_commit(code_dir: Optional[os.PathLike | str] = None) -> str:
    """`<git sha>` or `<git sha>-dirty`; falls back to a hash of *.py files when not a git repo."""
    d = Path(code_dir or Path(__file__).resolve().parent.parent)
    try:
        sha = subprocess.check_output(["git", "-C", str(d), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(["git", "-C", str(d), "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        return sha + ("-dirty" if status.strip() else "")
    except Exception:
        pys = sorted(f for f in d.rglob("*.py") if ".venv" not in f.parts and "site-packages" not in f.parts)
        h = hashlib.sha256()
        for f in pys:
            h.update(str(f.relative_to(d)).encode())
            h.update(f.read_bytes())
        return "nogit-" + h.hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Lineage record
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Lineage:
    model_sha: str
    router_sha: str
    cfg_hash: str
    recipe_hash: str
    data_manifest_sha: str
    code_commit: str

    @classmethod
    def build(cls, model_dir: Optional[str], router_path: Optional[str], cfg: Dict[str, Any],
              recipe: Any, manifest_path: Optional[str], code_dir: Optional[str] = None) -> "Lineage":
        return cls(
            model_sha=dir_fingerprint(model_dir) if model_dir else "none",
            router_sha=dir_fingerprint(router_path) if router_path else "none",
            cfg_hash=canonical_hash(cfg),
            recipe_hash=(sha256_file(recipe) if isinstance(recipe, (str, os.PathLike)) and os.path.exists(str(recipe))
                         else canonical_hash(recipe)),
            data_manifest_sha=sha256_file(manifest_path) if manifest_path else "none",
            code_commit=code_commit(code_dir),
        )

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Lineage":
        missing = [k for k in LINEAGE_KEYS if k not in row]
        if missing:
            raise LineageMismatch(f"rows file lacks lineage fields {missing}; regenerate it (E0-2)")
        return cls(**{k: str(row[k]) for k in LINEAGE_KEYS})

    def diff(self, other: "Lineage") -> Dict[str, tuple]:
        return {f.name: (getattr(self, f.name), getattr(other, f.name))
                for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)}


# --------------------------------------------------------------------------------------------
# rows I/O with lineage
# --------------------------------------------------------------------------------------------
def iter_rows(path: os.PathLike | str, tolerant: bool = False) -> Iterator[Dict[str, Any]]:
    """Rows of a JSONL file.  A file that another job is still appending to (or that a crash left half-written) can end
    in a truncated line or, on NFS/Lustre, in a NUL-filled block; those trailing artifacts are skipped with a warning.
    `tolerant=True` (summaries) also skips malformed lines in the middle; the default raises on those."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    n_bad = 0
    for i, line in enumerate(lines):
        line = line.strip().strip("\x00").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            rest = [x for x in lines[i + 1:] if x.strip().strip("\x00").strip()]
            if tolerant or not rest:
                n_bad += 1
                continue
            raise ValueError(f"{path}: malformed JSON at line {i + 1}: {e} (not the last line -> not a partial write)") from e
    if n_bad:
        print(f"[lineage] {path}: skipped {n_bad} malformed line(s) (partial write / NUL block)", file=sys.stderr, flush=True)


def read_first_lineage(path: os.PathLike | str) -> Optional[Lineage]:
    for row in iter_rows(path):
        return Lineage.from_row(row)
    return None


def assert_resume_compatible(rows_path: os.PathLike | str, lineage: Lineage,
                             keys: Sequence[str] = RESUME_KEYS) -> int:
    """Return the number of existing rows if compatible; raise LineageMismatch otherwise.

    Every row is checked (a rows file must be produced under exactly one model/config lineage).
    Keys outside `keys` (by default only code_commit) are reported once as a warning, never raised.
    """
    p = Path(rows_path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    n = 0
    soft: Dict[str, tuple] = {}
    for i, row in enumerate(iter_rows(p)):
        old = Lineage.from_row(row)
        diff = old.diff(lineage)
        d = {k: v for k, v in diff.items() if k in keys}
        if d:
            desc = "; ".join(f"{k}: existing={a[:12]}… current={b[:12]}…" for k, (a, b) in d.items())
            raise LineageMismatch(f"{p} row {i}: lineage mismatch -> {desc}. "
                                  f"Refusing to resume/rescore; use a new output path.")
        for k, v in diff.items():
            if k not in keys and k not in soft:
                soft[k] = v
        n += 1
    if soft:
        desc = "; ".join(f"{k}: existing={a[:12]}… current={b[:12]}…" for k, (a, b) in soft.items())
        print(f"[lineage] {p.name}: resuming {n} rows across a code change (model/cfg identical): {desc}",
              file=sys.stderr, flush=True)
    return n


class RowsWriter:
    """Append-only JSONL writer that stamps the lineage on every row."""

    def __init__(self, path: os.PathLike | str, lineage: Lineage, resume: bool = False, flush_every: int = 50):
        self.path = Path(path)
        self.lineage = lineage.as_dict()
        self.resume = resume
        self.flush_every = flush_every
        self._f = None
        self._n = 0

    def __enter__(self) -> "RowsWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume:
            assert_resume_compatible(self.path, Lineage(**self.lineage))
            self._f = open(self.path, "a", encoding="utf-8")
        else:
            if self.path.exists() and self.path.stat().st_size > 0:
                raise FileExistsError(f"{self.path} exists; pass resume=True (lineage is checked) or a new path")
            self._f = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, row: Dict[str, Any]) -> None:
        clash = [k for k in LINEAGE_KEYS if k in row and row[k] != self.lineage[k]]
        if clash:
            raise LineageMismatch(f"row carries different lineage fields {clash}")
        out = dict(row)
        out.update(self.lineage)
        out.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._f.write(json.dumps(out, ensure_ascii=False, allow_nan=False) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()

    def __exit__(self, *exc) -> None:
        if self._f:
            self._f.flush()
            self._f.close()


def dedupe_rows(rows, key: str = "id"):
    """One row per id: two writers appending to the same rows file (b1-start 9/19: the 40 GB and 48 GB arrays both ran
    gpt-oss v2main) leave duplicate ids -- keep the scored row (pred / verdict not None) when there is one, else the first."""
    out, pos = [], {}
    for r in rows:
        k = r.get(key)
        if k is None:
            out.append(r)
            continue
        scored = r.get("pred") is not None or r.get("verdict") is not None
        if k not in pos:
            pos[k] = len(out)
            out.append(r)
        elif scored:
            prev = out[pos[k]]
            if prev.get("pred") is None and prev.get("verdict") is None:
                out[pos[k]] = r
    return out


def done_ids(rows_path: os.PathLike | str, key: str = "id") -> set:
    p = Path(rows_path)
    if not p.exists():
        return set()
    return {row[key] for row in iter_rows(p) if key in row}


# --------------------------------------------------------------------------------------------
# CLI:  python -m lgs1.lineage check --rows R --model M --cfg cfg.json ...
# --------------------------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="E0-2 lineage tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="verify a rows file can be resumed under the current lineage (exit 2 on mismatch)")
    c.add_argument("--rows", required=True)
    c.add_argument("--model", default=None)
    c.add_argument("--router", default=None)
    c.add_argument("--cfg", default=None, help="json file with the run config")
    c.add_argument("--recipe", default=None, help="recipe json file (hashed by content)")
    c.add_argument("--manifest", default=None)
    c.add_argument("--code-dir", default=None)
    c.add_argument("--strict", action="store_true", help="also require an identical code_commit (default: warn only)")
    f = sub.add_parser("fingerprint", help="print the fingerprint of a checkpoint dir / file")
    f.add_argument("path")
    s = sub.add_parser("show", help="print the lineage of a rows file")
    s.add_argument("rows")
    a = ap.parse_args(argv)
    if a.cmd == "fingerprint":
        print(dir_fingerprint(a.path))
        return 0
    if a.cmd == "show":
        lin = read_first_lineage(a.rows)
        print(json.dumps(lin.as_dict() if lin else None, indent=2))
        return 0
    cfg = json.load(open(a.cfg)) if a.cfg else {}
    lin = Lineage.build(a.model, a.router, cfg, a.recipe or {}, a.manifest, a.code_dir)
    try:
        n = assert_resume_compatible(a.rows, lin, keys=LINEAGE_KEYS if a.strict else RESUME_KEYS)
    except LineageMismatch as e:
        print(f"LINEAGE MISMATCH: {e}", file=sys.stderr)
        return 2
    print(f"ok: {n} existing rows are compatible with the current lineage")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
