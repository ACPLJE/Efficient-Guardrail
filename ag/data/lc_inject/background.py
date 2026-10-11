"""Background documents for LC-Inject (plan 5.3 #2).

Rows are the LG/AG hf_stream format {"id", "text", "meta"} (lgs1.download.fetch_hf_stream), e.g.
$AG_SOURCES/bg_wiki_en/docs.jsonl.gz written on the login node by the S0 downloader. Paragraphs
are the stripped non-empty lines of the text (Wikipedia section headings become short paragraphs).
"""
from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DOWNLOAD_HINT = "fetch it on the login node: bash slurm/ag_download_login.sh --only bg_wiki_en"
_UNEXPANDED = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    paragraphs: tuple[str, ...]


def split_paragraphs(text: str) -> tuple[str, ...]:
    """Split on newlines (blank lines included), strip, drop empty pieces."""
    return tuple(p for p in (x.strip() for x in text.splitlines()) if p)


def resolve_path(path: str | os.PathLike) -> Path:
    """os.path.expandvars (+ ~); FileNotFoundError with the download hint for an unset $VAR or a missing file."""
    raw = os.fspath(path)
    p = os.path.expanduser(os.path.expandvars(raw))
    m = _UNEXPANDED.search(p)
    if m:
        raise FileNotFoundError(f"background docs path {raw!r}: environment variable {m.group(0)} is not set "
                                f"(source env/ag.env); {DOWNLOAD_HINT}")
    if not Path(p).is_file():
        raise FileNotFoundError(f"background docs file not found: {p!r} (from {raw!r}); {DOWNLOAD_HINT}")
    return Path(p)


def _open_text(p: Path):
    return gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")


def load_docs(paths: str | os.PathLike | Sequence[str | os.PathLike], *, text_field: str = "text",
              min_chars: int = 2000, max_docs: int | None = None) -> list[Doc]:
    """Docs from .jsonl.gz / .jsonl / .jsonl.txt files, in file and row order.

    Rows whose text is missing, not a string or shorter than min_chars (after strip) are skipped;
    max_docs caps the total over all files. A malformed JSON line raises ValueError(path:line).
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files = [resolve_path(p) for p in paths]           # fail before reading anything
    out: list[Doc] = []
    for p in files:
        with _open_text(p) as f:
            for ln, line in enumerate(f, 1):
                if max_docs is not None and len(out) >= max_docs:
                    return out
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{ln}: malformed JSON row ({e.msg})") from None
                if not isinstance(row, dict):
                    raise ValueError(f"{p}:{ln}: expected a JSON object, got {type(row).__name__}")
                text = row.get(text_field)
                if not isinstance(text, str) or len(text.strip()) < min_chars:
                    continue
                paras = split_paragraphs(text)
                if paras:
                    out.append(Doc(str(row.get("id", f"{p.name}:{ln}")), text, paras))
    return out
