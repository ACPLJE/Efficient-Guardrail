"""Output chunking (SPEC 1.4).

Serving chunks: chunk 0 is the first `first` tokens, later chunks are `next` tokens (or end at a
sentence boundary when `sentence` is on). A tool call is always one chunk. Standard chunk
definitions (first8, first32, all) are single chunks used for offline analysis.
"""
from __future__ import annotations

from collections.abc import Sequence

from ag.data.types import ChunkInfo, ChunkingConfig

STANDARD_CHUNK_DEFS = ("first8", "first32", "all")


def serving_chunks(n_tokens: int, cfg: ChunkingConfig = ChunkingConfig(),
                   tool_spans: Sequence[tuple[int, int]] = (),
                   sentence_ends: Sequence[int] = ()) -> tuple[ChunkInfo, ...]:
    """Split [0, n_tokens) into serving chunks.

    tool_spans: [start, end) output-token spans of tool calls; each becomes exactly one chunk and
    no boundary is placed inside it. sentence_ends: token indices just after a sentence end.
    """
    spans = sorted((max(0, a), min(n_tokens, b)) for a, b in tool_spans if b > a)
    ends = sorted(set(sentence_ends))
    out: list[ChunkInfo] = []
    i = 0
    while i < n_tokens:
        size = cfg.first if not out else cfg.next
        boundary = min(i + size, n_tokens)
        nxt = next(((a, b) for a, b in spans if b > i), None)
        if nxt is not None and nxt[0] <= i:
            out.append(ChunkInfo(len(out), i, nxt[1], True))
            i = nxt[1]
            continue
        if nxt is not None and nxt[0] < boundary:
            boundary = nxt[0]
        elif cfg.sentence:
            cut = next((e for e in ends if i < e < boundary), None)
            if cut is not None:
                boundary = cut
        out.append(ChunkInfo(len(out), i, boundary, False))
        i = boundary
    return tuple(out)


def standard_chunk(chunk_def: str, n_tokens: int) -> ChunkInfo:
    if chunk_def == "first8":
        return ChunkInfo(0, 0, min(8, n_tokens))
    if chunk_def == "first32":
        return ChunkInfo(0, 0, min(32, n_tokens))
    if chunk_def == "all":
        return ChunkInfo(0, 0, n_tokens)
    raise ValueError(f"unknown chunk_def {chunk_def!r}")
