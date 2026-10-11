"""Request -> RenderedPrompt (SPEC 2.2).

A chat template turns segments into one prompt string and reports where each segment's content
sits in it. The tokenizer's offset mapping then turns char spans (trust boundaries, payload,
harmful needle) into token spans. Rendering never changes the length; length classes are the
data generator's job.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from ag.data import window_map as wm
from ag.data.types import RenderedPrompt, Request, Segment

CharSpan = tuple[int, int]


class Tokenizer(Protocol):
    vocab_size: int

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[CharSpan]]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


ChatTemplate = Callable[[Sequence[Segment]], tuple[str, list[CharSpan]]]


def simple_template(segments: Sequence[Segment]) -> tuple[str, list[CharSpan]]:
    """Mock chat template: '<|role|>' header token, newline, content, newline."""
    parts: list[str] = []
    spans: list[CharSpan] = []
    pos = 0
    for seg in segments:
        head = f"<|{seg.role}|>\n"
        parts.append(head)
        pos += len(head)
        spans.append((pos, pos + len(seg.text)))
        parts.append(seg.text + "\n")
        pos += len(seg.text) + 1
    parts.append("<|assistant|>\n")
    return "".join(parts), spans


def _token_mask(offsets: Sequence[CharSpan], intervals: Sequence[CharSpan]) -> np.ndarray:
    m = np.zeros(len(offsets), dtype=bool)
    if not intervals:
        return m
    for i, (s, e) in enumerate(offsets):
        for a, b in intervals:
            if s < b and e > a:
                m[i] = True
                break
    return m


def _token_span(offsets: Sequence[CharSpan], a: int, b: int) -> tuple[int, int] | None:
    idx = [i for i, (s, e) in enumerate(offsets) if s < b and e > a]
    return (idx[0], idx[-1] + 1) if idx else None


def render(req: Request, tokenizer: Tokenizer, template: ChatTemplate = simple_template,
           target_id: str = "MT", split_tokens: int = 128) -> RenderedPrompt:
    text, content = template(req.segments)
    ids, offsets = tokenizer.encode_with_offsets(text)
    n = len(ids)

    untrusted_iv: list[CharSpan] = []
    payload_char: list[CharSpan] = []
    harm_char: list[CharSpan] = []
    for seg, (a, b) in zip(req.segments, content):
        if not seg.trusted:
            untrusted_iv.append((a, b))
        payload_char += [(a + s, a + e) for s, e in seg.payload_char_spans]
        harm_char += [(a + s, a + e) for s, e in seg.harm_char_spans]

    untrusted = _token_mask(offsets, untrusted_iv)
    payload_spans = tuple(sp for sp in (_token_span(offsets, a, b) for a, b in payload_char) if sp)
    harm_spans = tuple(sp for sp in (_token_span(offsets, a, b) for a, b in harm_char) if sp)

    bu = wm.block_untrusted(untrusted, split_tokens)
    n_sites = req.attack.n_sites if req.attack else 1
    pw, per_site, rule = wm.payload_windows(payload_spans, n, split_tokens, n_sites=n_sites)
    return RenderedPrompt(
        request_id=req.request_id,
        target_id=target_id,
        prompt_ids=tuple(ids),
        untrusted_mask=np.packbits(untrusted, bitorder="little").tobytes(),
        payload_spans=payload_spans,
        harm_spans=harm_spans,
        split_tokens=split_tokens,
        n_blocks=len(bu),
        block_untrusted=bu.tobytes(),
        payload_windows=pw,
        payload_windows_per_site=per_site,
        payload_rule=rule,
        eligible_windows=int(wm.eligible_windows(bu).sum()),
    )


def window_text(rp: RenderedPrompt, w: int, tokenizer: Tokenizer) -> tuple[str, tuple[int, ...]]:
    """Window text by decoding its tokens (SPEC 2.2)."""
    a, b = wm.window_token_span(w, rp.n_tokens, rp.split_tokens)
    ids = rp.prompt_ids[a:b]
    return tokenizer.decode(ids), tuple(ids)
