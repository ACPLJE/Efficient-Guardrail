"""Tokenizer wrappers.  Real runs use the HF tokenizers from the offline cache (Qwen3 for every length target,
Llama 3 for the second count); tests use a whitespace tokenizer with the same interface."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import TOKENIZERS

_WORD = re.compile(r"\S+")


class WhitespaceTok:
    """1 token = 1 whitespace-delimited word (tests / dry-runs without model files)."""
    name = "whitespace"

    def count(self, texts: Sequence[str]) -> List[int]:
        return [len(_WORD.findall(t or "")) for t in texts]

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        offs = [(m.start(), m.end()) for m in _WORD.finditer(text or "")]
        return list(range(len(offs))), offs

    def decode_span(self, text: str, offsets: Sequence[Tuple[int, int]], a: int, b: int) -> str:
        if a >= b or not offsets:
            return ""
        return text[offsets[a][0]:offsets[b - 1][1]]


class HFTok:
    def __init__(self, repo_id: str, name: str):
        from transformers import AutoTokenizer
        self.name = name
        self.repo_id = repo_id
        self.tok = AutoTokenizer.from_pretrained(repo_id)
        if not getattr(self.tok, "is_fast", False):
            raise RuntimeError(f"{repo_id}: a fast tokenizer is required for offset mappings")

    def count(self, texts: Sequence[str], batch: int = 256) -> List[int]:
        out: List[int] = []
        for i in range(0, len(texts), batch):
            enc = self.tok(list(texts[i:i + batch]), add_special_tokens=False, return_attention_mask=False,
                           return_length=True)
            out += [int(x) for x in enc["length"]]
        return out

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False)
        return list(enc["input_ids"]), [tuple(x) for x in enc["offset_mapping"]]

    def decode_span(self, text: str, offsets: Sequence[Tuple[int, int]], a: int, b: int) -> str:
        if a >= b or not offsets:
            return ""
        return text[offsets[a][0]:offsets[b - 1][1]]


def get_tok(kind: str = "qwen3"):
    if kind in ("whitespace", "ws"):
        return WhitespaceTok()
    if kind in TOKENIZERS:
        return HFTok(TOKENIZERS[kind], kind)
    return HFTok(kind, kind)          # a repo id


def token_span_for_chars(offsets: Sequence[Tuple[int, int]], span: Tuple[int, int]) -> Tuple[int, int]:
    """Smallest token span [a, b) that covers the character span (offsets from encode_with_offsets)."""
    a = len(offsets)
    for i, (s, e) in enumerate(offsets):
        if e > span[0]:
            a = i
            break
    b = len(offsets)
    for i in range(a, len(offsets)):
        if offsets[i][0] >= span[1]:
            b = i
            break
    return a, max(b, min(a + 1, len(offsets)))


def chunk_by_tokens(text: str, tok, chunk_tokens: int) -> List[Tuple[int, int, int]]:
    """Consecutive token chunks -> [(chunk_idx, char_start, char_end)] on the tokenizer's grid."""
    ids, offs = tok.encode_with_offsets(text)
    out = []
    for c, start in enumerate(range(0, len(ids), chunk_tokens)):
        seg = offs[start:start + chunk_tokens]
        if not seg:
            break
        out.append((c, seg[0][0], seg[-1][1]))
    return out
