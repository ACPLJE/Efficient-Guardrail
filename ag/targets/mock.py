"""Mock tokenizer and mock targets (SPEC 4.3).

MockTokenizer: deterministic whitespace tokenizer with offset mapping. Ids are folded into
vocab_size (512) so the tiny Qwen3 test model can consume them.

MockTarget(kind="synthetic"): numpy only. The ledger puts a configurable share of attention mass
on the payload windows (attack requests) or on one untrusted window (benign requests), so the
hit@k path can be checked at any length (8k / 32k / 128k window counts).

MockTarget(kind="tiny"): the 4-layer tiny Qwen3 (lgs1.lgcore.model.tiny_qwen3) with real eager
attention through ag.reference.ledger_oracle. Positions are limited to 4,096 tokens.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ag import SCHEMA_VERSION
from ag.data.render import ChatTemplate, render, simple_template
from ag.data.types import LedgerRecord, RenderedPrompt, Request
from ag.lineage_util import canonical_hash
from ag.runtime.chunking import standard_chunk
from ag.selector.layers import resolve_layers
from ag.targets.base import RecordSpec, Target, TargetOutput

_WORD = re.compile(r"\S+")
N_SPECIAL = 4


def stable_hash(s: str, bits: int = 32) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=bits // 8).digest(), "little")


class MockTokenizer:
    """Whitespace tokenizer; one word = one token; ids folded into vocab_size."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self._rev: dict[int, str] = {}

    def token_id(self, word: str) -> int:
        i = N_SPECIAL + stable_hash(word) % (self.vocab_size - N_SPECIAL)
        self._rev.setdefault(i, word)
        return i

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offs = [], []
        for m in _WORD.finditer(text):
            ids.append(self.token_id(m.group(0)))
            offs.append((m.start(), m.end()))
        return ids, offs

    def encode(self, text: str) -> list[int]:
        return self.encode_with_offsets(text)[0]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._rev.get(int(i), f"<{int(i)}>") for i in ids)


def mock_target_layers(cfg) -> tuple[int, tuple[str, ...]]:
    """(n_layers, layer_attn_kind) of a mock target from its system-file config (flags.mock.target)."""
    n = 4 if cfg.get("kind", "synthetic") == "tiny" else int(cfg.get("n_layers", 8))
    return n, tuple(["full"] * n)


class MockTarget(Target):
    """Deterministic mock target. See module docstring."""

    def __init__(self, kind: Literal["synthetic", "tiny"] = "synthetic", *, target_id: str = "MT",
                 n_layers: int = 8, n_heads: int = 4, concentration: float = 0.6, gen_share: float = 0.1,
                 success_rate: float = 0.7, seed: int = 0, split_tokens: int = 128,
                 template: ChatTemplate = simple_template):
        self.kind = kind
        self.target_id = target_id
        # the tiny Qwen3 has 512 ids; the synthetic target uses a large id space so decode is lossless
        self.tokenizer = MockTokenizer(512 if kind == "tiny" else 1 << 24)
        self.template = template
        self.concentration = concentration
        self.gen_share = gen_share
        self.success_rate = success_rate
        self.seed = seed
        self.split_tokens = split_tokens
        self._model = None
        if kind == "tiny":
            from lgs1.lgcore.model import tiny_qwen3  # needs torch + transformers
            self._model = tiny_qwen3(seed=seed, attn="eager")
            n_layers, n_heads = self._model.config.num_hidden_layers, self._model.config.num_attention_heads
            self.max_context = 4096
        else:
            self.max_context = 1 << 20
        self.n_layers, self.n_q_heads, self.n_kv_heads = n_layers, n_heads, n_heads
        self.layer_attn_kind = tuple(["full"] * n_layers)

    # ---------------------------------------------------------------- rendering
    def render(self, req: Request) -> RenderedPrompt:
        rp = render(req, self.tokenizer, self.template, self.target_id, self.split_tokens)
        if rp.n_tokens > self.max_context:
            raise ValueError(f"{req.request_id}: {rp.n_tokens} tokens > max_context {self.max_context}")
        return rp

    # ---------------------------------------------------------------- output text
    def _attack_succeeds(self, rp: RenderedPrompt) -> bool:
        if not rp.payload_windows:
            return False
        u = (stable_hash(f"succ:{self.seed}:{rp.request_id}") % 10_000) / 10_000
        return u < self.success_rate

    def output_words(self, rp: RenderedPrompt, n: int) -> list[str]:
        """Mock output: attack successes contain PWNED at token 10 (chunk 1), benign outputs
        contain ANSWER at token 3 (chunk 0)."""
        words = [f"w{(stable_hash(f'{rp.request_id}:{i}') % 50):02d}" for i in range(n)]
        if n > 3:
            words[3] = "ANSWER"
        if self._attack_succeeds(rp) and n > 10:
            words[3] = f"w{stable_hash(rp.request_id) % 50:02d}"
            words[10] = "PWNED"
        return words

    # ---------------------------------------------------------------- generation
    def generate(self, rp: RenderedPrompt, max_new_tokens: int, *, record: RecordSpec | None,
                 forced_output: Sequence[int] | None = None,
                 drop_tokens: Sequence[tuple[int, int]] = ()) -> TargetOutput:
        if record is not None and record.split_tokens != rp.split_tokens:
            raise ValueError(f"RecordSpec.split_tokens={record.split_tokens} != rendered split {rp.split_tokens}")
        if self.kind == "tiny":
            return self._generate_tiny(rp, max_new_tokens, record, forced_output)
        if forced_output is None:
            words = self.output_words(rp, max_new_tokens)
            ids = tuple(self.tokenizer.token_id(w) for w in words)
            text = " ".join(words)
        else:                    # teacher forcing: the text is the forced ids
            ids = tuple(int(i) for i in forced_output)
            text = self.tokenizer.decode(ids)
        rec = self._synthetic_record(rp, ids, record) if record is not None else None
        return TargetOutput(ids, text, (), 0, tuple(0 for _ in ids), 0, rec)

    def _synthetic_record(self, rp: RenderedPrompt, ids: Sequence[int], spec: RecordSpec) -> LedgerRecord:
        T, NB = len(ids), rp.n_blocks
        layers = resolve_layers(spec.layers, self.n_layers, self.layer_attn_kind)
        L, H = len(layers), self.n_q_heads
        rng = np.random.default_rng(stable_hash(f"rec:{self.seed}:{rp.request_id}"))
        untr = np.flatnonzero(rp.block_untrusted_array())
        if rp.payload_windows:
            blocks = sorted({b for w in rp.payload_windows for b in (w, min(w + 1, NB - 1))})
        elif len(untr):
            w = int(rng.choice(untr))
            blocks = sorted({w, min(w + 1, NB - 1)})       # w = NB-1 would otherwise count twice
        else:
            blocks = [NB - 1]
        focus = np.zeros(NB)
        focus[blocks] = 1.0 / len(blocks)
        # per-request distractor profile (persists over tokens) + per-row noise
        profile = rng.dirichlet(np.full(NB, 0.3))
        base = 0.5 * profile[None, None, :] + 0.5 * rng.dirichlet(np.full(NB, 0.5), size=(T, L))
        depth = np.linspace(0.5, 1.0, L)[None, :, None]           # deeper layers focus more
        c = self.concentration * depth
        gen_share = np.where(np.arange(T) > 0, self.gen_share, 0.0)[:, None, None]
        block_mass = (1.0 - gen_share) * ((1 - c) * base + c * focus[None, None, :])
        gen = None
        if spec.gen_mass:
            gen = np.zeros((T, L, T))
            for t in range(1, T):
                gen[t, :, :t] = rng.dirichlet(np.ones(t), size=L) * self.gen_share
        if not spec.first_token:
            block_mass[0] = 0.0
        head = None
        if spec.per_chunk_heads:
            head = np.zeros((len(spec.chunk_defs), L, H, NB))
            mix = rng.uniform(0.0, 0.5, size=H)
            for ci, cd in enumerate(spec.chunk_defs):
                ch = standard_chunk(cd, T)
                lo = max(ch.tok_start, 1)
                mean = block_mass[lo:ch.tok_end].mean(axis=0) if ch.tok_end > lo else block_mass[:1].mean(axis=0)
                for h in range(H):
                    noise = rng.dirichlet(np.full(NB, 0.5), size=L) * mean.sum(axis=-1, keepdims=True)
                    head[ci, :, h] = (1 - mix[h]) * mean + mix[h] * noise
        dt = np.float16 if spec.precision == "fp16" else np.float32
        return LedgerRecord(
            request_id=rp.request_id, target_id=self.target_id, producer="mock", schema_version=SCHEMA_VERSION,
            record_spec_hash=canonical_hash(spec.to_json())[:16], precision=spec.precision,
            first_token_policy="include" if spec.first_token else "skip", prompt_len=rp.n_tokens,
            split_tokens=rp.split_tokens, chunk_defs=tuple(spec.chunk_defs),
            output_ids=np.asarray(ids, dtype=np.int32), t_ns=np.zeros(T, dtype=np.int64),
            layers=np.asarray(layers, dtype=np.int16), block_mass=block_mass.astype(dt),
            block_untrusted=rp.block_untrusted_array().astype(np.uint8),
            gen_mass=None if gen is None else gen.astype(np.float16),
            chunk_head_mass=None if head is None else head.astype(np.float16),
        )

    def _generate_tiny(self, rp, max_new_tokens, spec, forced_output) -> TargetOutput:
        from ag.reference.ledger_oracle import decode_with_ledger
        res = decode_with_ledger(self._model, list(rp.prompt_ids), max_new_tokens,
                                 split_tokens=rp.split_tokens, spec=spec or RecordSpec(),
                                 forced_output=forced_output)
        rec = None
        if spec is not None:
            rec = res.to_record(rp, self.target_id, "reference", spec)
        text = self.tokenizer.decode(res.output_ids)
        return TargetOutput(tuple(res.output_ids), text, (), 0, tuple(0 for _ in res.output_ids), 0, rec)
