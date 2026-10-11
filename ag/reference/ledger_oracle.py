"""Reference ledger (oracle) for differential testing (SPEC 4.5, 5.2; E2.1, E2.2).

Greedy decode with eager attention and output_attentions=True. Row t of the ledger is the
attention row of the query that produces output token t:
  t = 0  the last prompt position n-1 (the prefill pass)          -> first_token_policy "include"
  t >= 1 the input token y_{t-1} at position n+t-1 (a decode step)
Keys are [0, n+t): prompt keys are summed per 128-token block (block_mass), generated keys
[n, n+t) go to gen_mass[t, l, s] for s < t. Heads are averaged over Q heads. Rows sum to 1.

S1 runs this on the tiny Qwen3 (lgs1.lgcore.model.tiny_qwen3, eager everywhere). The real-model
path (SDPA prefill + eager decode, last-row recompute) is S2 work.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ag import SCHEMA_VERSION
from ag.data.types import LedgerRecord, RenderedPrompt
from ag.lineage_util import canonical_hash
from ag.runtime.chunking import standard_chunk
from ag.selector.layers import resolve_layers


@dataclass(eq=False)
class LedgerResult:
    output_ids: list[int]
    layers: tuple[int, ...]
    prompt_len: int
    split_tokens: int
    head_block_mass: np.ndarray           # float64 [T, L, H, NB]
    head_gen_mass: np.ndarray             # float64 [T, L, H, T]

    @property
    def block_mass(self) -> np.ndarray:   # [T, L, NB]
        return self.head_block_mass.mean(axis=2)

    @property
    def gen_mass(self) -> np.ndarray:     # [T, L, T]
        return self.head_gen_mass.mean(axis=2)

    def chunk_head_mass(self, chunk_defs: Sequence[str]) -> np.ndarray:
        """[C, L, H, NB]: mean over rows t >= 1 of each standard chunk (SPEC 5.2)."""
        T = len(self.output_ids)
        out = np.zeros((len(chunk_defs),) + self.head_block_mass.shape[1:], dtype=np.float64)
        for ci, cd in enumerate(chunk_defs):
            ch = standard_chunk(cd, T)
            lo = max(ch.tok_start, 1)
            if ch.tok_end > lo:
                out[ci] = self.head_block_mass[lo:ch.tok_end].mean(axis=0)
        return out

    def to_record(self, rp: RenderedPrompt, target_id: str, producer: str, spec) -> LedgerRecord:
        dt = np.float16 if spec.precision == "fp16" else np.float32
        T = len(self.output_ids)
        bm = self.block_mass
        if not spec.first_token:
            bm = bm.copy()
            bm[0] = 0.0
        head = self.chunk_head_mass(spec.chunk_defs) if spec.per_chunk_heads else None
        return LedgerRecord(
            request_id=rp.request_id, target_id=target_id, producer=producer, schema_version=SCHEMA_VERSION,
            record_spec_hash=canonical_hash(spec.to_json())[:16], precision=spec.precision,
            first_token_policy="include" if spec.first_token else "skip", prompt_len=self.prompt_len,
            split_tokens=self.split_tokens, chunk_defs=tuple(spec.chunk_defs),
            output_ids=np.asarray(self.output_ids, dtype=np.int32), t_ns=np.zeros(T, dtype=np.int64),
            layers=np.asarray(self.layers, dtype=np.int16), block_mass=bm.astype(dt),
            block_untrusted=rp.block_untrusted_array().astype(np.uint8),
            gen_mass=self.gen_mass.astype(np.float16) if spec.gen_mass else None,
            chunk_head_mass=None if head is None else head.astype(np.float16),
            token_head_mass=self.head_block_mass.astype(np.float16) if spec.per_token_heads else None,
        )


def _block_sum(row: np.ndarray, n: int, split_tokens: int) -> np.ndarray:
    """row [..., K] -> [..., NB]: sum of prompt keys [0, n) per block."""
    nb = max(1, -(-n // split_tokens))
    pad = nb * split_tokens - n
    p = row[..., :n]
    if pad:
        p = np.concatenate([p, np.zeros(p.shape[:-1] + (pad,), dtype=p.dtype)], axis=-1)
    return p.reshape(p.shape[:-1] + (nb, split_tokens)).sum(axis=-1)


def decode_with_ledger(model, prompt_ids: Sequence[int], max_new_tokens: int, *, split_tokens: int = 128,
                       spec=None, forced_output: Sequence[int] | None = None,
                       drop_tokens: Sequence[tuple[int, int]] = ()) -> LedgerResult:
    """Greedy (or teacher-forced) decode that records the ledger. Needs torch and an eager model."""
    import torch

    if drop_tokens:
        raise NotImplementedError("drop_tokens (leave-one-out scorer) is S2 work")
    n = len(prompt_ids)
    n_layers = model.config.num_hidden_layers
    attn_kind = getattr(model.config, "layer_types", None)
    layers = resolve_layers(spec.layers if spec is not None else "all", n_layers,
                            [("full" if k == "full_attention" else "sliding") for k in attn_kind] if attn_kind else None)
    T = len(forced_output) if forced_output is not None else max_new_tokens
    H = model.config.num_attention_heads
    nb = max(1, -(-n // split_tokens))
    head_bm = np.zeros((T, len(layers), H, nb))
    head_gm = np.zeros((T, len(layers), H, T))
    out: list[int] = []

    with torch.no_grad():
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
        res = model(input_ids=ids, use_cache=True, output_attentions=True)
        past = res.past_key_values
        for t in range(T):
            rows = [res.attentions[l][0, :, -1, :].double().cpu().numpy() for l in layers]   # [H, n+t]
            for li, row in enumerate(rows):
                head_bm[t, li] = _block_sum(row, n, split_tokens)
                if t:
                    head_gm[t, li, :, :t] = row[:, n:n + t]
            nxt = int(forced_output[t]) if forced_output is not None else int(res.logits[0, -1].argmax())
            out.append(nxt)
            if t + 1 < T:
                res = model(input_ids=torch.tensor([[nxt]], dtype=torch.long), past_key_values=past,
                            use_cache=True, output_attentions=True)
                past = res.past_key_values
    return LedgerResult(out, tuple(layers), n, split_tokens, head_bm, head_gm)
