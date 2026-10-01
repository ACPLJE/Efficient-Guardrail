"""Reference Landmark Guard forward path on a Hugging Face Qwen3-style causal LM.

This module is the executable specification used by the E0-1 equivalence tests.  It deliberately
re-implements the pieces whose correctness the 2026-09-09/10 code reviews questioned:

  * manual layer-by-layer forward (`manual_forward`)  -> must equal the HF forward       (E0-1 a)
  * block-local trunk on windows (`trunk_windows`)     -> must equal standalone runs     (E0-1 b)
  * landmark invariance to other blocks                -> checked by the tests           (E0-1 c)
  * window-unit trunk + in-window token-max pooling    (`Router`, `pool_scores`, `route`) (E0-1 d)

Design constants (report §6.1): b=128 (256-token windows, stride 128), ell=9, k=32, pool=max.
The tiny model used in tests has 4 layers / hidden 64, so tests use ell=2, b=4..8, k=2..3.

If the v0.4 code base is importable, `lgs1.lgcore.adapter` can redirect the tests to it; the
functions here then serve as the reference the v0.4 implementation must agree with.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .windows import make_windows, windows_containing

Window = Tuple[int, int]


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------
@dataclass
class LGConfig:
    b: int = 128                 # half window; window length 2b
    stride: Optional[int] = None # None -> b (overlap); 2b -> disjoint
    ell: int = 9                 # trunk depth: layers [0, ell) are block-local
    k: int = 32                  # windows routed to the deep head
    pool: str = "max"            # max | mean | top2 | top4 | lse
    tau: float = 0.5             # verdict threshold on head probability
    pad_id: int = 0

    def to_dict(self) -> Dict:
        return dict(b=self.b, stride=self.stride, ell=self.ell, k=self.k, pool=self.pool,
                    tau=self.tau, pad_id=self.pad_id)


# --------------------------------------------------------------------------------------------
# Tiny model factory (tests only)
# --------------------------------------------------------------------------------------------
def tiny_qwen3(seed: int = 0, n_layers: int = 4, hidden: int = 64, vocab: int = 512,
               heads: int = 4, kv_heads: int = 2, attn: str = "eager"):
    """A randomly initialised 4-layer Qwen3 model (~0.2M params) for equivalence tests."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=n_layers, num_attention_heads=heads,
                      num_key_value_heads=kv_heads, head_dim=hidden // heads,
                      max_position_embeddings=4096, tie_word_embeddings=False,
                      attn_implementation=attn)
    model = Qwen3ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------------
def _inner(model):
    """Return the decoder (Qwen3Model) of a *ForCausalLM or the model itself."""
    return model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model


def build_attn_mask(valid: torch.Tensor, causal: bool, dtype: torch.dtype) -> torch.Tensor:
    """Additive [B,1,T,T] float mask from a [B,T] validity mask (1 = real token)."""
    B, T = valid.shape
    neg = torch.finfo(dtype).min
    allowed = valid[:, None, None, :].bool().expand(B, 1, T, T).clone()
    if causal:
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=valid.device))
        allowed &= tri[None, None]
    # a fully-masked query row would give NaN in softmax; let padded queries attend to themselves
    eye = torch.eye(T, dtype=torch.bool, device=valid.device)[None, None]
    allowed |= eye
    mask = torch.zeros(B, 1, T, T, dtype=dtype, device=valid.device)
    mask.masked_fill_(~allowed, neg)
    return mask


def manual_forward(model, input_ids: Optional[torch.Tensor] = None, *,
                   inputs_embeds: Optional[torch.Tensor] = None,
                   valid: Optional[torch.Tensor] = None,
                   position_ids: Optional[torch.Tensor] = None,
                   layer_range: Optional[Tuple[int, int]] = None,
                   apply_norm: bool = False,
                   causal: bool = True) -> torch.Tensor:
    """Run decoder layers [start, end) by hand and return hidden states [B,T,d].

    * layer_range=None  -> all layers.  apply_norm=True adds the final RMSNorm (== HF last_hidden_state)
    * layer_range=(0,ell) with apply_norm=False == HF output_hidden_states[ell]
    * works with transformers 4.4x (tuple outputs, per-layer rotary) and 5.x (tensor outputs)
    """
    dec = _inner(model)
    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError("input_ids or inputs_embeds required")
        inputs_embeds = dec.embed_tokens(input_ids)
    h = inputs_embeds
    B, T, _ = h.shape
    if valid is None:
        valid = torch.ones(B, T, dtype=torch.long, device=h.device)
    if position_ids is None:
        position_ids = torch.arange(T, device=h.device)[None].expand(B, T)
    mask = build_attn_mask(valid, causal, h.dtype)
    n = len(dec.layers)
    start, end = (0, n) if layer_range is None else layer_range
    end = min(end, n)
    pos_emb = dec.rotary_emb(h, position_ids) if hasattr(dec, "rotary_emb") else None
    for i in range(start, end):
        layer = dec.layers[i]
        kwargs = dict(attention_mask=mask, position_ids=position_ids)
        if pos_emb is not None:
            kwargs["position_embeddings"] = pos_emb
        out = layer(h, **kwargs)
        h = out[0] if isinstance(out, (tuple, list)) else out
    if apply_norm:
        h = dec.norm(h)
    return h


def hf_hidden_states(model, input_ids: torch.Tensor, valid: Optional[torch.Tensor] = None):
    """HF reference: tuple of hidden states (len = n_layers + 1; last one is normed)."""
    dec = _inner(model)
    kwargs = dict(input_ids=input_ids, output_hidden_states=True)
    if valid is not None:
        kwargs["attention_mask"] = valid
    with torch.no_grad():
        out = dec(**kwargs)
    return out.hidden_states, out.last_hidden_state


# --------------------------------------------------------------------------------------------
# Block-local trunk
# --------------------------------------------------------------------------------------------
def gather_windows(doc_ids: torch.Tensor, windows: Sequence[Window], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack windows of a 1-D token tensor into [W, L] with right padding + validity mask."""
    L = max(e - s for s, e in windows)
    W = len(windows)
    ids = torch.full((W, L), pad_id, dtype=doc_ids.dtype, device=doc_ids.device)
    valid = torch.zeros((W, L), dtype=torch.long, device=doc_ids.device)
    for i, (s, e) in enumerate(windows):
        ids[i, : e - s] = doc_ids[s:e]
        valid[i, : e - s] = 1
    return ids, valid


def trunk_windows(model, doc_ids: torch.Tensor, windows: Sequence[Window], ell: int, pad_id: int = 0,
                  batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-local shallow trunk: layers [0, ell) on every window independently, positions reset to 0.

    Returns (hidden [W, L, d], valid [W, L]).  Because every window is an independent forward with
    its own position ids, the result for window w equals `manual_forward` on that window alone
    (E0-1 b) and does not depend on tokens outside the window (E0-1 c).
    """
    ids, valid = gather_windows(doc_ids, windows, pad_id)
    outs = []
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch_size):
            h = manual_forward(model, ids[i:i + batch_size], valid=valid[i:i + batch_size],
                               layer_range=(0, ell), apply_norm=False)
            outs.append(h)
    return torch.cat(outs, 0), valid


# --------------------------------------------------------------------------------------------
# Router (window-unit landmarks, token-max pooling) and routing
# --------------------------------------------------------------------------------------------
class Router(nn.Module):
    """LayerNorm(d) -> Linear(d, 1) probe on per-token trunk hidden states (~3k params at d=1024)."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # [W, L, d] -> [W, L]
        return self.proj(self.norm(hidden)).squeeze(-1)


def pool_scores(token_scores: torch.Tensor, valid: torch.Tensor, pool: str = "max") -> torch.Tensor:
    """In-window pooling of token scores [W, L] -> window scores [W]. Padding is ignored."""
    neg = torch.finfo(token_scores.dtype).min
    v = valid.bool()
    if pool == "max":
        return token_scores.masked_fill(~v, neg).max(dim=1).values
    if pool == "mean":
        return (token_scores * v).sum(1) / v.sum(1).clamp(min=1)
    if pool in ("top2", "top4"):
        m = int(pool[3:])
        s = token_scores.masked_fill(~v, neg)
        top = s.topk(min(m, s.shape[1]), dim=1).values
        top = torch.where(top <= neg / 2, torch.zeros_like(top), top)  # drop padded slots
        return top.sum(1) / v.sum(1).clamp(max=m, min=1)
    if pool == "lse":
        s = token_scores.masked_fill(~v, neg)
        return torch.logsumexp(s, dim=1)
    raise ValueError(f"unknown pool {pool}")


def route(window_scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, float]:
    """Top-k routing. Returns (selected window indices sorted by score desc, theta_k).

    theta_k is the k-th largest window score = the admission threshold on the *router score
    scale*.  Any evasion / decoy analysis must compare candidate scores with this theta (the
    2026-09-09 bug compared a mean-pool theta with token-max scores).
    """
    k = max(1, min(int(k), window_scores.numel()))
    top = torch.topk(window_scores, k)
    return top.indices, float(top.values[-1])


# --------------------------------------------------------------------------------------------
# Deep per-window head
# --------------------------------------------------------------------------------------------
class Head(nn.Module):
    """Probe on the last valid token of the deep (full-depth, normed) window representation."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:  # -> [W] logits
        last = valid.sum(1).clamp(min=1) - 1
        h = hidden[torch.arange(hidden.shape[0]), last]
        return self.proj(self.norm(h)).squeeze(-1)


def head_forward(model, trunk_hidden: torch.Tensor, valid: torch.Tensor, ell: int, head: Head) -> torch.Tensor:
    """Continue layers [ell, N) from the trunk hidden states of the selected windows, then score."""
    with torch.no_grad():
        h = manual_forward(model, inputs_embeds=trunk_hidden, valid=valid,
                           layer_range=(ell, len(_inner(model).layers)), apply_norm=True)
        return head(h, valid)


# --------------------------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------------------------
@dataclass
class LGOutput:
    windows: List[Window]
    token_scores: torch.Tensor      # [W, L]
    valid: torch.Tensor             # [W, L]
    window_scores: torch.Tensor     # [W]
    selected: torch.Tensor          # [k] indices (score desc)
    theta: float                    # k-th largest window score (router scale)
    head_logits: torch.Tensor       # [k] aligned with `selected`
    head_probs: torch.Tensor        # [k]
    verdict_unsafe: bool
    n_deep: int                     # windows that went through the deep head (= constant k)

    def score(self) -> float:
        return float(self.head_probs.max()) if self.head_probs.numel() else 0.0


class LandmarkGuard(nn.Module):
    """Reference two-stage guard: block-local trunk -> router top-k -> per-window head -> max."""

    def __init__(self, model, cfg: LGConfig, router: Optional[Router] = None, head: Optional[Head] = None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        d = _inner(model).config.hidden_size
        self.router = router or Router(d)
        self.head = head or Head(d)

    # ---- stage 1: landmarks + routing (this is the `route()` the evasion runner must use) ----
    @torch.no_grad()
    def landmarks(self, doc_ids: torch.Tensor):
        windows = make_windows(int(doc_ids.numel()), self.cfg.b, self.cfg.stride)
        hidden, valid = trunk_windows(self.model, doc_ids, windows, self.cfg.ell, self.cfg.pad_id)
        token_scores = self.router(hidden)
        window_scores = pool_scores(token_scores, valid, self.cfg.pool)
        return windows, hidden, valid, token_scores, window_scores

    @torch.no_grad()
    def route(self, doc_ids: torch.Tensor):
        """Router-only pass. Returns dict(windows, token_scores, valid, window_scores, selected, theta)."""
        windows, hidden, valid, token_scores, window_scores = self.landmarks(doc_ids)
        selected, theta = route(window_scores, self.cfg.k)
        return dict(windows=windows, hidden=hidden, valid=valid, token_scores=token_scores,
                    window_scores=window_scores, selected=selected, theta=theta)

    # ---- stage 2: deep head on the routed windows, max verdict ----
    @torch.no_grad()
    def forward(self, doc_ids: torch.Tensor) -> LGOutput:
        r = self.route(doc_ids)
        sel = r["selected"]
        logits = head_forward(self.model, r["hidden"][sel], r["valid"][sel], self.cfg.ell, self.head)
        probs = torch.sigmoid(logits)
        return LGOutput(windows=r["windows"], token_scores=r["token_scores"], valid=r["valid"],
                        window_scores=r["window_scores"], selected=sel, theta=r["theta"],
                        head_logits=logits, head_probs=probs,
                        verdict_unsafe=bool(probs.max() > self.cfg.tau) if probs.numel() else False,
                        n_deep=int(sel.numel()))

    # ---- helpers used by tests / attack runner ----
    def needle_windows(self, windows: Sequence[Window], span: Tuple[int, int], max_n: int = 3) -> List[int]:
        """Windows that wholly contain the needle span (<= 3 with the overlapping grid)."""
        return windows_containing(span, windows)[:max_n]


def sliding_scan(lg: LandmarkGuard, doc_ids: torch.Tensor) -> LGOutput:
    """B5 baseline: same head on every window (k = all)."""
    cfg = LGConfig(**{**lg.cfg.to_dict(), "k": 10 ** 9})
    tmp = LandmarkGuard(lg.model, cfg, lg.router, lg.head)
    return tmp(doc_ids)
