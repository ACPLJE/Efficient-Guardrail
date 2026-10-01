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
import time
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
    ell: int = 9                 # trunk depth: layers [0, ell) are block-local; the head continues from this state
    k: int = 32                  # windows routed to the deep head
    pool: str = "max"            # max | mean | top2 | top4 | lse
    tau: float = 0.5             # verdict threshold on head probability
    pad_id: int = 0
    router_ells: Optional[List[int]] = None   # E6-3 (S8): depths whose block-local states feed the router (concatenated
                                              # along d); None = [ell].  The trunk runs to max(ell, router_ells) on every
                                              # window; the head still starts from the depth-`ell` state.

    def to_dict(self) -> Dict:
        """The hashed configuration (lineage cfg_hash of every LG rows file).  `router_ells` is written only when set: the
        default reading must hash exactly as it did before S8 added the field (2026-09-27: an unconditional `router_ells:
        None` changed every cfg_hash and the LG3-8B e2 resume was refused with a LineageMismatch)."""
        d = dict(b=self.b, stride=self.stride, ell=self.ell, k=self.k, pool=self.pool, tau=self.tau, pad_id=self.pad_id)
        if self.router_ells:
            d["router_ells"] = list(self.router_ells)
        return d

    @property
    def router_depths(self) -> List[int]:
        """Sorted unique depths the router reads (the concat order of `router_features`, the same order the trainer used)."""
        return sorted({int(x) for x in self.router_ells}) if self.router_ells else [int(self.ell)]

    @property
    def trunk_depth(self) -> int:
        """Layers every window goes through before routing."""
        return max([int(self.ell)] + self.router_depths)


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


def trunk_windows_multi(model, doc_ids: torch.Tensor, windows: Sequence[Window], ells: Sequence[int], pad_id: int = 0,
                        batch_size: int = 64) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Block-local trunk that returns the hidden state after EVERY depth in `ells` (one pass, segments [0,e1), [e1,e2) ...).
    `{e: hidden [W, L, d]}` -- each entry equals `trunk_windows(..., ell=e)` (segment-wise forward = one forward; tests).
    E6-3 (S8): the router reads the state at other / several depths while the head continues from the depth-`ell` state."""
    ells_sorted = sorted(set(int(e) for e in ells))
    if not ells_sorted:
        raise ValueError("ells must not be empty")
    ids, valid = gather_windows(doc_ids, windows, pad_id)
    outs: Dict[int, List[torch.Tensor]] = {e: [] for e in ells_sorted}
    dec = _inner(model)
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch_size):
            v = valid[i:i + batch_size]
            h = dec.embed_tokens(ids[i:i + batch_size])
            prev = 0
            for e in ells_sorted:
                if e > prev:
                    h = manual_forward(model, inputs_embeds=h, valid=v, layer_range=(prev, e), apply_norm=False)
                outs[e].append(h)
                prev = e
    return {e: torch.cat(xs, 0) for e, xs in outs.items()}, valid


# --------------------------------------------------------------------------------------------
# Router (window-unit landmarks, token-max pooling) and routing
# --------------------------------------------------------------------------------------------
class Router(nn.Module):
    """LayerNorm(in_dim) -> Linear(in_dim, 1) probe on per-token trunk hidden states (~3k params at d=1024).
    `in_dim` > d when several trunk depths are concatenated (LGConfig.router_ells, E6-3)."""

    arch = "linear"

    def __init__(self, d: int, in_dim: Optional[int] = None):
        super().__init__()
        in_dim = int(in_dim or d)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, 1)

    @property
    def in_dim(self) -> int:
        return int(self.proj.in_features)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # [W, L, in_dim] -> [W, L]
        hidden = hidden.to(self.proj.weight.dtype)             # bf16 backbone features -> fp32 probe (S4 checkpoints)
        return self.proj(self.norm(hidden)).squeeze(-1)


class RouterMLP(nn.Module):
    """E6-5 capacity ablation: LayerNorm(in_dim) -> Linear(in_dim, hidden) -> GELU -> Linear(hidden, 1) (~0.26M params at
    d=1024, hidden=256) -- the same per-token score / in-window pooling as `Router`."""

    arch = "mlp"

    def __init__(self, d: int, in_dim: Optional[int] = None, hidden: int = 256):
        super().__init__()
        in_dim = int(in_dim or d)
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, int(hidden))
        self.proj = nn.Linear(int(hidden), 1)

    @property
    def in_dim(self) -> int:
        return int(self.fc1.in_features)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.fc1.weight.dtype)
        return self.proj(F.gelu(self.fc1(self.norm(hidden)))).squeeze(-1)


ROUTER_ARCHS = ("linear", "mlp")


def make_router(d: int, arch: str = "linear", in_dim: Optional[int] = None, hidden: int = 256) -> nn.Module:
    if arch == "linear":
        return Router(d, in_dim)
    if arch == "mlp":
        return RouterMLP(d, in_dim, hidden)
    raise ValueError(f"router arch must be one of {ROUTER_ARCHS}, got {arch!r}")


def router_in_dim(cfg: "LGConfig", d: int) -> int:
    """Input width of the router under `cfg`: d per router depth (concatenated)."""
    return int(d) * len(cfg.router_depths)


def router_features(states: Dict[int, torch.Tensor], cfg: "LGConfig") -> torch.Tensor:
    """The router input from per-depth trunk states: the depth-`ell` state (default) or the concatenation over
    cfg.router_ells along the hidden dimension."""
    depths = cfg.router_depths
    if len(depths) == 1:
        return states[depths[0]]
    return torch.cat([states[e] for e in depths], dim=-1)


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
    """Probe on the last valid token of the deep (full-depth, normed) window representation.

    probe="last"      (v1, default) the last content token of the window is read.
    probe="landmark"  (v2, 2026-09-21) a learned end-of-window vector is appended AFTER the last content token at the
                      trunk depth (layer ell); the deep layers attend from it to the whole window (causal, so the content
                      tokens' states are unchanged) and the probe reads it.  Motivation (S5 framing diagnostic): the v1
                      probe needs 16+ tokens after the needle -- a needle ending at the window end (every needle at the
                      document end, depth 1.0) scored 0.72 vs 0.85, and benign twins showed the same position effect.
                      With the landmark every window is read at the same position with all of its content behind it,
                      like the chat-template suffix the base guard generates its verdict after.  +d parameters, +1
                      position per routed window in the deep layers (~0.4 % of the head FLOPs)."""

    def __init__(self, d: int, probe: str = "last"):
        super().__init__()
        if probe not in ("last", "landmark"):
            raise ValueError(f"probe must be 'last' or 'landmark', got {probe!r}")
        self.probe = probe
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)
        if probe == "landmark":
            self.landmark = nn.Parameter(torch.zeros(d))     # zero init: the first deep layer attends uniformly (a mean query)

    def inject(self, hidden: torch.Tensor, valid: torch.Tensor):
        """probe='landmark': append the landmark vector after the last valid token, [W, L, d] -> [W, L+1, d] (valid too).
        probe='last': identity."""
        if self.probe != "landmark":
            return hidden, valid
        W, L, d = hidden.shape
        h2 = torch.cat([hidden, hidden.new_zeros(W, 1, d)], dim=1)
        v2 = torch.cat([valid, valid.new_zeros(W, 1)], dim=1)
        n = valid.sum(1).clamp(max=L).long()                   # insertion index = number of valid content tokens
        idx = torch.arange(W, device=hidden.device)
        h2 = h2.index_put((idx, n), self.landmark.to(h2.dtype).expand(W, d))
        v2[idx, n] = 1
        return h2, v2

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:  # -> [W] logits
        last = valid.sum(1).clamp(min=1) - 1
        h = hidden[torch.arange(hidden.shape[0]), last].to(self.proj.weight.dtype)
        return self.proj(self.norm(h)).squeeze(-1)


def head_forward(model, trunk_hidden: torch.Tensor, valid: torch.Tensor, ell: int, head: Head) -> torch.Tensor:
    """Continue layers [ell, N) from the trunk hidden states of the selected windows, then score
    (probe='landmark': the landmark vector is appended to every window before the deep layers)."""
    with torch.no_grad():
        trunk_hidden, valid = head.inject(trunk_hidden, valid)
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


class PhaseTimer:
    """Accumulates wall time per phase (tokenize / trunk / router / head / aggregate) for the E4-4 breakdown.  On CUDA
    every phase boundary synchronises, so use it for the breakdown measurement only, never inside a latency run."""

    def __init__(self, enabled: bool = True, device: Optional[str] = None):
        self.enabled = enabled
        self.device = device
        self.ms: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}

    def _sync(self):
        if self.device and str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    class _Phase:
        def __init__(self, outer: "PhaseTimer", name: str):
            self.outer, self.name = outer, name

        def __enter__(self):
            if self.outer.enabled:
                self.outer._sync()
                self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if self.outer.enabled:
                self.outer._sync()
                dt = (time.perf_counter() - self.t0) * 1000.0
                self.outer.ms[self.name] = self.outer.ms.get(self.name, 0.0) + dt
                self.outer.calls[self.name] = self.outer.calls.get(self.name, 0) + 1
            return False

    def phase(self, name: str) -> "_Phase":
        return PhaseTimer._Phase(self, name)

    def total(self) -> float:
        return float(sum(self.ms.values()))

    def reset(self) -> "PhaseTimer":
        self.ms, self.calls = {}, {}
        return self


def _pad_windows_to(h1: torch.Tensor, v1: torch.Tensor, h2: torch.Tensor, v2: torch.Tensor):
    """Concatenate two window batches whose padded lengths may differ (only when a document is shorter than one window)."""
    L = max(h1.shape[1], h2.shape[1])
    if h1.shape[1] != L:
        h1 = F.pad(h1, (0, 0, 0, L - h1.shape[1]))
        v1 = F.pad(v1, (0, L - v1.shape[1]))
    if h2.shape[1] != L:
        h2 = F.pad(h2, (0, 0, 0, L - h2.shape[1]))
        v2 = F.pad(v2, (0, L - v2.shape[1]))
    return torch.cat([h1, h2], 0), torch.cat([v1, v2], 0)


def _cat_padded(ts: List[torch.Tensor], vs: List[torch.Tensor]):
    if not ts:
        return torch.zeros(0, 0), torch.zeros(0, 0, dtype=torch.long)
    L = max(t.shape[1] for t in ts)
    ts = [F.pad(t, (0, L - t.shape[1])) if t.shape[1] != L else t for t in ts]
    vs = [F.pad(v, (0, L - v.shape[1])) if v.shape[1] != L else v for v in vs]
    return torch.cat(ts, 0), torch.cat(vs, 0)


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
    def _trunk(self, doc_ids: torch.Tensor, windows: Sequence[Window], batch_size: int = 64):
        """Block-local trunk on `windows` -> (head state at depth ell [W, L, d], router input [W, L, in_dim], valid).
        With the default cfg (router_ells None) the two states are the same tensor and nothing extra is computed."""
        depths = sorted(set([int(self.cfg.ell)] + self.cfg.router_depths))
        if depths == [int(self.cfg.ell)]:
            h, v = trunk_windows(self.model, doc_ids, windows, self.cfg.ell, self.cfg.pad_id, batch_size=batch_size)
            return h, h, v
        states, v = trunk_windows_multi(self.model, doc_ids, windows, depths, self.cfg.pad_id, batch_size=batch_size)
        return states[int(self.cfg.ell)], router_features(states, self.cfg), v

    @torch.no_grad()
    def landmarks(self, doc_ids: torch.Tensor):
        windows = make_windows(int(doc_ids.numel()), self.cfg.b, self.cfg.stride)
        hidden, h_router, valid = self._trunk(doc_ids, windows)
        token_scores = self.router(h_router)
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
    def forward(self, doc_ids: torch.Tensor, stream: bool = False, chunk_windows: int = 64,
                timer: Optional["PhaseTimer"] = None) -> LGOutput:
        """stream=False: every window's trunk state is kept ([W, 2b, d]) and the top-k are gathered afterwards.
        stream=True: `forward_stream` -- windows go through the trunk in chunks and only the running top-k states are
        kept, so peak memory does not grow with L (S6 64k/128k rows, E4-3).  Both return the same verdict."""
        if stream:
            return self.forward_stream(doc_ids, chunk_windows=chunk_windows, timer=timer)
        tm = timer or PhaseTimer(enabled=False)
        with tm.phase("trunk"):
            windows = make_windows(int(doc_ids.numel()), self.cfg.b, self.cfg.stride)
            hidden, h_router, valid = self._trunk(doc_ids, windows)
        with tm.phase("router"):
            token_scores = self.router(h_router)
            window_scores = pool_scores(token_scores, valid, self.cfg.pool)
            sel, theta = route(window_scores, self.cfg.k)
        with tm.phase("head"):
            logits = head_forward(self.model, hidden[sel], valid[sel], self.cfg.ell, self.head)
        with tm.phase("aggregate"):
            probs = torch.sigmoid(logits)
            verdict = bool(probs.max() > self.cfg.tau) if probs.numel() else False
        return LGOutput(windows=windows, token_scores=token_scores, valid=valid, window_scores=window_scores, selected=sel,
                        theta=theta, head_logits=logits, head_probs=probs, verdict_unsafe=verdict, n_deep=int(sel.numel()))

    @torch.no_grad()
    def forward_stream(self, doc_ids: torch.Tensor, chunk_windows: int = 64, timer: Optional["PhaseTimer"] = None,
                       skipping: bool = True, pad_to_k: bool = False, head_batch: Optional[int] = None) -> LGOutput:
        """Window-streaming forward: the trunk runs on `chunk_windows` windows at a time, the router scores them, and
        a running top-k buffer keeps only the k best trunk states (plus every window's scores, which are O(W)).  The
        deep head then reads the buffer, so the verdict equals `forward(stream=False)` (up to top-k tie order).
        `head_batch` runs the deep head on that many buffered windows at a time (windows are independent, so the
        logits are identical); it bounds the head's transient activations the way `chunk_windows` bounds the trunk's
        (E4-3 / G2 (c), 9/26: with chunk 8 the 1k -> 128k peak growth still had a +200 MB floor = the k=32-window head batch).

        Cost-model switches (E4 only, never used for accuracy rows):
          skipping=False  every window also runs the deep layers [ell, N) (no early exit) -- the FLOPs of a guard that
                          cannot skip; the head still reads the routed windows (E6-13 measures the saving).
          pad_to_k=True   when the document has fewer than k windows, the head batch is padded with copies up to k
                          windows (constant deep work at every L; the copies are dropped from the output)."""
        tm = timer or PhaseTimer(enabled=False)
        n = int(doc_ids.numel())
        windows = make_windows(n, self.cfg.b, self.cfg.stride)
        W = len(windows)
        k = max(1, min(int(self.cfg.k), W))
        n_layers = len(_inner(self.model).layers)
        all_scores = torch.empty(W, dtype=torch.float32, device=doc_ids.device)
        tok_chunks: List[torch.Tensor] = []
        valid_chunks: List[torch.Tensor] = []
        buf_h = buf_v = buf_s = buf_i = None
        for c0 in range(0, W, max(1, int(chunk_windows))):
            wins = windows[c0:c0 + chunk_windows]
            with tm.phase("trunk"):
                h, hr, v = self._trunk(doc_ids, wins, batch_size=len(wins))
            if not skipping:
                with tm.phase("head"):                          # full depth on every window: the no-skip cost model
                    manual_forward(self.model, inputs_embeds=h, valid=v, layer_range=(self.cfg.ell, n_layers), apply_norm=True)
            with tm.phase("router"):
                ts = self.router(hr)
                ws = pool_scores(ts, v, self.cfg.pool).float()
                all_scores[c0:c0 + len(wins)] = ws
                tok_chunks.append(ts)
                valid_chunks.append(v)
                idx = torch.arange(c0, c0 + len(wins), device=doc_ids.device)
                if buf_h is None:
                    cand = (h, v, ws, idx)
                else:
                    h2, v2 = _pad_windows_to(buf_h, buf_v, h, v)
                    cand = (h2, v2, torch.cat([buf_s, ws]), torch.cat([buf_i, idx]))
                if cand[2].numel() > k:
                    keep = torch.topk(cand[2], k).indices
                    cand = tuple(t[keep] for t in cand)
                buf_h, buf_v, buf_s, buf_i = cand
        with tm.phase("router"):
            order = torch.argsort(buf_s, descending=True)
            buf_h, buf_v, buf_s, buf_i = buf_h[order], buf_v[order], buf_s[order], buf_i[order]
            theta = float(buf_s[-1])
        with tm.phase("head"):
            hh, vv = buf_h, buf_v
            n_real = hh.shape[0]
            if pad_to_k and n_real < int(self.cfg.k):
                reps = -(-int(self.cfg.k) // n_real)
                hh, vv = hh.repeat(reps, 1, 1)[: int(self.cfg.k)], vv.repeat(reps, 1)[: int(self.cfg.k)]
            hb = max(1, int(head_batch)) if head_batch else hh.shape[0]
            if hb >= hh.shape[0]:
                logits = head_forward(self.model, hh, vv, self.cfg.ell, self.head)[:n_real]
            else:
                logits = torch.cat([head_forward(self.model, hh[i:i + hb], vv[i:i + hb], self.cfg.ell, self.head)
                                    for i in range(0, hh.shape[0], hb)], 0)[:n_real]
        with tm.phase("aggregate"):
            probs = torch.sigmoid(logits)
            verdict = bool(probs.max() > self.cfg.tau) if probs.numel() else False
        token_scores, valid = _cat_padded(tok_chunks, valid_chunks)
        return LGOutput(windows=windows, token_scores=token_scores, valid=valid, window_scores=all_scores, selected=buf_i,
                        theta=theta, head_logits=logits, head_probs=probs, verdict_unsafe=verdict, n_deep=int(n_real))

    @torch.no_grad()
    def forward_batch(self, docs: Sequence[torch.Tensor], chunk_windows: int = 64, head_batch: int = 64,
                      timer: Optional["PhaseTimer"] = None) -> List[LGOutput]:
        """Several documents at once (E4-2 throughput): each document streams through the trunk + router on its own
        (memory stays flat), then the routed windows of ALL documents go through the deep head in batches of
        `head_batch` windows.  Verdicts equal `forward_stream` per document."""
        tm = timer or PhaseTimer(enabled=False)
        parts = []
        for ids in docs:
            o = self._route_stream(ids, chunk_windows, tm)
            parts.append(o)
        hs = [p["buf_h"] for p in parts]
        vs = [p["buf_v"] for p in parts]
        L = max(h.shape[1] for h in hs)
        hs = [torch.nn.functional.pad(h, (0, 0, 0, L - h.shape[1])) for h in hs]
        vs = [torch.nn.functional.pad(v, (0, L - v.shape[1])) for v in vs]
        H, V = torch.cat(hs, 0), torch.cat(vs, 0)
        with tm.phase("head"):
            logits = torch.cat([head_forward(self.model, H[i:i + head_batch], V[i:i + head_batch], self.cfg.ell, self.head)
                                for i in range(0, H.shape[0], head_batch)], 0) if H.shape[0] else H.new_zeros(0)
        outs = []
        off = 0
        with tm.phase("aggregate"):
            for p in parts:
                m = p["buf_h"].shape[0]
                lg = logits[off:off + m]
                off += m
                probs = torch.sigmoid(lg)
                outs.append(LGOutput(windows=p["windows"], token_scores=p["token_scores"], valid=p["valid"], window_scores=p["scores"],
                                     selected=p["buf_i"], theta=p["theta"], head_logits=lg, head_probs=probs,
                                     verdict_unsafe=bool(probs.max() > self.cfg.tau) if probs.numel() else False, n_deep=int(m)))
        return outs

    @torch.no_grad()
    def _route_stream(self, doc_ids: torch.Tensor, chunk_windows: int, tm: "PhaseTimer") -> Dict:
        n = int(doc_ids.numel())
        windows = make_windows(n, self.cfg.b, self.cfg.stride)
        W = len(windows)
        k = max(1, min(int(self.cfg.k), W))
        all_scores = torch.empty(W, dtype=torch.float32, device=doc_ids.device)
        tok_chunks, valid_chunks = [], []
        buf_h = buf_v = buf_s = buf_i = None
        for c0 in range(0, W, max(1, int(chunk_windows))):
            wins = windows[c0:c0 + chunk_windows]
            with tm.phase("trunk"):
                h, hr, v = self._trunk(doc_ids, wins, batch_size=len(wins))
            with tm.phase("router"):
                ts = self.router(hr)
                ws = pool_scores(ts, v, self.cfg.pool).float()
                all_scores[c0:c0 + len(wins)] = ws
                tok_chunks.append(ts)
                valid_chunks.append(v)
                idx = torch.arange(c0, c0 + len(wins), device=doc_ids.device)
                if buf_h is None:
                    cand = (h, v, ws, idx)
                else:
                    h2, v2 = _pad_windows_to(buf_h, buf_v, h, v)
                    cand = (h2, v2, torch.cat([buf_s, ws]), torch.cat([buf_i, idx]))
                if cand[2].numel() > k:
                    keep = torch.topk(cand[2], k).indices
                    cand = tuple(t[keep] for t in cand)
                buf_h, buf_v, buf_s, buf_i = cand
        order = torch.argsort(buf_s, descending=True)
        token_scores, valid = _cat_padded(tok_chunks, valid_chunks)
        return dict(windows=windows, scores=all_scores, token_scores=token_scores, valid=valid, buf_h=buf_h[order], buf_v=buf_v[order],
                    buf_s=buf_s[order], buf_i=buf_i[order], theta=float(buf_s[order][-1]))

    # ---- helpers used by tests / attack runner ----
    def needle_windows(self, windows: Sequence[Window], span: Tuple[int, int], max_n: int = 3) -> List[int]:
        """Windows that wholly contain the needle span (<= 3 with the overlapping grid)."""
        return windows_containing(span, windows)[:max_n]


@torch.no_grad()
def sliding_scan(lg: LandmarkGuard, doc_ids: torch.Tensor, chunk_windows: int = 64, timer: Optional[PhaseTimer] = None) -> LGOutput:
    """B5 baseline: the same head on EVERY window (k = all).  Streams window chunks through trunk -> router -> head, so
    the memory stays flat while the deep work grows with L (that is the point of the comparison, E4-1/E4-3).
    `selected` lists every window in score order; head_probs are aligned with it."""
    tm = timer or PhaseTimer(enabled=False)
    n = int(doc_ids.numel())
    windows = make_windows(n, lg.cfg.b, lg.cfg.stride)
    W = len(windows)
    scores = torch.empty(W, dtype=torch.float32, device=doc_ids.device)
    logits_all = torch.empty(W, dtype=torch.float32, device=doc_ids.device)
    tok_chunks, valid_chunks = [], []
    for c0 in range(0, W, max(1, int(chunk_windows))):
        wins = windows[c0:c0 + chunk_windows]
        with tm.phase("trunk"):
            h, hr, v = lg._trunk(doc_ids, wins, batch_size=len(wins))
        with tm.phase("router"):
            ts = lg.router(hr)
            scores[c0:c0 + len(wins)] = pool_scores(ts, v, lg.cfg.pool).float()
            tok_chunks.append(ts)
            valid_chunks.append(v)
        with tm.phase("head"):
            logits_all[c0:c0 + len(wins)] = head_forward(lg.model, h, v, lg.cfg.ell, lg.head).float()
    with tm.phase("aggregate"):
        order = torch.argsort(scores, descending=True)
        logits = logits_all[order]
        probs = torch.sigmoid(logits)
        verdict = bool(probs.max() > lg.cfg.tau) if probs.numel() else False
    token_scores, valid = _cat_padded(tok_chunks, valid_chunks)
    return LGOutput(windows=windows, token_scores=token_scores, valid=valid, window_scores=scores, selected=order,
                    theta=float(scores[order][-1]) if W else 0.0, head_logits=logits, head_probs=probs, verdict_unsafe=verdict, n_deep=W)
