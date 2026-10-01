"""Backbone loading, freezing policy, training forwards and checkpoint I/O shared by the S4 trainers.

Freezing (plan §3): the block-local trunk = layers [0, ell) and the embeddings are frozen for the LG head; the deep
layers [ell, N) + final norm are fine-tuned in fp32 master weights (mode=full) or through LoRA (mode=lora); the probe
(lgs1.lgcore.Head = LayerNorm + Linear on the last valid token) is always trained.  Frozen parts stay bf16.
Forwards run under bf16 autocast.  The trunk pass is no-grad and identical to lgcore.trunk_windows (E0-1: the router
trained later on the FT checkpoint sees exactly the base model's shallow features).
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from lgs1.lgcore.model import Head, LGConfig, Router, _inner, build_attn_mask, router_in_dim
from lgs1.lineage import sha256_file
from lgs4 import lora as LORA
from lgs4.config import BACKBONES, Backbone, WINDOW_B

HEAD_FILE, ROUTER_FILE, DENSE_FILE = "head.pt", "router.pt", "dense_ft.pt"


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------
YARN_FACTOR = 4.0


def apply_yarn(cfg, factor: float = YARN_FACTOR):
    """YaRN x`factor` on a HF config in place (transformers 4.x `rope_scaling` + `rope_theta`, 5.x `rope_parameters` that
    carries rope_theta itself -- dropping it made the inverse frequencies `None ** x`, 2026-09-22).  A text_config
    (VLM wrappers) is handled through the wrapper."""
    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        apply_yarn(tc, factor)
        return cfg
    orig = int(getattr(cfg, "max_position_embeddings", 32768) or 32768)
    rp = getattr(cfg, "rope_parameters", None)
    old = dict(rp) if isinstance(rp, dict) else dict(getattr(cfg, "rope_scaling", None) or {})
    if old.get("rope_type") == "yarn":
        return cfg
    new = {"rope_type": "yarn", "factor": float(factor), "original_max_position_embeddings": orig}
    if isinstance(rp, dict):                                   # transformers >= 5: theta lives in the same dict
        theta = old.get("rope_theta", getattr(cfg, "rope_theta", None))
        if theta is not None:
            new["rope_theta"] = theta
        cfg.rope_parameters = new
    else:
        cfg.rope_scaling = new
    cfg.max_position_embeddings = int(orig * factor)
    return cfg


def rope_config(repo: str, rope: Optional[str]):
    """Config override for a rope policy: `yarn` = YaRN factor 4 over the model's native window (32k -> 128k for the Qwen3
    family; B4 dense-FT 64k/128k rows and the dense zero-shot E4 rows), None = unchanged."""
    if not rope or rope == "native":
        return None
    if rope != "yarn":
        raise ValueError(f"rope policy must be yarn or native, got {rope!r}")
    from transformers import AutoConfig
    return apply_yarn(AutoConfig.from_pretrained(repo))


def load_backbone(bb: Backbone | str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, attn: str = "sdpa",
                  rope: Optional[str] = None):
    """Backbone (causal LM) + tokenizer from the local HF cache (offline on compute nodes).  `rope="yarn"` extends the
    context x4 (rope_config) -- inference only."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = bb.repo if isinstance(bb, Backbone) else bb
    tok = AutoTokenizer.from_pretrained(repo)
    kw = {"dtype": dtype} if _tf_ge("4.56") else {"torch_dtype": dtype}
    cfg = rope_config(repo, rope)
    if cfg is not None:
        kw["config"] = cfg
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, attn_implementation=attn, **kw)
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _tf_ge(v: str) -> bool:
    import transformers
    from packaging.version import Version
    return Version(transformers.__version__) >= Version(v)


def n_layers(model) -> int:
    return len(_inner(model).layers)


def hidden_size(model) -> int:
    return int(_inner(model).config.hidden_size)


def pad_id_of(model, tok) -> int:
    pid = getattr(tok, "pad_token_id", None)
    if pid is None:
        pid = getattr(tok, "eos_token_id", None)
    return int(pid if pid is not None else 0)


# ------------------------------------------------------------------------------------------------
# trainable parameter policy
# ------------------------------------------------------------------------------------------------
def deep_layer_indices(model, ell: int) -> List[int]:
    return list(range(ell, n_layers(model)))


def prepare_lg_trainable(model, bb: Backbone, ell: int) -> Dict[str, List[nn.Parameter]]:
    """Make layers [ell, N) (+ final norm) trainable.  Returns param groups by learning-rate key."""
    dec = _inner(model)
    groups: Dict[str, List[nn.Parameter]] = {"lr_deep": [], "lr_lora": []}
    if bb.mode == "full":
        for li in deep_layer_indices(model, ell):
            dec.layers[li].float()
            for p in dec.layers[li].parameters():
                p.requires_grad_(True)
                groups["lr_deep"].append(p)
        dec.norm.float()
        for p in dec.norm.parameters():
            p.requires_grad_(True)
            groups["lr_deep"].append(p)
    elif bb.mode == "lora":
        LORA.inject_lora(model, deep_layer_indices(model, ell), r=bb.lora_r, alpha=bb.lora_alpha)
        groups["lr_lora"] = LORA.lora_parameters(model)
    else:
        raise ValueError(bb.mode)
    return groups


def prepare_dense_trainable(model, bb: Backbone) -> Dict[str, List[nn.Parameter]]:
    """dense-FT: every decoder layer (+ norm) trainable (full) or LoRA on every layer."""
    dec = _inner(model)
    groups: Dict[str, List[nn.Parameter]] = {"lr_deep": [], "lr_lora": []}
    if bb.dense_mode == "full":
        for layer in dec.layers:
            layer.float()
            for p in layer.parameters():
                p.requires_grad_(True)
                groups["lr_deep"].append(p)
        dec.norm.float()
        for p in dec.norm.parameters():
            p.requires_grad_(True)
            groups["lr_deep"].append(p)
    else:
        LORA.inject_lora(model, range(n_layers(model)), r=bb.lora_r, alpha=bb.lora_alpha)
        groups["lr_lora"] = LORA.lora_parameters(model)
    return groups


def count_params(params: Iterable[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


# ------------------------------------------------------------------------------------------------
# forwards
# ------------------------------------------------------------------------------------------------
def run_layers(model, h: torch.Tensor, valid: torch.Tensor, start: int, end: int, apply_norm: bool,
               grad_ckpt: bool = False, causal: bool = True) -> torch.Tensor:
    """Decoder layers [start, end) on hidden states h [B, T, d] with positions reset to 0 (block-local windows).
    Same maths as lgs1.lgcore.manual_forward, with optional per-layer gradient checkpointing."""
    dec = _inner(model)
    B, T, _ = h.shape
    position_ids = torch.arange(T, device=h.device)[None].expand(B, T)
    mask = build_attn_mask(valid, causal, h.dtype)
    pos_emb = dec.rotary_emb(h, position_ids) if hasattr(dec, "rotary_emb") else None
    end = min(end, len(dec.layers))
    ac = _autocast_on(h.device.type)
    for i in range(start, end):
        layer = dec.layers[i]
        p_dtype = next(layer.parameters()).dtype
        if not ac and h.dtype != p_dtype:            # no autocast (CPU tests): keep dtypes consistent by hand
            h, mask = h.to(p_dtype), mask.to(p_dtype)

        def fn(x, layer=layer):
            kw = dict(attention_mask=mask, position_ids=position_ids)
            if pos_emb is not None:
                kw["position_embeddings"] = pos_emb
            out = layer(x, **kw)
            return out[0] if isinstance(out, (tuple, list)) else out

        if grad_ckpt and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(fn, h, use_reentrant=False)
        else:
            h = fn(h)
    if apply_norm:
        if not ac and h.dtype != dec.norm.weight.dtype:
            h = h.to(dec.norm.weight.dtype)
        h = dec.norm(h)
    return h


def _autocast_on(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:                                    # older torch: no device argument
        return bool(torch.is_autocast_enabled()) if device_type == "cuda" else bool(torch.is_autocast_cpu_enabled())


def lg_window_logits(model, head: Head, ids: torch.Tensor, valid: torch.Tensor, ell: int, grad_ckpt: bool = False,
                     autocast: bool = True) -> torch.Tensor:
    """Windows [B, 2b] -> head logits [B]: frozen trunk (no grad) -> deep layers (grad) -> probe."""
    dec = _inner(model)
    dev_type = "cuda" if ids.is_cuda else "cpu"
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=autocast and dev_type == "cuda"):
        with torch.no_grad():
            h = dec.embed_tokens(ids)
            h = run_layers(model, h, valid, 0, ell, apply_norm=False)
        h, valid = head.inject(h, valid)                     # probe='landmark': trainable end-of-window vector (grad flows)
        h = run_layers(model, h, valid, ell, n_layers(model), apply_norm=True, grad_ckpt=grad_ckpt)
        logits = head(h.float(), valid)
    return logits.float()


def dense_logits(model, head: Head, ids: torch.Tensor, valid: torch.Tensor, autocast: bool = True) -> torch.Tensor:
    """Full-context forward (HF, causal, right padding) -> probe on the last valid token.  An unpadded batch passes no
    attention mask: HF then uses the fused causal kernel instead of materialising a [B, 1, L, L] mask (32 GB at 128k)."""
    dev_type = "cuda" if ids.is_cuda else "cpu"
    dec = _inner(model)
    mask = None if bool(valid.all()) else valid
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=autocast and dev_type == "cuda"):
        out = dec(input_ids=ids, attention_mask=mask, use_cache=False)
        h = out.last_hidden_state
        logits = head(h.float(), valid)
    return logits.float()


DENSE_PREFILL_MIN = 32768         # single rows longer than this go through the chunked prefill (dense-FT yarn 64k/128k)
DENSE_PREFILL_CHUNK = 2048


def dense_logits_chunked(model, head: Head, ids: torch.Tensor, chunk: int = DENSE_PREFILL_CHUNK, autocast: bool = True) -> torch.Tensor:
    """One unpadded row [1, L] -> probe logit [1], prefilling `chunk` tokens at a time into a KV cache.  Attention memory is
    O(chunk x L) per layer instead of the [L, L] mask / L^2 weights that a single-pass forward needs whenever the fused
    causal kernel is not taken (2026-09-23: dense-FT YaRN rows OOMed at 128k with a 28 GB [L, L] mask on a 48 GB card).
    The final hidden state of the last chunk is the same last-token state the single pass would produce."""
    from lgs1.zeroshot.prefill import _cache
    if ids.shape[0] != 1:
        raise ValueError("dense_logits_chunked takes one row at a time")
    dev_type = "cuda" if ids.is_cuda else "cpu"
    dec = _inner(model)
    cache = _cache(model)
    L = int(ids.shape[1])
    last = None
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=autocast and dev_type == "cuda"):
        for s in range(0, L, max(1, int(chunk))):
            out = dec(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last = out.last_hidden_state[:, -1:]
        valid = torch.ones((1, 1), dtype=torch.long, device=ids.device)
        logits = head(last.float(), valid)
    del cache
    return logits.float()


def dense_logits_auto(model, head: Head, ids: torch.Tensor, valid: torch.Tensor, prefill_min: int = DENSE_PREFILL_MIN,
                      chunk: int = DENSE_PREFILL_CHUNK, autocast: bool = True) -> torch.Tensor:
    """dense_logits for batches / short rows, dense_logits_chunked for a single row longer than `prefill_min` tokens."""
    if ids.shape[0] == 1 and int(valid.sum()) > int(prefill_min) and bool(valid.all()):
        return dense_logits_chunked(model, head, ids, chunk, autocast)
    return dense_logits(model, head, ids, valid, autocast)


# ------------------------------------------------------------------------------------------------
# checkpoints
# ------------------------------------------------------------------------------------------------
def trainable_state(model, bb: Backbone, ell: Optional[int], dense: bool = False,
                    dtype: Optional[torch.dtype] = torch.bfloat16) -> Dict[str, torch.Tensor]:
    """State of the trained backbone parts (bf16 for full FT, adapter tensors for LoRA).  `dtype=None` keeps the
    modules' own dtype (fp32 master weights) -- used by the resume checkpoints."""
    mode = bb.dense_mode if dense else bb.mode
    if mode == "lora":
        return LORA.lora_state_dict(model)
    dec = _inner(model)
    sd: Dict[str, torch.Tensor] = {}
    idx = range(n_layers(model)) if dense else deep_layer_indices(model, ell or 0)

    def cp(v):
        return (v.detach().to(dtype) if dtype is not None else v.detach()).cpu()
    for li in idx:
        for k, v in dec.layers[li].state_dict().items():
            sd[f"layers.{li}.{k}"] = cp(v)
    for k, v in dec.norm.state_dict().items():
        sd[f"norm.{k}"] = cp(v)
    return sd


def apply_trainable_state(model, bb: Backbone, sd: Dict[str, torch.Tensor], ell: Optional[int], dense: bool = False) -> int:
    """Load a `trainable_state` dict into the model (cast to each module's current device/dtype).  Returns #tensors."""
    mode = bb.dense_mode if dense else bb.mode
    if mode == "lora":
        idx = range(n_layers(model)) if dense else deep_layer_indices(model, ell or 0)
        if not LORA.lora_modules(model):
            LORA.inject_lora(model, idx, r=bb.lora_r, alpha=bb.lora_alpha, dropout=0.0)
        return LORA.load_lora_state(model, sd)
    dec = _inner(model)
    per_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    norm_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("layers."):
            _, li, rest = k.split(".", 2)
            per_layer.setdefault(int(li), {})[rest] = v
        elif k.startswith("norm."):
            norm_sd[k.split(".", 1)[1]] = v
    n = 0
    for li, sub in per_layer.items():
        layer = dec.layers[li]
        ref = next(layer.parameters())
        layer.load_state_dict({k: v.to(ref.device, ref.dtype) for k, v in sub.items()}, strict=True)
        n += len(sub)
    if norm_sd:
        ref = dec.norm.weight
        dec.norm.load_state_dict({k: v.to(ref.device, ref.dtype) for k, v in norm_sd.items()}, strict=True)
        n += len(norm_sd)
    return n


def save_head_ckpt(path: Path, model, head: Head, bb: Backbone, cfg: LGConfig, hp: Dict, meta: Dict) -> Path:
    obj = {"kind": "lg_head", "backbone": bb.name, "repo": bb.repo, "mode": bb.mode, "ell": cfg.ell, "cfg": cfg.to_dict(),
           "head_probe": getattr(head, "probe", "last"),
           "trained": trainable_state(model, bb, cfg.ell), "probe": {k: v.detach().cpu() for k, v in head.state_dict().items()},
           "hp": hp, "meta": meta, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def save_dense_ckpt(path: Path, model, head: Head, bb: Backbone, hp: Dict, meta: Dict) -> Path:
    obj = {"kind": "dense_ft", "backbone": bb.name, "repo": bb.repo, "mode": bb.dense_mode, "max_tokens": bb.dense_max_tokens,
           "trained": trainable_state(model, bb, None, dense=True), "probe": {k: v.detach().cpu() for k, v in head.state_dict().items()},
           "hp": hp, "meta": meta, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def save_router_ckpt(path: Path, router: nn.Module, bb: Backbone, cfg: LGConfig, head_path: Path, hp: Dict, meta: Dict) -> Path:
    # cfg carries pool / router_ells (E6-3/4 ablations: the router's own reading of the trunk); arch + in_dim rebuild the module
    obj = {"kind": "lg_router", "backbone": bb.name, "repo": bb.repo, "ell": cfg.ell, "cfg": cfg.to_dict(),
           "arch": getattr(router, "arch", "linear"), "in_dim": int(getattr(router, "in_dim", 0)) or None,
           "hidden": int(router.fc1.out_features) if hasattr(router, "fc1") else None,
           "state": {k: v.detach().cpu() for k, v in router.state_dict().items()},
           "head_sha": sha256_file(head_path) if head_path.exists() else None, "hp": hp, "meta": meta,
           "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def load_ckpt(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


RESUME_SUFFIX = ".resume.pt"


def save_resume(path: Path, model, bb: Backbone, ell: Optional[int], head: nn.Module, opt, sched, state: Dict,
                dense: bool = False) -> Path:
    """Mid-training checkpoint (fp32 master weights + optimizer + scheduler + RNG + position) for `load_resume`.
    2026-09-19: a 0.6B head run is 58k steps; the 12-24 h job limits and pre-emption would otherwise lose everything."""
    obj = {"kind": "resume", "trained": trainable_state(model, bb, ell, dense=dense, dtype=None),
           "probe": {k: v.detach().cpu() for k, v in head.state_dict().items()},
           "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state,
           "rng": {"python": __import__("random").getstate(), "torch": torch.get_rng_state(),
                   "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
           "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def load_resume(path: Path, model, bb: Backbone, ell: Optional[int], head: nn.Module, opt, sched, dense: bool = False) -> Dict:
    """Restore a `save_resume` checkpoint into live objects; returns its `state` (step / epoch / batch_idx ...)."""
    ck = load_ckpt(path)
    if ck.get("kind") != "resume":
        raise SystemExit(f"{path}: not a resume checkpoint")
    apply_trainable_state(model, bb, ck["trained"], ell, dense=dense)
    head.load_state_dict(ck["probe"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    try:
        __import__("random").setstate(ck["rng"]["python"])
        torch.set_rng_state(ck["rng"]["torch"])
        if ck["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    except Exception as e:                                          # noqa: BLE001  (different device count: fine)
        print(f"[resume] RNG state not restored: {e}")
    return ck["state"]


def load_lg(run_dir: Path, device: str = "cuda", k: int = 32, repo_override: Optional[str] = None,
            with_router: bool = True, dtype: torch.dtype = torch.bfloat16, attn: str = "sdpa") -> Tuple[object, object, Head, Optional[Router], LGConfig, Dict]:
    """Rebuild (model, tok, head, router, cfg, info) from runs/v2/<backbone>_s<seed>/ for evaluation (S5+).
    The backbone is bf16 (or `dtype`); probe and router stay fp32 (they cast their inputs)."""
    run_dir = Path(run_dir)
    hc = load_ckpt(run_dir / HEAD_FILE)
    bb = BACKBONES.get(hc["backbone"]) or Backbone(hc["backbone"], hc["repo"], hc["ell"], mode=hc["mode"])
    model, tok = load_backbone(repo_override or bb.repo, device=device, dtype=dtype, attn=attn)
    apply_trainable_state(model, bb, hc["trained"], hc["ell"])
    model.eval()
    head = Head(hidden_size(model), probe=hc.get("head_probe", "landmark" if "landmark" in hc["probe"] else "last")).to(device)
    head.load_state_dict(hc["probe"])
    head.eval()
    cfg = LGConfig(**{**hc["cfg"], "k": k, "pad_id": pad_id_of(model, tok)})
    router = None
    if with_router and (run_dir / ROUTER_FILE).exists():
        rc = load_ckpt(run_dir / ROUTER_FILE)
        router, cfg = router_from_ckpt(rc, hidden_size(model), cfg, device)
    info = {"head_meta": hc.get("meta", {}), "hp": hc.get("hp", {}), "run_dir": str(run_dir)}
    return model, tok, head, router, cfg, info


def router_from_ckpt(rc: Dict, d: int, cfg: LGConfig, device: str = "cpu") -> Tuple[nn.Module, LGConfig]:
    """Rebuild the router of a `save_router_ckpt` checkpoint and merge its trunk reading into `cfg`: the router decides
    the in-window pooling and the depths it reads (E6-3/4/5 ablations train those; the head checkpoint's cfg carries the
    defaults).  Older checkpoints (no arch / cfg extras) load as the linear depth-`ell` max-pool router."""
    from lgs1.lgcore.model import make_router
    rcfg = rc.get("cfg") or {}
    cfg = dataclasses.replace(cfg, pool=rcfg.get("pool", cfg.pool), router_ells=rcfg.get("router_ells") or None)
    in_dim = rc.get("in_dim") or router_in_dim(cfg, d)
    router = make_router(d, rc.get("arch", "linear"), in_dim, rc.get("hidden") or 256).to(device)
    router.load_state_dict(rc["state"])
    router.eval()
    return router, cfg


def load_dense(run_dir: Path, device: str = "cuda", repo_override: Optional[str] = None, dtype: torch.dtype = torch.bfloat16,
               rope: Optional[str] = None):
    run_dir = Path(run_dir)
    dc = load_ckpt(run_dir / DENSE_FILE)
    bb = BACKBONES.get(dc["backbone"]) or Backbone(dc["backbone"], dc["repo"], 0, dense_mode=dc["mode"])
    model, tok = load_backbone(repo_override or bb.repo, device=device, dtype=dtype, rope=rope)
    apply_trainable_state(model, bb, dc["trained"], None, dense=True)
    model.eval()
    head = Head(hidden_size(model)).to(device)
    head.load_state_dict(dc["probe"])
    head.eval()
    return model, tok, head, dc


def write_hashes(run_dir: Path, extra: Optional[Dict] = None) -> Path:
    """hashes.json = sha256 of every checkpoint in the run dir + data-manifest shas (plan §3 'hashes.json')."""
    run_dir = Path(run_dir)
    h = {f: sha256_file(run_dir / f) for f in (HEAD_FILE, ROUTER_FILE, DENSE_FILE) if (run_dir / f).exists()}
    h.update(extra or {})
    h["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "hashes.json").write_text(json.dumps(h, indent=2), encoding="utf-8")
    return run_dir / "hashes.json"
