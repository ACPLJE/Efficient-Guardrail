"""Target configs and factories (SPEC 4.3; plan 6.1; build plan S2 step 0).

configs/targets/<target_id>.yaml holds the structure values of a target (plan 6.1, D-031, D-037).
`create_hf_target` builds the HF reference target, `length_fn_for` gives the data generators the
rendered prompt token count of a segment list (same tokenizer + chat template as the target).
Mock targets ("MT*") have no YAML: MockTokenizer + simple_template. The special target "tiny-hf"
(`build: tiny_hf`) resolves to the cached tiny model of ag.reference.tiny_hf.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from ag.data.types import Segment

REPO = Path(__file__).resolve().parents[2]
TARGETS_DIR = REPO / "configs" / "targets"
REQUIRED_KEYS = ("target_id", "hf_repo", "revision", "n_layers", "n_q_heads", "n_kv_heads", "max_context",
                 "layer_attn_kind", "tracked8", "dtype", "verify", "notes")

log = logging.getLogger("ag.targets.registry")
LengthFn = Callable[[Sequence[Segment]], int]


def resolve_targets_dir(override: str | os.PathLike | None = None) -> Path:
    return Path(override or os.environ.get("AG_TARGETS_DIR") or TARGETS_DIR)


def known_targets(override: str | os.PathLike | None = None) -> list[str]:
    d = resolve_targets_dir(override)
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


def is_mock_target(target_id: str) -> bool:
    return str(target_id).startswith("MT")


def load_target_config(target_id: str, targets_dir: str | os.PathLike | None = None) -> dict[str, Any]:
    """configs/targets/<target_id>.yaml (override dir: argument, then $AG_TARGETS_DIR)."""
    d = resolve_targets_dir(targets_dir)
    p = d / f"{target_id}.yaml"
    if not p.is_file():
        raise KeyError(f"unknown target {target_id!r}: {p} not found; known: {known_targets(d)}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{p}: missing keys {missing}")
    if cfg["target_id"] != target_id:
        raise ValueError(f"{p}: target_id {cfg['target_id']!r} != file name {target_id!r}")
    return cfg


def resolve_model_path(target_id: str, cfg: dict[str, Any], model_path: str | os.PathLike | None = None) -> str:
    """Explicit model_path (system flag) > tiny_hf build > hf_repo (resolved in the HF cache later)."""
    if model_path:
        return os.fspath(model_path)
    if cfg.get("build") == "tiny_hf":
        from ag.reference.tiny_hf import ensure_tiny_hf
        return str(ensure_tiny_hf(template=cfg.get("template", "chatml")))
    if cfg.get("hf_repo"):
        return str(cfg["hf_repo"])
    raise ValueError(f"target {target_id!r} has neither hf_repo nor build; pass model_path")


def create_hf_target(target_id: str, *, model_path: str | os.PathLike | None = None, device: str | None = None,
                     dtype: str | None = None, targets_dir: str | os.PathLike | None = None, split_tokens: int = 128,
                     now_ns: Callable[[], int] | None = None, local_files_only: bool = True,
                     attn_implementation: str = "sdpa"):
    """HFReferenceTarget for a configured target. dtype None = the YAML dtype; max_context from the YAML.
    Structure values that differ between the YAML and the HF config are logged (an error when verify
    is false would block S2 on a typo in a comment-level field, so it is a warning here)."""
    from ag.reference.hf_target import HFReferenceTarget
    cfg = load_target_config(target_id, targets_dir)
    tgt = HFReferenceTarget(resolve_model_path(target_id, cfg, model_path), target_id=target_id,
                            revision=None if model_path else cfg.get("revision"), dtype=dtype or cfg["dtype"],
                            device=device, attn_implementation=attn_implementation, split_tokens=split_tokens,
                            now_ns=now_ns, local_files_only=local_files_only, max_context=int(cfg["max_context"]))
    for k in ("n_layers", "n_q_heads", "n_kv_heads"):
        if int(cfg[k]) != getattr(tgt, k):
            log.warning("target %s: %s=%s in %s.yaml but %s in the HF config (verify=%s)", target_id, k, cfg[k],
                        target_id, getattr(tgt, k), cfg.get("verify"))
    return tgt


def length_fn_for(target_id: str, *, model_path: str | os.PathLike | None = None,
                  targets_dir: str | os.PathLike | None = None) -> LengthFn:
    """segments -> len(render(...).prompt_ids) for the target, without loading model weights."""
    from ag.data.render import simple_template
    if is_mock_target(target_id) and not model_path:
        from ag.targets.mock import MockTokenizer
        tok = MockTokenizer()

        def mock_len(segments: Sequence[Segment]) -> int:
            return len(tok.encode(simple_template(segments)[0]))
        return mock_len

    from ag.reference.hf_target import HFTokenizerAdapter, hf_chat_template, load_tokenizer, resolve_model_dir
    if is_mock_target(target_id):
        cfg: dict[str, Any] = {}
    else:
        cfg = load_target_config(target_id, targets_dir)
    path = resolve_model_path(target_id, cfg, model_path)
    tok = load_tokenizer(resolve_model_dir(path, revision=None if model_path else cfg.get("revision")))
    adapter, template = HFTokenizerAdapter(tok), hf_chat_template(tok)

    def hf_len(segments: Sequence[Segment]) -> int:
        return adapter.count(template(segments)[0])
    return hf_len
