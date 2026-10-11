"""Experiment and system YAML (SPEC 8).

Experiment file: configs/exp/<id>.yaml (SPEC 8.1). System file: configs/systems/<name>.yaml,
either a base ({system_id, impl, flags}) or an overlay ({base, system_id, override}); the overlay
deep-merges `override` into the base flags (SPEC 8.2).
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
EXP_REQUIRED = ("exp_id", "systems", "targets")
VALID_TOP = {
    "exp_id", "plan_ref", "notes", "outputs", "targets", "systems", "judges", "data", "record", "scorers",
    "selector", "chunking", "closed_loop", "workload", "topology", "operating_point", "repeats", "seeds",
    "fairness", "fairness_exempt", "sweep", "slurm", "clock", "success_judge",
}


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return d


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_dotted(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def del_dotted(d: dict[str, Any], key: str) -> None:
    parts = key.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def load_exp(path: str | Path) -> dict[str, Any]:
    d = load_yaml(path)
    missing = [k for k in EXP_REQUIRED if k not in d]
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")
    unknown = sorted(set(d) - VALID_TOP)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}")
    for k in ("systems", "targets"):
        if not isinstance(d[k], list) or not d[k]:
            raise ConfigError(f"{path}: {k} must be a non-empty list")
    return d


def system_path(name: str, systems_dir: str | Path | None = None) -> Path:
    d = Path(systems_dir) if systems_dir else CONFIGS / "systems"
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = d / (name if name.endswith(".yaml") else f"{name.lower()}.yaml")
    if not cand.exists():
        raise ConfigError(f"system {name!r}: {cand} not found")
    return cand


def load_system(name: str, systems_dir: str | Path | None = None, _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Resolved system: {system_id, impl, flags}."""
    path = system_path(name, systems_dir)
    if str(path) in _seen:
        raise ConfigError(f"system overlay cycle: {_seen + (str(path),)}")
    d = load_yaml(path)
    if "base" in d:
        base = load_system(str(path.parent / d["base"]), systems_dir, _seen + (str(path),))
        flags = deep_merge(base["flags"], d.get("override") or {})
        out = {"system_id": d.get("system_id", base["system_id"]), "impl": d.get("impl", base["impl"]),
               "flags": flags}
    else:
        if "impl" not in d or "system_id" not in d:
            raise ConfigError(f"{path}: base system file needs system_id and impl")
        out = {"system_id": d["system_id"], "impl": d["impl"], "flags": dict(d.get("flags") or {})}
    validate_flags(out["flags"], path)
    return out


def validate_flags(flags: Mapping[str, Any], where: Any = "") -> None:
    """SPEC 9.1 validation rules."""
    if flags.get("release_before_verdict") and flags.get("block_before_generation"):
        raise ConfigError(f"{where}: release_before_verdict and block_before_generation are exclusive")
    sel = flags.get("selector") or {}
    if sel.get("head_weighting") == "learned" and sel.get("first_token", "skip") != "skip":
        raise ConfigError(f"{where}: selector.head_weighting=learned needs selector.first_token=skip")
