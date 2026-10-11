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


SPLITS = ("calib", "test", "pilot")
LAYER_SPECS = ("all", "upper_half", "tracked8", "last")
PRECISIONS = ("fp16", "fp32")
SELECTOR_RULES = ("topk", "mass_threshold", "all", "oracle", "random")
CLOCKS = ("monotonic", "virtual")


def _each(v) -> list:
    return list(v) if isinstance(v, list) else [v]


def validate_exp_values(d: Mapping[str, Any], where: Any = "") -> None:
    """Value checks that would otherwise fail only after generation (SPEC 2.1, 4.5, 5.2, 8.1)."""
    split = get_dotted(d, "data.split", "test")
    if split not in SPLITS:
        raise ConfigError(f"{where}: data.split must be one of {SPLITS}, got {split!r}")
    rec = d.get("record") or {}
    layers = rec.get("layers", "all")
    if not ((isinstance(layers, list) and all(isinstance(x, int) for x in layers)) or layers in LAYER_SPECS):
        raise ConfigError(f"{where}: record.layers must be {LAYER_SPECS} or a list of ints, got {layers!r}")
    if rec.get("precision", "fp16") not in PRECISIONS:
        raise ConfigError(f"{where}: record.precision must be one of {PRECISIONS}")
    sel = d.get("selector") or {}
    for rule in _each(sel.get("rule", "topk")):
        if rule not in SELECTOR_RULES:
            raise ConfigError(f"{where}: selector.rule {rule!r} not in {SELECTOR_RULES}")
    for ls in _each(sel.get("layers", "tracked8")):
        if not (isinstance(ls, list) or ls in LAYER_SPECS):
            raise ConfigError(f"{where}: selector.layers {ls!r} not in {LAYER_SPECS} or a list of ints")
    if d.get("clock", "monotonic") not in CLOCKS:
        raise ConfigError(f"{where}: clock must be one of {CLOCKS}")
    for inc in (d.get("sweep") or {}).get("include") or []:
        if not isinstance(inc, Mapping):
            raise ConfigError(f"{where}: sweep.include entries must be mappings of dotted keys, got {inc!r}")


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
    validate_exp_values(d, path)
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
    out["flags"] = validate_flags(out["flags"], path)
    return out


# SPEC 9.1 enum flags. YAML 1.1 reads unquoted off/on as booleans; they are mapped back first.
ENUM_FLAGS = {
    "harm_scan": ("off", "parallel", "blocking"), "output_guard": ("off", "on"), "inj_path": ("off", "on"),
    "verdict_cache": ("off", "per_request", "global"), "scheduler": ("none", "fifo", "edf"),
    "ledger_export": ("off", "sync", "async"), "scorer_combine": ("none", "union_topk"),
}


def validate_flags(flags: Mapping[str, Any], where: Any = "") -> dict[str, Any]:
    """SPEC 9.1 validation rules. Returns the flags with YAML booleans mapped back to off / on."""
    out = dict(flags)
    for k, allowed in ENUM_FLAGS.items():
        if k not in out:
            continue
        v = out[k]
        if v is False:
            v = "off"
        elif v is True and "on" in allowed:
            v = "on"
        if v not in allowed:
            raise ConfigError(f"{where}: {k} must be one of {allowed}, got {out[k]!r}")
        out[k] = v
    if out.get("release_before_verdict") and out.get("block_before_generation"):
        raise ConfigError(f"{where}: release_before_verdict and block_before_generation are exclusive")
    sel = out.get("selector") or {}
    if sel.get("head_weighting") == "learned" and sel.get("first_token", "skip") != "skip":
        raise ConfigError(f"{where}: selector.head_weighting=learned needs selector.first_token=skip")
    if out.get("block_before_generation") and sel.get("rule", "topk") in ("topk", "mass_threshold"):
        raise ConfigError(f"{where}: block_before_generation judges before prefill (no ledger yet); "
                          "selector.rule must be all | oracle | random")
    return out
