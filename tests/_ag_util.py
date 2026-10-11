"""Shared helpers for the AG CPU tests (not a test module)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SKELETON = ROOT / "configs" / "exp" / "skeleton.yaml"
SYSTEMS = ROOT / "configs" / "systems"


def deep_update(d: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def small_exp(tmp_path: Path, name: str = "exp.yaml", **upd: Any) -> Path:
    """skeleton.yaml shrunk for fast tests (2k prompts, 4 pairs), with overrides."""
    exp = yaml.safe_load(SKELETON.read_text())
    exp["data"].update(length_classes=["2k"], n_attack=4, n_benign=4)
    exp["fairness"]["max_new_tokens"] = 48
    deep_update(exp, copy.deepcopy(upd))
    p = tmp_path / name
    p.write_text(yaml.safe_dump(exp, sort_keys=False))
    return p


def system_file(tmp_path: Path, name: str, base: str, system_id: str, override: dict) -> Path:
    """An overlay system file next to copies of the repo's mock base files."""
    sysdir = tmp_path / "systems"
    sysdir.mkdir(exist_ok=True)
    for f in SYSTEMS.glob("mock_*.yaml"):
        (sysdir / f.name).write_text(f.read_text())
    p = sysdir / f"{name}.yaml"
    p.write_text(yaml.safe_dump({"base": base, "system_id": system_id, "override": override}))
    return p


def run_dirs(out: Path, exp_id: str = "mock") -> dict[str, Path]:
    """system_id -> run dir."""
    out_d = {}
    for d in sorted((out / "raw" / exp_id).iterdir()):
        if d.is_dir() and d.name != "jobs":
            out_d[d.name.split("__")[1]] = d
    return out_d


def load_json(p: Path) -> Any:
    return json.loads(Path(p).read_text())


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
