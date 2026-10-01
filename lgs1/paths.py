"""Central path/config resolution for Stage S1 (reads env/s1.env, then the process environment)."""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

PKG_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PKG_ROOT / "env" / "s1.env"

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_env_file(path: Path = ENV_FILE, override: bool = True) -> Dict[str, str]:
    """Parse `export KEY=VALUE` lines with ${VAR} expansion and export them.

    The file wins over values already in the process environment (same as `source env/s1.env` in
    bash), so a stale `source` in an interactive shell cannot shadow an `env-set` edit.  Set
    LG_ENV_FILE_NO_OVERRIDE=1 to keep shell values instead (e.g. for one-off experiments)."""
    if os.environ.get("LG_ENV_FILE_NO_OVERRIDE") == "1":
        override = False
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip() if not raw.strip().startswith("#") else ""
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val and val[0] in "\"'" and val[-1] == val[0]:
            val = val[1:-1]
        val = os.path.expandvars(os.path.expanduser(val)) if val else val
        values[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
        # make later ${VAR} references resolve to this value
        os.environ.setdefault(key, val)
    return values


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


@dataclass
class S1Paths:
    out: Path
    sources: Path
    hf_home: Path
    v04_root: Optional[Path]
    v04_runs: Optional[Path]
    paper_root: Optional[Path]
    data_v1: Optional[Path]
    external_code: Optional[Path]
    external_data: Optional[Path]
    symlink_06b: Optional[Path]
    round3_root: Optional[Path]
    quota_home: Path
    python: str

    @classmethod
    def load(cls) -> "S1Paths":
        load_env_file()
        def p(k):
            v = env(k)
            return Path(v) if v else None
        py = env("LG_PYTHON", "python3")
        if not Path(py).exists():
            py = "python3"
        return cls(
            out=Path(env("LG_OUT", str(Path.home() / "lg_s1_out"))),
            sources=Path(env("LG_SOURCES", str(Path.home() / "lg_data_v2" / "sources"))),
            hf_home=Path(env("HF_HOME", str(Path.home() / ".cache" / "huggingface"))),
            v04_root=p("LG_V04_ROOT"), v04_runs=p("LG_V04_RUNS"), paper_root=p("LG_PAPER_ROOT"),
            data_v1=p("LG_DATA_V1"), external_code=p("LG_EXTERNAL_CODE"), external_data=p("LG_EXTERNAL_DATA"),
            symlink_06b=p("LG_06B_DATA_SYMLINK"), round3_root=p("LG_ROUND3_ROOT"),
            quota_home=Path(env("LG_HOME", str(Path.home()))), python=py,
        )

    def ensure_out(self) -> Path:
        for sub in ("reports", "rows", "logs", "slurm", "zeroshot"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        return self.out
