"""S5 configuration: benches, length grids, model specs and output paths.

Model spec strings (s5ctl --models, lgs5.run --model):
    lg:<run>[:k<K>][:s<S>][:tp<N>]
                         Landmark Guard from runs/v2/<run> (head.pt + router.pt), top-k routing (default k=32).  The rows
                         keep the head probability of every routed window in router order, so k=8/16 are read off the
                         k=32 rows (lgs5.tables) -- one pass per checkpoint.  S8 inference-only ablations: s<S> = window
                         stride override (E6-2: s256 = disjoint 256-token windows), tp<N> = tail pad: N tokens of a neutral
                         suffix appended to every document before windowing (the depth-1.0 / document-end deficit, S5-S6
                         diagnosis: the last-token probe needs content after the needle).  Method names carry the options
                         (lg_k32_s256, lg_k32_tp32) so the rows never mix with the default run's.
    sliding:<run>        B5: the same head on every window (k = all).
    oracle:<run>         E2-7 (ii): the head on the windows that contain the needle (router replaced by the truth).
    needle:<run>         E2-7 (i): the needle text alone through LG (no background).
    dense:<run>[:policy] dense-FT baseline from runs/v2/<run>/dense_ft.pt (full context, last-token probe).  Beyond the
                         model's window (32k) the policy decides (S6 B4 64k/128k rows): head (default; keep the first 32k
                         tokens; method dense_ft, the <=28k rows are native either way), tail (keep the last 32k;
                         dense_ft_tail), yarn (YaRN x4 -> 128k context; dense_ft_yarn).
    base:<guard>         zero-shot guard from lgs1.zeroshot.guards (E1 only here; long-context zero-shot rows come from
                         lgs3.longbench).  Qwen3Guard verdicts are kept as safe/controversial/unsafe and the tables show
                         strict (controversial = unsafe) and loose rows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lgs1.paths import env
from lgs2.config import S2Paths
from lgs4.config import LG_K_DEFAULT, S4Paths

# ---- E1: short-prompt parity benches = the test-pool needle sources (lgs2.config.NEEDLE_SOURCES, pool=test) ----
E1_BENCHES: Dict[str, Tuple[str, Optional[int]]] = {      # pool source -> (display name, published size)
    "toxicchat_test": ("ToxicChat test", 2853),
    "openai_mod": ("OpenAI Moderation", 1680),
    "aegis1_test": ("Aegis 1.0 test", None),
    "aegis2_test": ("Aegis 2.0 test", 1984),
    "sst": ("SimpleSafetyTests", 100),
    "harmbench": ("HarmBench prompts", 239),
    "wgtest": ("WildGuardTest", 1756),
}
E1_OVERREFUSAL: Dict[str, Tuple[str, Optional[int]]] = {   # FPR on the benign rows only (E1-2)
    "xstest": ("XSTest safe", 250),
    "orbench_hard": ("OR-Bench-Hard", 1319),
}
E1_SOURCES = list(E1_BENCHES) + list(E1_OVERREFUSAL)

# ---- E2: length grids ----
E2_L_LE28K = [1024, 2048, 4096, 8192, 16384, 28672]
E2_L_LONG = [65536, 131072]                      # S6 E2-1 64k/128k
E2_L_ALL = E2_L_LE28K + E2_L_LONG
K_REPORT = (8, 16, 32)
MATCHED_TARGETS = (0.05, 0.01)
DENSE_POLICIES = ("head", "tail", "yarn")

# ---- S6 variant subsets (lgs3.variants, test only, L in {4k, 28k, 128k}) ----
VAR_L = [4096, 28672, 131072]
VAR_SUBSETS = ["needle_repeat", "decoy", "decoy_flood", "fragmented", "multi_needle", "related", "long_needle", "ood_domain", "framed"]
VAR_S6 = ["needle_repeat", "decoy", "framed", "ood_domain"]          # E2-4, E3-2, E3-3 (+ E7-5 domain rows)


LG_KINDS = ("lg", "sliding", "oracle", "needle")


@dataclass(frozen=True)
class ModelSpec:
    kind: str                 # lg | sliding | oracle | needle | dense | base
    name: str                 # run dir name (runs/v2/<name>) or guard name
    k: int = LG_K_DEFAULT
    policy: str = ""          # dense only: '' | head | tail | yarn (context policy beyond the model window)
    stride: Optional[int] = None   # LG kinds: window stride override (E6-2 inference arm; None = the run's cfg)
    tail_pad: int = 0              # LG kinds: neutral suffix tokens appended before windowing (S8 tailpad arm)

    @property
    def options(self) -> str:
        """Method-name suffix of the inference-only options (empty for the defaults)."""
        out = ""
        if self.kind in LG_KINDS and self.stride:
            out += f"_s{self.stride}"
        if self.kind in LG_KINDS and self.tail_pad:
            out += f"_tp{self.tail_pad}"
        return out

    @property
    def method(self) -> str:
        if self.kind == "lg":
            return f"lg_k{self.k}{self.options}"
        if self.kind == "dense":
            return "dense_ft" if self.policy in ("", "head") else f"dense_ft_{self.policy}"
        if self.kind == "base":
            return f"base_{self.name}"
        return f"{self.kind}{self.options}"

    @property
    def tag(self) -> str:      # file-safe: <method>__<name>
        return f"{self.method}__{self.name}"

    def __str__(self) -> str:
        s = f"{self.kind}:{self.name}"
        if self.kind == "lg" and self.k != LG_K_DEFAULT:
            s += f":k{self.k}"
        if self.kind in LG_KINDS and self.stride:
            s += f":s{self.stride}"
        if self.kind in LG_KINDS and self.tail_pad:
            s += f":tp{self.tail_pad}"
        if self.kind == "dense" and self.policy not in ("", "head"):
            s += f":{self.policy}"
        return s


METHOD_RE = re.compile(r"^lg_k(\d+)(_[A-Za-z0-9_]+)?$")


def split_lg_method(method: str) -> Optional[Tuple[int, str]]:
    """'lg_k32_s256' -> (32, '_s256'); 'lg_k8' -> (8, ''); None for a non-LG method.  The suffix is kept by every
    derivation (k' < k, aggregation), so an inference arm's rows never collide with the default run's."""
    m = METHOD_RE.match(str(method))
    if not m:
        return None
    return int(m.group(1)), (m.group(2) or "")


def parse_spec(s: str) -> ModelSpec:
    parts = s.strip().split(":")
    if len(parts) < 2 or parts[0] not in ("lg", "sliding", "oracle", "needle", "dense", "base"):
        raise SystemExit(f"bad model spec {s!r}: kind:name[:kK|:sS|:tpN|:policy] with kind in lg|sliding|oracle|needle|dense|base")
    k = LG_K_DEFAULT
    policy = ""
    stride: Optional[int] = None
    tail_pad = 0
    for extra in parts[2:]:
        m = re.fullmatch(r"k(\d+)", extra)
        ms = re.fullmatch(r"s(\d+)", extra)
        mt = re.fullmatch(r"tp(\d+)", extra)
        if m and parts[0] == "lg":
            k = int(m.group(1))
        elif ms and parts[0] in LG_KINDS:
            stride = int(ms.group(1))
        elif mt and parts[0] in LG_KINDS:
            tail_pad = int(mt.group(1))
        elif parts[0] == "dense" and extra in DENSE_POLICIES:
            policy = extra
        else:
            raise SystemExit(f"bad model spec {s!r}: unknown option {extra!r}")
    return ModelSpec(parts[0], parts[1], k, policy, stride, tail_pad)


def parse_specs(s: str) -> List[ModelSpec]:
    return [parse_spec(x) for x in s.split(",") if x.strip()]


def parse_Ls(s: str) -> List[int]:
    if s in ("le28k", "28k", ""):
        return list(E2_L_LE28K)
    if s == "all":
        return list(E2_L_ALL)
    if s == "long":
        return list(E2_L_LONG)
    if s == "var":
        return list(VAR_L)
    return [int(x) for x in s.split(",") if x]


@dataclass
class S5Paths:
    data: S2Paths
    s4: S4Paths
    out: Path                 # $LG_OUT
    eval: Path                # $LG_OUT/eval/v2/<exp>/<tag>.rows.jsonl
    reports: Path             # $LG_OUT/reports/s5
    longbench: Path           # $LG_OUT/longbench (lgs3.longbench rows: zero-shot + CD)

    @classmethod
    def load(cls) -> "S5Paths":
        s4 = S4Paths.load()
        out = s4.out
        return cls(data=s4.data, s4=s4, out=out, eval=Path(env("LG_EVAL", str(out / "eval" / "v2"))), reports=out / "reports" / "s5",
                   longbench=out / "longbench")

    def ensure(self) -> "S5Paths":
        for p in (self.eval, self.reports, self.out / "logs", self.out / "slurm"):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def rows_path(self, exp: str, spec: ModelSpec) -> Path:
        return self.eval / exp / f"{spec.tag}.rows.jsonl"

    def run_dir(self, name: str) -> Path:
        return self.s4.runs / name
