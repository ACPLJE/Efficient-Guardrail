"""The ablation registry (plan S8-S9, report E6): every arm is one change to the main 0.6B s0 run.

Three kinds of arm:
    infer    the base checkpoint evaluated under an inference-time option (k, stride, tail pad, k = all)
    router   the base HEAD with a retrained router (pooling, reading depth, objective, capacity); the arm's run dir holds a
             symlink to the base head.pt and its own router.pt
    head     a retrained head (window size, hard negatives, initialisation, data recipe) + its own router (plan §3: every
             router is trained on the head it routes for)
An arm lives in runs/v2/<backbone>_<arm>_s<seed>/ (lgs4 `--variant <arm>`) and its S5 rows in eval/v2/e2/<method>__<run>.rows.jsonl
on the ablation subset (lgs8.config.ABL: 200 pairs x 8 L x 5 depths + benign-long, calib benign-long for the thresholds).
Table 4 (lgs8.table4) puts the arms of one axis next to the baseline measured on the same subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from lgs2.config import TRAIN_DOMAINS
from lgs4.config import S4Paths
from lgs5.config import ModelSpec, parse_spec

BASE_BACKBONE = "qwen3guard_0.6b"
BASE_SEED = 0


@dataclass(frozen=True)
class Arm:
    name: str
    axis: str                                   # E6-1 k | E6-2 window | ... | tailpad
    kind: str                                   # infer | router | head
    desc: str
    head: Dict = field(default_factory=dict)    # lgs4.head_train knobs (CLI names without dashes; True = flag)
    router: Dict = field(default_factory=dict)  # lgs4.router_train knobs
    spec_kind: str = "lg"                       # lg | sliding (infer arms)
    spec_opts: str = ""                         # ":k64" | ":s256" | ":tp32" (infer arms)
    gpu_gb: int = 24                            # tier of the training / eval jobs (0.6B)
    on: Optional[str] = None                    # infer arms: evaluate on THIS arm's checkpoints instead of the base run

    @property
    def variant(self) -> str:
        """The lgs4 run-dir variant this arm's checkpoints live in ('' = the base run)."""
        if self.kind == "infer":
            return self.on or ""
        return self.name

    def run_name(self, backbone: str = BASE_BACKBONE, seed: int = BASE_SEED) -> str:
        v = self.variant
        return f"{backbone}{('_' + v) if v else ''}_s{seed}"

    def run_dir(self, P4: S4Paths, backbone: str = BASE_BACKBONE, seed: int = BASE_SEED) -> Path:
        return P4.run_dir(backbone, seed, self.variant)

    def spec_str(self, backbone: str = BASE_BACKBONE, seed: int = BASE_SEED) -> str:
        return f"{self.spec_kind}:{self.run_name(backbone, seed)}{self.spec_opts}"

    def spec(self, backbone: str = BASE_BACKBONE, seed: int = BASE_SEED) -> ModelSpec:
        return parse_spec(self.spec_str(backbone, seed))

    @property
    def trains_head(self) -> bool:
        return self.kind == "head"

    @property
    def trains_router(self) -> bool:
        return self.kind in ("head", "router")


def _arm(name, axis, kind, desc, **kw) -> Arm:
    return Arm(name, axis, kind, desc, **kw)


DOM1 = "govreport" if "govreport" in TRAIN_DOMAINS else (TRAIN_DOMAINS[0] if TRAIN_DOMAINS else "govreport")

ARMS: Dict[str, Arm] = {a.name: a for a in [
    # ---- E6-1 k (k = 4/8/16 are read off the k=32 rows by lgs5.tables; these need their own pass) ----
    _arm("k64", "E6-1 k", "infer", "top-64 routing", spec_opts=":k64"),
    _arm("kall", "E6-1 k", "infer", "the head on every window (k = all; B5 sliding)", spec_kind="sliding"),
    # ---- E6-2 window ----
    _arm("b64", "E6-2 window", "head", "128-token windows (b=64), head + router retrained", head={"b": 64}, router={}),
    _arm("b256", "E6-2 window", "head", "512-token windows (b=256), head + router retrained", head={"b": 256}, router={}, gpu_gb=48),
    _arm("stride256", "E6-2 window", "infer", "disjoint 256-token windows (stride 2b) at inference", spec_opts=":s256"),
    # ---- E6-3 router reading depth ----
    _arm("ell6", "E6-3 depth", "router", "router reads the layer-6 state (head still from 9)", router={"router-ells": "6"}),
    _arm("ell12", "E6-3 depth", "router", "router reads the layer-12 state (trunk runs to 12 on every window)", router={"router-ells": "12"}),
    _arm("ell16", "E6-3 depth", "router", "router reads the layer-16 state", router={"router-ells": "16"}),
    _arm("ell6_12", "E6-3 depth", "router", "router reads layers 6 and 12 concatenated", router={"router-ells": "6,12"}),
    # ---- E6-4 in-window pooling ----
    _arm("pool_mean", "E6-4 pooling", "router", "mean of the token scores", router={"pool": "mean"}),
    _arm("pool_top2", "E6-4 pooling", "router", "mean of the top-2 token scores", router={"pool": "top2"}),
    _arm("pool_top4", "E6-4 pooling", "router", "mean of the top-4 token scores", router={"pool": "top4"}),
    _arm("pool_lse", "E6-4 pooling", "router", "log-sum-exp of the token scores", router={"pool": "lse"}),
    # ---- E6-5 router objective x capacity (7 routers) ----
    _arm("obj_token", "E6-5 objective", "router", "token-BCE (needle tokens vs the rest), linear", router={"objective": "token_bce"}),
    _arm("obj_window", "E6-5 objective", "router", "window-BCE (needle windows), linear", router={"objective": "window_bce"}),
    _arm("obj_distill", "E6-5 objective", "router", "head distillation without the listwise term, linear", router={"objective": "distill"}),
    _arm("mlp", "E6-5 capacity", "router", "MLP 1024->256->1, distill + listwise", router={"arch": "mlp"}),
    _arm("mlp_token", "E6-5 capacity", "router", "MLP, token-BCE", router={"arch": "mlp", "objective": "token_bce"}),
    _arm("mlp_window", "E6-5 capacity", "router", "MLP, window-BCE", router={"arch": "mlp", "objective": "window_bce"}),
    _arm("mlp_distill", "E6-5 capacity", "router", "MLP, distill without listwise", router={"arch": "mlp", "objective": "distill"}),
    # ---- E6-6 head training: hard negatives / initialisation ----
    _arm("hardneg", "E6-6 head", "head", "head retrained from the Gen init with 4 mined hard negatives per row (lgs4.hardneg on the base run)",
         head={"hard-neg": "@hardneg", "hard-per-row": 4}),
    _arm("hardneg_ft", "E6-6 head", "head", "the base head fine-tuned one more epoch on the mined hard negatives",
         head={"hard-neg": "@hardneg", "hard-per-row": 4, "init-from": "@base_head", "epochs": 1}),
    _arm("init_dense", "E6-6 head", "head", "deep layers initialised from the dense-FT s0 checkpoint, then window-FT",
         head={"init-from": "@base_dense"}),
    # ---- E6-10 training schedule (9/29: len4k and data50 -- both fewer optimiser steps than the full recipe -- beat it at 128k
    #      by 7 pt while the seeds spread 2 pt: is the base head over-trained for 128k?  val (<= 8k) cannot see it) ----
    _arm("ep1", "E6-10 epochs", "head", "full recipe, 1 epoch (a third of the base schedule)", head={"epochs": 1}),
    _arm("ep2", "E6-10 epochs", "head", "full recipe, 2 epochs", head={"epochs": 2}),
    # ---- E6-7 training-length recipe ----
    _arm("len4k", "E6-7 recipe", "head", "presence rows <= 4k only", head={"max-L": 4096}),
    _arm("len8k", "E6-7 recipe", "head", "presence rows <= 8k only", head={"max-L": 8192}),
    # ---- E6-8 data amount ----
    _arm("data10", "E6-8 data", "head", "10 % of the pairs", head={"needle-frac": 0.10}),
    _arm("data25", "E6-8 data", "head", "25 % of the pairs", head={"needle-frac": 0.25}),
    _arm("data50", "E6-8 data", "head", "50 % of the pairs", head={"needle-frac": 0.50}),
    # ---- E6-9 backgrounds / twins ----
    _arm("dom1", "E6-9 domains", "head", f"one background domain ({DOM1}) only", head={"domains": DOM1}),
    _arm("notwin", "E6-9 domains", "head", "benign twins dropped from the presence rows", head={"no-twin": True}),
    # ---- document-end deficit (S5/S6 diagnosis): tail pad ----
    _arm("tp32", "tailpad", "infer", "32 neutral suffix tokens appended before windowing", spec_opts=":tp32"),
    _arm("tp64", "tailpad", "infer", "64 neutral suffix tokens appended before windowing", spec_opts=":tp64"),
    _arm("hardneg_tp32", "tailpad", "infer", "the hardneg head with 32 suffix tokens (both G2 (a) levers; after hardneg trained)",
         spec_opts=":tp32", on="hardneg"),
    # 9/30: b256 alone is +4.3 pt R@128k / +2.6 pt R@1k over the base head (Table 4) -- the two levers that help at 128k stacked
    _arm("b256_tp64", "tailpad", "infer", "the b=256 head with 64 suffix tokens (window size + tail pad; after b256 trained)",
         spec_opts=":tp64", on="b256", gpu_gb=48),
]}

AXES: List[str] = []
for _a in ARMS.values():
    if _a.axis not in AXES:
        AXES.append(_a.axis)


def arms_of(names: Optional[Sequence[str]] = None, axis: Optional[str] = None) -> List[Arm]:
    """Arms by name list ('all' / None = every arm) and/or axis prefix (e.g. 'E6-4')."""
    out = list(ARMS.values())
    if names and list(names) != ["all"]:
        bad = [n for n in names if n not in ARMS]
        if bad:
            raise SystemExit(f"unknown arm(s) {bad}; one of {list(ARMS)}")
        out = [ARMS[n] for n in names]
    if axis:
        out = [a for a in out if a.axis.startswith(axis)]
    return out


def resolve_head_flags(arm: Arm, P4: S4Paths, backbone: str = BASE_BACKBONE, seed: int = BASE_SEED) -> Dict:
    """Head-trainer knobs with the '@' placeholders resolved against the base run (hard negatives dir, base head / dense)."""
    base = P4.run_dir(backbone, seed)
    out = {}
    for k, v in arm.head.items():
        if v == "@hardneg":
            v = str(base / "hardneg")
        elif v == "@base_head":
            v = str(base / "head.pt")
        elif v == "@base_dense":
            v = str(base / "dense_ft.pt")
        out[k] = v
    return out


def cli_flags(knobs: Dict) -> str:
    """{'b': 64, 'no-twin': True, 'router-ells': '6,12'} -> ' --b 64 --no-twin --router-ells 6,12'."""
    parts = []
    for k, v in knobs.items():
        if v is True:
            parts.append(f"--{k}")
        elif v is False or v is None:
            continue
        else:
            parts.append(f"--{k} {v}")
    return (" " + " ".join(parts)) if parts else ""
