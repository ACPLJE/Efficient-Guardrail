"""S4 configuration: backbones, fixed training settings (plan §3 table -> Table A2), paths.

Plan §3 (fixed for every S4 run; changes only in E6):
  backbone   Qwen3Guard-Gen-0.6B (main, 3 seeds), 4B, 8B (LoRA r=16), Llama Guard 3-8B (LoRA r=16)
  structure  ell = 9 for 0.6B (layers 0-8 frozen = block-local trunk), ~N/3 for the others; b = 128 (256-token windows,
             stride 128); in-window token-max pooling; k = 32 default (8, 16 reported); joint head off
  head       v2 train presence rows (train/main 32,000 + train/main16k 2,000, all <= 16k) + native rows (~75k);
             LG and dense-FT see the same rows and the same number of optimizer steps
  router     LayerNorm(d) -> Linear(d, 1) (~3k params), head-distillation = BCE + listwise softmax ranking,
             ALWAYS trained on the FT checkpoint (the head run of the same backbone/seed)
  seeds      0/1/2 = probe init + data order; 0.6B only 3 seeds, other backbones s0
  saving     runs/v2/<backbone>_s<seed>/{head.pt, router.pt, dense_ft.pt, hashes.json, *.log.jsonl, val_report.json}
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lgs1.paths import env
from lgs2.config import RECIPE_V2, S2Paths

WINDOW_B = RECIPE_V2["window_b"]          # 128


@dataclass(frozen=True)
class Backbone:
    name: str
    repo: str
    ell: int                              # trunk depth; layers [0, ell) frozen and block-local
    mode: str = "full"                    # full = fine-tune layers [ell, N) ; lora = LoRA r on layers [ell, N)
    lora_r: int = 16
    lora_alpha: int = 32
    min_gpu_gb: int = 24
    grad_ckpt: bool = False
    micro_windows: int = 32               # windows per forward/backward (head training); batch = HEAD_HP.batch_windows
    dense_max_tokens: int = 16384         # dense-FT context cap (plan: 0.6B <= 16k, 8B LoRA <= 8k)
    dense_micro_tokens: int = 16384       # tokens per dense micro-batch (padded)
    dense_mode: str = "full"              # full = all layers ; lora = LoRA on all layers
    tok: str = "qwen3"                    # tokenizer family of the prepped token arrays (lgs2.config.TOKENIZERS / D8 tok/<family>)
    note: str = ""


BACKBONES: Dict[str, Backbone] = {
    "qwen3guard_0.6b": Backbone("qwen3guard_0.6b", "Qwen/Qwen3Guard-Gen-0.6B", ell=9, mode="full", min_gpu_gb=24,
                                micro_windows=32, dense_max_tokens=16384, dense_micro_tokens=24576,
                                note="main backbone; 28 layers, d=1024; 3 seeds (dense micro 24k padded tokens: 9.8 GB peak at 16k)"),
    "qwen3guard_4b": Backbone("qwen3guard_4b", "Qwen/Qwen3Guard-Gen-4B", ell=12, mode="full", min_gpu_gb=48, grad_ckpt=True,
                              micro_windows=8, dense_max_tokens=16384, dense_micro_tokens=8192,
                              note="36 layers, d=2560; full FT of layers 12-35 with gradient checkpointing (48 GB)"),
    "qwen3guard_8b": Backbone("qwen3guard_8b", "Qwen/Qwen3Guard-Gen-8B", ell=12, mode="lora", lora_r=16, min_gpu_gb=48,
                              grad_ckpt=True, micro_windows=8, dense_max_tokens=8192, dense_micro_tokens=8192, dense_mode="lora",
                              note="36 layers, d=4096; LoRA r=16 on layers 12-35"),
    "llamaguard3_8b": Backbone("llamaguard3_8b", "meta-llama/Llama-Guard-3-8B", ell=11, mode="lora", lora_r=16, min_gpu_gb=48,
                               grad_ckpt=True, micro_windows=8, dense_max_tokens=8192, dense_micro_tokens=8192, dense_mode="lora",
                               tok="llama3", note="32 layers, d=4096; LoRA r=16 on layers 11-31; Llama 3 tokenizer -> s4/prep_llama3 (T5)"),
}

# ---- fixed hyper-parameters (Table A2) ----------------------------------------------------------------------
HEAD_HP: Dict = {
    "epochs": 3,                    # passes over the presence rows (native rows are interleaved at native_ratio)
    "batch_windows": 32,            # windows per optimizer step (micro-batched by Backbone.micro_windows)
    "lr_deep": 2e-5,                # fine-tuned deep layers (mode=full)
    "lr_lora": 2e-4,                # LoRA A/B (mode=lora)
    "lr_probe": 1e-3,               # LayerNorm + Linear probe (+ the landmark vector when probe="landmark")
    "probe": "last",                # last | landmark (lgs1.lgcore.model.Head; head v2 ablation 2026-09-21)
    "weight_decay": 0.01,
    "warmup_frac": 0.03,
    "schedule": "cosine",
    "grad_clip": 1.0,
    "neg_per_row": 6,               # random non-needle windows per presence row
    "near_neg": 2,                  # windows adjacent to the needle windows (hard negatives)
    "native_ratio": 0.5,            # native rows drawn per presence row (75k native / 34k presence ~ 2.2 -> capped)
    "pos_weight": "auto",           # BCE positive weight = n_neg / n_pos of the sampled epoch (capped at 8)
    "pos_weight_cap": 8.0,
    "bf16": True,
    "eval_every": 2000,             # window-level val AUPRC on val/main needle+twin windows
    "log_every": 50,
    # S8 ablation knobs (E6-2/6/7/8/9); the defaults reproduce the S4 head
    "b": WINDOW_B,                  # window half-size
    "hard_neg": None,               # dir of mined hard negatives (lgs4.hardneg); hard_per_row of them replace random negatives
    "hard_per_row": 0,
    "init_from": None,              # head.pt / dense_ft.pt to initialise the deep layers from
    "max_L": None,                  # presence rows up to this length
    "needle_frac": 1.0,             # seeded fraction of the pairs
    "domains": None,                # background domains kept
    "twin": True,                   # benign twins on
}
ROUTER_HP: Dict = {
    "epochs": 2,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "lambda_rank": 1.0,             # listwise softmax ranking (KL over windows of one row) weight
    "temperature": 1.0,             # softmax temperature on the head logits
    "target": "head",               # head = distil P_head(window) ; label = needle windows only (ablation E6)
    "pos_threshold": 0.5,           # BCE target = 1[P_head > pos_threshold]
    "batch_rows": 8,
    "router_rows": 12000,           # presence rows for the router (seeded uniform subsample of the head's rows; 2026-09-20:
                                    # all 66k rows = 2.1M windows took > 4 h on an RTX 3090 for the targets alone)
    "native_ratio": 0.5,            # native rows per presence row (also subsampled)
    "bf16": True,
    "log_every": 100,
    # S8 ablation knobs (E6-3/4/5); the defaults reproduce the S4 router
    "pool": "max",                  # in-window pooling of the token scores: max | mean | top2 | top4 | lse
    "arch": "linear",               # router capacity: linear (LN -> Linear) | mlp (LN -> Linear -> GELU -> Linear)
    "mlp_hidden": 256,
    "objective": "distill_listwise",   # distill_listwise | distill | window_bce | token_bce (lgs4.router_train.OBJECTIVES)
    "router_ells": None,            # trunk depths the router reads (list), None = the head's ell
}
DENSE_HP: Dict = {
    "lr_deep": 2e-5, "lr_lora": 2e-4, "lr_probe": 1e-3, "weight_decay": 0.01, "warmup_frac": 0.03, "schedule": "cosine",
    "grad_clip": 1.0, "bf16": True, "log_every": 50, "eval_every": 2000,
    "native_ratio": 0.5,            # same sampling policy as the LG head
    # training budget relative to the LG head of the same backbone/seed (steps derived from head.pt):
    #   tokens = same number of training tokens (head: steps x 32 windows x 256; dense: steps x rows/step x mean row length)
    #   flops  = same training FLOPs (tokens x layer passes: head ell + 3(N-ell), dense 3N (+N with checkpointing))
    #   steps  = same optimizer-step count (the original Table A2 wording; 2026-09-20 a dense step is 8 rows x ~5k tokens =
    #            ~5x the head's tokens and 28/28 layers, 15 s/step on an RTX 3090 -> 58k steps = 240 h per seed)
    "budget": "tokens",
}
LG_K_DEFAULT = 32
LG_K_REPORT = (8, 16, 32)

# data used by S4 (all from the sealed lg_data_v2)
PRESENCE_SETS: List[Tuple[str, str]] = [("train", "main"), ("train", "main16k")]
NATIVE_SETS: List[Tuple[str, str]] = [("train", "native")]
VAL_SETS: List[Tuple[str, str]] = [("val", "main"), ("val", "benign_long")]
CALIB_SETS: List[Tuple[str, str]] = [("calib", "main"), ("calib", "benign_long")]
CALIB_L = 28672                       # T1 pass criterion: val/calib hit@8 @28k >= 0.95 (round-3 reference 0.96-1.00)
HIT_PASS = 0.95


@dataclass
class S4Paths:
    data: S2Paths
    out: Path                 # $LG_OUT
    prep: Path                # $LG_OUT/s4/prep/<split>/<subset>/{ids.npy, offsets.npy, meta.jsonl.gz}
    runs: Path                # $LG_OUT/runs/v2
    reports: Path

    @classmethod
    def load(cls) -> "S4Paths":
        d = S2Paths.load()
        runs = Path(env("LG_RUNS", str(d.out / "runs" / "v2")))
        return cls(data=d, out=d.out, prep=d.out / "s4" / "prep", runs=runs, reports=d.out / "reports")

    def ensure(self) -> "S4Paths":
        for p in (self.prep, self.runs, self.reports, self.out / "logs", self.out / "slurm"):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, backbone: str, seed: int, variant: str = "") -> Path:
        """runs/<backbone>[_<variant>]_s<seed>: a variant (e.g. `lm` = landmark-probe head, 2026-09-21) trains next to the
        default run without touching it; S5 addresses it by the directory name (lg:qwen3guard_0.6b_lm_s0)."""
        return self.runs / f"{backbone}{('_' + variant) if variant else ''}_s{seed}"

    def for_tok(self, family: str = "qwen3") -> "S4Paths":
        """Paths whose prepped token arrays belong to `family`: the Qwen3 arrays stay at s4/prep (every run so far), another
        tokenizer (T5: Llama Guard 3-8B = llama3) gets s4/prep_<family>.  Token ids of one vocabulary must never feed a
        backbone of another, so every trainer picks its prep through the backbone's `tok`."""
        if not family or family == "qwen3":
            return self
        return dataclasses.replace(self, prep=self.out / "s4" / f"prep_{family}")

    def for_backbone(self, name: str) -> "S4Paths":
        return self.for_tok(BACKBONES[name].tok if name in BACKBONES else "qwen3")

    def sealed(self) -> bool:
        return (self.data.root / "SEALED").exists()


def backbone(name: str) -> Backbone:
    """Backbone by name.  `LG_BB_MODE_<name>=lora` (env / env/s1.env) switches a full-FT backbone to LoRA r=16 without a
    code change -- the fallback if the 4B fp32-master run still does not fit a 48 GB card."""
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone {name}; one of {', '.join(BACKBONES)}")
    bb = BACKBONES[name]
    override = os.environ.get(f"LG_BB_MODE_{name}", "").strip().lower()
    if override and override != bb.mode:
        if override not in ("full", "lora"):
            raise SystemExit(f"LG_BB_MODE_{name}={override}: expected full or lora")
        bb = dataclasses.replace(bb, mode=override, note=(bb.note + f" [mode overridden to {override} by LG_BB_MODE_{name}]").strip())
        print(f"[config] {name}: mode {override} (LG_BB_MODE_{name})", flush=True)
    return bb
