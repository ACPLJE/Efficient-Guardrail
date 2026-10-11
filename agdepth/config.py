"""Configuration of the depth-asymmetry PoC: paths, backbones, snippet classes, contrasts, profiles, sources.

Every number that shapes the experiment lives here.  The window recipe is hashed into datacard.json (recipe_hash) and
the windows themselves get a content hash (data_manifest_sha); every derived file records the latter, so features,
probes and scores of a rebuilt dataset are never mixed with the new windows.

Output root (never the SPEC 7.1 layout of $AG_OUT itself -- this PoC is not yet a SPEC experiment, DECISIONS D-041):
    $AGDEPTH_OUT, else $AG_OUT/poc/agdepth, else $LG_OUT/agdepth
Directory layout under the root:
    <run>/data/windows.jsonl.gz      train/val/test windows (one inserted snippet each)
    <run>/data/bigneg.jsonl.gz       optional large negative pool (test snippets x test backgrounds)
    <run>/data/datacard.json         counts, length stats, split audit, recipe_hash, data_manifest_sha
    <run>/data/lineage.json          lineage of the build run (CLAUDE.md 3.3)
    <run>/feats/<bb>/<framing>/      <pool>.f16 memmaps [L+1, N, d], meta.json, progress.json
    <run>/probes/<bb>/               lineage.json + run_id.txt of the backbone run (extract + fit + score)
    <run>/probes/<bb>/<framing>/     <pool>.probes.npz (weights), <pool>.test_scores.npz, <pool>.bigneg_scores.f32
    <run>/lex/                       lexical (hashed n-gram) baseline scores
    <run>/report/                    lineage.json, probe_results.jsonl, layer_metrics.csv, summary.json, report.md, fig_*.png
    <run>/slurm/, <run>/logs/        rendered sbatch scripts and their logs
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lgs1.paths import S1Paths, env, load_env_file


# ----------------------------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------------------------
@dataclass
class AGPaths:
    root: Path          # every PoC output
    out: Path           # $LG_OUT (logs, slurm scripts)
    sources: Path       # $LG_SOURCES (S1 download layout: <name>/revision.json)
    data_v2: Path       # $LG_DATA_V2 (LG v2 needle pool + haystack); may be absent
    hf_home: Path
    python: str

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "AGPaths":
        load_env_file()
        p1 = S1Paths.load()
        if root is None:
            if env("AGDEPTH_OUT"):
                root = Path(env("AGDEPTH_OUT"))
            elif env("AG_OUT"):
                root = Path(env("AG_OUT")) / "poc" / "agdepth"      # beside, never inside, the SPEC 7.1 raw/ tree
            else:
                root = p1.out / "agdepth"
        data_v2 = Path(env("LG_DATA_V2", str(p1.sources.parent)))
        return cls(root=Path(root).expanduser().resolve(), out=p1.out, sources=p1.sources, data_v2=data_v2,
                   hf_home=p1.hf_home, python=p1.python)

    def run(self, name: str) -> "RunPaths":
        return RunPaths(self.root / name)


@dataclass
class RunPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def windows(self) -> Path:
        return self.data / "windows.jsonl.gz"

    @property
    def bigneg(self) -> Path:
        return self.data / "bigneg.jsonl.gz"

    @property
    def datacard(self) -> Path:
        return self.data / "datacard.json"

    def feats(self, bb: str, framing: str) -> Path:
        return self.root / "feats" / bb / framing

    def probes(self, bb: str, framing: str) -> Path:
        return self.root / "probes" / bb / framing

    def bb_run(self, bb: str) -> Path:
        """Run directory of one backbone (lineage.json, run_id.txt)."""
        return self.root / "probes" / bb

    @property
    def slurm(self) -> Path:
        return self.root / "slurm"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def lex(self) -> Path:
        return self.root / "lex"

    @property
    def report(self) -> Path:
        return self.root / "report"


# ----------------------------------------------------------------------------------------------
# backbones (layer counts are read from the model config at run time; the numbers here are documentation)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BackboneSpec:
    """A registry entry (named *Spec so it does not clash with the SPEC 2.6 `Backbone` protocol)."""
    key: str
    repo: str
    role: str
    n_layers: int
    hidden: int
    gated: bool = False
    min_gpu_gb: int = 24
    tokens_per_batch: int = 32768       # extraction batch budget (padded tokens); halved on OOM
    # task framing: "tokenizer" = the tokenizer's chat template (instruct targets); "chatml" / "llama3" = a neutral
    # user/assistant turn of the model family, used for guards whose own template is a moderation prompt
    chat: str = "tokenizer"


BACKBONES: Dict[str, BackboneSpec] = {b.key: b for b in [
    BackboneSpec("qwen3guard_0.6b", "Qwen/Qwen3Guard-Gen-0.6B", "guard, LG backbone (plan E0.3 / E1.1)", 28, 1024,
                 tokens_per_batch=65536, chat="chatml"),
    BackboneSpec("qwen3guard_4b", "Qwen/Qwen3Guard-Gen-4B", "guard (plan E1.1)", 36, 2560, chat="chatml"),
    BackboneSpec("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct", "target T1 (plan E0.3 / E1.1)", 32, 4096, gated=True,
                 min_gpu_gb=40, tokens_per_batch=16384),
    BackboneSpec("qwen3_4b_2507", "Qwen/Qwen3-4B-Instruct-2507", "target T2 (plan E1.1)", 36, 2560),
    BackboneSpec("qwen3guard_8b", "Qwen/Qwen3Guard-Gen-8B", "guard (optional)", 36, 4096, min_gpu_gb=40,
                 tokens_per_batch=16384, chat="chatml"),
    BackboneSpec("llamaguard3_8b", "meta-llama/Llama-Guard-3-8B", "guard (optional, cross-family)", 32, 4096,
                 gated=True, min_gpu_gb=40, tokens_per_batch=16384, chat="llama3"),
    BackboneSpec("qwen3_8b", "Qwen/Qwen3-8B", "target (optional; 32k native)", 36, 4096, min_gpu_gb=40,
                 tokens_per_batch=16384),
]}
E03_BACKBONES = ["qwen3guard_0.6b", "llama31_8b"]                                   # plan E0.3
E11_BACKBONES = ["qwen3guard_0.6b", "qwen3guard_4b", "llama31_8b", "qwen3_4b_2507"]  # plan E1.1
BACKBONE_ALIASES = {"e03": E03_BACKBONES, "e11": E11_BACKBONES}


def is_local_model(key: str) -> bool:
    return Path(key).expanduser().is_dir()


def backbone(key: str) -> BackboneSpec:
    if key in BACKBONES:
        return BACKBONES[key]
    # a local directory or an unlisted repo id (tests, ad-hoc runs)
    name = Path(key).name if is_local_model(key) else key.replace("/", "_")
    repo = str(Path(key).expanduser().resolve()) if is_local_model(key) else key
    return BackboneSpec(name, repo, "ad-hoc", 0, 0)


# ----------------------------------------------------------------------------------------------
# snippet classes and the contrasts the probes are trained / evaluated on
# ----------------------------------------------------------------------------------------------
CLASSES: Dict[str, str] = {
    "harm": "harmful request (LG v2 needle pool, label harmful)",
    "harm_hard": "benign request, incl. seemingly-toxic over-refusal prompts (LG v2 needle pool, label benign)",
    "inj": "prompt-injection payload from public benchmarks (BIPIA, deepset/prompt-injections, LLMail-Inject, Gandalf)",
    "instr": "instruction-heavy benign text addressed to a human reader (READMEs, emails, procedures, policies)",
    "notinject": "NotInject benign prompts with injection trigger words",
    "plain": "benign paragraph without instructions (another background document)",
}
POS_CLASSES = ("harm", "inj")
NEG_CLASSES = ("harm_hard", "instr", "notinject", "plain")


@dataclass(frozen=True)
class Task:
    name: str
    pos: Tuple[str, ...]
    train_neg: Tuple[str, ...]
    eval_neg: Dict[str, Tuple[str, ...]]

    def classes(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(self.pos + self.train_neg + tuple(c for v in self.eval_neg.values() for c in v)))


TASKS: Dict[str, Task] = {
    "harm": Task("harm", ("harm",), ("plain", "harm_hard"),
                 {"all": ("plain", "harm_hard"), "plain": ("plain",), "hard": ("harm_hard",)}),
    "inj": Task("inj", ("inj",), ("plain", "instr", "notinject"),
                {"all": ("plain", "instr", "notinject"), "plain": ("plain",), "instr": ("instr",),
                 "notinject": ("notinject",), "hard": ("instr", "notinject")}),
}

# plan §8.2 E0.3: l95(injection, instruction-heavy negatives) >= 1.5 x l95(harmful)
GATE = {"shallow": ("harm", "all"), "deep": ("inj", "instr"), "min_ratio": 1.5}
# secondary comparisons reported next to the gate (both sides with hard negatives / both with plain negatives)
COMPARISONS = [
    ("gate", ("harm", "all"), ("inj", "instr")),
    ("hard_vs_hard", ("harm", "hard"), ("inj", "hard")),
    ("plain_vs_plain", ("harm", "plain"), ("inj", "plain")),
]
FPR_TARGETS = (1e-2, 1e-3, 2e-4, 5e-5)       # 2e-4 = window FPR that keeps document FPR at 5 % for 32k (plan §2.2)
PRIMARY_FPR = 2e-4                           # plan §8.2 E0.3 / §9.1 Fig. 3 / SPEC 6.5: l95 and the gate are read here
L95_FRAC = 0.95


# ----------------------------------------------------------------------------------------------
# profiles
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    name: str
    window: int                      # target window length (reference-tokenizer tokens)
    snip_tok: Tuple[int, int]        # inserted snippet length band (inclusive)
    n_train: int                     # windows per class
    n_val: int
    n_test: int
    n_bigneg: int                    # windows per class in BIGNEG_CLASSES (0 = no big pool)
    seeds: Tuple[int, ...]
    n_boot: int
    max_len: int                     # hard truncation for extraction (tokens of the backbone tokenizer)
    max_reuse: int = 3               # a snippet appears in at most this many windows of one split


PROFILES: Dict[str, Profile] = {p.name: p for p in [
    Profile("smoke", 96, (4, 40), 40, 12, 24, 30, (0,), 40, 160),
    # quick: 30k big-pool windows per class so TPR @ 2e-4 sits on >= 5 expected false positives (plan E0.3)
    Profile("quick", 256, (12, 112), 1500, 300, 600, 30_000, (0, 1, 2), 500, 320),
    Profile("full", 256, (12, 112), 4000, 500, 1500, 100_000, (0, 1, 2, 3, 4), 1000, 320),
]}
BIGNEG_CLASSES = ("plain", "instr", "harm_hard")
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.1, "test": 0.3}       # by snippet group and by background document
DEPTH_RANGE = (0.05, 0.95)                                    # relative insertion position inside the window
SEED = 20261011
EXP_OF_PROFILE = {"smoke": "E0.3", "quick": "E0.3", "full": "E1.1"}
FRAMINGS = ("raw", "task")
POOLINGS = ("mean", "last", "span")
DEFAULT_POOLINGS = ("mean", "last")
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)                       # L2 grid for the probes (picked on val AUROC)
TRAIN_SUBSAMPLE = 0.85                                        # every seed trains on this share of train groups


# ----------------------------------------------------------------------------------------------
# snippet sources
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SnipSource:
    name: str
    cls: str
    kind: str                                  # lg_pool | registry | builtin | mined
    loader: str = ""
    registry: str = ""                         # $LG_SOURCES/<registry>/revision.json -> local_path
    hf_repo: str = ""                          # HF hub cache fallback (offline)
    file_globs: Tuple[str, ...] = ()
    max_items: Optional[int] = None
    holdout: bool = False                      # split_mode=source: this source is test-only
    test_only: bool = False                    # test-only in every split mode (plan 5.2: LLMail-Inject is held out)
    note: str = ""


SNIP_SOURCES: List[SnipSource] = [
    SnipSource("lg_harm", "harm", "lg_pool", note="LG v2 needles/pool.jsonl.gz label=harmful; source mode holds out pool=test"),
    SnipSource("lg_benign", "harm_hard", "lg_pool", note="LG v2 needles/pool.jsonl.gz label=benign (OR-Bench, XSTest, ...)"),
    SnipSource("bipia_text", "inj", "registry", "load_bipia", "bipia", file_globs=("*text_attack_*.json",),
               note="BIPIA text attack instructions (75 train + 75 test); group = attack category"),
    SnipSource("bipia_code", "inj", "registry", "load_bipia", "bipia", file_globs=("*code_attack_*.json",),
               note="BIPIA code attack instructions (50 + 50)"),
    SnipSource("deepset_pi", "inj", "registry", "load_text_label", "deepset_pi", "deepset/prompt-injections",
               ("*.parquet", "*.csv", "*.jsonl"), note="label == 1 rows (English and German)"),
    SnipSource("llmail", "inj", "registry", "load_llmail", "llmail", "microsoft/llmail-inject-challenge",
               ("*.jsonl", "*.json", "*.parquet"), max_items=20000, holdout=True, test_only=True,
               note="LLMail-Inject submissions (subject + body); human-written adaptive attacks"),
    SnipSource("gandalf", "inj", "registry", "load_text_label", "gandalf", "Lakera/gandalf_ignore_instructions",
               ("*.parquet", "*.csv", "*.jsonl"), max_items=1000, note="every row is an injection attempt"),
    SnipSource("gen_instr", "instr", "builtin", note="instruction-heavy benign templates (README, email, recipe, "
               "policy, support, manual, trigger-word notices); group = template"),
    SnipSource("mined_instr", "instr", "mined", note="background paragraphs with >= 2 imperative sentences"),
    SnipSource("notinject", "notinject", "registry", "load_notinject", "notinject", "leolee99/NotInject",
               ("NotInject_*.json", "*.parquet", "*.jsonl"), note="339 benign prompts with trigger words"),
    SnipSource("bg_plain", "plain", "mined", note="background paragraphs with no imperative sentence"),
]
SNIP_SOURCE_BY_NAME = {s.name: s for s in SNIP_SOURCES}

# background (haystack) domains: LG v2 haystack/<domain>/docs.jsonl.gz first, then $LG_SOURCES/<name>/docs.jsonl.gz
BG_DOMAINS = ("govreport", "pg19", "arxiv", "wiki_en", "pubmed", "stack")
MINE_INSTR_DOMAINS = ("stack", "govreport", "agentdojo", "wiki_en", "pubmed", "pg19", "arxiv")


# ----------------------------------------------------------------------------------------------
# extra downloads (login node; the LG v2 entries come from the S1 registry)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Download:
    name: str
    kind: str                                  # hf_dataset | hf_model | urls
    repo_id: str = ""
    urls: Tuple[str, ...] = ()
    role: str = ""
    gated: bool = False
    verify: bool = False
    license: str = ""
    note: str = ""


_BIPIA = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/"
_NOTINJECT = "https://raw.githubusercontent.com/leolee99/PIGuard/main/datasets/"
DOWNLOADS: List[Download] = [
    Download("bipia", "urls", urls=tuple(_BIPIA + f"{k}_attack_{s}.json" for k in ("text", "code") for s in ("train", "test")),
             role="inj payloads", license="MIT", note="verified 2026-10-11 (dict: category -> list of 5 instructions)"),
    Download("notinject", "urls", urls=tuple(_NOTINJECT + f"NotInject_{k}.json" for k in ("one", "two", "three")),
             role="notinject negatives", license="MIT", note="verified 2026-10-11 (list of {prompt, word_list, category})"),
    Download("deepset_pi", "hf_dataset", "deepset/prompt-injections", role="inj payloads", license="Apache-2.0"),
    Download("llmail", "hf_dataset", "microsoft/llmail-inject-challenge", role="inj payloads (held-out source)",
             verify=True, license="MIT", note="~208k submissions; the loader reads subject/body from any json/jsonl/parquet"),
    Download("gandalf", "hf_dataset", "Lakera/gandalf_ignore_instructions", role="inj payloads", verify=True, license="MIT"),
    Download("llama31_8b", "hf_model", "meta-llama/Llama-3.1-8B-Instruct", role="backbone T1", gated=True),
    Download("qwen3_4b_2507", "hf_model", "Qwen/Qwen3-4B-Instruct-2507", role="backbone T2"),
]
# S1 registry entries this PoC reads when the LG v2 needle pool / haystack are NOT built yet
S1_DEPENDENCIES = ["qwen3guard_gen_0.6b", "qwen3guard_gen_4b", "wildguardmix", "aegis2", "toxicchat", "orbench",
                   "harmbench", "xstest", "govreport", "pg19", "arxiv", "wiki_en", "pubmed", "stack"]


def recipe(profile: Profile, split_mode: str, bg_domains, ref_tok: str, seed: int = SEED,
           sources: Optional[List[str]] = None) -> Dict:
    """The data recipe.  `bg_domains` are the domains actually found, `sources` the snippet sources actually read."""
    return {"version": "agdepth/recipe-1.1", "profile": asdict(profile), "split_mode": split_mode,
            "split_fractions": SPLIT_FRACTIONS, "depth_range": DEPTH_RANGE, "seed": seed, "bg_domains": list(bg_domains),
            "sources": sorted(sources or []), "ref_tokenizer": ref_tok,
            "insert_rule": "background span + blank line + snippet + blank line + background span; "
            "cut points snapped to whitespace", "label_rule": "window label = class of the inserted snippet"}


def recipe_hash(r: Dict) -> str:
    return hashlib.sha1(json.dumps(r, sort_keys=True).encode()).hexdigest()[:12]
