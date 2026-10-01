"""Stage S2 configuration: paths, the 10 background domains, the 13 needle sources, split ratios, and the
synthesis recipe.  Every number here comes from the experiment report §5 / plan S2–S3; change them here only
(the recipe dict is hashed into every synthesized row as `recipe_hash`, so an edit is visible in the data).

Directory layout (report §5.8), rooted at $LG_DATA_V2 (default: parent of $LG_SOURCES):
    sources/<name>/                 D3 downloads (revision.json)               [S1]
    haystack/<domain>/docs.jsonl.gz background document pieces + paragraphs    [S2 D4]
    screen/<domain>.<model>.jsonl.gz chunk-level screening rows                [S2 D4]
    screen/screen_summary.json, screen/manual_review_50.md, screen/keep_<domain>.json
    related/<domain>.emb.npy + .chunks.jsonl.gz   bge-m3 chunk embeddings      [S2 D4 -> S3 related subset]
    needles/pool.jsonl.gz, needles/pool_stats.json                             [S2 D5]
    splits/needles.json, splits/haystack.json, splits/audit.json               [S2 D5]
    synth_dryrun/<split>/<subset>/rows.jsonl.gz                                 [S2 D6 dry-run]
    synth/<split>/<subset>/rows.jsonl.gz + manifests/*.manifest.json           [S3 D6–D7]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from lgs1.paths import S1Paths, env, load_env_file

# ----------------------------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------------------------
@dataclass
class S2Paths:
    root: Path              # lg_data_v2
    sources: Path
    haystack: Path
    screen: Path
    related: Path
    needles: Path
    splits: Path
    synth: Path
    synth_dryrun: Path
    manifests: Path
    out: Path               # $LG_OUT (logs, slurm scripts, reports)
    hf_home: Path
    python: str
    s1: S1Paths

    @classmethod
    def load(cls) -> "S2Paths":
        p1 = S1Paths.load()
        root = Path(env("LG_DATA_V2", str(p1.sources.parent)))
        return cls(root=root, sources=p1.sources, haystack=root / "haystack", screen=root / "screen",
                   related=root / "related", needles=root / "needles", splits=root / "splits", synth=root / "synth",
                   synth_dryrun=root / "synth_dryrun", manifests=root / "manifests", out=p1.out, hf_home=p1.hf_home,
                   python=p1.python, s1=p1)

    def ensure(self) -> "S2Paths":
        for d in (self.haystack, self.screen, self.related, self.needles, self.splits, self.synth_dryrun,
                  self.manifests, self.out / "logs", self.out / "slurm", self.out / "reports"):
            d.mkdir(parents=True, exist_ok=True)
        return self


# ----------------------------------------------------------------------------------------------
# tokenizers (the length grid is defined on Qwen3 tokens; Llama 3 counts are recorded alongside)
# ----------------------------------------------------------------------------------------------
TOKENIZERS: Dict[str, str] = {
    "qwen3": "Qwen/Qwen3Guard-Gen-0.6B",          # same vocabulary as every Qwen3Guard size
    "llama3": "meta-llama/Llama-Guard-3-8B",
}

# ----------------------------------------------------------------------------------------------
# background (haystack) domains — report §5.3
# ----------------------------------------------------------------------------------------------
@dataclass
class Domain:
    name: str
    role: str                    # "train" (also test in-domain) | "ood" (test only)
    source: str                  # registry entry name under sources/
    budget_tokens: int           # target pool size after screening (Qwen3 tokens)
    text_field: str = "text"     # hf_stream docs.jsonl.gz always use "text"; parquet kinds name their column here
    kind: str = "stream"         # stream (sources/<name>/docs.jsonl.gz) | parquet (HF cache snapshot) | agentdojo
    hf_repo: str = ""            # for kind=parquet
    hf_config_dir: str = ""      # sub-directory pattern inside the snapshot (e.g. "ledgar")
    paragraph: str = "blank"     # blank (\n\n) | line (\n) | code (blank line; never split inside a line)
    max_piece_tokens: int = 8192 # long documents are cut into pieces at paragraph boundaries
    min_doc_tokens: int = 64
    filters: Dict = field(default_factory=dict)   # e.g. {"language": "English", "toxic": False} on meta
    note: str = ""


DOMAINS: List[Domain] = [
    Domain("govreport", "train", "govreport", 20_000_000, text_field="report", kind="parquet",
           hf_repo="ccdv/govreport-summarization", note="re-screened with the 8B pair (v1 used LG3-1B only)"),
    Domain("pg19", "train", "pg19", 20_000_000, note="books; pieces of <=8k tokens"),
    Domain("wildchat", "train", "wildchat", 20_000_000, paragraph="line",
           filters={"language": "English", "toxic": False}, note="one conversation = one doc; English, non-toxic flag"),
    Domain("stack", "train", "stack", 15_000_000, paragraph="code",
           note="the-stack-smol (permissive); insertion only at blank lines"),
    Domain("arxiv", "train", "arxiv", 15_000_000, note="ccdv/arxiv-summarization article (streamed as text)"),
    Domain("wiki_en", "ood", "wiki_en", 10_000_000, note="SafetyNIAH background domain -> test only"),
    Domain("wiki_zh", "ood", "wiki_zh", 5_000_000, note="LongGuard filler domain"),
    Domain("pubmed", "ood", "pubmed", 5_000_000, note="MedRAG/pubmed content (sensitive-topic benign)"),
    Domain("agentdojo", "ood", "agentdojo", 3_000_000, kind="agentdojo", paragraph="line",
           note="benign tool outputs rendered from the AgentDojo suite environments (JSON / email / calendar)"),
    Domain("ledgar", "ood", "ledgar", 5_000_000, text_field="text", kind="parquet", hf_repo="coastalcph/lex_glue",
           hf_config_dir="ledgar", note="contract clauses (short) -> many per document"),
]
DOMAIN_BY_NAME: Dict[str, Domain] = {d.name: d for d in DOMAINS}
TRAIN_DOMAINS = [d.name for d in DOMAINS if d.role == "train"]
OOD_DOMAINS = [d.name for d in DOMAINS if d.role == "ood"]

SCREEN = {
    "chunk_tokens": 512,                      # report §5.3
    "models": ["llamaguard3_8b", "qwen3guard_gen_8b"],   # GuardSpecs from lgs1.zeroshot.guards
    "qwen3guard_mode": "strict",              # controversial counts as unsafe
    "collect_factor": 1.35,                   # pool 35% more than the budget so rejections still leave the budget
    "manual_review_docs": 50,
    "batch_size": 16,
}

# ----------------------------------------------------------------------------------------------
# needle sources — report §5.2 (pool = train | test; public test sources never enter train/val/calib)
# ----------------------------------------------------------------------------------------------
@dataclass
class NeedleSource:
    name: str
    pool: str                   # "train" | "test"
    registry: str               # sources/<name> entry (hf_dataset) or "" for external dirs
    hf_repo: str
    loader: str                 # function name in lgs2.needles
    file_globs: List[str]
    expected_n: Optional[int] = None    # published size (informational; mismatch -> warning, not failure)
    note: str = ""


NEEDLE_SOURCES: List[NeedleSource] = [
    NeedleSource("wildguardtrain", "train", "wildguardmix", "allenai/wildguardmix", "load_wildguard",
                 ["*wildguard_train*.parquet", "*wildguardtrain*.parquet", "train/*.parquet"], 86_759,
                 "prompt + prompt_harm_label (+adversarial); 48,783 prompts once response-pair duplicates collapse"),
    NeedleSource("aegis2_train", "train", "aegis2", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "load_aegis2",
                 ["train.json", "validation.json", "*train*.parquet", "*validation*.parquet"], None,
                 "prompt_label safe/unsafe; 'Needs Caution' excluded; main branch ships train/validation/test.json"),
    NeedleSource("toxicchat_train", "train", "toxicchat", "lmsys/toxic-chat", "load_toxicchat",
                 ["*0124*train*.csv", "*0124*/*train*.csv"], 5_082, "toxicity label, jailbreaking flag kept"),
    NeedleSource("orbench_80k", "train", "orbench", "bench-llm/or-bench", "load_orbench_80k",
                 ["or-bench-80k.csv", "*or-bench-80k*.parquet", "*or-bench-80k*/*.parquet"], 10_000, "10,000 sampled from the 80K (hard benign)"),
    NeedleSource("wgtest", "test", "wildguardmix", "allenai/wildguardmix", "load_wildguard",
                 ["*wildguard_test*.parquet", "*wildguardtest*.parquet", "test/*.parquet"], 1_725, "prompt-labelled rows"),
    NeedleSource("toxicchat_test", "test", "toxicchat", "lmsys/toxic-chat", "load_toxicchat",
                 ["*0124*test*.csv", "*0124*/*test*.csv"], 2_853, "human-annotated rows only"),
    NeedleSource("harmbench", "test", "harmbench", "walledai/HarmBench", "load_harmbench",
                 ["standard/*.parquet", "contextual/*.parquet", "*.parquet", "*.csv"], None,
                 "standard (200) + contextual prompts; copyright config skipped; config kept in meta"),
    NeedleSource("sst", "test", "simplesafetytests", "Bertievidgen/SimpleSafetyTests", "load_sst",
                 ["*.parquet", "*.csv"], 100, "all harmful"),
    NeedleSource("jbb", "test", "jbb", "JailbreakBench/JBB-Behaviors", "load_jbb",
                 ["harmful-behaviors.csv", "benign-behaviors.csv", "*behaviors*/*.parquet", "*harmful*.csv", "*benign*.csv"], 200,
                 "100 harmful + 100 benign goals"),
    NeedleSource("openai_mod", "test", "openai_moderation", "mmathys/openai-moderation-api-evaluation", "load_openai_mod",
                 ["*.parquet", "*.jsonl", "*.jsonl.gz", "*.csv"], 1_680, "harmful if any category flag == 1"),
    NeedleSource("aegis2_test", "test", "aegis2", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "load_aegis2",
                 ["test.json", "*test*.parquet"], 1_680, "1,984 rows minus 'Needs Caution' and duplicate prompts = 1,680 (2026-09-17)"),
    NeedleSource("xstest", "test", "xstest", "Paul/XSTest", "load_xstest", ["*.parquet", "*.csv"], 450, "250 safe + 200 unsafe"),
    NeedleSource("orbench_hard", "test", "orbench", "bench-llm/or-bench", "load_orbench_hard",
                 ["or-bench-hard-1k.csv", "*or-bench-hard-1k*.parquet", "*or-bench-hard-1k*/*.parquet"], 1_319, "benign (over-refusal)"),
    NeedleSource("aegis1_test", "test", "aegis1", "nvidia/Aegis-AI-Content-Safety-Dataset-1.0", "load_aegis1",
                 ["*test*.parquet"], None, "user_message rows, majority label; Needs Caution excluded"),
]
NEEDLE_SOURCE_BY_NAME = {s.name: s for s in NEEDLE_SOURCES}
TEST_SOURCES = [s.name for s in NEEDLE_SOURCES if s.pool == "test"]
TRAIN_SOURCES = [s.name for s in NEEDLE_SOURCES if s.pool == "train"]

# external long-context needle sets the train pool is audited against (exact + near-dup), report §5.6
EXTERNAL_AUDIT = {
    "safetyniah": {"registry": "safetyniah", "hf_repo": "caskcsg/SafetyNIAH",
                   "columns": ["origin_prompt", "origin_response"], "note": "LongGuard needles (17 public test sets)"},
    "longsafetybench_harmfulniah": {"registry": "longsafetybench", "hf_repo": "LutherXD/LongSafetyBench",
                                    "columns": ["reference"], "file_glob": "*HarmfulNIAH*.parquet", "note": "optional"},
}

LENGTH_BUCKETS = [("le32", 0, 32), ("33_64", 33, 64), ("65_128", 65, 128), ("129_512", 129, 512)]
NEEDLE_MAX_TOK = 128           # = b: guarantees whole containment in one 256-token window (Prop 1 premise)
LONG_NEEDLE_MAX_TOK = 512
ORBENCH_80K_SAMPLE = 10_000

MINHASH = {"shingle_words": 5, "num_perm": 128, "bands": 32, "jaccard": 0.8, "seed": 20260917}

# ----------------------------------------------------------------------------------------------
# splits — report §5.6
# ----------------------------------------------------------------------------------------------
SPLITS = {
    "seed": 20260917,
    # train-pool needle GROUPS -> val / calib / train (test sources are test only)
    "needle_group_fractions": {"val": 0.04, "calib": 0.04},
    # train-domain background DOCS -> split by token share (OOD domains are 100% test)
    "haystack_fractions": {"test": 0.30, "calib": 0.15, "val": 0.05},   # remainder -> train
}

# ----------------------------------------------------------------------------------------------
# synthesis recipe — report §5.4 / §5.7, plan S3 D6.  Hashed into every row (recipe_hash).
# ----------------------------------------------------------------------------------------------
L_GRID_ALL = [1024, 2048, 4096, 8192, 16384, 28672, 65536, 131072]   # 28k = 28,672 content-token cap
RECIPE_V2 = {
    "version": "lg_data_v2/recipe-2.0",
    "seed": 20260917,
    "tokenizer": TOKENIZERS["qwen3"],
    "window_b": 128,
    "needle_max_tok": NEEDLE_MAX_TOK,
    "length_tol": 0.10,                      # |n_tok - L_target| <= 10% (gate E0-4 / D8)
    "twin_len_tol": 0.20,                    # benign twin needle length within ±20% of the harmful needle
    "insert_rule": "nearest paragraph boundary to depth*len(background); code: blank lines only",
    "background_rule": "documents of ONE domain concatenated with blank lines; every haystack doc id recorded; "
                       "a background document belongs to exactly one split",
    "label_rule": "presence label = label of the inserted needle (not a moderation verdict of the whole document)",
    "sets": {
        # pairs = harmful needle + benign twin (2 rows per pair per (L, depth))
        "train":   {"subset": "main", "pairs": 2000, "L": [1024, 2048, 4096, 8192],
                    "positions": "random2+ends", "benign_long": 0},
        "train16k": {"subset": "main16k", "pairs": 500, "L": [16384], "positions": "random2", "benign_long": 0},
        "val":     {"subset": "main", "pairs": 300, "L": [1024, 2048, 4096, 8192],
                    "positions": [0.0, 0.5, 1.0], "benign_long": 300},          # 300 per L = 1,200
        "calib":   {"subset": "main", "pairs": 200, "L": L_GRID_ALL,
                    "positions": [0.0, 0.25, 0.5, 0.75, 1.0], "benign_long": 500},  # 500 per L = 4,000
        "test":    {"subset": "main", "pairs": 200, "L": L_GRID_ALL,
                    "positions": [round(0.1 * i, 1) for i in range(11)], "benign_long": 500},   # 500 per L = 4,000
    },
    "native": {"train": "all", "val": "all"},   # needle text as-is (E1 moderation task); test native = public benches
    "domains_per_split": {"train": TRAIN_DOMAINS, "val": TRAIN_DOMAINS, "calib": TRAIN_DOMAINS, "test": TRAIN_DOMAINS},
    "benign_long_domains": {"test": TRAIN_DOMAINS + OOD_DOMAINS, "calib": TRAIN_DOMAINS, "val": TRAIN_DOMAINS},
}

def missing_depths(rows, depths) -> List[float]:
    """Requested insertion depths that no row carries.  The test grid is 0.0, 0.1, ..., 1.0 (11 positions) and calib/main is
    {0, .25, .5, .75, 1}: `--depths 0,0.25,0.5,0.75,1` on test rows quietly covered 3 positions, not 5 (found 2026-09-27:
    the S8 ablation subset, the B1-long 64k/128k dense subset, the B2 chunk sweep).  Callers warn or refuse."""
    have = {round(float(r["depth"]), 4) for r in rows if r.get("depth") is not None}
    return [d for d in depths if round(float(d), 4) not in have]


def depth_warning(rows, depths, where: str) -> str:
    """'' when every requested depth matches a row, else the one-line warning the runners print."""
    miss = missing_depths(rows, depths) if depths else []
    if not miss:
        return ""
    have = sorted({round(float(r["depth"]), 4) for r in rows if r.get("depth") is not None})
    return (f"[depths] WARNING {where}: --depths {list(miss)} match no row (rows cover {have}); the test grid is "
            f"0.0, 0.1, ..., 1.0 -- use grid values (e.g. 0,0.2,0.5,0.8,1) for 5 positions")


SYNTH_ROW_FIELDS = ["id", "split", "subset", "L_target", "depth", "needle_id", "twin_id", "pair_id", "haystack_domain",
                    "haystack_doc_ids", "needle_char_span", "label", "needle_share", "n_tok_qwen3", "n_tok_needle",
                    "recipe_hash", "text"]


def data_root() -> Path:
    load_env_file()
    return S2Paths.load().root
