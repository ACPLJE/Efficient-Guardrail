"""D3 registry: every dataset and model Stage S1 needs, with how to fetch it.

`kind`:
  hf_model     snapshot_download(repo_id) with weight/tokenizer patterns; revision sha recorded
  hf_dataset   snapshot_download(repo_id, repo_type="dataset") -> raw parquet/jsonl kept as-is
  hf_stream    datasets.load_dataset(..., streaming=True) capped at max_docs -> docs.jsonl.gz
  github       git clone (commit sha recorded)
  manual       cannot be fetched automatically; instructions printed, status "manual"

`verify: True` marks ids I could not confirm from a primary source at writing time.  The downloader
tries them, records the exact error, and lists `alternatives`; nothing else in S1 depends on them
(they feed S2-S3 data generation, not the E0 tests or the zero-shot harness).

`gated: True` needs `HF_TOKEN` (or `huggingface-cli login`) on the LOGIN node after accepting the
licence on the model/dataset page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja", "tokenizer*",
                  "*.tiktoken", "merges.txt", "vocab*", "chat_template*", "LICENSE*", "README.md"]
MODEL_IGNORE = ["original/*", "*.gguf", "*.pth", "consolidated*", "*.bin"]


@dataclass
class Entry:
    name: str
    kind: str
    repo_id: str = ""
    config: Optional[str] = None
    split: Optional[str] = None
    role: str = ""
    gated: bool = False
    verify: bool = False
    max_docs: Optional[int] = None
    text_field: Optional[str] = None
    url: str = ""
    note: str = ""
    alternatives: List[str] = field(default_factory=list)
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    shuffle_seed: Optional[int] = None     # hf_stream: shuffle shard order + a buffer so max_docs is not the first N shards only


# ----------------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------------
DATASETS: List[Entry] = [
    # needle sources (train)
    Entry("wildguardmix", "hf_dataset", "allenai/wildguardmix", role="needle train/test (WildGuardTrain, WildGuardTest)", gated=True,
          note="configs wildguardtrain / wildguardtest; accept the licence on the dataset page"),
    Entry("aegis2", "hf_dataset", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", role="needle train/test", gated=True),
    Entry("aegis1", "hf_dataset", "nvidia/Aegis-AI-Content-Safety-Dataset-1.0", role="needle test (Aegis 1.0 test)", gated=True),
    Entry("toxicchat", "hf_dataset", "lmsys/toxic-chat", role="needle train/test; E0-6 bench", note="config toxicchat0124"),
    Entry("orbench", "hf_dataset", "bench-llm/or-bench", role="hard-benign needles (80K sample) + OR-Bench-Hard-1K test"),
    Entry("openai_moderation", "hf_dataset", "mmathys/openai-moderation-api-evaluation", role="needle test (OpenAI Mod 1,680)"),
    Entry("simplesafetytests", "hf_dataset", "Bertievidgen/SimpleSafetyTests", role="needle test (SST 100)"),
    Entry("harmbench", "hf_dataset", "walledai/HarmBench", role="needle test (HarmBench prompts 239 std)", verify=True,
          alternatives=["https://github.com/centerforaisafety/HarmBench (data/behavior_datasets/harmbench_behaviors_text_all.csv)"]),
    Entry("jbb", "hf_dataset", "JailbreakBench/JBB-Behaviors", role="needle test (JBB 100+100)", note="config behaviors"),
    Entry("xstest", "hf_dataset", "Paul/XSTest", role="over-refusal test + decoy source", alternatives=["walledai/XSTest"]),
    # background (haystack) sources
    Entry("govreport", "hf_dataset", "ccdv/govreport-summarization", role="background train (in-domain), re-screened",
          note="existing download_v1 copy is reused when present"),
    Entry("pg19", "hf_stream", "emozilla/pg19", split="train", role="background train (narrative)", max_docs=6000, text_field="text",
          note="parquet mirror of deepmind/pg19 (the original is a script dataset, unsupported by datasets>=3); 6k books >> 20M tokens",
          alternatives=["deepmind/pg19 (script-based)"]),
    Entry("wildchat", "hf_stream", "allenai/WildChat-1M", split="train", role="background train (chat logs)", gated=True,
          max_docs=150000, text_field="conversation", note="keep language=='English' and toxic==False rows"),
    Entry("stack", "hf_stream", "bigcode/the-stack-smol", split="train", role="background train (code)", verify=True, gated=True,
          max_docs=120000, text_field="content", shuffle_seed=0,   # 30 languages x 10k rows: shuffle shards so all languages appear
          note="gated: accept the terms at https://huggingface.co/datasets/bigcode/the-stack-smol, then huggingface-cli login",
          alternatives=["bigcode/the-stack-v2-dedup (gated, huge)", "codeparrot/github-code-clean"]),
    Entry("arxiv", "hf_stream", "ccdv/arxiv-summarization", config="document", split="train", role="background train (papers)",
          max_docs=40000, text_field="article", note="verified streaming 2026-09-17 (article + abstract columns)",
          alternatives=["armanc/scientific_papers (script-based)", "togethercomputer/RedPajama-Data-1T-Sample"]),
    Entry("wiki_en", "hf_stream", "wikimedia/wikipedia", config="20231101.en", split="train", role="background OOD test",
          max_docs=60000, text_field="text"),
    Entry("wiki_zh", "hf_stream", "wikimedia/wikipedia", config="20231101.zh", split="train", role="background OOD test",
          max_docs=60000, text_field="text"),
    Entry("pubmed", "hf_stream", "MedRAG/pubmed", split="train", role="background OOD test (medical)", verify=True,
          max_docs=200000, text_field="content", alternatives=["qiaojin/PubMedQA (config pqa_artificial, field context)"]),
    Entry("ledgar", "hf_dataset", "coastalcph/lex_glue", config="ledgar", role="background OOD test (legal)", verify=True,
          alternatives=["theatticusproject/cuad-qa"]),
    Entry("agentdojo", "github", url="https://github.com/ethz-spylab/agentdojo", role="background OOD test (tool outputs)",
          note="benign tool outputs are generated from the suites (pip install agentdojo); see S3 synth notes"),
    # external long-context benches
    Entry("safetyniah", "hf_dataset", "caskcsg/SafetyNIAH", role="E2-3 external long bench (30,400 rows)"),
    Entry("longsafetybench", "hf_dataset", "LutherXD/LongSafetyBench", role="E2-5 HarmfulNIAH (4K-100K) + ManyShotJailbreak",
          note="parquet per task: data/HarmfulNIAH-*.parquet, data/ManyShotJailbreak-*.parquet, DocAttack, ... (verified 2026-09-17)"),
    Entry("ninja", "manual", url="https://arxiv.org/abs/2511.04707", role="E2-5 NINJA 960", verify=True),
    Entry("msj", "manual", url="https://www.anthropic.com/research/many-shot-jailbreaking", role="E2-5 MSJ 320",
          note="covered by LongSafetyBench data/ManyShotJailbreak-*.parquet (same download as longsafetybench); construct extra rows in S3 only if needed"),
]

# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
MODELS: List[Entry] = [
    Entry("qwen3guard_gen_0.6b", "hf_model", "Qwen/Qwen3Guard-Gen-0.6B", role="LG backbone (main); B1"),
    Entry("qwen3guard_gen_4b", "hf_model", "Qwen/Qwen3Guard-Gen-4B", role="LG backbone; B1"),
    Entry("qwen3guard_gen_8b", "hf_model", "Qwen/Qwen3Guard-Gen-8B", role="LG backbone; B1; CD; screening"),
    Entry("qwen3guard_stream_8b", "hf_model", "Qwen/Qwen3Guard-Stream-8B", role="B1 (token-level head); E6-6 init", note="trust_remote_code"),
    Entry("llamaguard3_1b", "hf_model", "meta-llama/Llama-Guard-3-1B", role="optional backbone; v1 screener", gated=True),
    Entry("llamaguard3_8b", "hf_model", "meta-llama/Llama-Guard-3-8B", role="LG backbone (cross-family); B1; screening", gated=True),
    Entry("llamaguard4_12b", "hf_model", "meta-llama/Llama-Guard-4-12B", role="B1", gated=True, note="Llama4ForConditionalGeneration + AutoProcessor"),
    Entry("wildguard_7b", "hf_model", "allenai/wildguard", role="B1", gated=True),
    Entry("guardreasoner_8b", "hf_model", "yueliu1999/GuardReasoner-8B", role="B1 (reasoning)"),
    Entry("polyguard_qwen_7b", "hf_model", "ToxicityPrompts/PolyGuard-Qwen", role="B1", note="7B; the id has no -7B suffix (verified 2026-09-17)"),
    Entry("granite_guardian_3.3_8b", "hf_model", "ibm-granite/granite-guardian-3.3-8b", role="B1"),
    Entry("gpt_oss_safeguard_20b", "hf_model", "openai/gpt-oss-safeguard-20b", role="B1 (131k native)",
          note="MXFP4 weights; needs transformers>=4.55 + kernels; ~13 GB"),
    Entry("nemotron_safety_guard_v3", "hf_model", "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3", role="B1 (8K ctx, truncation regime)", gated=True),
    Entry("shieldgemma_9b", "hf_model", "google/shieldgemma-9b", role="B1 optional (Yes/No prob)", gated=True),
    Entry("bingoguard_8b", "hf_model", "fanyin3639/bingoguard-llama-8b", role="B1 optional (4k ctx)",
          note="author release (Fan Yin), not gated (verified 2026-09-17)", alternatives=["fanyin3639/bingoguard-phi3-3B"]),
    Entry("prompt_guard_2_86m", "hf_model", "meta-llama/Llama-Prompt-Guard-2-86M", role="B6 small encoder", gated=True),
    Entry("gliguard", "hf_model", "fastino/gliguard-LLMGuardrails-300M", role="B6 small encoder",
          note="GLiClass-style encoder guard (verified id 2026-09-17); inference API to be wired in S6 (B6)"),
    Entry("bge_m3", "hf_model", "BAAI/bge-m3", role="related-subset embeddings (S3)",
          allow_patterns=MODEL_PATTERNS + ["pytorch_model.bin"],
          ignore_patterns=["original/*", "*.gguf", "*.pth", "consolidated*", "onnx/*", "imgs/*", "*.pt"],
          note="the Hub repo has no safetensors (2026-09-18: pytorch_model.bin 2.3 GB only) -> the default *.bin ignore "
               "left an empty snapshot and embed died in from_pretrained; re-fetch with --only bge_m3 --no-skip"),
    Entry("qwen3_8b", "hf_model", "Qwen/Qwen3-8B", role="E5 serving target; B8 summariser"),
]


def by_name(name: str) -> Entry:
    for e in DATASETS + MODELS:
        if e.name == name:
            return e
    raise KeyError(name)


def all_entries() -> Dict[str, Entry]:
    return {e.name: e for e in DATASETS + MODELS}
