"""AG asset registry (build plan 4.2 S0; plan 5.2, 6.1, 6.2, 7.2; SPEC 9.2).

Every model, dataset, trace and baseline code base AG needs, with how the login node fetches it
(`ag/data/download_ag.py`, `slurm/ag_download_login.sh`). Entries reuse `lgs1.registry.Entry` (D3), so
the LG fetchers in `lgs1/download.py` take them unchanged.

`kind` (same as LG):
  hf_model     snapshot_download(repo_id) with weight/tokenizer patterns; revision sha recorded
  hf_dataset   snapshot_download(repo_id, repo_type="dataset"); raw files kept as-is
  hf_stream    datasets.load_dataset(..., streaming=True) capped at max_docs -> docs.jsonl.gz
  github       git clone --depth 1 (commit sha recorded)
  manual       not fetched automatically; status "manual" with the url / note

`stage` says when an asset is first needed:
  pilot     S2 step 0 pilot run (T2 model, background documents)
  s2        S2 E0 critical path (E0.1, E0.2, W0-grader, LC-Inject sources, Mooncake)
  s3        S3 / S3b / S3c (T3, harm path, baseline code)
  later     S4 and after (E3.6, E6.2, held-out sets)
  optional  only if time allows (T4, small judge variants)

`verify: True` marks ids or formats not confirmed from a primary source at writing time (no network
from the machine this was written on). The downloader tries them, records the exact error and lists
`alternatives`. Ids taken from the plan's Sources list (plan 13) say so in `note`.

`gated: True` needs `HF_TOKEN` (or `huggingface-cli login`) on the LOGIN node after the licence was
accepted on the model page (F-10).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lgs1.registry import MODEL_IGNORE, MODEL_PATTERNS, Entry

KINDS = ("hf_model", "hf_dataset", "hf_stream", "github", "manual")
STAGES = ("pilot", "s2", "s3", "later", "optional")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class AGEntry(Entry):
    stage: str = "later"
    ag_role: str = ""       # plan id: T1-T4, J1-J4, grader, harm_guard, B2-B5, W-QA source, ...


_NO_BIN = [p for p in MODEL_IGNORE if p != "*.bin"]

# ----------------------------------------------------------------------------------------------
# models (plan 6.1, 6.2)
# ----------------------------------------------------------------------------------------------
MODELS: list[AGEntry] = [
    AGEntry("t1_llama31_8b_instruct", "hf_model", "meta-llama/Llama-3.1-8B-Instruct", gated=True, stage="s2",
            ag_role="T1", role="main target (plan 6.1); 32/32/8, ctx 131072",
            note="gated (F-10); D-006: until approved, E0.1 runs T2-only"),
    AGEntry("t2_qwen3_4b_instruct_2507", "hf_model", "Qwen/Qwen3-4B-Instruct-2507", stage="pilot",
            ag_role="T2", role="small target (plan 6.1); 36/32/8, ctx 262144; S2 step-0 pilot target; J2 base model"),
    AGEntry("t3_mistral_small_32_24b", "hf_model", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", stage="s3",
            verify=True, ag_role="T3", role="large target, tool use (plan 6.1); 40/32/8, ctx 131072, 2 GPUs",
            note="multimodal Mistral3 checkpoint: confirm the HF-format weights (model-*.safetensors) exist next to "
                 "consolidated.safetensors (ignored by MODEL_IGNORE) and whether the tokenizer is tokenizer.json or "
                 "tekken.json (mistral-common); a snapshot without weight files is recorded as failed",
            alternatives=["mistralai/Mistral-Small-3.1-24B-Instruct-2503 (previous release, same size class)"]),
    AGEntry("t4_gpt_oss_20b", "hf_model", "openai/gpt-oss-20b", stage="optional", ag_role="T4",
            role="optional target, sliding/full alternation (plan 6.1); 24/64/8, ctx 131072",
            ignore_patterns=MODEL_IGNORE + ["metal/*"],
            note="MXFP4 weights; needs transformers>=4.55 + kernels (same as LG gpt_oss_safeguard_20b)"),
    AGEntry("grader_qwen3_8b", "hf_model", "Qwen/Qwen3-8B", stage="s2", ag_role="grader",
            role="attack-success LLM grader (D-005, plan 5.3 #7)",
            note="same repo as LG qwen3_8b: a no-op when the shared HF cache already has it"),
    AGEntry("harm_qwen3guard_gen_0_6b", "hf_model", "Qwen/Qwen3Guard-Gen-0.6B", stage="s3", ag_role="harm_guard",
            role="harm path backbone (LG, design 1; plan 6.2); ctx 32768",
            note="same repo as LG qwen3guard_gen_0.6b"),
    AGEntry("outguard_qwen3guard_stream_0_6b", "hf_model", "Qwen/Qwen3Guard-Stream-0.6B", stage="later", verify=True,
            ag_role="output_guard", role="output guard candidate (plan 6.2); token-level head, ctx 8192",
            note="trust_remote_code (like LG qwen3guard_stream_8b); the 0.6B id is not confirmed",
            alternatives=["Qwen/Qwen3Guard-Stream-4B", "Qwen/Qwen3Guard-Stream-8B"]),
    AGEntry("j1_prompt_guard2_86m", "hf_model", "meta-llama/Llama-Prompt-Guard-2-86M", gated=True, stage="s2",
            ag_role="J1", role="judge J1, 512-token window classifier (plan 6.2)",
            note="gated (F-10); D-006: until approved, E0.2 runs J2-only"),
    AGEntry("prompt_guard2_22m", "hf_model", "meta-llama/Llama-Prompt-Guard-2-22M", gated=True, stage="optional",
            ag_role="J1-small", role="smaller J1 variant (plan 6.2 latency note)"),
    AGEntry("j2_agentwatcher_monitor", "hf_model", "SecureLLMSys/AgentWatcher-Qwen3-4B-Instruct-2507", stage="s2",
            verify=True, ag_role="J2", role="judge J2: AgentWatcher monitor (Qwen3-4B-Instruct-2507 LoRA, plan 6.2)",
            url="https://arxiv.org/abs/2604.01194",
            allow_patterns=MODEL_PATTERNS + ["adapter_model.bin"], ignore_patterns=_NO_BIN + ["pytorch_model*.bin"],
            note="id from plan 13 Sources; LoRA adapter vs merged weights not confirmed (base = T2)",
            alternatives=["https://github.com/wang-yanting/AgentWatcher (code, monitor prompt)"]),
    AGEntry("j3_piguard", "hf_model", "leolee99/PIGuard", stage="later", verify=True, ag_role="J3",
            role="judge J3 (DeBERTa-v3-base 184M, plan 6.2; E6.2)",
            allow_patterns=MODEL_PATTERNS + ["pytorch_model.bin"], ignore_patterns=_NO_BIN,
            note="HF id guessed from the code repo; may ship pytorch_model.bin only (cf. LG bge_m3), so *.bin is allowed",
            alternatives=["https://github.com/leolee99/PIGuard (plan 13)", "leolee99/InjecGuard"]),
    AGEntry("j4_datasentinel", "manual", stage="later", ag_role="J4", url="https://arxiv.org/abs/2504.11358",
            role="judge J4 (Mistral-7B fine-tune, plan 6.2; E6.2)", verify=True,
            note="checkpoint location unknown; check the paper and the Open-Prompt-Injection README (same authors)"),
]

# ----------------------------------------------------------------------------------------------
# datasets, traces, background (plan 5.2, 5.3, 5.5)
# ----------------------------------------------------------------------------------------------
DATASETS: list[AGEntry] = [
    AGEntry("bg_wiki_en", "hf_stream", "wikimedia/wikipedia", config="20231101.en", split="train", text_field="text",
            max_docs=20000, stage="pilot", ag_role="background",
            role="background documents for the S2 step-0 draft LC-Inject pilot (plan 5.3 #2)",
            note="same writer as LG wiki_en: docs.jsonl.gz rows {id, text, meta}; first max_docs rows of the stream"),
    AGEntry("injecagent", "github", url="https://github.com/uiuc-kang-lab/InjecAgent", stage="s2",
            ag_role="LC-Inject source", role="tool-response injection, 1,054 cases (plan 5.2); tool sink"),
    AGEntry("bipia", "github", url="https://github.com/microsoft/BIPIA", stage="s2", ag_role="LC-Inject source",
            role="email/web/table QA, summarisation, code QA (plan 5.2); stream sink",
            note="some BIPIA tasks are built from external corpora by its scripts; check benchmark/ after the clone"),
    AGEntry("open_prompt_injection", "github", url="https://github.com/liu00222/Open-Prompt-Injection", stage="s2",
            ag_role="payloads", role="attack strategies x tasks toolkit (plan 5.2, 5.3 #3); payload generator"),
    AGEntry("longpibench", "hf_dataset", "RainWatcher/LongPIBench", stage="s2", verify=True,
            ag_role="LC-Inject source", role="long-input PI benchmark, public synthetic 4 x 100 (plan 5.2)",
            note="id from plan 13 Sources; file layout unknown (see probe.json)",
            alternatives=["https://github.com/liu00222/LongPIBench (code)", "https://arxiv.org/abs/2608.28411"]),
    AGEntry("piarena", "github", url="https://github.com/sleeepeer/PIArena", stage="s2", verify=True,
            ag_role="LC-Inject source", role="13 datasets / 1,700 samples, long sets + task-aligned attacks (plan 5.2)",
            note="url from plan 13 Sources; the data may live outside the repo (HF), see its README",
            alternatives=["https://arxiv.org/abs/2604.08499"]),
    AGEntry("agentdojo", "manual", stage="s2", ag_role="LC-Inject source (AgentDojo-L)",
            url="https://github.com/ethz-spylab/agentdojo",
            role="tool-agent suites, 97 tasks / 629 security cases (plan 5.2); AgentDojo-L",
            note="pip install agentdojo==0.1.35 into the ag311 venv on the login node (F-15); suites ship in the package"),
    AGEntry("mooncake", "github", url="https://github.com/kvcache-ai/Mooncake", stage="s2", ag_role="W-QA/W-Agent/W-MT",
            role="FAST '25 traces: conversation / toolagent arrival + length + hash_ids (plan 5.5; W0-mooncake)",
            note="traces under FAST25-release/traces/ (F-11)"),
    AGEntry("tasktracker", "github", url="https://github.com/microsoft/TaskTracker", stage="later",
            ag_role="depth negatives", role="clean/poisoned pair notebooks, instruction-heavy negatives (plan 5.2, 5.4)"),
    AGEntry("llmail_inject", "hf_dataset", "microsoft/llmail-inject-challenge", stage="later", verify=True,
            ag_role="held-out", role="208,095 human adaptive attack submissions, never used for calibration (plan 5.2)",
            note="id also listed in plan 13 Sources"),
    AGEntry("notinject", "hf_dataset", "leolee99/NotInject", stage="later", verify=True, ag_role="FP stress",
            role="339 benign prompts with trigger words (plan 5.2, 5.4)", note="id also listed in plan 13 Sources"),
    AGEntry("longpibench_code", "github", url="https://github.com/liu00222/LongPIBench", stage="later", verify=True,
            ag_role="LC-Inject source", role="LongPIBench generation code (plan 13)",
            note="only needed if the HF data (longpibench) lacks the task / judge definitions"),
    AGEntry("copilot_agent_traces", "manual", stage="later", verify=True, ag_role="W-Agent tool wait",
            url="https://github.com/Azure/AzurePublicDataset",
            role="Copilot coding-agent traces 2026: per-tool-call durations (plan 5.2, 5.5)",
            note="trace files are linked from the repo README (blob storage), not in the git tree; confirm the format"),
    AGEntry("tracelab", "manual", stage="later", verify=True, ag_role="W-Agent tool wait",
            url="https://github.com/uw-syfi/TraceLab", role="tool-call durations (plan 5.2, 5.5)",
            note="alternative to the Copilot traces; confirm licence and format before use"),
    AGEntry("resumeshield", "manual", stage="later", ag_role="E3.6", url="https://github.com/jbarach2012/ResumeShield",
            role="104 resumes, 72 with hidden instructions (plan 5.2; E3.6)"),
    AGEntry("crackedpdfs", "manual", stage="later", ag_role="E3.6",
            url="https://huggingface.co/datasets/volkthienpreecha/crackedpdfs",
            role="29,322 PDFs, 9,774 injected (plan 5.2; E3.6)",
            note="large; fetch the subset E3.6 needs by hand instead of a full snapshot"),
    AGEntry("lg_v2_harm_data", "manual", stage="s3", ag_role="harm path data",
            role="LG v2 harmful-needle data (plan 5.2, 5.3 #8; E3.2, E3.5)",
            note="internal: produced by LG S2-S3 under $LG_SOURCES / $LG_OUT; reused as-is, not downloaded"),
]

# ----------------------------------------------------------------------------------------------
# baseline code (plan 7.2, SPEC 9.2; ported in S3c)
# ----------------------------------------------------------------------------------------------
BASELINES: list[AGEntry] = [
    AGEntry("pishield_code", "github", url="https://github.com/weizou52/PIShield", stage="s3", ag_role="B2",
            role="B2: Llama-3.1-8B layer-13 linear probe (plan 7.2)", note="url from plan 13 Sources"),
    AGEntry("attntrace_code", "github", url="https://github.com/Wang-Yanting/AttnTrace", stage="s3", ag_role="B3",
            role="B3: AttnTrace attribution (separate Python 3.10.15 venv, SPEC 9.2 / 13)",
            note="url from plan 13 Sources; F-16: Python 3.10.15 + torch 2.5.1 pinned"),
    AGEntry("agentwatcher_code", "github", url="https://github.com/wang-yanting/AgentWatcher", stage="s3",
            ag_role="B3/J2", role="B3: AgentWatcher original pipeline; J2 monitor prompt (plan 7.2)",
            note="url from plan 13 Sources"),
    AGEntry("pisanitizer_code", "github", url="https://github.com/sleeepeer/PISanitizer", stage="s3", ag_role="B4",
            role="B4: PISanitizer (plan 7.2)", note="url from plan 13 Sources"),
    AGEntry("rennervate", "manual", stage="s3", verify=True, ag_role="B4", url="https://arxiv.org/abs/2512.08417",
            role="B4: Rennervate, re-implemented from the paper (plan 7.2)", note="code release unknown (F-09)"),
    AGEntry("melon_code", "github", url="https://github.com/kaijiezhu11/MELON", stage="s3", ag_role="B5",
            role="B5: MELON, runs inside AgentDojo (plan 7.2)", note="url from plan 13 Sources"),
    AGEntry("attention_tracker_code", "github", url="https://github.com/khhung-906/Attention-Tracker", stage="later",
            ag_role="related", role="Attention Tracker (closest attention-based detector; Table 1)",
            note="url from plan 13 Sources"),
]


def all_entries() -> dict[str, AGEntry]:
    return {e.name: e for e in MODELS + DATASETS + BASELINES}


def by_name(name: str) -> AGEntry:
    try:
        return all_entries()[name]
    except KeyError:
        raise KeyError(f"unknown AG asset {name!r}; known: {sorted(all_entries())}") from None


def by_stage(*stages: str) -> list[AGEntry]:
    bad = set(stages) - set(STAGES)
    if bad:
        raise ValueError(f"unknown stage(s) {sorted(bad)}; known: {list(STAGES)}")
    return [e for e in MODELS + DATASETS + BASELINES if e.stage in stages]


def validate(entries: list[AGEntry] | None = None) -> list[str]:
    """Problems in the registry (empty list = consistent). Checked by tests and before every download."""
    entries = MODELS + DATASETS + BASELINES if entries is None else entries
    probs: list[str] = []
    seen: set[str] = set()
    for e in entries:
        if e.name in seen:
            probs.append(f"{e.name}: duplicate name")
        seen.add(e.name)
        if not _NAME.match(e.name):
            probs.append(f"{e.name}: name is not snake_case")
        if e.kind not in KINDS:
            probs.append(f"{e.name}: unknown kind {e.kind!r}")
        if e.stage not in STAGES:
            probs.append(f"{e.name}: unknown stage {e.stage!r}")
        if e.kind.startswith("hf_") and "/" not in e.repo_id:
            probs.append(f"{e.name}: {e.kind} needs repo_id 'org/name'")
        if e.kind == "hf_stream" and not (e.text_field and e.max_docs):
            probs.append(f"{e.name}: hf_stream needs text_field and max_docs")
        if e.kind == "github" and not e.url.startswith("https://github.com/"):
            probs.append(f"{e.name}: github needs an https://github.com/ url")
        if e.kind == "manual" and not (e.url or e.note):
            probs.append(f"{e.name}: manual needs a url or a note")
    return probs
