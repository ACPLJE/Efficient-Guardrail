"""Metric names: the single source for JSONL keys, summary.json keys, numbers.json keys and the
docs/metrics.md table (SPEC 6). Parametric names (hit@k[d]) are built by the helpers below.
"""
from __future__ import annotations

from enum import StrEnum


class M(StrEnum):
    # quality (SPEC 6.1)
    ASR = "asr"
    ASR_POOLED = "asr_pooled"
    ASR_MACRO = "asr_macro"
    FPR_INJ = "fpr_inj"
    FPR_ALL = "fpr_all"
    UTILITY = "utility"
    UTILITY_ATTACK = "utility_attack"
    RECALL = "recall"
    JUDGED_WINDOWS = "judged_windows"
    JUDGED_WINDOWS_UNCACHED = "judged_windows_uncached"
    JUDGE_CALLS = "judge_calls"
    DEADLINE_MISS = "deadline_miss_rate"
    # latency (SPEC 6.2)
    TTFT_GEN = "ttft_gen"
    PREFILL_TIME = "prefill_time"
    TPOT_GEN = "tpot_gen"
    TTFV = "ttfv"
    TTFV_ENGINE = "ttfv_engine"
    RELEASE_LAG = "release_lag"
    BLOCK_LAG = "block_lag"
    TPOT_VIS = "tpot_vis"
    ITL_VIS = "itl_vis"
    EXPOSED_TOKENS = "exposed_tokens"
    GATE_DELAY = "gate_delay"
    E2E_VIS = "e2e_vis"
    CLIENT_WAIT = "client_wait"


CHUNK_DEFS = ("first8", "first32", "all")
HIT_X = ("S", "A\\S", "A")

# abstract numbers (plan 9.3); only plots/abstract_numbers.py writes numbers.json
ABSTRACT_KEYS = ("abs1", "abs2", "abs3", "abs4", "abs5", "abs6", "abs7")
ABSTRACT_DESC = {
    "abs1": "doc FPR on instruction-heavy benign 128k: all-window vs AG (same judge, same ASR)",
    "abs2": "ASR: undefended vs AG, mean over three targets",
    "abs3": "hit@8 of successful attacks (32k, 128k)",
    "abs4": "TTFT(gen) increase and P95 release lag",
    "abs5": "goodput at SLO scale 1.25: no guard, blocking all-window, second-model attribution, AG",
    "abs6": "ledger TPOT overhead and engine LoC",
    "abs7": "mean judged windows per request (vs 1,023 windows at 128k)",
}


def hit_key(k: int, chunk_def: str, x: str = "S", site_rule: str = "single") -> str:
    """hit@8[union] (X = S), hit@8[first8][A\\S], hit@8[union_any], ..."""
    d = chunk_def if site_rule == "single" else f"{chunk_def}_{site_rule}"
    return f"hit@{k}[{d}]" if x == "S" else f"hit@{k}[{d}][{x}]"


def random_hit_key(k: int, chunk_def: str) -> str:
    return f"hit@{k}[{chunk_def}][random]"


def pct_key(m: M | str, p: int) -> str:
    return f"{m}_p{p}"
