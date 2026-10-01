"""Landmark Guard core reference implementation + adapter (used by the E0-1 tests)."""
from .windows import make_windows, windows_containing, spans_fully_covered, window_lengths  # noqa: F401
from .model import (LGConfig, LandmarkGuard, Router, Head, tiny_qwen3, manual_forward,  # noqa: F401
                    hf_hidden_states, trunk_windows, pool_scores, route, head_forward, sliding_scan)
from .attack import run_attack, evaluate, insert_slot, random_search, AttackEval  # noqa: F401
