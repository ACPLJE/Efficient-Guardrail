"""Adapter between the E0-1 tests and the production code base (landmark_guard v0.4).

The tests in `tests/test_correctness.py` call the functions exposed HERE, never the reference
module directly.  By default every name resolves to the reference implementation in
`lgs1.lgcore.model` / `windows`, so the package is self-checking out of the box.

To run the same tests against the v0.4 code (`/home/212aig08/landmark_guard_mlsys`):

  1. set   LG_V04_ROOT=/home/212aig08/landmark_guard_mlsys   (or edit env/s1.env)
  2. fill in the `_wire_v04()` mapping below so that each name points at the v0.4 function with
     the SAME contract (documented next to each name).  Leave a name unmapped to keep the
     reference for it; the test report prints which side each name came from.
  3. run  `python s1ctl.py tests`  ->  `pytest tests/ -x -q` prints `adapter: <name> <- v04|ref`.

Contracts (all token tensors are 1-D LongTensors unless stated):
  make_windows(n_tokens, b, stride=None) -> list[(start, end)]    length 2b, grid from 0, last
                                                                    window shifted to end at n
  trunk_windows(model, doc_ids, windows, ell, pad_id) -> (hidden [W,L,d], valid [W,L])
                                                                    block-local, positions reset
  manual_forward(model, input_ids, valid=None, layer_range=None, apply_norm=False) -> [B,T,d]
  hf_hidden_states(model, input_ids, valid=None) -> (tuple_of_hidden_states, last_hidden_state)
  pool_scores(token_scores [W,L], valid [W,L], pool) -> [W]
  route(window_scores [W], k) -> (indices sorted desc, theta_k)
  LandmarkGuard(model, cfg, router=None, head=None) with .route(doc_ids) and __call__(doc_ids)
  run_attack(lg, doc_ids, needle_span, adv_ids, slot) -> (AttackEval before, AttackEval after)
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Dict

from . import model as _ref_model
from . import windows as _ref_windows
from . import attack as _ref_attack

SOURCE: Dict[str, str] = {}

make_windows = _ref_windows.make_windows
windows_containing = _ref_windows.windows_containing
spans_fully_covered = _ref_windows.spans_fully_covered
manual_forward = _ref_model.manual_forward
hf_hidden_states = _ref_model.hf_hidden_states
trunk_windows = _ref_model.trunk_windows
pool_scores = _ref_model.pool_scores
route = _ref_model.route
Router = _ref_model.Router
Head = _ref_model.Head
LandmarkGuard = _ref_model.LandmarkGuard
LGConfig = _ref_model.LGConfig
tiny_qwen3 = _ref_model.tiny_qwen3
run_attack = _ref_attack.run_attack
evaluate = _ref_attack.evaluate
insert_slot = _ref_attack.insert_slot

for _n in ("make_windows", "windows_containing", "spans_fully_covered", "manual_forward",
           "hf_hidden_states", "trunk_windows", "pool_scores", "route", "Router", "Head",
           "LandmarkGuard", "LGConfig", "tiny_qwen3", "run_attack", "evaluate", "insert_slot"):
    SOURCE[_n] = "ref"


def _wire_v04() -> None:
    """Map names to the v0.4 code base.  Edit the body; keep the contracts above."""
    root = os.environ.get("LG_V04_ROOT")
    if not root or not os.path.isdir(root):
        return
    if root not in sys.path:
        sys.path.insert(0, root)
    global make_windows, SOURCE
    try:
        v04 = importlib.import_module("landmark_guard")  # v0.4 entry module
    except Exception as e:  # pragma: no cover - depends on the user's checkout
        print(f"[adapter] LG_V04_ROOT set but `landmark_guard` not importable: {e}", file=sys.stderr)
        return
    # --- mapping block: uncomment / adjust once the v0.4 names are confirmed -----------------
    if hasattr(v04, "make_windows"):
        def _mw(n_tokens, b=128, stride=None):
            # v0.4 signature is believed to be make_windows(n, B, stride=None); adjust if needed
            return [tuple(w) for w in v04.make_windows(n_tokens, b, stride=stride)]
        make_windows = _mw
        SOURCE["make_windows"] = "v04"
    # trunk_windows / pool_scores / route / LandmarkGuard: add the same pattern here when the
    # v0.4 functions are identified (see contracts in the module docstring).


_wire_v04()


def report() -> str:
    return "\n".join(f"adapter: {k} <- {v}" for k, v in sorted(SOURCE.items()))
