"""S8 configuration: the ablation evaluation subset, paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from lgs5.config import E2_L_ALL, S5Paths

# plan S8: "ablation 평가 부분집합은 E2-8과 같은 200쌍 × 8 L × 5 위치 = 16,000행 + benign-long 4,000행".  The test grid is
# 0.0, 0.1, ..., 1.0 (lgs2.config RECIPE_V2), so the five positions are grid values: {0, .25, .5, .75, 1} matched only three
# of them (2026-09-27; the arms evaluated that day are resumed for 0.2 / 0.8 by id).
ABL_DEPTHS = [0.0, 0.2, 0.5, 0.8, 1.0]
ABL_MAX_PAIRS = 200
ABL_L = list(E2_L_ALL)
ABL_DEPTHS_STR = ",".join(str(d) for d in ABL_DEPTHS)
ABL_MAIN_ROWS = ABL_MAX_PAIRS * len(ABL_DEPTHS) * len(ABL_L) * 2      # 16,000 test/main rows (harmful + twin)


def eval_args() -> str:
    """lgs5.run arguments of an arm's e2 pass (thresholds come from calib/benign_long; calib/main is not needed)."""
    return f"--L all --depths {ABL_DEPTHS_STR} --max-pairs {ABL_MAX_PAIRS} --no-calib-main"


def pair_index(pair_id: Optional[str]) -> Optional[int]:
    if not pair_id:
        return None
    try:
        return int(str(pair_id).rsplit("-p", 1)[1])
    except (ValueError, IndexError):
        return None


def subset_row(r: Dict, depths: Sequence[float] = ABL_DEPTHS, max_pairs: int = ABL_MAX_PAIRS) -> bool:
    """Is an eval row part of the ablation subset?  test/main: the 5 grid depths and the first `max_pairs` pairs; test and calib
    benign-long: all; calib/main (E2-7 twins) and everything else: no.  Applied to the baseline's full rows so the
    baseline and the arms are compared on identical rows."""
    split, subset = r.get("split"), r.get("subset")
    if subset == "benign_long":
        return split in ("test", "calib")
    if split == "test" and subset == "main":
        d = r.get("depth")
        if d is None or round(float(d), 4) not in {round(float(x), 4) for x in depths}:
            return False
        pi = pair_index(r.get("pair_id"))
        return pi is None or pi < max_pairs
    return False


@dataclass
class S8Paths:
    s5: S5Paths
    reports: Path

    @classmethod
    def load(cls) -> "S8Paths":
        s5 = S5Paths.load()
        return cls(s5=s5, reports=s5.out / "reports" / "s8")

    def ensure(self) -> "S8Paths":
        self.s5.ensure()
        self.reports.mkdir(parents=True, exist_ok=True)
        return self
