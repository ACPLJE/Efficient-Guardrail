"""Time-limit stop for resumable row writers (lgs1.interrupt.Stop): the row loop flushes and exits 3 after the current row,
the rows written so far survive, and the resumed run continues from the next id.  The sbatch side (RUN_FORWARD forwarding
SIGTERM to the python child, rc=3 -> scontrol requeue) is covered by tests/test_sbatch_signal.py."""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from lgs1.interrupt import RC_REQUEUE, Stop
from lgs1.lineage import Lineage
from lgs5.config import ModelSpec
from lgs5.runners import write_rows


class _Runner:
    method = "fake"
    lineage = Lineage(model_sha="m", router_sha="r", cfg_hash="c", recipe_hash="p", data_manifest_sha="d", code_commit="x")

    def __init__(self, stop_after: int):
        self.stop_after, self.n = stop_after, 0

    def score_row(self, r):
        self.n += 1
        if self.n == self.stop_after:
            os.kill(os.getpid(), signal.SIGTERM)          # what the sbatch wrapper forwards at the time limit
        return {"score": 0.5, "verdict": False}


def test_stop_flushes_exits_3_and_resumes(tmp_path: Path):
    Stop.reset()
    rows = [{"id": f"r{i}", "split": "test", "subset": "main", "L_target": 1024, "depth": 0.0, "label": "harmful"} for i in range(10)]
    out = tmp_path / "fake.rows.jsonl"
    spec = ModelSpec("lg", "tiny_s0")
    with pytest.raises(SystemExit) as ex:
        write_rows(_Runner(stop_after=4), spec, rows, out)
    assert ex.value.code == RC_REQUEUE and Stop.requested
    got = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["id"] for r in got] == ["r0", "r1", "r2", "r3"]           # the signalled row is still written, then the loop stops
    Stop.reset()
    n = write_rows(_Runner(stop_after=10 ** 6), spec, rows, out)         # the requeued run resumes by id
    assert n == 6
    got = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["id"] for r in got] == [f"r{i}" for i in range(10)]
    Stop.reset()
    Stop._installed = False
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
