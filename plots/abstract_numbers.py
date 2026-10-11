"""numbers.json: the seven abstract numbers (plan 9.3, SPEC 7.4). Only this script writes it.

S1: every key is a stub (value null) unless a source run is given, e.g.
  python plots/abstract_numbers.py --source abs3=<E0.1 B0 run_id>
abs3 reads hit@8[union] from that run's summary.json. Mock runs are rejected unless --allow-mock
and are then tagged "mock" (never "측정").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import _common as C  # noqa: E402

from ag.harness.writer import atomic_write_json, read_json  # noqa: E402
from ag.metrics.names import ABSTRACT_DESC, ABSTRACT_KEYS, hit_key  # noqa: E402

# key -> (summary.json metric, unit); keys without a source stay null until their stage lands
SOURCES = {"abs3": (hit_key(8, "union"), "ratio")}


def stub(key: str) -> dict:
    return {"value": None, "unit": None, "ci": None, "tag": None, "source_run_ids": [], "code_commit": None,
            "target_id": None, "desc": ABSTRACT_DESC[key]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="abstract_numbers")
    ap.add_argument("--out", default=None, help="results root ($AG_OUT)")
    ap.add_argument("--source", action="append", default=[], help="absN=<run_id>")
    ap.add_argument("--allow-mock", action="store_true")
    args = ap.parse_args(argv)
    import os
    out_root = Path(args.out or os.environ.get("AG_OUT", "results"))
    nums = {k: stub(k) for k in ABSTRACT_KEYS}
    for spec in args.source:
        key, rid = spec.split("=", 1)
        if key not in SOURCES:
            raise C.RunRejected(f"{key}: no source defined yet (S1 supports {sorted(SOURCES)})")
        d = C.run_dir_of(out_root, rid, exp="")
        lin = C.check_run(d, allow_mock=args.allow_mock)
        metric, unit = SOURCES[key]
        m = read_json(d / "summary.json").get(metric)
        if not m or m.get("value") is None:
            raise C.RunRejected(f"{rid}: summary.json has no {metric} value (abs3 needs selector.k with 8 and a "
                                f"resolved S source); nothing written")
        nums[key].update(value=m.get("value"), unit=unit, ci=[m.get("ci_low"), m.get("ci_high")],
                         tag="mock" if C.is_mock(lin) else "측정", source_run_ids=[rid],
                         code_commit=lin.get("code_commit"), target_id=rid.split("__")[2])
    p = out_root / "numbers.json"
    atomic_write_json(p, nums)
    print(p)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except C.RunRejected as e:
        print(f"abstract_numbers: {e}", file=sys.stderr)
        raise SystemExit(1) from None
