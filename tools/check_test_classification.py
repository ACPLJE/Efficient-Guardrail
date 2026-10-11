#!/usr/bin/env python3
"""Every test module is listed in exactly one of tests/manifests/{cpu,gpu,integration}.txt, and
every manifest entry exists. GPU tests live under tests/gpu/ and only in gpu.txt.

python tools/check_test_classification.py   # exit 1 on a problem
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = ("cpu", "gpu", "integration")


def problems(repo: Path = REPO) -> list[str]:
    tests = {str(p.relative_to(repo)) for p in (repo / "tests").rglob("test_*.py")}
    listed: dict[str, list[str]] = {}
    out: list[str] = []
    for m in MANIFESTS:
        path = repo / "tests" / "manifests" / f"{m}.txt"
        entries = [l.strip() for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
        for e, n in Counter(entries).items():
            if n > 1:
                out.append(f"{m}.txt lists {e} {n} times")
        for e in entries:
            listed.setdefault(e, []).append(m)
            if e not in tests:
                out.append(f"{m}.txt lists missing file {e}")
            if (m == "gpu") != e.startswith("tests/gpu/"):
                out.append(f"{e}: tests/gpu/ modules belong in gpu.txt only")
    for t in sorted(tests):
        where = listed.get(t, [])
        if len(where) != 1:
            out.append(f"{t} is listed in {where or 'no manifest'}")
    return out


def main() -> int:
    p = problems()
    for x in p:
        print(x)
    return 1 if p else 0


if __name__ == "__main__":
    sys.exit(main())
