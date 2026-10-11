#!/usr/bin/env python3
"""W0-mooncake: length distribution, hash_ids block unit and prefix reuse of Mooncake traces
(plan 5.5, 12.2; SPEC 7.1, 11 W0-mooncake; F-11, A-12, D-020, D-022). CPU only, no network.

python scripts/mooncake_stats.py --name fast25 --out $AG_OUT \\
    --trace $AG_SOURCES/mooncake/repo/FAST25-release/traces/conversation_trace.jsonl \\
    --trace $AG_SOURCES/mooncake/repo/FAST25-release/traces/toolagent_trace.jsonl

Writes $AG_OUT/raw/W0-mooncake/<run_id>/{trace_stats.json, lineage.json, cmd.txt} and prints the run dir.
  run_id             W0-mooncake__<name>__<config_hash[:8]>__<UTC YYYYmmddTHHMMSS> (--ts fixes the time)
  --trace KEY=PATH   names a trace; the default key is the file name without .gz / .txt / .jsonl
  config_hash        canonical_hash of the normalised arguments: name, block/window settings and
                     (key, file name) per trace. Directories stay out (the same analysis of a copy is
                     the same config); file contents are in data_manifest_sha.
  data_manifest_sha  sha256 over the bytes of every trace file in the given order, first 16 hex
                     (a single file -> its own sha256[:16]).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ag import SCHEMA_VERSION  # noqa: E402
from ag.data.traces.mooncake import BLOCK_TOKENS, WINDOW_S, WINDOW_STEP_S, load_trace, trace_stats  # noqa: E402
from ag.harness.lineage import (LineageMismatch, build_lineage, close_lineage, fail_on_mismatch,  # noqa: E402
                                open_lineage)
from ag.harness.writer import atomic_write_json, atomic_write_text  # noqa: E402
from ag.lineage_util import canonical_hash  # noqa: E402

EXP_ID = "W0-mooncake"
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TS = re.compile(r"^\d{8}T\d{6}$")


def trace_key(path: str) -> str:
    name = Path(path).name
    for suf in (".gz", ".txt", ".jsonl", ".json"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name or "trace"


def parse_trace_arg(arg: str) -> tuple[str, Path]:
    """`PATH` or `KEY=PATH` (only when PATH is not itself an existing file with '=' in its name)."""
    if "=" in arg and not Path(arg).exists():
        key, path = arg.split("=", 1)
        if _KEY.match(key):
            return key, Path(path)
    return trace_key(arg), Path(arg)


def sha256_file(path: Path, h: Any = None) -> str:
    """sha256 of one file; also feeds the bytes into `h` (the running data_manifest_sha) when given."""
    own = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            own.update(buf)
            if h is not None:
                h.update(buf)
    return own.hexdigest()


def check_component(name: str, value: str) -> str:
    if not value or "__" in value or "/" in value or any(c.isspace() for c in value):
        raise SystemExit(f"--{name} {value!r}: must be non-empty without '__', '/' or whitespace (run_id part)")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mooncake_stats", description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", action="append", required=True, help="trace file (repeatable), PATH or KEY=PATH")
    ap.add_argument("--name", required=True, help="run name in run_id, e.g. fast25")
    ap.add_argument("--out", default=os.environ.get("AG_OUT", "results"), help="results root ($AG_OUT)")
    ap.add_argument("--block-tokens", type=int, default=BLOCK_TOKENS, help="hash_ids block size to check (A-12)")
    ap.add_argument("--window-s", type=float, default=WINDOW_S, help="rate window for slice choice (D-022)")
    ap.add_argument("--window-step-s", type=float, default=WINDOW_STEP_S)
    ap.add_argument("--ts", default=None, help="UTC YYYYmmddTHHMMSS for run_id (default: now)")
    args = ap.parse_args(argv)

    name = check_component("name", args.name)
    traces = [parse_trace_arg(a) for a in args.trace]
    keys = [k for k, _ in traces]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit(f"duplicate trace keys {dup}; name them with --trace KEY=PATH")
    for _, p in traces:
        if not p.is_file():
            raise SystemExit(f"trace file not found: {p} (download on the login node: "
                             f"bash slurm/ag_download_login.sh --only mooncake)")
    ts = args.ts or time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    if not _TS.match(ts):
        raise SystemExit(f"--ts {ts!r}: expected YYYYmmddTHHMMSS")

    cfg = {"exp_id": EXP_ID, "name": name, "block_tokens": args.block_tokens, "window_s": args.window_s,
           "window_step_s": args.window_step_s, "traces": [[k, p.name] for k, p in traces]}
    config_hash = canonical_hash(cfg)
    run_id = f"{EXP_ID}__{name}__{config_hash[:8]}__{ts}"
    run_dir = Path(args.out) / "raw" / EXP_ID / run_id

    data_h = hashlib.sha256()
    inputs, stats = [], {}
    for key, p in traces:
        rows = load_trace(p)
        inputs.append({"key": key, "path": str(p.resolve()), "sha256": sha256_file(p, data_h), "n_rows": len(rows)})
        stats[key] = trace_stats(rows, block_tokens=args.block_tokens, window_s=args.window_s,
                                 window_step_s=args.window_step_s)

    lin = build_lineage(run_id, EXP_ID, config_hash, clock="monotonic", seed=0, producer="analysis",
                        model_sha={}, data_manifest_sha=data_h.hexdigest()[:16], code_dir=REPO)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        lin, _ = open_lineage(run_dir, lin)
    except LineageMismatch as e:
        fail_on_mismatch(e)
    cmd = " ".join([sys.executable, str(Path(__file__).resolve())] + (argv if argv is not None else sys.argv[1:]))
    atomic_write_text(run_dir / "cmd.txt", cmd + "\n")
    atomic_write_json(run_dir / "trace_stats.json", {
        "run_id": run_id, "schema_version": SCHEMA_VERSION, "exp_id": EXP_ID, "name": name, "config": cfg,
        "inputs": inputs, "traces": stats})
    close_lineage(run_dir, lin)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
