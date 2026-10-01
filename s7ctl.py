#!/usr/bin/env python
"""S7 — E4 cost measurement jobs (plan S7: E4-1 latency, E4-2 throughput, E4-3 peak memory, E4-4 breakdown, E4-9 k sweep;
RTX 6000 Ada + L40S + A100 via --partition) and the cost tables (Fig 7, Fig 9, Table 5, G2 (b)(c)).

    python s7ctl.py cost --methods lg:qwen3guard_0.6b_s0,sliding:qwen3guard_0.6b_s0 --exps latency,memory,breakdown --sbatch
    python s7ctl.py cost --methods lg:qwen3guard_0.6b_s0 --exps throughput --L 8192,32768,131072 --sbatch
    python s7ctl.py cost --methods lg:qwen3guard_0.6b_s0 --exps ksweep --L 28672,131072 --sbatch
    python s7ctl.py e4 --sbatch [--partition <L40S partition>] [--gpu l40s]     # the plan's method list, one exclusive job per tier
    python s7ctl.py tables
    python s7ctl.py status

Every E4 job takes the node exclusively (--exclusive; plan: GPU D runs latency jobs only) and records the GPU / driver /
CUDA / torch / dtype / attention implementation / batch / repeat metadata in every row.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import s1ctl  # noqa: E402
import s5ctl  # noqa: E402
from lgs1.paths import env, load_env_file  # noqa: E402
from lgs5.config import S5Paths  # noqa: E402
from lgs7.cost import parse_cost_spec  # noqa: E402

TOK_ENV = "export TOKENIZERS_PARALLELISM=true\n"
# --exclusive hands the job every GPU of the node; an E4 row is a one-GPU measurement (device_map='auto' would balance the
# guard over all of them -- 9/24).  Keep the first allocated device; lgs7.cost repeats the check when the variable is unset.
GPU_ENV = 'if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"; fi\n'
EXPS = ["latency", "throughput", "memory", "breakdown", "ksweep"]
L_THROUGHPUT = "8192,32768,131072"
L_KSWEEP = "28672,131072"
# Every E4 row must come from ONE GPU type (plan: GPU D = RTX 6000 Ada; Fig 7 / G2 (b)(c) compare methods on it).  9/24: the
# unpinned tier list spread the first e4 over A5000 + RTX 6000 Ada, so the rows split into cost/rtxa5000 and cost/rtx6000adageneration.
# `--partition any` restores the tier list (a different GPU for the 3-GPU side panel: `--partition L40S`).
S7_PARTITION = env("LG_S7_PARTITION", "RTX6000ADA")


def s7_partition(a) -> Optional[str]:
    p = getattr(a, "partition", None) or S7_PARTITION
    return None if p.lower() in ("any", "tier", "") else p


def _tier(spec) -> int:
    n = spec.name
    if spec.kind in ("base", "cd"):
        from lgs1.zeroshot import guards as G
        gb = G.get(n).min_gpu_gb if n in G.GUARDS else 24
        return max(gb, 48) if spec.policy == "yarn" or spec.kind == "cd" and "8b" in n else gb
    return 48 if ("4b" in n or "8b" in n or spec.policy == "yarn") else 24


PREFILL_NATIVE = 8192      # 128k-native zero-shot guards: the same chunked prefill B1 used (s6ctl.PREFILL); 48 GB fits 8B-20B at 128k


def _prefill_for(method: str, a) -> Optional[int]:
    if getattr(a, "prefill_chunk", None):
        return int(a.prefill_chunk)
    spec = parse_cost_spec(method)
    return PREFILL_NATIVE if spec.kind == "base" and (spec.policy or "native") == "native" else None


def _body(method: str, exps: List[str], a) -> str:
    lines = []
    common = f" --warmup {a.warmup} --iters {a.iters} --n-docs {a.n_docs}" + (f" --gpu {a.gpu}" if a.gpu else "") + (" --force" if a.force else "")
    pc = _prefill_for(method, a)
    if pc:
        common += f" --prefill-chunk {pc}"
    for exp in exps:
        L = a.L if exp not in ("throughput", "ksweep") or a.L != "all" else (L_THROUGHPUT if exp == "throughput" else L_KSWEEP)
        extra = f" --batches {a.batches}" if exp == "throughput" else (f" --ks {a.ks}" if exp == "ksweep" else "")
        lines.append(f"${{PY}} -m lgs7.cost --exp {exp} --method {method} --L {L}{extra}{common}")
    return "\n".join(lines)


def _submit_cost(a, methods: List[str], exps: List[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for m in methods:
        spec = parse_cost_spec(m)
        job = f"s7_cost_{spec.tag}"[:60].replace(":", "_")
        body = _body(m, exps, a)
        if a.sbatch:
            out[m] = s1ctl.write_sbatch(job, TOK_ENV + GPU_ENV + body, env("LG_TIME_S7_COST", "12:00:00"), dep=a.dep, dry_run=a.dry_run_sbatch,
                                        min_gpu_gb=_tier(spec), exclusive=not a.no_exclusive, partition=s7_partition(a))
        else:
            rc = 0
            for ln in body.splitlines():
                args = ln.split()[3:]                    # drop "${PY} -m lgs7.cost"
                rc |= s5ctl._run_local("lgs7.cost", args)
            out[m] = "ok" if rc == 0 else f"rc={rc}"
    return out


def cmd_cost(a):
    methods = [m for m in a.methods.split(",") if m]
    exps = EXPS if a.exps == "all" else [e for e in a.exps.split(",") if e]
    bad = [e for e in exps if e not in EXPS]
    if bad:
        raise SystemExit(f"unknown exps {bad}; one of {EXPS}")
    out = _submit_cost(a, methods, exps)
    print(f"[cost] {out}")
    return 0


def e4_methods(P: S5Paths) -> List[str]:
    """Plan S7 E4-1 method list, restricted to the checkpoints / guards present."""
    from lgs1.zeroshot import guards as G
    ms: List[str] = []
    r06 = "qwen3guard_0.6b_s0"
    cfg = E4_CFG
    if (P.run_dir(r06) / "router.pt").exists():
        ms += [f"lg:{r06}:k32{cfg}", f"lg:{r06}:k16{cfg}", f"lg:{r06}:k8{cfg}", f"lg:{r06}:k32{cfg}:noskip", f"lg:{r06}:k32{cfg}:pad",
               f"lg:{r06}:k32:nostream", f"sliding:{r06}:c16"]
    if (P.run_dir(r06) / "dense_ft.pt").exists():
        ms += [f"dense:{r06}:yarn"]
    for big in ("qwen3guard_8b_s0", "llamaguard3_8b_s0", "qwen3guard_4b_s0"):
        if (P.run_dir(big) / "router.pt").exists():
            ms.append(f"lg:{big}:k32{cfg}")
    for g, pol in (("qwen3guard_gen_0.6b", "yarn"), ("qwen3guard_gen_8b", "yarn"), ("llamaguard3_8b", "native"), ("gpt_oss_safeguard_20b", "native"),
                   ("guardreasoner_8b", "native")):
        if g in G.GUARDS:
            ms.append(f"base:{g}:{pol}")
    if "qwen3guard_gen_8b" in G.GUARDS:
        ms += ["cd:qwen3guard_gen_8b:512:256", "cd:qwen3guard_gen_8b:1024:512"]
    if "prompt_guard_2_86m" in G.GUARDS:
        ms.append("cd:prompt_guard_2_86m:512:0")
    return ms


E4_CFG = ":c16:h8"                         # 9/27: streaming chunk 16 windows + deep head in batches of 8 -> peak 1k->128k +8.6 % (G2 (c) PASS),
                                           # 128k p50 1.10 s vs 1.35 s at the c64 default; verdicts identical (tests)
E4_MAIN = f"lg:qwen3guard_0.6b_s0:k32{E4_CFG}"   # the main config: breakdown / k sweep / throughput run on this spec only


def e4_plan(methods: List[str]) -> List[tuple]:
    """(methods, exps) groups for `e4`; every (method, exp) pair appears exactly once.

    latency+memory: every method.  breakdown: main LG + 4B/8B LG.  ksweep: main LG.
    throughput: main LG + dense-FT / base yarn / CD-512 (the batchable baselines).
    (9/23: the old filter `m.count(":") == 3` picked the noskip/pad/nostream cost switches
    instead of the plain k32 spec and re-submitted them three times.)
    """
    lg_main = [m for m in methods if m == E4_MAIN]
    big = [m for m in methods if m.startswith("lg:") and "0.6b" not in m]
    dense = [m for m in methods if m.startswith(("dense:", "base:qwen3guard_gen_0.6b", "base:qwen3guard_gen_8b", "cd:qwen3guard_gen_8b:512"))]
    plan = [(list(methods), ["latency", "memory"])]
    if lg_main:
        plan += [(lg_main + big, ["breakdown"]), (lg_main, ["ksweep"]), (lg_main + dense, ["throughput"])]
    seen = set()
    out = []
    for ms, exps in plan:
        keep = []
        for m in ms:
            key = (m, tuple(exps))
            if key not in seen:
                seen.add(key)
                keep.append(m)
        if keep:
            out.append((keep, exps))
    return out


def cmd_e4(a):
    P = S5Paths.load()
    methods = e4_methods(P)
    if not methods:
        raise SystemExit("nothing to measure (no runs/v2/*/router.pt, no guards)")
    a.L = "all"
    out: Dict[str, Optional[str]] = {}
    for ms, exps in e4_plan(methods):
        for m, job in _submit_cost(a, ms, exps).items():
            out[f"{'+'.join(exps)}|{m}"] = job
    print(f"[e4] {len(out)} jobs:")
    for k, v in out.items():
        print(f"  {v}  {k}")
    print("[e4] when they finish: python s7ctl.py tables")
    return 0


def cmd_tables(a):
    return s5ctl._run_local("lgs7.tables", [])


def cmd_status(a):
    P = S5Paths.load()
    print(f"\nS7 status  ({time.strftime('%Y-%m-%d %H:%M')}, cost={P.out / 'cost'})\n")
    for d in sorted(glob.glob(str(P.out / "cost" / "*"))):
        files = sorted(glob.glob(str(Path(d) / "*.rows.jsonl")))
        print(f"  {Path(d).name}: {len(files)} rows files")
        for f in files:
            n = err = 0
            for line in open(f, encoding="utf-8", errors="replace"):
                if line.strip():
                    n += 1
                    if '"error": "' in line:
                        err += 1
            print(f"     {Path(f).name:70s} {n:>5} rows  {err} errors")
    reps = sorted(glob.glob(str(P.out / "reports" / "s7" / "*.md")))
    print(f"  reports/s7: {[Path(r).name for r in reps]}")
    os.system("squeue -u $USER -o '%.12i %.40j %.8T %.10M %.6D %R' 2>/dev/null | grep -E 'JOBID|s7_' || true")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--sbatch", action="store_true")
        p.add_argument("--local", action="store_true")
        p.add_argument("--dep", default=None)
        p.add_argument("--dry-run-sbatch", action="store_true")
        p.add_argument("--L", default="all")
        p.add_argument("--batches", default="1,2,4,8,16,32")
        p.add_argument("--ks", default="4,8,16,32,64,all")
        p.add_argument("--warmup", type=int, default=5)
        p.add_argument("--iters", type=int, default=20)
        p.add_argument("--n-docs", type=int, default=20)
        p.add_argument("--gpu", default=env("LG_COST_GPU", "") or None, help="rows dir slug (default: device name)")
        p.add_argument("--partition", default=None, help=f"Slurm partition = GPU type (default {S7_PARTITION}; 'any' = the memory tier list)")
        p.add_argument("--no-exclusive", action="store_true")
        p.add_argument("--prefill-chunk", type=int, default=None)
        p.add_argument("--force", action="store_true")

    p = sub.add_parser("cost"); common(p); p.add_argument("--methods", required=True); p.add_argument("--exps", default="latency,memory"); p.set_defaults(fn=cmd_cost)
    p = sub.add_parser("e4"); common(p); p.set_defaults(fn=cmd_e4)
    p = sub.add_parser("tables"); p.set_defaults(fn=cmd_tables)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    return ap


def main(argv=None) -> int:
    load_env_file()
    a = build_parser().parse_args(argv)
    r = a.fn(a)
    return int(r) if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
