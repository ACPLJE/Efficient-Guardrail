#!/usr/bin/env python
"""S5 — evaluate the trained checkpoints (plan S5: E1 parity, E2-1/E2-2 <=28k, E2-7, B2 CD, B4 dense-FT, T4 LG 8B).

    python s5ctl.py e1 --models lg:qwen3guard_0.6b_s0,dense:qwen3guard_0.6b_s0,base:qwen3guard_gen_0.6b --sbatch
    python s5ctl.py e2 --models lg:qwen3guard_0.6b_s0,dense:qwen3guard_0.6b_s0 --L le28k --sbatch   # --L long|all: S6 64k/128k rows (s6ctl e2-long)
    python s5ctl.py e27 --runs qwen3guard_0.6b_s0 --sbatch          # oracle + needle-only rows (E2-7)
    python s5ctl.py b2 --sbatch [--sweep]                           # CD: Qwen3Guard-8B W=512 overlap 256 (<=28k full) [+ sweep subset]
    python s5ctl.py b1-calib --sbatch                               # zero-shot 12 + CD on calib/benign_long -> matched-FPR for the baselines
    python s5ctl.py t4 --sbatch                                     # LG 8B (LoRA) head -> router -> valcheck
    python s5ctl.py all --sbatch                                    # e1 + e2 + e27 for the 0.6B seeds, dense-FT, base 0.6B, b2, t4
    python s5ctl.py tables [--L all]                                # Table 1, Table 2, Fig 3/4 data, E2-7 -> reports/s5/
    python s5ctl.py status

Model specs: lg:<run>[:kK] | sliding:<run> | oracle:<run> | needle:<run> | dense:<run> | base:<guard>  (lgs5.config).
Every job resumes by row id; rerunning a command only scores what is missing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import s1ctl  # noqa: E402
from lgs1.paths import env  # noqa: E402
from lgs4.config import S4Paths  # noqa: E402,F401
from lgs5.config import E2_L_LE28K, ModelSpec, S5Paths, parse_specs  # noqa: E402

SEEDS_06B = [0, 1, 2]
TOK_ENV = "export TOKENIZERS_PARALLELISM=true\n"
L_LE28K = ",".join(str(x) for x in E2_L_LE28K)


def _submit(job: str, body: str, time_limit: str, a, dep: Optional[str] = None, **kw) -> Optional[str]:
    return s1ctl.write_sbatch(job, TOK_ENV + body, time_limit, dep=dep if dep is not None else getattr(a, "dep", None),
                              dry_run=getattr(a, "dry_run_sbatch", False), **kw)


def _run_local(module: str, args: List[str]) -> int:
    from s3ctl import _run_local as rl
    return rl(module, args)


def _tier(spec: ModelSpec) -> int:
    if spec.kind == "base":
        from lgs1.zeroshot import guards as G
        return G.get(spec.name).min_gpu_gb if spec.name in G.GUARDS else 24
    return 48 if ("4b" in spec.name or "8b" in spec.name) else 24


def _job_name(exp: str, spec: ModelSpec) -> str:
    return f"s5_{exp}_{spec.tag}"[:60].replace(":", "_")


EVAL_SIGNAL = "B:TERM@300"       # 9/25: eval rows are resumable by id -> SIGTERM 5 min before the limit, flush, rc=3, self-requeue
                                 # (same job id); before, e2-long / YaRN / LG 8B e2 jobs died at 24-36 h and were resubmitted by hand


def _eval_jobs(a, exp: str, specs: List[ModelSpec], extra: str = "", time_key: str = "LG_TIME_S5_EVAL", default_time: str = "24:00:00") -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for spec in specs:
        body = f"${{PY}} -m lgs5.run --exp {exp} --model {spec}{extra}"
        if a.sbatch:
            out[str(spec)] = _submit(_job_name(exp, spec), body, env(time_key, default_time), a, min_gpu_gb=_tier(spec),
                                     signal=EVAL_SIGNAL, requeue=True)
        else:
            rc = _run_local("lgs5.run", ["--exp", exp, "--model", str(spec)] + extra.split())
            out[str(spec)] = "ok" if rc == 0 else f"rc={rc}"
    return out


# ------------------------------------------------------------------------------------------------
def cmd_e1(a):
    specs = parse_specs(a.models)
    out = _eval_jobs(a, "e1", specs, time_key="LG_TIME_S5_E1", default_time="06:00:00")
    print(f"[e1] {out}")
    return 0


def cmd_e2(a):
    specs = parse_specs(a.models)
    extra = f" --L {a.L}" + (f" --depths {a.depths}" if a.depths else "") + (f" --max-pairs {a.max_pairs}" if a.max_pairs else "")
    out = _eval_jobs(a, "e2", specs, extra=extra)
    print(f"[e2] {out}")
    return 0


def cmd_e27(a):
    runs = [r for r in a.runs.split(",") if r]
    specs = [ModelSpec(kind, r) for r in runs for kind in ("oracle", "needle")]
    out = _eval_jobs(a, "e2", specs, extra=f" --L {a.L}")
    print(f"[e27] {out}")
    return 0


def cmd_b2(a):
    """B2 chunk-and-decode baselines through lgs3.longbench (rows land in longbench/v2main, read by the tables)."""
    import s3ctl
    tasks = [["qwen3guard_gen_8b", "v2main", "chunk_or:512:256", L_LE28K, "-"]]          # default config, full <=28k
    if a.sweep:
        sub = "--max-pairs_128_--depths_0,0.25,0.5,0.75,1"       # on the 0.1-step test grid this is 3 positions (0, .5, 1); see s6ctl.SUB_LONG
        for W in (256, 1024, 2048):
            for ov in (0, W // 2):
                tasks.append(["qwen3guard_gen_8b", "v2main", f"chunk_or:{W}:{ov}", L_LE28K, sub])
        tasks.append(["qwen3guard_gen_8b", "v2main", "chunk_or:512:0", L_LE28K, sub])
        for g in ("llamaguard3_8b", "gpt_oss_safeguard_20b"):
            tasks.append([g, "v2main", "chunk_or:512:256", L_LE28K, sub])
    if not a.sbatch:
        for t in tasks:
            print(" ".join(t))
        return 0
    jids = s3ctl._submit_longbench(a, tasks, job="s5_b2")
    print(f"[b2] {len(tasks)} tasks in {len(jids)} arrays: {jids}")
    return 0


def cmd_b1_calib(a):
    """Zero-shot guards + CD default config on calib/benign_long <=28k: the calibration scores that turn the baselines'
    default-verdict rows into matched-FPR rows in Table 2 (3,000 rows per guard, ~10 % of the B1 test run)."""
    import s3ctl
    from lgs3.longbench import CTX_DOC
    tasks = []
    for g in s3ctl.B1_ALL:
        if CTX_DOC.get(g, 32768) < 28672 + 512:
            tasks.append([g, "v2calib", "trunc_keep_head", L_LE28K, "-"])
            tasks.append([g, "v2calib", "trunc_keep_tail", L_LE28K, "-"])
        else:
            tasks.append([g, "v2calib", "native", L_LE28K, "-"])
    tasks.append(["qwen3guard_gen_8b", "v2calib", "chunk_or:512:256", L_LE28K, "-"])
    if not a.sbatch:
        for t in tasks:
            print(" ".join(t))
        return 0
    jids = s3ctl._submit_longbench(a, tasks, job="s5_b1calib")
    print(f"[b1-calib] {len(tasks)} tasks in {len(jids)} arrays: {jids}")
    return 0


def cmd_t4(a):
    """T4: LG 8B (LoRA r=16) head -> router -> valcheck via the S4 chain."""
    import s4ctl
    s4ctl._check_sealed(a)
    if not a.sbatch:
        raise SystemExit("t4 is an sbatch chain")
    out = s4ctl._chain(a, "qwen3guard_8b", 0, a.dep, start=getattr(a, "from_stage", "head"))
    print(f"[t4] {out}")
    return 0


def cmd_all(a):
    P = S5Paths.load()
    lg = [f"lg:qwen3guard_0.6b_s{s}" for s in SEEDS_06B if (P.run_dir(f"qwen3guard_0.6b_s{s}") / "router.pt").exists()]
    dense = [f"dense:qwen3guard_0.6b_s{s}" for s in SEEDS_06B if (P.run_dir(f"qwen3guard_0.6b_s{s}") / "dense_ft.pt").exists()]
    for big in ("qwen3guard_4b_s0", "qwen3guard_8b_s0"):
        if (P.run_dir(big) / "router.pt").exists():
            lg.append(f"lg:{big}")
    base = ["base:qwen3guard_gen_0.6b"]
    if not lg:
        raise SystemExit("no trained LG run with router.pt under runs/v2 (S4 T1 first)")
    a.models = ",".join(lg + dense + base)
    cmd_e1(a)
    a.models = ",".join(lg + dense)
    cmd_e2(a)
    a.runs = ",".join(x.split(":")[1] for x in lg)
    cmd_e27(a)
    if not getattr(a, "no_b2", False):
        a.sweep = getattr(a, "sweep", False)
        cmd_b2(a)
        cmd_b1_calib(a)
    if getattr(a, "with_t4", False):
        cmd_t4(a)
    print("[all] when the jobs finish: python s5ctl.py tables")
    return 0


def cmd_tables(a):
    return _run_local("lgs5.tables", ["--L", a.L])


def cmd_relineage(a):
    """Rewrite model_sha in existing rows files to the stable definition (head.pt + router.pt for LG kinds, dense_ft.pt for
    dense) so a resume is not refused after dense-FT wrote checkpoints into the run directory.  Only model_sha changes, and
    only when the rows' router_sha still matches the run's router.pt (same run)."""
    import os as _os
    from lgs1.lineage import iter_rows, sha256_file
    from lgs4 import backbone as BB
    from lgs5.runners import dense_model_sha, lg_model_sha
    P = S5Paths.load()
    for spec in parse_specs(a.models):
        path = P.rows_path(a.exp, spec)
        if not path.exists():
            print(f"[relineage] {path.name}: no rows file")
            continue
        run_dir = P.run_dir(spec.name)
        new_sha = dense_model_sha(run_dir) if spec.kind == "dense" else lg_model_sha(run_dir)
        rfile = run_dir / BB.ROUTER_FILE
        router_sha = "none" if spec.kind == "dense" or not rfile.exists() else sha256_file(rfile)
        rows = list(iter_rows(path, tolerant=True))
        old = sorted({str(r.get("model_sha")) for r in rows})
        if old == [new_sha]:
            print(f"[relineage] {path.name}: already {new_sha[:12]}… ({len(rows):,} rows)")
            continue
        bad = sum(1 for r in rows if str(r.get("router_sha")) != router_sha)
        if bad:
            print(f"[relineage] {path.name}: REFUSED -- {bad:,} rows carry a different router_sha (not the same run)")
            continue
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                r["model_sha"] = new_sha
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        _os.replace(tmp, path)
        print(f"[relineage] {path.name}: model_sha {[x[:12] + '…' for x in old]} -> {new_sha[:12]}… ({len(rows):,} rows)")
    return 0


_PROG = re.compile(r"^\s+([\d,]+)/([\d,]+)\s+(\d+)s\s+\((\d+) ms/row\)")


def progress_from_log(log: Path) -> str:
    """'done' | 'running n/todo  <ms> ms/row  eta <m>m' | 'queued' from the newest lgs5.run log of a rows file."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    last_prog, state = None, "queued"
    for ln in lines:
        if ln.startswith("[s5] ") and "done=" in ln:
            state, last_prog = "running", None
            if "todo=0" in ln:
                state = "done"
        elif ln.startswith("[s5] ") and " rows in " in ln:
            state = "done"
        else:
            m = _PROG.match(ln)
            if m:
                last_prog = m
    if state != "running" or last_prog is None:
        return state
    n, todo, ms = int(last_prog.group(1).replace(",", "")), int(last_prog.group(2).replace(",", "")), int(last_prog.group(4))
    eta = (todo - n) * ms / 1000 / 60
    return f"running {n:,}/{todo:,}  {ms} ms/row  eta {eta:.0f} m ({time.strftime('%H:%M', time.localtime(time.time() + eta * 60))})"


def _newest_log(P: S5Paths, exp: str, tag: str) -> Optional[Path]:
    logs = glob.glob(str(P.out / "logs" / f"s5_{exp}_{tag}_[0-9]*.out"))
    return Path(max(logs, key=os.path.getmtime)) if logs else None


def _queued_names() -> Optional[set]:
    """Names of this user's PENDING/RUNNING jobs (None when squeue is unavailable)."""
    user = env("LG_USER") or os.environ.get("USER") or ""
    try:
        txt = subprocess.run(["squeue", "-h", "-u", user, "-o", "%j"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    return {l.strip() for l in txt.splitlines() if l.strip()}


def cmd_status(a):
    P = S5Paths.load()
    print(f"\nS5 status  ({time.strftime('%Y-%m-%d %H:%M')}, eval={P.eval})\n")
    queued = _queued_names()
    for exp in ("e1", "e2"):
        files = sorted(glob.glob(str(P.eval / exp / "*.rows.jsonl")))
        print(f"  {exp}: {len(files)} rows files")
        for f in files:
            n = sum(1 for line in open(f, encoding="utf-8", errors="replace") if line.strip())
            tag = Path(f).name[: -len(".rows.jsonl")]
            log = _newest_log(P, exp, tag)
            st = progress_from_log(log) if log else ""
            job = f"s5_{exp}_{tag}"[:60].replace(":", "_")
            if st.startswith("running") and queued is not None and job not in queued:
                # the log says the job started and never finished, but nothing with that name is queued -> it died
                st = f"STOPPED (no job in queue) -> {Path(log).with_suffix('.err').name}"
            print(f"     {Path(f).name:60s} {n:>8,} rows   {st}")
    for bench in ("v2main", "v2calib"):
        lb = sorted(glob.glob(str(P.longbench / bench / "*.rows.jsonl")))
        print(f"  longbench/{bench}: {len(lb)} rows files ({'zero-shot + CD' if bench == 'v2main' else 'calibration benign-long'})")
        for f in lb:
            n = sum(1 for line in open(f, encoding="utf-8", errors="replace") if line.strip())
            print(f"     {Path(f).name:60s} {n:>8,} rows")
    reps = sorted(glob.glob(str(P.reports / "*.md")))
    print(f"  reports/s5: {[Path(r).name for r in reps]}")
    os.system("squeue -u $USER -o '%.12i %.40j %.8T %.10M %.6D %R' 2>/dev/null | grep -E 'JOBID|s5_' || true")
    return 0


# ------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--sbatch", action="store_true")
        p.add_argument("--local", action="store_true")
        p.add_argument("--dep", default=None)
        p.add_argument("--dry-run-sbatch", action="store_true")
        p.add_argument("--allow-unsealed", action="store_true")

    p = sub.add_parser("e1"); common(p); p.add_argument("--models", required=True); p.set_defaults(fn=cmd_e1)
    p = sub.add_parser("e2"); common(p); p.add_argument("--models", required=True); p.add_argument("--L", default="le28k")
    p.add_argument("--depths", default=None); p.add_argument("--max-pairs", type=int, default=None); p.set_defaults(fn=cmd_e2)
    p = sub.add_parser("e27"); common(p); p.add_argument("--runs", required=True); p.add_argument("--L", default="le28k"); p.set_defaults(fn=cmd_e27)
    p = sub.add_parser("b2"); common(p); p.add_argument("--sweep", action="store_true"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_b2)
    p = sub.add_parser("b1-calib"); common(p); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_b1_calib)
    p = sub.add_parser("relineage"); p.add_argument("--exp", default="e2", choices=["e1", "e2"]); p.add_argument("--models", required=True)
    p.set_defaults(fn=cmd_relineage)
    p = sub.add_parser("t4"); common(p); p.add_argument("--from", dest="from_stage", default="head", choices=["head", "router", "valcheck"]); p.set_defaults(fn=cmd_t4)
    p = sub.add_parser("all"); common(p); p.add_argument("--L", default="le28k"); p.add_argument("--depths", default=None)
    p.add_argument("--max-pairs", type=int, default=None); p.add_argument("--sweep", action="store_true"); p.add_argument("--no-b2", action="store_true")
    p.add_argument("--with-t4", action="store_true"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_all)
    p = sub.add_parser("tables"); p.add_argument("--L", default="le28k"); p.set_defaults(fn=cmd_tables)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    r = a.fn(a)
    return int(r) if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
