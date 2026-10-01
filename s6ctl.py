#!/usr/bin/env python
"""S6 — 64k/128k rows, dilution / long-benign FPR / decoy / framed, long baselines, LG3-8B + dense-FT 8B training
(plan S6: E2-1·2 64k·128k, E2-4, E2-6, E3-1·2·3, B1 64k·128k, B2 CD long, B4·B5·B6·B7, T5·T6).

    python s6ctl.py e2-long --models lg:qwen3guard_0.6b_s0,dense:qwen3guard_0.6b_s0:yarn --sbatch   # E2-1 64k/128k rows (same rows file as S5)
    python s6ctl.py var --models lg:qwen3guard_0.6b_s0,dense:qwen3guard_0.6b_s0 --sbatch              # needle_repeat / decoy / framed / ood_domain
    python s6ctl.py b1-long --sbatch                    # zero-shot 12 at 64k/128k (128 pairs x 5 depths + 250 benign-long per L; yarn / trunc / native)
    python s6ctl.py b2-long --sbatch                    # CD default config (8B W=512 o=256) on the same 64k/128k subset
    python s6ctl.py b1-var --sbatch                     # zero-shot 3 + CD on the S6 variant subsets (4k, 28k; 128k with the long policies)
    python s6ctl.py b6 --sbatch                         # Prompt Guard 2-86M (512-token chunk-OR) <=28k + 64k/128k subset
    python s6ctl.py t5 --sbatch                         # LG3-8B: prep (llama3 ids) -> head -> router -> valcheck
    python s6ctl.py t6 --sbatch                         # dense-FT 8B (LoRA, <=8k)
    python s6ctl.py all --sbatch                        # everything above that is ready (t5/t6 with --with-t5 / --with-t6)
    python s6ctl.py tables                              # reports/s5/table2_allL.md (+fig3/4 all L) and reports/s6/{fig5,fig6,table_variants,table2_cascade,g2_check}
    python s6ctl.py status

Every job resumes by row id.  The S5 rows files are extended in place (the 64k/128k ids are the only todo), so `s5ctl
tables --L all` / `s6ctl tables` see all 8 lengths in one file.
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
from lgs5.config import E2_L_LONG, VAR_L, VAR_S6, ModelSpec, S5Paths, parse_specs  # noqa: E402

TOK_ENV = "export TOKENIZERS_PARALLELISM=true\nexport PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\n"
L_LONG = ",".join(str(x) for x in E2_L_LONG)
L_VAR_LE28K = ",".join(str(x) for x in VAR_L if x <= 28672)
L_VAR_LONG = ",".join(str(x) for x in VAR_L if x > 28672)
SUB_LONG = "--max-pairs_128_--depths_0,0.25,0.5,0.75,1_--max-bl_250"          # plan S6 B1: 128 pairs x 5 depths x 2 L + 250 bl / L
# NOTE (2026-09-27): the test grid is 0.0, 0.1, ..., 1.0, so 0.25 / 0.75 match nothing and the B1-long rows cover THREE positions
# (0, .5, 1) = 768 main rows per L.  Kept as is: every 64k/128k guard row (Table 2/3) was produced on this subset; changing it
# now would mix subsets across guards.  Report it as 3 positions.
PREFILL = "--prefill-chunk_8192"
B1_VAR_GUARDS = ["qwen3guard_gen_0.6b", "qwen3guard_gen_8b", "llamaguard3_8b"]
SEEDS_06B = s5ctl.SEEDS_06B


def _submit(job: str, body: str, time_limit: str, a, dep: Optional[str] = None, **kw) -> Optional[str]:
    return s1ctl.write_sbatch(job, TOK_ENV + body, time_limit, dep=dep if dep is not None else getattr(a, "dep", None),
                              dry_run=getattr(a, "dry_run_sbatch", False), **kw)


def _tier(spec: ModelSpec, long: bool = True) -> int:
    big = "4b" in spec.name or "8b" in spec.name
    if spec.kind == "dense":
        # head / tail truncation processes <= 32k tokens; yarn runs the full 128k context
        return 48 if (big or (spec.policy == "yarn" and long)) else 24
    return 48 if big else 24                     # LG streams window chunks: 0.6B at 128k fits 24 GB


def _job_name(exp: str, spec: ModelSpec) -> str:
    return f"s6_{exp}_{spec.tag}"[:60].replace(":", "_")


def _eval_jobs(a, exp: str, specs: List[ModelSpec], extra: str, time_key: str, default_time: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for spec in specs:
        body = f"${{PY}} -m lgs5.run --exp {exp} --model {spec}{extra}"
        if a.sbatch:
            out[str(spec)] = _submit(_job_name(exp, spec), body, env(time_key, default_time), a, min_gpu_gb=_tier(spec),
                                     signal=s5ctl.EVAL_SIGNAL, requeue=True)     # self-requeue at the limit (9/25)
        else:
            rc = s5ctl._run_local("lgs5.run", ["--exp", exp, "--model", str(spec)] + extra.split())
            out[str(spec)] = "ok" if rc == 0 else f"rc={rc}"
    return out


# ------------------------------------------------------------------------------------------------
def cmd_e2_long(a):
    """E2-1 64k/128k (+ calib benign-long / calib main at those L) for LG / sliding / dense rows: appended to the S5 rows files."""
    specs = parse_specs(a.models)
    extra = f" --L {a.L}" + (f" --depths {a.depths}" if a.depths else "") + (f" --max-pairs {a.max_pairs}" if a.max_pairs else "")
    if a.stream != "auto":
        extra += f" --stream {a.stream}"
    if getattr(a, "prefill_chunk", None):
        extra += f" --prefill-chunk {a.prefill_chunk}"
    out = _eval_jobs(a, "e2", specs, extra, "LG_TIME_S6_E2", "36:00:00")
    print(f"[e2-long] {out}")
    return 0


def cmd_var(a):
    """Variant subsets (default needle_repeat, decoy, framed, ood_domain; L 4k/28k/128k) for lg / sliding / dense rows."""
    specs = [s for s in parse_specs(a.models) if s.kind in ("lg", "sliding", "dense")]
    extra = f" --subsets {a.subsets}" + (f" --L {a.L}" if a.L else "")
    out = _eval_jobs(a, "var", specs, extra, "LG_TIME_S6_VAR", "24:00:00")
    print(f"[var] {out}")
    return 0


def long_policies(guard: str) -> List[str]:
    """Context policies of a zero-shot guard at 64k/128k (plan S6 B1, Table 3): 32k-native Qwen-family guards get YaRN x4 and
    both truncations; other <128k guards the two truncations; 128k-native guards run natively (chunked prefill)."""
    from lgs1.zeroshot import guards as G
    from lgs3.longbench import CTX_DOC, YARN_FAMILY
    spec = G.get(guard)
    ctx = CTX_DOC.get(guard)
    if ctx is None:
        ctx = 32768 if guard.startswith("qwen3guard") else 131072
    if ctx >= 131072:
        return ["native"]
    pol = ["trunc_keep_head", "trunc_keep_tail"]
    if guard in YARN_FAMILY and spec.kind == "causal":
        pol = ["yarn"] + pol
    return pol


def _b1_long_tasks(bench: str, extra: str) -> List[List[str]]:
    import s3ctl
    tasks = []
    for g in s3ctl.B1_ALL:
        for pol in long_policies(g):
            tasks.append([g, bench, pol, L_LONG, extra])
    return tasks


def cmd_b1_long(a):
    """B1 64k/128k: the 12 zero-shot guards on the dense subset (test) + calib benign-long at those L (matched-FPR)."""
    import s3ctl
    tasks = _b1_long_tasks("v2main", f"{SUB_LONG}_{PREFILL}") + _b1_long_tasks("v2calib", f"--max-bl_250_{PREFILL}")
    if not a.sbatch:
        for t in tasks:
            print(" ".join(t))
        return 0
    jids = s3ctl._submit_longbench(a, tasks, job="s6_b1long")
    print(f"[b1-long] {len(tasks)} tasks in {len(jids)} arrays: {jids}")
    return 0


def cmd_b2_long(a):
    """B2 at 64k/128k: CD default config on the subset (+ calib), sweep configs with --sweep."""
    import s3ctl
    tasks = [["qwen3guard_gen_8b", "v2main", "chunk_or:512:256", L_LONG, SUB_LONG],
             ["qwen3guard_gen_8b", "v2calib", "chunk_or:512:256", L_LONG, "--max-bl_250"]]
    if a.sweep:
        for W in (256, 1024, 2048):
            for ov in (0, W // 2):
                tasks.append(["qwen3guard_gen_8b", "v2main", f"chunk_or:{W}:{ov}", L_LONG, SUB_LONG])
        for g in ("llamaguard3_8b", "gpt_oss_safeguard_20b"):
            tasks.append([g, "v2main", "chunk_or:512:256", L_LONG, SUB_LONG])
    if not a.sbatch:
        for t in tasks:
            print(" ".join(t))
        return 0
    jids = s3ctl._submit_longbench(a, tasks, job="s6_b2long")
    print(f"[b2-long] {len(tasks)} tasks in {len(jids)} arrays: {jids}")
    return 0


def cmd_b1_var(a):
    """Zero-shot guards + CD on the S6 variant subsets: 4k/28k under the S5 policies, 128k under the long policies."""
    import s3ctl
    from lgs3.longbench import CTX_DOC
    bench = "v2:" + ",".join(a.subsets.split(","))
    tasks = []
    for g in [x for x in a.guards.split(",") if x]:
        if CTX_DOC.get(g, 32768) < 28672 + 512:
            tasks.append([g, bench, "trunc_keep_head", L_VAR_LE28K, "-"])
        else:
            tasks.append([g, bench, "native", L_VAR_LE28K, "-"])
        for pol in long_policies(g):
            tasks.append([g, bench, pol, L_VAR_LONG, PREFILL])
    tasks.append(["qwen3guard_gen_8b", bench, "chunk_or:512:256", ",".join(str(x) for x in VAR_L), "-"])
    if not a.sbatch:
        for t in tasks:
            print(" ".join(t))
        return 0
    jids = s3ctl._submit_longbench(a, tasks, job="s6_b1var")
    print(f"[b1-var] {len(tasks)} tasks in {len(jids)} arrays: {jids}")
    return 0


def cmd_b6(a):
    """B6 small encoders: Prompt Guard 2-86M (512-token chunk-OR) on <=28k (full) and the 64k/128k subset (+ calib)."""
    import s3ctl
    from lgs1.zeroshot import guards as G
    guards = [g for g in ("prompt_guard_2_86m", "gliguard") if g in G.GUARDS]
    tasks = []
    for g in guards:
        tasks.append([g, "v2main", "chunk_or:512:0", s3ctl.L_LE28K, "-"])
        tasks.append([g, "v2calib", "chunk_or:512:0", s3ctl.L_LE28K, "-"])
        tasks.append([g, "v2main", "chunk_or:512:0", L_LONG, SUB_LONG])
        tasks.append([g, "v2calib", "chunk_or:512:0", L_LONG, "--max-bl_250"])
    if not a.sbatch:
        for t in tasks:
            print(" ".join(t))
        return 0
    jids = s3ctl._submit_longbench(a, tasks, job="s6_b6")
    print(f"[b6] {len(tasks)} tasks in {len(jids)} arrays: {jids}")
    return 0


def cmd_t5(a):
    """T5 LG3-8B: Llama-3 token arrays (s4/prep_llama3, CPU) -> head (LoRA r=16) -> router -> valcheck."""
    import s4ctl
    s4ctl._check_sealed(a)
    if not a.sbatch:
        raise SystemExit("t5 is an sbatch chain")
    from lgs4.config import S4Paths
    P4 = S4Paths.load().for_tok("llama3")
    dep = a.dep
    start = getattr(a, "from_stage", "head")
    if start == "head" and not (P4.prep / "train" / "main" / "ids.npy").exists():
        a.tok = "llama3"
        dep = s4ctl.cmd_prep(a)
    a.backbone = "llamaguard3_8b"
    out = s4ctl._chain(a, "llamaguard3_8b", 0, dep, start=start)
    print(f"[t5] prep_dep={dep} {out}")
    return 0


def cmd_t6(a):
    """T6 dense-FT 8B (LoRA on every layer, <=8k): step budget from the 8B head.pt (token-matched), so it waits for the head."""
    import s4ctl
    s4ctl._check_sealed(a)
    if not a.sbatch:
        raise SystemExit("t6 is an sbatch chain")
    from lgs4.config import S4Paths
    P4 = S4Paths.load()
    dep = a.dep
    if dep is None and not (P4.run_dir("qwen3guard_8b", 0) / "head.pt").exists():
        dep = s4ctl._head_job_id("qwen3guard_8b", 0)
        if dep is None:
            raise SystemExit("[t6] no runs/v2/qwen3guard_8b_s0/head.pt and no queued 8B head job: run `python s5ctl.py t4 --sbatch` first")
    a.backbone, a.seed = "qwen3guard_8b", 0
    extra = (f" --steps {a.steps}" if getattr(a, "steps", None) else "") + (f" --limit-rows {a.limit_rows}" if getattr(a, "limit_rows", None) else "")
    jid = s4ctl._stage_job(a, "dense", "lgs4.dense_train", "qwen3guard_8b", 0, "LG_TIME_S4_DENSE", "48:00:00", dep=dep, extra=extra)
    print(f"[t6] dense 8B -> {jid} (dep {dep})")
    return 0


def _ready_specs(P: S5Paths) -> Dict[str, List[str]]:
    lg = [f"lg:qwen3guard_0.6b_s{s}" for s in SEEDS_06B if (P.run_dir(f"qwen3guard_0.6b_s{s}") / "router.pt").exists()]
    for big in ("qwen3guard_4b_s0", "qwen3guard_8b_s0", "llamaguard3_8b_s0"):
        if (P.run_dir(big) / "router.pt").exists():
            lg.append(f"lg:{big}")
    dense = []
    for s in SEEDS_06B:
        if (P.run_dir(f"qwen3guard_0.6b_s{s}") / "dense_ft.pt").exists():
            dense += [f"dense:qwen3guard_0.6b_s{s}", f"dense:qwen3guard_0.6b_s{s}:tail", f"dense:qwen3guard_0.6b_s{s}:yarn"]
    if (P.run_dir("qwen3guard_8b_s0") / "dense_ft.pt").exists():
        dense += ["dense:qwen3guard_8b_s0", "dense:qwen3guard_8b_s0:yarn"]
    sliding = [f"sliding:qwen3guard_0.6b_s0"] if (P.run_dir("qwen3guard_0.6b_s0") / "router.pt").exists() else []
    return {"lg": lg, "dense": dense, "sliding": sliding}


def cmd_all(a):
    P = S5Paths.load()
    r = _ready_specs(P)
    if not r["lg"]:
        raise SystemExit("no trained LG run with router.pt under runs/v2 (S4 T1 first)")
    a.models = ",".join(r["lg"] + r["dense"] + r["sliding"])
    a.L, a.depths, a.max_pairs, a.stream = "long", None, None, "auto"
    cmd_e2_long(a)
    a.models = ",".join(r["lg"][:1] + [x for x in r["lg"] if "0.6b" not in x] + [x for x in r["dense"] if x.endswith("_s0")][:1])
    a.subsets, a.L = ",".join(VAR_S6), ""
    cmd_var(a)
    a.sweep = getattr(a, "sweep", False)
    cmd_b1_long(a)
    cmd_b2_long(a)
    a.guards, a.subsets = ",".join(B1_VAR_GUARDS), ",".join(VAR_S6)
    cmd_b1_var(a)
    cmd_b6(a)
    if getattr(a, "with_t5", False):
        cmd_t5(a)
    if getattr(a, "with_t6", False):
        cmd_t6(a)
    print("[all] when the jobs finish: python s6ctl.py tables")
    return 0


def cmd_tables(a):
    """S5 all-L + S6 tables.  --sbatch runs the aggregation as a CPU job (9/26: ~2M rows across e2 / longbench / var
    files take 10+ min and a few GB on the login node; the output lands in reports/ either way)."""
    args = ["--L", a.L] + (["--no-s5"] if a.no_s5 else [])
    if getattr(a, "sbatch", False):
        body = f"${{PY}} -m lgs6.tables {' '.join(args)}"
        jid = s1ctl.write_sbatch("s6_tables", TOK_ENV + body, env("LG_TIME_S6_TABLES", "06:00:00"),
                                 dry_run=getattr(a, "dry_run_sbatch", False), min_gpu_gb=8, cpu=True)
        print(f"[tables] job {jid}: reports/s5/*_allL.md + reports/s6/*.md when it ends (log: logs/s6_tables_{jid}.out)")
        return 0
    return s5ctl._run_local("lgs6.tables", args)


def _count_L(path: Path, min_L: int) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("L_target") or 0) >= min_L:
                n += 1
    return n


def _newest_log(P: S5Paths, prefix: str, tag: str) -> Optional[Path]:
    logs = glob.glob(str(P.out / "logs" / f"{prefix}_{tag}_[0-9]*.out"))
    return Path(max(logs, key=os.path.getmtime)) if logs else None


def cmd_status(a):
    P = S5Paths.load()
    print(f"\nS6 status  ({time.strftime('%Y-%m-%d %H:%M')}, eval={P.eval})\n")
    queued = s5ctl._queued_names()
    for exp, minL in (("e2", 65536), ("var", 0)):
        files = sorted(glob.glob(str(P.eval / exp / "*.rows.jsonl")))
        print(f"  {exp}: {len(files)} rows files" + ("  (rows at L >= 64k)" if minL else ""))
        for f in files:
            n = _count_L(Path(f), minL)
            tag = Path(f).name[: -len(".rows.jsonl")]
            # the newest log of either controller (s6ctl e2-long / var, or s5ctl e2 for oracle / needle / sliding)
            cands = [(p, _newest_log(P, f"{p}_{exp}", tag)) for p in ("s6", "s5")]
            cands = [(p, l) for p, l in cands if l is not None]
            st, job = "", ""
            if cands:
                pfx, log = max(cands, key=lambda x: os.path.getmtime(x[1]))
                job = f"{pfx}_{exp}_{tag}"[:60].replace(":", "_")
                st = s5ctl.progress_from_log(log)
                if st.startswith("running") and queued is not None and job not in queued:
                    st = f"STOPPED (no job in queue) -> {Path(log).with_suffix('.err').name}"
            print(f"     {Path(f).name:60s} {n:>8,} rows   {st}")
    for d in sorted(glob.glob(str(P.longbench / "v2*"))):
        lb = sorted(glob.glob(str(Path(d) / "*.rows.jsonl")))
        n_long = sum(_count_L(Path(f), 65536) for f in lb)
        print(f"  longbench/{Path(d).name}: {len(lb)} rows files, {n_long:,} rows at L >= 64k")
        if getattr(a, "verbose", False):
            for f in lb:
                print(f"     {Path(f).name:60s} {_count_L(Path(f), 0):>8,} rows  ({_count_L(Path(f), 65536):,} at >= 64k)")
    from lgs4.config import S4Paths
    P4 = S4Paths.load()
    for run in ("llamaguard3_8b_s0", "qwen3guard_8b_s0"):
        d = P4.runs / run
        have = [f for f in ("head.pt", "router.pt", "dense_ft.pt", "val_report.json") if (d / f).exists()]
        print(f"  runs/v2/{run}: {have or 'nothing yet'}")
    reps = sorted(glob.glob(str(P.out / "reports" / "s6" / "*.md")))
    print(f"  reports/s6: {[Path(r).name for r in reps]}")
    os.system("squeue -u $USER -o '%.12i %.40j %.8T %.10M %.6D %R' 2>/dev/null | grep -E 'JOBID|s6_|s5_e2|s5_b|s3_longbench|s4_head_llama|s4_router_llama|s4_valcheck_llama|s4_dense_qwen3guard_8b|s4_prep' || true")
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
        p.add_argument("--force", action="store_true", help="longbench: submit even if the same task is queued")
        p.add_argument("--variant", default="")
        p.add_argument("--probe", default=None)

    p = sub.add_parser("e2-long"); common(p); p.add_argument("--models", required=True); p.add_argument("--L", default="long")
    p.add_argument("--depths", default=None); p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--stream", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--prefill-chunk", type=int, default=None, help="dense: KV-cache prefill chunk for rows > 32k (default 2048; 1024 for 8B)"); p.set_defaults(fn=cmd_e2_long)
    p = sub.add_parser("var"); common(p); p.add_argument("--models", required=True); p.add_argument("--subsets", default=",".join(VAR_S6))
    p.add_argument("--L", default="", help="default: every L of the subsets (4k, 28k, 128k)"); p.set_defaults(fn=cmd_var)
    p = sub.add_parser("b1-long"); common(p); p.set_defaults(fn=cmd_b1_long)
    p = sub.add_parser("b2-long"); common(p); p.add_argument("--sweep", action="store_true"); p.set_defaults(fn=cmd_b2_long)
    p = sub.add_parser("b1-var"); common(p); p.add_argument("--guards", default=",".join(B1_VAR_GUARDS)); p.add_argument("--subsets", default=",".join(VAR_S6))
    p.set_defaults(fn=cmd_b1_var)
    p = sub.add_parser("b6"); common(p); p.set_defaults(fn=cmd_b6)
    p = sub.add_parser("t5"); common(p); p.add_argument("--from", dest="from_stage", default="head", choices=["head", "router", "valcheck"]); p.set_defaults(fn=cmd_t5)
    p = sub.add_parser("t6"); common(p); p.add_argument("--steps", type=int, default=None); p.add_argument("--limit-rows", type=int, default=None); p.set_defaults(fn=cmd_t6)
    p = sub.add_parser("all"); common(p); p.add_argument("--sweep", action="store_true"); p.add_argument("--with-t5", action="store_true")
    p.add_argument("--with-t6", action="store_true"); p.set_defaults(fn=cmd_all)
    p = sub.add_parser("tables"); p.add_argument("--L", default="all"); p.add_argument("--no-s5", action="store_true")
    p.add_argument("--sbatch", action="store_true", help="aggregate as a CPU job instead of on the login node")
    p.add_argument("--dry-run-sbatch", action="store_true"); p.set_defaults(fn=cmd_tables)
    p = sub.add_parser("status"); p.add_argument("--verbose", action="store_true"); p.set_defaults(fn=cmd_status)
    return ap


def main(argv=None) -> int:
    load_env_file()
    a = build_parser().parse_args(argv)
    r = a.fn(a)
    return int(r) if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
