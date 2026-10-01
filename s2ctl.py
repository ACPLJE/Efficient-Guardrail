#!/usr/bin/env python3
"""Stage S2 controller — background screening (D4), needle pool / splits / audit (D5), synthesis dry-run (D6).

    python s2ctl.py haystack --sbatch               # D4a: 10 domain pools (CPU job; --local runs here)
    python s2ctl.py screen --sbatch [--dep <haystack jobid>]   # D4b: array of 20 (domain x model) GPU tasks
    python s2ctl.py screen --summarize              # merge -> keep lists, screen_summary.json, manual_review_50.md
    python s2ctl.py embed --sbatch                  # bge-m3 chunk embeddings (array over domains; after summarize)
    python s2ctl.py needles --local|--sbatch        # D5a: needles/pool.jsonl.gz
    python s2ctl.py splits --local|--sbatch         # D5b: splits/*.json + audit.json (+ gates)
    python s2ctl.py synth --dry-run [--allow-unscreened] --local|--sbatch     # D6: ~200 rows + schema checks
    python s2ctl.py status                          # S2 checklist
    python s2ctl.py all --sbatch                    # haystack -> screen array -> (summarize, splits, dry-run); needles in parallel

Every --sbatch job takes --gres=gpu:1 (QOS normal requires it) and no array concurrency limit; partitions
come from the GPU-memory tiers in lgs1.slurmutil.  Paths: env/s1.env (LG_SOURCES -> LG_DATA_V2 root).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import s1ctl  # noqa: E402  (write_sbatch, partitions, env helpers)
from lgs1.paths import env, load_env_file  # noqa: E402
from lgs2.config import DOMAINS, SCREEN, S2Paths  # noqa: E402

CPU_JOB = dict(min_gpu_gb=8, cpu=True)     # tokenisation / synthesis: LG_CPU_PARTITIONS when set, else the smallest GPU tier (QOS normal needs --gres=gpu:1)
TOK_ENV = "export TOKENIZERS_PARALLELISM=true\n"


def _submit(job: str, body: str, time_limit: str, a, array: Optional[str] = None, **kw) -> Optional[str]:
    return s1ctl.write_sbatch(job, TOK_ENV + body, time_limit, array=array, dep=getattr(a, "dep", None),
                              dry_run=getattr(a, "dry_run_sbatch", False), **kw)


def _run_local(module: str, args: List[str]) -> int:
    """Run a lgs2 module in this shell — always under $LG_PYTHON (the venv with torch/pyarrow), even when
    `python s2ctl.py` itself was started with the system python."""
    import runpy
    import subprocess
    py = S2Paths.load().python
    if py and Path(py).exists() and Path(py).resolve() != Path(sys.executable).resolve():
        env_ = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.run([py, "-m", module] + args, cwd=str(ROOT), env=env_).returncode
    sys.argv = [module] + args
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as e:
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            return e.code
        print(f"[s2ctl] {module}: {e.code}", file=sys.stderr)
        return 1
    return 0


def _domains(a) -> List[str]:
    return [d.name for d in DOMAINS] if a.domains == "all" else [x.strip() for x in a.domains.split(",") if x.strip()]


# ------------------------------------------------------------------------------------------------
def cmd_haystack(a):
    doms = ",".join(_domains(a))
    extra = f" --limit-docs {a.limit_docs}" if a.limit_docs else ""
    if a.sbatch:
        return _submit("s2_haystack", f"${{PY}} -m lgs2.haystack --domains {doms}{extra}", env("LG_TIME_S2_CPU", "06:00:00"), a, **CPU_JOB)
    return _run_local("lgs2.haystack", ["--domains", doms] + (["--limit-docs", str(a.limit_docs)] if a.limit_docs else []))


def cmd_screen(a):
    P = S2Paths.load().ensure()
    if a.summarize:
        return _run_local("lgs2.screen", ["--summarize"])
    models = SCREEN["models"] if a.models == "both" else [m for m in a.models.split(",") if m]
    tasks = [(d, m) for d in _domains(a) for m in models]
    if a.sbatch:
        stamp = time.strftime("%m%d_%H%M%S")
        lst = P.out / "slurm" / f"screen_tasks_{stamp}.txt"
        lst.write_text("\n".join(f"{d} {m}" for d, m in tasks) + "\n")
        body = (f"read DOM MODEL <<< \"$(sed -n \"$((SLURM_ARRAY_TASK_ID+1))p\" {lst})\"\n"
                f"echo \"[s2] screen domain=$DOM model=$MODEL\"\n"
                f"${{PY}} -m lgs2.screen --domain $DOM --model $MODEL" + (f" --limit {a.limit}" if a.limit else ""))
        jid = _submit("s2_screen", body, env("LG_TIME_S2_SCREEN", "12:00:00"), a, array=f"0-{len(tasks) - 1}", min_gpu_gb=24)
        print(f"         {len(tasks)} tasks: {tasks[:4]}{' ...' if len(tasks) > 4 else ''}")
        return jid
    rc = 0
    for d, m in tasks:
        rc |= _run_local("lgs2.screen", ["--domain", d, "--model", m] + (["--limit", str(a.limit)] if a.limit else []))
    return rc


def cmd_embed(a):
    P = S2Paths.load().ensure()
    doms = _domains(a)
    if a.sbatch:
        stamp = time.strftime("%m%d_%H%M%S")
        lst = P.out / "slurm" / f"embed_domains_{stamp}.txt"
        lst.write_text("\n".join(doms) + "\n")
        body = (f"DOM=$(sed -n \"$((SLURM_ARRAY_TASK_ID+1))p\" {lst})\n"
                f"${{PY}} -m lgs2.embed --domain $DOM" + (f" --limit {a.limit}" if a.limit else ""))
        return _submit("s2_embed", body, env("LG_TIME_S2_EMBED", "06:00:00"), a, array=f"0-{len(doms) - 1}", min_gpu_gb=24)
    rc = 0
    for d in doms:
        rc |= _run_local("lgs2.embed", ["--domain", d] + (["--limit", str(a.limit)] if a.limit else []))
    return rc


def cmd_needles(a):
    if a.sbatch:
        return _submit("s2_needles", "${PY} -m lgs2.needles", env("LG_TIME_S2_CPU", "06:00:00"), a, **CPU_JOB)
    return _run_local("lgs2.needles", [])


def cmd_splits(a):
    if a.sbatch:
        return _submit("s2_splits", "${PY} -m lgs2.splits", "02:00:00", a, **CPU_JOB)
    return _run_local("lgs2.splits", [])


def cmd_synth(a):
    args = []
    if a.dry_run:
        args.append("--dry-run")
        if a.allow_unscreened:
            args.append("--allow-unscreened")
    else:
        args += ["--split", a.split]
        if a.set:
            args += ["--set", a.set]
        if a.native:
            args.append("--native")
        if a.pairs:
            args += ["--pairs", str(a.pairs)]
    if a.sbatch:
        return _submit("s2_synth", "${PY} -m lgs2.synth " + " ".join(args), env("LG_TIME_S2_SYNTH", "24:00:00"), a, **CPU_JOB)
    return _run_local("lgs2.synth", args)


def cmd_all(a):
    """haystack -> screen array -> finish job (summarize + splits + dry-run); needles pool in parallel."""
    a.dep = None
    a.domains, a.limit_docs = "all", None
    j_hay = cmd_haystack(a)
    a.dep = j_hay
    a.models, a.limit, a.summarize = "both", None, False
    j_screen = cmd_screen(a)
    a.dep = None
    j_needles = cmd_needles(a)
    a.dep = f"{j_screen}:{j_needles}" if j_screen and j_needles else (j_screen or j_needles)
    body = ("${PY} -m lgs2.screen --summarize\n"
            "${PY} -m lgs2.splits\n"
            "${PY} -m lgs2.synth --dry-run")
    j_fin = _submit("s2_finish", body, "04:00:00", a, **CPU_JOB)
    a.dep = j_fin
    a.limit = None
    j_emb = cmd_embed(a)
    print(f"[all] haystack={j_hay} screen={j_screen} needles={j_needles} finish={j_fin} embed={j_emb}")
    return 0


def cmd_status(a):
    P = S2Paths.load()
    rows = []
    hay = {d.name: (P.haystack / d.name / "stats.json") for d in DOMAINS}
    n_hay = sum(1 for f in hay.values() if f.exists())
    toks = 0
    for f in hay.values():
        if f.exists():
            toks += json.loads(f.read_text()).get("tokens", 0)
    rows.append(("D4a haystack", n_hay == len(DOMAINS), f"{n_hay}/{len(DOMAINS)} domains, {toks:,} tokens collected"))
    n_rows_files = sum(1 for d in DOMAINS for m in SCREEN["models"] if (P.screen / f"{d.name}.{m}.rows.jsonl").exists())
    ss = P.screen / "screen_summary.json"
    if ss.exists():
        s = json.loads(ss.read_text())
        comp = [d for d, v in s["domains"].items() if v.get("status") == "complete"]
        met = [d for d, v in s["domains"].items() if v.get("budget_met")]
        rows.append(("D4b screening", len(comp) == len(DOMAINS), f"complete {len(comp)}/{len(DOMAINS)}, budget met {len(met)}/{len(DOMAINS)}; "
                     f"rows files {n_rows_files}/{len(DOMAINS) * len(SCREEN['models'])}; manual_review_50.md"))
    else:
        rows.append(("D4b screening", False, f"rows files {n_rows_files}/{len(DOMAINS) * len(SCREEN['models'])}; run `s2ctl.py screen --summarize` when the array is done"))
    n_emb = sum(1 for d in DOMAINS if (P.related / f"{d.name}.emb.npy").exists())
    rows.append(("D4c bge-m3 related", n_emb == len(DOMAINS), f"{n_emb}/{len(DOMAINS)} domains embedded"))
    ps = P.needles / "pool_stats.json"
    if ps.exists():
        st = json.loads(ps.read_text())
        miss = [k for k, v in st["sources"].items() if v.get("status") != "ok"]
        rows.append(("D5a needle pool", not miss, f"{st['n_total']:,} rows, {st['groups']['n_groups']:,} groups; "
                     f"sources not ok: {miss or 'none'}"))
    else:
        rows.append(("D5a needle pool", False, "run `s2ctl.py needles`"))
    au = P.splits / "audit.json"
    if au.exists():
        ad = json.loads(au.read_text())
        g = ad["gates"]
        rows.append(("D5b splits/audit", g["pass"] and ad.get("haystack_screened", False),
                     f"gates {'PASS' if g['pass'] else 'FAIL'}; removed {ad['overlap']['n_removed']} "
                     f"(exact {ad['overlap']['n_removed_exact']}, near {ad['overlap']['n_removed_near_dup']}); "
                     f"haystack {'screened' if ad.get('haystack_screened') else 'PROVISIONAL'}"))
    else:
        rows.append(("D5b splits/audit", False, "run `s2ctl.py splits`"))
    dr = P.synth_dryrun / "dryrun_report.json"
    if dr.exists():
        r = json.loads(dr.read_text())
        v = r["validation"]
        rows.append(("D6 synth dry-run", v["pass"] and not r.get("haystack_provisional"),
                     f"{v['n_rows']} rows, {'PASS' if v['pass'] else 'FAIL'}"
                     f"{' (provisional haystack)' if r.get('haystack_provisional') else ''}; problems: {v['problems'] or 'none'}"))
    else:
        rows.append(("D6 synth dry-run", False, "run `s2ctl.py synth --dry-run`"))
    print(f"\nS2 status  ({time.strftime('%Y-%m-%d %H:%M')}, data={P.root})\n")
    w = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"  {name:{w}s}  {'DONE   ' if ok else 'PENDING'}  {detail}")
    print()
    os.system("squeue -u $USER -o '%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null | grep -E 'JOBID|s2_' || true")
    return 0


# ------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, domains=True):
        p.add_argument("--sbatch", action="store_true")
        p.add_argument("--local", action="store_true", help="run in this shell (login node / interactive GPU)")
        p.add_argument("--dep", default=None, help="sbatch afterok dependency job id(s), colon-separated")
        p.add_argument("--dry-run-sbatch", action="store_true", help="print the sbatch script instead of submitting")
        if domains:
            p.add_argument("--domains", default="all")

    p = sub.add_parser("haystack"); common(p); p.add_argument("--limit-docs", type=int, default=None); p.set_defaults(fn=cmd_haystack)
    p = sub.add_parser("screen"); common(p)
    p.add_argument("--models", default="both"); p.add_argument("--limit", type=int, default=None)
    p.add_argument("--summarize", action="store_true"); p.set_defaults(fn=cmd_screen)
    p = sub.add_parser("embed"); common(p); p.add_argument("--limit", type=int, default=None); p.set_defaults(fn=cmd_embed)
    p = sub.add_parser("needles"); common(p, domains=False); p.set_defaults(fn=cmd_needles)
    p = sub.add_parser("splits"); common(p, domains=False); p.set_defaults(fn=cmd_splits)
    p = sub.add_parser("synth"); common(p, domains=False)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--allow-unscreened", action="store_true")
    p.add_argument("--split", default="test"); p.add_argument("--set", default=""); p.add_argument("--native", action="store_true")
    p.add_argument("--pairs", type=int, default=None); p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("all"); common(p); p.set_defaults(fn=cmd_all)
    return ap


def main(argv=None) -> int:
    load_env_file()
    a = build_parser().parse_args(argv)
    if getattr(a, "sbatch", False) and getattr(a, "local", False):
        raise SystemExit("choose --sbatch or --local")
    r = a.fn(a)
    return 0 if r is None or isinstance(r, str) else int(r)


if __name__ == "__main__":
    sys.exit(main())
