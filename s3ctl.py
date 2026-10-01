#!/usr/bin/env python3
"""Stage S3 controller — full synthesis (D6), variants (D7), per-model tokens (D8), E0-4 gates + seal, data card (D10),
E0-5 pilot, and the long-context baseline jobs (B1/B2).

    python s3ctl.py synth --sbatch [--splits test,calib,val,train]     # 4 CPU jobs (one per split; train = main+16k+native)
    python s3ctl.py variants --sbatch --dep <test synth jobid>          # array over the 9 variant subsets (GPU: bge-m3 for related)
    python s3ctl.py tokspans --sbatch --dep <jobids>                    # array over 5 tokenizers x 4 splits
    python s3ctl.py finish --sbatch --dep <tokspans jobid>              # gates + datacard + pilot (what `all` chains last)
    python s3ctl.py gates [--seal]                                      # E0-4 (login node, venv); --seal makes lg_data_v2 read-only
    python s3ctl.py datacard                                            # DATACARD.md / Table A1 draft
    python s3ctl.py pilot --sbatch|--local                              # E0-5 tiny pilot on the dry-run rows
    python s3ctl.py longbench --guards qwen3guard_gen_8b --bench v2main --policy chunk_or:512:256 --L 1024,...,28672 --sbatch
    python s3ctl.py b1-start --sbatch                                   # plan S3 item 7: 12 guards on v2 test <=28k + SafetyNIAH (+CD)
    python s3ctl.py longbench --summarize --bench v2main
    python s3ctl.py status
    python s3ctl.py all --sbatch                                        # synth -> variants -> tokspans -> (gates, datacard, pilot)

Every sbatch job takes --gres=gpu:1 (QOS normal) with no array concurrency limit; the GPU-memory tier comes from the
guard (24 GB: <=9B, 40 GB: LG4-12B, 48 GB: gpt-oss-20b) and from L (>= 64k tokens -> 48 GB).
"""
from __future__ import annotations

import argparse
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
from lgs1.paths import env, load_env_file  # noqa: E402
from lgs2.config import RECIPE_V2, S2Paths  # noqa: E402
from lgs3.tokspans import TOK_MODELS  # noqa: E402
from lgs3.variants import VARIANT_NAMES  # noqa: E402

TOK_ENV = "export TOKENIZERS_PARALLELISM=true\n"
SPLIT_SETS = {"test": "--split test --set test", "calib": "--split calib --set calib", "val": "--split val --set val",
              "train": "--split train --set train,train16k --native"}
B1_LOW = ["qwen3guard_gen_0.6b", "qwen3guard_gen_4b", "qwen3guard_gen_8b", "qwen3guard_stream_8b", "llamaguard3_8b", "wildguard_7b",
          "guardreasoner_8b", "polyguard_qwen_7b", "granite_guardian_3.3_8b", "nemotron_safety_guard_v3"]
B1_ALL = B1_LOW + ["llamaguard4_12b", "gpt_oss_safeguard_20b"]
L_LE28K = "1024,2048,4096,8192,16384,28672"


def _submit(job: str, body: str, time_limit: str, a, array: Optional[str] = None, **kw) -> Optional[str]:
    return s1ctl.write_sbatch(job, TOK_ENV + body, time_limit, array=array, dep=getattr(a, "dep", None),
                              dry_run=getattr(a, "dry_run_sbatch", False), **kw)


def _run_local(module: str, args: List[str]) -> int:
    py = S2Paths.load().python
    if py and Path(py).exists() and Path(py).resolve() != Path(sys.executable).resolve():
        env_ = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.run([py, "-m", module] + args, cwd=str(ROOT), env=env_).returncode
    import runpy
    sys.argv = [module] + args
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as e:
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            return e.code
        print(f"[s3ctl] {module}: {e.code}", file=sys.stderr)
        return 1
    return 0


def _tier_for(guard: str, Ls: List[int]) -> int:
    from lgs1.zeroshot import guards as G
    gb = G.get(guard).min_gpu_gb if guard in G.GUARDS else 24
    if Ls and max(Ls) >= 65536:
        gb = max(gb, 48)
    elif Ls and max(Ls) >= 28672 and gb < 24:
        gb = 24
    return gb


LONGBENCH_SIGNAL = "B:TERM@900"


def _array_job(a, job: str, lines: List[str], body_tpl: str, time_limit: str, min_gpu_gb: int, list_name: str, cpu: bool = False) -> Optional[str]:
    P = S2Paths.load().ensure()
    # one list per ARRAY: the name carries the job (tier) -- a second-resolution stamp alone let two tiers submitted in the
    # same second overwrite each other's list (b1-start 9/19, b1-calib 9/20: the 24 GB array read the 40 GB tier's one line).
    lst = P.out / "slurm" / f"{list_name}_{time.strftime('%m%d_%H%M%S')}_{job}.txt"
    lst.write_text("\n".join(lines) + "\n")
    body = f"LINE=$(sed -n \"$((SLURM_ARRAY_TASK_ID+1))p\" {lst})\n" + body_tpl
    # longbench tasks are resumable by id: SIGTERM 15 min before the limit (a GuardReasoner 128k row takes ~6 min),
    # flush, rc=3, and the task requeues itself with the same id (9/25; before, tasks died at 48 h and were resubmitted)
    kw = dict(signal=LONGBENCH_SIGNAL, requeue=True) if list_name == "longbench" else {}
    return _submit(job, body, time_limit, a, array=f"0-{len(lines) - 1}", min_gpu_gb=min_gpu_gb, cpu=cpu, **kw)


# ------------------------------------------------------------------------------------------------
def cmd_synth(a):
    splits = [s for s in a.splits.split(",") if s]
    if a.sbatch:
        jids = {}
        for sp in splits:
            jids[sp] = _submit(f"s3_synth_{sp}", f"${{PY}} -m lgs2.synth {SPLIT_SETS[sp]}", env("LG_TIME_S3_SYNTH", "24:00:00"), a, min_gpu_gb=8, cpu=True)
        print(f"[synth] jobs: {jids}")
        return ":".join(j for j in jids.values() if j)
    rc = 0
    for sp in splits:
        rc |= _run_local("lgs2.synth", SPLIT_SETS[sp].split())
    return rc


def cmd_variants(a):
    names = VARIANT_NAMES if a.variant == "all" else [x for x in a.variant.split(",") if x]
    if a.sbatch:
        return _array_job(a, "s3_variants", names, "echo \"[s3] variant=$LINE\"\n${PY} -m lgs3.variants --variant $LINE",
                          env("LG_TIME_S3_VARIANTS", "12:00:00"), 24, "variants")
    rc = 0
    for n in names:
        rc |= _run_local("lgs3.variants", ["--variant", n])
    return rc


def cmd_tokspans(a):
    models = list(TOK_MODELS) if a.models == "all" else [x for x in a.models.split(",") if x]
    splits = ["test", "calib", "val", "train"] if a.splits == "all" else [x for x in a.splits.split(",") if x]
    lines = [f"{m} {sp}" for m in models for sp in splits]
    if a.sbatch:
        return _array_job(a, "s3_tokspans", lines, "read MODEL SPLIT <<< \"$LINE\"\n${PY} -m lgs3.tokspans --model $MODEL --split $SPLIT",
                          env("LG_TIME_S3_TOK", "12:00:00"), 8, "tokspans", cpu=True)
    rc = 0
    for m in models:
        for sp in splits:
            rc |= _run_local("lgs3.tokspans", ["--model", m, "--split", sp])
    return rc


def cmd_dedupe(a):
    return _run_local("lgs3.dedupe", ["--split", a.split, "--subset", a.subset])


def cmd_gates(a):
    return _run_local("lgs3.gates", (["--seal"] if a.seal else []) + (["--dry-run"] if a.dry_run else []))


def cmd_datacard(a):
    return _run_local("lgs3.datacard", [])


def cmd_pilot(a):
    if a.sbatch:
        return _submit("s3_pilot", "${PY} -m lgs3.pilot", "02:00:00", a, min_gpu_gb=8, cpu=True)
    return _run_local("lgs3.pilot", ["--device", "cpu"] if a.cpu else [])


FINISH_BODY = ("${PY} -m lgs3.gates\n"
               "${PY} -m lgs3.datacard\n"
               "${PY} -m lgs3.pilot")


def cmd_finish(a):
    """gates (E0-4) + data card (D10) + pilot (E0-5) in one job; `--dep <tokspans jobid>` after a partial rerun."""
    if a.sbatch:
        return _submit("s3_finish", FINISH_BODY, "04:00:00", a, min_gpu_gb=8, cpu=True)
    rc = _run_local("lgs3.gates", [])
    rc |= _run_local("lgs3.datacard", [])
    rc |= _run_local("lgs3.pilot", ["--device", "cpu"] if getattr(a, "cpu", False) else [])
    return rc


LONGBENCH_JOB_PREFIXES = ("s3_longbench", "s5_b2", "s5_b1calib", "s6_b1long", "s6_b2long", "s6_b1var", "s6_b6")


def queued_longbench() -> set:
    """(guard, bench, policy) triples of PENDING/RUNNING longbench array tasks (task files of the arrays), whatever
    controller submitted them (S3 b1-start, S5 b2 / b1-calib, S6 b1-long / b2-long / b1-var / b6)."""
    user = env("LG_USER") or os.environ.get("USER") or ""
    try:
        txt = subprocess.run(["squeue", "-r", "-h", "-u", user, "-o", "%i|%j|%T"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return set()
    out = set()
    cache: Dict[str, List[str]] = {}
    for line in txt.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 3 or not cols[1].startswith(LONGBENCH_JOB_PREFIXES) or cols[2] not in ("PENDING", "RUNNING", "CONFIGURING"):
            continue
        base, _, task = cols[0].partition("_")
        if base not in cache:
            lines: List[str] = []
            try:
                sc = subprocess.run(["scontrol", "show", "job", base, "-o"], capture_output=True, text=True, timeout=20).stdout
                m = re.search(r"Command=(\S+)", sc)
                if m and Path(m.group(1)).exists():
                    mm = re.search(r"([^\s)]*longbench_\d+_\d+[^\s)]*\.txt)", Path(m.group(1)).read_text())
                    if mm and Path(mm.group(1)).exists():
                        lines = [l.strip() for l in Path(mm.group(1)).read_text().splitlines() if l.strip()]
            except Exception:
                lines = []
            cache[base] = lines
        lines = cache[base]
        idx = int(task) if task.isdigit() else None
        if idx is not None and idx < len(lines):
            out.add(tuple(lines[idx].split()[:3]))
    return out


def _submit_longbench(a, tasks: List[List[str]], job: str = "s3_longbench") -> List[str]:
    """tasks: [guard, bench, policy, L-list(or '-'), extra-args...]; one array per GPU tier."""
    queued = queued_longbench() if not getattr(a, "force", False) else set()
    by_tier: Dict[int, List[str]] = {}
    for t in tasks:
        if tuple(t[:3]) in queued:
            print(f"[longbench] {t[:3]} already queued/running -> skipped (--force to duplicate)")
            continue
        Ls = [int(x) for x in t[3].split(",")] if len(t) > 3 and t[3] != "-" else []
        by_tier.setdefault(_tier_for(t[0], Ls), []).append(" ".join(t))
    jids = []
    for gb, lines in sorted(by_tier.items()):
        body = ("read GUARD BENCH POLICY LS EXTRA <<< \"$LINE\"\n"
                "LARG=\"\"; if [[ \"$LS\" != \"-\" ]]; then LARG=\"--L $LS\"; fi\n"
                "EARG=\"\"; if [[ \"$EXTRA\" != \"-\" ]]; then EARG=\"${EXTRA//_/ }\"; fi\n"
                "echo \"[s3] longbench guard=$GUARD bench=$BENCH policy=$POLICY L=$LS extra=$EARG\"\n"
                "${PY} -m lgs3.longbench --guard $GUARD --bench $BENCH --policy $POLICY $LARG $EARG")
        jid = _array_job(a, f"{job}_{gb}gb", lines, body, env("LG_TIME_S3_LONGBENCH", "48:00:00"), gb, "longbench")
        print(f"         tier {gb} GB: {len(lines)} tasks")
        jids.append(jid)
    return [j for j in jids if j]


def cmd_longbench(a):
    if a.summarize:
        return _run_local("lgs3.longbench", ["--summarize", "--bench", a.bench])
    guards = [g for g in a.guards.split(",") if g]
    if not guards:
        raise SystemExit("--guards required")
    extra = []
    if a.max_pairs is not None:
        extra += [f"--max-pairs_{a.max_pairs}"]
    if a.depths:
        extra += [f"--depths_{a.depths}"]
    if a.max_rows:
        extra += [f"--max-rows_{a.max_rows}"]
    if getattr(a, "max_bl", None):
        extra += [f"--max-bl_{a.max_bl}"]
    if getattr(a, "prefill_chunk", None):
        extra += [f"--prefill-chunk_{a.prefill_chunk}"]
    tasks = [[g, a.bench, a.policy, a.L or "-", " ".join(extra) or "-"] for g in guards]
    if a.sbatch:
        jids = _submit_longbench(a, tasks)
        return ":".join(jids)
    rc = 0
    for g in guards:
        args = ["--guard", g, "--bench", a.bench, "--policy", a.policy] + (["--L", a.L] if a.L else []) + \
               (["--max-pairs", str(a.max_pairs)] if a.max_pairs is not None else []) + (["--depths", a.depths] if a.depths else []) + \
               (["--max-rows", str(a.max_rows)] if a.max_rows else []) + (["--max-bl", str(a.max_bl)] if getattr(a, "max_bl", None) else []) + \
               (["--prefill-chunk", str(a.prefill_chunk)] if getattr(a, "prefill_chunk", None) else [])
        rc |= _run_local("lgs3.longbench", args)
    return rc


def cmd_b1_start(a):
    """Plan S3 item 7 (9/25 evening): B1 zero-shot 12 on v2 test <=28k (native, or head/tail truncation for <32k-ctx
    guards) + SafetyNIAH: CD (Qwen3Guard-8B W=256/512) then zero-shot 12 native."""
    from lgs3.longbench import CTX_DOC
    tasks = []
    for g in B1_ALL:
        if CTX_DOC.get(g, 32768) < 28672 + 512:
            tasks.append([g, "v2main", "trunc_keep_head", L_LE28K, "-"])
            tasks.append([g, "v2main", "trunc_keep_tail", L_LE28K, "-"])
        else:
            tasks.append([g, "v2main", "native", L_LE28K, "-"])
    tasks.append(["qwen3guard_gen_8b", "safetyniah", "chunk_or:512:256", "-", "-"])
    tasks.append(["qwen3guard_gen_8b", "safetyniah", "chunk_or:256:128", "-", "-"])
    for g in B1_ALL:
        tasks.append([g, "safetyniah", "native", "-", "-"])
    if not a.sbatch:
        for t in tasks:
            print(" ".join(t))
        return 0
    jids = _submit_longbench(a, tasks)
    print(f"[b1-start] {len(tasks)} tasks in {len(jids)} arrays: {jids}")
    return 0


def cmd_all(a):
    a.dep = None
    a.splits = "test,calib,val,train"
    j_synth = cmd_synth(a)
    a.dep = j_synth.split(":")[0] if j_synth else None        # variants need the test split only
    a.variant = "all"
    j_var = cmd_variants(a)
    a.dep = ":".join(x for x in [j_synth, j_var] if x) or None
    a.models, a.splits = "all", "all"
    j_tok = cmd_tokspans(a)
    a.dep = j_tok
    j_fin = _submit("s3_finish", FINISH_BODY, "04:00:00", a, min_gpu_gb=8, cpu=True)
    print(f"[all] synth={j_synth} variants={j_var} tokspans={j_tok} finish={j_fin}")
    print("[all] after s3_finish: python s3ctl.py gates --seal   (seals lg_data_v2 when the 7 gates pass) -> then b1-start --sbatch")
    return 0


def cmd_status(a):
    P = S2Paths.load()
    rows = []
    want = {"test": ["main", "benign_long"], "calib": ["main", "benign_long"], "val": ["main", "benign_long"], "train": ["main", "main16k", "native"]}
    have, n_rows = [], 0
    for sp, subs in want.items():
        for sub in subs:
            m = P.manifests / f"{sp}.{sub}.manifest.json"
            if m.exists():
                have.append(f"{sp}/{sub}")
                n_rows += json.loads(m.read_text()).get("rows", 0)
    rows.append(("D6 synthesis", len(have) == sum(len(v) for v in want.values()), f"{len(have)}/{sum(len(v) for v in want.values())} sets, {n_rows:,} rows"))
    vh = [v for v in VARIANT_NAMES if (P.manifests / f"test.{v}.manifest.json").exists()]
    rows.append(("D7 variants", len(vh) == len(VARIANT_NAMES), f"{len(vh)}/{len(VARIANT_NAMES)}: {vh}"))
    tk, tk_fail = [], []
    for m in TOK_MODELS:
        reps = list((P.root / "tok" / m).glob("*/report.json")) if (P.root / "tok" / m).exists() else []
        if not reps:
            continue
        tk.append(m)
        for rp in reps:
            rep = json.loads(rp.read_text())
            for sub, v in rep["subsets"].items():
                if not v.get("recon_pass") or (m == "qwen3" and v.get("len_within_tol") not in (None, 1.0)):
                    tk_fail.append(f"{m}/{rep['split']}/{sub}")
    ok8 = len(tk) == len(TOK_MODELS) and not tk_fail
    rows.append(("D8 token spans", ok8, f"{len(tk)}/{len(TOK_MODELS)} tokenizers: {tk}" + (f"; CHECK FAILED {len(tk_fail)}: {tk_fail[:6]}" if tk_fail else "")))
    au = P.splits / "audit.json"
    g = json.loads(au.read_text()).get("gates_e04") if au.exists() else None
    sealed = (P.root / "SEALED").exists()
    rows.append(("E0-4 gates", bool(g and g.get("pass")), (f"{'PASS' if g['pass'] else 'FAIL'}: " + ", ".join(f"{k.split('_')[0]}={'ok' if v['pass'] else 'FAIL'}" for k, v in g['gates'].items())) if g else "run `s3ctl.py gates`"))
    rows.append(("G0 seal", sealed, "lg_data_v2 read-only" if sealed else "run `s3ctl.py gates --seal` after the gates pass"))
    rows.append(("D10 data card", (P.root / "DATACARD.md").exists(), str(P.root / "DATACARD.md")))
    pj = P.out / "reports" / "e05_pilot.json"
    pr = json.loads(pj.read_text()) if pj.exists() else None
    rows.append(("E0-5 pilot", bool(pr and pr.get("pass")), (f"{'PASS' if pr['pass'] else 'FAIL'} {list(pr['checks'])}" if pr else "run `s3ctl.py pilot`")))
    lb = P.out / "longbench"
    n_lb = sum(1 for _ in lb.glob("*/*.rows.jsonl")) if lb.exists() else 0
    rows.append(("B1/B2 longbench", False, f"{n_lb} rows files under {lb} (summaries: s3ctl.py longbench --summarize --bench <bench>)"))
    print(f"\nS3 status  ({time.strftime('%Y-%m-%d %H:%M')}, data={P.root})\n")
    w = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"  {name:{w}s}  {'DONE   ' if ok else 'PENDING'}  {detail}")
    print()
    os.system("squeue -u $USER -o '%.12i %.22j %.8T %.10M %.6D %R' 2>/dev/null | grep -E 'JOBID|s3_' || true")
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
        p.add_argument("--force", action="store_true", help="submit even if the same task is already queued")

    p = sub.add_parser("synth"); common(p); p.add_argument("--splits", default="test,calib,val,train"); p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("variants"); common(p); p.add_argument("--variant", default="all"); p.set_defaults(fn=cmd_variants)
    p = sub.add_parser("tokspans"); common(p); p.add_argument("--models", default="all"); p.add_argument("--splits", default="all"); p.set_defaults(fn=cmd_tokspans)
    p = sub.add_parser("gates"); p.add_argument("--seal", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_gates)
    p = sub.add_parser("dedupe"); p.add_argument("--split", required=True); p.add_argument("--subset", required=True); p.set_defaults(fn=cmd_dedupe)
    p = sub.add_parser("datacard"); p.set_defaults(fn=cmd_datacard)
    p = sub.add_parser("pilot"); common(p); p.add_argument("--cpu", action="store_true"); p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("finish"); common(p); p.add_argument("--cpu", action="store_true"); p.set_defaults(fn=cmd_finish)
    p = sub.add_parser("longbench"); common(p)
    p.add_argument("--guards", default=""); p.add_argument("--bench", default="v2main"); p.add_argument("--policy", default="native")
    p.add_argument("--L", default=""); p.add_argument("--depths", default=""); p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None); p.add_argument("--max-bl", type=int, default=None)
    p.add_argument("--prefill-chunk", type=int, default=None); p.add_argument("--summarize", action="store_true"); p.set_defaults(fn=cmd_longbench)
    p = sub.add_parser("b1-start"); common(p); p.set_defaults(fn=cmd_b1_start)
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
