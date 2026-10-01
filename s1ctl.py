#!/usr/bin/env python3
"""Stage S1 controller: run each S1 task locally or submit it as a Slurm job.

    python s1ctl.py assets   [--seal] [--sbatch]                # D1  asset check -> reports/assets_check_<MMDD>.md
    python s1ctl.py quota    [--apply-move] [--delete --yes]    # D2  quota + clean-up candidates (dry-run by default)
    python s1ctl.py tests    [--sbatch] [--v04] [-k EXPR]       # E0-1/2/3 + parser tests (pytest); GPU job with --sbatch
    python s1ctl.py download [--list] [--only a,b] [--models-only|--datasets-only]   # D3  LOGIN NODE ONLY
    python s1ctl.py zeroshot [--b1|--all|--guards a,b] [--bench wgtest,toxicchat] [--limit N] [--sbatch] [--dep JOBID]
    python s1ctl.py compare                                     # E0-6 roll-up -> reports/e06_reproduction.md
    python s1ctl.py m1                                          # M1 decision sheet -> reports/M1_decisions.md
    python s1ctl.py status                                      # what is done / pending, from the files on disk
    python s1ctl.py doctor                                      # pre-flight: venv packages, cached guards, free space
    python s1ctl.py locate [--write]                            # find v0.4 code + Round-3 dirs; write them into env/s1.env
    python s1ctl.py env-set KEY=VALUE ...                       # edit env/s1.env safely
    python s1ctl.py partitions                                  # partition list per GPU-memory tier (QOS-filtered)
    python s1ctl.py all --sbatch                                # assets+quota now; tests job; zeroshot array after tests

Every --sbatch writes the generated script to $LG_OUT/slurm/ and prints the job id.  --dry-run prints
the script instead of submitting it.  Slurm resources come from env/s1.env (partition, QOS, gres).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lgs1.paths import S1Paths, env, load_env_file  # noqa: E402

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition={partition}
{qos_line}{gres_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logdir}/{job}_{jobpat}.out
#SBATCH --error={logdir}/{job}_{jobpat}.err
{nodelist}{array}{dep}{signal_line}{requeue_line}{extra_lines}
set -euo pipefail
source {root}/slurm/common.sh
echo "[s1] job={job} host=$(hostname) jobid=${{SLURM_JOB_ID:-none}} task=${{SLURM_ARRAY_TASK_ID:-none}}"
set +e
{run_block}
echo "[s1] done rc=$rc"
if [[ $rc -eq 3 && "{requeue}" == "1" && -n "${{SLURM_JOB_ID:-}}" ]]; then
  echo "[s1] rc=3 = checkpoint written before the time limit -> scontrol requeue $SLURM_JOB_ID (resumes with the same job id)"
  scontrol requeue "$SLURM_JOB_ID" && sleep "${{LG_REQUEUE_SLEEP:-60}}"
fi
exit $rc
"""

# ordinary jobs: the body in a subshell (set -e inside, rc captured outside)
RUN_PLAIN = """(
set -e
{body}
)
rc=$?"""

# requeue-able jobs (S4 head/dense): the body runs in the background and the batch shell forwards SIGTERM/SIGUSR1 to
# it -- Slurm's --signal=B:TERM@N reaches the batch shell only, and a trap is the only way the python child sees it.
# `wait` returns early (>128) when the trap fires, so wait again for the child's real exit code (3 = checkpoint saved).
RUN_FORWARD = """set -m
(
trap '' TERM USR1
set -e
{body}
) &
child=$!
set +m
trap 'echo "[s1] signal received -> SIGTERM to process group $child (the trainer checkpoints and exits 3)"; kill -TERM -- -$child 2>/dev/null' TERM USR1
wait $child; rc=$?
while [[ $rc -gt 128 ]] && kill -0 $child 2>/dev/null; do wait $child; rc=$?; done"""


TESTS_STATUS_TAIL = ('rc=${PIPESTATUS[0]}\n'
                     'if [[ $rc -eq 0 ]]; then echo PASS > $LG_OUT/reports/pytest_e0.status; '
                     'else echo "FAIL rc=$rc" > $LG_OUT/reports/pytest_e0.status; fi\n'
                     'exit $rc')


def _slurm_cfg():
    load_env_file()
    return dict(partition=env("LG_PARTITION", "RTX6000ADA"), qos=env("LG_QOS", "normal"), gres=env("LG_GRES", "gpu:1"),
                cpus=env("LG_CPUS", "8"), mem=env("LG_MEM", "64G"), nodelist=env("LG_NODELIST", ""),
                # LG_EXCLUDE=n44,n45 -> #SBATCH --exclude (9/30: a node with an uncorrectable-ECC GPU killed two evals at startup)
                exclude=env("LG_EXCLUDE", ""),
                # CPU-only jobs (synthesis, token spans, prep, gates): optional CPU partitions, no GPU reservation.
                # Unset -> such jobs take the smallest GPU tier as before (QOS normal on this cluster needs --gres=gpu:1).
                cpu_partitions=env("LG_CPU_PARTITIONS", ""), cpu_qos=env("LG_CPU_QOS", ""), cpu_gres=env("LG_CPU_GRES", ""))


def partitions_for(min_gpu_gb: int, bf16: Optional[bool] = None) -> str:
    """Comma-separated partition list for this GPU-memory tier (LG_PARTITIONS_OVERRIDE wins).  `bf16` (default: every
    tier >= 24 GB, i.e. every job that runs a real model) drops GPUs without bf16 tensor cores -- TITAN RTX / RTX 2080 Ti
    (Turing) ran the 0.6B head at 2.8 s/step, ~10x an Ampere card (2026-09-19); LG_ALLOW_NO_BF16=1 re-admits them."""
    from lgs1 import slurmutil
    c = _slurm_cfg()
    if bf16 is None:
        bf16 = min_gpu_gb >= 24 and env("LG_ALLOW_NO_BF16", "") not in ("1", "true", "yes")
    chosen, _ = slurmutil.pick_partitions(min_gpu_gb, c["qos"], bf16=bf16)
    if not chosen:
        chosen = [c["partition"]]
    return ",".join(chosen)


def write_sbatch(job: str, body: str, time_limit: str, array: Optional[str] = None, dep: Optional[str] = None,
                 dry_run: bool = False, cpus: Optional[str] = None, mem: Optional[str] = None,
                 min_gpu_gb: int = 24, cpu: bool = False, signal: Optional[str] = None, requeue: bool = False,
                 exclusive: bool = False, partition: Optional[str] = None) -> Optional[str]:
    """Write + submit one sbatch script.  `cpu=True` marks a job that needs no GPU: with LG_CPU_PARTITIONS set
    (e.g. cpu-max24,cpu-max16) it goes there without --gres (QOS from LG_CPU_QOS, omitted when empty, so the
    partition default applies); otherwise it takes the `min_gpu_gb` GPU tier like before.  `exclusive` adds
    --exclusive (S7 E4 latency jobs own the node); `partition` pins one partition (a named GPU type for E4)."""
    P = S1Paths.load()
    P.ensure_out()
    c = _slurm_cfg()
    logdir = P.out / "logs"
    qos, gres = c["qos"], c["gres"]
    if cpu and c["cpu_partitions"]:
        partition, qos, gres = c["cpu_partitions"], c["cpu_qos"], c["cpu_gres"]
        print(f"[sbatch] {job}: CPU job -> partitions {partition} (LG_CPU_PARTITIONS; qos={qos or 'partition default'}, gres={gres or 'none'})")
    elif partition:
        print(f"[sbatch] {job}: partition pinned -> {partition}")
    else:
        partition = partitions_for(min_gpu_gb)
        print(f"[sbatch] {job}: partitions for >= {min_gpu_gb} GB -> {partition}")
    extra_lines = "#SBATCH --exclusive\n" if exclusive else ""
    if c.get("exclude"):
        extra_lines += f"#SBATCH --exclude={c['exclude']}\n"
        print(f"[sbatch] {job}: excluding nodes {c['exclude']} (LG_EXCLUDE)")
    if requeue:
        # the batch shell must be the one signalled (B:), and it forwards to the python child (RUN_FORWARD)
        if signal and not signal.startswith("B:"):
            signal = "B:" + signal
        run_block = RUN_FORWARD.format(body=body)
    else:
        run_block = RUN_PLAIN.format(body=body)
    script = SBATCH_TEMPLATE.format(
        job=job, partition=partition, qos_line=f"#SBATCH --qos={qos}\n" if qos else "", gres_line=f"#SBATCH --gres={gres}\n" if gres else "",
        signal_line=f"#SBATCH --signal={signal}\n" if signal else "",
        requeue_line="#SBATCH --requeue\n#SBATCH --open-mode=append\n" if requeue else "", requeue="1" if requeue else "0",
        run_block=run_block,
        cpus=cpus or c["cpus"], mem=mem or c["mem"],
        time=time_limit, logdir=logdir, root=ROOT, jobpat="%A_%a" if array else "%j",
        nodelist=f"#SBATCH --nodelist={c['nodelist']}\n" if c["nodelist"] else "",
        array=f"#SBATCH --array={array}\n" if array else "",
        dep=f"#SBATCH --dependency=afterok:{dep}\n" if dep else "",
        extra_lines=extra_lines,
        )
    path = P.out / "slurm" / f"{job}_{time.strftime('%m%d_%H%M%S')}.sbatch"
    path.write_text(script, encoding="utf-8")
    if dry_run:
        print(script)
        print(f"[dry-run] script written to {path} (not submitted)")
        return None
    out = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        raise SystemExit(f"sbatch failed for {path}")
    jid = out.stdout.strip().split()[-1]
    print(f"[sbatch] {job} -> job {jid}  ({path})")
    lj = P.out / "slurm" / "last_jobs.json"
    try:
        d = json.loads(lj.read_text()) if lj.exists() else {}
        d[job] = jid
        lj.write_text(json.dumps(d, indent=2))
    except Exception:
        pass
    return jid


def last_job(job: str) -> Optional[str]:
    P = S1Paths.load()
    lj = P.out / "slurm" / "last_jobs.json"
    if lj.exists():
        return json.loads(lj.read_text()).get(job)
    return None


def _py() -> str:
    return S1Paths.load().python


# ------------------------------------------------------------------------------------------------
# sub-commands
# ------------------------------------------------------------------------------------------------
def cmd_assets(a):
    if a.sbatch:
        body = f"${{PY}} -m lgs1.assets_check {'--seal' if a.seal else ''}"
        return write_sbatch("s1_assets", body, "00:30:00", dry_run=a.dry_run, cpus="2", mem="8G", min_gpu_gb=8)
    from lgs1 import assets_check
    return assets_check.main(["--seal"] if a.seal else [])


def cmd_quota(a):
    from lgs1 import quota
    argv = []
    if a.apply_move:
        argv.append("--apply-move")
    if a.delete:
        argv.append("--delete")
    if a.yes:
        argv.append("--yes")
    return quota.main(argv)


def cmd_tests(a):
    extra = f" -k {shlex.quote(a.k)}" if a.k else ""
    v04 = f"export LG_V04_ROOT={shlex.quote(a.v04)}\n" if a.v04 else ""
    if a.sbatch:
        body = (f"{v04}cd {ROOT}\n${{PY}} -m pytest tests/ -x -q --junitxml=$LG_OUT/reports/pytest_e0.xml{extra} "
                f"2>&1 | tee $LG_OUT/reports/pytest_e0.log\n" + TESTS_STATUS_TAIL)
        return write_sbatch("s1_tests", body, env("LG_TIME_TESTS", "01:00:00"), dry_run=a.dry_run, min_gpu_gb=8)
    if a.v04:
        os.environ["LG_V04_ROOT"] = a.v04
    P = S1Paths.load(); P.ensure_out()
    cmd = [sys.executable, "-m", "pytest", "tests/", "-x", "-q", f"--junitxml={P.out / 'reports' / 'pytest_e0.xml'}"]
    if a.k:
        cmd += ["-k", a.k]
    print("[tests]", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    (P.out / "reports" / "pytest_e0.status").write_text("PASS\n" if rc == 0 else f"FAIL rc={rc}\n")
    return rc


def cmd_download(a):
    if os.environ.get("SLURM_JOB_ID"):
        print("[download] refuses to run inside a Slurm job (compute nodes are offline). Run on the login node.", file=sys.stderr)
        return 2
    from lgs1 import download
    argv = []
    if a.list:
        argv.append("--list")
    if a.only:
        argv += ["--only", a.only]
    if a.models_only:
        argv.append("--models-only")
    if a.datasets_only:
        argv.append("--datasets-only")
    if a.no_skip:
        argv.append("--no-skip")
    if a.force_online:
        argv.append("--force-online")
    return download.main(argv)


def _guard_list(a) -> List[str]:
    from lgs1.zeroshot import guards as G
    if a.guards:
        return [g.strip() for g in a.guards.split(",") if g.strip()]
    if a.all:
        return G.B1_ORDER + G.OPTIONAL
    return G.B1_ORDER


EXPECTED_N = {"wgtest": 1725, "toxicchat": 2853}


def _redo_mismatch(a) -> Optional[tuple]:
    """Find (guard, bench) summaries whose n differs from the published benchmark size; delete them."""
    P = S1Paths.load()
    zs = P.out / "zeroshot"
    by_bench: dict = {}
    for f in sorted(zs.glob("*/*.summary.json")) if zs.exists() else []:
        try:
            sm = json.loads(f.read_text())
        except Exception:
            continue
        b, g = sm.get("bench"), sm.get("guard")
        if b in EXPECTED_N and sm.get("n") != EXPECTED_N[b]:
            by_bench.setdefault(b, []).append(g)
            for x in (f, f.with_name(f"{b}.rows.jsonl")):
                if x.exists():
                    x.unlink()
            print(f"[redo] {g}/{b}: n={sm.get('n')} != {EXPECTED_N[b]} -> deleted rows+summary")
    if not by_bench:
        print("[redo] nothing to redo (all benchmark sizes match)")
        return None
    return by_bench


_GUARD_FILE = re.compile(r"(\S*guards_\d+gb_\d+_\d+\.txt)")


def parse_queued_guards(squeue_lines: List[str], guards_of_job) -> dict:
    """{guard: [jobid_task, ...]} from `squeue -r -h -o '%i|%j|%T'` lines (arrays expanded one task per line).

    `guards_of_job(base_jobid) -> list of guard names` in array-task order (None when unknown)."""
    out: dict = {}
    for line in squeue_lines:
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 3 or not cols[1].startswith("s1_zeroshot"):
            continue
        jid, state = cols[0], cols[2]
        if state not in ("PENDING", "RUNNING", "CONFIGURING", "SUSPENDED", "REQUEUED"):
            continue
        base, _, task = jid.partition("_")
        names = guards_of_job(base) or []
        idx: List[int] = []
        if task.startswith("["):                        # unexpanded range (squeue without -r): "[7-8%1]" / "[0,2%1]"
            for part in task.strip("[]").split("%")[0].split(","):
                lo, _, hi = part.partition("-")
                if lo.isdigit():
                    idx += list(range(int(lo), int(hi) + 1 if hi.isdigit() else int(lo) + 1))
        elif task.isdigit():
            idx = [int(task)]
        elif not task and len(names) == 1:              # non-array job with one guard
            idx = [0]
        for i in idx:
            if i < len(names):
                out.setdefault(names[i], []).append(jid)
    return out


def queued_guards() -> dict:
    """Live view: which B1 guards already have a PENDING/RUNNING zeroshot task (avoids duplicate submissions)."""
    user = env("LG_USER") or os.environ.get("USER") or ""
    try:
        txt = subprocess.run(["squeue", "-r", "-h", "-u", user, "-o", "%i|%j|%T"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    cache: dict = {}

    def guards_of_job(base: str):
        if base in cache:
            return cache[base]
        names = None
        try:
            sc = subprocess.run(["scontrol", "show", "job", base, "-o"], capture_output=True, text=True, timeout=20).stdout
            cmd = re.search(r"Command=(\S+)", sc)
            if cmd and Path(cmd.group(1)).exists():
                m = _GUARD_FILE.search(Path(cmd.group(1)).read_text(encoding="utf-8"))
                if m and Path(m.group(1)).exists():
                    names = [l.strip() for l in Path(m.group(1)).read_text().splitlines() if l.strip()]
        except Exception:
            names = None
        cache[base] = names
        return names
    return parse_queued_guards(txt.splitlines(), guards_of_job)


def _missing_guards(bench_list: List[str]) -> List[str]:
    from lgs1.zeroshot import guards as G
    P = S1Paths.load()
    zs = P.out / "zeroshot"
    out = []
    for g in G.B1_ORDER:
        if not all((zs / g / f"{b}.summary.json").exists() for b in bench_list):
            out.append(g)
    return out


def cmd_zeroshot(a):
    if getattr(a, "redo_missing", False):
        bench_list = [b.strip() for b in a.bench.split(",") if b.strip()]
        miss = _missing_guards(bench_list)
        if not miss:
            print("[redo] every B1 guard has summaries for", bench_list)
            return 0
        print(f"[redo] guards without summaries: {miss}")
        a.guards, a.all = ",".join(miss), False
        return cmd_zeroshot_inner(a)
    if getattr(a, "redo_mismatch", False):
        by_bench = _redo_mismatch(a)
        if not by_bench:
            return 0
        rc = 0
        for b, gs in by_bench.items():
            a.bench, a.guards, a.all = b, ",".join(sorted(set(gs))), False
            print(f"[redo] resubmitting {a.guards} for {b}")
            rc |= int(bool(cmd_zeroshot_inner(a)))
        return rc
    return cmd_zeroshot_inner(a)


def cmd_zeroshot_inner(a):
    if getattr(a, "dep_last", False) and not a.dep:
        a.dep = last_job("s1_tests")
        print(f"[zeroshot] dependency on the last tests job: {a.dep or 'none found'}")
    guards = _guard_list(a)
    bench = a.bench
    if (a.sbatch or getattr(a, "fresh", False)) and not getattr(a, "force", False):
        # never touch or resubmit a guard that already has a PENDING/RUNNING task (it is writing its rows file)
        q = queued_guards()
        dup = [g for g in guards if g in q]
        for g in dup:
            print(f"[zeroshot] {g}: already queued/running as {','.join(q[g])} -> skipped (scancel it first, or --force)")
        guards = [g for g in guards if g not in q]
        if not guards:
            print("[zeroshot] nothing to do (every requested guard is already queued or running)")
            return []
    if getattr(a, "fresh", False):
        # set the previous rows/summary of these guards aside (never deleted) so the run starts from zero
        P = S1Paths.load(); P.ensure_out()
        stamp = time.strftime("%m%d_%H%M%S")
        for g in guards:
            for b in [x.strip() for x in bench.split(",") if x.strip()]:
                for suffix in ("rows.jsonl", "summary.json"):
                    f = P.out / "zeroshot" / g / f"{b}.{suffix}"
                    if f.exists():
                        dst = f.with_name(f"{b}.{suffix.split('.')[0]}.stale_{stamp}.{suffix.split('.')[1]}")
                        f.rename(dst)
                        print(f"[fresh] {g}/{f.name} -> {dst.name}")
    lim = f" --limit {a.limit}" if a.limit else ""
    local = ""
    if a.wgtest:
        local += f" --wgtest {shlex.quote(a.wgtest)}"
    if a.toxicchat:
        local += f" --toxicchat {shlex.quote(a.toxicchat)}"
    if a.sbatch:
        from lgs1.zeroshot import guards as G
        P = S1Paths.load(); P.ensure_out()
        # one job array per GPU-memory tier so 8B guards can use 24 GB cards while 12B/20B wait for 40/48 GB
        tiers: dict = {}
        for g in guards:
            tiers.setdefault(G.get(g).min_gpu_gb, []).append(g)
        jids = []
        for gb, names in sorted(tiers.items()):
            stamp = time.strftime('%m%d_%H%M%S')
            lst = P.out / "slurm" / f"guards_{gb}gb_{stamp}.txt"
            lst.write_text("\n".join(names) + "\n")
            body = (f"GUARD=$(sed -n \"$((SLURM_ARRAY_TASK_ID+1))p\" {lst})\n"
                    f"echo \"[s1] guard=$GUARD (tier {gb} GB)\"\n"
                    f"${{PY}} -m lgs1.zeroshot.run --guard $GUARD --bench {bench}{lim}{local}\n"
                    f"${{PY}} -m lgs1.zeroshot.compare || true")
            # no throttle by default: every task may start as soon as any listed partition has a free GPU
            # (Jieui, 9/17: the cluster has plenty of GPUs and the scheduler does the waiting) — --parallel N adds %N
            array = f"0-{len(names) - 1}" + (f"%{a.parallel}" if a.parallel and a.parallel > 0 else "")
            jid = write_sbatch(f"s1_zeroshot_{gb}gb", body, env("LG_TIME_ZEROSHOT", "12:00:00"), array=array, dep=a.dep,
                               dry_run=a.dry_run, min_gpu_gb=gb)
            print(f"         tier {gb} GB: {names}")
            jids.append(jid)
            time.sleep(1)
        return jids
    from lgs1.zeroshot import run as R
    rc = 0
    for g in guards:
        argv = ["--guard", g, "--bench", bench]
        if a.limit:
            argv += ["--limit", str(a.limit)]
        if a.wgtest:
            argv += ["--wgtest", a.wgtest]
        if a.toxicchat:
            argv += ["--toxicchat", a.toxicchat]
        if a.show_prompt:
            argv.append("--show-prompt")
        if a.online:
            argv.append("--online")
        try:
            rc |= R.main(argv)
        except Exception as e:  # keep going with the next guard; the failure is visible in status
            print(f"[zeroshot] {g} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            rc |= 1
    return rc


def cmd_compare(a):
    from lgs1.zeroshot import compare
    return compare.main([])


def cmd_m1(a):
    P = S1Paths.load(); P.ensure_out()
    src = ROOT / "docs" / "M1_decisions.md"
    dst = P.out / "reports" / "M1_decisions.md"
    if not dst.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[m1] decision sheet created: {dst}")
    else:
        print(f"[m1] decision sheet exists: {dst}")
    print(dst.read_text(encoding="utf-8"))
    return 0


def cmd_status(a):
    P = S1Paths.load(); P.ensure_out()
    rep = P.out / "reports"
    rows = []
    def add(item, ok, detail):
        rows.append((item, "DONE" if ok else "PENDING", detail))
    ac = sorted(rep.glob("assets_check_*.md"))
    add("D1 assets", bool(ac), ac[-1].name if ac else "run `s1ctl.py assets`")
    qa = sorted(rep.glob("quota_*.md"))
    add("D2 quota", bool(qa), qa[-1].name if qa else "run `s1ctl.py quota`")
    st = rep / "pytest_e0.status"
    lg = rep / "pytest_e0.log"
    if st.exists():
        add("E0-1/2/3 tests", st.read_text().startswith("PASS"), st.read_text().strip())
    elif lg.exists():
        tail = lg.read_text(errors="ignore")[-600:]
        m = re.search(r"(\d+) passed", tail)
        ok = bool(m) and "failed" not in tail and "error" not in tail.lower()
        add("E0-1/2/3 tests", ok, (f"PASS ({m.group(1)} passed, from pytest_e0.log)" if ok else "FAIL — see reports/pytest_e0.log"))
        if ok:
            st.write_text("PASS\n")
    else:
        add("E0-1/2/3 tests", False, "run `s1ctl.py tests --sbatch`")
    ds = P.sources / "download_status.json"
    from lgs1.registry import all_entries
    from lgs1.download import rebuild_status
    reg = all_entries()
    d = rebuild_status(P.sources)            # per-entry revision.json files are the truth
    if d:
        ok = [k for k, v in d.items() if v.get("status") == "ok"]
        bad = [k for k, v in d.items() if v.get("status") == "failed"]
        man = [k for k, v in d.items() if v.get("status") == "manual"]
        missing = [k for k in reg if k not in d]
        add("D3 download", not bad and not missing,
            f"ok={len(ok)} manual={len(man)} failed={len(bad)} not-yet={len(missing)} of {len(reg)}"
            + (f" -> failed {bad}" if bad else "") + (f"; next: `bash slurm/s1_download_login.sh`" if missing else ""))
    else:
        add("D3 download", False, f"0/{len(reg)} — run `bash slurm/s1_download_login.sh` on the login node")
    from lgs1.zeroshot import guards as G
    zs = P.out / "zeroshot"
    done_g = sorted(p.parent.name for p in zs.glob("*/toxicchat.summary.json")) if zs.exists() else []
    add("E0-6 zero-shot", len(set(done_g) & set(G.B1_ORDER)) == len(G.B1_ORDER),
        f"{len(set(done_g) & set(G.B1_ORDER))}/{len(G.B1_ORDER)} B1 guards have summaries: {done_g}")
    e06 = rep / "e06_reproduction.json"
    if e06.exists():
        c = json.loads(e06.read_text())
        n_pass = sum(r["status"] == "PASS" for r in c)
        n_fail = sum(str(r["status"]).startswith("FAIL") for r in c)            # includes FAIL(other mode PASS)
        n_flag = sum(r["status"] == "FAIL(other mode PASS)" for r in c)
        n_na = sum(r["status"] == "n/a" for r in c)
        n_rerun = sum(str(r["status"]).startswith("RERUN") for r in c)
        detail = f"PASS={n_pass} FAIL={n_fail} n/a={n_na} (rows={len(c)})"
        if n_flag:
            detail += f"; {n_flag} FAIL is within ±1pt in the other mode -> footnote + M1 (see reports/e06_reproduction.md)"
        if n_rerun:
            detail += f"; RERUN={n_rerun} -> zeroshot --redo-mismatch --sbatch"
        add("E0-6 ±1pt", bool(c) and n_fail == 0 and n_rerun == 0,
            detail if c else "no summaries yet — wait for the zeroshot array")
    m1 = rep / "M1_decisions.md"
    add("M1 decisions", m1.exists() and "DECIDED" in m1.read_text(), "fill reports/M1_decisions.md (mark DECIDED)" if m1.exists() else "run `s1ctl.py m1`")
    w = max(len(r[0]) for r in rows)
    print(f"\nS1 status  ({time.strftime('%Y-%m-%d %H:%M')}, out={P.out})\n")
    for item, s, detail in rows:
        print(f"  {item:{w}s}  {s:8s} {detail}")
    try:
        sq = subprocess.run(["squeue", "-u", env("LG_USER", os.environ.get("USER", "")), "-o", "%.10i %.20j %.8T %.10M %.6D %R"],
                            capture_output=True, text=True, timeout=20)
        if sq.returncode == 0:
            print("\n" + sq.stdout)
    except Exception:
        pass
    return 0


def cmd_doctor(a):
    """Pre-flight: interpreter, packages, HF cache coverage of the B1 guards, free space, host."""
    P = S1Paths.load(); P.ensure_out()
    from lgs1.zeroshot import guards as G
    from lgs1.registry import MODELS
    print(f"host={socket.gethostname()}  python={P.python}")
    chk = subprocess.run([P.python, "-c", "import torch, transformers, pytest, datasets, yaml, pandas; "
                          "print('ok torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"],
                         capture_output=True, text=True)
    print("packages:", chk.stdout.strip() if chk.returncode == 0 else f"MISSING -> {P.python} -m pip install -r requirements.txt\n{chk.stderr.strip()[-300:]}")
    # candidate interpreters that already have torch+transformers (for LG_PYTHON)
    import glob as _glob
    home = Path.home()
    cands = [P.python]
    for pat in ([str(P.v04_root / ".venv/bin/python")] if P.v04_root else []) + [
            str(home / "venvs/*/bin/python"), str(home / "miniconda3/envs/*/bin/python"), str(home / "anaconda3/envs/*/bin/python"),
            str(home / ".conda/envs/*/bin/python"), "/home3/{u}/servsec/exp/secu/bin/python".format(u=env("LG_USER", "")),
            "/home3/{u}/venvs/*/bin/python".format(u=env("LG_USER", "")), "/home3/{u}/miniconda3/envs/*/bin/python".format(u=env("LG_USER", ""))]:
        cands += _glob.glob(pat)
    seen = set()
    import shutil as _sh
    print("interpreters with torch+transformers+pytest (use one as LG_PYTHON):")
    for c in cands:
        c = _sh.which(c) or c
        if c in seen or not Path(c).exists():
            continue
        seen.add(c)
        r = subprocess.run([c, "-c", "import torch, transformers, pytest, datasets, yaml; print(torch.__version__, transformers.__version__)"],
                           capture_output=True, text=True)
        print(f"  {'OK ' if r.returncode == 0 else 'no '} {c}  {r.stdout.strip() if r.returncode == 0 else ''}")
    hub = P.hf_home / "hub"
    have = {x.name for x in hub.glob("models--*")} if hub.exists() else set()
    def cached(repo):
        return f"models--{repo.replace('/', '--')}" in have
    print(f"HF_HOME={P.hf_home} ({len(have)} models)")
    miss = []
    for name in G.B1_ORDER + G.OPTIONAL:
        spec = G.get(name)
        flag = "cached " if cached(spec.repo_id) else "MISSING"
        if flag == "MISSING":
            miss.append(name)
        print(f"  {flag} {name:26s} {spec.repo_id}")
    if miss:
        print(f"download on the login node: bash slurm/s1_download_login.sh --only {','.join(miss)}")
    for label, path in (("LG_OUT", P.out), ("LG_SOURCES", P.sources), ("HF_HOME", P.hf_home)):
        try:
            st = os.statvfs(path if path.exists() else path.parent)
            print(f"{label}={path}  free={st.f_bavail * st.f_frsize / 2**30:.0f}G (filesystem; user quota may be lower: `quota -s`)")
        except Exception as e:
            print(f"{label}={path}  ({e})")
    return 0


LOCATE_FILES = ["landmark_guard.py", "run_all.sh", "test_correctness.py", "make_recipes.py", "run_eval.py",
                "analyze.py", "make_table.py", "train_head.py", "run_evasion.py", "filter_decoys.py", "smoke_tiny.py",
                "attacks.py"]
LOCATE_DIRS = ["results_r3", "data_clean", "results_clean", "lg_mlsys_v04", "landmark_guard_mlsys"]
PRUNE = [".venv", "venv", "site-packages", "node_modules", ".cache", "__pycache__", ".git", "third_party", ".nvm",
         ".npm", ".local", "tot311", "venvs", "servprun", "hub"]


def cmd_locate(a):
    """Find the v0.4 code base and the Round-3 project by their file names (prunes venvs/caches)."""
    load_env_file()
    roots = [r for r in (a.roots.split(",") if a.roots else [f"/home/{env('LG_USER', os.environ.get('USER'))}",
                                                              f"/home3/{env('LG_USER', os.environ.get('USER'))}"]) if Path(r).exists()]
    prune = []
    for n in PRUNE:
        prune += ["-name", n, "-o"]
    prune = prune[:-1]
    names_f = []
    for n in LOCATE_FILES:
        names_f += ["-name", n, "-o"]
    names_f = names_f[:-1]
    names_d = []
    for n in LOCATE_DIRS:
        names_d += ["-name", n, "-o"]
    names_d = names_d[:-1]
    cmd = ["find"] + roots + ["(", *prune, ")", "-prune", "-o", "(", "(", "-type", "f", "(", *names_f, ")", ")", "-o",
           "(", "-type", "d", "(", *names_d, ")", ")", ")", "-print"]
    print(f"[locate] scanning {roots} (a few minutes on NFS) ...", flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    hits = sorted(set(l.strip() for l in out.stdout.splitlines() if l.strip()))
    by_dir: dict = {}
    for h in hits:
        p = Path(h)
        by_dir.setdefault(str(p.parent if p.is_file() else p), []).append(p.name)
    print(f"[locate] {len(hits)} hits")
    for d, names in sorted(by_dir.items()):
        print(f"  {d}\n      {', '.join(sorted(set(names)))}")
    # v0.4 candidates: a dir with landmark_guard.py; best has tests/test_correctness.py and run_all.sh
    v04 = []
    for h in hits:
        p = Path(h)
        if p.name == "landmark_guard.py":
            root = p.parent
            score = (1 if (root / "tests" / "test_correctness.py").exists() else 0) + (1 if (root / "run_all.sh").exists() else 0)
            v04.append((score, str(root)))
    v04.sort(reverse=True)
    r3 = sorted({str(Path(h).parent) for h in hits if Path(h).name in ("results_r3", "data_clean", "results_clean")}
                | {str(Path(h).parent) for h in hits if Path(h).name in ("run_evasion.py", "filter_decoys.py", "train_head.py")})
    print("\n[locate] v0.4 code candidates (score 2 = has tests/test_correctness.py + run_all.sh):")
    for sc, d in v04 or []:
        print(f"  score={sc}  {d}")
    if not v04:
        print("  none — landmark_guard.py not found under the roots; upload the v0.4 ZIP (v0.4.0, 09-14) and unzip it, then rerun")
    print("[locate] Round-3 candidates (dir containing results_r3/ data_clean/ results_clean/ or run_evasion.py):")
    for d in r3:
        print(f"  {d}")
    if not r3:
        print("  none — Round 3 may live under another name; search: find / -maxdepth 8 -name 'results_r3' 2>/dev/null")
    sugg = []
    if v04 and (len(v04) == 1 or v04[0][0] > v04[1][0]):
        sugg.append(f"LG_V04_ROOT={v04[0][1]}")
    if len(r3) == 1:
        sugg.append(f"LG_ROUND3_ROOT={r3[0]}")
    if sugg:
        print("\n[locate] suggested env lines: " + " ".join(sugg))
        if a.write:
            _env_set(sugg)
        else:
            print("         apply with: python s1ctl.py env-set " + " ".join(sugg) + "   (or rerun locate --write)")
    else:
        print("\n[locate] ambiguous or empty; set by hand: python s1ctl.py env-set LG_V04_ROOT=<dir> LG_ROUND3_ROOT=<dir>")
    return 0


def _env_set(pairs: List[str]) -> None:
    from lgs1.paths import ENV_FILE
    text = ENV_FILE.read_text(encoding="utf-8")
    lines = text.splitlines()
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected KEY=VALUE, got {pair}")
        key, val = pair.split("=", 1)
        pat = re.compile(rf"^(export\s+)?{re.escape(key)}=")
        done = False
        for i, line in enumerate(lines):
            if pat.match(line.strip()):
                comment = ""
                if "#" in line and not line.strip().startswith("#"):
                    comment = "   " + line[line.index("#"):].rstrip()
                lines[i] = f"export {key}={val}{comment}"
                done = True
                break
        if not done:
            lines.append(f"export {key}={val}")
        print(f"[env-set] {key}={val}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[env-set] written {ENV_FILE}")


def cmd_partitions(a):
    from lgs1 import slurmutil
    load_env_file()
    qos = env("LG_QOS", "normal")
    print(f"QOS={qos}  (override with LG_PARTITIONS_OVERRIDE=A,B,C in env/s1.env)")
    try:
        for gb, (chosen, reasons) in slurmutil.tiers(qos).items():
            print(f"\n>= {gb} GB  ->  {','.join(chosen) if chosen else '(none)'}")
            for r in reasons:
                print(f"    {r}")
    except BrokenPipeError:
        pass
    return 0


def cmd_env_set(a):
    _env_set(a.pairs)
    return 0


def cmd_all(a):
    print("== D1 assets ==")
    from lgs1 import assets_check, quota
    assets_check.main([])
    print("== D2 quota (dry-run) ==")
    quota.main([])
    print("== M1 sheet ==")
    cmd_m1(a)
    if a.sbatch:
        jid = write_sbatch("s1_tests", f"cd {ROOT}\n${{PY}} -m pytest tests/ -x -q --junitxml=$LG_OUT/reports/pytest_e0.xml 2>&1 | tee $LG_OUT/reports/pytest_e0.log\n" + TESTS_STATUS_TAIL,
                           env("LG_TIME_TESTS", "01:00:00"), dry_run=a.dry_run, min_gpu_gb=8)
        a.dep = jid
        a.guards = None; a.all = False; a.limit = None; a.wgtest = None; a.toxicchat = None; a.parallel = 0
        cmd_zeroshot(a)
        print("\nNext: run `python s1ctl.py download` on the login node (if not done) and `python s1ctl.py status`.")
    else:
        print("\nLocal mode: run `python s1ctl.py tests` and `python s1ctl.py zeroshot` on a GPU node; `download` on the login node.")
    return 0


# ------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage S1 controller", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--sbatch", action="store_true", help="submit as a Slurm job")
        p.add_argument("--dry-run", action="store_true", help="with --sbatch: print the script, do not submit")

    p = sub.add_parser("assets"); common(p); p.add_argument("--seal", action="store_true"); p.set_defaults(fn=cmd_assets)
    p = sub.add_parser("quota"); p.add_argument("--apply-move", action="store_true"); p.add_argument("--delete", action="store_true")
    p.add_argument("--yes", action="store_true"); p.set_defaults(fn=cmd_quota)
    p = sub.add_parser("tests"); common(p); p.add_argument("--v04", default=None, help="LG_V04_ROOT to wire the adapter")
    p.add_argument("-k", default=None, help="pytest -k expression"); p.set_defaults(fn=cmd_tests)
    p = sub.add_parser("download"); p.add_argument("--list", action="store_true"); p.add_argument("--only", default=None)
    p.add_argument("--models-only", action="store_true"); p.add_argument("--datasets-only", action="store_true")
    p.add_argument("--no-skip", action="store_true"); p.add_argument("--force-online", action="store_true"); p.set_defaults(fn=cmd_download)
    p = sub.add_parser("zeroshot"); common(p)
    p.add_argument("--guards", default=None, help="comma-separated guard names")
    p.add_argument("--b1", action="store_true", help="the 12 B1 guards (default)")
    p.add_argument("--all", action="store_true", help="B1 + optional guards")
    p.add_argument("--bench", default="wgtest,toxicchat")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--parallel", type=int, default=0, help="array concurrency limit (%%N); 0 = no limit, every task may run at once (default)")
    p.add_argument("--dep", default=None, help="afterok job id (e.g. the tests job)")
    p.add_argument("--dep-last", action="store_true", help="afterok on the most recently submitted tests job")
    p.add_argument("--redo-mismatch", action="store_true", help="delete results whose benchmark size is wrong and resubmit them")
    p.add_argument("--redo-missing", action="store_true", help="resubmit B1 guards that have no summary yet (after the arrays finished)")
    p.add_argument("--force", action="store_true", help="submit even if the guard already has a PENDING/RUNNING zeroshot task")
    p.add_argument("--fresh", action="store_true", help="move the guards' existing rows/summary aside first (rerun after a harness fix)")
    p.add_argument("--wgtest", default=None); p.add_argument("--toxicchat", default=None)
    p.add_argument("--show-prompt", action="store_true"); p.add_argument("--online", action="store_true")
    p.set_defaults(fn=cmd_zeroshot)
    p = sub.add_parser("compare"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("m1"); p.set_defaults(fn=cmd_m1)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("doctor"); p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("locate", help="find the v0.4 code base and the Round-3 project; --write sets env/s1.env")
    p.add_argument("--roots", default=None, help="comma-separated roots (default /home/<user>,/home3/<user>)")
    p.add_argument("--write", action="store_true"); p.set_defaults(fn=cmd_locate)
    p = sub.add_parser("partitions", help="show which partitions each GPU-memory tier will use and why"); p.set_defaults(fn=cmd_partitions)
    p = sub.add_parser("env-set", help="set KEY=VALUE lines in env/s1.env")
    p.add_argument("pairs", nargs="+"); p.set_defaults(fn=cmd_env_set)
    p = sub.add_parser("all"); common(p); p.add_argument("--bench", default="wgtest,toxicchat"); p.add_argument("--dep", default=None)
    p.add_argument("--show-prompt", action="store_true"); p.add_argument("--online", action="store_true"); p.set_defaults(fn=cmd_all)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    rc = a.fn(a)
    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
