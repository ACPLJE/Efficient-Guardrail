#!/usr/bin/env python
"""S8-S9 — ablations (E6 axes) on the LG 0.6B s0 run, Table 4, Fig 1 (plan S8: E6-1~6, S9: E6-7~12).

    python s8ctl.py arms                                  # the registry: axis, kind (infer / router / head), spec, description
    python s8ctl.py mine --sbatch                         # hard negatives from the base run (lgs4.hardneg) -> runs/v2/qwen3guard_0.6b_s0/hardneg/
    python s8ctl.py run --arms k64,kall,stride256,tp32 --sbatch          # inference-only arms: eval on the ablation subset
    python s8ctl.py run --axis E6-4 --sbatch              # router arms of one axis: router job -> eval job (afterok)
    python s8ctl.py run --arms hardneg,hardneg_ft --sbatch               # head arms: (mine ->) head -> router -> eval
    python s8ctl.py run --arms all --sbatch               # everything (32 arms; ~25 GPU jobs of 1-8 h + evals of ~6 h each)
    python s8ctl.py train --arms b64 --sbatch             # training chain only;  python s8ctl.py eval --arms b64 --sbatch   # eval only
    python s8ctl.py tables [--sbatch]                     # reports/s8/table4_ablation.md, fig10_k_sweep.json + reports/s7/fig1_pareto.md
    python s8ctl.py fig1 [--png]                          # Fig 1 alone
    python s8ctl.py status

Every arm lives in runs/v2/qwen3guard_0.6b_<arm>_s0/ (router arms: a symlinked head.pt of the base run + their own router.pt)
and writes its rows to eval/v2/e2/<method>__<run>.rows.jsonl on the ablation subset (lgs8.config: 200 pairs x 8 L x 5 grid depths 0/.2/.5/.8/1 +
benign-long; calib benign-long for the thresholds).  Head jobs checkpoint and requeue themselves (rc=3); eval jobs resume by id
(lgs1.interrupt).  Table 4 compares each arm with the baseline measured on the identical rows.
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
from lgs1.paths import env  # noqa: E402
from lgs4 import backbone as BB  # noqa: E402
from lgs4.config import S4Paths, backbone  # noqa: E402
from lgs5.config import S5Paths  # noqa: E402
from lgs8.arms import ARMS, AXES, BASE_BACKBONE, BASE_SEED, Arm, arms_of, cli_flags, resolve_head_flags  # noqa: E402
from lgs8.config import ABL_DEPTHS_STR, ABL_L, ABL_MAIN_ROWS, S8Paths, eval_args, subset_row  # noqa: E402

TOK_ENV = "export TOKENIZERS_PARALLELISM=true\nexport PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\n"
HEAD_SIGNAL = "B:TERM@1500"                 # lgs4.head_train checkpoints + rc=3 on SIGTERM (Checkpointer) -> self-requeue


def _submit(job: str, body: str, time_limit: str, a, dep: Optional[str] = None, **kw) -> Optional[str]:
    return s1ctl.write_sbatch(job, TOK_ENV + body, time_limit, dep=dep if dep is not None else getattr(a, "dep", None),
                              dry_run=getattr(a, "dry_run_sbatch", False), **kw)


def _smoke(a) -> str:
    return " --allow-unsealed" if getattr(a, "allow_unsealed", False) else ""


def _bb_seed(a):
    return getattr(a, "backbone", None) or BASE_BACKBONE, int(getattr(a, "seed", BASE_SEED) or 0)


def _select(a) -> List[Arm]:
    names = [x for x in (getattr(a, "arms", "") or "").split(",") if x]
    axis = getattr(a, "axis", None)
    if not names and not axis:
        raise SystemExit("--arms a,b | --arms all | --axis E6-4")
    return arms_of(names or None, axis)


def _link_base_head(arm: Arm, P4: S4Paths, bb: str, seed: int) -> Path:
    """Router arms route for the BASE head: the arm's run dir gets a relative symlink to runs/v2/<base>/head.pt (the router
    trainer, load_lg and the lineage fingerprint all read through it)."""
    base = P4.run_dir(bb, seed)
    if not (base / BB.HEAD_FILE).exists():
        raise SystemExit(f"{base / BB.HEAD_FILE} missing: the base run must be trained first")
    d = arm.run_dir(P4, bb, seed)
    d.mkdir(parents=True, exist_ok=True)
    link = d / BB.HEAD_FILE
    if link.is_symlink() or link.exists():
        return link
    link.symlink_to(os.path.relpath(base / BB.HEAD_FILE, d))
    return link


def mine_done(P4: S4Paths, bb: str, seed: int) -> bool:
    return (P4.run_dir(bb, seed) / "hardneg" / "meta.json").exists()


def submit_mine(a, P4: S4Paths, bb: str, seed: int) -> Optional[str]:
    b = backbone(bb)
    body = f"${{PY}} -m lgs4.hardneg --backbone {bb} --seed {seed}"
    return _submit(f"s8_mine_{bb}_s{seed}", body, env("LG_TIME_S8_MINE", "06:00:00"), a, min_gpu_gb=max(24, b.min_gpu_gb))


def submit_train(a, arm: Arm, P4: S4Paths, bb: str, seed: int, dep: Optional[str] = None, mine_job: Optional[str] = None) -> Dict[str, Optional[str]]:
    """The training chain of one arm -> {'head': jid, 'router': jid} (absent stages omitted).  Complete stages are skipped by
    the trainers themselves (head.pt / router.pt with meta.complete), so re-running is idempotent."""
    out: Dict[str, Optional[str]] = {}
    b = backbone(bb)
    gb = max(arm.gpu_gb, b.min_gpu_gb)
    last = dep
    if arm.trains_head:
        flags = resolve_head_flags(arm, P4, bb, seed)
        needs_mine = any(str(v).endswith("/hardneg") for v in flags.values())
        if needs_mine and mine_job:
            last = mine_job
        body = f"${{PY}} -m lgs4.head_train --backbone {bb} --seed {seed} --variant {arm.name}{cli_flags(flags)}{_smoke(a)}"
        out["head"] = _submit(f"s8_head_{arm.name}", body, env("LG_TIME_S8_HEAD", "24:00:00"), a, dep=last, min_gpu_gb=gb,
                              signal=HEAD_SIGNAL, requeue=True)
        last = out["head"]
    elif arm.trains_router:
        if not getattr(a, "dry_run_sbatch", False):
            _link_base_head(arm, P4, bb, seed)
    if arm.trains_router:
        body = f"${{PY}} -m lgs4.router_train --backbone {bb} --seed {seed} --variant {arm.name}{cli_flags(arm.router)}{_smoke(a)}"
        out["router"] = _submit(f"s8_router_{arm.name}", body, env("LG_TIME_S8_ROUTER", "12:00:00"), a, dep=last, min_gpu_gb=gb)
    return out


_QUEUED: Optional[set] = None


def _queued() -> set:
    global _QUEUED
    if _QUEUED is None:
        _QUEUED = s5ctl._queued_names() or set()
    return _QUEUED


def submit_eval(a, arm: Arm, bb: str, seed: int, dep: Optional[str] = None) -> Optional[str]:
    """One e2 job on the ablation subset (resumes the arm's rows file by id).  An eval job of the same arm that is still
    queued / running is not duplicated (two writers on one rows file leave duplicate ids) unless --force."""
    b = backbone(bb)
    spec = arm.spec_str(bb, seed)
    job = f"s8_eval_{arm.name}"
    if job in _queued() and not getattr(a, "force", False) and not getattr(a, "dry_run_sbatch", False):
        print(f"[eval] {job}: already queued/running -> skipped (--force to submit anyway)")
        return None
    body = f"${{PY}} -m lgs5.run --exp e2 --model {spec} {eval_args()}"
    return _submit(job, body, env("LG_TIME_S8_EVAL", "24:00:00"), a, dep=dep, min_gpu_gb=max(arm.gpu_gb, b.min_gpu_gb),
                   signal=s5ctl.EVAL_SIGNAL, requeue=True)


# ------------------------------------------------------------------------------------------------
def cmd_arms(a):
    bb, seed = _bb_seed(a)
    arms = arms_of(None, getattr(a, "axis", None))
    print(f"{'arm':12s} {'axis':18s} {'kind':7s} {'spec':44s} description")
    for arm in arms:
        print(f"{arm.name:12s} {arm.axis:18s} {arm.kind:7s} {arm.spec_str(bb, seed):44s} {arm.desc}")
    print(f"\n{len(arms)} arms; axes: {AXES}")
    return 0


def cmd_mine(a):
    bb, seed = _bb_seed(a)
    P4 = S4Paths.load().ensure().for_backbone(bb)
    if a.sbatch:
        jid = submit_mine(a, P4, bb, seed)
        print(f"[mine] job {jid}")
        return 0
    return s5ctl._run_local("lgs4.hardneg", ["--backbone", bb, "--seed", str(seed)])


def _run(a, do_train: bool, do_eval: bool) -> Dict[str, Dict]:
    bb, seed = _bb_seed(a)
    P4 = S4Paths.load().ensure().for_backbone(bb)
    arms = _select(a)
    out: Dict[str, Dict] = {}
    mine_job = None
    if do_train and any(arm.trains_head and any(str(v) == "@hardneg" for v in arm.head.values()) for arm in arms):
        if not mine_done(P4, bb, seed) and not getattr(a, "no_mine", False):
            mine_job = submit_mine(a, P4, bb, seed) if a.sbatch else None
            if a.sbatch:
                print(f"[run] hard negatives not mined yet -> mine job {mine_job} (the hardneg heads wait for it)")
    chains: Dict[str, Optional[str]] = {}                   # arm -> last training job id (for infer arms hosted on it)
    for arm in arms:
        rec: Dict = {"kind": arm.kind, "spec": arm.spec_str(bb, seed)}
        last = None
        if arm.kind == "infer" and arm.on:
            host_dir = arm.run_dir(P4, bb, seed)
            if arm.on in chains:
                last = chains[arm.on]                          # evaluated after the host arm's chain
            elif not (host_dir / BB.ROUTER_FILE).exists() and not getattr(a, "dry_run_sbatch", False):
                rec["eval"] = f"skipped: {host_dir.name} has no router.pt yet (run --arms {arm.on} first)"
                out[arm.name] = rec
                continue
        if do_train and arm.kind != "infer":
            if a.sbatch:
                tr = submit_train(a, arm, P4, bb, seed, mine_job=mine_job)
                rec.update(tr)
                last = tr.get("router") or tr.get("head")
                chains[arm.name] = last
            else:
                flags = resolve_head_flags(arm, P4, bb, seed)
                if arm.trains_head:
                    rc = s5ctl._run_local("lgs4.head_train", ["--backbone", bb, "--seed", str(seed), "--variant", arm.name] + cli_flags(flags).split() + _smoke(a).split())
                    rec["head"] = "ok" if rc == 0 else f"rc={rc}"
                else:
                    _link_base_head(arm, P4, bb, seed)
                rc = s5ctl._run_local("lgs4.router_train", ["--backbone", bb, "--seed", str(seed), "--variant", arm.name] + cli_flags(arm.router).split() + _smoke(a).split())
                rec["router"] = "ok" if rc == 0 else f"rc={rc}"
        if do_eval:
            if a.sbatch:
                rec["eval"] = submit_eval(a, arm, bb, seed, dep=last)
            else:
                rc = s5ctl._run_local("lgs5.run", ["--exp", "e2", "--model", arm.spec_str(bb, seed)] + eval_args().split())
                rec["eval"] = "ok" if rc == 0 else f"rc={rc}"
        out[arm.name] = rec
    return out


def cmd_train(a):
    out = _run(a, True, False)
    print(f"[train] {json.dumps(out, indent=1)}")
    return 0


def cmd_eval(a):
    out = _run(a, False, True)
    print(f"[eval] {json.dumps(out, indent=1)}")
    return 0


def cmd_run(a):
    out = _run(a, True, True)
    print(f"[run] {json.dumps(out, indent=1)}")
    print("[run] when the evals finish: python s8ctl.py tables")
    return 0


def cmd_tables(a):
    bb, seed = _bb_seed(a)
    args = ["--backbone", bb, "--seed", str(seed)] + (["--arms", a.arms] if getattr(a, "arms", None) else [])
    if getattr(a, "sbatch", False):
        body = f"${{PY}} -m lgs8.table4 {' '.join(args)}\n${{PY}} -m lgs8.fig1 || true"
        jid = s1ctl.write_sbatch("s8_tables", TOK_ENV + body, env("LG_TIME_S8_TABLES", "04:00:00"), dry_run=getattr(a, "dry_run_sbatch", False),
                                 min_gpu_gb=8, cpu=True)
        print(f"[tables] job {jid}: reports/s8/table4_ablation.md, fig10_k_sweep.json, reports/s7/fig1_pareto.md (log: logs/s8_tables_{jid}.out)")
        return 0
    rc = s5ctl._run_local("lgs8.table4", args)
    rc2 = s5ctl._run_local("lgs8.fig1", [])
    return rc or (0 if rc2 in (0, 1) else rc2)


def cmd_table(a):
    """Print the Table 4 rows of a few arms (their +rg3 / +k8 derivations included) with the baseline and seed rows, from the
    last `tables` run's JSON -- `python s8ctl.py table --arms ep1,ep2` instead of cat-ing the whole markdown."""
    from lgs5.tables import _f
    P8 = S8Paths.load()
    js = P8.reports / "table4_ablation.json"
    if not js.exists():
        raise SystemExit(f"{js} missing: python s8ctl.py tables --sbatch first")
    res = json.loads(js.read_text(encoding="utf-8"))
    names = [x for x in (a.arms or "").split(",") if x]
    axis = getattr(a, "axis", None)
    base = [] if getattr(a, "no_base", False) else [r for r in res["rows"] if r["arm"] in ("base", "s1", "s2")]
    if axis and any(str(r["axis"]).split(" ", 1)[0] == axis for r in res["rows"]):
        axis_ok = lambda ax: str(ax).split(" ", 1)[0] == axis         # noqa: E731  exact code (E6-1 is not E6-12)
    else:
        axis_ok = lambda ax: str(ax).startswith(axis or "")            # noqa: E731  prefix, like arms_of
    pick = [r for r in res["rows"] if r["arm"] not in ("base", "s1", "s2")
            and (not names or r["arm"].split("+", 1)[0] in names or r["arm"] in names)
            and (not axis or axis_ok(r["axis"]))]
    if not names and not axis:
        pick = []
    cols = [("R@1k", 3), ("R@28k", 3), ("R@128k", 3), ("d_R@128k", "d"), ("pos_min@128k", 3), ("d_pos_min@128k", "d"), ("fpr@tau1k_bl@128k", 3),
            ("tau@128k", 4), ("hit@k@128k", 3), ("twin_fpr@28k", 3), ("eps_hat", 3), ("n_rows", "i")]
    head = ["arm", "R@1k", "R@28k", "R@128k", "dR", "pos-min", "d", "infl", "tau", "hit@k", "twinFPR", "eps", "n", "status"]
    print(f"Table 4 ({time.strftime('%Y-%m-%d %H:%M', time.localtime(js.stat().st_mtime))}, {res.get('base_run')}; d = arm - baseline on the same rows)")
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for r in base + pick:
        cells = []
        for k, fmt in cols:
            v = r.get(k)
            if v is None:
                cells.append("")
            elif fmt == "d":
                cells.append(f"{v:+.3f}")
            elif fmt == "i":
                cells.append(f"{v:,}")
            else:
                cells.append(_f(float(v), fmt))
        st = r.get("status", "")
        print(f"| {r['arm']} | " + " | ".join(cells) + f" | {'' if st == 'ok' else st} |")
    if names:
        missing = [n for n in names if not any(r["arm"].split("+", 1)[0] == n for r in pick)]
        if missing:
            print(f"(no rows in the last tables run for: {missing})")
    elif not axis:
        print("(baseline rows only: add --arms ep1,ep2 or --axis E6-10; the arm names: python s8ctl.py arms)")
    return 0


def cmd_fig1(a):
    return s5ctl._run_local("lgs8.fig1", (["--png"] if getattr(a, "png", False) else []) + (["--gpu", a.gpu] if getattr(a, "gpu", None) else []))


def _count_subset(path: Path) -> int:
    from lgs1.lineage import iter_rows
    try:
        return sum(1 for r in iter_rows(path, tolerant=True) if subset_row(r))
    except Exception:                                                # noqa: BLE001
        return 0


def _newest_log(P5: S5Paths, prefix: str) -> Optional[Path]:
    # <prefix>_<jobid>.out only: `s8_eval_b256_*` must not pick up `s8_eval_b256_tp64_<jid>.out` (9/30: b256 shown "queued")
    logs = glob.glob(str(P5.out / "logs" / f"{prefix}_[0-9]*.out"))
    return Path(max(logs, key=os.path.getmtime)) if logs else None


def cmd_status(a):
    bb, seed = _bb_seed(a)
    P4 = S4Paths.load().for_backbone(bb)
    P5 = S5Paths.load()
    print(f"\nS8 status  ({time.strftime('%Y-%m-%d %H:%M')}, base {bb}_s{seed}, subset rows target = {ABL_MAIN_ROWS:,} test/main "
          f"(depths {ABL_DEPTHS_STR}) + 4,000 test bl + 4,000 calib bl)\n")
    print(f"  hard negatives: {'mined' if mine_done(P4, bb, seed) else 'not mined (python s8ctl.py mine --sbatch)'}")
    queued = s5ctl._queued_names()
    base_file = P5.eval / "e2" / f"lg_k32__{bb}_s{seed}.rows.jsonl"
    print(f"  baseline rows on the subset: {_count_subset(base_file):,}" if base_file.exists() else "  baseline rows: missing")
    print(f"\n  {'arm':12s} {'axis':18s} {'kind':7s} {'head':8s} {'router':8s} {'rows':>8s}  eval")
    for arm in arms_of(None, getattr(a, "axis", None)):
        d = arm.run_dir(P4, bb, seed)
        if arm.kind == "infer":
            head = router = "base"
        else:
            hp = d / BB.HEAD_FILE
            # head.pt is written only when training completes (the mid-run state is head.resume.pt), so existence = done
            head = "link" if hp.is_symlink() else ("done" if hp.exists() else ("part" if (d / ("head" + BB.RESUME_SUFFIX)).exists() else "-"))
            rp = d / BB.ROUTER_FILE
            router = "done" if rp.exists() else "-"
        rows = P5.rows_path("e2", arm.spec(bb, seed))
        n = _count_subset(rows) if rows.exists() else 0
        log = _newest_log(P5, f"s8_eval_{arm.name}")
        st = s5ctl.progress_from_log(log) if log else ""
        if st.startswith("running") and queued is not None and f"s8_eval_{arm.name}" not in queued:
            st = f"STOPPED (no job in queue) -> {log.with_suffix('.err').name}"
        elif st == "queued" and queued is not None and f"s8_eval_{arm.name}" not in queued:
            # the log exists (the job started) but no rows were written and no job is left: it died at startup
            st = f"FAILED before the first rows (no job in queue) -> {log.with_suffix('.err').name}"
        print(f"  {arm.name:12s} {arm.axis:18s} {arm.kind:7s} {head:8s} {router:8s} {n:>8,}  {st}")
    reps = sorted(glob.glob(str(P5.out / "reports" / "s8" / "*.md")))
    print(f"\n  reports/s8: {[Path(r).name for r in reps]}")
    os.system("squeue -u $USER -o '%.12i %.40j %.8T %.10M %.6D %R' 2>/dev/null | grep -E 'JOBID|s8_' || true")
    return 0


# ------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, arms=True):
        p.add_argument("--sbatch", action="store_true")
        p.add_argument("--dep", default=None)
        p.add_argument("--dry-run-sbatch", action="store_true")
        p.add_argument("--allow-unsealed", action="store_true")
        p.add_argument("--backbone", default=BASE_BACKBONE)
        p.add_argument("--seed", type=int, default=BASE_SEED)
        if arms:
            p.add_argument("--arms", default="", help="comma list of arm names, or all")
            p.add_argument("--axis", default=None, help="every arm of an axis (prefix match, e.g. E6-4)")

    p = sub.add_parser("arms"); p.add_argument("--axis", default=None); p.add_argument("--backbone", default=BASE_BACKBONE)
    p.add_argument("--seed", type=int, default=BASE_SEED); p.set_defaults(fn=cmd_arms)
    p = sub.add_parser("mine"); common(p, arms=False); p.set_defaults(fn=cmd_mine)
    for name, fn in (("train", cmd_train), ("eval", cmd_eval), ("run", cmd_run)):
        p = sub.add_parser(name); common(p); p.add_argument("--no-mine", action="store_true", help="do not submit the mining job automatically")
        p.add_argument("--force", action="store_true", help="submit an eval job even if one of the same arm is queued / running")
        p.set_defaults(fn=fn)
    p = sub.add_parser("tables"); p.add_argument("--sbatch", action="store_true"); p.add_argument("--dry-run-sbatch", action="store_true")
    p.add_argument("--arms", default=None); p.add_argument("--backbone", default=BASE_BACKBONE); p.add_argument("--seed", type=int, default=BASE_SEED)
    p.set_defaults(fn=cmd_tables)
    p = sub.add_parser("table", help="print a few arms' Table 4 rows from the last tables run"); p.add_argument("--arms", default="")
    p.add_argument("--axis", default=None); p.add_argument("--no-base", action="store_true"); p.set_defaults(fn=cmd_table)
    p = sub.add_parser("fig1"); p.add_argument("--png", action="store_true"); p.add_argument("--gpu", default=None); p.set_defaults(fn=cmd_fig1)
    p = sub.add_parser("status"); p.add_argument("--axis", default=None); p.add_argument("--backbone", default=BASE_BACKBONE)
    p.add_argument("--seed", type=int, default=BASE_SEED); p.set_defaults(fn=cmd_status)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
