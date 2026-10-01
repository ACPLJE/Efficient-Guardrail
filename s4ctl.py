#!/usr/bin/env python3
"""Stage S4 controller — LG head + router (T1 0.6B x 3 seeds, T3 4B), dense-FT (T2), E0-7 FLOPs, checkpoint checks.

    python s4ctl.py prep --sbatch                               # tokenise v2 train/val/calib rows once (CPU tier)
    python s4ctl.py head   --backbone qwen3guard_0.6b --seed 0 --sbatch
    python s4ctl.py router --backbone qwen3guard_0.6b --seed 0 --sbatch [--dep <head jobid>]
    python s4ctl.py dense  --backbone qwen3guard_0.6b --seed 0 --sbatch [--dep <head jobid>]
    python s4ctl.py valcheck --backbone qwen3guard_0.6b --seed 0 --sbatch
    python s4ctl.py flops --sbatch                              # E0-7 on the base 0.6B (weights do not change FLOPs)
    python s4ctl.py t1 --sbatch                                 # 0.6B seeds 0/1/2: head -> router -> valcheck chains
    python s4ctl.py t2 --sbatch                                 # dense-FT 0.6B seeds 0/1/2 (each after its head job)
    python s4ctl.py t3 --sbatch                                 # LG 4B s0: head -> router -> valcheck
    python s4ctl.py all --sbatch                                # prep -> (t1, t2, t3, e07)
    python s4ctl.py status

Training refuses to start unless lg_data_v2/SEALED exists (G0).  Jobs: --gres=gpu:1, no array throttle; tiers from
the backbone (0.6B: 24 GB, 4B/8B: 48 GB).  `--local` runs in this shell under $LG_PYTHON (smoke tests: add --smoke).
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
from lgs4.config import BACKBONES, S4Paths, backbone  # noqa: E402

TOK_ENV = "export TOKENIZERS_PARALLELISM=true\n"
SEEDS_06B = [0, 1, 2]


def _submit(job: str, body: str, time_limit: str, a, dep: Optional[str] = None, **kw) -> Optional[str]:
    return s1ctl.write_sbatch(job, TOK_ENV + body, time_limit, dep=dep if dep is not None else getattr(a, "dep", None),
                              dry_run=getattr(a, "dry_run_sbatch", False), **kw)


def _run_local(module: str, args: List[str]) -> int:
    py = S4Paths.load().data.python
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
        print(f"[s4ctl] {module}: {e.code}", file=sys.stderr)
        return 1
    return 0


def _check_sealed(a) -> None:
    P = S4Paths.load()
    if not P.sealed() and not getattr(a, "allow_unsealed", False):
        raise SystemExit(f"[s4ctl] {P.data.root}/SEALED missing: run `python s3ctl.py gates --seal` first (G0). "
                         f"Smoke runs only: --allow-unsealed")


def _smoke_args(a) -> str:
    return " --allow-unsealed" if getattr(a, "allow_unsealed", False) else ""


# ------------------------------------------------------------------------------------------------
def cmd_prep(a):
    tok = getattr(a, "tok", "") or "qwen3"
    extra = f" --tok {tok}" if tok != "qwen3" else ""
    if a.sbatch:
        return _submit("s4_prep" + (f"_{tok}" if extra else ""), "${PY} -m lgs4.prep" + extra, env("LG_TIME_S4_PREP", "06:00:00"), a,
                       min_gpu_gb=8, cpu=True)
    return _run_local("lgs4.prep", extra.split())


def _variant_args(a, stage: str) -> str:
    """--variant <v> for every stage of a variant chain (+ --probe for the head); '' for the default run."""
    v = getattr(a, "variant", "") or ""
    out = f" --variant {v}" if v else ""
    if stage == "head" and getattr(a, "probe", None):
        out += f" --probe {a.probe}"
    return out


def _run_name(bb: str, seed: int, a=None) -> str:
    v = (getattr(a, "variant", "") or "") if a is not None else ""
    return f"{bb}{('_' + v) if v else ''}_s{seed}"


def _stage_job(a, stage: str, module: str, bb: str, seed: int, time_key: str, default_time: str, dep: Optional[str] = None,
               extra: str = "") -> Optional[str]:
    b = backbone(bb)
    body = f"${{PY}} -m {module} --backbone {bb} --seed {seed}{extra}{_variant_args(a, stage)}{_smoke_args(a)}"
    # head/dense: SIGTERM 25 min before the limit -> checkpoint + rc=3 -> the script requeues ITSELF (same job id, so
    # router/valcheck/dense dependants keep waiting) and the requeued run resumes from the checkpoint
    long_stage = stage in ("head", "dense")
    return _submit(f"s4_{stage}_{_run_name(bb, seed, a)}", body, env(time_key, default_time), a, dep=dep, min_gpu_gb=b.min_gpu_gb,
                   signal="TERM@1500" if long_stage else None, requeue=long_stage)


def cmd_head(a):
    _check_sealed(a)
    extra = f" --max-steps {a.max_steps}" if a.max_steps else ""
    extra += f" --limit-rows {a.limit_rows}" if a.limit_rows else ""
    if a.sbatch:
        return _stage_job(a, "head", "lgs4.head_train", a.backbone, a.seed, "LG_TIME_S4_HEAD", "24:00:00", extra=extra)
    return _run_local("lgs4.head_train", ["--backbone", a.backbone, "--seed", str(a.seed)] + extra.split() + _variant_args(a, "head").split()
                      + _smoke_args(a).split())


def cmd_router(a):
    _check_sealed(a)
    extra = f" --limit-rows {a.limit_rows}" if a.limit_rows else ""
    if a.sbatch:
        return _stage_job(a, "router", "lgs4.router_train", a.backbone, a.seed, "LG_TIME_S4_ROUTER", "12:00:00", extra=extra)
    return _run_local("lgs4.router_train", ["--backbone", a.backbone, "--seed", str(a.seed)] + extra.split() + _variant_args(a, "router").split()
                      + _smoke_args(a).split())


def cmd_dense(a):
    _check_sealed(a)
    extra = f" --steps {a.steps}" if a.steps else ""
    extra += f" --limit-rows {a.limit_rows}" if a.limit_rows else ""
    if a.sbatch:
        return _stage_job(a, "dense", "lgs4.dense_train", a.backbone, a.seed, "LG_TIME_S4_DENSE", "48:00:00", extra=extra)
    return _run_local("lgs4.dense_train", ["--backbone", a.backbone, "--seed", str(a.seed)] + extra.split() + _variant_args(a, "dense").split()
                      + _smoke_args(a).split())


def cmd_valcheck(a):
    if a.sbatch:
        return _stage_job(a, "valcheck", "lgs4.valcheck", a.backbone, a.seed, "LG_TIME_S4_VAL", "06:00:00")
    return _run_local("lgs4.valcheck", ["--backbone", a.backbone, "--seed", str(a.seed)] + _variant_args(a, "valcheck").split())


def cmd_flops(a):
    b = backbone(a.backbone)
    body = f"${{PY}} -m lgs4.flops --backbone {a.backbone} --L {a.L} --k {a.k}" + (f" --run {a.run}" if a.run else "")
    if a.sbatch:
        return _submit("s4_e07_flops", body, env("LG_TIME_S4_FLOPS", "02:00:00"), a, min_gpu_gb=max(24, b.min_gpu_gb))
    return _run_local("lgs4.flops", ["--backbone", a.backbone, "--L", a.L, "--k", str(a.k)] + (["--run", a.run] if a.run else []))


def _chain(a, bb: str, seed: int, dep: Optional[str], start: str = "head") -> Dict[str, Optional[str]]:
    """head -> router -> valcheck for one (backbone, seed); `start` skips the finished front of the chain
    (`--from router` after a router job died: no head job is queued just to print 'already complete')."""
    order = ["head", "router", "valcheck"]
    if start not in order:
        raise SystemExit(f"--from must be one of {order}")
    out: Dict[str, Optional[str]] = {"head": None, "router": None, "valcheck": None}
    prev = dep
    for stage in order[order.index(start):]:
        if stage == "head":
            prev = _stage_job(a, "head", "lgs4.head_train", bb, seed, "LG_TIME_S4_HEAD", "24:00:00", dep=prev)
        elif stage == "router":
            prev = _stage_job(a, "router", "lgs4.router_train", bb, seed, "LG_TIME_S4_ROUTER", "12:00:00", dep=prev)
        else:
            prev = _stage_job(a, "valcheck", "lgs4.valcheck", bb, seed, "LG_TIME_S4_VAL", "06:00:00", dep=prev)
        out[stage] = prev
    return out


def _seeds(a) -> List[int]:
    """--seeds 0,2 restricts t1/t2 to those seeds (resubmitting one seed's chain after a failed or slow job)."""
    v = getattr(a, "seeds", None)
    return [int(x) for x in v.split(",") if x] if v else list(SEEDS_06B)


def cmd_t1(a):
    _check_sealed(a)
    if not a.sbatch:
        raise SystemExit("t1 is an sbatch chain (use head/router/valcheck --local for single runs)")
    out = {s: _chain(a, "qwen3guard_0.6b", s, a.dep, start=getattr(a, "from_stage", "head")) for s in _seeds(a)}
    print(f"[t1] {out}")
    return ":".join(j["valcheck"] for j in out.values() if j["valcheck"])


def cmd_t2(a):
    _check_sealed(a)
    if not a.sbatch:
        raise SystemExit("t2 is an sbatch chain")
    P = S4Paths.load()
    out = {}
    for s in _seeds(a):
        dep = a.dep
        if dep is None and not (P.run_dir("qwen3guard_0.6b", s) / "head.pt").exists():
            hj = _head_job_id("qwen3guard_0.6b", s)
            dep = hj
            if hj is None:
                print(f"[t2] s{s}: no head.pt and no queued head job -> dense needs --steps; submit `python s4ctl.py t1 --sbatch` first")
                continue
        out[s] = _stage_job(a, "dense", "lgs4.dense_train", "qwen3guard_0.6b", s, "LG_TIME_S4_DENSE", "48:00:00", dep=dep)
    print(f"[t2] {out}")
    return ":".join(j for j in out.values() if j)


def cmd_t3(a):
    _check_sealed(a)
    if not a.sbatch:
        raise SystemExit("t3 is an sbatch chain")
    out = _chain(a, "qwen3guard_4b", 0, a.dep, start=getattr(a, "from_stage", "head"))
    print(f"[t3] {out}")
    return out["valcheck"]


def cmd_all(a):
    _check_sealed(a)
    if not a.sbatch:
        raise SystemExit("all is an sbatch chain")
    prep = cmd_prep(a)
    a.dep = prep
    t1 = {s: _chain(a, "qwen3guard_0.6b", s, prep) for s in SEEDS_06B}
    t2 = {s: _stage_job(a, "dense", "lgs4.dense_train", "qwen3guard_0.6b", s, "LG_TIME_S4_DENSE", "48:00:00", dep=t1[s]["head"]) for s in SEEDS_06B}
    t3 = _chain(a, "qwen3guard_4b", 0, prep)
    a.dep = prep
    e07 = cmd_flops(a)
    print(f"[all] prep={prep} t1={t1} t2={t2} t3={t3} e07={e07}")
    return prep


def _head_job_id(bb: str, seed: int) -> Optional[str]:
    """Queued/running s4_head job for this backbone/seed (dependency target for dense)."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"], capture_output=True, text=True).stdout
    except Exception:
        return None
    for line in out.splitlines():
        jid, name = (line.split() + ["", ""])[:2]
        if name == f"s4_head_{bb}_s{seed}":
            return jid
    return None


# ------------------------------------------------------------------------------------------------
def _progress(run_dir: Path, key: str) -> str:
    """In-progress stage from its jsonl log: last `step`/`eta_h` line (head/dense) or targets parts (router)."""
    log = run_dir / f"{key}.log.jsonl"
    if key == "router":
        parts = sorted((run_dir / "router_targets").glob("*.part*.npy")) if (run_dir / "router_targets").exists() else []
        if parts and not log.exists():
            return f"running (targets: {len(parts)} parts cached)"
    if not log.exists():
        return "-"
    last = None
    try:
        with open(log, encoding="utf-8") as f:
            for line in f:
                if '"step"' in line:
                    last = json.loads(line)
    except Exception:
        return "-"
    if not last:
        return "-"
    if last.get("done"):
        return "finishing"
    eta = f" eta {last['eta_h']} h" if last.get("eta_h") is not None else ""
    return f"running {last.get('step')}{'/' + str(last['total']) if last.get('total') else ''}{eta}"


def cmd_status(a):
    P = S4Paths.load()
    print(f"\nS4 status  ({time.strftime('%Y-%m-%d %H:%M')}, runs={P.runs}, sealed={'yes' if P.sealed() else 'NO'})\n")
    prepped = [f"{s}/{u}" for s in ("train", "val", "calib") for u in ("main", "main16k", "native", "benign_long")
               if (P.prep / s / u / "ids.npy").exists()]
    print(f"  prep      {len(prepped)} subsets: {prepped}")
    rows = []
    run_dirs = [P.run_dir(bb, seed) for bb in BACKBONES for seed in (SEEDS_06B if bb == "qwen3guard_0.6b" else [0])]
    if P.runs.exists():                                    # variant runs (runs/<bb>_<variant>_s<seed>) next to the defaults
        run_dirs += sorted(d for d in P.runs.iterdir() if d.is_dir() and re.fullmatch(r".+_s\d+", d.name) and d not in run_dirs)
    for d in run_dirs:
        if True:
            if not d.exists():
                continue
            rec = {"run": d.name}
            for f, key in (("head.pt", "head"), ("router.pt", "router"), ("dense_ft.pt", "dense")):
                p = d / f
                if p.exists():
                    try:
                        import torch
                        m = torch.load(p, map_location="cpu", weights_only=False).get("meta", {})
                        rec[key] = f"{m.get('steps')} steps{'' if m.get('complete') else ' (incomplete)'}"
                        if key == "router" and m.get("val_hit"):
                            rec[key] += f" hit@8={m['val_hit'].get('all', {}).get('hit@8', float('nan')):.3f}"
                        if key == "head" and m.get("val"):
                            rec[key] += f" auprc={m['val'].get('auprc_needle_vs_twin')}"
                    except Exception as e:
                        rec[key] = f"unreadable ({e})"
                else:
                    rec[key] = _progress(d, key)
            vr = d / "val_report.json"
            if vr.exists():
                v = json.loads(vr.read_text())
                rec["valcheck"] = f"hit@8@28k={v.get('hit8_at_28k')} {'PASS' if v.get('pass') else 'FAIL'}"
                if v.get("matched"):
                    rec["valcheck"] += f" recall@5pct={v['matched'].get('recall@5pct'):.3f}"
            else:
                rec["valcheck"] = "-"
            rows.append(rec)
    for r in rows:
        print(f"  {r['run']:22s} head: {r['head']:40s} router: {r['router']:32s} dense: {r['dense']:20s} val: {r['valcheck']}")
    if not rows:
        print("  (no runs yet)")
    e07 = P.reports / "e07_flops.json"
    if e07.exists():
        rep = json.loads(e07.read_text())
        print(f"  E0-7 FLOPs  {'PASS' if rep['pass'] else 'FAIL'}  " + ", ".join(f"L={r['L']}: {r['ratio_total']:.3f}" for r in rep["rows"]))
    else:
        print("  E0-7 FLOPs  PENDING  run `s4ctl.py flops --sbatch`")
    print()
    os.system("squeue -u $USER -o '%.10i %.34j %.8T %.10M %.6D %R' 2>/dev/null | grep -E 'JOBID|s4_' || true")
    return 0


# ------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, bb=True):
        p.add_argument("--sbatch", action="store_true")
        p.add_argument("--local", action="store_true")
        p.add_argument("--dep", default=None, help="afterok dependency job id(s), colon-separated")
        p.add_argument("--dry-run-sbatch", action="store_true")
        p.add_argument("--allow-unsealed", action="store_true", help="smoke runs before G0 only")
        p.add_argument("--variant", default="", help="run-dir suffix for an ablation chain: runs/<bb>_<variant>_s<seed> (e.g. lm)")
        p.add_argument("--probe", default=None, choices=[None, "last", "landmark"], help="head probe for --variant chains (head v2 = landmark)")
        if bb:
            p.add_argument("--backbone", default="qwen3guard_0.6b", choices=list(BACKBONES))
            p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prep"); common(p, bb=False); p.add_argument("--tok", default="qwen3", help="tokenizer family (llama3 for the LG3-8B backbone, T5)"); p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("head"); common(p); p.add_argument("--max-steps", type=int, default=None); p.add_argument("--limit-rows", type=int, default=None); p.set_defaults(fn=cmd_head)
    p = sub.add_parser("router"); common(p); p.add_argument("--limit-rows", type=int, default=None); p.set_defaults(fn=cmd_router)
    p = sub.add_parser("dense"); common(p); p.add_argument("--steps", type=int, default=None); p.add_argument("--limit-rows", type=int, default=None); p.set_defaults(fn=cmd_dense)
    p = sub.add_parser("valcheck"); common(p); p.set_defaults(fn=cmd_valcheck)
    p = sub.add_parser("flops"); common(p); p.add_argument("--L", default="4096,28672,131072"); p.add_argument("--k", type=int, default=32)
    p.add_argument("--run", default=None); p.set_defaults(fn=cmd_flops)
    p = sub.add_parser("t1"); common(p, bb=False); p.add_argument("--seeds", default=None, help="e.g. 0 or 0,2 (default all)")
    p.add_argument("--from", dest="from_stage", default="head", choices=["head", "router", "valcheck"]); p.set_defaults(fn=cmd_t1)
    p = sub.add_parser("t2"); common(p, bb=False); p.add_argument("--seeds", default=None); p.set_defaults(fn=cmd_t2)
    p = sub.add_parser("t3"); common(p, bb=False); p.add_argument("--from", dest="from_stage", default="head", choices=["head", "router", "valcheck"]); p.set_defaults(fn=cmd_t3)
    p = sub.add_parser("all"); common(p); p.add_argument("--L", default="4096,28672,131072"); p.add_argument("--k", type=int, default=32)
    p.add_argument("--run", default=None); p.set_defaults(fn=cmd_all)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
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
