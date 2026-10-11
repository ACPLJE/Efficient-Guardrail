#!/usr/bin/env python3
"""Depth-asymmetry PoC (plan 2026-10-10, E0.3 / E1.1): does the SAME backbone separate harmful windows at shallow
layers but injection windows (vs instruction-heavy benign text) only at deeper layers?

    python depthctl.py doctor                                   # what is cached / missing
    python depthctl.py download [--with-s1]                     # login node only
    python depthctl.py smoke                                    # offline end-to-end check (tiny model, CPU, ~1 min)
    python depthctl.py run --run e03 --profile quick --backbones qwen3guard_0.6b,llama31_8b --sbatch
        = build (CPU) -> per backbone: extract + fit + score (GPU) -> analyze (CPU), chained with afterok
    python depthctl.py build|extract|fit|score|analyze|pipeline ... [--sbatch] [--dry-run]

Outputs under $AG_OUT/<run> (default $LG_OUT/agdepth/<run>); the verdict is in report/report.md.
See docs/depth_poc_runbook.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lgs1.paths import env  # noqa: E402
from agdepth.config import (BACKBONES, BG_DOMAINS, DEFAULT_POOLINGS, E03_BACKBONES, E11_BACKBONES, PROFILES,  # noqa: E402
                            SNIP_SOURCES, AGPaths, backbone)

TIME = {"build": "04:00:00", "pipeline": "24:00:00", "analyze": "06:00:00"}


def _submit(job: str, body: str, time_key: str, a, dep: Optional[str] = None, cpu: bool = False,
            min_gpu_gb: int = 24, mem: Optional[str] = None, cpus: Optional[str] = None) -> Optional[str]:
    import s1ctl
    t = env(f"AG_TIME_{time_key.upper()}", TIME[time_key])
    return s1ctl.write_sbatch(job, body, t, dep=dep, dry_run=getattr(a, "dry_run", False), cpu=cpu,
                              min_gpu_gb=min_gpu_gb, mem=mem, cpus=cpus)


def _q(xs) -> str:
    return " ".join(shlex.quote(str(x)) for x in xs)


# ------------------------------------------------------------------------------------------------
# sub-commands
# ------------------------------------------------------------------------------------------------
def _build_args(a) -> List[str]:
    out = ["--run", a.run, "--profile", a.profile, "--split-mode", a.split_mode, "--tok", a.tok, "--bg-domains", a.bg_domains]
    if a.sources:
        out += ["--sources", a.sources]
    for k in ("window", "snip_tok", "n_train", "n_val", "n_test", "n_bigneg", "max_len"):
        if getattr(a, k, None) is not None:
            out += ["--" + k.replace("_", "-"), str(getattr(a, k))]
    return out


def cmd_build(a):
    if a.sbatch:
        return _submit(f"ag_build_{a.run}", f"${{PY}} depthctl.py build {_q(_build_args(a))}", "build", a, cpu=True,
                       mem="32G", cpus="4")
    from agdepth.windows import build, profile_from_args
    build(a.run, profile_from_args(a), a.split_mode, a.tok, [d for d in a.bg_domains.split(",") if d],
          sources_only=[s for s in a.sources.split(",") if s] or None)


def _bb_args(a, bb: str) -> List[str]:
    return ["--run", a.run, "--backbone", bb, "--framings", a.framings, "--poolings", a.poolings]


def cmd_pipeline(a, dep: Optional[str] = None):
    """extract -> fit -> score for one backbone in one process (the model is loaded once)."""
    bb = backbone(a.backbone)
    if a.sbatch:
        extra = (["--skip-score"] if a.skip_score else []) + (["--drop-feats"] if a.drop_feats else [])
        return _submit(f"ag_{bb.key}_{a.run}", f"${{PY}} depthctl.py pipeline {_q(_bb_args(a, a.backbone) + extra)}",
                       "pipeline", a, dep=dep, min_gpu_gb=bb.min_gpu_gb, mem="96G")
    from agdepth import extract, fit, score
    fr = [f for f in a.framings.split(",") if f]
    po = [p for p in a.poolings.split(",") if p]
    M = extract.load(bb.repo, a.device, a.dtype)
    extract.extract(a.run, a.backbone, fr, po, M=M)
    fit.fit(a.run, a.backbone, fr, po, device=a.device)
    if not a.skip_score:
        score.score(a.run, a.backbone, fr, po, M=M)
    if a.drop_feats:
        extract.drop_feats(a.run, a.backbone, fr, po)


def cmd_extract(a):
    bb = backbone(a.backbone)
    if a.sbatch:
        return _submit(f"ag_extract_{bb.key}_{a.run}", f"${{PY}} depthctl.py extract {_q(_bb_args(a, a.backbone))}",
                       "pipeline", a, min_gpu_gb=bb.min_gpu_gb, mem="96G")
    from agdepth import extract
    extract.extract(a.run, a.backbone, a.framings.split(","), a.poolings.split(","), device=a.device, dtype=a.dtype)


def cmd_fit(a):
    bb = backbone(a.backbone)
    if a.sbatch:
        return _submit(f"ag_fit_{bb.key}_{a.run}", f"${{PY}} depthctl.py fit {_q(_bb_args(a, a.backbone))}", "pipeline", a,
                       min_gpu_gb=24, mem="96G")
    from agdepth import fit
    fit.fit(a.run, a.backbone, a.framings.split(","), a.poolings.split(","), device=a.device)


def cmd_score(a):
    bb = backbone(a.backbone)
    if a.sbatch:
        return _submit(f"ag_score_{bb.key}_{a.run}", f"${{PY}} depthctl.py score {_q(_bb_args(a, a.backbone))}",
                       "pipeline", a, min_gpu_gb=bb.min_gpu_gb, mem="64G")
    from agdepth import score
    score.score(a.run, a.backbone, a.framings.split(","), a.poolings.split(","), a.device, a.dtype)


def _analyze_args(a) -> List[str]:
    out = ["--run", a.run]
    if a.primary_fpr:
        out += ["--primary-fpr", str(a.primary_fpr)]
    if a.n_boot is not None:
        out += ["--n-boot", str(a.n_boot)]
    return out


def cmd_analyze(a, dep: Optional[str] = None):
    if a.sbatch:
        return _submit(f"ag_analyze_{a.run}", f"${{PY}} depthctl.py analyze {_q(_analyze_args(a))}", "analyze", a,
                       dep=dep, cpu=True, mem="64G", cpus="8")
    from agdepth.analyze import analyze
    analyze(a.run, a.primary_fpr, a.n_boot)


def cmd_run(a):
    """build (optional) -> one GPU job per backbone -> analyze."""
    bbs = {"e03": E03_BACKBONES, "e11": E11_BACKBONES}.get(a.backbones, [b for b in a.backbones.split(",") if b])
    if not a.sbatch:
        if not a.skip_build:
            cmd_build(a)
        for bb in bbs:
            a.backbone = bb
            cmd_pipeline(a)
        return cmd_analyze(a)
    dep = None
    if not a.skip_build:
        dep = cmd_build(a)
    jids = []
    for bb in bbs:
        a.backbone = bb
        jids.append(cmd_pipeline(a, dep=dep))
    jids = [j for j in jids if j]
    cmd_analyze(a, dep=":".join(jids) if jids else None)


def cmd_smoke(a):
    """Offline end-to-end run: placeholder classes, tiny random model, CPU.  Checks the plumbing, not the claim."""
    base = Path(a.dir) if a.dir else Path(tempfile.mkdtemp(prefix="agdepth_smoke_"))
    os.environ["AG_OUT"] = str(base / "out")
    from agdepth import analyze, extract, fit, score
    from agdepth.tiny import make_tiny_repo
    from agdepth.windows import build
    repo = str(make_tiny_repo(base / "tiny-model"))
    build("smoke", PROFILES["smoke"], smoke=True)
    M = extract.load(repo, "cpu", "fp32")
    fr, po = ["raw", "task"], ["mean", "last", "span"]
    extract.extract("smoke", repo, fr, po, M=M, budget=4096)
    fit.fit("smoke", repo, fr, po, device="cpu")
    score.score("smoke", repo, fr, po, M=M, budget=4096)
    s = analyze.analyze("smoke", n_boot=a.n_boot if a.n_boot is not None else 20)
    print(f"[smoke] OK -> {AGPaths.load().run('smoke').report / 'report.md'}")
    return s


def cmd_doctor(a):
    from agdepth.sources import registry_files
    P = AGPaths.load()
    print(f"AG_OUT       {P.root}")
    print(f"LG_SOURCES   {P.sources}")
    print(f"LG_DATA_V2   {P.data_v2}")
    print(f"HF_HOME      {P.hf_home}")
    try:
        import torch
        import transformers
        print(f"python       {sys.executable}  torch {torch.__version__} (cuda={torch.cuda.is_available()})  "
              f"transformers {transformers.__version__}")
    except Exception as e:
        print(f"python       {sys.executable}  torch/transformers MISSING ({e})")
    try:
        import matplotlib
        print(f"matplotlib   {matplotlib.__version__}")
    except Exception:
        print("matplotlib   MISSING (figures are skipped; pip install matplotlib)")
    pool = P.data_v2 / "needles" / "pool.jsonl.gz"
    print(f"\nLG v2 needle pool  {'OK' if pool.exists() else 'missing (raw S2 needle sources are read instead)'}  {pool}")
    for dom in BG_DOMAINS:
        h = P.data_v2 / "haystack" / dom / "docs.jsonl.gz"
        s = P.sources / dom / "docs.jsonl.gz"
        print(f"background {dom:10s} {'haystack' if h.exists() else ('stream' if s.exists() else 'MISSING')}")
    print()
    for src in SNIP_SOURCES:
        if src.kind == "registry":
            files = registry_files(src, P)
            print(f"source {src.name:12s} {src.cls:9s} {'OK (%d files)' % len(files) if files else 'MISSING -> python depthctl.py download --only ' + src.registry}")
        else:
            print(f"source {src.name:12s} {src.cls:9s} {src.kind}")
    print()
    hub = Path(os.environ.get("HF_HUB_CACHE") or (P.hf_home / "hub"))
    for k, b in BACKBONES.items():
        d = hub / f"models--{b.repo.replace('/', '--')}" / "snapshots"
        ok = d.exists() and any(d.iterdir())
        print(f"backbone {k:16s} {b.repo:40s} {'cached' if ok else 'not cached'}{'  (gated)' if b.gated else ''}")


def cmd_download(a):
    from agdepth.download import run
    r = run(only=[x for x in a.only.split(",") if x] or None, with_s1=a.with_s1, token=a.token,
            force_online=a.force_online, skip_existing=not a.no_skip)
    return 0 if not r.get("error") and not r.get("failed") else 1


# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=True):
        if run:
            p.add_argument("--run", required=True, help="run name (directory under $AG_OUT)")
        p.add_argument("--sbatch", action="store_true", help="submit to Slurm instead of running here")
        p.add_argument("--dry-run", action="store_true", help="with --sbatch: write the script, do not submit")

    def data_opts(p):
        p.add_argument("--profile", default="quick", choices=sorted(PROFILES))
        p.add_argument("--split-mode", default="group", choices=["group", "source"])
        p.add_argument("--tok", default="qwen3", help="reference tokenizer for window lengths")
        p.add_argument("--bg-domains", default=",".join(BG_DOMAINS))
        p.add_argument("--sources", default="", help="comma list of snippet sources (default: all available)")
        from agdepth.windows import add_profile_overrides
        add_profile_overrides(p)

    def model_opts(p, bb=True):
        if bb:
            p.add_argument("--backbone", required=True, help=f"one of {sorted(BACKBONES)}, a repo id or a local dir")
        p.add_argument("--framings", default="raw,task")
        p.add_argument("--poolings", default=",".join(DEFAULT_POOLINGS))
        p.add_argument("--device", default=None)
        p.add_argument("--dtype", default=None)
        p.add_argument("--skip-score", action="store_true", help="pipeline: skip the big negative pool")
        p.add_argument("--drop-feats", action="store_true",
                       help="pipeline: delete the feature memmaps after fit/score (probes + scores are kept)")

    def analyze_opts(p):
        p.add_argument("--primary-fpr", type=float, default=None)
        p.add_argument("--n-boot", type=int, default=None)

    p = sub.add_parser("doctor"); p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("download"); p.set_defaults(fn=cmd_download)
    p.add_argument("--only", default=""); p.add_argument("--with-s1", action="store_true")
    p.add_argument("--token", default=None); p.add_argument("--force-online", action="store_true")
    p.add_argument("--no-skip", action="store_true")
    p = sub.add_parser("smoke"); p.set_defaults(fn=cmd_smoke)
    p.add_argument("--dir", default=None); p.add_argument("--n-boot", type=int, default=None)
    p = sub.add_parser("build"); common(p); data_opts(p); p.set_defaults(fn=cmd_build)
    for name, fn in (("extract", cmd_extract), ("fit", cmd_fit), ("score", cmd_score), ("pipeline", cmd_pipeline)):
        p = sub.add_parser(name); common(p); model_opts(p); p.set_defaults(fn=fn)
    p = sub.add_parser("analyze"); common(p); analyze_opts(p); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("run"); common(p); data_opts(p); model_opts(p, bb=False); analyze_opts(p); p.set_defaults(fn=cmd_run)
    p.add_argument("--backbones", default="e03", help="e03 | e11 | comma list of backbone keys")
    p.add_argument("--skip-build", action="store_true", help="reuse the existing windows of this run")
    a = ap.parse_args(argv)
    r = a.fn(a)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
