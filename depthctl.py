#!/usr/bin/env python3
"""Depth-asymmetry PoC (plan 2026-10-10, E0.3 / E1.1): does the SAME backbone separate harmful windows at shallow
layers but injection windows (vs instruction-heavy benign text) only at deeper layers?

    python depthctl.py doctor                                   # what is cached / missing, where jobs will write
    python depthctl.py download [--with-s1]                     # login node only
    python depthctl.py smoke                                    # offline end-to-end check (tiny model, CPU, ~1 min)
    python depthctl.py run --run e03 --profile quick --backbones e03 --sbatch [--submit]
        = build -> per backbone: extract + fit + score -> analyze; --sbatch renders the scripts and runs
          `sbatch --test-only` on each (agctl convention), --submit then submits them chained with afterok
    python depthctl.py build|extract|fit|score|analyze|pipeline ... [--sbatch [--submit]]

Jobs source slurm/ag_common.sh (env/ag.env, the Python 3.11 AG venv, offline mode; CLAUDE.md 4, D-001).  Outputs go
to $AGDEPTH_OUT/<run> (default $AG_OUT/poc/agdepth/<run>, else $LG_OUT/agdepth/<run>); the verdict is in
report/report.md.  This PoC is not yet a SPEC experiment (DECISIONS D-041).  See docs/depth_poc_runbook.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lgs1.paths import env  # noqa: E402
from agdepth.config import (BACKBONE_ALIASES, BACKBONES, BG_DOMAINS, DEFAULT_POOLINGS, FRAMINGS, LAMBDAS,  # noqa: E402
                            POOLINGS, PROFILES, SNIP_SOURCES, TRAIN_SUBSAMPLE, AGPaths, backbone, is_local_model)

TIME = {"build": "04:00:00", "pipeline": "24:00:00", "analyze": "06:00:00"}
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


# ------------------------------------------------------------------------------------------------
# validation (everything is checked on the login node, before any script is rendered)
# ------------------------------------------------------------------------------------------------
def _abs_if_local(x: str) -> str:
    p = Path(x).expanduser()
    return str(p.resolve()) if p.exists() else x


def _expand_backbones(spec: str) -> List[str]:
    out: List[str] = []
    for item in [x for x in spec.split(",") if x]:
        out += BACKBONE_ALIASES.get(item, [item])
    bad = [b for b in out if b not in BACKBONES and not is_local_model(b) and not _REPO_ID.match(b)]
    if bad:
        raise SystemExit(f"[depthctl] unknown backbone(s) {bad}: use {sorted(BACKBONES)}, an alias {sorted(BACKBONE_ALIASES)}, "
                         f"an org/name repo id or a local model directory")
    return [_abs_if_local(b) for b in dict.fromkeys(out)]


def _check_tok(tok: str) -> None:
    """--tok must resolve offline on the compute node: a known name, a local directory, or a cached repo id."""
    from lgs2.config import TOKENIZERS
    if tok in ("whitespace", "ws") or Path(tok).is_dir():
        return
    repo = TOKENIZERS.get(tok, tok)
    if not _REPO_ID.match(repo):
        raise SystemExit(f"[depthctl] --tok {tok!r}: use {sorted(TOKENIZERS)}, a local directory or an org/name repo id")
    hub = Path(os.environ.get("HF_HUB_CACHE") or (AGPaths.load().hf_home / "hub"))
    if not (hub / f"models--{repo.replace('/', '--')}").exists():
        raise SystemExit(f"[depthctl] --tok {tok!r}: {repo} is not in the HF cache {hub} (jobs run offline); "
                         f"download it on the login node first")


def _validate(a) -> None:
    for name, allowed in (("framings", FRAMINGS), ("poolings", POOLINGS)):
        if hasattr(a, name):
            vals = [x for x in getattr(a, name).split(",") if x]
            bad = [x for x in vals if x not in allowed]
            if bad or not vals:
                raise SystemExit(f"[depthctl] --{name} {getattr(a, name)!r}: allowed {list(allowed)}")
    if getattr(a, "backbone", None):
        bbs = _expand_backbones(a.backbone)
        if len(bbs) != 1:
            raise SystemExit(f"[depthctl] --backbone takes one backbone, got {a.backbone!r} -> {bbs}; "
                             f"use `run --backbones {a.backbone}` for several")
        a.backbone = bbs[0]
    if getattr(a, "backbones", None):
        a.backbones = ",".join(_expand_backbones(a.backbones))
    if getattr(a, "dtype", None):
        from agdepth.extract import DTYPES
        if a.dtype not in DTYPES:
            raise SystemExit(f"[depthctl] --dtype {a.dtype!r}: allowed {sorted(DTYPES)}")
    if getattr(a, "tok", None):
        a.tok = _abs_if_local(a.tok)
        _check_tok(a.tok)
    if hasattr(a, "profile"):
        from agdepth.windows import profile_from_args
        profile_from_args(a)                                  # raises on an inconsistent override
    if getattr(a, "primary_fpr", None) is not None and not (0 < a.primary_fpr < 1):
        raise SystemExit("[depthctl] --primary-fpr must be in (0, 1)")


# ------------------------------------------------------------------------------------------------
# sbatch (agctl convention: render -> sbatch --test-only -> optional submit with afterok)
# ------------------------------------------------------------------------------------------------
def _q(xs) -> str:
    return " ".join(shlex.quote(str(x)) for x in xs)


def _render(P: AGPaths, run: str, name: str, cmd: List[str], time_key: str, cpus: int = 8) -> Path:
    import agctl
    from agdepth.lineage import now_ts
    R = P.run(run)
    R.logs.mkdir(parents=True, exist_ok=True)                 # Slurm cannot open -o in a missing directory
    R.slurm.mkdir(parents=True, exist_ok=True)
    out = os.environ.get("AG_OUT") or str(P.root)
    text = agctl.SBATCH_HEADER.format(name=name, partition=env("AG_PARTITION", "RTX6000ADA"), gres="gpu:1", cpus=cpus,
                                      time=env(f"AG_TIME_{time_key.upper()}", TIME[time_key]), log_dir=str(R.logs),
                                      array_line="", out=out, repo=agctl.REPO)
    text += f'export AGDEPTH_OUT="{P.root}"\n"$PY" depthctl.py {_q(cmd)}\n'
    path = R.slurm / f"{name}.{now_ts()}.sbatch"
    path.write_text(text)
    return path


def _dispatch(steps: List[Tuple[str, Path, List[str]]], submit: bool) -> int:
    """steps = [(name, script, dependency names)].  --test-only every script; with submit, submit in order."""
    import agctl
    for _, p, _ in steps:
        print(f"[sbatch] rendered {p}")
    if shutil.which("sbatch") is None:
        print("[sbatch] sbatch not found here: scripts rendered only (sbatch --test-only skipped)")
        return 0
    ok = True
    for name, p, _ in steps:
        r = agctl._sbatch(["--test-only", str(p)])
        print(f"[test-only] {name}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
        ok &= r.returncode == 0
    if not ok:
        print("[sbatch] --test-only failed: nothing submitted", file=sys.stderr)
        return 1
    if not submit:
        print("[sbatch] all scripts pass --test-only; add --submit to submit them")
        return 0
    jid: Dict[str, str] = {}
    for name, p, deps in steps:
        dj = [jid[d] for d in deps if d in jid]
        r = agctl._sbatch(([f"--dependency=afterok:{':'.join(dj)}"] if dj else []) + [str(p)])
        print(r.stdout.strip() or r.stderr.strip())
        m = re.search(r"Submitted batch job (\d+)", r.stdout)
        if r.returncode or not m:
            print(f"[sbatch] submitting {name} failed; later steps not submitted", file=sys.stderr)
            return 1
        jid[name] = m.group(1)
    print(f"[sbatch] submitted {jid}")
    return 0


# ------------------------------------------------------------------------------------------------
# argument lists that reproduce a command inside a job
# ------------------------------------------------------------------------------------------------
def _build_args(a) -> List[str]:
    out = ["--run", a.run, "--profile", a.profile, "--split-mode", a.split_mode, "--tok", a.tok, "--bg-domains", a.bg_domains]
    if a.sources:
        out += ["--sources", a.sources]
    for k in ("window", "snip_tok", "n_train", "n_val", "n_test", "n_bigneg", "max_len"):
        if getattr(a, k, None) is not None:
            out += ["--" + k.replace("_", "-"), str(getattr(a, k))]
    return out


def _bb_args(a, bb: str) -> List[str]:
    out = ["--run", a.run, "--backbone", bb, "--framings", a.framings, "--poolings", a.poolings]
    for k in ("device", "dtype"):
        if getattr(a, k, None):
            out += [f"--{k}", getattr(a, k)]
    for k in ("skip_score", "drop_feats", "fresh"):
        if getattr(a, k, False):
            out.append("--" + k.replace("_", "-"))
    return out


def _analyze_args(a) -> List[str]:
    out = ["--run", a.run]
    if a.primary_fpr:
        out += ["--primary-fpr", str(a.primary_fpr)]
    if a.n_boot is not None:
        out += ["--n-boot", str(a.n_boot)]
    return out


# ------------------------------------------------------------------------------------------------
# the backbone run: extract -> fit -> score (one process, the model is loaded once), with lineage
# ------------------------------------------------------------------------------------------------
def open_bb_run(P: AGPaths, run: str, bb_key: str, dtype_name: str, model_sha: str, fresh: bool = False):
    """Start (or resume) the lineage of one backbone run: extract, fit and score of that backbone share it.  The run id
    is kept in probes/<bb>/run_id.txt; a resume under different resume keys (code, env, data, model, dtype) exits 2."""
    from agdepth import lineage as LN
    from agdepth.analyze import exp_of_card
    from agdepth.windows import load_card
    R = P.run(run)
    card = load_card(R)
    bb = backbone(bb_key)
    run_dir = R.bb_run(bb.key)
    if fresh:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(R.root / "feats" / bb.key, ignore_errors=True)
    prof = card["recipe"]["profile"]
    seed = int(card["recipe"].get("seed", 0))
    exp_id = exp_of_card(card)
    # framings/poolings are outputs of one backbone run, not part of its identity (adding one resumes the same run)
    cfg = {"data_manifest_sha": card["data_manifest_sha"], "backbone": bb.repo, "chat": bb.chat, "dtype": dtype_name,
           "max_len": prof["max_len"], "lambdas": list(LAMBDAS), "seeds": list(prof["seeds"]),
           "train_subsample": TRAIN_SUBSAMPLE}
    chash = LN.config_hash(cfg)
    run_id, _ = LN.stable_run_id(run_dir, lambda: LN.make_run_id(exp_id, "probe", bb.key, f"depthwin-{prof['window']}",
                                                                 chash, seed))
    lin = LN.start(run_dir, run_id, exp_id, chash, seed=seed, producer="mock" if card.get("smoke") else "reference",
                   model_sha={"backbone": model_sha}, data_manifest_sha=card["data_manifest_sha"],
                   parents=[card["run_id"]] if card.get("run_id") else [])
    return lin, run_dir, card


def _fit_identity(P: AGPaths, run: str, bb_key: str) -> Tuple[str, str]:
    """(dtype, model_sha) of an existing backbone run, for a standalone fit (no model is loaded)."""
    R = P.run(run)
    bb = backbone(bb_key)
    lp = R.bb_run(bb.key) / "lineage.json"
    for fr in FRAMINGS:
        mp = R.feats(bb.key, fr) / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
            sha = (json.loads(lp.read_text()).get("model_sha") or {}).get("backbone") if lp.exists() else None
            from agdepth.lineage import model_sha_of
            return meta["dtype_model"], sha or model_sha_of(bb.repo, meta.get("model_sha", ""))
    raise SystemExit(f"[depthctl] no features for {bb.key} in run {run}: run extract (or pipeline) first")


def run_pipeline(P: AGPaths, run: str, bb_key: str, framings: Sequence[str], poolings: Sequence[str],
                 device: Optional[str] = None, dtype: Optional[str] = None, skip_score: bool = False,
                 drop_feats: bool = False, fresh: bool = False, budget: Optional[int] = None) -> None:
    from agdepth import extract, fit, score
    from agdepth import lineage as LN
    from agdepth.fit import probes_current
    bb = backbone(bb_key)
    if fresh:
        R0 = P.run(run)
        shutil.rmtree(R0.bb_run(bb.key), ignore_errors=True)
        shutil.rmtree(R0.root / "feats" / bb.key, ignore_errors=True)
    M = extract.load(bb.repo, device, dtype)
    lin, run_dir, card = open_bb_run(P, run, bb_key, M["dtype"], LN.model_sha_of(bb.repo, M["sha"]))
    R = P.run(run)
    dsha = card["data_manifest_sha"]
    prof = card["recipe"]["profile"]
    for fr in framings:
        need_fit = not all(probes_current(R, bb.key, fr, p, dsha) for p in poolings)
        if need_fit:
            extract.extract(run, bb.key if bb.key in BACKBONES else bb.repo, [fr], poolings, device=device, dtype=dtype,
                            budget=budget, P=P, M=M)
            fit.fit(run, bb.key if bb.key in BACKBONES else bb.repo, [fr], poolings, device=device, P=P)
        else:
            print(f"[pipeline] {bb.key}/{fr}: probes are current for this dataset, extract + fit skipped")
        if not skip_score:
            score.score(run, bb.key if bb.key in BACKBONES else bb.repo, [fr], poolings, device, dtype, budget, P=P, M=M)
        if drop_feats:
            extract.drop_feats(run, bb.key if bb.key in BACKBONES else bb.repo, [fr], poolings, P=P)
    fit.fit_lex(run, prof["seeds"], LAMBDAS, device, P)
    steps = {"framings": sorted(set(framings)), "poolings": sorted(set(poolings)), "run_id": lin.run_id}
    (run_dir / "steps.json").write_text(json.dumps(steps, indent=2))
    LN.finish(run_dir, lin)


# ------------------------------------------------------------------------------------------------
# sub-commands
# ------------------------------------------------------------------------------------------------
def cmd_build(a):
    _validate(a)
    P = AGPaths.load()
    if a.sbatch:
        p = _render(P, a.run, f"agd-build-{a.run}", ["build"] + _build_args(a), "build", cpus=4)
        return _dispatch([("build", p, [])], a.submit)
    from agdepth.windows import build, profile_from_args
    build(a.run, profile_from_args(a), a.split_mode, a.tok, [d for d in a.bg_domains.split(",") if d],
          sources_only=[s for s in a.sources.split(",") if s] or None, P=P)


def cmd_pipeline(a):
    _validate(a)
    P = AGPaths.load()
    if a.sbatch:
        bb = backbone(a.backbone)
        p = _render(P, a.run, f"agd-{bb.key}-{a.run}", ["pipeline"] + _bb_args(a, a.backbone), "pipeline")
        return _dispatch([(bb.key, p, [])], a.submit)
    run_pipeline(P, a.run, a.backbone, a.framings.split(","), a.poolings.split(","), a.device, a.dtype, a.skip_score,
                 a.drop_feats, a.fresh)


def _single(step: str):
    def cmd(a):
        _validate(a)
        P = AGPaths.load()
        bb = backbone(a.backbone)
        if a.sbatch:
            p = _render(P, a.run, f"agd-{step}-{bb.key}-{a.run}", [step] + _bb_args(a, a.backbone), "pipeline")
            return _dispatch([(step, p, [])], a.submit)
        from agdepth import extract, fit, score
        from agdepth import lineage as LN
        fr, po = a.framings.split(","), a.poolings.split(",")
        if a.fresh:
            raise SystemExit("[depthctl] --fresh starts a whole backbone run: use `pipeline --fresh`")
        if step == "fit":                                     # same lineage as the pipeline run of this backbone
            dt, sha = _fit_identity(P, a.run, a.backbone)
            lin, run_dir, _ = open_bb_run(P, a.run, a.backbone, dt, sha)
            fit.fit(a.run, a.backbone, fr, po, device=a.device, P=P)
        else:
            M = extract.load(bb.repo, a.device, a.dtype)
            lin, run_dir, _ = open_bb_run(P, a.run, a.backbone, M["dtype"], LN.model_sha_of(bb.repo, M["sha"]))
            if step == "extract":
                extract.extract(a.run, a.backbone, fr, po, device=a.device, dtype=a.dtype, P=P, M=M)
            else:
                score.score(a.run, a.backbone, fr, po, a.device, a.dtype, P=P, M=M)
        LN.finish(run_dir, lin)
    return cmd


def cmd_analyze(a):
    _validate(a)
    P = AGPaths.load()
    if a.sbatch:
        p = _render(P, a.run, f"agd-analyze-{a.run}", ["analyze"] + _analyze_args(a), "analyze")
        return _dispatch([("analyze", p, [])], a.submit)
    from agdepth.analyze import analyze
    analyze(a.run, a.primary_fpr, a.n_boot, P=P)


def cmd_run(a):
    """build (optional) -> one job per backbone -> analyze."""
    _validate(a)
    P = AGPaths.load()
    bbs = [x for x in a.backbones.split(",") if x]
    if not a.sbatch:
        if not a.skip_build:
            cmd_build(a)
        for bb in bbs:
            run_pipeline(P, a.run, bb, a.framings.split(","), a.poolings.split(","), a.device, a.dtype, a.skip_score,
                         a.drop_feats, a.fresh)
        from agdepth.analyze import analyze
        analyze(a.run, a.primary_fpr, a.n_boot, P=P)
        return 0
    steps: List[Tuple[str, Path, List[str]]] = []
    if not a.skip_build:
        steps.append(("build", _render(P, a.run, f"agd-build-{a.run}", ["build"] + _build_args(a), "build", cpus=4), []))
    names = []
    for bb in bbs:
        key = backbone(bb).key
        names.append(key)
        steps.append((key, _render(P, a.run, f"agd-{key}-{a.run}", ["pipeline"] + _bb_args(a, bb), "pipeline"),
                      ["build"] if not a.skip_build else []))
    steps.append(("analyze", _render(P, a.run, f"agd-analyze-{a.run}", ["analyze"] + _analyze_args(a), "analyze"), names))
    return _dispatch(steps, a.submit)


def cmd_smoke(a):
    """Offline end-to-end run: placeholder classes, tiny random model, CPU.  Checks the plumbing, not the claim.
    Nothing outside --dir (default: a new temp dir) is touched, and the process environment is left alone."""
    from agdepth.analyze import analyze
    from agdepth.config import PROFILES as PR
    from agdepth.tiny import make_tiny_repo
    from agdepth.windows import build
    base = Path(a.dir).expanduser().resolve() if a.dir else Path(tempfile.mkdtemp(prefix="agdepth_smoke_"))
    P = AGPaths.load(root=base / "out")
    shutil.rmtree(P.run("smoke").root, ignore_errors=True)     # only the smoke run's own directory
    repo = str(make_tiny_repo(base / "tiny-model"))
    build("smoke", PR["smoke"], smoke=True, P=P)
    run_pipeline(P, "smoke", repo, ["raw", "task"], ["mean", "last", "span"], device="cpu", dtype="fp32", budget=4096)
    s = analyze("smoke", primary_fpr=a.primary_fpr, n_boot=a.n_boot if a.n_boot is not None else 20, P=P)
    print(f"[smoke] OK -> {P.run('smoke').report / 'report.md'}")
    return s


def cmd_doctor(a):
    from agdepth.sources import registry_files
    P = AGPaths.load()
    print(f"output root  {P.root}   (AGDEPTH_OUT > $AG_OUT/poc/agdepth > $LG_OUT/agdepth)")
    print(f"LG_SOURCES   {P.sources}")
    print(f"LG_DATA_V2   {P.data_v2}")
    print(f"HF_HOME      {P.hf_home}")
    ag_env = ROOT / "env" / "ag.env"
    print(f"jobs         source slurm/ag_common.sh -> {ag_env} ({'present' if ag_env.exists() else 'MISSING: cp env/ag.env.example env/ag.env'})"
          f", venv {env('AG_VENV', '/home/212aig08/ag_venv/ag311')}")
    try:
        import torch
        import transformers
        print(f"python       {sys.executable} {sys.version.split()[0]}  torch {torch.__version__} "
              f"(cuda={torch.cuda.is_available()})  transformers {transformers.__version__}")
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
        keep = (P.data_v2 / "screen" / f"keep_{dom}.json").exists()
        state = "haystack" + ("" if keep else " (UNSCREENED)") if h.exists() else ("stream (UNSCREENED)" if s.exists() else "MISSING")
        print(f"background {dom:10s} {state}")
    print()
    for src in SNIP_SOURCES:
        if src.kind == "registry":
            files = registry_files(src, P)
            print(f"source {src.name:12s} {src.cls:9s} "
                  f"{'OK (%d files)' % len(files) if files else 'MISSING -> python depthctl.py download --only ' + src.registry}")
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
    rc = 0 if not r.get("error") and not r.get("failed") else 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)                       # streaming datasets threads can hang at interpreter exit (as in lgs1.download)


# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=True):
        if run:
            p.add_argument("--run", required=True, help="run name (directory under the output root)")
        p.add_argument("--sbatch", action="store_true", help="render sbatch script(s) and run sbatch --test-only")
        p.add_argument("--submit", action="store_true", help="with --sbatch: submit after every --test-only passes")

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
        p.add_argument("--fresh", action="store_true",
                       help="pipeline: delete this backbone's run (features, probes, lineage) and start a new run id")

    def analyze_opts(p):
        p.add_argument("--primary-fpr", type=float, default=None, help="override the plan's 2e-4 (reported as such)")
        p.add_argument("--n-boot", type=int, default=None)

    p = sub.add_parser("doctor"); p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("download"); p.set_defaults(fn=cmd_download)
    p.add_argument("--only", default=""); p.add_argument("--with-s1", action="store_true")
    p.add_argument("--token", default=None); p.add_argument("--force-online", action="store_true")
    p.add_argument("--no-skip", action="store_true")
    p = sub.add_parser("smoke"); p.set_defaults(fn=cmd_smoke)
    p.add_argument("--dir", default=None); analyze_opts(p)
    p = sub.add_parser("build"); common(p); data_opts(p); p.set_defaults(fn=cmd_build)
    p = sub.add_parser("pipeline"); common(p); model_opts(p); p.set_defaults(fn=cmd_pipeline)
    for name in ("extract", "fit", "score"):
        p = sub.add_parser(name); common(p); model_opts(p); p.set_defaults(fn=_single(name))
    p = sub.add_parser("analyze"); common(p); analyze_opts(p); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("run"); common(p); data_opts(p); model_opts(p, bb=False); analyze_opts(p); p.set_defaults(fn=cmd_run)
    p.add_argument("--backbones", default="e03", help="e03 | e11 | comma list of backbone keys (aliases expand)")
    p.add_argument("--skip-build", action="store_true", help="reuse the existing windows of this run")
    a = ap.parse_args(argv)
    if getattr(a, "submit", False) and not getattr(a, "sbatch", False):
        ap.error("--submit needs --sbatch")
    r = a.fn(a)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
