"""handover.py -- pack this work for a hand-over and put the code on GitHub.

    python handover.py sizes                                        # what exists where and how big (writes nothing)
    python handover.py pack --what code,results                     # archives + sha256 + file lists + HANDOVER.md
    python handover.py pack --what all --dest /home3/212aig08/handover
    python handover.py pack --what checkpoints --runs all           # the S8 arm checkpoints too (default: the base runs)
    python handover.py git --remote https://github.com/ACPLJE/Efficient-Guardrail.git          # init + commit + remote
    python handover.py git --remote https://github.com/ACPLJE/Efficient-Guardrail.git --push   # ... and push

Categories (paths resolved from env/s1.env through lgs1.paths, like every ctl):
    code         this package (LG_S1_ROOT) without env/s1.env -- a masked copy env/s1.env.example goes in instead -- and caches
    code_v04     the v0.4 code base (LG_V04_ROOT) without its .venv
    data         LG_DATA_V2: sources/, needles/, splits/, synth/, manifests/, tok/ ... (the sealed S2/S3 data)
    results      LG_OUT without checkpoints and prepped arrays: eval rows, cost rows, longbench rows, reports, logs, slurm
                 scripts, runs/v2/*/{meta.json, router.pt, *.json, *.log.jsonl, hardneg/}
    checkpoints  runs/v2/<run>/ with head.pt / dense_ft.pt: --runs main (default; qwen3guard_{0.6b,4b,8b}_s*, llamaguard3_8b_s*)
                 or --runs all (the S8 arms too; their head.pt symlinks stay relative)
    prep         LG_OUT/s4/prep: prepped token arrays (re-creatable with `python s4ctl.py prep`; not in `all`)
    legacy       LG_V04_RUNS, LG_PAPER_ROOT, LG_EXTERNAL_CODE, LG_EXTERNAL_DATA, LG_DATA_V1 (earlier rounds; not in `all`)
    hf / venv    never packed: HANDOVER.md lists the Hugging Face models and the venv comes from requirements.txt

Archives: code / code_v04 as zip (`-ry`: symlinks kept), the rest as tar.gz (`--format` overrides); every archive gets
<name>.sha256 and <name>.files.txt (the member list with sizes).  Restore = the commands HANDOVER.md prints.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lgs1.paths import S1Paths, env  # noqa: E402
from lgs2.config import S2Paths  # noqa: E402
from lgs4.config import BACKBONES, S4Paths  # noqa: E402

ENV_FILE = ROOT / "env" / "s1.env"
ENV_EXAMPLE = ROOT / "env" / "s1.env.example"
SECRET_RE = re.compile(r"TOKEN|SECRET|PASS|API_KEY|_KEY$", re.I)
NOT_SECRET = {"TOKENIZERS_PARALLELISM"}
TOKEN_VALUE_RE = re.compile(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
CODE_EXCLUDES = ["env/s1.env", "__pycache__", "*.pyc", ".pytest_cache", "*.tmp", ".git", "docs/.s8_project_doc.md", ".venv",
                 "*.zip", "*.tar.gz", "*.rows.jsonl", "*.pt", "*.npz"]
RESULTS_EXCLUDES = ["runs/v2/*/head*.pt", "runs/v2/*/dense_ft*.pt", "*.resume.pt", "s4/prep", "*.tmp", "tmp"]
DATA_EXCLUDES = ["*.tmp", "tmp", "__pycache__"]
CKPT_EXCLUDES = ["*.resume.pt", "*.tmp"]
MAIN_RUN_RE = re.compile(r"^(qwen3guard_(0\.6b|4b|8b)|llamaguard3_8b)_s\d+$")
GITIGNORE = """# hand-over: the env file carries machine paths and (optionally) an HF token; env/s1.env.example is the template
env/s1.env
__pycache__/
*.pyc
.pytest_cache/
*.tmp
docs/.s8_project_doc.md
.venv/
*.zip
*.tar.gz
*.rows.jsonl
*.pt
*.npz
"""


# ------------------------------------------------------------------------------------------------
@dataclass
class Cat:
    name: str
    desc: str
    root: Optional[Path]              # directory the archive is made of (archive root = root.name)
    excludes: List[str] = field(default_factory=list)
    members: Optional[List[str]] = None   # explicit top-level members under root (checkpoints: run dirs); None = root itself
    in_all: bool = True
    env_key: str = ""                 # the env key whose value is `root` on the receiving machine
    note: str = ""

    @property
    def exists(self) -> bool:
        return bool(self.root) and self.root.exists()


def _excluded(rel: str, patterns: Sequence[str]) -> bool:
    """GNU-tar-like unanchored matching: a pattern excludes a path when it matches the path, any suffix of it (component
    aligned) or any of its directory prefixes' suffixes; `*` matches `/`."""
    parts = rel.split("/")
    cands = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts) + 1):
            cands.append("/".join(parts[i:j]))
    return any(fnmatch.fnmatchcase(c, p) for c in cands for p in patterns)


def _walk(root: Path, excludes: Sequence[str], members: Optional[Sequence[str]] = None):
    """(relative path, size) of every file / symlink the archive would hold (no following of symlinks)."""
    tops = [root / m for m in members] if members else [root]
    for top in tops:
        if not top.exists():
            continue
        if top.is_file() or top.is_symlink():
            yield str(top.relative_to(root)), top.lstat().st_size
            continue
        for dp, dns, fns in os.walk(top):
            d = Path(dp)
            rel_d = str(d.relative_to(root))
            dns[:] = sorted(x for x in dns if not _excluded(f"{rel_d}/{x}" if rel_d != "." else x, excludes))
            for f in sorted(fns):
                rel = f"{rel_d}/{f}" if rel_d != "." else f
                if _excluded(rel, excludes):
                    continue
                try:
                    yield rel, (d / f).lstat().st_size
                except OSError:
                    continue


def _du(root: Path, excludes: Sequence[str] = (), members: Optional[Sequence[str]] = None) -> Tuple[int, int]:
    n = total = 0
    for _, sz in _walk(root, excludes, members):
        n += 1
        total += sz
    return total, n


def _human(b: Optional[int]) -> str:
    if b is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1024
    return str(b)


# ------------------------------------------------------------------------------------------------
def masked_env(src: Optional[Path] = None) -> str:
    """env/s1.env with secret values blanked (HF_TOKEN=..., anything *TOKEN*/*SECRET*/*PASS*/*KEY* except
    TOKENIZERS_PARALLELISM) -- the template the receiving side copies to env/s1.env."""
    src = src or ENV_FILE
    out = ["# env/s1.env.example -- copy to env/s1.env and set the paths of the new machine (python handover.py wrote this;",
           "# secret values were blanked).  LG_HOME / LG_OUT / LG_SOURCES / HF_HOME / LG_PYTHON are the ones every stage reads.", ""]
    for raw in src.read_text(encoding="utf-8").splitlines() if src.exists() else []:
        m = re.match(r"^(\s*(?:#\s*)?export\s+)([A-Za-z_][A-Za-z0-9_]*)=(.*)$", raw)
        if m and SECRET_RE.search(m.group(2)) and m.group(2) not in NOT_SECRET:
            comment = m.group(3).split(" #", 1)[1] if " #" in m.group(3) else ""
            raw = f"{m.group(1)}{m.group(2)}=" + (f"  #{comment}" if comment else "   # <set on the new machine>")
        elif TOKEN_VALUE_RE.search(raw):
            raw = TOKEN_VALUE_RE.sub("<masked>", raw)
        out.append(raw)
    return "\n".join(out) + "\n"


def write_env_example() -> Path:
    ENV_EXAMPLE.write_text(masked_env(), encoding="utf-8")
    return ENV_EXAMPLE


def categories(runs: str = "main") -> Dict[str, Cat]:
    P1 = S1Paths.load()
    P2 = S2Paths.load()
    P4 = S4Paths.load()
    run_dirs = sorted(d.name for d in P4.runs.iterdir() if d.is_dir()) if P4.runs.exists() else []
    main_runs = [r for r in run_dirs if MAIN_RUN_RE.match(r)]
    ck_members = run_dirs if runs == "all" else main_runs
    cats = {
        "code": Cat("code", "this package (lg_s1): lgs1-lgs8, s*ctl.py, slurm/, docs/, tests/, requirements.txt; env/s1.env -> env/s1.env.example",
                    ROOT, CODE_EXCLUDES, env_key="LG_S1_ROOT"),
        "code_v04": Cat("code_v04", "v0.4 code base (adapter target of S1; its .venv is not packed)", P1.v04_root,
                        [".venv", "__pycache__", "*.pyc", ".pytest_cache", ".git", "*.tmp"], env_key="LG_V04_ROOT"),
        "data": Cat("data", "LG_DATA_V2: sources/ (downloads), needles/, splits/, synth/ (S3 rows), manifests/, tok/ (spans), screen/, related/",
                    P2.root, DATA_EXCLUDES, env_key="LG_DATA_V2"),
        "results": Cat("results", "LG_OUT without checkpoints / prepped arrays: eval/, cost/, longbench/, reports/, logs/, slurm/, "
                       "runs/v2/*/{meta.json, router.pt, hashes.json, *.log.jsonl, val_report.json, hardneg/}",
                       P1.out, RESULTS_EXCLUDES, env_key="LG_OUT"),
        "checkpoints": Cat("checkpoints", f"runs/v2/<run>/ with head.pt / dense_ft.pt / router.pt ({'all runs incl. S8 arms' if runs == 'all' else 'base runs only; --runs all for the S8 arms'})",
                           P4.runs, CKPT_EXCLUDES, members=ck_members, env_key="LG_OUT/runs/v2",
                           note="restore into $LG_OUT/runs/v2 (the archive root is the run names); arm head.pt files are relative symlinks to the base run"),
        "prep": Cat("prep", "LG_OUT/s4/prep: prepped token arrays (re-creatable: python s4ctl.py prep)", P4.prep, ["*.tmp"], in_all=False,
                    env_key="LG_OUT/s4/prep"),
    }
    for key, p in (("LG_V04_RUNS", P1.v04_runs), ("LG_PAPER_ROOT", P1.paper_root), ("LG_EXTERNAL_CODE", P1.external_code),
                   ("LG_EXTERNAL_DATA", P1.external_data), ("LG_DATA_V1", P1.data_v1), ("LG_ROUND3_ROOT", P1.round3_root)):
        if p:
            cats[f"legacy_{key.lower()}"] = Cat(f"legacy_{key.lower()}", f"earlier round ({key})", p,
                                                [".venv", "__pycache__", "*.pyc", ".git", "*.tmp"], in_all=False, env_key=key)
    return cats


def _venv_dir() -> Optional[Path]:
    py = env("LG_PYTHON")
    if py and Path(py).exists():
        return Path(py).resolve().parent.parent
    return None


# ------------------------------------------------------------------------------------------------
def cmd_sizes(a) -> int:
    cats = categories(a.runs)
    P1 = S1Paths.load()
    print(f"\nhand-over sizes  ({_dt.datetime.now():%Y-%m-%d %H:%M}, {socket.gethostname()}; symlinks counted as links, nothing written)\n")
    print(f"  {'category':24s} {'size':>10s} {'files':>8s}  in 'all'  path")
    tot_all = 0
    for c in cats.values():
        if not c.exists:
            print(f"  {c.name:24s} {'-':>10s} {'-':>8s}  {'yes' if c.in_all else 'no ':8s}  {c.root} (missing)")
            continue
        sz, n = _du(c.root, c.excludes, c.members)
        if c.in_all:
            tot_all += sz
        print(f"  {c.name:24s} {_human(sz):>10s} {n:>8,}  {'yes' if c.in_all else 'no ':8s}  {c.root}{' (' + c.env_key + ')' if c.env_key else ''}")
        if c.name in ("data", "results") and c.root.exists():
            for sub in sorted(x for x in c.root.iterdir() if x.is_dir()):
                rel = sub.name
                if _excluded(rel, c.excludes):
                    continue
                ssz, sn = _du(c.root, c.excludes, [rel])
                if sn:                                   # s4/ shows nothing once s4/prep is excluded
                    print(f"    {rel + '/':22s} {_human(ssz):>10s} {sn:>8,}")
        if c.name == "checkpoints" and c.members:
            print(f"    runs: {', '.join(c.members)}")
    print(f"\n  total of 'all' (code, code_v04, data, results, checkpoints): {_human(tot_all)}")
    for label, p in (("hf cache (HF_HOME; re-download, see HANDOVER.md)", P1.hf_home), ("venv (LG_PYTHON; requirements.txt)", _venv_dir())):
        if p and p.exists():
            sz, n = _du(p, [])
            print(f"  not packed: {label:52s} {_human(sz):>10s} {n:>8,}  {p}")
    print("\n  next: python handover.py pack --what code,results [--dest DIR]   (then data, checkpoints as the disk allows)")
    return 0


# ------------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive(c: Cat, dest: Path, fmt: str, dry_run: bool = False) -> Optional[Path]:
    """One archive of a category: tar.gz (`tar -C parent name --exclude ...`, verbose listing -> .files.txt) or zip
    (`zip -ry`, symlinks kept, excludes as -x patterns)."""
    root = c.root
    members = c.members if c.members is not None else [root.name]
    parent = root if c.members is not None else root.parent
    if fmt == "zip":
        out = dest / f"{c.name}.zip"
        ex = []
        for p in c.excludes:
            ex += [p, f"*/{p}", f"*/{p}/*", f"{p}/*"] if "/" not in p or p.startswith("*") else [f"*/{p}", f"*/{p}/*"]
        cmd = ["zip", "-ry", "-q", str(out)] + members + ["-x"] + sorted(set(ex))
    else:
        out = dest / f"{c.name}.tar.gz"
        cmd = ["tar", "-cvvzf", str(out)] + [f"--exclude={p}" for p in c.excludes] + members
    print(f"[pack] {c.name}: {' '.join(cmd[:4])} ... ({len(members)} member{'s' if len(members) != 1 else ''}, cwd {parent})")
    if dry_run:
        return None
    if out.exists():
        out.unlink()
    with open(dest / f"{c.name}.files.txt", "w", encoding="utf-8") as lst:
        if fmt == "zip":
            rc = subprocess.run(cmd, cwd=parent).returncode
            if rc not in (0, 12):           # 12 = nothing to do
                raise SystemExit(f"zip failed rc={rc} for {c.name}")
            for rel, sz in _walk(root, c.excludes, c.members):
                lst.write(f"{sz:>14,}  {root.name + '/' if c.members is None else ''}{rel}\n")
        else:
            rc = subprocess.run(cmd, cwd=parent, stdout=lst, stderr=subprocess.PIPE, text=True)
            if rc.returncode != 0:
                raise SystemExit(f"tar failed rc={rc.returncode} for {c.name}: {rc.stderr[-2000:]}")
    sha = _sha256(out)
    (dest / f"{c.name}.sha256").write_text(f"{sha}  {out.name}\n", encoding="utf-8")
    print(f"[pack] {c.name}: {out} ({_human(out.stat().st_size)}, sha256 {sha[:12]}...)")
    return out


def _models() -> List[Tuple[str, str, str]]:
    rows = [(b.name, b.repo, "backbone (S4 head / router / dense-FT)") for b in BACKBONES.values()]
    try:
        from lgs1.zeroshot import guards as G
        rows += [(g.name, g.repo_id, "zero-shot / B1 guard") for g in G.GUARDS.values() if g.repo_id not in {r[1] for r in rows}]
    except Exception:                                                   # noqa: BLE001
        pass
    return rows


def _versions() -> str:
    import importlib.metadata as md
    out = [f"python {sys.version.split()[0]}"]
    for pkg in ("torch", "transformers", "datasets", "accelerate", "peft", "numpy"):
        try:
            out.append(f"{pkg} {md.version(pkg)}")
        except md.PackageNotFoundError:
            pass
    return ", ".join(out)


def handover_md(cats: Dict[str, Cat], packed: Dict[str, Path], dest: Path) -> str:
    P1 = S1Paths.load()
    data_root = S2Paths.load().root
    L = [f"# Landmark Guard hand-over ({_dt.datetime.now():%Y-%m-%d %H:%M}, {socket.gethostname()})", "",
         f"Package `{ROOT}`; outputs `{P1.out}` (LG_OUT); data `{data_root}` (LG_SOURCES = its sources/).  Env: {_versions()}.", "",
         "## Archives", "", "| archive | what | size | sha256 | unpack root (env key) |", "| --- | --- | --- | --- | --- |"]
    for name, c in cats.items():                    # every archive in dest, this call's and earlier ones'
        shaf = dest / f"{name}.sha256"
        if not shaf.exists():
            continue
        sha, fname = shaf.read_text().split()[:2]
        p = dest / fname
        if not p.exists():
            continue
        L.append(f"| `{p.name}` | {c.desc} | {_human(p.stat().st_size)} | `{sha[:16]}…` | `{c.env_key}` |")
    L += ["", "Every archive has `<name>.files.txt` (member list) and `<name>.sha256` (`sha256sum -c <name>.sha256`).", "",
          "## Not packed", "",
          "* **Hugging Face models** (`HF_HOME`): re-download with `huggingface-cli download <repo>`; `meta-llama/*` and some others are gated",
          "  (accept the licence on the Hub and set `HF_TOKEN` in env/s1.env).",
          "", "| name | repo | role |", "| --- | --- | --- |"]
    L += [f"| {n} | `{r}` | {role} |" for n, r, role in _models()]
    L += ["", "* **venv**: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (" + _versions() + ").",
          "* `s4/prep` token arrays (`--what prep` if wanted; else `python s4ctl.py prep`) and the earlier rounds (`--what legacy_*`).", "",
          "## Restore on the new machine", "", "```bash",
          f"# 1. code  (archive root = {ROOT.name})", f"unzip code.zip -d $HOME/        # or: git clone <repo> {ROOT.name}",
          f"cd $HOME/{ROOT.name} && cp env/s1.env.example env/s1.env",
          "#    edit env/s1.env: LG_USER, LG_HOME, LG_S1_ROOT, LG_OUT, LG_SOURCES, HF_HOME, LG_PYTHON, LG_PARTITION/QOS/GRES (+ HF_TOKEN for gated models)",
          "python3 -m venv $HOME/landmark_guard_mlsys/.venv && $HOME/landmark_guard_mlsys/.venv/bin/pip install -r requirements.txt",
          f"# 2. data  (archive root = {data_root.name}; LG_SOURCES must point at <where you unpack>/{data_root.name}/sources)",
          f"tar -xzf data.tar.gz -C <parent dir>          # -> <parent dir>/{data_root.name}/{{sources,needles,splits,synth,manifests,tok,...}}",
          f"# 3. results + checkpoints  (archive root = {P1.out.name}; LG_OUT = <parent dir>/{P1.out.name})",
          f"tar -xzf results.tar.gz -C <parent dir>       # -> <parent dir>/{P1.out.name}/{{eval,cost,longbench,reports,logs,slurm,runs}}",
          f"mkdir -p $LG_OUT/runs/v2 && tar -xzf checkpoints.tar.gz -C $LG_OUT/runs/v2",
          "# 4. models: huggingface-cli download Qwen/Qwen3Guard-Gen-0.6B  (etc., table above)",
          "# 5. verify", "python -m pytest tests -x -q", "python s5ctl.py status", "python s8ctl.py status",
          "python s6ctl.py tables ; python s8ctl.py tables   # re-derive every table from the rows", "```", "",
          "Stage docs: `README.md` (dated change log), `docs/S*_runbook.md`, `docs/plan_osdi_2026-09-29.md` (plan v2), "
          "`docs/E0_pass_criteria.md`; results tables under `reports/` are re-derivable from `eval/`, `cost/`, `longbench/` rows.", ""]
    return "\n".join(L)


def cmd_pack(a) -> int:
    cats = categories(a.runs)
    what = [x for x in (a.what or "").split(",") if x]
    if not what:
        raise SystemExit("--what code,results,... | all  (python handover.py sizes lists the categories)")
    if what == ["all"]:
        what = [c.name for c in cats.values() if c.in_all]
    bad = [w for w in what if w not in cats]
    if bad:
        raise SystemExit(f"unknown categories {bad}; one of {list(cats)}")
    dest = Path(a.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    write_env_example()
    packed: Dict[str, Path] = {}
    for w in what:
        c = cats[w]
        if not c.exists:
            print(f"[pack] {w}: {c.root} missing -> skipped")
            continue
        if c.members is not None and not c.members:
            print(f"[pack] {w}: nothing to pack under {c.root}")
            continue
        fmt = a.format if a.format != "auto" else ("zip" if w.startswith("code") else "tar")
        sz, n = _du(c.root, c.excludes, c.members)
        free = shutil.disk_usage(dest).free
        print(f"[pack] {w}: {_human(sz)} in {n:,} files -> {dest} (free {_human(free)})")
        if not a.dry_run and sz > free:
            print(f"[pack] WARNING {w}: {_human(sz)} uncompressed vs {_human(free)} free at {dest} -- rows/json compress ~5x, .pt hardly; "
                  f"if the archive stops with 'No space left', pack this category to another --dest")
        out = _archive(c, dest, fmt, a.dry_run)
        if out:
            packed[w] = out
    if a.dry_run:
        print("[pack] dry run: nothing written")
        return 0
    (dest / "HANDOVER.md").write_text(handover_md(cats, packed, dest), encoding="utf-8")
    print(f"[pack] {dest / 'HANDOVER.md'} (archives, sha256, model list, restore steps)")
    return 0


# ------------------------------------------------------------------------------------------------
def _git(args: List[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=check)


def cmd_git(a) -> int:
    """Make this package a git repo pointed at --remote: .gitignore (env/s1.env never tracked), env/s1.env.example,
    one commit of everything else, origin set; --push pushes (HTTPS asks for a token, or use the SSH remote)."""
    if shutil.which("git") is None:
        raise SystemExit("git is not installed here")
    gi = ROOT / ".gitignore"
    if not gi.exists():
        gi.write_text(GITIGNORE, encoding="utf-8")
        print(f"[git] wrote {gi}")
    elif "env/s1.env" not in gi.read_text(encoding="utf-8"):
        gi.write_text(gi.read_text(encoding="utf-8").rstrip() + "\nenv/s1.env\n", encoding="utf-8")
        print(f"[git] added env/s1.env to {gi}")
    print(f"[git] wrote {write_env_example()} (secrets blanked)")
    if not (ROOT / ".git").exists():
        r = _git(["init", "-b", a.branch], ROOT, check=False)
        if r.returncode != 0:                      # git < 2.28
            _git(["init"], ROOT)
            _git(["checkout", "-b", a.branch], ROOT)
        print(f"[git] init ({a.branch}) in {ROOT}")
    _git(["add", "-A"], ROOT)
    tracked = _git(["ls-files", "env/s1.env"], ROOT).stdout.strip()
    if tracked:
        raise SystemExit("env/s1.env is tracked -- `git rm --cached env/s1.env` first (it may hold HF_TOKEN)")
    staged = _git(["diff", "--cached", "--name-only"], ROOT).stdout.split()
    for f in staged:
        p = ROOT / f
        if p.is_file() and p.stat().st_size < 5_000_000:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if TOKEN_VALUE_RE.search(txt):
                raise SystemExit(f"{f} contains what looks like an API token -- remove it before committing")
    ident: List[str] = []
    if a.name:
        ident = ["-c", f"user.name={a.name}", "-c", f"user.email={a.email or a.name + '@users.noreply.github.com'}"]
    elif not _git(["config", "user.email"], ROOT, check=False).stdout.strip():
        user = env("LG_USER") or os.environ.get("USER") or "lg"
        ident = ["-c", f"user.name={user}", "-c", f"user.email={user}@users.noreply.github.com"]
    if _git(["diff", "--cached", "--quiet"], ROOT, check=False).returncode == 0:
        print("[git] nothing new to commit")
    else:
        msg = a.message or f"Landmark Guard S1-S8 ({_dt.date.today():%Y-%m-%d} hand-over)"
        _git(ident + ["commit", "-q", "-m", msg], ROOT)
        print(f"[git] committed: {msg} ({len(staged)} files)")
    remotes = _git(["remote"], ROOT).stdout.split()
    if a.remote:
        _git(["remote", "set-url" if "origin" in remotes else "add", "origin", a.remote], ROOT)
        print(f"[git] origin = {a.remote}")
    if a.push:
        r = _git(["push", "-u", "origin", a.branch], ROOT, check=False)
        print(r.stdout + r.stderr)
        if r.returncode != 0:
            print("[git] push failed: HTTPS needs a GitHub personal access token as the password (Settings > Developer settings > "
                  "Fine-grained tokens, Contents: read/write on the repo), or switch the remote to SSH:\n"
                  "      git remote set-url origin git@github.com:ACPLJE/Efficient-Guardrail.git   (after adding your key on GitHub)\n"
                  "      and if the login node has no outbound HTTPS, run `python handover.py git --remote ... --push` from the laptop "
                  "on the unzipped code.zip instead")
            return 1
        print(f"[git] pushed {a.branch} -> origin")
    else:
        print(f"[git] not pushed; when ready:  cd {ROOT} && git push -u origin {a.branch}   (or rerun with --push)")
    return 0


# ------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sizes"); p.add_argument("--runs", default="main", choices=["main", "all"]); p.set_defaults(fn=cmd_sizes)
    p = sub.add_parser("pack")
    p.add_argument("--what", default="", help="comma list: code,code_v04,data,results,checkpoints,prep,legacy_* | all")
    p.add_argument("--dest", default=str(Path(env("LG_HOME", str(Path.home()))) / "handover"))
    p.add_argument("--format", default="auto", choices=["auto", "tar", "zip"])
    p.add_argument("--runs", default="main", choices=["main", "all"], help="checkpoints: base runs (main) or every run dir (all)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("git")
    p.add_argument("--remote", default="https://github.com/ACPLJE/Efficient-Guardrail.git")
    p.add_argument("--branch", default="main"); p.add_argument("--push", action="store_true")
    p.add_argument("--message", default=None); p.add_argument("--name", default=None); p.add_argument("--email", default=None)
    p.set_defaults(fn=cmd_git)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
