"""Repository hygiene (CLAUDE.md 3.2, 3.4; build plan 4.2 S0): no secrets, .gitignore rules, pyproject core deps,
the CPU workflow and the local pre-PR gate. Git-dependent checks skip outside a git checkout (e.g. a code.zip)."""
from __future__ import annotations

import ast
import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from _ag_util import ROOT

CHECK_SECRETS = ROOT / "tools" / "check_secrets.py"
PR_CHECK = ROOT / "tools" / "pr_check.sh"
CPU_YML = ROOT / ".github" / "workflows" / "cpu.yml"
GATES = ("python -m pytest tests -x -q", "pytest $(cat tests/manifests/cpu.txt)")   # CLAUDE.md 3.4, verbatim
# the planted tokens are built at run time so this tracked file never matches the patterns itself
HF_FAKE = "hf_" + "Zq7" * 13 + "w"                                  # 40 chars after the prefix
GHP_FAKE = "ghp_" + "a1B2" * 9                                      # 36
PAT_FAKE = "github_pat_" + "11ABCDEFG0_" * 7                        # 77


def _git_env(**extra: str) -> dict[str, str]:
    """Isolate git from the user's global/system config (excludesFile, init templates, hooks)."""
    return {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, **extra}


def _git(*args: str, cwd: Path, check: bool = True, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check, env=_git_env(**env))


def _in_git_checkout(path: Path) -> bool:
    if shutil.which("git") is None:
        return False
    r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, capture_output=True, text=True,
                       env=_git_env())
    return r.returncode == 0 and r.stdout.strip() == "true"


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
needs_repo = pytest.mark.skipif(not _in_git_checkout(ROOT), reason="not a git checkout (git missing or code.zip)")


def _secrets(cwd: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECK_SECRETS), *args], cwd=cwd, capture_output=True, text=True,
                          env=_git_env(**env))


def _tmp_repo(tmp_path: Path, files: dict[str, str | bytes], *, add: bool = True) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", ".", cwd=repo)
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content) if isinstance(content, bytes) else p.write_text(content)
    if add and files:
        _git("add", "-f", *files, cwd=repo)                          # -f: an env/*.env must be tracked on purpose
    return repo


# ---- check_secrets ----------------------------------------------------------------------------------------------

@needs_repo
def test_check_secrets_passes_on_repo():
    r = _secrets(ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().endswith("0 findings")


@needs_git
def test_check_secrets_catches_planted_hf_token(tmp_path):
    repo = _tmp_repo(tmp_path, {
        "cfg.py": f'# settings\nHF_TOKEN = "{HF_FAKE}"\n',
        "notes.md": "set HF_TOKEN=hf_... or hf_xxx; never paste hf_… or ghp_… here\n",   # placeholders
    })
    r = _secrets(repo)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "cfg.py:2: hf_token" in r.stdout.splitlines()
    assert HF_FAKE not in out and HF_FAKE[3:20] not in out           # the token itself is never printed
    assert "notes.md" not in out


@needs_git
def test_check_secrets_kinds_env_files_binary_untracked_args(tmp_path):
    repo = _tmp_repo(tmp_path, {
        "a/gh.txt": f"x\ny\nclassic={GHP_FAKE} fine={PAT_FAKE}\n",
        "env/ag.env": "export AG_OUT=/x\n",                          # tracked env file: refused even without a token
        "env/ag.env.example": "export AG_OUT=/x\n# HF_TOKEN=\n",
        "lib/blob.bin": b"\x00\x01" + HF_FAKE.encode() + b"\n",      # binary: skipped
        "sub/ok.txt": "nothing here\n",
    })
    r = _secrets(repo / "sub")                                       # from a subdirectory: still the whole repo
    lines = r.stdout.splitlines()
    assert r.returncode == 1
    assert "a/gh.txt:3: github_token" in lines and "a/gh.txt:3: github_pat" in lines
    assert any(ln.startswith("env/ag.env:0: tracked_env_file") for ln in lines)
    assert not any("ag.env.example" in ln or "blob.bin" in ln for ln in lines)
    assert "1 skipped" in lines[-1]
    for tok in (GHP_FAKE, PAT_FAKE):
        assert tok not in r.stdout + r.stderr

    _git("rm", "-q", "--cached", "env/ag.env", "a/gh.txt", cwd=repo)  # now clean in the index ...
    (repo / "new.py").write_text(f"t = '{HF_FAKE}'\n")                # ... but an untracked file holds a token
    assert _secrets(repo).returncode == 0
    r = _secrets(repo, "--untracked")
    assert r.returncode == 1 and "new.py:1: hf_token" in r.stdout
    assert "env/ag.env:0: env_file_not_ignored" in r.stdout          # git would add it: no .gitignore in this repo

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text(f"{HF_FAKE}\n")
    r = _secrets(repo, str(outside))                                  # extra paths are scanned too
    assert r.returncode == 1 and f"{outside / 'leak.txt'}:1: hf_token" in r.stdout


def test_check_secrets_outside_git(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "clean.txt").write_text("no tokens\n")
    ceiling = {"GIT_CEILING_DIRECTORIES": str(tmp_path)}              # never discover a repo above tmp_path
    r = _secrets(plain, **ceiling)
    assert r.returncode == 2 and "not inside a git checkout" in r.stderr
    assert _secrets(plain, "clean.txt", **ceiling).returncode == 0
    assert _secrets(plain, "missing.txt", **ceiling).returncode == 2


def test_check_secrets_patterns_unit():
    sys.path.insert(0, str(ROOT / "tools"))
    import check_secrets as cs
    text = "\n".join([f"a {HF_FAKE}", "hf_" + "x" * 29, "hf_reference_target", f"{GHP_FAKE[:-1]}", PAT_FAKE])
    assert list(cs.scan_text(text)) == [(1, "hf_token"), (5, "github_pat")]
    assert cs.is_env_file("env/ag.env") and cs.is_env_file("env/.env")
    assert not cs.is_env_file("env/ag.env.example") and not cs.is_env_file("lg_s1/env/s1.env")
    assert not cs.is_env_file("env/sub/x.env")


# ---- .gitignore ---------------------------------------------------------------------------------------------------

IGNORED = ("results/x", "a.jsonl", "raw/E0.1/run/events.jsonl", "env/ag.env", "env/s1.env", "figures/f.pdf",
           "x.npz", "ckpt.pt")
TRACKABLE = ("tests/fixtures/results_mock/x.jsonl.txt", "tests/data/mooncake_conversation_50.jsonl.txt",
             "env/ag.env.example", "env/s1.env.example", "plots/fig09.py")


@needs_repo
@pytest.mark.parametrize("path,ignored", [(p, True) for p in IGNORED] + [(p, False) for p in TRACKABLE])
def test_gitignore_rules(path, ignored):
    # --no-index: judge the rules themselves, not whether the path happens to be tracked
    r = _git("check-ignore", "--no-index", "-q", path, cwd=ROOT, check=False)
    if r.returncode == 129:                                          # usage error: git older than 1.8.5
        pytest.skip(f"git check-ignore --no-index unsupported: {r.stderr.strip()[:80]}")
    assert r.returncode in (0, 1), r.stderr
    assert (r.returncode == 0) is ignored, f"{path}: expected {'ignored' if ignored else 'not ignored'}"


@needs_repo
def test_no_tracked_file_is_ignored():
    r = _git("ls-files", "-z", "--cached", "--ignored", "--exclude-standard", cwd=ROOT)
    assert [p for p in r.stdout.split("\0") if p] == []


# ---- pyproject ----------------------------------------------------------------------------------------------------

def _dist(req: str) -> str:
    return re.sub(r"[-_.]+", "-", re.split(r"[\s<>=!~;\[(@]", req.strip(), maxsplit=1)[0]).lower()


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_pyproject_python_and_extras():
    from packaging.specifiers import SpecifierSet
    proj = _pyproject()["project"]
    spec = SpecifierSet(proj["requires-python"])
    assert "3.11" in spec and "3.10" not in spec                     # D-001
    core = {_dist(d) for d in proj["dependencies"]}
    assert {"numpy", "pyyaml"} <= core
    extras = {k: {_dist(d) for d in v} for k, v in proj["optional-dependencies"].items()}
    assert {"torch", "transformers"} <= extras["reference"]          # cpu.yml installs .[reference,test]
    assert "pytest" in extras["test"]


IMPORT_TO_DIST = {"yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn"}
FIRST_PARTY = {"ag", "plots", "tools", "agctl", "handover"} | {f"lgs{i}" for i in range(1, 9)}
# what a core install must import: the harness CLI, the mock plugins the skeleton config loads lazily, the plots
HARNESS_ROOTS = ("ag.harness.run", "ag.systems.mock_system", "ag.targets.mock", "ag.judges.mock_judge", "agctl")


def _module_file(mod: str) -> Path | None:
    p = ROOT.joinpath(*mod.split("."))
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    return p / "__init__.py" if (p / "__init__.py").is_file() else None


def _module_level_imports(path: Path, mod: str) -> list[str]:
    """Dotted names imported by top-level statements (imports inside functions or try blocks are lazy/optional)."""
    pkg = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    names: list[str] = []
    for node in ast.parse(path.read_text(), str(path)).body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                parent = ".".join(base + ([node.module] if node.module else []))
            else:
                parent = node.module or ""
            names += [parent] + [f"{parent}.{a.name}" for a in node.names]   # `from pkg import submodule`
    return names


def _third_party_closure(roots: list[str]) -> dict[str, str]:
    """{third-party top-level module: first first-party module importing it} over the module-level import closure."""
    todo, seen, found = list(roots), set(), {}
    while todo:
        mod = todo.pop()
        path = _module_file(mod)
        if mod in seen or path is None:
            continue
        seen.add(mod)
        parts = mod.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]  # importing a.b.c runs a/__init__ and a/b/__init__
        for name in _module_level_imports(path, mod):
            top = name.split(".")[0]
            if top == "__future__" or top in sys.stdlib_module_names:
                continue
            if top in FIRST_PARTY:
                todo.append(name)                                    # unresolvable names (attributes) are dropped
            else:
                found.setdefault(top, mod)
    return found


def test_pyproject_core_deps_cover_module_level_imports():
    core = {_dist(d) for d in _pyproject()["project"]["dependencies"]}
    roots = list(HARNESS_ROOTS) + [f"plots.{p.stem}" for p in sorted((ROOT / "plots").glob("*.py"))]
    found = _third_party_closure(roots)
    assert {"numpy", "yaml"} <= set(found), found                    # sanity: the walk reaches the harness
    missing = {m: by for m, by in found.items() if _dist(IMPORT_TO_DIST.get(m, m)) not in core}
    assert not missing, f"imported at module level but not in [project].dependencies: {missing}"


# ---- CI workflow and the local gate -------------------------------------------------------------------------------

def _workflow() -> dict:
    return yaml.safe_load(CPU_YML.read_text())


def test_cpu_workflow():
    wf = _workflow()
    on = wf.get("on", wf.get(True))                                  # YAML 1.1 reads the bare key `on` as True
    assert on["push"] == {"branches": ["main"]} and "pull_request" in on      # PR branches run once (pull_request)
    group = wf["concurrency"]["group"]
    assert "github.sha" in group and "github.event.pull_request.number" in group   # main: one group per commit
    assert wf["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    (job,) = wf["jobs"].values()
    assert job["runs-on"] == "ubuntu-latest" and 0 < job["timeout-minutes"] <= 40
    assert {job["env"][k] for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")} == {"1"}
    assert job["env"]["TOKENIZERS_PARALLELISM"] == "false"
    steps = job["steps"]
    uses = [s["uses"] for s in steps if "uses" in s]
    assert all(re.fullmatch(r"actions/[\w-]+@v\d+", u) for u in uses) and len(uses) == 2
    (setup,) = [s for s in steps if s.get("uses", "").startswith("actions/setup-python@")]
    assert str(setup["with"]["python-version"]) == "3.11" and setup["with"]["cache"] == "pip"
    runs = [ln.strip() for s in steps for ln in s.get("run", "").splitlines() if ln.strip()]
    for cmd in ("pip install torch --index-url https://download.pytorch.org/whl/cpu", "pip install -r requirements.txt",
                'pip install -e ".[reference,test]"', "python tools/check_secrets.py",
                "python tools/check_test_classification.py", *GATES):
        assert cmd in runs, cmd
    order = [runs.index(c) for c in ("python tools/check_secrets.py", "python tools/check_test_classification.py",
                                     *GATES)]
    assert order == sorted(order)


def test_gate_commands_match_claude_md():
    text = (ROOT / "CLAUDE.md").read_text()
    for cmd in GATES:
        assert f"`{cmd}`" in text, f"CLAUDE.md 3.4 no longer names `{cmd}`; update cpu.yml and pr_check.sh"


def test_pr_check_is_executable_and_parses():
    mode = PR_CHECK.stat().st_mode
    assert mode & stat.S_IXUSR and os.access(PR_CHECK, os.X_OK)
    text = PR_CHECK.read_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    for frag in ("tools/check_secrets.py", "tools/check_test_classification.py", "--select F,E9",
                 "-m pytest tests -x -q", "-m pytest $(cat tests/manifests/cpu.txt)"):
        assert frag in text, frag
    if shutil.which("bash"):
        r = subprocess.run(["bash", "-n", str(PR_CHECK)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def _stub(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_pr_check_runs_every_step_and_reports(tmp_path):
    """Stub python/ruff record the calls, so the gate's flow is checked without running the suite twice."""
    log = tmp_path / "calls.log"
    py = _stub(tmp_path / "python", f"""if [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi
echo "py $*" >> "{log}"
if [ -n "$STUB_FAIL" ]; then case "$*" in *"$STUB_FAIL"*) exit 3 ;; esac; fi
exit 0
""")
    ruff = _stub(tmp_path / "ruff", f'echo "ruff $*" >> "{log}"\nexit 0\n')

    def run(**env: str) -> subprocess.CompletedProcess:
        log.write_text("")
        e = _git_env(**{"PYTHON": str(py), "RUFF": str(ruff), "STUB_FAIL": "", **env})
        return subprocess.run(["bash", str(PR_CHECK)], cwd=tmp_path, capture_output=True, text=True, env=e)

    r = run()
    calls = log.read_text().splitlines()
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls[0] == "py tools/check_secrets.py --untracked"
    assert calls[1] == "py tools/check_test_classification.py"
    assert calls[2].startswith("ruff check --select F,E9 ") and " ag " in calls[2] + " "
    assert calls[3] == "py -m pytest tests -x -q"
    manifest = (ROOT / "tests" / "manifests" / "cpu.txt").read_text().split()
    assert calls[4] == "py -m pytest " + " ".join(manifest)
    summary = r.stdout.split("---- summary ----")[1].splitlines()
    status = {ln.split()[2]: ln.split()[1] for ln in summary if ln.startswith("[pr_check] ") and len(ln.split()) > 2}
    stubbed = ("secrets", "test-classification", "ruff", "pytest-all", "pytest-cpu-manifest")
    assert {status[s] for s in stubbed} == {"PASS"}
    expect_grep = "PASS" if _in_git_checkout(ROOT) else "SKIP"
    assert status["conflict-markers"] == status["abs-home-paths"] == expect_grep
    assert r.stdout.rstrip().endswith("[pr_check] PASS")

    r = run(STUB_FAIL="check_test_classification")                  # one failing step: the rest still runs
    assert r.returncode == 1
    assert "FAIL  test-classification" in r.stdout and "(exit 3," in r.stdout
    assert len(log.read_text().splitlines()) == 5                    # pytest steps ran after the failure
    assert r.stdout.rstrip().endswith("[pr_check] FAIL")

    r = run(PYTHON=sys.executable + "-does-not-exist")               # wrong interpreter: refused before any step
    assert r.returncode == 2 and "not Python >= 3.11" in r.stderr and log.read_text() == ""


def test_check_secrets_long_line_is_scanned_in_pieces(tmp_path, monkeypatch):
    """A GB-sized file without newlines must not be read whole; a token across a piece boundary still matches."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_secrets", ROOT / "tools" / "check_secrets.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    monkeypatch.setattr(C, "LINE_PIECE", 64)
    f = tmp_path / "one_line.txt"
    f.write_text("a" * 50 + "hf_" + "A" * 34 + "b" * 500 + "\nok\nline3 ghp_" + "x" * 36 + "\n")
    assert C.scan_file(f) == [(1, "hf_token"), (3, "github_token")]
