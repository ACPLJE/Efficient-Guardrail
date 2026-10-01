"""handover.py: sizes, pack (code zip without env/s1.env, results without checkpoints, checkpoints of the base runs with the
arm symlink kept), the masked env example, and git (init + commit + remote, env/s1.env never tracked, token scan)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE_HF = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"       # assembled so this file itself never contains a token-shaped string
FAKE_HF2 = "hf_" + "Z" * 26


@pytest.fixture
def tree(tmp_path, monkeypatch):
    # a small copy of the package as LG_S1_ROOT (handover.ROOT is the real package, so we point its ENV_FILE at a fake env)
    out = tmp_path / "lg_s1_out"
    data = tmp_path / "lg_data_v2"
    for d in ("eval/v2/e2", "cost/rtx6000ada", "reports/s8", "logs", "slurm", "s4/prep/train/main",
              "runs/v2/qwen3guard_0.6b_s0", "runs/v2/qwen3guard_0.6b_s1", "runs/v2/qwen3guard_0.6b_pool_mean_s0", "runs/v2/llamaguard3_8b_s0"):
        (out / d).mkdir(parents=True)
    for d in ("sources/toxicchat", "needles", "synth/test/main", "manifests", "tok/qwen3"):
        (data / d).mkdir(parents=True)
    (out / "eval/v2/e2/lg_k32__qwen3guard_0.6b_s0.rows.jsonl").write_text('{"id": 1}\n' * 100)
    (out / "cost/rtx6000ada/x.rows.jsonl").write_text("{}\n")
    (out / "reports/s8/table4_ablation.md").write_text("| a |\n")
    (out / "logs/s8_eval_tp64_1.out").write_text("[s1] done rc=0\n")
    (out / "s4/prep/train/main/ids.npy").write_bytes(b"\0" * 4096)
    base = out / "runs/v2/qwen3guard_0.6b_s0"
    (base / "head.pt").write_bytes(b"\1" * 8192)
    (base / "head.resume.pt").write_bytes(b"\2" * 8192)
    (base / "dense_ft.pt").write_bytes(b"\3" * 8192)
    (base / "router.pt").write_bytes(b"\4" * 512)
    (base / "meta.json").write_text("{}")
    (base / "hardneg").mkdir()
    (base / "hardneg/meta.json").write_text("{}")
    s1 = out / "runs/v2/qwen3guard_0.6b_s1"
    (s1 / "head.pt").write_bytes(b"\5" * 8192)
    (s1 / "router.pt").write_bytes(b"\6" * 512)
    arm = out / "runs/v2/qwen3guard_0.6b_pool_mean_s0"
    (arm / "head.pt").symlink_to(os.path.relpath(base / "head.pt", arm))
    (arm / "router.pt").write_bytes(b"\7" * 512)
    (out / "runs/v2/llamaguard3_8b_s0/head.pt").write_bytes(b"\8" * 100)
    (data / "sources/toxicchat/revision.json").write_text("{}")
    (data / "synth/test/main/rows.jsonl.gz").write_bytes(b"\0" * 2048)
    (data / "tok/qwen3/spans.npz").write_bytes(b"\0" * 1024)
    monkeypatch.setenv("LG_ENV_FILE_NO_OVERRIDE", "1")
    monkeypatch.setenv("LG_OUT", str(out))
    monkeypatch.setenv("LG_SOURCES", str(data / "sources"))
    monkeypatch.setenv("LG_DATA_V2", str(data))
    monkeypatch.setenv("LG_HOME", str(tmp_path))
    monkeypatch.setenv("LG_PYTHON", "")
    for k in ("LG_V04_ROOT", "LG_V04_RUNS", "LG_PAPER_ROOT", "LG_EXTERNAL_CODE", "LG_EXTERNAL_DATA", "LG_DATA_V1", "LG_ROUND3_ROOT", "HF_HOME"):
        monkeypatch.delenv(k, raising=False)
    # a package copy with a fake env file (the real env/s1.env must not be read or packed)
    pkg = tmp_path / "lg_s1"
    pkg.mkdir()
    for name in ("handover.py", "requirements.txt", "README.md"):
        shutil.copy(ROOT / name, pkg / name)
    for d in ("lgs1", "lgs2", "lgs4", "lgs5", "lgs8", "env", "docs", "tests"):
        (pkg / d).mkdir()
    (pkg / "lgs1/__init__.py").write_text("")
    (pkg / "lgs1/__pycache__").mkdir()
    (pkg / "lgs1/__pycache__/x.pyc").write_bytes(b"\0")
    (pkg / "docs/.s8_project_doc.md").write_text("private")
    (pkg / "docs/S8_runbook.md").write_text("# S8\n")
    (pkg / "env/s1.env").write_text(f"export LG_USER=u\nexport TOKENIZERS_PARALLELISM=false\nexport HF_TOKEN={FAKE_HF}  # hub\n"
                                   f"# HF_TOKEN={FAKE_HF2}\nexport LG_API_KEY=secret\nexport LG_OUT=/x\n")
    (pkg / "big.rows.jsonl").write_text("{}\n")
    import handover as H
    monkeypatch.setattr(H, "ROOT", pkg)
    monkeypatch.setattr(H, "ENV_FILE", pkg / "env/s1.env")
    monkeypatch.setattr(H, "ENV_EXAMPLE", pkg / "env/s1.env.example")
    return H, pkg, out, data, tmp_path


def test_excluded_and_masked_env(tree):
    H, pkg, out, data, tmp = tree
    assert H._excluded("runs/v2/x/head.pt", H.RESULTS_EXCLUDES) and H._excluded("runs/v2/x/head.resume.pt", H.RESULTS_EXCLUDES)
    assert H._excluded("runs/v2/x/dense_ft.pt", H.RESULTS_EXCLUDES) and H._excluded("s4/prep/train/main/ids.npy", H.RESULTS_EXCLUDES)
    assert not H._excluded("runs/v2/x/router.pt", H.RESULTS_EXCLUDES) and not H._excluded("eval/v2/e2/a.rows.jsonl", H.RESULTS_EXCLUDES)
    assert H._excluded("env/s1.env", H.CODE_EXCLUDES) and H._excluded("lgs1/__pycache__/x.pyc", H.CODE_EXCLUDES)
    assert not H._excluded("env/s1.env.example", H.CODE_EXCLUDES) and not H._excluded("lgs1/paths.py", H.CODE_EXCLUDES)
    m = H.masked_env()
    assert FAKE_HF not in m and FAKE_HF2[:8] not in m and "=secret" not in m
    assert "export HF_TOKEN=  # hub" in m and "export TOKENIZERS_PARALLELISM=false" in m and "export LG_OUT=/x" in m and "export LG_API_KEY=" in m


def test_sizes_and_pack(tree, capsys):
    H, pkg, out, data, tmp = tree
    assert H.main(["sizes"]) == 0
    s = capsys.readouterr().out
    assert "code " in s and "results" in s and "checkpoints" in s and "qwen3guard_0.6b_s0, qwen3guard_0.6b_s1" in s and "pool_mean" not in s.split("runs:")[1].split("\n")[0]
    assert "eval/" in s and "s4/" not in s.split("results")[1].split("checkpoints")[0]      # s4/prep excluded from results
    dest = tmp / "handover"
    assert H.main(["pack", "--what", "code,results,checkpoints,data", "--dest", str(dest)]) == 0
    s = capsys.readouterr().out
    assert "HANDOVER.md" in s
    # code: zip, no env/s1.env, no caches, example present
    z = zipfile.ZipFile(dest / "code.zip")
    names = z.namelist()
    assert "lg_s1/env/s1.env.example" in names and "lg_s1/handover.py" in names and "lg_s1/docs/S8_runbook.md" in names
    assert "lg_s1/env/s1.env" not in names and not any("__pycache__" in n for n in names) and "lg_s1/docs/.s8_project_doc.md" not in names
    assert "lg_s1/big.rows.jsonl" not in names
    assert FAKE_HF not in z.read("lg_s1/env/s1.env.example").decode()
    # results: rows, reports, router.pt, meta.json; no head.pt / dense_ft.pt / resume / s4/prep
    t = tarfile.open(dest / "results.tar.gz")
    rn = t.getnames()
    assert "lg_s1_out/eval/v2/e2/lg_k32__qwen3guard_0.6b_s0.rows.jsonl" in rn and "lg_s1_out/runs/v2/qwen3guard_0.6b_s0/router.pt" in rn
    assert "lg_s1_out/runs/v2/qwen3guard_0.6b_s0/meta.json" in rn and "lg_s1_out/runs/v2/qwen3guard_0.6b_s0/hardneg/meta.json" in rn
    assert not any(n.endswith("head.pt") or n.endswith("dense_ft.pt") or ".resume." in n or "/s4/prep" in n for n in rn)
    # checkpoints: base runs only (main), symlink arm not included; resume excluded
    t = tarfile.open(dest / "checkpoints.tar.gz")
    cn = t.getnames()
    assert "qwen3guard_0.6b_s0/head.pt" in cn and "qwen3guard_0.6b_s0/dense_ft.pt" in cn and "qwen3guard_0.6b_s1/head.pt" in cn and "llamaguard3_8b_s0/head.pt" in cn
    assert not any(n.startswith("qwen3guard_0.6b_pool_mean_s0") for n in cn) and "qwen3guard_0.6b_s0/head.resume.pt" not in cn
    # data
    t = tarfile.open(dest / "data.tar.gz")
    assert "lg_data_v2/synth/test/main/rows.jsonl.gz" in t.getnames() and "lg_data_v2/sources/toxicchat/revision.json" in t.getnames()
    for n in ("code", "results", "checkpoints", "data"):
        assert (dest / f"{n}.sha256").exists() and (dest / f"{n}.files.txt").stat().st_size > 0
    md = (dest / "HANDOVER.md").read_text()
    assert "Qwen/Qwen3Guard-Gen-0.6B" in md and "checkpoints.tar.gz" in md and "sha256sum -c" in md and "python -m pytest tests -x -q" in md
    # --runs all: the arm dir with its relative symlink
    assert H.main(["pack", "--what", "checkpoints", "--runs", "all", "--dest", str(dest), "--format", "tar"]) == 0
    t = tarfile.open(dest / "checkpoints.tar.gz")
    m = t.getmember("qwen3guard_0.6b_pool_mean_s0/head.pt")
    assert m.issym() and m.linkname == "../qwen3guard_0.6b_s0/head.pt"
    capsys.readouterr()
    assert H.main(["pack", "--what", "code", "--dest", str(dest), "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out


def test_git(tree, capsys):
    H, pkg, out, data, tmp = tree
    if shutil.which("git") is None:
        pytest.skip("no git")
    assert H.main(["git", "--remote", "https://example.invalid/x.git", "--name", "t", "--email", "t@x"]) == 0
    s = capsys.readouterr().out
    assert "committed" in s and "origin = https://example.invalid/x.git" in s and "not pushed" in s
    tracked = subprocess.run(["git", "ls-files"], cwd=pkg, capture_output=True, text=True).stdout.split()
    assert "env/s1.env.example" in tracked and "handover.py" in tracked and "env/s1.env" not in tracked
    assert "big.rows.jsonl" not in tracked and "docs/.s8_project_doc.md" not in tracked
    assert (pkg / ".gitignore").read_text().count("env/s1.env\n") == 1
    # second run: nothing to commit, remote re-set; a staged token aborts
    assert H.main(["git", "--remote", "https://example.invalid/y.git", "--name", "t", "--email", "t@x"]) == 0
    assert "nothing new to commit" in capsys.readouterr().out
    (pkg / "notes.txt").write_text(f"token {FAKE_HF}\n")
    with pytest.raises(SystemExit):
        H.main(["git", "--remote", "https://example.invalid/y.git", "--name", "t", "--email", "t@x"])
