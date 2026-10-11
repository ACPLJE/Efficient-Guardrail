"""AG asset registry + login-node downloader + agctl assets (build plan 4.2 S0). No network: fetchers are fakes."""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _ag_util import ROOT

sys.path.insert(0, str(ROOT))
import agctl  # noqa: E402
import lgs1.download as lgd  # noqa: E402
from ag.data import download_ag as D  # noqa: E402
from ag.data import registry_ag as R  # noqa: E402
from lgs1.registry import Entry  # noqa: E402

T1, T2, J1 = "t1_llama31_8b_instruct", "t2_qwen3_4b_instruct_2507", "j1_prompt_guard2_86m"


READ_ENV_FILE = D.read_env_file          # the real parser (the autouse fixture below stubs D.read_env_file)


@pytest.fixture(autouse=True)
def _no_env_file(monkeypatch):
    """main() merges env/ag.env into os.environ; on a checkout that has one (the cluster) that would leak AG_*/HF_*
    into later tests. Tests that need the merge patch read_env_file themselves."""
    monkeypatch.setattr(D, "read_env_file", lambda path=None, keep_set=False: {})
SHA = "abc123def4567890aaaa"


# ----------------------------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------------------------
def test_registry_consistent():
    entries = R.MODELS + R.DATASETS + R.BASELINES
    assert R.validate() == []
    names = [e.name for e in entries]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", n) for n in names)
    assert {e.kind for e in entries} <= set(R.KINDS) and {e.stage for e in entries} <= set(R.STAGES)
    assert all(isinstance(e, Entry) and isinstance(e, R.AGEntry) for e in entries)
    assert all(e.url.startswith("https://github.com/") for e in entries if e.kind == "github")
    assert all("/" in e.repo_id for e in entries if e.kind.startswith("hf_"))


def test_validate_catches_problems():
    bad = [R.AGEntry("a", "hf_model", "org/x"), R.AGEntry("a", "ftp", stage="soon"), R.AGEntry("Bad-Name", "github",
                                                                                               url="http://x")]
    probs = "\n".join(R.validate(bad))
    for frag in ("a: duplicate name", "unknown kind 'ftp'", "unknown stage 'soon'", "Bad-Name: name is not snake_case",
                 "Bad-Name: github needs"):
        assert frag in probs


def test_gated_llama_family():
    reg = R.all_entries()
    assert reg[T1].repo_id == "meta-llama/Llama-3.1-8B-Instruct" and reg[T1].gated
    assert reg[J1].repo_id == "meta-llama/Llama-Prompt-Guard-2-86M" and reg[J1].gated
    assert all(e.gated for e in reg.values() if e.repo_id.startswith("meta-llama/"))
    assert not reg[T2].gated


def test_pilot_stage_and_roles():
    pilot = {e.name: e for e in R.by_stage("pilot")}
    assert set(pilot) == {T2, "bg_wiki_en"}
    assert pilot[T2].repo_id == "Qwen/Qwen3-4B-Instruct-2507" and pilot[T2].kind == "hf_model"
    bg = pilot["bg_wiki_en"]
    assert (bg.kind, bg.repo_id, bg.config, bg.split, bg.text_field, bg.max_docs) == \
        ("hf_stream", "wikimedia/wikipedia", "20231101.en", "train", "text", 20000)
    roles = {e.ag_role: e.name for e in R.all_entries().values()}
    for r in ("T1", "T2", "T3", "T4", "J1", "J2", "J3", "J4", "grader", "harm_guard", "output_guard"):
        assert r in roles, r
    reg = R.all_entries()
    assert reg["grader_qwen3_8b"].repo_id == "Qwen/Qwen3-8B" and reg["t3_mistral_small_32_24b"].verify
    assert reg["mooncake"].kind == "github" and "FAST25-release/traces" in reg["mooncake"].note
    assert reg["agentdojo"].kind == "manual" and "agentdojo==0.1.35" in reg["agentdojo"].note
    with pytest.raises(ValueError):
        R.by_stage("soon")
    with pytest.raises(KeyError):
        R.by_name("nope")


# ----------------------------------------------------------------------------------------------
# downloader with fake fetchers
# ----------------------------------------------------------------------------------------------
@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("AG_SOURCES", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")                 # main() setdefaults it; restored after the test
    return tmp_path


def fake_hf_model(tmp: Path, weights: bool = True):
    def f(e, dst, token):
        snap = tmp / "hub" / f"models--{e.repo_id.replace('/', '--')}" / "snapshots" / SHA
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text("{}")
        if weights:
            (snap / "model.safetensors").write_bytes(b"\0" * 8)
        return {"sha": SHA, "local_path": str(snap), "n_files": 2, "bytes": 10}
    return f


def fake_github(e, dst, token):
    repo = dst / "repo"
    (repo / "data").mkdir(parents=True, exist_ok=True)
    (repo / ".git").mkdir(exist_ok=True)
    (repo / ".git" / "packed.json").write_text("{}")
    (repo / "data" / "cases.jsonl").write_text(
        json.dumps({"id": 1, "tool": "GmailReadEmail", "payload": "x" * 500}) + "\n"
        + json.dumps({"id": 2, "tool": None, "payload": "y"}) + "\n")
    (repo / "data" / "table.csv").write_text("id,score,label\n1,0.5,yes\n2,1.5,no\n")
    (repo / "README.md").write_text("# readme\n")
    (repo / "requirements.txt").write_text("numpy\n")
    return {"sha": "f00d" * 10, "local_path": str(repo), "n_files": 5, "bytes": 100}


def fake_hf_stream(e, dst, token):
    out = dst / "docs.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"id": f"{e.name}-{i}", "text": f"doc {i} " * 50, "meta": {"title": f"T{i}"}}) + "\n")
    return {"sha": "beef" * 10, "local_path": str(out), "n_files": 1, "bytes": out.stat().st_size, "n_docs": 3}


def boom(e, dst, token):
    raise AssertionError(f"fetcher must not be called for {e.name}")


@pytest.fixture()
def fakes(env, monkeypatch):
    monkeypatch.setitem(D.FETCHERS, "hf_model", fake_hf_model(env))
    monkeypatch.setitem(D.FETCHERS, "github", fake_github)
    monkeypatch.setitem(D.FETCHERS, "hf_stream", fake_hf_stream)
    monkeypatch.setitem(D.FETCHERS, "hf_dataset", boom)
    return env


ONLY = [T2, "bg_wiki_en", "injecagent", "agentdojo"]


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def test_run_writes_revision_status_snapshot_probe(fakes):
    src = fakes / "src"
    assert D.FETCHERS is not lgd.FETCHERS                         # a copy: patching it never touches LG
    res = D.run(only=ONLY, sources=src, force_online=True)
    assert res["ok"] == [T2, "bg_wiki_en", "injecagent"] and res["manual"] == ["agentdojo"] and res["failed"] == []
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["HF_DATASETS_OFFLINE"] == "1"   # restored
    for n in ONLY:
        rev = _read(src / n / "revision.json")
        assert rev["name"] == n and rev["stage"] == R.by_name(n).stage and rev["ag_role"] == R.by_name(n).ag_role
    assert _read(src / T2 / "revision.json")["sha"] == SHA and _read(src / "agentdojo" / "revision.json")["status"] == "manual"
    assert set(_read(src / "download_status.json")) == set(ONLY)
    # dated snapshot: every registry entry, gated state, counts
    snaps = list(src.glob("ag_assets_*.json"))
    assert len(snaps) == 1 and re.fullmatch(r"ag_assets_\d{4}\.json", snaps[0].name)
    snap = _read(snaps[0])
    assert {"created_at", "host", "entries", "gated", "counts"} <= set(snap)
    assert set(snap["entries"]) == set(R.all_entries())
    assert snap["entries"][T2]["status"] == "ok" and snap["entries"][T2]["sha"] == SHA
    assert snap["entries"][T2]["stage"] == "pilot" and snap["entries"][T1]["status"] == "missing"
    assert snap["gated"][T1] == "missing" and snap["gated"][J1] == "missing" and T2 not in snap["gated"]
    assert snap["counts"]["ok"] == 3 and snap["counts"]["manual"] == 1
    assert snap["counts"]["missing"] == len(R.all_entries()) - 4
    assert snap["counts"]["by_stage"]["pilot"] == {"ok": 2}
    # probe: github repo
    assert not (src / T2 / "probe.json").exists()                  # models are not probed
    pr = _read(src / "injecagent" / "probe.json")
    assert [f["path"] for f in pr["files"]] == ["data/cases.jsonl", "data/table.csv"]    # no .git, README, requirements
    assert all(f["bytes"] > 0 for f in pr["files"])
    by = {s["path"]: s for s in pr["samples"]}
    js = by["data/cases.jsonl"]
    assert js["format"] == "jsonl" and js["n_rows_read"] == 2 and js["keys"] == ["id", "tool", "payload"]
    assert js["types"]["tool"] == "NoneType|str" and len(js["example"]["payload"]) == 200
    cs = by["data/table.csv"]
    assert cs["keys"] == ["id", "score", "label"] and cs["types"] == {"id": "int", "score": "float", "label": "str"}
    assert _read(src / "injecagent" / "revision.json")["probe"].endswith("probe.json")
    # probe: streamed docs (root is a single file)
    pb = _read(src / "bg_wiki_en" / "probe.json")
    assert pb["files"] == [{"path": "docs.jsonl.gz", "bytes": (src / "bg_wiki_en" / "docs.jsonl.gz").stat().st_size}]
    assert pb["samples"][0]["format"] == "jsonl.gz" and pb["samples"][0]["keys"] == ["id", "text", "meta"]
    # refs/main repaired for the single-snapshot repo in the (tmp) HF cache
    refs = fakes / "hub" / "models--Qwen--Qwen3-4B-Instruct-2507" / "refs" / "main"
    assert refs.read_text() == SHA


def test_second_run_skips_ok_entries_and_reprobes(fakes, monkeypatch):
    src = fakes / "src"
    D.run(only=ONLY, sources=src, force_online=True)
    (src / "injecagent" / "probe.json").unlink()
    for k in ("hf_model", "github", "hf_stream"):
        monkeypatch.setitem(D.FETCHERS, k, boom)
    res = D.run(only=ONLY, sources=src, force_online=True)
    assert res["skipped"] == [T2, "bg_wiki_en", "injecagent"] and res["attempted"] == ["agentdojo"]
    assert (src / "injecagent" / "probe.json").exists()             # re-probed from the recorded local_path
    assert not res["failed"]


def test_stage_filter(fakes):
    res = D.run(stage="pilot", sources=fakes / "src", force_online=True)
    assert res["attempted"] == [T2, "bg_wiki_en"] and res["ok"] == [T2, "bg_wiki_en"]
    assert D.run(stage="soon", sources=fakes / "src", force_online=True)["error"].startswith("unknown stage")
    assert D.run(only=["nope"], sources=fakes / "src", force_online=True)["error"].startswith("unknown names")


def test_failing_gated_fetcher_recorded_with_hint(fakes, monkeypatch, capsys):
    def denied(e, dst, token):
        raise OSError(f"401 Client Error: Unauthorized for url https://huggingface.co/api/models/{e.repo_id} (gated repo)")
    monkeypatch.setitem(D.FETCHERS, "hf_model", denied)
    src = fakes / "src"
    res = D.run(only=[T1], sources=src, force_online=True)
    assert res["failed"] == [T1] and res["failed_total"] == [T1]
    rev = _read(src / T1 / "revision.json")
    assert rev["status"] == "failed" and "401" in rev["error"] and "gated: accept the licence" in rev["error"]
    snap = _read(next(src.glob("ag_assets_*.json")))
    assert snap["gated"][T1] == "failed" and snap["entries"][T1]["error"] == rev["error"]
    assert "FAILED" in capsys.readouterr().err
    assert D.main(["--only", T1, "--force-online", "--sources", str(src)]) == 1        # exit 1 on any failure


def test_failed_refetch_keeps_previous_ok_record(fakes, monkeypatch):
    src = fakes / "src"
    D.run(only=[T2], sources=src, force_online=True)

    def flaky(e, dst, token):
        raise ConnectionError("connection reset")
    monkeypatch.setitem(D.FETCHERS, "hf_model", flaky)
    res = D.run(only=[T2], sources=src, force_online=True, skip_existing=False)
    rev = _read(src / T2 / "revision.json")
    assert res["failed"] == [T2] and res["failed_total"] == []      # this run failed; the asset is still there
    assert rev["status"] == "ok" and rev["sha"] == SHA and "connection reset" in rev["refetch_error"]


def test_model_snapshot_without_weights_fails(fakes, monkeypatch):
    monkeypatch.setitem(D.FETCHERS, "hf_model", fake_hf_model(fakes, weights=False))
    res = D.run(only=[T2], sources=fakes / "src", force_online=True)
    assert res["failed"] == [T2]
    assert "no weight files" in _read(fakes / "src" / T2 / "revision.json")["error"]


def test_probe_error_never_fails_download(fakes, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(D, "probe_path", broken)
    res = D.run(only=["injecagent"], sources=fakes / "src", force_online=True)
    rev = _read(fakes / "src" / "injecagent" / "revision.json")
    assert res["ok"] == ["injecagent"] and rev["status"] == "ok" and "probe exploded" in rev["probe_error"]


def test_no_probe_flag(fakes):
    D.run(only=["injecagent"], sources=fakes / "src", force_online=True, probe=False)
    assert not (fakes / "src" / "injecagent" / "probe.json").exists()


def test_refuses_offline_and_inside_sbatch(env, monkeypatch, capsys):
    monkeypatch.setitem(D.FETCHERS, "hf_model", boom)
    monkeypatch.setattr(D, "_online_ok", lambda *a, **k: False)
    src = env / "src"
    assert D.run(only=[T2], sources=src) == {"error": "offline"}
    assert "login node" in capsys.readouterr().err and not src.exists()
    monkeypatch.setattr(D, "_online_ok", lambda *a, **k: True)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert D.run(only=[T2], sources=src) == {"error": "offline"}
    assert D.main(["--only", T2, "--sources", str(src)]) == 1


def test_sources_required(env, monkeypatch):
    assert D.run(only=[T2], force_online=True) == {"error": "no_sources"}
    monkeypatch.setenv("AG_SOURCES", str(env / "from_env"))
    monkeypatch.setitem(D.FETCHERS, "hf_model", fake_hf_model(env))
    assert D.run(only=[T2], force_online=True)["ok"] == [T2]
    assert (env / "from_env" / T2 / "revision.json").exists()


def test_list_exits_zero(capsys):
    assert D.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-4B-Instruct-2507" in out and "bg_wiki_en" in out and "gated" in out
    line = next(x for x in out.splitlines() if x.startswith(T2))
    assert "pilot" in line and "hf_model" in line
    r = subprocess.run([sys.executable, "-m", "ag.data.download_ag", "--list"], cwd=ROOT, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert T1 in r.stdout


# ----------------------------------------------------------------------------------------------
# probe readers
# ----------------------------------------------------------------------------------------------
def test_sample_file_formats_and_errors(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a": 1}\n{not json\n')
    s = D.sample_file(bad, "bad.jsonl")
    assert "line 2" in s["error"]
    lst = tmp_path / "l.json"
    lst.write_text(json.dumps([{"q": "x", "n": 1}, {"q": "y", "n": 2.0}]))
    s = D.sample_file(lst, "l.json")
    assert s["top_level"] == "list" and s["n_items"] == 2 and s["types"] == {"q": "str", "n": "float|int"}
    dct = tmp_path / "d.json"
    dct.write_text(json.dumps({"train": [1, 2, 3], "meta": {"v": 1}}))
    s = D.sample_file(dct, "d.json")
    assert s["top_level"] == "dict" and s["keys"] == ["train", "meta"] and s["list_lengths"] == {"train": 3}
    jl = tmp_path / "lines.json"                                    # JSON lines with a .json suffix
    jl.write_text('{"a": 1}\n{"a": 2}\n')
    s = D.sample_file(jl, "lines.json")
    assert s["parsed_as"] == "jsonl" and s["n_rows_read"] == 2
    y = tmp_path / "c.yaml"
    y.write_text("tasks:\n  - name: a\n  - name: b\nversion: 2\n")
    s = D.sample_file(y, "c.yaml")
    assert s["keys"] == ["tasks", "version"] and s["list_lengths"] == {"tasks": 2}


def test_sample_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    p = tmp_path / "t.parquet"
    pq.write_table(pa.table({"prompt": ["a", "b", "c"], "label": [0, 1, 0]}), p)
    s = D.sample_file(p, "t.parquet")
    assert s["keys"] == ["prompt", "label"] and s["n_rows_total"] == 3 and s["schema"]["label"] == "int64"


def test_probe_sample_order_prefers_data_dirs(tmp_path):
    for rel in ("configs/a.yaml", "data/x.jsonl", "data/y.jsonl", "data/z.csv", "traces/t.jsonl", "zz.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text('{"k": 1}\n' if not rel.endswith(".csv") else "k\n1\n")
    pr = D.probe_path(tmp_path, max_samples=4)
    assert pr["n_data_files"] == 6
    assert [s["path"] for s in pr["samples"]] == ["data/x.jsonl", "data/z.csv", "traces/t.jsonl", "data/y.jsonl"]


def test_read_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AG_TEST_UNSET_VAR", raising=False)
    monkeypatch.setenv("AG_TEST_HOME", "/h")
    f = tmp_path / "ag.env"
    f.write_text("# comment\nexport A=/x            # trailing comment\nexport B=${A}/y\n"
                 "export C=${AG_TEST_UNSET_VAR:-/def}\nD=\"$AG_TEST_HOME/q\"\n")
    assert READ_ENV_FILE(f) == {"A": "/x", "B": "/x/y", "C": "/def", "D": "/h/q"}
    assert "A" not in os.environ
    ex = READ_ENV_FILE(ROOT / "env" / "ag.env.example")
    assert ex["AG_SOURCES"] == "/home/212aig08/ag_sources"
    monkeypatch.setenv("A", "/shell")                 # keep_set: derived values follow the shell's value, as bash would
    assert READ_ENV_FILE(f, keep_set=True)["B"] == "/shell/y" and READ_ENV_FILE(f)["B"] == "/x/y"


# ----------------------------------------------------------------------------------------------
# agctl assets
# ----------------------------------------------------------------------------------------------
def test_agctl_assets_table(fakes, capsys):
    src = fakes / "src"
    D.run(only=ONLY, sources=src, force_online=True)
    capsys.readouterr()
    assert agctl.main(["assets", "--sources", str(src)]) == 0
    out = capsys.readouterr().out
    t2 = next(x for x in out.splitlines() if x.startswith(T2))
    assert t2.split()[:4] == [T2, "pilot", "ok", SHA[:12]]
    t1 = next(x for x in out.splitlines() if x.startswith(T1))
    assert t1.split() == [T1, "s2", "missing", "-", "yes"]
    assert "counts:" in out and "ok=3" in out
    assert agctl.main(["assets", "--sources", str(src), "--stage", "pilot"]) == 0
    out = capsys.readouterr().out
    assert T2 in out and T1 not in out


def test_agctl_assets_without_status_file(tmp_path, capsys):
    assert agctl.main(["assets", "--sources", str(tmp_path / "none")]) == 0
    out = capsys.readouterr().out
    assert "no " in out and "ag_download_login.sh" in out and "missing" in out


# ----------------------------------------------------------------------------------------------
# login-node shell wrapper
# ----------------------------------------------------------------------------------------------
@pytest.fixture()
def wrapper_env(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    stub = venv / "python"
    stub.write_text('#!/bin/sh\necho "$@" > "$STUB_OUT"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env_file = tmp_path / "ag.env"
    env_file.write_text(f"export AG_SOURCES={tmp_path / 'src'}\nexport AG_OUT={tmp_path / 'out'}\n"
                        f"export AG_VENV={tmp_path / 'venv'}\nexport HF_HOME={tmp_path / 'hf'}\n"
                        f"export HF_HUB_CACHE={tmp_path / 'hf' / 'hub'}\n")
    e = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    e.update(AG_ENV_FILE=str(env_file), STUB_OUT=str(tmp_path / "stub_args.txt"))
    return tmp_path, e


SCRIPT = str(ROOT / "slurm" / "ag_download_login.sh")


def test_wrapper_refuses_inside_slurm(wrapper_env):
    tmp, e = wrapper_env
    r = subprocess.run(["bash", SCRIPT, "--only", "bg_wiki_en"], env={**e, "SLURM_JOB_ID": "77"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "refusing" in r.stderr and "login node" in r.stderr
    assert not (tmp / "stub_args.txt").exists() and not (tmp / "out").exists() and not (tmp / "src").exists()


def test_wrapper_passes_args_in_background(wrapper_env):
    tmp, e = wrapper_env
    r = subprocess.run(["bash", SCRIPT, "--only", "bg_wiki_en", "--stage", "pilot"], env=e, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "started in background" in r.stdout
    args = tmp / "stub_args.txt"
    deadline = time.time() + 20
    while not (args.exists() and args.read_text().strip()) and time.time() < deadline:
        time.sleep(0.05)
    assert args.read_text().split() == ["-m", "ag.data.download_ag", "--only", "bg_wiki_en", "--stage", "pilot"]
    assert len(list((tmp / "out" / "logs").glob("ag_download_*.log"))) == 1 and (tmp / "src").is_dir()


def test_wrapper_list_runs_in_foreground(wrapper_env):
    tmp, e = wrapper_env
    r = subprocess.run(["bash", SCRIPT, "--list"], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert (tmp / "stub_args.txt").read_text().split() == ["-m", "ag.data.download_ag", "--list"]
    assert not (tmp / "out").exists()


@pytest.mark.parametrize("argv, stages", [
    ([], ["pilot", "s2"]),                       # default: the multi-10-GB T3 / T4 / later sets need --stage all
    (["--stage", "all"], None),
    (["--stage", "pilot"], ["pilot"]),
    (["--only", T2], None),                      # --only alone is not narrowed by the default stages
    (["--only", T2, "--stage", "s2"], ["s2"]),
])
def test_cli_default_stages(argv, stages, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(D, "run", lambda **kw: seen.update(kw) or {})
    assert D.main([*argv, "--sources", str(tmp_path)]) == 0
    assert seen["stage"] == stages


# ----------------------------------------------------------------------------------------------
# review fixes: staged artifacts, registry changes, oversized JSON probe, env merge
# ----------------------------------------------------------------------------------------------
def _stream_failing_after(k: int):
    """Like lgs1 fetch_hf_stream: opens dst/docs.jsonl.gz for writing first, then the stream dies at row k."""
    def f(e, dst, token):
        with gzip.open(dst / "docs.jsonl.gz", "wt", encoding="utf-8") as fh:
            for i in range(k):
                fh.write(json.dumps({"id": f"x-{i}", "text": "partial", "meta": {}}) + "\n")
            raise ConnectionError("stream reset by peer")
    return f


def _n_docs(p: Path) -> int:
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def test_failed_stream_refetch_keeps_the_previous_file(fakes, monkeypatch):
    src = fakes / "src"
    D.run(only=["bg_wiki_en"], sources=src, force_online=True)
    docs = src / "bg_wiki_en" / "docs.jsonl.gz"
    assert _n_docs(docs) == 3
    monkeypatch.setitem(D.FETCHERS, "hf_stream", _stream_failing_after(1))
    res = D.run(only=["bg_wiki_en"], sources=src, force_online=True, skip_existing=False)
    assert res["failed"] == ["bg_wiki_en"]
    assert _n_docs(docs) == 3                                     # the pilot's input was not truncated
    rev = _read(src / "bg_wiki_en" / "revision.json")
    assert rev["status"] == "ok" and rev["n_docs"] == 3 and "stream reset" in rev["refetch_error"]
    assert rev["local_path"] == str(docs) and not list((src / "bg_wiki_en").glob(".partial-*"))


def test_failed_first_stream_leaves_no_file_at_the_pilot_path(fakes, monkeypatch):
    src = fakes / "src"
    monkeypatch.setitem(D.FETCHERS, "hf_stream", _stream_failing_after(2))
    D.run(only=["bg_wiki_en"], sources=src, force_online=True)
    assert _read(src / "bg_wiki_en" / "revision.json")["status"] == "failed"
    assert not (src / "bg_wiki_en" / "docs.jsonl.gz").exists()


def test_github_refetch_is_staged(fakes, monkeypatch):
    src = fakes / "src"
    D.run(only=["injecagent"], sources=src, force_online=True)
    repo = src / "injecagent" / "repo"
    (repo / "marker").write_text("old")
    (src / "injecagent" / ".partial-1").mkdir()                    # a killed earlier run left this

    def half_clone(e, dst, token):
        (dst / "repo" / ".git").mkdir(parents=True, exist_ok=True)
        raise subprocess.CalledProcessError(128, ["git", "clone"])
    monkeypatch.setitem(D.FETCHERS, "github", half_clone)
    D.run(only=["injecagent"], sources=src, force_online=True, skip_existing=False)
    assert (repo / "marker").read_text() == "old" and (repo / "data" / "cases.jsonl").exists()
    assert not list((src / "injecagent").glob(".partial-*"))
    monkeypatch.setitem(D.FETCHERS, "github", fake_github)
    D.run(only=["injecagent"], sources=src, force_online=True, skip_existing=False)
    assert not (repo / "marker").exists() and (repo / "data" / "cases.jsonl").exists()   # fresh clone swapped in
    assert _read(src / "injecagent" / "revision.json")["local_path"] == str(repo)


def test_registry_change_forces_refetch(fakes, monkeypatch, capsys):
    src = fakes / "src"
    D.run(only=[T2], sources=src, force_online=True)
    rev_path = src / T2 / "revision.json"
    rev = _read(rev_path)
    rev["repo_id"] = "Qwen/Some-Other-Model"                       # as if the registry id was corrected since
    rev_path.write_text(json.dumps(rev))
    res = D.run(only=[T2], sources=src, force_online=True)        # default skip_existing=True
    assert res["attempted"] == [T2] and "registry changed since the last fetch (repo_id)" in capsys.readouterr().out
    assert _read(rev_path)["repo_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    rev = _read(rev_path)
    rev["max_docs"] = 7
    rev_path.write_text(json.dumps(rev))
    D.rebuild_status(src)
    snap = D.snapshot(src, json.loads((src / "download_status.json").read_text()), host="h", run_info={})
    assert snap["entries"][T2]["registry_mismatch"] is True


def test_probe_does_not_load_a_one_line_json_array(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "MAX_WHOLE_FILE", 100)
    big = tmp_path / "samples.json"
    big.write_text(json.dumps([{"id": i, "context": "c" * 50} for i in range(20)]))     # one line, > 100 bytes
    out = D.sample_file(big, "samples.json")
    assert "not parsed" in out["skipped"] and out["head"].startswith('[{"id": 0')
    lines = tmp_path / "rows.json"
    lines.write_text("\n".join(json.dumps({"id": i, "q": "x" * 30}) for i in range(10)) + "\n")
    out = D.sample_file(lines, "rows.json")
    assert out["parsed_as"] == "jsonl" and "q" in out["keys"]


def test_env_file_fills_hf_home_even_with_sources(monkeypatch, tmp_path):
    for k in ("HF_HOME", "AG_SOURCES"):          # setenv first so teardown restores the real state (main setdefaults)
        monkeypatch.setenv(k, "placeholder")
        monkeypatch.delenv(k)
    monkeypatch.setattr(D, "read_env_file", lambda path=None, keep_set=False: {"HF_HOME": "/big/disk/hf",
                                                                                "AG_SOURCES": "/x"})
    monkeypatch.setattr(D, "run", lambda **kw: {})
    assert D.main(["--stage", "pilot", "--sources", str(tmp_path)]) == 0
    assert os.environ["HF_HOME"] == "/big/disk/hf"


def _lg_like_github(e, dst, token):
    """Like lgs1 fetch_github: skips the clone when repo/ exists, then needs a complete checkout."""
    repo = dst / "repo"
    if not repo.exists():
        fake_github(e, dst, token)
    if not (repo / "README.md").exists():
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
    return {"sha": "f00d" * 10, "local_path": str(repo), "n_files": 5, "bytes": 100}


def test_first_run_half_clone_does_not_poison_later_runs(fakes, monkeypatch):
    src = fakes / "src"

    def half_clone(e, dst, token):
        (dst / "repo" / ".git").mkdir(parents=True, exist_ok=True)
        raise subprocess.CalledProcessError(128, ["git", "clone"])
    monkeypatch.setitem(D.FETCHERS, "github", half_clone)
    D.run(only=["injecagent"], sources=src, force_online=True)
    assert _read(src / "injecagent" / "revision.json")["status"] == "failed"
    assert not (src / "injecagent" / "repo").exists()               # nothing half-done at the final path
    monkeypatch.setitem(D.FETCHERS, "github", _lg_like_github)
    D.run(only=["injecagent"], sources=src, force_online=True)
    assert _read(src / "injecagent" / "revision.json")["status"] == "ok"


def test_failed_swap_restores_the_old_copy(fakes, monkeypatch):
    src = fakes / "src"
    D.run(only=["injecagent"], sources=src, force_online=True)
    real = os.replace

    def flaky(a, b):
        if Path(a).parent.name.startswith(".partial-"):
            raise OSError(16, "Device or resource busy")
        return real(a, b)
    monkeypatch.setattr(D.os, "replace", flaky)
    D.run(only=["injecagent"], sources=src, force_online=True, skip_existing=False)
    repo = src / "injecagent" / "repo"
    assert (repo / "README.md").exists() and (repo / "data" / "cases.jsonl").exists()
    assert not list((src / "injecagent").glob(".old-*")) and not list((src / "injecagent").glob(".partial-*"))


def test_changed_entry_with_failed_refetch_is_not_kept_ok(fakes, monkeypatch, capsys):
    src = fakes / "src"
    D.run(only=[T2], sources=src, force_online=True)
    rev_path = src / T2 / "revision.json"
    rev = _read(rev_path)
    rev["repo_id"] = "Qwen/Old-Id"
    rev_path.write_text(json.dumps(rev))

    def down(e, dst, token):
        raise ConnectionError("hub unreachable")
    monkeypatch.setitem(D.FETCHERS, "hf_model", down)
    res = D.run(only=[T2], sources=src, force_online=True)
    assert _read(rev_path)["status"] == "failed" and res["failed_total"] == [T2]
    assert "kept the previous ok record" not in capsys.readouterr().err


def test_records_without_newer_identity_keys_are_still_skipped(fakes, monkeypatch):
    src = fakes / "src"
    D.run(only=[T2, "injecagent", "bg_wiki_en"], sources=src, force_online=True)
    for name in (T2, "injecagent", "bg_wiki_en"):                   # LG-shaped / older records
        p = src / name / "revision.json"
        rev = _read(p)
        for k in ("url", "max_docs", "text_field", "shuffle_seed", "allow_patterns", "ignore_patterns"):
            rev.pop(k, None)
        p.write_text(json.dumps(rev))
    for kind in ("hf_model", "github", "hf_stream"):
        monkeypatch.setitem(D.FETCHERS, kind, boom)
    res = D.run(only=[T2, "injecagent", "bg_wiki_en"], sources=src, force_online=True)
    assert sorted(res["skipped"]) == sorted([T2, "injecagent", "bg_wiki_en"])


def test_probe_reads_json_lines_with_any_row_type(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "MAX_WHOLE_FILE", 100)
    p = tmp_path / "rows.json"
    p.write_text("\n" + "\n".join(json.dumps([i, "x" * 40]) for i in range(10)) + "\n")   # leading blank, array rows
    out = D.sample_file(p, "rows.json")
    assert out.get("parsed_as") == "jsonl" and out["n_rows_read"] == 10
