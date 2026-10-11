"""Build plan 4.1: one mock experiment through config -> registry -> MockSystem -> events -> hit@k ->
raw/mock/<run_id>/ (with lineage.json) -> plots/fig09.py -> fig09.pdf + numbers.json, under 60 s."""
import json
import subprocess
import sys
import time

from _ag_util import ROOT, load_json, rows, run_dirs, small_exp
from ag.harness.run import main as run_main

RUN_FILES = ("lineage.json", "resolved.yaml", "cmd.txt", "events.jsonl", "outcomes_run.jsonl", "success.jsonl",
             "outcomes.jsonl", "summary.json", "checks.json")


def _rescore_with_error(run_dir, rid):
    from ag.metrics.score import score_outcomes
    rs = rows(run_dir / "outcomes_run.jsonl")
    for r in rs:
        if r["request_id"] == rid:
            r.update(error_kind="RuntimeError", generation=None)
    (run_dir / "outcomes_run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    return score_outcomes(run_dir)


def test_error_units_are_counted_and_left_out_of_s(tmp_path):
    exp = small_exp(tmp_path)
    out = tmp_path / "out"
    assert run_main(["--config", str(exp), "--out", str(out)]) == 0
    runs = run_dirs(out)
    s_before = load_json(runs["AG"] / "summary.json")
    victim = next(o["unit_id"] for o in rows(runs["AG"] / "outcomes.jsonl") if o["attack_success_b0"])
    s = _rescore_with_error(runs["AG"], victim)
    assert s["error_units"]["value"] == 1 and s["error_units"]["attack"] == 1
    assert s["_meta"]["n_error"] == 1 and s["_meta"]["n_S"] == s_before["_meta"]["n_S"] - 1
    assert s["hit@8[union]"]["den"] == s["_meta"]["n_S"] and s["asr"]["den"] == s_before["asr"]["den"] - 1


def test_missing_parent_is_reported_not_silent(tmp_path):
    exp = small_exp(tmp_path)
    out = tmp_path / "out"
    assert run_main(["--config", str(exp), "--out", str(out), "--ts", "20261011T000005"]) == 0
    runs = run_dirs(out)
    (runs["B0"] / "success.jsonl").unlink()                        # parent not scored
    (runs["AG"] / "lineage.json").write_text((runs["AG"] / "lineage.json").read_text())
    from ag.metrics.score import score_outcomes
    s = score_outcomes(runs["AG"])
    assert s["_meta"]["s_source_run_id"] is None and "hit@8[union]" not in s


def test_abstract_numbers_rejects_a_missing_metric(tmp_path):
    exp = small_exp(tmp_path, systems=["mock_b0"], selector={"k": [1, 2, 4, 16]})
    out = tmp_path / "out"
    assert run_main(["--config", str(exp), "--out", str(out)]) == 0
    b0 = run_dirs(out)["B0"]
    r = subprocess.run([sys.executable, str(ROOT / "plots" / "abstract_numbers.py"), "--out", str(out),
                        "--source", f"abs3={b0.name}", "--allow-mock"], capture_output=True, text=True)
    assert r.returncode == 1 and "hit@8[union]" in r.stderr
    assert not (out / "numbers.json").exists()


def test_index_append_is_safe_across_processes(tmp_path):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(str(tmp_path), p, 40)) for p in range(4)]
    [p.start() for p in procs]
    [p.join(60) for p in procs]
    assert all(p.exitcode == 0 for p in procs)
    idx = [json.loads(l) for l in (tmp_path / "raw" / "index.jsonl").read_text().splitlines()]
    assert len(idx) == 160 and len({r["run_id"] for r in idx}) == 160


def _append_many(root, p, n):
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from pathlib import Path
    from ag.harness.runner import _append_index
    from ag.sweep.expand import Job
    root = Path(root)
    for i in range(n):
        rid = f"E__S__T__d__abcdef01__s00r0__{p}__{i:04d}"
        d = root / "raw" / "E" / rid
        d.mkdir(parents=True, exist_ok=True)
        _append_index(root, Job(rid, "E", "S", "T", "d", "h", 0, 1, 0, p, {}), d)
        _append_index(root, Job(rid, "E", "S", "T", "d", "h", 0, 1, 0, p, {}), d)      # second call is a no-op


def test_end_to_end_under_60s(tmp_path):
    t0 = time.monotonic()
    exp = small_exp(tmp_path, data={"length_classes": ["8k"]})
    out = tmp_path / "out"
    assert run_main(["--config", str(exp), "--out", str(out)]) == 0
    runs = run_dirs(out)
    assert set(runs) == {"B0", "AG"}
    for sid, d in runs.items():
        for f in RUN_FILES:
            assert (d / f).exists(), f"{sid}: {f} missing"
        assert all(c["ok"] for c in load_json(d / "checks.json")), sid
        lin = load_json(d / "lineage.json")
        assert lin["clock"] == "virtual" and lin["finished_unix_ns"] and lin["run_id"] == d.name
        assert all(r["run_id"] == d.name for r in rows(d / "events.jsonl"))
    b0, ag = runs["B0"], runs["AG"]
    assert (b0 / "hits.jsonl").exists() and (b0 / "records" / "index.jsonl").exists()
    assert not (ag / "records").exists()
    assert load_json(ag / "lineage.json")["parent_run_ids"] == [b0.name]
    s_b0, s_ag = load_json(b0 / "summary.json"), load_json(ag / "summary.json")
    assert s_b0["hit@8[union]"]["den"] == s_b0["_meta"]["n_S"] > 0
    assert s_b0["hit@8[union]"]["value"] >= 0.9                 # concentrated mock ledger
    assert s_ag["asr"]["value"] < s_b0["asr"]["value"]           # the gate blocks successful attacks
    assert s_ag["exposed_tokens_p99"]["value"] == 0
    assert s_b0["exposed_tokens_p50"]["value"] is None and s_b0["exposed_tokens_p50"]["not_applicable"]
    o_ag = [o for o in rows(ag / "outcomes.jsonl") if o["is_attack"]]
    assert all("serving_best" in o["payload_rank"] and "union" in o["hit"] for o in o_ag)
    assert {o["judged_windows_spec"] for o in rows(ag / "outcomes.jsonl")} == {"128x2"}
    assert all(r["source"] == "mock" for r in rows(b0 / "hits.jsonl"))
    idx = [json.loads(l) for l in (out / "raw" / "index.jsonl").read_text().splitlines()]
    assert {r["run_id"] for r in idx} == {b0.name, ag.name}
    assert not (out / "raw" / "mock" / "latest.txt").exists()

    py = sys.executable
    r = subprocess.run([py, str(ROOT / "plots" / "fig09.py"), "--out", str(out), "--exp", "mock",
                        "--run-id", b0.name, "--allow-mock"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "figures" / "fig09.pdf").stat().st_size > 1000
    assert (out / "processed" / "fig09" / b0.name / "hit_at_k.csv").exists()
    r = subprocess.run([py, str(ROOT / "plots" / "abstract_numbers.py"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    nums = load_json(out / "numbers.json")
    assert sorted(nums) == [f"abs{i}" for i in range(1, 8)] and all(v["value"] is None for v in nums.values())
    assert time.monotonic() - t0 < 60
