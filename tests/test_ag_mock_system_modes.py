"""MockSystem flag combinations on the harness (SPEC 3.4, 6.2 identity 3, 9.1)."""
from pathlib import Path

import pytest

from _ag_util import load_json, rows, run_dirs, small_exp, system_file
from ag.harness.events import EventType as E, by_request, event_from_row
from ag.harness.run import main as run_main
from ag.metrics import per_request as pr


def run_overlay(tmp_path: Path, name: str, system_id: str, override: dict, **exp_upd) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = system_file(tmp_path, name, "mock_ag.yaml", system_id, override)
    exp = small_exp(tmp_path, systems=[str(p.parent / "mock_b0.yaml"), str(p)], **exp_upd)
    out = tmp_path / "out"
    assert run_main(["--config", str(exp), "--out", str(out)]) == 0
    return run_dirs(out)


def events(run_dir: Path):
    return by_request(event_from_row(r) for r in rows(run_dir / "events.jsonl"))


def checks_ok(run_dir: Path) -> bool:
    return all(c["ok"] for c in load_json(run_dir / "checks.json"))


def test_n5_release_lag_zero_and_b1b_exposed(tmp_path):
    d = run_overlay(tmp_path, "n5", "N5", {"release_before_verdict": True, "mock": {"gate_us": 0.0}})["N5"]
    assert checks_ok(d)
    per = events(d)
    lags = [x for evs in per.values() for x in pr.release_lags(pr.times_from_events(evs))]
    assert lags and max(lags) == 0                      # identity 3: zero on the virtual clock
    exposed = [pr.exposed_tokens(pr.times_from_events(evs)) for evs in per.values()]
    assert sum(exposed) > 0                              # B1b / N5: tokens shown before the verdict
    s = load_json(d / "summary.json")
    assert s["exposed_tokens_p50"]["value"] > 0


def test_ag_has_no_exposed_tokens(tmp_path):
    d = run_overlay(tmp_path, "agx", "AGX", {})["AGX"]
    assert checks_ok(d)
    per = events(d)
    assert all(pr.exposed_tokens(pr.times_from_events(evs)) == 0 for evs in per.values())


def test_block_before_generation(tmp_path):
    d = run_overlay(tmp_path, "b1a", "B1a", {"block_before_generation": True, "selector": {"rule": "all"}})["B1a"]
    assert checks_ok(d)
    per = events(d)
    outcomes = {o["unit_id"]: o for o in rows(d / "outcomes.jsonl")}
    blocked = [rid for rid, o in outcomes.items() if o["blocked"]]
    assert blocked and all(outcomes[r]["is_attack"] or outcomes[r]["false_block"] for r in blocked)
    for rid in blocked:
        types = {e.type for e in per[rid]}
        assert E.REQUEST_BLOCKED in types and E.PREFILL_START not in types
    passed = [rid for rid, o in outcomes.items() if not o["blocked"]]
    for rid in passed:
        evs = per[rid]
        g0 = next(e for e in evs if e.type == E.CHUNK_GENERATED and e.chunk_id == 0)
        assert g0.required_jobs
        jd = next(e for e in evs if e.type == E.JOB_DONE)
        assert next(e for e in evs if e.type == E.PREFILL_START).t_ns >= jd.t_ns
    nw = rows(d / "outcomes_run.jsonl")[0]["n_eligible"]
    assert max(o["judged_windows"] for o in outcomes.values()) == nw


def test_deadline_miss_fails_closed(tmp_path):
    d = run_overlay(tmp_path, "dl", "DL", {"deadline_budget_ms": {"release": 1.0}})["DL"]
    assert checks_ok(d)
    outcomes = rows(d / "outcomes.jsonl")
    assert all(o["block_reason"] == "deadline_miss" for o in outcomes)
    s = load_json(d / "summary.json")
    assert s["deadline_miss_rate"]["value"] == 1.0
    assert s["fpr_inj"]["value"] == 0.0 and s["fpr_all"]["value"] == 1.0     # deadline misses are not FPR(inj)


def test_overlap_off_is_slower(tmp_path):
    on = run_overlay(tmp_path / "a", "on", "ON", {})["ON"]
    off = run_overlay(tmp_path / "b", "off", "OFF", {"overlap": False})["OFF"]
    e2e = lambda d: load_json(d / "summary.json")["e2e_vis_p50"]["value"]   # noqa: E731
    assert e2e(off) > e2e(on)


def test_rbv_and_bbg_rejected(tmp_path):
    from ag.config.schema import ConfigError
    with pytest.raises(ConfigError):
        run_overlay(tmp_path, "bad", "BAD", {"release_before_verdict": True, "block_before_generation": True})
