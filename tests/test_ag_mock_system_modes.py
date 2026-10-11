"""MockSystem flag combinations on the harness (SPEC 3.4, 6.2 identity 3, 9.1)."""
import json
from pathlib import Path

import pytest

from _ag_util import load_json, rows, run_dirs, small_exp, system_file
from ag.harness.events import EventType as E, by_request, event_from_row
from ag.harness.run import main as run_main
from ag.metrics import per_request as pr
from ag.metrics.score import score_outcomes


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
    d = run_overlay(tmp_path, "b1a", "B1a", {"block_before_generation": True, "selector": {"rule": "all"},
                                             "mock": {"judge": {"false_positive_rate": 0.0}}})["B1a"]
    assert checks_ok(d)
    per = events(d)
    outcomes = {o["unit_id"]: o for o in rows(d / "outcomes.jsonl")}
    blocked = [rid for rid, o in outcomes.items() if o["blocked"]]
    assert blocked and all(outcomes[r]["is_attack"] and outcomes[r]["block_reason"] == "inj_window" for r in blocked)
    assert not any(o["false_block"] for o in outcomes.values())
    assert {r for r, o in outcomes.items() if o["is_attack"]} == set(blocked)
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


def test_block_before_generation_passed_requests_keep_selection(tmp_path):
    # tau so high that nothing is blocked: chunk 0 must still report the pre-generation selection
    d = run_overlay(tmp_path, "b1a", "B1a", {"block_before_generation": True, "selector": {"rule": "all"},
                                             "thresholds": {"MJ": {"F5": 0.99999}}})["B1a"]
    assert checks_ok(d)
    for o in rows(d / "outcomes.jsonl"):
        assert not o["blocked"] and len(o["selected"]["0"]) == o["judged_windows"] > 0
        if o["is_attack"]:
            assert o["hit_live"] == {"8": True}


def test_block_before_generation_counts_cache_hits(tmp_path):
    d = run_overlay(tmp_path, "b1g", "B1G", {"block_before_generation": True, "selector": {"rule": "all"},
                                             "verdict_cache": "global",
                                             "thresholds": {"MJ": {"F5": 0.99999}}})["B1G"]
    assert checks_ok(d)
    runrows = rows(d / "outcomes_run.jsonl")
    assert all(r["judged_windows"] == r["n_eligible"] for r in runrows)      # cache hits count as judged
    assert sum(r["judged_windows_uncached"] for r in runrows) < sum(r["judged_windows"] for r in runrows)


def test_block_before_generation_rejects_ledger_rules(tmp_path):
    from ag.config.schema import ConfigError
    with pytest.raises(ConfigError, match="before prefill"):
        run_overlay(tmp_path, "bad", "BAD", {"block_before_generation": True})


def test_rbv_stops_judging_once_unsafe_is_known(tmp_path):
    d = run_overlay(tmp_path, "n5", "N5", {"release_before_verdict": True})["N5"]
    assert checks_ok(d)
    for rid, evs in events(d).items():
        unsafe = [e.t_ns for e in evs if e.type == E.JOB_DONE and any(v.unsafe for v in e.verdicts)]
        if unsafe:
            assert not [e for e in evs if e.type == E.JOB_SUBMITTED and e.t_ns > min(unsafe)], rid


@pytest.mark.parametrize("cache", ["off", "per_request"])
def test_late_unsafe_verdict_counts_for_fpr_inj(tmp_path, cache):
    # B1b with a fast target: every chunk is out before the verdict; false positives still count.
    # With the cache off no chunk carries cached_unsafe_windows, so only the JOB_DONE rule can add the reason.
    d = run_overlay(tmp_path, "b1b", "B1b", {"release_before_verdict": True, "verdict_cache": cache,
                                             "mock": {"tpot_ms": 0.1, "judge": {"false_positive_rate": 0.15}}})["B1b"]
    assert checks_ok(d)
    per = events(d)
    if cache == "off":
        assert not any(e.payload.get("cached_windows") for evs in per.values() for e in evs
                       if e.type == E.CHUNK_GENERATED)
    outcomes = {o["unit_id"]: o for o in rows(d / "outcomes.jsonl")}
    flagged = [rid for rid, evs in per.items()
               if any(e.type == E.JOB_DONE and any(v.unsafe for v in e.verdicts) for e in evs)]
    benign_flagged = [r for r in flagged if not outcomes[r]["is_attack"]]
    assert benign_flagged
    assert all("inj_window" in outcomes[r]["block_reasons"] for r in flagged)
    assert load_json(d / "summary.json")["fpr_inj"]["num"] == len(benign_flagged)


def _rewrite(path: Path, rs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rs))


def test_blocked_units_have_zero_utility(tmp_path):
    # a benign unit whose chunk 0 (with ANSWER) was delivered and a later chunk blocked
    d = run_overlay(tmp_path, "agu", "AGU", {"mock": {"judge": {"false_positive_rate": 0.0}}})["AGU"]
    rs = rows(d / "outcomes_run.jsonl")
    r = next(x for x in rs if not x["is_attack"] and not x["blocked"])
    gen = r["generation"]
    gen.update(blocked=True, block_reason="inj_window", text_delivered=" ".join(gen["text_generated"].split()[:8]))
    r.update(blocked=True, block_reason="inj_window", block_reasons=["inj_window"])
    _rewrite(d / "outcomes_run.jsonl", rs)
    score_outcomes(d)
    o = next(x for x in rows(d / "outcomes.jsonl") if x["unit_id"] == r["request_id"])
    succ = next(x for x in rows(d / "success.jsonl") if x["unit_id"] == r["request_id"])
    assert o["blocked"] and o["utility"] is False and succ["utility"] is True     # raw judge verdict kept
    s = load_json(d / "summary.json")
    assert s["utility"]["num"] == sum(1 for x in rows(d / "outcomes.jsonl") if not x["is_attack"] and x["utility"])


def test_request_blocked_after_cache_hits_counts_them(tmp_path):
    d = run_overlay(tmp_path, "b1g", "B1G", {"block_before_generation": True, "selector": {"rule": "all"},
                                             "verdict_cache": "global",
                                             "mock": {"judge": {"false_positive_rate": 0.15}}})["B1G"]
    assert checks_ok(d)
    runrows = rows(d / "outcomes_run.jsonl")
    assert any(r["blocked"] and not r["is_attack"] and r["judged_windows_uncached"] < r["judged_windows"]
               for r in runrows)                                     # REQUEST_BLOCKED on cache hits happens
    assert all(r["judged_windows"] == r["n_eligible"] for r in runrows)


def test_rbv_fpr_does_not_depend_on_cache_policy(tmp_path):
    nums = {}
    for pol in ("per_request", "global"):
        d = run_overlay(tmp_path / pol, "b1b", "B1b", {"release_before_verdict": True, "verdict_cache": pol,
                                                       "mock": {"tpot_ms": 0.1, "judge": {"false_positive_rate": 0.05}}},
                        data={"n_attack": 16, "n_benign": 16})["B1b"]
        assert checks_ok(d)
        nums[pol] = load_json(d / "summary.json")["fpr_inj"]["num"]
    assert nums["per_request"] == nums["global"] > 0


def test_judged_windows_spec_follows_the_judged_grid(tmp_path):
    # the mock judges the target's own split_tokens x 2 grid; the spec reported is that grid
    runs = run_overlay(tmp_path / "a", "w64", "W64", {}, record={"split_tokens": 64})
    assert {o["judged_windows_spec"] for o in rows(runs["W64"] / "outcomes.jsonl")} == {"64x2"}
    assert {o["judged_windows_spec"] for o in rows(runs["B0"] / "outcomes.jsonl")} == {""}
    with pytest.raises(ValueError, match="judge_windows"):         # a different spec is not silently relabelled
        run_overlay(tmp_path / "b", "w256", "W256", {"judge_windows": ["256x2"]})


def test_record_split_tokens_reaches_the_target(tmp_path):
    runs = run_overlay(tmp_path, "s64", "S64", {}, record={"split_tokens": 64})
    for sid, d in runs.items():
        assert checks_ok(d), sid
        assert all(o["error_kind"] is None for o in rows(d / "outcomes.jsonl")), sid
    idx = rows(runs["B0"] / "records" / "index.jsonl")
    assert idx and all(r["n_blocks"] == -(-r["prompt_len"] // 64) for r in idx)


def test_target_only_split_tokens_reaches_the_records(tmp_path):
    for name in ("mock_b0", "mock_ag"):
        system_file(tmp_path, f"{name}_64", f"{name}.yaml", "B0" if name == "mock_b0" else "AG",
                    {"mock": {"target": {"split_tokens": 64}}})
    sysdir = tmp_path / "systems"
    exp = small_exp(tmp_path, systems=[str(sysdir / "mock_b0_64.yaml"), str(sysdir / "mock_ag_64.yaml")])
    assert run_main(["--config", str(exp), "--out", str(tmp_path / "out")]) == 0
    runs = run_dirs(tmp_path / "out")
    for sid, d in runs.items():
        assert checks_ok(d), sid
        assert all(o["error_kind"] is None for o in rows(d / "outcomes.jsonl")), sid
    idx = rows(runs["B0"] / "records" / "index.jsonl")
    assert all(r["n_blocks"] == -(-r["prompt_len"] // 64) for r in idx)
    assert {h["variant"]["split_tokens"] for h in rows(runs["B0"] / "hits.jsonl")} == {64}
    att = [o for o in rows(runs["AG"] / "outcomes.jsonl") if o["is_attack"]]
    assert att and all(o["hit"].get("union") for o in att)          # AG reads the parent's 64-split hits


def test_cache_only_requests_keep_their_window_spec(tmp_path):
    from ag.data.types import Verdict
    from ag.harness.runner import outcome_run_row
    from ag.judges.base import cache_key
    from ag.data import window_map as wm
    s, req = _mock_ag(tmp_path, verdict_cache="global", block_before_generation=True, selector={"rule": "all"})
    rp = s.target.render(req)
    eligible = [int(w) for w in range(len(wm.eligible_windows(rp.block_untrusted_array())))
                if wm.eligible_windows(rp.block_untrusted_array())[w]]
    for x in s._judge_inputs(req, rp, None, eligible, None, None):          # every window already judged safe
        s.cache.put("other", cache_key(s.judge, x),
                    Verdict("MJ", "other", None, x.window_id, "128x2", 0.1, 0.5, "F5", False, 5, 1, cached=True))
    row = outcome_run_row(req, list(s.run(req)))
    assert row["judged_windows_uncached"] == 0 and row["judged_windows"] == len(eligible) > 0
    assert row["judged_windows_spec"] == "128x2"


def test_split_tokens_conflict_fails_at_setup(tmp_path):
    with pytest.raises(ValueError, match="split_tokens"):
        p = system_file(tmp_path, "b0s", "mock_b0.yaml", "B0S", {"mock": {"target": {"split_tokens": 64}}})
        exp = small_exp(tmp_path, systems=[str(p)], record={"split_tokens": 128})
        run_main(["--config", str(exp), "--out", str(tmp_path / "out")])


def test_partial_record_layers_scored_with_target_layers(tmp_path):
    # B0 records only tracked8 of a 32-layer mock target; scoring resolves tracked8 over the target
    from ag.data import window_map as wm
    from ag.reference.record import RecordStore
    from ag.runtime.chunking import standard_chunk
    from ag.selector.select import SelectorConfig, chunk_block_mass, payload_rank
    p = system_file(tmp_path, "b0t", "mock_b0.yaml", "B0T",
                    {"mock": {"target": {"n_layers": 32, "concentration": 0.05}}})     # noisy: ranks depend on layers
    exp = small_exp(tmp_path, systems=[str(p)], record={"layers": "tracked8"}, selector={"layers": ["tracked8"]})
    assert run_main(["--config", str(exp), "--out", str(tmp_path / "out")]) == 0
    d = run_dirs(tmp_path / "out")["B0T"]
    tracked = [16, 18, 20, 22, 25, 27, 29, 31]
    assert rows(d / "records" / "index.jsonl")[0]["layers"] == tracked
    hits = rows(d / "hits.jsonl")
    payload = {r["request_id"]: r["payload_windows"] for r in rows(d / "outcomes_run.jsonl")}
    checked = 0
    for rec in RecordStore(d / "records").iter():
        if not payload[rec.request_id]:
            continue
        el = wm.eligible_windows(rec.block_untrusted)
        for cd in ("first8", "first32", "all"):     # tracked8 of 32 layers = every recorded layer
            m = chunk_block_mass(rec, standard_chunk(cd, rec.n_tokens), SelectorConfig(layers=tuple(tracked)))
            want = payload_rank(wm.window_scores_from_blocks(m, el), payload[rec.request_id])
            got = next(h["payload_rank"] for h in hits if h["request_id"] == rec.request_id and h["chunk_def"] == cd)
            assert got == want, (rec.request_id, cd)
            checked += 1
    assert checked >= 6


def test_per_site_hits_reach_outcomes(tmp_path):
    runs = run_overlay(tmp_path, "agp", "AGP", {})
    b0, ag = runs["B0"], runs["AGP"]
    rs = rows(b0 / "outcomes_run.jsonl")
    for r in rs:
        if r["payload_windows"]:
            r["payload_windows_per_site"] = [r["payload_windows"], [0]]       # a second, far-away site
    _rewrite(b0 / "outcomes_run.jsonl", rs)
    score_outcomes(b0)
    score_outcomes(ag)
    for d in (b0, ag):
        att = [o for o in rows(d / "outcomes.jsonl") if o["is_attack"]]
        assert att and all({"union_any", "union_all"} <= set(o["hit"]) for o in att)
    assert "hit@8[union_any]" in load_json(b0 / "summary.json")


def _mock_ag(tmp_path, emit=lambda e: None, clock=None, length="1k", **flag_upd):
    """A set-up mock AG MockSystem outside the runner, and one attack request."""
    import logging
    from ag.config.schema import load_system
    from ag.data import mock_lc_inject
    from ag.harness.context import RunContext, VirtualClock
    from ag.harness.lineage import build_lineage
    from ag.systems.base import SystemConfig
    from ag.systems.mock_system import MockSystem
    sysd = load_system("mock_ag")
    flags = {**sysd["flags"], **flag_upd}
    lin = build_lineage("r", "mock", "c", clock="virtual", seed=0, producer="mock", model_sha={}, data_manifest_sha="")
    ctx = RunContext("r", "mock", tmp_path, clock or VirtualClock(), 0, lin, None, emit, logging.getLogger("t"))
    s = MockSystem()
    s.setup(SystemConfig("AG", "mock", flags, exp={"operating_point": "F5", "fairness": {"max_new_tokens": 48}}), ctx)
    return s, mock_lc_inject.generate(n_pairs=1, length_classes=[length])[0]


def test_global_cache_hits_are_restamped(tmp_path):
    from ag.data.types import Verdict
    from ag.judges.base import cache_key
    from ag.systems.mock_system import _Timeline
    s, req = _mock_ag(tmp_path, verdict_cache="global")
    rp = s.target.render(req)
    w = rp.payload_windows[0]
    x = s._judge_inputs(req, rp, 3, [w], None, None)[0]
    s.cache.put("other", cache_key(s.judge, x),
                Verdict("MJ", "other/request", 7, 99, "128x2", 0.95, 0.5, "F5", True, 5, 1, cached=True))
    job, done, cached_ns, vs, _ = s._judge(_Timeline("r", req.request_id, "virtual"), req, rp, 3, [w], 100, 0,
                                           {"attempt": 0}, deadline_ns=None)
    assert job is None and done is None and cached_ns == 5
    assert (vs[0].request_id, vs[0].chunk_id, vs[0].window_id, vs[0].cached) == (req.request_id, 3, w, True)


def test_concurrent_runs_keep_job_ids_unique_and_timing_sequential(tmp_path):
    # SPEC 4.2: System.run is thread-safe; openai_front drives one system from many handler threads.
    # verdict_cache off + 4k prompts: every chunk submits its own job, so a shared counter would collide.
    import dataclasses
    import threading
    from ag.data import mock_lc_inject
    from ag.harness.events import by_request
    from ag.metrics import per_request as pr

    base = mock_lc_inject.generate(n_pairs=3, length_classes=["4k"], seed=1)
    reqs = [dataclasses.replace(r, request_id=f"{r.request_id}#{i}") for i, r in enumerate(base * 8)]

    def drive(threads: int) -> dict:
        got, lock = [], threading.Lock()
        s, _ = _mock_ag(tmp_path, length="4k", verdict_cache="off")

        def work(part):
            for r in part:
                evs = list(s.run(r))
                with lock:
                    got.extend(evs)
        ths = [threading.Thread(target=work, args=(reqs[i::threads],)) for i in range(threads)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        return by_request(got)

    seq, par = drive(1), drive(8)
    assert set(seq) == set(par) == {r.request_id for r in reqs}
    benign = {r.request_id for r in reqs if not r.is_attack}
    for rid, evs in par.items():
        ids = [e.job_id for e in evs if e.type == E.JOB_SUBMITTED]
        assert len(ids) == len(set(ids)), rid
        assert rid not in benign or len(ids) > 1, rid          # benign requests pass every chunk: one job each
        assert sorted(ids) == sorted(e.job_id for e in seq[rid] if e.type == E.JOB_SUBMITTED), rid
        a, b = pr.times_from_events(seq[rid]), pr.times_from_events(evs)
        assert pr.gate_delays(a) == pr.gate_delays(b) and pr.exposed_tokens(b) == 0, rid


def test_oracle_live_run_passes_checks(tmp_path):
    d = run_overlay(tmp_path, "orc", "ORC", {"selector": {"rule": "oracle"}, "evaluation_only": True})["ORC"]
    assert checks_ok(d)
    outcomes = rows(d / "outcomes.jsonl")
    assert all(o["blocked"] for o in outcomes if o["is_attack"])
    assert all(not o["blocked"] and o["judged_windows"] == 0 for o in outcomes if not o["is_attack"])


def test_evaluation_only_never_qualifies_for_latest():
    from ag.harness.runner import latest_eligible
    clean = {"clock": "monotonic", "code_dirty": False}
    assert latest_eligible(clean, "sglang-0.5.21", True, {})
    assert not latest_eligible(clean, "sglang-0.5.21", True, {"evaluation_only": True})
    assert not latest_eligible(clean, "sglang-0.5.21", True, {}, partial=True)  # --limit / incomplete runs never qualify


def test_outcome_row_written_after_finished_is_on_disk(tmp_path, monkeypatch):
    import ag.harness.runner as R
    seen = []
    real = R.outcome_run_row

    def spy(req, evs):
        d = next((tmp_path / "out" / "raw" / "mock").glob("*__B0__*"))
        on_disk = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
        seen.append(any(e["request_id"] == req.request_id and e["type"] == "FINISHED" for e in on_disk))
        return real(req, evs)
    monkeypatch.setattr(R, "outcome_run_row", spy)
    exp = small_exp(tmp_path, systems=["mock_b0"])
    assert run_main(["--config", str(exp), "--out", str(tmp_path / "out")]) == 0
    assert seen and all(seen)


def test_record_must_cover_selector_layers_before_generation(tmp_path):
    exp = small_exp(tmp_path, systems=["mock_b0"], record={"layers": "tracked8"},
                    selector={"layers": ["upper_half", "tracked8"]})
    p = system_file(tmp_path, "b0t", "mock_b0.yaml", "B0T", {"mock": {"target": {"n_layers": 32}}})
    exp.write_text(exp.read_text().replace("- mock_b0", f"- {p}"))
    with pytest.raises(ValueError, match="does not cover"):
        run_main(["--config", str(exp), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out" / "raw" / "mock").exists() or \
        not list((tmp_path / "out" / "raw" / "mock").glob("*/events.jsonl"))       # refused before any run


def test_deadline_miss_fails_closed(tmp_path):
    d = run_overlay(tmp_path, "dl", "DL", {"deadline_budget_ms": {"release": 1.0},
                                           "mock": {"judge": {"false_positive_rate": 0.0}}})["DL"]
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


def test_blocked_units_without_a_verdict_keep_utility_none(tmp_path):
    """Pending judge: a blocked unit's utility stays None (unlabelled) instead of 0, so the utility denominator is
    not made of blocked units only; with a real judge blocked units still get 0 (test above)."""
    from ag.metrics.score import _blocked_utility
    assert _blocked_utility(None, True) is None and _blocked_utility(True, True) is False
    assert _blocked_utility(True, False) is True
    d = run_overlay(tmp_path, "agp", "AGP", {"mock": {"judge": {"false_positive_rate": 0.5}}},
                    success_judge="pending")["AGP"]
    outs = rows(d / "outcomes.jsonl")
    assert any(o["blocked"] for o in outs) and all(o["utility"] is None for o in outs)
    s = load_json(d / "summary.json")
    assert s["utility"]["den"] == 0 and s["utility"]["n_unlabelled"] == sum(1 for o in outs if not o["is_attack"])
