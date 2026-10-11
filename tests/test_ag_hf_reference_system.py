"""hf_reference system + runner wiring + pending success judge on the CPU smoke config
(configs/exp/w0_grader_smoke.yaml, tiny local HF model; build plan S2 step 0)."""
import hashlib
import importlib.util
import logging

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from _ag_util import ROOT, load_json, rows  # noqa: E402
from ag.config.schema import load_exp, load_system  # noqa: E402
from ag.data.success_judge import get_success_judge  # noqa: E402
from ag.data.types import Request, Segment  # noqa: E402
from ag.harness.context import MonotonicClock, RunContext  # noqa: E402
from ag.harness.events import EventType as E, by_request, event_from_row  # noqa: E402
from ag.harness.run import main as run_main  # noqa: E402
from ag.harness.runner import GENERATORS, PRODUCERS, build_requests  # noqa: E402
from ag.metrics import quality as q  # noqa: E402
from ag.runtime.chunking import serving_chunks  # noqa: E402
from ag.sweep.expand import expand, fairness_mismatches  # noqa: E402
from ag.systems.base import SystemConfig  # noqa: E402
from ag.systems.registry import create_system  # noqa: E402

SMOKE = ROOT / "configs" / "exp" / "w0_grader_smoke.yaml"
PILOT = ROOT / "configs" / "exp" / "w0_grader_pilot.yaml"
BG = ROOT / "tests" / "data" / "bg_docs_small.jsonl.txt"
TS = "20261011T090000"
RUN_FILES = ("lineage.json", "resolved.yaml", "cmd.txt", "events.jsonl", "outcomes_run.jsonl", "success.jsonl",
             "outcomes.jsonl", "summary.json", "checks.json", "server.log")


@pytest.fixture(scope="module", autouse=True)
def _env(tmp_path_factory):
    """AG_CACHE in tmp (the tiny model is built there, never under ~); one torch thread for the tiny model."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AG_CACHE", str(tmp_path_factory.mktemp("agcache")))
        yield
    torch.set_num_threads(n)


def _events(d):
    return [event_from_row(r) for r in rows(d / "events.jsonl")]


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """The smoke config run twice with the same --ts: first --limit 2, then the rest (resume)."""
    out = tmp_path_factory.mktemp("smoke") / "out"
    assert run_main(["--config", str(SMOKE), "--out", str(out), "--ts", TS, "--limit", "2"]) == 0
    d = next(p for p in (out / "raw" / "W0-smoke").iterdir() if p.is_dir() and p.name != "jobs")
    first = _events(d)
    assert run_main(["--config", str(SMOKE), "--out", str(out), "--ts", TS]) == 0
    return out, d, first


# ------------------------------------------------------------------------------------------------ configs
def test_configs_load_and_expand():
    for path, target, n in ((PILOT, "T2", 200), (SMOKE, "tiny-hf", 4)):
        exp = load_exp(path)
        jobs = expand(exp, ts=TS)
        assert len(jobs) == 1 and not fairness_mismatches(jobs)
        j = jobs[0].resolved
        assert j["target"] == target and j["data"]["n_attack"] == n and j["success_judge"] == "pending"
        assert j["system"]["impl"] == "hf_reference" and j["system"]["flags"]["gate"] == "none"
        assert j["system"]["flags"]["write_records"] is False and j["clock"] == "monotonic"
    pilot = load_exp(PILOT)
    assert pilot["data"]["background_paths"] == ["${AG_SOURCES}/bg_wiki_en/docs.jsonl.gz"]
    assert pilot["data"]["length_classes"] == ["8k"] and pilot["data"]["n_benign"] == 0
    assert pilot["fairness"] == {"max_new_tokens": 256, "temperature": 0.0, "ignore_eos": False, "dtype": "bfloat16"}
    assert load_system("b0_ref")["flags"]["harm_scan"] == "off"


# ------------------------------------------------------------------------------------------------ end to end
def test_smoke_run_files_lineage_checks(smoke):
    out, d, _ = smoke
    for f in RUN_FILES:
        assert (d / f).exists(), f
    assert not (d / "records").exists() and not (d / "hits.jsonl").exists()
    lin = load_json(d / "lineage.json")
    import transformers
    assert lin["producer"] == "reference" and lin["engine_version"] == f"transformers-{transformers.__version__}"
    assert lin["model_sha"]["target"].startswith("local:") and lin["placement"] == {"target": "cpu"}
    assert lin["clock"] == "monotonic" and lin["finished_unix_ns"] and lin["run_id"] == d.name
    bad = [c for c in load_json(d / "checks.json") if not c["ok"]]
    assert not bad, bad
    assert [r["run_id"] for r in rows(out / "raw" / "index.jsonl")] == [d.name]


def test_smoke_events_tokens_text(smoke):
    _, d, _ = smoke
    per = by_request(_events(d))
    assert len(per) == 6
    for rid, evs in per.items():
        fin = next(e for e in evs if e.type == E.FINISHED)
        gen = fin.generation
        toks = [e for e in evs if e.type == E.TOKENS]
        assert len(toks) == len(gen.output_ids) == fin.payload["perf"]["n_out"] > 0
        assert [e.tok_range for e in toks] == [(i, i + 1) for i in range(len(toks))]
        assert sum(e.type == E.CHUNK_DELIVERED for e in evs) == len(toks)
        assert "".join(e.payload["text"] for e in toks) == gen.text_delivered == gen.text_generated
        t = {e.type: e.t_ns for e in reversed(evs)}         # first occurrence of each type
        assert t[E.ARRIVED] <= t[E.ADMITTED] <= t[E.PREFILL_START] <= t[E.PREFILL_DONE] == toks[0].t_ns
        assert {e.t_source for e in evs if e.type in (E.PREFILL_START, E.TOKENS, E.FINISHED)} == {"engine"}
        perf = fin.payload["perf"]
        assert perf["n_prompt"] == fin.payload["prompt"]["n_tokens"] and perf["peak_mem_bytes"] == 0
        assert perf["prefill_ns"] == t[E.PREFILL_DONE] - t[E.PREFILL_START] >= 0


def test_smoke_outcomes_success_summary_manifest(smoke):
    out, d, _ = smoke
    live = rows(d / "outcomes_run.jsonl")
    assert sorted(r["request_id"].rsplit("/", 2)[1] for r in live) == ["a"] * 4 + ["b"] * 2
    for r in live:
        if r["is_attack"]:
            assert 0.98 * 1024 <= r["prompt_len"] <= 1024 and r["payload_windows"]
    succ = rows(d / "success.jsonl")
    assert len(succ) == 6 and all(s["attack_success"] is None and s["utility"] is None for s in succ)
    assert {s["success_judge"] for s in succ} == {"pending@1"}
    assert all(o["attack_success"] is None for o in rows(d / "outcomes.jsonl"))
    s = load_json(d / "summary.json")
    assert s["asr"]["value"] is None and s["asr"]["den"] == 0 and s["asr"]["n_unlabelled"] == 4
    assert s["utility"]["n_unlabelled"] == 2 and s["_meta"]["n_A"] == 4 and s["_meta"]["n_S"] == 0
    assert s["_meta"]["system_info"]["engine"] == "hf" and s["_meta"]["checks_ok"] is True
    lin = load_json(d / "lineage.json")
    man = load_json(out / "raw" / "data" / lin["data_manifest_sha"] / "manifest.json")
    p = man["params"]
    assert man["generator"] == "lc_inject_draft" and p["target"] == "tiny-hf" and man["n_requests"] == 6
    assert p["background_paths"] == ["tests/data/bg_docs_small.jsonl.txt"]
    assert p["background_sha256"] == {"tests/data/bg_docs_small.jsonl.txt": hashlib.sha256(BG.read_bytes()).hexdigest()}


def test_smoke_resume_skips_done_requests(smoke):
    out, d, first = smoke
    assert len({e.request_id for e in first}) == 2 and {e.attempt for e in first} == {0}
    evs = _events(d)
    fins = [e for e in evs if e.type == E.FINISHED]
    assert len(fins) == 6 and len({e.request_id for e in fins}) == 6         # nothing re-run
    assert evs[:len(first)] == first
    assert {e.attempt for e in evs[len(first):]} == {1}
    n = len(evs)
    assert run_main(["--config", str(SMOKE), "--out", str(out), "--ts", TS]) == 0     # all done: no new events
    assert len(rows(d / "events.jsonl")) == n and len(rows(d / "outcomes_run.jsonl")) == 6


def test_smoke_label_sheet(smoke, tmp_path):
    _, d, _ = smoke
    spec = importlib.util.spec_from_file_location("export_label_sheet", ROOT / "scripts" / "export_label_sheet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--run-dir", str(d), "--out", str(tmp_path / "ls.csv")]) == 0
    sheet = rows(tmp_path / "ls.jsonl")
    gens = {r["request_id"]: r["generation"] for r in rows(d / "outcomes_run.jsonl")}
    assert [r["request_id"] for r in sheet] == sorted(gens)
    assert all(r["output_text"] == gens[r["request_id"]]["text_delivered"] for r in sheet)


def test_record_with_write_records_raises_at_setup(tmp_path):
    over = tmp_path / "b0_ref_rec.yaml"
    over.write_text(yaml.safe_dump({"base": str(ROOT / "configs" / "systems" / "b0_ref.yaml"), "system_id": "B0",
                                    "override": {"write_records": True}}))
    exp = yaml.safe_load(SMOKE.read_text())
    exp.update(systems=[str(over)], record={"layers": "all"})
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(exp))
    with pytest.raises(NotImplementedError, match="S2 step 3"):
        run_main(["--config", str(p), "--out", str(tmp_path / "out"), "--ts", TS])
    assert not list((tmp_path / "out" / "raw" / "W0-smoke").glob("*/lineage.json"))


# ------------------------------------------------------------------------------------------------ unit level
def _ctx(tmp_path, sink: list):
    return RunContext("r0", "W0-smoke", tmp_path, MonotonicClock(), 0, None, None, sink.append,
                      logging.getLogger("ag.test.hfref"))


def _cfg(flags: dict | None = None, **exp) -> SystemConfig:
    f = {**load_system("b0_ref")["flags"], **(flags or {})}
    e = {"target": "tiny-hf", "fairness": {"max_new_tokens": 6, "temperature": 0.0, "ignore_eos": False,
                                           "dtype": "float32"}, **exp}
    return SystemConfig("B0", "hf_reference", f, exp=e)


@pytest.mark.parametrize("flags,exp,err", [
    ({"write_records": True}, {"record": {"layers": "all"}}, NotImplementedError),
    ({}, {"record": {"layers": "all"}, "_no_write_flag": True}, NotImplementedError),
    ({"gate": "ag"}, {}, ValueError),
    ({"harm_scan": "parallel"}, {}, ValueError),
    ({}, {"fairness": {"max_new_tokens": 6, "temperature": 0.7}}, ValueError),
])
def test_setup_refusals(tmp_path, flags, exp, err):
    cfg = _cfg(flags, **{k: v for k, v in exp.items() if not k.startswith("_")})
    if exp.get("_no_write_flag"):          # write_records missing counts as "not false"
        cfg = SystemConfig(cfg.system_id, cfg.impl, {k: v for k, v in cfg.flags.items() if k != "write_records"},
                           exp=cfg.exp)
    with pytest.raises(err):
        create_system("hf_reference").setup(cfg, _ctx(tmp_path, []))


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    s = create_system("hf_reference")
    s.setup(_cfg(), _ctx(tmp_path_factory.mktemp("sys"), []))
    yield s
    s.teardown()


def _req(sink: str = "stream", max_new: int = 4) -> Request:
    segs = (Segment("system", "You are a helpful assistant.", True), Segment("user", "Summarize the document.", True),
            Segment("document", "The war began in 1914.\n\nThe river is long and the city is old.", False))
    rid = "t/pilot/512/000000/b/t00"
    return Request(rid, rid.rsplit("/", 2)[0], rid.rsplit("/", 1)[0], 0, "t", "pilot", "512", None, "summary3", segs,
                   sink, max_new, True, None)


def test_info_and_max_new_tokens(system):
    info = system.info()
    assert info.engine == "hf" and info.attn_backend == "sdpa" and info.mode == "live"
    assert info.flags["b0_chunks"] is True and info.flags["uses_judge"] is False
    assert info.flags["engine_version"].startswith("transformers-") and info.placement == {"target": "cpu"}
    evs = list(system.run(_req(max_new=50)))                # fairness.max_new_tokens = 6 caps it
    assert sum(e.type == E.TOKENS for e in evs) == 6


def test_tool_sink_is_one_tokens_event(system):
    evs = list(system.run(_req("tool", 5)))
    toks = [e for e in evs if e.type == E.TOKENS]
    fin = evs[-1]
    assert fin.type == E.FINISHED and len(toks) == 1 and toks[0].tok_range == (0, 5)
    assert toks[0].payload["text"] == fin.generation.text_delivered
    assert fin.generation.chunks == serving_chunks(5, tool_spans=[(0, 5)]) and fin.generation.chunks[0].is_tool_call


def test_failure_is_one_fatal_error(system, caplog):
    old = system.target.max_context
    system.target.max_context = 5
    try:
        with caplog.at_level(logging.ERROR, logger="ag.test.hfref"):
            evs = list(system.run(_req()))
    finally:
        system.target.max_context = old
    assert len(evs) == 1 and evs[0].type == E.ERROR and evs[0].fatal and evs[0].reason.startswith("ValueError")
    assert "Traceback (most recent call last)" in caplog.text


# ------------------------------------------------------------------------------------------------ runner, judge
def test_build_requests_registry_and_env_paths(tmp_path, monkeypatch):
    assert set(GENERATORS) == {"mock_lc_inject", "lc_inject_draft"} and PRODUCERS == {"mock": "mock", "hf": "reference"}
    monkeypatch.setenv("AG_TEST_BG_DIR", str(BG.parent))
    raw = "${AG_TEST_BG_DIR}/bg_docs_small.jsonl.txt"
    job = {"target": "MT", "system": {"flags": {"engine": "mock"}}, "fairness": {"max_new_tokens": 32},
           "data": {"generator": "lc_inject_draft", "n_attack": 3, "n_benign": 1, "length_classes": ["1k"],
                    "background_paths": [raw]}}
    reqs, params = build_requests(job)
    assert [r.is_attack for r in reqs] == [True, False, True, True]
    assert params["background_paths"] == [raw] and params["split"] == "pilot" and params["source"] == "lc_draft"
    assert params["background_sha256"][raw] == hashlib.sha256(BG.read_bytes()).hexdigest()
    assert all(r.max_new_tokens == 32 and r.split == "pilot" for r in reqs)
    monkeypatch.delenv("AG_TEST_BG_DIR")
    with pytest.raises(FileNotFoundError, match="ag_download_login.sh"):
        build_requests(job)
    with pytest.raises(ValueError, match="background_paths"):
        build_requests({**job, "data": {"generator": "lc_inject_draft"}})
    with pytest.raises(NotImplementedError, match="known"):
        build_requests({**job, "data": {"generator": "nope"}})


def test_pending_judge_and_none_in_rates():
    for name in ("pending", "pending@1"):
        res = get_success_judge(name).judge(_req(), None)
        assert res.attack_success is None and res.utility is None and res.judge == "pending@1"
        assert "W0-grader" in res.raw["reason"]
    assert get_success_judge("mock@1").name == "mock@1"
    with pytest.raises(KeyError):
        get_success_judge("nope")
    base = {"labels": {"inj": True}, "is_attack": True, "error_kind": None, "source": "s"}
    out = [{**base, "unit_id": f"u{i}", "attack_success": v, "utility": v} for i, v in enumerate([True, False, None, None])]
    a = q.asr(out)
    assert (a["num"], a["den"], a["value"], a["n_unlabelled"]) == (1, 2, 0.5, 2)
    u = q.utility(out, attack=True)
    assert (u["num"], u["den"], u["n_unlabelled"]) == (1, 2, 2)
    none = q.asr([{**base, "unit_id": "x", "attack_success": None}])
    assert none["value"] is None and none["den"] == 0 and none["n_unlabelled"] == 1
