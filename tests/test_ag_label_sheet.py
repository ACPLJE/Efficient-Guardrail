"""scripts/export_label_sheet.py on a synthetic run directory (W0-grader, plan 5.3 #7). No torch."""
import csv
import importlib.util
import json

import pytest

from _ag_util import ROOT, rows
from ag.data.lc_inject.background import load_docs
from ag.data.lc_inject.draft import generate
from ag.data.manifest import stamp, write_manifest
from ag.data.types import Generation
from ag.targets.registry import length_fn_for

COLUMNS = ["request_id", "pair_key", "is_attack", "attack_type", "position", "goal_id", "injected_instruction",
           "goal", "task_text", "output_text", "n_output_tokens", "marker_hint", "label_attack_success",
           "label_utility", "labeler", "notes"]
RUN_ID = "W0-grader__B0__T2__lcdraft-512__deadbeef__s00r0__0__20261011T000000"


def _script():
    spec = importlib.util.spec_from_file_location("export_label_sheet", ROOT / "scripts" / "export_label_sheet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def run(tmp_path):
    """4 attack + 4 benign requests; one benign ends in a fatal error; one attack row was resumed."""
    docs = load_docs(ROOT / "tests" / "data" / "bg_docs_small.jsonl.txt")
    reqs = generate(n_pairs=4, n_benign=4, length_classes=["512"], docs=docs, length_fn=length_fn_for("MT"))
    reqs, sha = stamp(reqs)
    out = tmp_path / "out"
    write_manifest(out, reqs, sha, generator="lc_inject_draft", params={})
    d = out / "raw" / "W0-grader" / RUN_ID
    d.mkdir(parents=True)
    (d / "lineage.json").write_text(json.dumps({"run_id": RUN_ID, "data_manifest_sha": sha}))
    lines = []
    for r in reversed(reqs):                                   # file order is not request_id order
        if r.request_id.endswith("000003/b/t00"):
            gen = None                                         # fatal error: no generation
        elif r.task_meta.get("goal_id") == "access_granted":
            gen = Generation(r.request_id, (5, 6), "ACCESS GRANTED", "ACCESS GRANTED", (), (), False, None)
        else:
            text = "Résumé: a short summary,\nwith \"quotes\" and 東京."
            gen = Generation(r.request_id, (1, 2, 3), "x", text, (), (), False, None)
        lines.append({"request_id": r.request_id, "generation": gen.to_json() if gen else None, "run_id": RUN_ID})
    stale = dict(lines[0], generation=Generation(lines[0]["request_id"], (9,), "old", "old", (), (), False,
                                                 None).to_json())
    lines.insert(0, stale)                                     # an earlier attempt's row: the later row wins
    (d / "outcomes_run.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return d, reqs


def test_columns_order_bom_and_values(run):
    d, reqs = run
    assert _script().main(["--run-dir", str(d)]) == 0
    raw = (d / "label_sheet.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                     # utf-8-sig
    with open(d / "label_sheet.csv", encoding="utf-8-sig", newline="") as f:
        table = list(csv.reader(f))
    assert table[0] == COLUMNS
    recs = [dict(zip(COLUMNS, r)) for r in table[1:]]
    by_id = {r.request_id: r for r in reqs}
    assert [r["request_id"] for r in recs] == sorted(set(by_id) - {"lc_draft/pilot/512/000003/b/t00"})
    for r in recs:
        req = by_id[r["request_id"]]
        assert r["pair_key"] == req.pair_key and r["task_text"] == req.task_meta["task_text"]
        assert all(r[c] == "" for c in ("label_attack_success", "label_utility", "labeler", "notes"))
        if req.is_attack:
            assert r["is_attack"] == "TRUE" and r["attack_type"] == req.attack.attack_type
            assert r["goal_id"] == req.task_meta["goal_id"] and r["goal"] == req.attack.goal
            assert r["injected_instruction"] == req.task_meta["injected_instruction"]
            assert float(r["position"]) == req.attack.position
            assert r["marker_hint"] == ("TRUE" if req.task_meta["goal_id"] == "access_granted" else "FALSE")
        else:
            assert r["is_attack"] == "FALSE" and r["attack_type"] == r["goal_id"] == r["marker_hint"] == ""
    first = recs[0]                                            # 000000/a: access_granted, the resumed row
    assert first["output_text"] == "ACCESS GRANTED" and first["n_output_tokens"] == "2"
    other = recs[1]
    assert other["output_text"] == "Résumé: a short summary,\nwith \"quotes\" and 東京."


def test_jsonl_matches_csv(run):
    d, _ = run
    _script().main(["--run-dir", str(d)])
    js = rows(d / "label_sheet.jsonl")
    with open(d / "label_sheet.csv", encoding="utf-8-sig", newline="") as f:
        table = list(csv.DictReader(f))
    assert [r["request_id"] for r in js] == [r["request_id"] for r in table] and len(js) == 7
    assert all(set(r) == set(COLUMNS) | {"run_id", "schema_version"} and r["run_id"] == RUN_ID for r in js)
    assert all(isinstance(r["is_attack"], bool) and r["label_utility"] is None for r in js)
    assert {r["marker_hint"] for r in js if not r["is_attack"]} == {None}


def test_sample_is_seeded_and_ordered(run, tmp_path):
    d, _ = run
    mod = _script()
    full, _ = mod.build_rows(d)
    for out in ("s1.csv", "s2.csv"):
        assert mod.main(["--run-dir", str(d), "--sample", "3", "--seed", "7", "--out", str(tmp_path / out)]) == 0
    assert (tmp_path / "s1.csv").read_bytes() == (tmp_path / "s2.csv").read_bytes()
    picked = [r["request_id"] for r in rows(tmp_path / "s1.jsonl")]
    assert len(picked) == 3 and picked == sorted(picked) and set(picked) <= {r["request_id"] for r in full}
    assert mod.sample_rows(full, 10, 0) == full and mod.sample_rows(full, None) == full


def test_rejects_rows_outside_the_manifest(run):
    d, _ = run
    with open(d / "outcomes_run.jsonl", "a") as f:
        f.write(json.dumps({"request_id": "lc_draft/pilot/512/000099/a/t00", "generation": None}) + "\n")
    with pytest.raises(ValueError, match="not in"):
        _script().build_rows(d)
