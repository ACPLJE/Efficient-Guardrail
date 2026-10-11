"""Depth-asymmetry PoC (agdepth): metrics, window builder, encoders, loaders, and a planted-gap end-to-end check.

All CPU, offline.  The planted test writes synthetic per-layer features with a known answer (harm separable from layer
1, injection vs instruction-heavy only from layer 6) and requires fit + analyze to recover it.
"""
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from agdepth import metrics as MX


@pytest.fixture
def agpaths(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_ENV_FILE_NO_OVERRIDE", "1")
    monkeypatch.setenv("AG_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("LG_OUT", str(tmp_path / "lgout"))
    monkeypatch.setenv("LG_SOURCES", str(tmp_path / "sources"))
    monkeypatch.setenv("LG_DATA_V2", str(tmp_path / "data_v2"))
    from agdepth.config import AGPaths
    return AGPaths.load()


# ---------------------------------------------------------------------------------------------- metrics
def test_tpr_matches_strict_matched_fpr():
    from lgs1.metrics import strict_matched_fpr
    rng = np.random.default_rng(0)
    neg, pos = rng.normal(0, 1, 5000), rng.normal(1.5, 1, 800)
    neg[:50] = 2.0                                     # ties at the threshold region
    for t in (1e-2, 1e-3, 2e-4):
        th = strict_matched_fpr(neg, t)
        tpr, fpr, tau = MX.tpr_at(pos, neg, t)
        assert tau == th.tau and fpr == th.achieved_fpr and fpr <= t
        assert tpr == float(np.mean(pos > th.tau))


def test_auroc_matches_pairwise():
    rng = np.random.default_rng(1)
    pos, neg = rng.integers(0, 5, 60).astype(float), rng.integers(0, 5, 70).astype(float)
    pair = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(MX.auroc(pos, neg) - pair) < 1e-12


def test_curveboot_unit_weights_equal_plain_curves():
    rng = np.random.default_rng(2)
    S, n_p, n_n, L = 2, 300, 4000, 5
    Sp = rng.normal(1, 1, (S, n_p, L)).astype(np.float32)
    Sn = rng.normal(0, 1, (S, n_n, L)).astype(np.float32)
    cb = MX.CurveBoot(Sp, Sn, 1e-2, auroc_max_neg=n_n)
    got_t = cb.tpr(np.ones(n_p), np.ones(n_n))
    got_a = cb.auroc(np.ones(n_p), np.ones(n_n))
    ref = [MX.curves(Sp[s], Sn[s], [1e-2]) for s in range(S)]
    assert np.allclose(got_t, np.mean([r["tpr@0.01"] for r in ref], 0))
    assert np.allclose(got_a, np.mean([r["auroc"] for r in ref], 0), atol=1e-6)
    # integer weights == duplicated samples
    wp, wn = rng.integers(0, 3, n_p).astype(float), rng.integers(0, 3, n_n).astype(float)
    dup_p = np.repeat(np.arange(n_p), wp.astype(int))
    dup_n = np.repeat(np.arange(n_n), wn.astype(int))
    ref2 = np.mean([MX.curves(Sp[s][dup_p], Sn[s][dup_n], [1e-2])["tpr@0.01"] for s in range(S)], 0)
    assert np.allclose(cb.tpr(wp, wn), ref2)


def test_ell95():
    assert MX.ell95([0.1, 0.5, 0.96, 1.0, 0.99]) == 2
    assert MX.ell95([0.91, 0.95, 0.94]) == 0          # 0.91 >= 0.95 * 0.95
    assert math.isnan(MX.ell95([0.0, 0.0]))
    from agdepth.analyze import l95_of
    assert l95_of("auroc", [0.5, 0.9, 0.99, 1.0]) == 2       # chance-corrected: 0.8 < 0.95, 0.98 >= 0.95
    assert l95_of("tpr@0.01", [0.5, 0.9, 0.99, 1.0]) == 2


# ---------------------------------------------------------------------------------------------- windows
def test_build_smoke_dataset(agpaths):
    from agdepth.config import PROFILES
    from agdepth.windows import build, read_jsonl_gz
    card = build("t", PROFILES["smoke"], smoke=True, P=agpaths)
    assert card["audit"]["pass"]
    rows = read_jsonl_gz(agpaths.run("t").windows)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    for cls, d in card["windows"]["test"].items():
        assert d == PROFILES["smoke"].n_test, cls
    for r in rows:
        a, b = r["snip_span"]
        assert 0 <= a < b <= len(r["text"])
        assert r["text"][a:b].strip()
        assert abs(r["n_tok"] - PROFILES["smoke"].window) <= 4
    # every class follows the same snippet-length histogram (within sampling noise)
    means = {c: v["test"]["mean"] for c, v in card["snippet_tokens"].items()}
    assert abs(means["harm"] - means["inj"]) < 4, means          # the two positive classes define the target histogram
    assert any(r["sid"].endswith("#half") for r in rows)          # short benign variants fill the short bins
    assert (agpaths.run("t").bigneg).exists()


def test_fill_empty_splits():
    from agdepth.windows import _fill_empty_splits
    items = [{"group": f"g{i}", "split": "train"} for i in range(5)]
    _fill_empty_splits(items, 0)
    assert {it["split"] for it in items} == {"train", "val", "test"}


def test_assign_splits_is_group_consistent_across_classes():
    from agdepth.windows import assign_splits
    by = {"harm": [{"group": f"g{i}", "holdout": i % 5 == 0, "split": None} for i in range(40)],
          "harm_hard": [{"group": f"g{i}", "holdout": i % 7 == 0, "split": None} for i in range(40)]}
    for mode in ("source", "group"):
        for items in by.values():
            for it in items:
                it["split"] = None
        assign_splits(by, mode)
        seen = {}
        for items in by.values():
            for it in items:
                assert seen.setdefault(it["group"], it["split"]) == it["split"], (mode, it)
        if mode == "source":
            hold = {it["group"] for items in by.values() for it in items if it["holdout"]}
            assert all(it["split"] == "test" for items in by.values() for it in items if it["group"] in hold)


def test_stratified_draw_balances_sources():
    import random
    from agdepth.windows import stratified_draw
    items = [{"source": "big", "n_tok": 20 + i % 30, "sid": f"b{i}"} for i in range(1000)]
    items += [{"source": "small", "n_tok": 20 + i % 30, "sid": f"s{i}"} for i in range(100)]
    hist = {b: 1 / 4 for b in range(2, 6)}
    out = stratified_draw(items, hist, 150, 1, random.Random(0))
    share = sum(o["source"] == "small" for o in out) / len(out)
    assert len(out) == 150 and 0.3 < share < 0.7


def test_imperative_count():
    from agdepth.sources import imperative_count
    assert imperative_count("Preheat the oven. Mix the flour. The cake is good.") == (2, 3)
    assert imperative_count("1. Install the package.\n2. Run the tests.") == (2, 2)
    assert imperative_count("The committee met on Monday.")[0] == 0


# ---------------------------------------------------------------------------------------------- loaders
def test_loaders(tmp_path):
    from agdepth.sources import load_bipia, load_llmail, load_notinject, load_text_label
    f = tmp_path / "text_attack_test.json"
    f.write_text(json.dumps({"Cat A": ["do x", "do y"], "Cat B": ["do z"]}))
    rows = load_bipia([f])
    assert len(rows) == 3 and rows[0]["group"] == "bipia:text:Cat A" and rows[0]["meta"]["bipia_split"] == "test"
    f = tmp_path / "NotInject_one.json"
    f.write_text(json.dumps([{"prompt": "Can I ignore this warning?", "word_list": ["ignore"], "category": "Q"}]))
    assert load_notinject([f])[0]["meta"]["trigger_words"] == ["ignore"]
    f = tmp_path / "subs.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"subject": "Hi", "body": "text one", "objectives": json.dumps({"a": True, "b": True})},
        {"subject": None, "body": "text two", "objectives": {"a": True, "b": False}},
        {"other": 1}]))
    ll = load_llmail([f])
    assert [r["text"] for r in ll] == ["Subject: Hi\n\ntext one", "text two"]
    assert [r["meta"]["success"] for r in ll] == [True, False]
    f = tmp_path / "pi.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [{"text": "a", "label": 1}, {"text": "b", "label": 0}]))
    assert [r["text"] for r in load_text_label([f])] == ["a"]


def test_instruction_generator_is_benign_and_deterministic():
    from agdepth import synth
    a, b = synth.gen_instructions(3, seed=0), synth.gen_instructions(3, seed=0)
    assert a == b and len({r["group"] for r in a}) == len(synth.INSTR_TEMPLATES)
    assert all("{" not in r["text"] for r in a)


# ---------------------------------------------------------------------------------------------- encoders
def test_encode_masks(tmp_path):
    from agdepth.extract import encode
    from agdepth.tiny import make_tiny_repo
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(make_tiny_repo(tmp_path / "m"))
    text = "The river near the city was built in 1900.\n\nPreheat the oven to 180 degrees. <|im_start|> stir.\n\nThe end."
    a = text.index("Preheat")
    b = text.index("\n\nThe end")
    row = {"id": "w1", "text": text, "snip_span": [a, b]}
    raw = encode(tok, row, "raw", 512)
    assert not raw["win"][0] and raw["win"][1:].all() and raw["span"].any() and not raw["span"][0]
    special = tok.convert_tokens_to_ids("<|im_start|>")
    assert special not in raw["ids"]                       # special-token strings inside documents stay text
    task = encode(tok, row, "task", 512)
    assert task["ids"].count(special) == 2                  # user turn + generation prompt, none from the window
    w = np.where(task["win"])[0]
    assert task["span"][w[0]:w[-1] + 1].sum() == task["span"].sum()     # span inside the window
    assert task["last"] == len(task["ids"]) - 1 and not task["win"][-1]


# ---------------------------------------------------------------------------------------------- planted gap
def _planted_run(P, name: str, harm_from: int, inj_from: int, L1: int = 9, d: int = 16, n=(300, 100, 600), seed=0):
    """Windows + fake features: harm vs benign separable from layer `harm_from`, inj vs instr from `inj_from`
    (inj vs plain is easy at every layer)."""
    from agdepth.config import PROFILES, recipe, recipe_hash
    R = P.run(name)
    R.data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for split, k in zip(("train", "val", "test"), n):
        for cls in ("harm", "harm_hard", "inj", "instr", "notinject", "plain"):
            for j in range(k):
                rows.append({"id": f"{split}-{cls}-{j}", "split": split, "cls": cls, "source": cls, "sid": f"{cls}{j}",
                             "group": f"{split}-{cls}-{j // 3}", "holdout": False, "text": f"{cls} window {j}",
                             "snip_span": [0, 3], "n_tok": 256, "n_tok_snip": 40, "bg_doc": "d", "depth": 0.5})
    with gzip.open(R.windows, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    prof = PROFILES["smoke"]
    rec = recipe(prof, "group", ["synthetic"], "none")
    card = {"recipe": rec, "recipe_hash": recipe_hash(rec), "windows": {}, "snippet_tokens": {}, "windows_by_source": {},
            "audit": {"pass": True}, "bigneg": {}, "shortfall": {}, "data_manifest_sha": f"planted{harm_from}{inj_from}",
            "run_id": ""}
    R.datacard.write_text(json.dumps(card))
    D = R.feats("planted", "raw")
    D.mkdir(parents=True, exist_ok=True)
    N = len(rows)
    mm = np.memmap(D / "mean.f16", dtype=np.float16, mode="w+", shape=(L1, N, d))
    cls = np.array([r["cls"] for r in rows])
    for l in range(L1):
        X = rng.normal(0, 1, (N, d))
        X[cls == "inj", 1] += 3.0                     # injection vs plain: separable everywhere ...
        X[cls == "instr", 1] += 3.0                   # ... but instruction-heavy text shares that direction
        X[cls == "notinject", 1] += 3.0
        if l >= harm_from:
            X[cls == "harm", 0] += 4.0
        if l >= inj_from:
            X[cls == "inj", 2] += 4.0
        mm[l] = X.astype(np.float16)
    mm.flush()
    (D / "ids.json").write_text(json.dumps([r["id"] for r in rows]))
    (D / "meta.json").write_text(json.dumps({"complete": True, "poolings": ["mean"], "layers": L1, "n": N, "d": d,
                                             "data_manifest_sha": card["data_manifest_sha"], "model_sha": ""}))
    return R


@pytest.mark.parametrize("harm_from,inj_from,expect", [(1, 6, "PASS"), (5, 5, "FAIL")])
def test_planted_depth_gap_is_recovered(agpaths, harm_from, inj_from, expect):
    from agdepth.analyze import analyze
    from agdepth.fit import fit
    _planted_run(agpaths, "p", harm_from, inj_from)
    fit("p", "planted", ["raw"], ["mean"], seeds=[0, 1], device="cpu", P=agpaths)
    s = analyze("p", primary_fpr=1e-2, n_boot=30, P=agpaths)     # 600 test negatives: 2e-4 is not resolvable
    g = next(x for x in s["comparisons"] if x["comparison"] == "gate" and x["metric"].startswith("tpr"))
    assert g["l95_shallow"] == harm_from and g["l95_deep"] == inj_from, g
    assert g["verdict"] == expect, g
    assert (agpaths.run("p").report / "report.md").exists()
    # shuffled-label control ~ chance
    ctrl = json.loads((agpaths.run("p").probes("planted", "raw") / "mean.fit.json").read_text())["controls_shuffled_auroc"]
    assert all(abs(v - 0.5) < 0.15 for d in ctrl.values() for v in d.values()), ctrl


# ---------------------------------------------------------------------------------------------- smoke
def test_smoke_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_ENV_FILE_NO_OVERRIDE", "1")
    before = os.environ.get("AG_OUT"), os.environ.get("AGDEPTH_OUT")
    import depthctl
    s = depthctl.cmd_smoke(type("A", (), {"dir": str(tmp_path), "n_boot": 5, "primary_fpr": None})())
    assert (os.environ.get("AG_OUT"), os.environ.get("AGDEPTH_OUT")) == before      # no environment side effects
    run = tmp_path / "out" / "smoke"
    assert s["comparisons"] and (run / "report" / "report.md").exists()
    assert s["primary_metric"] == "tpr@0.0002"                                       # the plan's metric, never switched
    for d in (run / "data", run / "probes" / "tiny-model", run / "report"):         # CLAUDE.md 3.3
        lin = json.loads((d / "lineage.json").read_text())
        assert lin["finished_unix_ns"] and lin["producer"] == "mock" and lin["data_manifest_sha"] == s["data_manifest_sha"]
    rows = [json.loads(l) for l in open(run / "report" / "probe_results.jsonl")]
    assert rows and all(r["run_id"] == s["run_id"] and "2e-4" in r["tpr_at_fpr"] for r in rows)
    json.loads((run / "report" / "summary.json").read_text())                       # strict JSON (no NaN literal)
    # a second smoke run on the same dir starts from scratch (the extraction code really runs again)
    s2 = depthctl.cmd_smoke(type("A", (), {"dir": str(tmp_path), "n_boot": 5, "primary_fpr": None})())
    assert s2["run_id"] != s["run_id"] or s2["data_manifest_sha"] == s["data_manifest_sha"]


# ---------------------------------------------------------------------------------------------- regressions (review)
def test_stratified_draw_is_not_biased_by_file_order():
    import random
    from agdepth.windows import stratified_draw
    items = [{"source": "big", "n_tok": 20 + i % 30, "sid": f"b{i}", "pos": i} for i in range(30000)]
    out = stratified_draw(items, {b: 1 / 4 for b in range(2, 6)}, 400, 1, random.Random(0))
    mean_pos = sum(o["pos"] for o in out) / len(out)
    assert 0.35 * 30000 < mean_pos < 0.65 * 30000, mean_pos          # log-space keys: no underflow to file order


def test_stratified_draw_fills_short_bins_from_near_bins():
    import random
    from agdepth.windows import BIN, stratified_draw
    # target: short snippets (bins 1-6); the class has few short and many long ones (reviewer scenario)
    items = [{"source": "s", "n_tok": BIN * b + 1, "sid": f"{b}-{i}"} for b in range(1, 7) for i in range(30)]
    items += [{"source": "s", "n_tok": BIN * b + 1, "sid": f"{b}-{i}"} for b in range(7, 14) for i in range(300)]
    out = stratified_draw(items, {b: 1 / 6 for b in range(1, 7)}, 1500, 3, random.Random(0))
    near = sum(o["n_tok"] // BIN <= 7 for o in out) / len(out)
    assert len(out) == 1500 and near >= 0.36, near          # in-bin reuse before far-away long snippets
    # unused distinct snippets one bin away are still taken before any reuse
    items2 = [{"source": "s", "n_tok": BIN * 3 + 1, "sid": f"a{i}"} for i in range(10)]
    items2 += [{"source": "s", "n_tok": BIN * 4 + 1, "sid": f"b{i}"} for i in range(100)]
    out2 = stratified_draw(items2, {3: 0.5, 4: 0.5}, 100, 3, random.Random(0))
    assert len({o["sid"] for o in out2}) == 100


def test_half_variants_keep_only_valid_labels():
    from agdepth.windows import half_variants
    para = ("The committee met on Tuesday to review the budget. The figures were discussed at length. Several "
            "concerns were raised about the timeline. Submit your comments by Friday. Send them to the clerk.")
    it = {"source": "mined_instr", "sid": "m:1", "text": para, "meta": {}}
    assert half_variants([it]) == []                                  # the half has no imperative sentence left


def test_llmail_is_test_only_in_every_split_mode():
    from agdepth.windows import assign_splits
    for mode in ("group", "source"):
        by = {"inj": [{"group": f"ll{i}", "holdout": True, "test_only": True, "split": None} for i in range(30)]
              + [{"group": f"b{i}", "holdout": False, "test_only": False, "split": None} for i in range(300)]}
        assign_splits(by, mode)
        assert all(it["split"] == "test" for it in by["inj"] if it["test_only"])
        other = [it["split"] for it in by["inj"] if not it["test_only"]]
        if mode == "group":                    # the other injection sources keep their own test split
            assert other.count("test") > 50 and other.count("train") > 100, other.count("test")


def test_heldout_positives_get_their_own_contrast():
    from agdepth.analyze import build_contrasts
    cls = np.array(["inj"] * 6 + ["instr"] * 4 + ["plain"] * 4 + ["harm"] * 3 + ["harm_hard"] * 3 + ["notinject"] * 2)
    ho = np.zeros(len(cls), bool)
    ho[:2] = True                                              # two LLMail windows
    sc = np.random.default_rng(0).normal(size=(2, 1, len(cls), 3))
    C = build_contrasts(sc, ["harm", "inj"], cls, np.arange(len(cls)), te_heldout=ho)
    assert C["inj/instr"].Sp.shape[1] == 4 and C["inj@heldout/instr"].Sp.shape[1] == 2
    assert C["harm/all"].Sp.shape[1] == 3 and "harm@heldout/all" not in C


def test_fpr_key_is_exact():
    from agdepth.analyze import fpr_key
    assert [fpr_key(t) for t in (1e-2, 1e-3, 2e-4, 5e-5)] == ["1e-2", "1e-3", "2e-4", "5e-5"]
    assert fpr_key(1.6e-4) == "1.6e-4" and fpr_key(0.015) == "1.5e-2"


def test_registry_files_ignore_failed_download(agpaths):
    from agdepth.config import SNIP_SOURCE_BY_NAME
    from agdepth.sources import registry_files
    d = agpaths.sources / "notinject"
    d.mkdir(parents=True)
    (d / "NotInject_one.json").write_text("[]")
    (d / "revision.json").write_text(json.dumps({"status": "failed"}))
    assert registry_files(SNIP_SOURCE_BY_NAME["notinject"], agpaths) == []
    (d / "revision.json").write_text(json.dumps({"status": "ok", "local_path": str(d)}))
    assert registry_files(SNIP_SOURCE_BY_NAME["notinject"], agpaths)


def test_added_tokens_stay_plain_text_and_span_skips_sink(tmp_path):
    from transformers import AutoTokenizer
    from agdepth.extract import encode
    from agdepth.tiny import make_tiny_repo
    tok = AutoTokenizer.from_pretrained(make_tiny_repo(tmp_path / "m"))
    tok.add_tokens(["<think>"])                                       # non-special added token (as in Qwen3)
    think = tok.convert_tokens_to_ids("<think>")
    text = "Preheat the oven. <think> stir the sauce.\n\nThe river near the city."
    enc = encode(tok, {"id": "w", "text": text, "snip_span": [0, 40]}, "raw", 512)
    assert think not in enc["ids"]
    assert not enc["span"][0] and enc["span"][1:].any()              # snippet at position 0: sink excluded


def test_bootstrap_replicates_stay_aligned():
    from agdepth.analyze import Contrast, bootstrap
    rng = np.random.default_rng(0)
    S, L = 1, 3
    a = Contrast("harm", "all", rng.normal(1, 1, (S, 5, L)), rng.normal(0, 1, (S, 200, L)), np.zeros(5, int),
                 np.arange(200) % 19 + 1, 0)
    b = Contrast("inj", "instr", rng.normal(1, 1, (S, 50, L)), rng.normal(0, 1, (S, 200, L)), np.arange(50) % 19 + 1,
                 np.arange(200) % 19 + 1, 0)
    out = bootstrap({"harm/all": a, "inj/instr": b}, 20, 1e-2, 40, L)
    assert out["harm/all"]["tpr"]["l95"].size == out["inj/instr"]["tpr"]["l95"].size == 40
    assert np.isnan(out["harm/all"]["tpr"]["l95"]).any()             # draws without group 0 stay as NaN rows


def test_primary_fpr_is_fixed_at_2e4(agpaths):
    from agdepth.analyze import analyze
    from agdepth.fit import fit
    _planted_run(agpaths, "q", 1, 6)
    fit("q", "planted", ["raw"], ["mean"], seeds=[0], device="cpu", P=agpaths)
    s = analyze("q", n_boot=10, P=agpaths)
    g = next(x for x in s["comparisons"] if x["comparison"] == "gate" and x["metric"].startswith("tpr"))
    assert s["primary_metric"] == "tpr@0.0002" and g["verdict"].startswith("INCONCLUSIVE"), g


def test_rebuild_invalidates_features_and_lexical(agpaths, tmp_path):
    from agdepth import extract, fit as F
    from agdepth.config import PROFILES
    from agdepth.tiny import make_tiny_repo
    from agdepth.windows import build
    repo = str(make_tiny_repo(tmp_path / "m"))
    c1 = build("r", PROFILES["smoke"], smoke=True, P=agpaths, seed=1)
    M = extract.load(repo, "cpu", "fp32")
    extract.extract("r", repo, ["raw"], ["mean"], M=M, P=agpaths, budget=4096)
    F.fit("r", repo, ["raw"], ["mean"], seeds=[0], device="cpu", controls=False, P=agpaths)
    c2 = build("r", PROFILES["smoke"], smoke=True, P=agpaths, seed=2)
    assert c1["data_manifest_sha"] != c2["data_manifest_sha"]
    with pytest.raises(SystemExit):                                   # stale features are refused, not reused
        F.fit("r", repo, ["raw"], ["mean"], seeds=[0], device="cpu", controls=False, P=agpaths)
    extract.extract("r", repo, ["raw"], ["mean"], M=M, P=agpaths, budget=4096)    # starts over, no skip
    meta = json.loads((agpaths.run("r").feats("m", "raw") / "meta.json").read_text())
    assert meta["data_manifest_sha"] == c2["data_manifest_sha"] and meta["complete"]
    F.fit("r", repo, ["raw"], ["mean"], seeds=[0], device="cpu", controls=False, P=agpaths)
    assert F.stored_sha(agpaths.run("r").lex / "lex.test_scores.npz") == c2["data_manifest_sha"]


def test_fresh_extract_never_inherits_progress(agpaths, tmp_path, monkeypatch):
    from agdepth import extract
    from agdepth.config import PROFILES
    from agdepth.tiny import make_tiny_repo
    from agdepth.windows import build
    repo = str(make_tiny_repo(tmp_path / "m"))
    build("f", PROFILES["smoke"], smoke=True, P=agpaths)
    M = extract.load(repo, "cpu", "fp32")
    extract.extract("f", repo, ["raw"], ["mean"], M=M, P=agpaths, budget=4096)
    D = agpaths.run("f").feats("m", "raw")
    assert json.loads((D / "progress.json").read_text())["done"] == [0]
    real = extract.iter_pooled

    def crash(*a, **k):
        raise KeyboardInterrupt("simulated preemption")
    monkeypatch.setattr(extract, "iter_pooled", crash)
    with pytest.raises(KeyboardInterrupt):                            # new pooling -> fresh start, killed in shard 0
        extract.extract("f", repo, ["raw"], ["mean", "span"], M=M, P=agpaths, budget=4096)
    assert not (D / "progress.json").exists()
    monkeypatch.setattr(extract, "iter_pooled", real)
    extract.extract("f", repo, ["raw"], ["mean", "span"], M=M, P=agpaths, budget=4096)
    meta = json.loads((D / "meta.json").read_text())
    mm = np.memmap(D / "span.f16", dtype=np.float16, mode="r", shape=(meta["layers"], meta["n"], meta["d"]))
    assert meta["complete"] and np.abs(np.asarray(mm[1])).sum(1).min() > 0     # every window really extracted


def test_sbatch_scripts_use_the_ag_environment(agpaths, tmp_path, monkeypatch):
    import depthctl
    monkeypatch.setenv("AGDEPTH_OUT", str(agpaths.root))
    (tmp_path / "hub" / "models--Qwen--Qwen3Guard-Gen-0.6B").mkdir(parents=True)     # the reference tokenizer is cached
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(depthctl.shutil, "which", lambda name: None)  # no sbatch here: render only
    assert depthctl.main(["run", "--run", "s", "--backbones", "e03", "--dtype", "fp16", "--sbatch"]) == 0
    scripts = sorted((agpaths.root / "s" / "slurm").glob("*.sbatch"))
    assert len(scripts) == 4                                          # build, 2 backbones, analyze
    for p in scripts:
        t = p.read_text()
        assert "slurm/ag_common.sh" in t and "slurm/common.sh" not in t.replace("ag_common", "")
        assert "--gres=gpu:1" in t and "-p RTX6000ADA" in t and f'AGDEPTH_OUT="{agpaths.root}"' in t
    assert any("--dtype fp16" in p.read_text() for p in scripts)
    with pytest.raises(SystemExit):
        depthctl.main(["run", "--run", "s", "--backbones", "qwen3guard_06b", "--sbatch"])
    with pytest.raises(SystemExit):
        depthctl.main(["run", "--run", "s", "--poolings", "mean,max", "--sbatch"])


def test_cli_rejects_ambiguous_backbone_and_bad_dtype(agpaths, monkeypatch):
    import depthctl
    monkeypatch.setenv("AGDEPTH_OUT", str(agpaths.root))
    monkeypatch.setattr(depthctl.shutil, "which", lambda name: None)
    for argv in (["pipeline", "--run", "x", "--backbone", "e03", "--sbatch"],
                 ["pipeline", "--run", "x", "--backbone", "qwen3guard_0.6b,llama31_8b", "--sbatch"],
                 ["run", "--run", "x", "--backbones", "e03", "--dtype", "float8", "--sbatch"],
                 ["run", "--run", "x", "--backbones", "e03", "--tok", "qwen-3", "--sbatch"]):
        with pytest.raises(SystemExit):
            depthctl.main(argv)


def test_hub_fallback_is_skipped_after_a_failed_download(agpaths, tmp_path, monkeypatch):
    from agdepth.config import SNIP_SOURCE_BY_NAME
    from agdepth.sources import registry_files
    hub = tmp_path / "hub"
    snap = hub / "datasets--deepset--prompt-injections" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "part.jsonl").write_text('{"text": "x", "label": 1}\n')
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    src = SNIP_SOURCE_BY_NAME["deepset_pi"]
    assert registry_files(src, agpaths)                       # no download record: the cached copy is used
    d = agpaths.sources / "deepset_pi"
    d.mkdir(parents=True)
    (d / "revision.json").write_text(json.dumps({"status": "failed"}))
    assert registry_files(src, agpaths) == []                 # a failed download never falls back to a partial copy


def test_interrupted_rebuild_leaves_no_stale_card(agpaths, monkeypatch):
    from agdepth import lineage as LN
    from agdepth.config import PROFILES
    from agdepth.windows import build, load_card
    build("i", PROFILES["smoke"], smoke=True, P=agpaths)
    assert load_card(agpaths.run("i"))["data_manifest_sha"]

    def boom(*a, **k):
        raise RuntimeError("killed")
    monkeypatch.setattr(LN, "start", boom)
    with pytest.raises(RuntimeError):
        build("i", PROFILES["smoke"], smoke=True, P=agpaths, seed=5)
    with pytest.raises(SystemExit):                            # consumers refuse instead of using the old hash
        load_card(agpaths.run("i"))


def test_standalone_fit_shares_the_backbone_lineage(tmp_path, monkeypatch):
    import depthctl
    monkeypatch.setenv("LG_ENV_FILE_NO_OVERRIDE", "1")
    depthctl.cmd_smoke(type("A", (), {"dir": str(tmp_path), "n_boot": 2, "primary_fpr": None})())
    run_dir = tmp_path / "out" / "smoke" / "probes" / "tiny-model"
    before = json.loads((run_dir / "lineage.json").read_text())
    monkeypatch.setenv("AGDEPTH_OUT", str(tmp_path / "out"))
    assert depthctl.main(["fit", "--run", "smoke", "--backbone", str(tmp_path / "tiny-model"), "--framings", "raw",
                          "--poolings", "mean", "--device", "cpu"]) == 0
    after = json.loads((run_dir / "lineage.json").read_text())
    assert after["run_id"] == before["run_id"] and after["finished_unix_ns"] >= before["finished_unix_ns"]
