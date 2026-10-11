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
            "audit": {"pass": True}, "bigneg": {}, "shortfall": {}}
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
    (D / "meta.json").write_text(json.dumps({"complete": True, "poolings": ["mean"], "layers": L1, "n": N, "d": d}))
    return R


@pytest.mark.parametrize("harm_from,inj_from,expect", [(1, 6, "PASS"), (5, 5, "FAIL")])
def test_planted_depth_gap_is_recovered(agpaths, harm_from, inj_from, expect):
    from agdepth.analyze import analyze
    from agdepth.fit import fit
    _planted_run(agpaths, "p", harm_from, inj_from)
    fit("p", "planted", ["raw"], ["mean"], seeds=[0, 1], device="cpu", P=agpaths)
    s = analyze("p", n_boot=30, P=agpaths)
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
    import depthctl
    s = depthctl.cmd_smoke(type("A", (), {"dir": str(tmp_path), "n_boot": 5})())
    assert s["comparisons"] and Path(tmp_path / "out" / "smoke" / "report" / "report.md").exists()
