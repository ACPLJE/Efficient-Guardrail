"""D8 token spans, E0-4 gates and the data card on the synthetic fixture (dry-run tree)."""
from __future__ import annotations

import gzip
import json
import random

from lgs2 import haystack as H
from lgs2 import needles as N
from lgs2 import screen as S
from lgs2 import splits as SP
from lgs2 import synth as SY
from lgs2.tok import WhitespaceTok
from lgs3 import datacard as DC
from lgs3 import gates as G
from lgs3 import tokspans as TK
from tests.test_s2_pipeline import TEST_DOMAINS, _fake_screen_rows, _needle_files, root  # noqa: F401


def _prepare(P, tmp_path, monkeypatch):
    tok = WhitespaceTok()
    for d in TEST_DOMAINS:
        H.build_domain(d, P, tok, budget_factor=1.0)
    _fake_screen_rows(P, [d.name for d in TEST_DOMAINS])
    S.summarize(P, n_review=6)
    files = _needle_files(tmp_path / "srcs", random.Random(1))
    monkeypatch.setattr(N, "source_files", lambda src, P_: files.get(src.name, []))
    N.build_pool(P, tok, tok)
    monkeypatch.setattr(SP, "external_texts", lambda P_: {"safetyniah": []})
    SP.run(P)
    monkeypatch.setattr(SY, "DRYRUN_PLAN", {"test": {"pairs": 4, "L": [1024, 2048], "positions": [0.0, 0.5, 1.0], "benign_long_per_L": 2},
                                            "train": {"pairs": 3, "L": [1024], "positions": "random2+ends", "benign_long_per_L": 0},
                                            "native": 5})
    rep = SY.run_dry(P, tok, seed=1, allow_unscreened=False)
    assert rep["validation"]["pass"]
    return tok


def test_tokspans_gates_datacard(root, tmp_path, monkeypatch):
    P = root
    tok = _prepare(P, tmp_path, monkeypatch)
    rep = TK.run(P, "qwen3", "test", tok=tok, root=P.synth_dryrun)
    assert set(rep["subsets"]) == {"main", "benign_long"}
    m = rep["subsets"]["main"]
    assert m["len_within_tol"] == 1.0 and m["recon_pass"] and m["needle_fully_contained"] == 1.0
    TK.run(P, "qwen3", "train", tok=tok, root=P.synth_dryrun)
    with gzip.open(P.synth_dryrun / "tok" / "qwen3" / "test" / "main" / "spans.jsonl.gz", "rt") as f:
        spans = [json.loads(l) for l in f]
    rows = {r["id"]: r for r in SY.read_rows(P.synth_dryrun / "test" / "main" / "rows.jsonl.gz")}
    for s in spans[:20]:
        r = rows[s["id"]]
        a, b = s["needle_tok_span"]
        assert s["n_tok"] == r["n_tok_qwen3"] and 0 <= a < b <= s["n_tok"] and s["window_ids_containing_needle"]
    res = G.evaluate(P, root=P.synth_dryrun, tok_root=P.synth_dryrun / "tok")
    g = res["gates"]
    for k in g:
        assert g[k]["pass"], (k, g[k])
    assert res["pass"]
    assert json.loads((P.splits / "audit.json").read_text())["gates_e04"]["pass"]
    # 2026-09-19: a native row carrying a >128-token prompt is exempt from gate 4 (counted, not failed); a duplicated
    # row id trips gate 3, and `lgs3.dedupe` repairs rows + manifest + every tokenizer's spans file
    native_p = P.synth_dryrun / "train" / "native" / "rows.jsonl.gz"
    nat = SY.read_rows(native_p)
    longrow = dict(nat[0])
    longrow["id"], longrow["n_tok_needle"], longrow["n_tok_qwen3"] = "train-native-long-1", 300, 300
    longrow["text"] = " ".join(["w"] * 300)
    longrow["needle_char_span"] = [0, len(longrow["text"])]
    with gzip.open(native_p, "at", encoding="utf-8") as f:
        f.write(json.dumps(longrow) + "\n")
    TK.run(P, "qwen3", "train", tok=tok, root=P.synth_dryrun)                      # spans for the new row
    res_n = G.evaluate(P, root=P.synth_dryrun, tok_root=P.synth_dryrun / "tok")
    g4 = res_n["gates"]["4_needle_length"]
    assert g4["pass"] and g4["native_rows_gt128_exempt"] == 1 and g4["long_needles_outside_subset"] == 0, g4
    train_main = P.synth_dryrun / "train" / "main" / "rows.jsonl.gz"
    first = SY.read_rows(train_main)[0]
    with gzip.open(train_main, "at", encoding="utf-8") as f:
        f.write(json.dumps(first) + "\n")
    res_d = G.evaluate(P, root=P.synth_dryrun, tok_root=P.synth_dryrun / "tok")
    assert res_d["gates"]["3_ids_twins_recipe_spans"]["duplicate_ids"] == 1 and not res_d["pass"]
    spans_p = P.synth_dryrun / "tok" / "qwen3" / "train" / "main" / "spans.jsonl.gz"        # tokspans ran on the duplicate too
    with gzip.open(spans_p, "rt", encoding="utf-8") as f:
        first_span = f.readline()
    with gzip.open(spans_p, "at", encoding="utf-8") as f:
        f.write(first_span)
    from lgs3 import dedupe as DD
    import types
    P2 = types.SimpleNamespace(root=P.synth_dryrun, synth=P.synth_dryrun, manifests=P.synth_dryrun / "manifests")
    out = DD.run(P2, "train", "main")
    assert out["dropped"] == 1 and out["spans"]["qwen3"]["dropped"] == 1, out
    ids = [r["id"] for r in SY.read_rows(train_main)]
    assert len(ids) == len(set(ids))
    mf = json.loads((P.synth_dryrun / "manifests" / "train.main.manifest.json").read_text())
    assert mf["rows"] == len(ids) and mf["dry_run"] is False
    res_ok = G.evaluate(P, root=P.synth_dryrun, tok_root=P.synth_dryrun / "tok")
    assert res_ok["gates"]["3_ids_twins_recipe_spans"]["pass"], res_ok["gates"]["3_ids_twins_recipe_spans"]
    # a leaked background piece must trip gate 1
    bad = dict(next(iter(rows.values())))
    bad["split"], bad["id"] = "train", "leak-1"
    with gzip.open(P.synth_dryrun / "train" / "main" / "rows.jsonl.gz", "at", encoding="utf-8") as f:
        f.write(json.dumps(bad) + "\n")
    res2 = G.evaluate(P, root=P.synth_dryrun, tok_root=P.synth_dryrun / "tok")
    assert not res2["gates"]["1_split_disjointness"]["pass"] and not res2["pass"]
    md = DC.build(P)
    assert "presence label" in md and "## 6. Synthesized sets" in md
