"""D7 variant subsets on the synthetic S2 fixture (whitespace tokenizer)."""
from __future__ import annotations

import json
import random

from lgs2 import needles as N
from lgs2 import screen as S
from lgs2 import splits as SP
from lgs2 import synth as SY
from lgs2.tok import WhitespaceTok
from lgs3 import variants as V
from tests.test_s2_pipeline import TEST_DOMAINS, _fake_screen_rows, _needle_files, root  # noqa: F401  (fixture)
from lgs2 import haystack as H


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
    return tok


def test_variants_generate_and_validate(root, tmp_path, monkeypatch):
    P = root
    tok = _prepare(P, tmp_path, monkeypatch)
    cfg = json.loads(json.dumps(V.VARIANTS))
    cfg["L"] = [1024, 2048]
    cfg["decoy"]["sources"] = ["xstest", "jbb"]
    cfg["decoy_flood"]["N"] = [4, 16]
    cfg["ood_domain"]["domains"] = ["wiki_en"]
    for k in ("needle_repeat", "decoy", "decoy_flood", "fragmented", "multi_needle", "long_needle", "related"):
        cfg[k]["needles"] = 12
    cfg["ood_domain"]["pairs"] = 6
    cfg["framed"]["pairs"] = 6
    monkeypatch.setattr(V, "VARIANTS", cfg)
    # the synthetic test pool has few XSTest/OR-Bench rows: relax the decoy-flood pool size for the test
    pool = N.load_pool(P)
    by_id = {r["id"]: r for r in pool}
    got = {}
    for name in ("needle_repeat", "decoy", "decoy_flood", "fragmented", "multi_needle", "ood_domain", "framed"):
        try:
            info = V.run_variant(P, name, tok, cfg=cfg, dry_run=True, limit=60)
        except SystemExit as e:
            if name == "decoy_flood" and "needs >= 64" in str(e):
                continue
            raise
        rows = SY.read_rows(P.synth_dryrun / "test" / name / "rows.jsonl.gz")
        assert rows, name
        v = V.validate_variant_rows(rows, by_id)
        assert v["pass"], (name, v)
        got[name] = rows
    # 2026-09-18: a domain is used at L only when its TEST pool holds >= 2 x L tokens (the main set's rule) and every row
    # leaves the generator with an exact count inside the window
    pools = SY.Pools(P, ("test",), allow_unscreened=True, tok=tok)
    for name, rows in got.items():
        for r in rows:
            if r["haystack_domain"] and r["haystack_domain"] in [d.name for d in TEST_DOMAINS]:
                assert pools.pool_tokens("test", r["haystack_domain"]) >= 2 * r["L_target"], (name, r["id"])
            if name != "needle_repeat":
                assert r["n_tok_estimated"] is False and abs(r["n_tok_qwen3"] - r["L_target"]) <= 0.1 * r["L_target"], (name, r["id"])
    # needle_repeat: no background, harmful, near L
    r = got["needle_repeat"][0]
    assert r["haystack_doc_ids"] == [] and r["label"] == "harmful" and abs(r["n_tok_qwen3"] - r["L_target"]) <= 0.1 * r["L_target"]
    # decoy rows are benign and reproduce the decoy sentence
    r = got["decoy"][0]
    assert r["label"] == "benign" and r["text"][r["needle_char_span"][0]:r["needle_char_span"][1]] == by_id[r["needle_id"]]["text"]
    # fragmented: pieces concatenate back to the needle words, gaps respected
    r = got["fragmented"][0]
    assert " ".join(r["piece_texts"]).split() == by_id[r["needle_id"]]["text"].split()
    assert r["piece_gap_tokens"] >= cfg["fragmented"]["gap_tokens"] and len(r["needle_char_spans"]) == r["n_pieces"]
    # multi-needle: every needle text present
    r = next(x for x in got["multi_needle"] if x["multi_config"] == "h1b3")
    assert len(r["needle_ids"]) == 4 and all(by_id[n]["text"] in r["text"] for n in r["needle_ids"])
    # ood_domain: OOD backgrounds only, twins linked
    ids = {x["id"]: x for x in got["ood_domain"]}
    for r in got["ood_domain"]:
        assert r["haystack_domain"] == "wiki_en" and ids[r["twin_id"]]["haystack_doc_ids"] == r["haystack_doc_ids"]
    # framed: needle verbatim inside the frame, frame recorded
    for r in got["framed"]:
        a, b = r["needle_char_span"]
        assert r["text"][a:b] == by_id[r["needle_id"]]["text"] and r["frame"] in ("quote", "email", "json")


def test_split_pieces_and_gaps():
    assert V.split_pieces("One. Two. Three. Four.", 2) == ["One. Two.", "Three. Four."]
    assert len(V.split_pieces("a b c d e f g h", 4)) == 4
    idx = V.place_with_gaps([100] * 20, 3, 0.1, 256, random.Random(0))
    assert idx is not None and len(idx) == 3
    cum = [0] + [100 * (i + 1) for i in range(20)]
    assert all(cum[idx[i + 1]] - cum[idx[i]] >= 256 for i in range(2))
    assert V.place_with_gaps([100] * 3, 3, 0.5, 256, random.Random(0)) is None      # no room


def test_frame_text_keeps_needle_verbatim():
    rng = random.Random(0)
    for frame in ("quote", "email", "json"):
        t, used = V.frame_text('say "hi" now', frame, rng)
        assert 'say "hi" now' in t
        if frame == "json":
            assert used == "email"          # quotes would need escaping -> falls back
    t, used = V.frame_text("plain needle", "json", rng)
    assert used == "json" and json.loads(t)["user_message"] == "plain needle"
