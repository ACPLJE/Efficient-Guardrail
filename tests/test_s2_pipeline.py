"""S2 end-to-end on synthetic sources (whitespace tokenizer, 3 domains, 5 needle sources):
haystack pools -> screening summary (fake rows) -> needle pool -> splits + audit -> synthesis dry-run + validation."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pandas as pd
import pytest

from lgs1.lineage import Lineage, RowsWriter
from lgs2 import config as C
from lgs2 import haystack as H
from lgs2 import needles as N
from lgs2 import screen as S
from lgs2 import splits as SP
from lgs2 import synth as SY
from lgs2.tok import WhitespaceTok

WORDS = [f"w{i}" for i in range(3000)]


def _para(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words)) + "."


def _doc(rng, n_par, style="blank"):
    pars = [_para(rng, rng.randint(20, 60)) for _ in range(n_par)]
    if style == "code":
        return "\n\n".join("\n".join(f"x{i} = {rng.randint(0, 9)}" for i in range(rng.randint(3, 8))) for _ in range(n_par))
    return "\n\n".join(pars)


TEST_DOMAINS = [
    C.Domain("govreport", "train", "govreport", 60_000, kind="stream", max_piece_tokens=800, min_doc_tokens=20),
    C.Domain("stack", "train", "stack", 40_000, kind="stream", paragraph="code", max_piece_tokens=800, min_doc_tokens=20),
    C.Domain("wiki_en", "ood", "wiki_en", 20_000, kind="stream", max_piece_tokens=800, min_doc_tokens=20),
]


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_ENV_FILE_NO_OVERRIDE", "1")
    monkeypatch.setenv("LG_DATA_V2", str(tmp_path / "lg_data_v2"))
    monkeypatch.setenv("LG_SOURCES", str(tmp_path / "lg_data_v2" / "sources"))
    monkeypatch.setenv("LG_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for mod in (C, H, S, SP):
        monkeypatch.setattr(mod, "DOMAINS", TEST_DOMAINS, raising=True)
    monkeypatch.setattr(C, "DOMAIN_BY_NAME", {d.name: d for d in TEST_DOMAINS})
    monkeypatch.setattr(C, "TRAIN_DOMAINS", ["govreport", "stack"])
    monkeypatch.setattr(C, "OOD_DOMAINS", ["wiki_en"])
    rec = json.loads(json.dumps(C.RECIPE_V2))
    rec["domains_per_split"] = {k: ["govreport", "stack"] for k in rec["domains_per_split"]}
    rec["benign_long_domains"] = {"test": ["govreport", "stack", "wiki_en"], "calib": ["govreport", "stack"], "val": ["govreport", "stack"]}
    monkeypatch.setattr(SY, "RECIPE_V2", rec)
    P = C.S2Paths.load().ensure()
    rng = random.Random(0)
    for d in TEST_DOMAINS:
        src = P.sources / d.source
        src.mkdir(parents=True, exist_ok=True)
        with gzip.open(src / "docs.jsonl.gz", "wt", encoding="utf-8") as f:
            for i in range(120):
                f.write(json.dumps({"id": f"{d.source}{i}", "text": _doc(rng, rng.randint(8, 40), d.paragraph), "meta": {}}) + "\n")
    return P


def _needle_files(tmp: Path, rng: random.Random):
    """Synthetic files for 5 sources (WildGuard train/test parquet, ToxicChat csv, XSTest csv, JBB parquet)."""
    def sent(n):
        return " ".join(rng.choice(WORDS) for _ in range(n))
    files = {}
    d = tmp / "wgtrain"; d.mkdir(parents=True)
    rows = [{"prompt": sent(rng.randint(5, 60)), "adversarial": bool(i % 3 == 0), "prompt_harm_label": "harmful" if i % 2 else "unharmful",
             "response": None, "subcategory": "x"} for i in range(400)]
    rows[0]["prompt"] = "EXACT COPY OF A TEST PROMPT number one"                   # exact overlap with wgtest row 0
    rows[1]["prompt"] = "near duplicate of test prompt two with many identical words in the same order here ok"
    rows[2]["prompt"] = "external needle from safety niah verbatim goes here today"   # exact overlap with SafetyNIAH
    pd.DataFrame(rows).to_parquet(d / "train-00000.parquet")
    files["wildguardtrain"] = [d / "train-00000.parquet"]
    d = tmp / "wgtest"; d.mkdir()
    rows = [{"prompt": sent(rng.randint(5, 60)), "adversarial": False, "prompt_harm_label": "harmful" if i % 2 else "unharmful"} for i in range(60)]
    rows[0]["prompt"] = "EXACT COPY OF A TEST PROMPT number one"
    rows[1]["prompt"] = "near duplicate of test prompt two with many identical words in the same order here okay"
    pd.DataFrame(rows).to_parquet(d / "test-00000.parquet")
    files["wgtest"] = [d / "test-00000.parquet"]
    d = tmp / "toxicchat" / "data" / "0124"; d.mkdir(parents=True)
    pd.DataFrame([{"conv_id": f"c{i}", "user_input": sent(rng.randint(3, 40)), "toxicity": int(i % 4 == 0), "jailbreaking": 0,
                   "human_annotation": True} for i in range(120)]).to_csv(d / "toxic-chat_annotation_train.csv", index=False)
    pd.DataFrame([{"conv_id": f"t{i}", "user_input": sent(rng.randint(3, 40)), "toxicity": int(i % 4 == 0), "jailbreaking": 0,
                   "human_annotation": i % 5 != 0} for i in range(50)]).to_csv(d / "toxic-chat_annotation_test.csv", index=False)
    files["toxicchat_train"] = [d / "toxic-chat_annotation_train.csv"]
    files["toxicchat_test"] = [d / "toxic-chat_annotation_test.csv"]
    d = tmp / "xstest"; d.mkdir()
    pd.DataFrame([{"id": i, "prompt": sent(rng.randint(4, 12)), "type": "homonyms", "label": "safe" if i % 2 else "unsafe",
                   "focus": "kill", "note": ""} for i in range(30)]).to_csv(d / "xstest_v2_prompts.csv", index=False)
    files["xstest"] = [d / "xstest_v2_prompts.csv"]
    d = tmp / "jbb" / "behaviors"; d.mkdir(parents=True)
    pd.DataFrame([{"Index": i, "Goal": sent(rng.randint(6, 20)), "Behavior": "b", "Category": "c", "Source": "s"} for i in range(10)]).to_parquet(d / "harmful-00000.parquet")
    pd.DataFrame([{"Index": i, "Goal": sent(rng.randint(6, 20)), "Behavior": "b", "Category": "c", "Source": "s"} for i in range(10)]).to_parquet(d / "benign-00000.parquet")
    files["jbb"] = [d / "harmful-00000.parquet", d / "benign-00000.parquet"]
    return files


def _fake_screen_rows(P, domains, reject_every=7):
    lin = Lineage(model_sha="x", router_sha="none", cfg_hash="c", recipe_hash="none", data_manifest_sha="none", code_commit="t")
    tok = WhitespaceTok()
    for d in domains:
        for m in C.SCREEN["models"]:
            with RowsWriter(P.screen / f"{d}.{m}.rows.jsonl", lin) as w:
                for k, ch in enumerate(S.iter_chunks(P, d, tok, chunk_tokens=200)):
                    flag = (m == "llamaguard3_8b" and k % reject_every == 3)
                    w.write({"id": ch["id"], "doc_id": ch["doc_id"], "chunk": ch["chunk"], "char_span": ch["char_span"],
                             "label": "unsafe" if flag else "safe", "flag": flag, "score": 0.9 if flag else 0.05, "raw": "",
                             "model": m, "domain": d})


def test_pipeline_end_to_end(root, tmp_path, monkeypatch):
    P = root
    tok = WhitespaceTok()
    # --- D4a haystack ---
    for d in TEST_DOMAINS:
        st = H.build_domain(d, P, tok, budget_factor=1.0)
        assert st["n_pieces"] > 0 and st["reached_target"], st
        recs = list(H.load_docs(P, d.name))
        assert all(r["n_tok"] <= d.max_piece_tokens for r in recs)
        assert all(sum(r["par_tok"]) == r["n_tok"] for r in recs)
        if d.paragraph == "code":
            assert all("\n\n" not in p for r in recs for p in r["pars"])          # code paragraphs = blank-line blocks
    # --- D4b screening summary from fake rows ---
    _fake_screen_rows(P, [d.name for d in TEST_DOMAINS])
    summ = S.summarize(P, n_review=6)
    for d in TEST_DOMAINS:
        v = summ["domains"][d.name]
        assert v["status"] == "complete" and 0 < v["pass_rate"] < 1 and v["n_kept"] + v["n_rejected"] == v["n_docs"]
        keep = S.load_keep(P, d.name)
        assert keep is not None and len(keep) == v["n_kept"]
    assert (P.screen / "manual_review_50.md").exists() and "verdict:" in (P.screen / "manual_review_50.md").read_text()
    # --- D5a needle pool ---
    files = _needle_files(tmp_path / "srcs", random.Random(1))
    monkeypatch.setattr(N, "source_files", lambda src, P_: files.get(src.name, []))
    st = N.build_pool(P, tok, tok)
    assert st["sources"]["wildguardtrain"]["status"] == "ok" and st["sources"]["harmbench"]["status"] == "missing"
    pool = N.load_pool(P)
    assert {r["pool"] for r in pool} == {"train", "test"}
    tc_test = [r for r in pool if r["source"] == "toxicchat_test"]
    assert len(tc_test) == 40                                                     # human-annotated rows only
    assert all(r["n_tok_qwen3"] == len(r["text"].split()) for r in pool)
    # --- D5b splits + audit (external: one exact copy of a train needle + one near-dup + one unrelated) ---
    ext = {"safetyniah": ["external needle from safety niah verbatim goes here today",
                          "near duplicate of test prompt two with many identical words in the same order here ok friend",
                          "completely unrelated external text about weather"]}
    monkeypatch.setattr(SP, "external_texts", lambda P_: ext)
    res = SP.run(P)
    ov = res["overlap"]
    removed = json.loads((P.splits / "audit.json").read_text())["removed"]
    assert "wildguardtrain:train-00000-0" in removed and removed["wildguardtrain:train-00000-0"]["reason"] == "exact"
    assert "wildguardtrain:train-00000-2" in removed and "external:safetyniah" in removed["wildguardtrain:train-00000-2"]["against"]
    assert "wildguardtrain:train-00000-1" in removed and removed["wildguardtrain:train-00000-1"]["reason"] == "near_dup"
    assert ov["n_removed"] == 3 and res["gates"]["pass"] and res["screened"]
    nsplit, hsplit = SP.load_splits(P)
    assert all(nsplit["assign"][r["id"]] == "test" for r in pool if r["pool"] == "test")
    assert all(nsplit["assign"][r["id"]] in ("train", "val", "calib", "removed") for r in pool if r["pool"] == "train")
    assert all(hsplit["assign"][d] == "test" for d in hsplit["assign"] if d.startswith("wiki_en"))
    assert {hsplit["assign"][d] for d in hsplit["assign"] if d.startswith("govreport")} == {"train", "val", "calib", "test"}
    # --- D6 dry-run synthesis + validation ---
    monkeypatch.setattr(SY, "DRYRUN_PLAN", {"test": {"pairs": 4, "L": [1024, 4096], "positions": [0.0, 0.5, 1.0], "benign_long_per_L": 2},
                                            "train": {"pairs": 3, "L": [1024], "positions": "random2+ends", "benign_long_per_L": 0},
                                            "native": 5})
    rep = SY.run_dry(P, tok, seed=1, allow_unscreened=False)
    assert rep["validation"]["pass"], rep["validation"]
    rows = SY.read_rows(P.synth_dryrun / "test" / "main" / "rows.jsonl.gz")
    assert len(rows) == 4 * 2 * 3 * 2
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        assert abs(r["n_tok_qwen3"] - r["L_target"]) <= 0.10 * r["L_target"]
        a, b = r["needle_char_span"]
        needle = next(n for n in pool if n["id"] == r["needle_id"])
        assert r["text"][a:b] == needle["text"] and r["n_tok_needle"] <= 128
        assert r["text"].split("\n\n").count(needle["text"]) >= 1                 # inserted as its own paragraph
        t = by_id[r["twin_id"]]
        assert t["haystack_doc_ids"] == r["haystack_doc_ids"] and t["depth"] == r["depth"] and t["label"] != r["label"]
        assert all(hsplit["assign"][d] == "test" for d in r["haystack_doc_ids"])
        assert all(d in S.load_keep(P, r["haystack_domain"]) for d in r["haystack_doc_ids"])   # screened pieces only
        assert abs(r["needle_share"] - r["n_tok_needle"] / r["n_tok_qwen3"]) < 1e-6
    # same background reused across the depths of a pair; depth 0 / 1 put the needle first / last
    first = [r for r in rows if r["label"] == "harmful" and r["L_target"] == 1024 and r["pair_id"].endswith("-p0")]
    assert len({tuple(r["haystack_doc_ids"]) for r in first}) == 1
    d0 = next(r for r in first if r["depth"] == 0.0); d1 = next(r for r in first if r["depth"] == 1.0)
    assert d0["needle_char_span"][0] == 0 and d1["needle_char_span"][1] == len(d1["text"])
    bl = SY.read_rows(P.synth_dryrun / "test" / "benign_long" / "rows.jsonl.gz")
    assert len(bl) == 2 * 2 and all(r["label"] == "benign" and r["needle_id"] is None for r in bl)
    assert {r["haystack_domain"] for r in bl} <= {"govreport", "stack", "wiki_en"}
    nat = SY.read_rows(P.synth_dryrun / "train" / "native" / "rows.jsonl.gz")
    assert len(nat) == 5 and all(r["needle_share"] == 1.0 for r in nat)
    man = json.loads((P.synth_dryrun / "manifests" / "test.main.manifest.json").read_text())
    assert man["rows"] == len(rows) and man["recipe_hash"] == rep["recipe_hash"] and len(man["sha256"]) == 64


def test_validate_rows_catches_broken_twins(root):
    rows = [
        {"id": "a", "split": "test", "subset": "main", "L_target": 1000, "depth": 0.5, "needle_id": "n1", "twin_id": "b", "pair_id": "p",
         "haystack_domain": "govreport", "haystack_doc_ids": ["d1"], "needle_char_span": [0, 3], "label": "harmful", "needle_share": 0.01,
         "n_tok_qwen3": 1000, "n_tok_needle": 10, "recipe_hash": "r", "text": "xxx", "n_tok_estimated": False},
        {"id": "b", "split": "test", "subset": "main", "L_target": 1000, "depth": 0.7, "needle_id": "n2", "twin_id": "a", "pair_id": "p",
         "haystack_domain": "govreport", "haystack_doc_ids": ["d1"], "needle_char_span": [0, 3], "label": "benign", "needle_share": 0.01,
         "n_tok_qwen3": 1500, "n_tok_needle": 10, "recipe_hash": "r", "text": "yyy", "n_tok_estimated": False},
    ]
    v = SY.validate_rows(rows)
    assert not v["pass"]
    assert any("twin" in p for p in v["problems"]) and any("L_target" in p for p in v["problems"])


def test_screen_flag_semantics():
    assert S.flag_of("qwen3guard_gen_8b", "controversial") is True          # strict mode
    assert S.flag_of("llamaguard3_8b", "safe") is False
    assert S.flag_of("llamaguard3_8b", None) is None
    assert N.bucket_of(32) == "le32" and N.bucket_of(33) == "33_64" and N.bucket_of(128) == "65_128" and N.bucket_of(129) == "129_512" and N.bucket_of(600) == "gt512"


def test_score_batch_label_step_with_tiny_model(tmp_path):
    """score_batch runs the generate + label-step path end to end on a tiny random Qwen3 (CPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lgs1.zeroshot import guards as G
    from tests.test_zeroshot import _tiny_repo
    repo = _tiny_repo(tmp_path)
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(repo).eval()
    spec = G.get("qwen3guard_gen_8b")
    words, flagged = S.label_words("qwen3guard_gen_8b", spec)
    assert words == S.QWEN_LABELS and flagged == (0, 2)
    res = S.score_batch(spec, tok, model, "cpu", [spec.build(tok, "some chunk text about cats"), spec.build(tok, "another chunk")], words, flagged)
    assert len(res) == 2 and all("raw" in r and (r["score"] is None or 0.0 <= r["score"] <= 1.0) for r in res)
    words, flagged = S.label_words("llamaguard3_8b", G.get("llamaguard3_8b"))
    assert words == ("unsafe", "safe") and flagged == (0,)
