"""B1/B2 runner on the dry-run rows with a tiny causal guard: native, truncation, chunk-OR, summaries."""
from __future__ import annotations

import dataclasses
import json
import random

from lgs1.zeroshot import guards as G
from lgs2 import haystack as H
from lgs2 import needles as N
from lgs2 import screen as S
from lgs2 import splits as SP
from lgs2 import synth as SY
from lgs2.tok import HFTok
from lgs3 import longbench as LB
from tests.test_s2_pipeline import TEST_DOMAINS, _fake_screen_rows, _needle_files, root  # noqa: F401
from tests.test_zeroshot import _tiny_repo


def test_longbench_policies(root, tmp_path, monkeypatch):
    P = root
    repo = _tiny_repo(tmp_path)
    tok = HFTok(str(repo), "tiny")
    for d in TEST_DOMAINS:
        H.build_domain(d, P, tok, budget_factor=1.0)
    _fake_screen_rows(P, [d.name for d in TEST_DOMAINS])
    S.summarize(P, n_review=6)
    files = _needle_files(tmp_path / "srcs", random.Random(1))
    monkeypatch.setattr(N, "source_files", lambda src, P_: files.get(src.name, []))
    N.build_pool(P, tok, tok)
    monkeypatch.setattr(SP, "external_texts", lambda P_: {"safetyniah": []})
    SP.run(P)
    monkeypatch.setattr(SY, "DRYRUN_PLAN", {"test": {"pairs": 2, "L": [512], "positions": [0.0, 1.0], "benign_long_per_L": 1},
                                            "train": {"pairs": 2, "L": [512], "positions": "random2", "benign_long_per_L": 0}, "native": 2})
    assert SY.run_dry(P, tok, seed=1, allow_unscreened=False)["validation"]["pass"]
    P2 = dataclasses.replace(P, synth=P.synth_dryrun)          # the runner reads synth/test/*; point it at the dry-run tree
    # tiny guard: the S1 smoke spec (Qwen3Guard-style template + parser) on a random tiny Qwen3
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = G.GuardSpec(name="tiny_guard", repo_id=str(repo), kind="causal", max_new_tokens=4, batch_size=4, published_key="",
                       build=G.build_user_chat, parse=G.parse_qwen3guard, first_token=("Unsafe", "Safe"))
    monkeypatch.setitem(G.GUARDS, "tiny_guard", spec)
    htok = AutoTokenizer.from_pretrained(repo); htok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(repo).eval()
    monkeypatch.setattr(LB, "load_guard", lambda name, dtype, policy: (spec, htok, model, "cpu", None))
    r1 = LB.run(P2, "tiny_guard", "v2main", "native", ctx_override=100000)
    r2 = LB.run(P2, "tiny_guard", "v2main", "trunc_keep_head", ctx_override=200)
    r3 = LB.run(P2, "tiny_guard", "v2main", "chunk_or:64:32", batch=4)
    assert r1["new"] == r2["new"] == r3["new"] == 2 * 2 * 2 + 1
    out = P.out / "longbench" / "v2main"
    nat = [json.loads(l) for l in open(out / "tiny_guard.native.rows.jsonl")]
    tr = [json.loads(l) for l in open(out / "tiny_guard.trunc_keep_head.rows.jsonl")]
    cd = [json.loads(l) for l in open(out / "tiny_guard.cd64o32.rows.jsonl")]
    assert all(r["error"] is None for r in nat + tr + cd)
    assert all(not r["truncated"] and r["n_chunks"] == 1 for r in nat)
    assert all(r["truncated"] and r["n_tok_input"] <= 200 for r in tr)
    assert all(r["n_chunks"] > 1 and r["budget_x"] > 1.0 and len(r["chunk_flags"]) == r["n_chunks"] for r in cd)
    for r in nat + tr + cd:
        assert set(("model_sha", "cfg_hash", "code_commit", "pred", "score", "ms", "label")) <= set(r)
    # resume: nothing new
    assert LB.run(P2, "tiny_guard", "v2main", "native", ctx_override=100000)["new"] == 0
    # native policy with a tiny context records an error row instead of crashing
    LB.run(P2, "tiny_guard", "v2:main", "native", ctx_override=50, max_rows=1)
    err = [json.loads(l) for l in open(P.out / "longbench" / "v2_main" / "tiny_guard.native.rows.jsonl")]
    assert err[0]["error"] and err[0]["pred"] is None
    # a VLM-style guard hands back a PROCESSOR as tok: processor(text) must not be used for counting (Llama Guard 4 raised
    # "You have to specify text" on every row, 2026-09-20) -- the run goes through processor.tokenizer instead
    class FakeProc:
        tokenizer = htok

        def __call__(self, *a, **kw):
            if "text" not in kw:
                raise ValueError("You have to specify text")
            return htok(kw["text"], return_tensors="pt", add_special_tokens=False)

        def apply_chat_template(self, msgs, **kw):
            if kw.get("tokenize"):
                raise ValueError("You have to specify text")          # the failing transformers path
            return msgs[0]["content"][0]["text"]

        def decode(self, ids, skip_special_tokens=True):
            return "unsafe\nS1"

    vspec = G.GuardSpec(name="tiny_vlm", repo_id=str(repo), kind="vlm", max_new_tokens=2, batch_size=1, published_key="",
                        build=None, parse=G.parse_safe_unsafe_first_line)
    monkeypatch.setitem(G.GUARDS, "tiny_vlm", vspec)
    monkeypatch.setattr(LB, "load_guard", lambda name, dtype, policy: (vspec, FakeProc(), model, "cpu", None))
    rv = LB.run(P2, "tiny_vlm", "v2:main", "native", ctx_override=100000, max_rows=2)
    vrows = [json.loads(l) for l in open(P.out / "longbench" / "v2_main" / "tiny_vlm.native.rows.jsonl")]
    assert rv["new"] == 2 and all(r["error"] is None and r["pred"] == 1 and r["n_tok_input"] > 16 for r in vrows)
    monkeypatch.setattr(LB, "load_guard", lambda name, dtype, policy: (spec, htok, model, "cpu", None))
    md = LB.summarize(P2, "v2main")
    text = md.read_text()
    assert "| tiny_guard | native | 512 |" in text and "| tiny_guard | cd64o32 | 512 |" in text
    table = json.loads((out / "summary.json").read_text())
    assert {t["policy"] for t in table} == {"native", "trunc_keep_head", "cd64o32"}
    assert all(t["n"] == 9 and 0.0 <= t["parse_fail_rate"] <= 1.0 for t in table)   # a random tiny model rarely parses


def test_policy_parsing_and_chunking():
    assert LB.parse_policy("chunk_or:512:256") == {"name": "chunk_or", "W": 512, "overlap": 256, "tag": "cd512o256"}
    assert LB.parse_policy("chunk_or")["W"] == 512 and LB.parse_policy("yarn")["tag"] == "yarn"
    ids = list(range(1000))
    spans = LB.chunk_ids(ids, 300, 100)
    assert spans[0] == (0, 300) and spans[1] == (200, 500) and spans[-1][1] == 1000
    assert LB.chunk_ids(ids[:100], 300, 100) == [(0, 100)]
