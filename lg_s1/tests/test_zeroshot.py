"""E0-6 harness: parser unit tests + an offline smoke run with a tiny random model (no network)."""
import json
import os
from pathlib import Path

import pytest

from lgs1.zeroshot import guards as G
from lgs1.zeroshot import compare as C
from lgs1.zeroshot.run import summarize


# ---- parsers ------------------------------------------------------------------------------
@pytest.mark.parametrize("text,exp", [
    ("Safety: Unsafe\nCategories: Violent\n", "unsafe"),
    ("Safety: Safe\nCategories: None", "safe"),
    ("safety: controversial\ncategories: Politically Sensitive Topics", "controversial"),
    ("I cannot classify this", None),
])
def test_parse_qwen3guard(text, exp):
    assert G.parse_qwen3guard(text) == exp


@pytest.mark.parametrize("text,exp", [
    ("\n\nunsafe\nS1", "unsafe"), ("safe", "safe"), ("Safe<|eot_id|>", "safe"), ("", None),
])
def test_parse_llamaguard(text, exp):
    assert G.parse_safe_unsafe_first_line(text) == exp


def test_parse_wildguard_polyguard():
    p = G.parse_yes_no_field("Harmful request")
    assert p("Harmful request: yes\nResponse refusal: no\nHarmful response: no") == "unsafe"
    assert p("Harmful request: no\nRequest safety violations: None") == "safe"
    assert p("garbage") is None


def test_parse_guardreasoner_nemotron_granite_gptoss():
    gr = "... reasoning ...\n---\nAnswers: \nRequest: harmful\nCompletion: refusal\nResponse: unharmful"
    assert G.parse_guardreasoner(gr) == "unsafe"
    assert G.parse_guardreasoner("Answers:\nRequest: unharmful\n") == "safe"
    assert G.parse_nemotron_json('{"User Safety": "unsafe", "Safety Categories": "S3"}') == "unsafe"
    assert G.parse_nemotron_json('{"User Safety": "safe"}') == "safe"
    assert G.parse_nemotron_json("not json") is None
    assert G.parse_granite("<think>...</think><score> yes </score>") == "unsafe"
    assert G.parse_granite("No") == "safe"
    assert G.parse_gpt_oss("<|channel|>analysis<|message|>thinking...<|end|><|start|>assistant<|channel|>final<|message|>UNSAFE<|return|>") == "unsafe"
    assert G.parse_gpt_oss("SAFE") == "safe"


def test_registry_shapes():
    for name in G.B1_ORDER:
        s = G.get(name)
        assert s.repo_id and s.kind in ("causal", "stream", "vlm", "classifier", "yesno")
        if s.kind in ("causal", "yesno"):
            assert callable(s.build)
        if s.kind == "causal":
            assert callable(s.parse)
    assert len(G.B1_ORDER) == 12


# ---- summarize / compare --------------------------------------------------------------------
def test_summarize_strict_loose_and_parse_fail():
    rows = [
        {"label": 1, "pred": "unsafe", "score": 0.9, "n_decode": 3, "meta": {"adversarial": True}},
        {"label": 1, "pred": "controversial", "score": 0.6, "n_decode": 3, "meta": {"adversarial": False}},
        {"label": 0, "pred": "safe", "score": 0.1, "n_decode": 3, "meta": {"adversarial": False}},
        {"label": 0, "pred": None, "score": 0.2, "n_decode": 3, "meta": {"adversarial": True}},
    ]
    s = summarize(rows, "wgtest")
    assert s["n"] == 4 and s["parse_fail"] == 1 and abs(s["parse_fail_rate"] - 0.25) < 1e-9
    assert s["recall_strict"] == 1.0 and s["recall_loose"] == 0.5
    assert "auprc" in s and "f1_loose_adversarial" in s and "f1_loose_vanilla" in s


def test_compare_pass_fail_na():
    pub = {"tolerance_pt": 1.0, "guards": {"g": {"wgtest": {"f1": 88.0, "mode": "loose"}}}}
    rows = [
        {"guard": "g", "bench": "wgtest", "n": 10, "f1_strict": 0.90, "f1_loose": 0.885, "parse_fail_rate": 0.0, "mean_decode_tokens": 2, "auprc": 0.9},
        {"guard": "g", "bench": "toxicchat", "n": 10, "f1_strict": 0.5, "f1_loose": 0.5, "parse_fail_rate": 0.1, "mean_decode_tokens": 2},
        {"guard": "h", "bench": "wgtest", "n": 10, "f1_strict": 0.5, "f1_loose": 0.5, "parse_fail_rate": 0.0, "mean_decode_tokens": 2},
    ]
    cmp = C.compare(rows, pub)
    st = {(r["guard"], r["bench"]): r["status"] for r in cmp}
    assert st[("g", "wgtest")] == "PASS" and st[("g", "toxicchat")] == "n/a" and st[("h", "wgtest")] == "n/a"
    pub["guards"]["g"]["wgtest"]["f1"] = 95.0
    assert {r["status"] for r in C.compare(rows[:1], pub)} == {"FAIL"}
    md = C.to_markdown(cmp, 1.0)
    assert "| g | wgtest |" in md


# ---- offline smoke run with a tiny random model ----------------------------------------------
def _tiny_repo(tmp_path: Path) -> Path:
    """Save a tiny Qwen3 + a tiny BPE tokenizer with a chat template into a local 'repo' dir."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from lgs1.lgcore import tiny_qwen3
    d = tmp_path / "tiny-guard"
    d.mkdir()
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<pad>", "<eos>", "<|im_start|>", "<|im_end|>"])
    corpus = ["how do i make a bomb", "what is the weather today", "Safety: Unsafe", "Safety: Safe",
              "user assistant system", "tell me a story about a cat", "categories none violent"] * 20
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    fast.save_pretrained(d)
    model = tiny_qwen3(seed=0, vocab=512)
    model.config.pad_token_id = fast.pad_token_id
    model.config.eos_token_id = fast.eos_token_id
    model.save_pretrained(d)
    return d


def test_smoke_run_offline(tmp_path, monkeypatch):
    from lgs1.zeroshot import run as R
    repo = _tiny_repo(tmp_path)
    bench = tmp_path / "bench.jsonl"
    with open(bench, "w") as f:
        for i in range(6):
            f.write(json.dumps({"prompt": f"prompt number {i} about cats", "prompt_harm_label": "harmful" if i % 2 else "unharmful",
                                "adversarial": bool(i % 3)}) + "\n")
    spec = G.GuardSpec(name="tiny_guard", repo_id=str(repo), kind="causal", max_new_tokens=4, batch_size=4,
                       published_key="", build=G.build_user_chat, parse=G.parse_qwen3guard, first_token=("Unsafe", "Safe"))
    monkeypatch.setitem(G.GUARDS, "tiny_guard", spec)
    monkeypatch.setenv("LG_OUT", str(tmp_path / "out"))
    out = tmp_path / "zs"
    s = R.run("tiny_guard", ["wgtest"], limit=None, batch=4, dtype="float32", show_prompt=False,
              paths={"wgtest": str(bench), "toxicchat": None}, allow_online=False, out_root=str(out))
    assert "wgtest" in s and s["wgtest"]["n"] == 6
    rows = [json.loads(l) for l in open(out / "tiny_guard" / "wgtest.rows.jsonl")]
    assert len(rows) == 6
    for r in rows:
        for k in ("model_sha", "cfg_hash", "code_commit", "pred", "raw", "score", "n_decode"):
            assert k in r
        assert r["score"] is None or 0.0 <= r["score"] <= 1.0
    # resume: a second run adds nothing and keeps the lineage
    s2 = R.run("tiny_guard", ["wgtest"], None, 4, "float32", False, {"wgtest": str(bench), "toxicchat": None}, False, str(out))
    rows2 = [json.loads(l) for l in open(out / "tiny_guard" / "wgtest.rows.jsonl")]
    assert len(rows2) == 6 and s2["wgtest"]["n"] == 6
    assert (out / "tiny_guard" / "wgtest.summary.json").exists()
