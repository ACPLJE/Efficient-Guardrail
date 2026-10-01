"""OOM guard + token-budget batching (2026-09-18: gpt-oss-20b OOM at 3.4k padded tokens on a 48 GB card)
and resume across a code change (code_commit is stamped but no longer blocks a resume)."""
import json

import pytest

import lgs1.lineage as LN
from lgs1.zeroshot import guards as G
from lgs1.zeroshot import run as R


def test_make_chunks_fixed_size_keeps_order():
    todo = [{"id": i} for i in range(7)]
    ch = R.make_chunks(todo, 3)
    assert [[r["id"] for r in c] for c in ch] == [[0, 1, 2], [3, 4, 5], [6]]


def test_make_chunks_token_budget_sorts_and_caps():
    todo = [{"id": i} for i in range(6)]
    lengths = [900, 100, 120, 1500, 110, 800]
    ch = R.make_chunks(todo, 4, lengths, max_tokens=1000)
    ids = [[r["id"] for r in c] for c in ch]
    assert sorted(x for c in ids for x in c) == list(range(6))          # every row exactly once
    for c in ch:
        L = [lengths[r["id"]] for r in c]
        assert len(c) <= 4 and (len(c) == 1 or max(L) * len(c) <= 1000)  # cap on batch x padded length
    assert ids[0] == [1, 4, 2]                                           # shortest first: 3 x 120 <= 1000
    assert [3] in ids and [0] in ids                                      # 900 and 1500 cannot share a batch


class _OOM(RuntimeError):
    pass


def test_score_chunk_safe_retries_singly_then_error_row(monkeypatch):
    spec = G.GuardSpec(name="t", repo_id="x", kind="classifier")
    calls = []

    def fake_classifier(model, prompts):
        calls.append(list(prompts))
        if len(prompts) > 1 or prompts[0] == "big":
            raise _OOM("CUDA out of memory. Tried to allocate 606.00 MiB")
        return [{"raw": "ok", "n_decode": 0, "score": 0.1, "label": "safe"} for _ in prompts]

    monkeypatch.setattr(R, "classifier_batch", fake_classifier)
    chunk = [{"id": "a", "prompt": "small"}, {"id": "b", "prompt": "big"}, {"id": "c", "prompt": "small2"}]
    res = R.score_chunk_safe(spec, None, None, "cpu", chunk)
    assert [x.get("label") for x in res] == ["safe", None, "safe"]
    assert res[1]["error"] == "oom" and res[1]["raw"].startswith("[error] CUDA OOM")
    assert calls[0] == ["small", "big", "small2"] and calls[1:] == [["small"], ["big"], ["small2"]]
    s = R.summarize([{"label": 1, "pred": x.get("label"), "score": x.get("score"), "meta": {}, **({"error": x["error"]} if x.get("error") else {})}
                     for x in res], "wgtest")
    assert s["n_error"] == 1 and s["parse_fail"] == 1


def test_score_chunk_safe_reraises_non_oom(monkeypatch):
    spec = G.GuardSpec(name="t", repo_id="x", kind="classifier")
    monkeypatch.setattr(R, "classifier_batch", lambda m, p: (_ for _ in ()).throw(ValueError("bad parse")))
    with pytest.raises(ValueError):
        R.score_chunk_safe(spec, None, None, "cpu", [{"id": "a", "prompt": "p"}])


def test_resume_across_code_change_warns_but_continues(tmp_path, capsys):
    lin_a = LN.Lineage("m", "r", "c", "p", "d", "nogit-aaaa")
    lin_b = LN.Lineage("m", "r", "c", "p", "d", "nogit-bbbb")      # same model/cfg, new code hash
    rows = tmp_path / "rows.jsonl"
    with LN.RowsWriter(rows, lin_a) as w:
        w.write({"split": "test", "subset": "s", "id": "a"})
    assert LN.assert_resume_compatible(rows, lin_b) == 1
    assert "across a code change" in capsys.readouterr().err
    with LN.RowsWriter(rows, lin_b, resume=True) as w:
        w.write({"split": "test", "subset": "s", "id": "b"})
    got = [json.loads(l) for l in rows.read_text().splitlines()]
    assert [r["code_commit"] for r in got] == ["nogit-aaaa", "nogit-bbbb"]   # both versions recorded row by row
    with pytest.raises(LN.LineageMismatch):                                  # strict mode still refuses
        LN.assert_resume_compatible(rows, lin_b, keys=LN.LINEAGE_KEYS)
    lin_c = LN.Lineage("m", "r", "c2", "p", "d", "nogit-bbbb")             # cfg change: still refused
    with pytest.raises(LN.LineageMismatch):
        LN.assert_resume_compatible(rows, lin_c)


def _fake_vlm_parts():
    import torch
    from transformers import AutoConfig
    from types import SimpleNamespace

    class Enc(dict):
        def to(self, dev):
            return self

    class Proc:
        def apply_chat_template(self, msgs, **kw):
            assert msgs[0]["content"][0]["text"] == "hello"
            return Enc(input_ids=torch.tensor([[5, 6, 7]]), attention_mask=torch.ones(1, 3, dtype=torch.long))

        def decode(self, ids, skip_special_tokens=True):
            return "unsafe\nS1" if len(ids) else ""

    class Model:
        config = AutoConfig.for_model("llama", num_hidden_layers=2, hidden_size=16, num_attention_heads=2, intermediate_size=32)
        generation_config = SimpleNamespace(cache_implementation="hybrid")     # what Llama-Guard-4 declares
        fail = {}          # strategy name -> exception to raise
        seen = []

        def generate(self, input_ids, attention_mask, **kw):
            name = ("nocache" if kw.get("use_cache") is False else "plain" if "past_key_values" in kw and getattr(kw["past_key_values"], "_tag", None) == "plain"
                    else "config" if "past_key_values" in kw else "default")
            Model.seen.append(name)
            if self.generation_config.cache_implementation is not None and "past_key_values" in kw:
                raise ValueError("Passing both `cache_implementation` (used to initialize certain caches) and "
                                 "`past_key_values` (a Cache object) is unsupported. Please use only one of the two.")
            if name in self.fail:
                raise self.fail[name]
            return torch.cat([input_ids, torch.tensor([[9, 9]])], dim=1)

    return Proc, Model


def test_vlm_one_clears_hybrid_declaration_and_uses_plain_dynamic_cache(monkeypatch):
    from transformers import DynamicCache
    Proc, Model = _fake_vlm_parts()
    spec = G.GuardSpec(name="lg4", repo_id="x", kind="vlm", max_new_tokens=16, parse=G.parse_safe_unsafe_first_line)
    m = Model()
    assert R.neutralise_cache_impl(m) == ["SimpleNamespace.cache_implementation=None"]   # cleared at load time (load_model)
    assert m.generation_config.cache_implementation is None
    R._VLM_STRATEGY.clear()
    r = R.vlm_one(spec, Proc(), m, "cpu", "hello")
    assert Model.seen == ["config"] or Model.seen == ["plain"]                     # a DynamicCache was passed first
    assert r["raw"] == "unsafe\nS1" and r["n_decode"] == 2


def test_vlm_generate_ladder_falls_through_and_remembers():
    Proc, Model = _fake_vlm_parts()
    spec = G.GuardSpec(name="lg4", repo_id="x", kind="vlm", max_new_tokens=16, parse=G.parse_safe_unsafe_first_line)
    m = Model()
    m.generation_config.cache_implementation = None
    # cluster 2026-09-18: both DynamicCache variants and the default path die inside the cache/mask code
    err = TypeError("'>=' not supported between instances of 'int' and 'NoneType'")
    Model.fail = {"plain": err, "config": err, "default": err}
    R._VLM_STRATEGY.clear()
    Model.seen.clear()
    r = R.vlm_one(spec, Proc(), m, "cpu", "hello")
    assert r["raw"] == "unsafe\nS1"
    assert Model.seen[-1] == "nocache" and len(Model.seen) == 4
    Model.seen.clear()
    R.vlm_one(spec, Proc(), m, "cpu", "hello")
    assert Model.seen == ["nocache"]                                              # remembered: no re-trying failures
    # a hybrid declaration that comes back is cleared on the way down the ladder
    m.generation_config.cache_implementation = "hybrid"
    Model.fail = {}
    R._VLM_STRATEGY.clear()
    Model.seen.clear()
    R.vlm_one(spec, Proc(), m, "cpu", "hello")
    assert m.generation_config.cache_implementation is None and Model.seen[-1] in ("config", "plain")


def test_vlm_generate_reraises_oom():
    Proc, Model = _fake_vlm_parts()
    spec = G.GuardSpec(name="lg4", repo_id="x", kind="vlm", max_new_tokens=16)
    m = Model()
    m.generation_config.cache_implementation = None
    Model.fail = {"plain": TypeError("CUDA out of memory. Tried to allocate 1 GiB"), "config": TypeError("CUDA out of memory")}
    R._VLM_STRATEGY.clear()
    with pytest.raises(TypeError):
        R.vlm_one(spec, Proc(), m, "cpu", "hello")


def test_llama4_null_chunk_size_patch_is_full_attention():
    """Llama-Guard-4-12B ships attention_chunk_size=null; transformers >= 4.53 then dies in create_chunked_causal_mask.
    neutralise_cache_impl sets a chunk no prompt reaches -> token-identical to an explicit full-attention model."""
    import torch
    pytest.importorskip("transformers.models.llama4")
    from transformers import Llama4Config, Llama4TextConfig, Llama4VisionConfig, Llama4ForConditionalGeneration

    def mk(chunk, layer_types=None):
        tc = Llama4TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, intermediate_size_mlp=64, num_hidden_layers=4,
                              num_attention_heads=2, num_key_value_heads=1, num_local_experts=2, num_experts_per_tok=1,
                              max_position_embeddings=512, attention_chunk_size=chunk, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                              **({"layer_types": layer_types} if layer_types else {}))
        vc = Llama4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32,
                                patch_size=16, vision_output_dim=32, projector_input_dim=32, projector_output_dim=32)
        return Llama4Config(text_config=tc, vision_config=vc, boi_token_index=200, eoi_token_index=201, image_token_index=202)

    torch.manual_seed(0)
    m = Llama4ForConditionalGeneration(mk(None)).eval()
    sd = m.state_dict()
    m.generation_config.cache_implementation = "hybrid"
    changed = R.neutralise_cache_impl(m)
    assert any("cache_implementation=None" in c for c in changed) and any("attention_chunk_size=None->" in c for c in changed)
    for obj in R._config_objects(m):
        assert getattr(obj, "cache_implementation", None) is None
        if hasattr(obj, "attention_chunk_size") and hasattr(obj, "num_hidden_layers"):
            assert obj.attention_chunk_size >= 1 << 20
    ids = torch.randint(3, 250, (1, 100))
    enc = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    R._VLM_STRATEGY.clear()
    out = R.vlm_generate(m, enc, dict(max_new_tokens=6, do_sample=False))
    full = Llama4ForConditionalGeneration(mk(1 << 20, ["full_attention"] * 4)).eval()
    full.load_state_dict(sd)
    assert torch.equal(out, full.generate(**enc, max_new_tokens=6, do_sample=False))
    assert R.neutralise_cache_impl(m) == []                                     # idempotent
