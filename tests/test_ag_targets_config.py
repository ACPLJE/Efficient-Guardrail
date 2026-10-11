"""configs/targets/*.yaml and ag.targets.registry (plan 6.1, D-031, D-037; build plan S2 step 0). No torch."""
import pytest
import yaml

from ag.data import mock_lc_inject
from ag.data.render import render
from ag.selector.layers import resolve_layers
from ag.targets.mock import MockTokenizer
from ag.targets.registry import REQUIRED_KEYS, known_targets, length_fn_for, load_target_config

# plan 6.1 [literature]: hf repo, layers, Q heads, KV heads, max context
PLAN = {"T1": ("meta-llama/Llama-3.1-8B-Instruct", 32, 32, 8, 131072),
        "T2": ("Qwen/Qwen3-4B-Instruct-2507", 36, 32, 8, 262144),
        "T3": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", 40, 32, 8, 131072),
        "T4": ("openai/gpt-oss-20b", 24, 64, 8, 131072)}
ALL = (*PLAN, "tiny-hf")


def test_known_targets():
    assert set(ALL) <= set(known_targets())


@pytest.mark.parametrize("tid", ALL)
def test_required_keys_and_tracked8(tid):
    cfg = load_target_config(tid)
    assert set(REQUIRED_KEYS) <= set(cfg) and cfg["target_id"] == tid
    kinds = cfg["layer_attn_kind"]
    if isinstance(kinds, list):
        assert len(kinds) == cfg["n_layers"] and set(kinds) <= {"full", "sliding"}
    else:
        assert kinds == "full"
        kinds = None
    if cfg["tracked8"] is None:                 # D-037: chosen among full layers from config.layer_types at load
        assert tid == "T4"
    else:                                       # D-031: U[round(i*(|U|-1)/7)] over the upper half
        assert tuple(cfg["tracked8"]) == resolve_layers("tracked8", cfg["n_layers"], kinds)


@pytest.mark.parametrize("tid", list(PLAN))
def test_plan_values(tid):
    cfg = load_target_config(tid)
    assert (cfg["hf_repo"], cfg["n_layers"], cfg["n_q_heads"], cfg["n_kv_heads"], cfg["max_context"]) == PLAN[tid]
    assert cfg["revision"] is None and cfg["dtype"] == "bfloat16"


def test_verify_flags_and_tiny():
    assert [load_target_config(t)["verify"] for t in PLAN] == [False, False, True, True]
    t4 = load_target_config("T4")["layer_attn_kind"]
    assert t4.count("sliding") == 12 and t4.count("full") == 12
    tiny = load_target_config("tiny-hf")
    assert tiny["hf_repo"] is None and tiny["build"] == "tiny_hf" and tiny["max_context"] == 4096
    assert tiny["dtype"] == "float32" and tiny["verify"] is False


def test_unknown_target_lists_known_ids():
    with pytest.raises(KeyError) as e:
        load_target_config("T9")
    assert "T2" in str(e.value) and "tiny-hf" in str(e.value)


def test_targets_dir_override(tmp_path, monkeypatch):
    cfg = {k: None for k in REQUIRED_KEYS}
    cfg.update(target_id="X1", n_layers=4, n_q_heads=4, n_kv_heads=2, max_context=64, layer_attn_kind="full",
               dtype="float32", verify=False, notes="")
    (tmp_path / "X1.yaml").write_text(yaml.safe_dump(cfg))
    assert load_target_config("X1", targets_dir=tmp_path)["max_context"] == 64
    monkeypatch.setenv("AG_TARGETS_DIR", str(tmp_path))
    assert load_target_config("X1")["n_layers"] == 4
    (tmp_path / "X2.yaml").write_text(yaml.safe_dump({**cfg, "target_id": "X1"}))
    with pytest.raises(ValueError, match="target_id"):
        load_target_config("X2")
    del cfg["notes"]
    (tmp_path / "X3.yaml").write_text(yaml.safe_dump({**cfg, "target_id": "X3"}))
    with pytest.raises(ValueError, match="notes"):
        load_target_config("X3")


def test_mock_length_fn_equals_render():
    fn = length_fn_for("MT")
    for req in mock_lc_inject.generate(n_pairs=2, length_classes=["1k"]):
        assert fn(req.segments) == len(render(req, MockTokenizer()).prompt_ids)
