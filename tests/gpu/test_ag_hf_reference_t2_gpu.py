"""GPU smoke of the real-model B0 path: T2 (Qwen3-4B-Instruct-2507) renders a ~2k-token draft LC-Inject request
and generates 16 tokens. Skipped without CUDA (tests/gpu/conftest.py) or without T2 in the HF cache."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
T2_REPO = "Qwen/Qwen3-4B-Instruct-2507"


def _t2_snapshot() -> str | None:
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(T2_REPO, local_files_only=True)
    except Exception:
        return None


def test_t2_renders_and_generates_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    snap = _t2_snapshot()
    if snap is None:
        pytest.skip(f"{T2_REPO} is not in the HF cache (download on the login node)")
    from ag.data.lc_inject.background import load_docs
    from ag.data.lc_inject.draft import generate
    from ag.targets.registry import create_hf_target, length_fn_for

    docs = load_docs(ROOT / "tests" / "data" / "bg_docs_small.jsonl.txt")
    req = generate(n_pairs=1, n_benign=0, length_classes=["2k"], docs=docs, length_fn=length_fn_for("T2"))[0]
    tgt = create_hf_target("T2", device="cuda")
    assert tgt.model_sha == Path(snap).name and tgt.dtype == "bfloat16"
    rp = tgt.render(req)
    assert 0.98 * 2048 <= rp.n_tokens <= 2048 and rp.payload_windows
    out = tgt.generate(rp, 16)
    assert out.text.strip() and 1 <= len(out.output_ids) <= 16
    assert out.t_token_ns[0] == out.t_prefill_done_ns and list(out.t_token_ns) == sorted(out.t_token_ns)
    assert tgt.last_timing["prefill_ns"] > 0 and out.peak_mem_bytes > 0
    assert "".join(tgt.token_texts(out.output_ids)) == out.text
    # oracle check (E2.2 needs this loop to be exact): HF generate's greedy tokens on the same prompt ids
    with torch.inference_mode():
        ids = torch.tensor([list(rp.prompt_ids)], device=tgt.device)
        ref = tgt.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=16,
                                 eos_token_id=sorted(tgt.eos_ids), pad_token_id=sorted(tgt.eos_ids)[0])
    assert list(out.output_ids) == ref[0, ids.shape[1]:].tolist()
