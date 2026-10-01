"""E0-2: every row carries the lineage; resume/rescore under a changed lineage is refused (exit 2)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lgs1 import lineage as LN


def _fake_ckpt(tmp_path: Path, name: str, payload: bytes) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text('{"hidden_size": 64}')
    (d / "model.safetensors").write_bytes(payload)
    return d


def test_rows_carry_all_lineage_fields(tmp_path):
    ck = _fake_ckpt(tmp_path, "ckpt", b"\x00" * 1024)
    lin = LN.Lineage.build(str(ck), None, {"k": 32, "b": 128}, {"L": [1024]}, None, str(tmp_path))
    rows = tmp_path / "rows.jsonl"
    with LN.RowsWriter(rows, lin) as w:
        for i in range(3):
            w.write({"split": "test", "subset": "main", "id": f"r{i}", "score": 0.1 * i})
    got = list(LN.iter_rows(rows))
    assert len(got) == 3
    for r in got:
        for k in LN.LINEAGE_KEYS:
            assert k in r and r[k]
        assert r["model_sha"] == lin.model_sha and r["cfg_hash"] == lin.cfg_hash


def test_resume_with_same_lineage_ok_and_dedup_ids(tmp_path):
    ck = _fake_ckpt(tmp_path, "ckpt", b"\x01" * 1024)
    lin = LN.Lineage.build(str(ck), None, {"k": 32}, {}, None, str(tmp_path))
    rows = tmp_path / "rows.jsonl"
    with LN.RowsWriter(rows, lin) as w:
        w.write({"split": "test", "subset": "main", "id": "a"})
    assert LN.assert_resume_compatible(rows, lin) == 1
    with LN.RowsWriter(rows, lin, resume=True) as w:
        w.write({"split": "test", "subset": "main", "id": "b"})
    assert LN.done_ids(rows) == {"a", "b"}


def test_resume_with_changed_cfg_is_refused(tmp_path):
    ck = _fake_ckpt(tmp_path, "ckpt", b"\x02" * 1024)
    lin_a = LN.Lineage.build(str(ck), None, {"k": 32}, {}, None, str(tmp_path))
    lin_b = LN.Lineage.build(str(ck), None, {"k": 16}, {}, None, str(tmp_path))   # deliberate change
    rows = tmp_path / "rows.jsonl"
    with LN.RowsWriter(rows, lin_a) as w:
        w.write({"split": "test", "subset": "main", "id": "a"})
    with pytest.raises(LN.LineageMismatch):
        LN.assert_resume_compatible(rows, lin_b)
    with pytest.raises(LN.LineageMismatch):
        with LN.RowsWriter(rows, lin_b, resume=True):
            pass


def test_model_change_detected_and_cache_is_used(tmp_path):
    ck = _fake_ckpt(tmp_path, "ckpt", b"\x03" * 4096)
    f1 = LN.dir_fingerprint(ck)
    assert (ck / ".lineage_cache.json").exists()
    assert LN.dir_fingerprint(ck) == f1                      # cache hit, same value
    (ck / "model.safetensors").write_bytes(b"\x04" * 4096)   # same size, new content/mtime
    assert LN.dir_fingerprint(ck) != f1


def test_cli_exit_code_2_on_mismatch(tmp_path):
    ck = _fake_ckpt(tmp_path, "ckpt", b"\x05" * 1024)
    lin = LN.Lineage.build(str(ck), None, {"k": 32}, {}, None, str(tmp_path))
    rows = tmp_path / "rows.jsonl"
    with LN.RowsWriter(rows, lin) as w:
        w.write({"split": "test", "subset": "main", "id": "a"})
    cfg_ok = tmp_path / "ok.json"; cfg_ok.write_text(json.dumps({"k": 32}))
    cfg_bad = tmp_path / "bad.json"; cfg_bad.write_text(json.dumps({"k": 8}))
    root = Path(__file__).resolve().parent.parent
    base = [sys.executable, "-m", "lgs1.lineage", "check", "--rows", str(rows), "--model", str(ck),
            "--code-dir", str(tmp_path)]
    ok = subprocess.run(base + ["--cfg", str(cfg_ok)], cwd=root, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run(base + ["--cfg", str(cfg_bad)], cwd=root, capture_output=True, text=True)
    assert bad.returncode == 2 and "MISMATCH" in bad.stderr


def test_rows_without_lineage_are_rejected(tmp_path):
    rows = tmp_path / "legacy.jsonl"
    rows.write_text(json.dumps({"id": "x", "score": 0.5}) + "\n")
    lin = LN.Lineage("m", "r", "c", "p", "d", "g")
    with pytest.raises(LN.LineageMismatch):
        LN.assert_resume_compatible(rows, lin)


def test_iter_rows_tolerates_partial_writes(tmp_path):
    """2026-09-20: `longbench --summarize` crashed on a rows file another job was appending (NUL block / cut line)."""
    from lgs1.lineage import iter_rows
    p = tmp_path / "x.rows.jsonl"
    p.write_text('{"id": 1}\n{"id": 2}\n{"id": 3, "x": "trunc', encoding="utf-8")
    assert [r["id"] for r in iter_rows(p)] == [1, 2]                                 # truncated last line skipped
    p.write_bytes(b'{"id": 1}\n\x00\x00\x00\x00\n{"id": 2}\n')
    assert [r["id"] for r in iter_rows(p, tolerant=True)] == [1, 2]                  # NUL block skipped
    assert [r["id"] for r in iter_rows(p)] == [1, 2]                                 # NUL-only lines are blank after stripping
    p.write_text('{"id": 1}\nnot json\n{"id": 2}\n', encoding="utf-8")
    assert [r["id"] for r in iter_rows(p, tolerant=True)] == [1, 2]
    import pytest
    with pytest.raises(ValueError):
        list(iter_rows(p))                                                            # malformed in the middle: strict raises


def test_dedupe_rows_keeps_one_scored_row_per_id():
    from lgs1.lineage import dedupe_rows
    rows = [{"id": "a", "pred": None, "error": "OOM"}, {"id": "b", "pred": 1}, {"id": "a", "pred": 0},   # a: error then scored
            {"id": "b", "pred": 0}, {"id": "c", "verdict": None}, {"id": "c", "verdict": True}, {"no_id": 1}]
    out = dedupe_rows(rows)
    assert [r.get("id") for r in out] == ["a", "b", "c", None]
    assert out[0]["pred"] == 0 and out[1]["pred"] == 1 and out[2]["verdict"] is True      # scored wins; first scored stays
