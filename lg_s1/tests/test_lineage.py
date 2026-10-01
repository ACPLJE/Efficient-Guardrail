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
