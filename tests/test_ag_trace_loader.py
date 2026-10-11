"""Mooncake trace loader, W0-mooncake stats and script (plan 5.5; SPEC 4.10, 11 W0-mooncake; F-11, A-12, D-020).

The fixtures tests/data/mooncake_{conversation,toolagent}_50.jsonl.txt are SYNTHETIC rows in the FAST25
format made by synth_rows() below (tests/data/README.md). Regenerate them with
    PYTHONPATH=. python tests/test_ag_trace_loader.py --write
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from ag.data.traces.mooncake import TraceRow, dump_trace, expected_blocks, load_trace, trace_stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"
SCRIPT = ROOT / "scripts" / "mooncake_stats.py"
FIXTURES = {"conversation": DATA / "mooncake_conversation_50.jsonl.txt",
            "toolagent": DATA / "mooncake_toolagent_50.jsonl.txt"}


# ---------------------------------------------------------------------------------------------------
# synthetic fixture generator (random.Random.random() only: its stream is stable across Python versions)
# ---------------------------------------------------------------------------------------------------
def synth_rows(kind: str, n: int = 50) -> list[TraceRow]:
    """conversation: multi-turn rows that extend an earlier row's full-block prefix, 4 inputs >= 32768
    (one in the D-020 128k bin). toolagent: a shared 3-block tool prefix on most rows, many outputs <= 3
    tokens, 3 inputs >= 32768. Both: timestamps strictly increasing from 0, hash_ids consistent with
    512-token blocks except exactly 2 rows (one id too many, one too few)."""
    conv = kind == "conversation"
    rnd = random.Random(1 if conv else 2)
    u = rnd.random
    big = {7: 33000, 19: 41000, 31: 70500, 44: 36000} if conv else {11: 34000, 27: 52000, 38: 33500}
    mismatch = {13: +1, 36: -1} if conv else {9: +1, 42: -1}
    next_id = [0 if conv else 3]                     # toolagent ids 0..2 are the shared tool prefix

    def fresh(k: int) -> list[int]:
        ids = list(range(next_id[0], next_id[0] + k))
        next_id[0] += k
        return ids

    rows: list[TraceRow] = []
    t = 0
    for i in range(n):
        t = 0 if i == 0 else t + 1 + int(u() * (600 if conv else 300))
        if conv:
            out = 1 + int(u() * u() * 800)
        else:
            out = 1 + int(u() * 3) if u() < 0.25 else 1 + int(u() * 300)
        if conv:
            prev = None
            if i > 0 and i not in big and u() < 0.35:
                j = i - 1 - int(u() * min(5, i))
                prev = rows[j] if j not in mismatch else None
            if prev is not None:
                length = prev.input_length + 200 + int(u() * 3000)
                full = prev.input_length // 512
                ids = list(prev.hash_ids[:full]) + fresh(expected_blocks(length) - full)
            else:
                length = big.get(i, 300 + int(u() * u() * 20000))
                ids = fresh(expected_blocks(length))
        else:
            length = big.get(i, 1536 + int(u() * 14000))
            ids = [0, 1, 2] + fresh(expected_blocks(length) - 3) if u() < 0.8 else fresh(expected_blocks(length))
        if i in mismatch:
            ids = ids + fresh(1) if mismatch[i] > 0 else ids[:-1]
        rows.append(TraceRow(t, length, out, tuple(ids)))
    return rows


def _row(ts: int, il: int, ol: int, ids=None) -> TraceRow:
    return TraceRow(ts, il, ol, tuple(ids) if ids is not None else tuple(range(1000 + ts, 1000 + ts + expected_blocks(il))))


# ---------------------------------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------------------------------
def test_roundtrip_jsonl_and_txt(tmp_path):
    rows = synth_rows("conversation", 12)
    for name in ("t.jsonl", "t.jsonl.txt"):
        p = dump_trace(rows, tmp_path / name)
        assert load_trace(p) == rows
    assert all(TraceRow.from_json(r.to_json()) == r for r in rows)
    first = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert list(first) == ["timestamp", "input_length", "output_length", "hash_ids"]   # FAST25 field names


def test_gz_by_suffix_and_by_magic(tmp_path):
    rows = synth_rows("toolagent", 9)
    p = dump_trace(rows, tmp_path / "t.jsonl.gz")
    assert p.read_bytes()[:2] == b"\x1f\x8b"
    assert load_trace(p) == rows
    assert dump_trace(rows, tmp_path / "u.jsonl.gz").read_bytes() == p.read_bytes()      # mtime 0 -> stable bytes
    q = tmp_path / "gz_without_suffix.jsonl"
    q.write_bytes(gzip.compress(b"\n" + "".join(json.dumps(r.to_json()) + "\n\n" for r in rows).encode()))
    assert load_trace(q) == rows


def test_blank_lines_skipped_and_file_order_kept(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [_row(50, 600, 2), _row(10, 100, 1), _row(30, 2000, 5)]                  # not sorted on purpose
    p.write_text("\n" + "\n   \n".join(json.dumps(r.to_json()) for r in rows) + "\n\n")
    assert load_trace(p) == rows
    s = trace_stats(load_trace(p))
    assert not s["timestamps_nondecreasing"] and s["n_timestamp_decreases"] == 1
    assert s["t_first_ms"] == 10 and s["t_last_ms"] == 50


@pytest.mark.parametrize("bad, lineno, needle", [
    ('{"timestamp": 3, "input_length": 4', 3, "malformed JSON"),
    ('{"timestamp": 3, "input_length": 4, "output_length": 1}', 3, "hash_ids"),
    ('[1, 2, 3]', 3, "not a JSON object"),
    ('{"timestamp": 3, "input_length": "4", "output_length": 1, "hash_ids": [1]}', 3, "input_length"),
    ('{"timestamp": 3, "input_length": 4, "output_length": true, "hash_ids": [1]}', 3, "output_length"),
    ('{"timestamp": 3, "input_length": -4, "output_length": 1, "hash_ids": []}', 3, "negative"),
    ('{"timestamp": 3, "input_length": 4, "output_length": 1, "hash_ids": [1.5]}', 3, "hash_ids"),
    ('{"timestamp": 3.5, "input_length": 4, "output_length": 1, "hash_ids": [1]}', 3, "timestamp"),
])
def test_malformed_row_reports_line_number(tmp_path, bad, lineno, needle):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(_row(0, 512, 1).to_json()) + "\n\n" + bad + "\n" + json.dumps(_row(9, 1, 1).to_json()) + "\n")
    with pytest.raises(ValueError) as ei:
        load_trace(p)
    msg = str(ei.value)
    assert f"{p}:{lineno}:" in msg and needle in msg


def test_integral_float_accepted_and_lfs_pointer_rejected(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"timestamp": 3.0, "input_length": 4, "output_length": 1, "hash_ids": [7], "extra": "x"}\n')
    assert load_trace(p) == [TraceRow(3, 4, 1, (7,))]
    q = tmp_path / "lfs.jsonl"
    q.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        load_trace(q)


# ---------------------------------------------------------------------------------------------------
# stats on hand-made traces (exact numbers)
# ---------------------------------------------------------------------------------------------------
LEN_ROWS = [_row(0, 1000, 2), _row(1000, 5000, 8), _row(2000, 40000, 50), _row(4000, 70000, 100)]


def test_stats_lengths_rates_bins_exact():
    s = trace_stats(LEN_ROWS)
    assert s["n"] == 4 and s["duration_s"] == 4.0 and s["timestamps_nondecreasing"]
    assert s["rate_req_s"] == 1.0 and s["input_tok_s"] == 29000.0 and s["output_tok_s"] == 40.0
    # numpy linear percentiles: position q * (n - 1) between sorted values (rounding in the last bits only)
    assert s["input_length"] == pytest.approx({"min": 1000, "p50": 22500.0, "p90": 61000.0, "p95": 65500.0,
                                               "p99": 69100.0, "max": 70000, "mean": 29000.0, "sum": 116000},
                                              rel=1e-12)
    assert (s["input_length"]["min"], s["input_length"]["max"], s["input_length"]["sum"]) == (1000, 70000, 116000)
    o = s["output_length"]
    assert (o["min"], o["p50"], o["max"], o["mean"], o["sum"]) == (2, 29.0, 100, 40.0, 160)
    assert {k: v["count"] for k, v in s["input_ge"].items()} == {"8192": 2, "32768": 2, "65536": 1, "131072": 0}
    assert s["input_ge"]["32768"]["frac"] == 0.5 and s["input_ge"]["65536"]["frac"] == 0.25
    assert s["input_ge"]["8192"]["rate_req_s"] == 0.5
    assert {k: v["count"] for k, v in s["output_le"].items()} == {"3": 1, "8": 2, "32": 2}
    bins = s["length_bins"]
    assert {k: v["count"] for k, v in bins.items()} == {"short": 1, "8k": 1, "32k": 1, "128k": 1}
    assert (bins["8k"]["lo"], bins["8k"]["hi"], bins["128k"]["hi"]) == (4096, 16384, None)
    assert sum(v["frac"] for v in bins.values()) == 1.0


def test_length_bin_edges():
    rows = [_row(i, il, 1) for i, il in enumerate([0, 4095, 4096, 16383, 16384, 65535, 65536, 131072])]
    s = trace_stats(rows)
    assert {k: v["count"] for k, v in s["length_bins"].items()} == {"short": 2, "8k": 2, "32k": 2, "128k": 2}
    assert s["input_ge"]["131072"]["count"] == 1 and s["input_ge"]["8192"]["count"] == 5


BLOCK_ROWS = [TraceRow(0, 1000, 1, (1, 2)),        # ceil(1000/512) = 2  ok
              TraceRow(10, 1536, 1, (1, 2, 3)),    # 3 ok; ids 1, 2 seen
              TraceRow(20, 600, 1, (7,)),          # expected 2 -> mismatch
              TraceRow(30, 512, 1, (1, 8)),        # expected 1 -> mismatch; id 1 seen
              TraceRow(40, 100, 1, (9,))]          # 1 ok


def test_block_check_and_prefix_reuse_exact():
    s = trace_stats(BLOCK_ROWS)
    b = s["hash_ids_block_check"]
    assert (b["block_tokens"], b["n_match"], b["n_mismatch"], b["frac_match"]) == (512, 3, 2, 0.6)
    assert b["frac_match_floor"] == 0.4
    assert b["mismatch_rows"] == [{"row": 2, "input_length": 600, "n_hash_ids": 1, "expected": 2},
                                  {"row": 3, "input_length": 512, "n_hash_ids": 2, "expected": 1}]
    assert b["scan"] == {"256": 0.4, "512": 0.6, "1024": 0.4}
    r = s["prefix_reuse"]
    assert (r["n_first_hash_seen"], r["first_hash_seen_frac"]) == (2, 0.4)
    assert (r["n_unique_hash_ids"], r["n_hash_ids"], r["n_rows_with_hash_ids"]) == (6, 9, 5)
    assert r["mean_seen_frac"] == pytest.approx(7 / 30, abs=1e-15)
    assert r["seen_hash_frac_total"] == pytest.approx(1 / 3, abs=1e-15)
    assert trace_stats(BLOCK_ROWS, block_tokens=1024)["hash_ids_block_check"]["scan"]["1024"] == 0.4


def test_empty_hash_ids_and_single_row():
    s = trace_stats([TraceRow(5, 0, 1, ())])
    assert s["duration_s"] == 0.0 and s["rate_req_s"] is None and s["input_tok_s"] is None
    assert s["hash_ids_block_check"]["n_match"] == 1                                 # ceil(0/512) = 0
    assert s["prefix_reuse"]["mean_seen_frac"] is None and s["prefix_reuse"]["seen_hash_frac_total"] is None
    assert s["windows"]["n_windows"] == 0
    with pytest.raises(ValueError, match="empty"):
        trace_stats([])


def test_windows_exact():
    w = trace_stats(LEN_ROWS, window_s=2.0, window_step_s=1.0)["windows"]
    # starts 0, 1, 2 s; counts 2, 2, 1; input tokens 6000, 45000, 40000 per 2 s
    assert w["n_windows"] == 3
    assert w["rate_req_s"] == {"min": 0.5, "p50": 1.0, "max": 1.0}
    assert w["input_tok_s"] == {"min": 3000.0, "p50": 20000.0, "max": 22500.0}
    assert (w["peak_rate_start_s"], w["peak_input_start_s"]) == (0.0, 1.0)
    assert trace_stats(LEN_ROWS)["windows"]["n_windows"] == 0                        # 4 s trace < 1,200 s window


# ---------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", sorted(FIXTURES))
def test_fixture_matches_generator(tmp_path, kind):
    assert dump_trace(synth_rows(kind), tmp_path / "x.jsonl.txt").read_bytes() == FIXTURES[kind].read_bytes()


@pytest.mark.parametrize("kind", sorted(FIXTURES))
def test_fixture_properties(kind):
    rows = load_trace(FIXTURES[kind])
    assert len(rows) == 50
    assert all(b > a for a, b in zip([r.timestamp_ms for r in rows], [r.timestamp_ms for r in rows][1:]))
    s = trace_stats(rows)
    assert s["hash_ids_block_check"]["n_mismatch"] == 2
    assert [m["n_hash_ids"] - m["expected"] for m in s["hash_ids_block_check"]["mismatch_rows"]] == [1, -1]
    assert 3 <= s["input_ge"]["32768"]["count"] <= 6 and s["input_ge"]["131072"]["count"] == 0
    assert s["prefix_reuse"]["n_first_hash_seen"] >= 5 and s["prefix_reuse"]["mean_seen_frac"] > 0.05
    if kind == "conversation":
        assert s["length_bins"]["128k"]["count"] >= 1 and s["length_bins"]["short"]["count"] >= 1
    else:
        assert s["output_le"]["3"]["count"] >= 5


# ---------------------------------------------------------------------------------------------------
# scripts/mooncake_stats.py
# ---------------------------------------------------------------------------------------------------
def _stats_main(argv):
    spec = importlib.util.spec_from_file_location("mooncake_stats_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(argv)


def _fixture_args(out, ts, name="fixture"):
    return ["--trace", str(FIXTURES["conversation"]), "--trace", str(FIXTURES["toolagent"]),
            "--name", name, "--out", str(out), "--ts", ts]


def _only_dir(out: Path) -> Path:
    dirs = [d for d in (out / "raw" / "W0-mooncake").iterdir() if d.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_script_end_to_end(tmp_path, capsys):
    out = tmp_path / "out"
    assert _stats_main(_fixture_args(out, "20261011T000000")) == 0
    d = _only_dir(out)
    assert capsys.readouterr().out.strip() == str(d)
    exp_id, name, chash, ts = d.name.split("__")
    assert (exp_id, name, ts, len(chash)) == ("W0-mooncake", "fixture", "20261011T000000", 8)
    lin = json.loads((d / "lineage.json").read_text())
    assert lin["run_id"] == d.name and lin["exp_id"] == "W0-mooncake" and lin["config_hash"].startswith(chash)
    assert lin["code_commit"] and lin["code_dirty"] == lin["code_commit"].endswith("-dirty")
    assert (lin["producer"], lin["clock"], lin["seed"], lin["model_sha"]) == ("analysis", "monotonic", 0, {})
    assert lin["finished_unix_ns"] is not None
    data = FIXTURES["conversation"].read_bytes() + FIXTURES["toolagent"].read_bytes()
    assert lin["data_manifest_sha"] == hashlib.sha256(data).hexdigest()[:16]
    st = json.loads((d / "trace_stats.json").read_text())
    assert set(st) == {"run_id", "schema_version", "exp_id", "name", "config", "inputs", "traces"}
    assert set(st["traces"]) == {"mooncake_conversation_50", "mooncake_toolagent_50"}
    assert [(i["key"], i["n_rows"]) for i in st["inputs"]] == [("mooncake_conversation_50", 50),
                                                               ("mooncake_toolagent_50", 50)]
    assert st["inputs"][0]["sha256"] == hashlib.sha256(FIXTURES["conversation"].read_bytes()).hexdigest()
    conv = st["traces"]["mooncake_conversation_50"]
    for k in ("n", "duration_s", "rate_req_s", "input_tok_s", "input_length", "output_length", "input_ge",
              "length_bins", "hash_ids_block_check", "prefix_reuse", "windows"):
        assert k in conv
    assert conv["hash_ids_block_check"]["n_mismatch"] == 2
    assert conv == json.loads(json.dumps(trace_stats(load_trace(FIXTURES["conversation"]))))
    assert "mooncake_stats.py" in (d / "cmd.txt").read_text()


def test_script_deterministic_and_resumable(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _stats_main(_fixture_args(a, "20261011T000001")) == 0
    assert _stats_main(_fixture_args(b, "20261011T000002")) == 0
    da, db = _only_dir(a), _only_dir(b)
    sa, sb = (json.loads((d / "trace_stats.json").read_text()) for d in (da, db))
    assert sa.pop("run_id") != sb.pop("run_id") and sa == sb
    assert da.name.split("__")[2] == db.name.split("__")[2]                         # same config_hash
    la, lb = (json.loads((d / "lineage.json").read_text()) for d in (da, db))
    assert la["data_manifest_sha"] == lb["data_manifest_sha"]
    assert _stats_main(_fixture_args(a, "20261011T000001")) == 0                    # same run_id: resume check passes
    assert _only_dir(a) == da
    # a different name or block size is a different config
    assert _stats_main(_fixture_args(tmp_path / "c", "20261011T000001", name="other")) == 0
    assert _stats_main(_fixture_args(tmp_path / "d", "20261011T000001") + ["--block-tokens", "256"]) == 0
    hashes = {_only_dir(tmp_path / x).name.split("__")[2] for x in ("a", "c", "d")}
    assert len(hashes) == 3


def test_script_trace_keys_and_errors(tmp_path):
    out = tmp_path / "out"
    assert _stats_main(["--trace", f"conv={FIXTURES['conversation']}", "--name", "k", "--out", str(out),
                        "--ts", "20261011T000003"]) == 0
    st = json.loads((_only_dir(out) / "trace_stats.json").read_text())
    assert list(st["traces"]) == ["conv"]
    with pytest.raises(SystemExit, match="duplicate trace keys"):
        _stats_main(["--trace", str(FIXTURES["conversation"]), "--trace", str(FIXTURES["conversation"]),
                     "--name", "k", "--out", str(out)])
    with pytest.raises(SystemExit, match="not found"):
        _stats_main(["--trace", str(tmp_path / "missing.jsonl"), "--name", "k", "--out", str(out)])
    with pytest.raises(SystemExit, match="run_id part"):
        _stats_main(["--trace", str(FIXTURES["conversation"]), "--name", "a__b", "--out", str(out)])


def test_script_runs_as_a_file_from_another_cwd(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "--trace", str(FIXTURES["toolagent"]), "--name", "sub",
                        "--out", str(out), "--ts", "20261011T000004"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    d = Path(r.stdout.strip().splitlines()[-1])
    assert (d / "trace_stats.json").exists() and (d / "lineage.json").exists() and (d / "cmd.txt").exists()


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: PYTHONPATH=. python tests/test_ag_trace_loader.py --write")
    for kind, path in FIXTURES.items():
        print(dump_trace(synth_rows(kind), path))
