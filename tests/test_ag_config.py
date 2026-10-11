"""SPEC 8: experiment YAML, system overlays, config_hash, run_id, job expansion."""
import pytest
import yaml

from _ag_util import SKELETON, SYSTEMS, small_exp
from ag.config.schema import ConfigError, deep_merge, load_exp, load_system
from ag.sweep.expand import config_hash, expand, make_run_id


def test_skeleton_loads_and_expands():
    exp = load_exp(SKELETON)
    jobs = expand(exp, ts="20261011T000000")
    assert [j.system_id for j in jobs] == ["B0", "AG"]
    b0, ag = jobs
    assert ag.parent_run_ids == (b0.run_id,) and b0.parent_run_ids == ()
    parts = ag.run_id.split("__")
    assert parts[0] == "mock" and parts[1] == "AG" and parts[2] == "MT" and parts[5] == "s00r0"
    assert len(parts[4]) == 8 and parts[4] == ag.config_hash[:8]


def test_overlay_merges_into_base():
    s = load_system("mock_b1a")
    assert s["system_id"] == "B1a" and s["impl"] == "mock"
    assert s["flags"]["block_before_generation"] is True
    assert s["flags"]["selector"]["rule"] == "all" and s["flags"]["selector"]["k"] == 8   # deep merge keeps k


def test_config_hash_excludes_notes_shard_repeat():
    exp = load_exp(SKELETON)
    j = expand(exp, ts="t")[0].resolved
    h = config_hash(j)
    assert config_hash({**j, "notes": "other", "shard": 3, "repeat": 2}) == h
    assert config_hash(deep_merge(j, {"slurm": {"time": "99:00:00"}})) == h
    assert config_hash(deep_merge(j, {"fairness": {"max_new_tokens": 1}})) != h


def test_job_axes_and_shards(tmp_path):
    p = small_exp(tmp_path, data={"length_classes": ["1k", "2k"]}, sweep={"job_axes": ["data.length_classes"]},
                  slurm={"shards": 2})
    jobs = expand(load_exp(p), ts="t")
    assert len(jobs) == 2 * 2 * 2                         # systems x length classes x shards
    assert len({j.run_id for j in jobs}) == 8
    assert all(len(j.resolved["data"]["length_classes"]) == 1 for j in jobs)
    for j in jobs:
        if j.system_id == "AG":
            (pid,) = j.parent_run_ids
            p_job = next(x for x in jobs if x.run_id == pid)
            assert (p_job.shard, p_job.data_tag) == (j.shard, j.data_tag)


def test_repeat_count_not_in_hash():
    exp = load_exp(SKELETON)
    h1 = expand(exp, ts="t")[0].config_hash
    h3 = [j.config_hash for j in expand({**exp, "repeats": 3}, ts="t") if j.repeat == 0][0]
    assert h1 == h3


def test_yaml_off_is_normalised(tmp_path):
    from ag.judges.base import VerdictCache
    (tmp_path / "base.yaml").write_text((SYSTEMS / "mock_ag.yaml").read_text())
    (tmp_path / "x.yaml").write_text("base: base.yaml\nsystem_id: X\noverride:\n  verdict_cache: off\n"
                                     "  output_guard: off\n")
    s = load_system(str(tmp_path / "x.yaml"))
    assert s["flags"]["verdict_cache"] == "off" and s["flags"]["output_guard"] == "off"
    (tmp_path / "y.yaml").write_text("base: base.yaml\nsystem_id: Y\noverride:\n  verdict_cache: sometimes\n")
    with pytest.raises(ConfigError, match="verdict_cache"):
        load_system(str(tmp_path / "y.yaml"))
    assert VerdictCache(False).policy == "off"
    with pytest.raises(ValueError):
        VerdictCache("bogus")


def test_validation_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"exp_id": "x", "systems": ["mock_b0"], "targets": ["MT"], "typo_key": 1}))
    with pytest.raises(ConfigError):
        load_exp(bad)
    with pytest.raises(ValueError):
        make_run_id("E__1", "B0", "T1", "d", "abcdef0123", 0, 0, 0, "ts")
