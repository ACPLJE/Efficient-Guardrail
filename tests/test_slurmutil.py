"""Partition tiers from a real `sinfo` snapshot (2026-09-17) with QOS filtering via the cluster notes."""
from lgs1 import slurmutil as SU

SINFO = """cpu-max10*|gpu:RTX2080Ti:8|up|drng|2
cpu-max16|gpu:RTX3090:4|up|mix|9
RTX2080Ti*|gpu:RTX2080Ti:8|up|drng|2
RTX2080Ti*|gpu:RTX2080Ti:8|up|mix|3
TITANRTX|gpu:TITANRTX:4|up|mix-|1
RTX3090|gpu:RTX3090:4|up|mix|9
RTX3090|gpu:RTX3090:8|up|mix|16
RTX3090|gpu:RTX3090:4|up|alloc|1
A5000|gpu:A5000:8|up|mix|7
A6000|gpu:A6000:8|up|mix-|6
A100-40GB-PCIe|gpu:A100-PCIe-40GB:4|up|mix-|2
4A100|gpu:A100-SXM4-40GB:8|up|drain*|1
A100-40GB|gpu:A100-SXM4-40GB:8|up|mix|1
A100-80GB|gpu:A100-SXM4-80GB:8|up|mix-|10
A100-80GB|gpu:A100-SXM4-80GB:8|up|alloc|3
RTX6000ADA|gpu:RTX6000ADA:8|up|mix-|10
RTX4090|gpu:RTX4090:4|up|mix|4
RTX4090|gpu:RTX4090:4|up|idle|1
L40S|gpu:L40S:8|up|mix-|3
L40S|gpu:L40S:8|up|alloc|3
H200|gpu:H200:8|up|mix-|1
H200|gpu:H200:8|up|drain*|1
H200-ZT|gpu:H200:8|up|mix-|2
H200-PCIe-ZT|gpu:H200-PCIe:8|up|mix-|1
"""


def test_parse_sinfo_states_and_memory():
    parts = SU.parse_sinfo(SINFO)
    assert parts["RTX6000ADA"]["mem_gb"] == 48 and parts["RTX6000ADA"]["mix"] == 10
    assert parts["RTX4090"]["idle"] == 1 and parts["RTX4090"]["mem_gb"] == 24
    assert parts["4A100"]["down"] == 1 and parts["4A100"]["idle"] + parts["4A100"]["mix"] + parts["4A100"]["alloc"] == 0
    assert parts["A100-40GB-PCIe"]["mem_gb"] == 40
    assert "cpu-max16" in parts  # excluded later by prefix


def test_tiers_without_scontrol_use_cluster_notes():
    parts = SU.parse_sinfo(SINFO)
    t8, _ = SU.pick_partitions(8, "normal", parts, use_scontrol=False)
    t24, _ = SU.pick_partitions(24, "normal", parts, use_scontrol=False)
    t40, _ = SU.pick_partitions(40, "normal", parts, use_scontrol=False)
    t48, _ = SU.pick_partitions(48, "normal", parts, use_scontrol=False)
    assert set(t24) == {"RTX6000ADA", "A6000", "L40S", "A5000", "RTX4090", "A100-40GB-PCIe"}
    assert set(t40) == {"RTX6000ADA", "A6000", "L40S", "A100-40GB-PCIe"}
    assert set(t48) == {"RTX6000ADA", "A6000", "L40S"}
    assert t8 == t24                       # nothing below 24 GB is on the QOS-normal list
    assert t24[0] == "RTX4090"             # the idle node sorts first (readability only)
    for lst in (t8, t24, t40, t48):
        assert "4A100" not in lst and "H200" not in lst and not any(p.startswith("cpu-") for p in lst)


def test_override_wins(monkeypatch):
    monkeypatch.setenv("LG_PARTITIONS_OVERRIDE", "L40S, A6000")
    chosen, why = SU.pick_partitions(48, "normal", SU.parse_sinfo(SINFO), use_scontrol=False)
    assert chosen == ["L40S", "A6000"] and "OVERRIDE" in why[0]


def test_bf16_tier_drops_turing_gpus(monkeypatch):
    """2026-09-19: TITAN RTX (Turing) ran the 0.6B head at 2.8 s/step; real-model jobs skip GPUs without bf16 cores."""
    parts = SU.parse_sinfo(SINFO)
    monkeypatch.setattr(SU, "KNOWN_NORMAL", SU.KNOWN_NORMAL + ["TITANRTX", "RTX3090"])
    with_t, _ = SU.pick_partitions(24, "normal", parts, use_scontrol=False, bf16=False)
    without, reasons = SU.pick_partitions(24, "normal", parts, use_scontrol=False, bf16=True)
    assert "TITANRTX" in with_t and "TITANRTX" not in without and "RTX3090" in without
    assert any("TITANRTX: TITANRTX has no bf16" in r for r in reasons)
    assert set(with_t) - set(without) == {"TITANRTX"}
