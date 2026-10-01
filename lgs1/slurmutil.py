"""Partition selection for sbatch: every partition the QOS allows whose GPUs are big enough.

Slurm accepts a comma-separated partition list; the job starts on whichever listed partition has a
free slot first.  We build that list from `sinfo` (GPU model per partition -> memory table) and
`scontrol show partition` (AllowQos/DenyQos), and fall back to the cluster notes when the queries
fail.  Override at any time with LG_PARTITIONS_OVERRIDE="A,B,C" in env/s1.env.

    python s1ctl.py partitions          # prints the tiers with the reasons

GPU memory tiers used by S1 (bf16, short prompts):
    8  GB  tests (tiny model; CPU would do, but QOS normal insists on --gres=gpu:1)
    24 GB  0.6B-9B guards (weights <= 18 GB)
    40 GB  Llama Guard 4-12B (24 GB weights + activations)
    48 GB  gpt-oss-safeguard-20b dequantised to bf16 (~42 GB) on non-Hopper GPUs
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

# GPU model (as it appears in sinfo GRES) -> memory in GB
GPU_MEM_GB: Dict[str, int] = {
    "RTX2080Ti": 11, "TITANRTX": 24, "RTX3090": 24, "A5000": 24, "RTX4090": 24,
    "A6000": 48, "RTX6000ADA": 48, "L40S": 48,
    "A100-PCIe-40GB": 40, "A100-SXM4-40GB": 40, "A100-SXM4-80GB": 80, "A100": 40,
    "H200": 141, "H200-PCIe": 141, "H100": 80,
}
# GPUs without bf16 tensor cores (Turing, sm_75): bf16 autocast falls back to slow kernels (0.6B head 2.8 s/step on a
# TITAN RTX vs ~0.3 s on Ampere/Ada, 2026-09-19).  Excluded from every real-model job unless LG_ALLOW_NO_BF16=1.
NO_BF16_GPUS = {"RTX2080Ti", "TITANRTX", "TITANV", "V100", "T4"}
# partitions the cluster notes say QOS `normal` may use (used when scontrol gives no answer)
KNOWN_NORMAL: List[str] = ["RTX6000ADA", "A6000", "L40S", "A5000", "RTX4090", "A100-40GB-PCIe"]
EXCLUDE_PREFIX = ("cpu-",)          # CPU partitions that happen to list GPUs


def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def parse_sinfo(text: str) -> Dict[str, Dict]:
    """Parse `sinfo -h -o '%P|%G|%a|%t|%D'` lines into {partition: {gpu, mem_gb, up, idle, mix, nodes}}."""
    parts: Dict[str, Dict] = {}
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 5:
            continue
        name, gres, avail, state, nodes = cols[:5]
        name = name.rstrip("*")
        m = re.search(r"gpu:([A-Za-z0-9\-]+):(\d+)", gres)
        gpu = m.group(1) if m else None
        p = parts.setdefault(name, {"gpu": gpu, "mem_gb": GPU_MEM_GB.get(gpu or "", 0), "up": avail.lower() == "up",
                                    "idle": 0, "mix": 0, "alloc": 0, "down": 0, "nodes": 0})
        try:
            n = int(nodes)
        except ValueError:
            n = 0
        st = state.rstrip("*~#!%$@^-")
        p["nodes"] += n
        if st == "idle":
            p["idle"] += n
        elif st == "mix":
            p["mix"] += n
        elif st == "alloc":
            p["alloc"] += n
        else:
            p["down"] += n
        if gpu and not p["gpu"]:
            p["gpu"], p["mem_gb"] = gpu, GPU_MEM_GB.get(gpu, 0)
    return parts


def sinfo_partitions() -> Dict[str, Dict]:
    txt = _run(["sinfo", "-h", "-o", "%P|%G|%a|%t|%D"])
    return parse_sinfo(txt) if txt else {}


def qos_allowed(partition: str, qos: str) -> Optional[bool]:
    """True/False from `scontrol show partition`; None when it cannot be determined."""
    txt = _run(["scontrol", "show", "partition", partition, "-o"])
    if not txt:
        return None
    allow = re.search(r"AllowQos=(\S+)", txt)
    deny = re.search(r"DenyQos=(\S+)", txt)
    if deny and qos in deny.group(1).split(","):
        return False
    if allow:
        v = allow.group(1)
        return True if v == "ALL" else qos in v.split(",")
    return None


def pick_partitions(min_gb: int, qos: str = "normal", parts: Optional[Dict[str, Dict]] = None,
                    use_scontrol: bool = True, bf16: bool = False) -> Tuple[List[str], List[str]]:
    """Return (chosen partitions, reasons).  Honors LG_PARTITIONS_OVERRIDE.  `bf16=True` skips GPUs in NO_BF16_GPUS."""
    override = os.environ.get("LG_PARTITIONS_OVERRIDE", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()], [f"LG_PARTITIONS_OVERRIDE={override}"]
    parts = parts if parts is not None else sinfo_partitions()
    reasons: List[str] = []
    if not parts:
        chosen = [p for p in KNOWN_NORMAL if _known_mem(p) >= min_gb and not (bf16 and p in NO_BF16_GPUS)]
        return chosen, [f"sinfo unavailable -> cluster notes list for QOS {qos}: {chosen}"]
    chosen: List[str] = []
    for name, p in parts.items():
        if name.startswith(EXCLUDE_PREFIX):
            continue
        if not p["up"] or p["mem_gb"] < min_gb:
            continue
        if bf16 and (p["gpu"] or "") in NO_BF16_GPUS:
            reasons.append(f"{name}: {p['gpu']} has no bf16 tensor cores (LG_ALLOW_NO_BF16=1 to allow)")
            continue
        if p["idle"] + p["mix"] + p["alloc"] == 0:
            reasons.append(f"{name}: no usable node (all drain/down)")
            continue
        ok = qos_allowed(name, qos) if use_scontrol else None
        if ok is None:
            ok = name in KNOWN_NORMAL
            src = "cluster notes"
        else:
            src = "scontrol AllowQos"
        if ok:
            chosen.append(name)
            reasons.append(f"{name}: {p['gpu']} {p['mem_gb']}GB, idle={p['idle']} mix={p['mix']} alloc={p['alloc']} ({src})")
        else:
            reasons.append(f"{name}: QOS {qos} not allowed ({src})")
    # bigger memory and more idle nodes first (Slurm ignores order, humans read it)
    chosen.sort(key=lambda n: (-parts[n]["idle"], -parts[n]["mem_gb"], n))
    if not chosen:
        chosen = [p for p in KNOWN_NORMAL if _known_mem(p) >= min_gb and not (bf16 and p in NO_BF16_GPUS)]
        reasons.append(f"nothing matched -> fallback {chosen}")
    return chosen, reasons


def _known_mem(partition: str) -> int:
    for gpu, gb in GPU_MEM_GB.items():
        if partition.replace("-40GB-PCIe", "-PCIe-40GB") == gpu or partition == gpu:
            return gb
    if "A100-40GB" in partition:
        return 40
    return 0


def tiers(qos: str = "normal") -> Dict[int, Tuple[List[str], List[str]]]:
    parts = sinfo_partitions()
    return {gb: pick_partitions(gb, qos, parts) for gb in (8, 24, 40, 48)}
