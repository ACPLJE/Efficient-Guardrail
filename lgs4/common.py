"""Shared trainer utilities: optimizer/schedule, logging, seeds, HF snapshot fingerprint, val AUPRC."""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from lgs1.lineage import code_commit, dir_fingerprint
from lgs1.metrics import auprc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_auto(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_optimizer(groups: Dict[str, List[torch.nn.Parameter]], probe_params: List[torch.nn.Parameter], hp: Dict) -> torch.optim.Optimizer:
    pg = []
    for key, params in groups.items():
        if params:
            pg.append({"params": params, "lr": hp[key], "weight_decay": hp.get("weight_decay", 0.0), "name": key})
    pg.append({"params": probe_params, "lr": hp["lr_probe"], "weight_decay": 0.0, "name": "lr_probe"})
    # fused AdamW updates in place; the default foreach path materialises sqrt(v) for EVERY parameter at once, which is
    # what pushed the 4B head (2.4B fp32 master params: 9 GiB weights + 9 grads + 18 Adam) over 47 GiB on 2026-09-19.
    fused = torch.cuda.is_available() and all(p.is_cuda for g in pg for p in g["params"])
    return torch.optim.AdamW(pg, betas=(0.9, 0.98), eps=1e-8, fused=fused)


def lr_lambda(total_steps: int, warmup_frac: float, schedule: str = "cosine"):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        if schedule == "constant":
            return 1.0
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


class JsonlLog:
    def __init__(self, path: Path, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a" if resume else "w", encoding="utf-8")

    def write(self, **rec) -> None:
        rec.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def snapshot_sha(repo: str) -> str:
    """Fingerprint of the local HF snapshot of `repo` (model_sha in the run lineage)."""
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo, local_files_only=True)
        return dir_fingerprint(d)
    except Exception:
        return "none"


def run_meta(P_root: Path, extra: Optional[Dict] = None) -> Dict:
    m = {"code_commit": code_commit(P_root), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "host": os.uname().nodename,
         "slurm_job": os.environ.get("SLURM_JOB_ID")}
    m.update(extra or {})
    return m


def window_auprc(logits: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if not logits or len(set(int(l) for l in labels)) < 2:
        return None
    return auprc([float(x) for x in logits], [int(l) for l in labels])


def gpu_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)


class Checkpointer:
    """Decides WHEN to write a resume checkpoint: every `every_sec` seconds, or as soon as SIGTERM/SIGUSR1 arrives
    (the sbatch scripts ask Slurm for `--signal=TERM@600`, ten minutes before the time limit).  After a signal the
    trainer saves and exits with rc=3, so dependants stay pending and `s4ctl.py all --sbatch` resumes it."""

    def __init__(self, every_sec: float = 1800.0):
        import signal as _signal
        self.every_sec = float(every_sec)
        self.t_last = time.time()
        self.stop = False
        self.signal_name: Optional[str] = None
        for name in ("SIGTERM", "SIGUSR1"):
            sig = getattr(_signal, name, None)
            if sig is None:
                continue
            try:
                _signal.signal(sig, self._on_signal)
            except (ValueError, OSError):                          # not the main thread (tests)
                pass

    def _on_signal(self, signum, frame):
        self.stop = True
        self.signal_name = str(signum)

    def due(self) -> bool:
        return self.stop or (time.time() - self.t_last) >= self.every_sec

    def mark(self) -> None:
        self.t_last = time.time()

