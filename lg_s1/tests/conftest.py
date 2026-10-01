import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(scope="session")
def torch_cpu():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch


@pytest.fixture(scope="session")
def tiny(torch_cpu):
    """4-layer / hidden-64 Qwen3 with eager attention (deterministic on CPU)."""
    from lgs1.lgcore import adapter as A
    return A.tiny_qwen3(seed=0)


@pytest.fixture(scope="session")
def adapter_report(capsys):
    from lgs1.lgcore import adapter as A
    return A.report()


def pytest_sessionstart(session):
    from lgs1.lgcore import adapter as A
    print("\n" + A.report())
