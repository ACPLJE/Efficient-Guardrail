"""GPU tests: skipped when CUDA is unavailable (CLAUDE.md 3.4). tests/conftest.py is untouched."""
import pytest


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _cuda():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "/tests/gpu/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip)
