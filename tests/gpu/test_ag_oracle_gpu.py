"""GPU smoke for the reference ledger: the tiny model on CUDA matches CPU within the E2.1 bound (1e-3)."""
import numpy as np
import pytest


def test_tiny_ledger_cuda_matches_cpu():
    torch = pytest.importorskip("torch")
    from lgs1.lgcore.model import tiny_qwen3
    from ag.reference.ledger_oracle import decode_with_ledger
    from ag.targets.base import RecordSpec

    ids = list(range(5, 5 + 400))
    cpu = decode_with_ledger(tiny_qwen3(seed=0, attn="eager"), ids, 8, spec=RecordSpec())
    gpu_model = tiny_qwen3(seed=0, attn="eager").to("cuda")

    class OnCuda:
        config = gpu_model.config

        def __call__(self, input_ids, **kw):
            return gpu_model(input_ids=input_ids.to("cuda"), **kw)

    gpu = decode_with_ledger(OnCuda(), ids, 0, spec=RecordSpec(), forced_output=cpu.output_ids)
    assert float(np.abs(gpu.block_mass - cpu.block_mass).max()) <= 1e-3
