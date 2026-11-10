from __future__ import annotations

import pytest
import torch

from dlexer.gpu import run_gpu_checks

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available"),
]


def test_device_properties() -> None:
    assert torch.cuda.device_count() >= 1
    props = torch.cuda.get_device_properties(0)
    assert props.major >= 8
    assert props.total_memory > 0


def test_matmul_matches_cpu() -> None:
    result = run_gpu_checks()
    assert result.available is True
    assert result.error is None, result.error
    assert result.matmul_ms is not None
    assert result.max_abs_diff is not None
    assert result.max_abs_diff < 1e-3


def test_memory_is_reported() -> None:
    free, total = torch.cuda.mem_get_info()
    assert total > 0
    assert free > 0
