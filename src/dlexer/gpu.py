"""GPU runtime checks used by the doctor command and the test suite."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

_SIZE = 512


@dataclass(frozen=True)
class GpuCheck:
    available: bool
    device_index: int | None = None
    device_name: str | None = None
    matmul_ms: float | None = None
    max_abs_diff: float | None = None
    bf16_supported: bool | None = None
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    error: str | None = None


def _memory(index: int) -> tuple[int | None, int | None]:
    try:
        free, total = torch.cuda.mem_get_info(index)
        return int(free // (1024 * 1024)), int(total // (1024 * 1024))
    except Exception:
        return None, None


def run_gpu_checks(device_index: int = 0, size: int = _SIZE) -> GpuCheck:
    """Run a small CUDA matmul, compare it with CPU, and report device facts."""
    if not torch.cuda.is_available():
        return GpuCheck(available=False)

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(0)
        left = torch.randn(size, size)
        right = torch.randn(size, size)
        expected = left @ right

        device = torch.device("cuda", device_index)
        gpu_left = left.to(device)
        gpu_right = right.to(device)
        torch.cuda.synchronize(device)
        gpu_left @ gpu_right  # warm-up
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        actual = gpu_left @ gpu_right
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        max_abs_diff = float((actual.cpu() - expected).abs().max().item())
        free_mb, total_mb = _memory(device_index)
        return GpuCheck(
            available=True,
            device_index=device_index,
            device_name=torch.cuda.get_device_name(device_index),
            matmul_ms=round(elapsed_ms, 3),
            max_abs_diff=max_abs_diff,
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
            memory_total_mb=total_mb,
            memory_free_mb=free_mb,
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        return GpuCheck(
            available=True,
            device_index=device_index,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
