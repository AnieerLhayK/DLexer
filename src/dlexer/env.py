"""Environment inspection for the DLexer workspace."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

_PACKAGE_NAMES = ("torch", "torchvision", "numpy", "transformers", "pytest")


@dataclass(frozen=True)
class PythonInfo:
    version: str
    executable: str
    platform: str


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    capability: str
    total_memory_mb: int


@dataclass(frozen=True)
class CudaInfo:
    available: bool
    torch_build_cuda: str | None
    device_count: int
    devices: list[DeviceInfo]
    cudnn_version: int | None
    driver_version: str | None
    bf16_supported: bool | None
    error: str | None = None


@dataclass(frozen=True)
class EnvironmentReport:
    python: PythonInfo
    packages: dict[str, str | None]
    cuda: CudaInfo


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _run_capture(command: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return result.returncode, (result.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return -1, ""


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        candidate = Path(system_root) / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            exe = str(candidate)
    return exe


@lru_cache(maxsize=1)
def get_driver_version() -> str | None:
    """Return the first NVIDIA driver version reported by nvidia-smi."""
    exe = _nvidia_smi()
    if exe is None:
        return None
    code, output = _run_capture(
        [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=10.0,
    )
    if code != 0 or not output:
        return None
    return output.splitlines()[0].strip() or None


def _collect_cuda(torch_module: Any) -> CudaInfo:
    build = getattr(torch_module.version, "cuda", None)
    try:
        available = bool(torch_module.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive
        return CudaInfo(
            available=False,
            torch_build_cuda=build,
            device_count=0,
            devices=[],
            cudnn_version=None,
            driver_version=get_driver_version(),
            bf16_supported=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    devices: list[DeviceInfo] = []
    error: str | None = None
    if available:
        try:
            for index in range(torch_module.cuda.device_count()):
                props = torch_module.cuda.get_device_properties(index)
                devices.append(
                    DeviceInfo(
                        index=index,
                        name=props.name,
                        capability=f"{props.major}.{props.minor}",
                        total_memory_mb=int(props.total_memory // (1024 * 1024)),
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"

    cudnn_version: int | None = None
    bf16_supported: bool | None = None
    if available and error is None:
        try:
            cudnn_version = torch_module.backends.cudnn.version()
            bf16_supported = bool(torch_module.cuda.is_bf16_supported())
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"

    return CudaInfo(
        available=available,
        torch_build_cuda=build,
        device_count=len(devices),
        devices=devices,
        cudnn_version=cudnn_version,
        driver_version=get_driver_version(),
        bf16_supported=bf16_supported,
        error=error,
    )


def collect_environment() -> EnvironmentReport:
    """Collect the current Python, PyTorch and CUDA state."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PyTorch is not importable: {exc}") from exc

    return EnvironmentReport(
        python=PythonInfo(
            version=platform.python_version(),
            executable=sys.executable,
            platform=platform.platform(),
        ),
        packages={name: _package_version(name) for name in _PACKAGE_NAMES},
        cuda=_collect_cuda(torch),
    )


def report_to_dict(report: EnvironmentReport) -> dict[str, Any]:
    """Return a JSON-serializable representation of an environment report."""
    return asdict(report)
