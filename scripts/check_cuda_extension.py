"""Check whether a tiny custom CUDA extension can be built.

This optional second-level CUDA check is not part of the default test suite.
It exits with code 2 when the toolchain is incomplete so that a missing MSVC
installation is reported as "skipped", not as a code failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")


def _msvc_installation() -> str | None:
    if not _VSWHERE.is_file():
        return None
    command = [
        str(_VSWHERE),
        "-latest",
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or "").strip()
    return output.splitlines()[0].strip() if output else None


def _preflight() -> list[str]:
    try:
        import torch
    except Exception as exc:
        return [f"torch is not importable: {exc}"]

    problems: list[str] = []
    if torch.version.cuda is None:
        problems.append("torch is a CPU build; install the cu130 wheel first")
    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is False")
    if shutil.which("nvcc") is None:
        problems.append("nvcc is not on PATH")
    if shutil.which("cl") is None:
        msvc = _msvc_installation()
        if msvc:
            problems.append(
                f"MSVC found at {msvc}, but cl.exe is not on PATH "
                "(run from a Developer PowerShell)"
            )
        else:
            problems.append("MSVC Build Tools with the C++ workload are not installed")
    if shutil.which("ninja") is None:
        print("warning: ninja is not installed; the build may fall back to setuptools")
    return problems


def main() -> int:
    problems = _preflight()
    if problems:
        print("custom CUDA extension is not ready:")
        for problem in problems:
            print(f"  - {problem}")
        print("see docs/cuda.md; exit code 2 means skipped")
        return 2

    import torch
    from torch.utils.cpp_extension import load

    sources = [str(ROOT / "scripts" / "csrc" / name) for name in ("vecadd.cpp", "vecadd.cu")]
    missing = [path for path in sources if not Path(path).is_file()]
    if missing:
        print("missing extension sources:", missing)
        return 1

    extension = load(
        name="dlexer_vecadd",
        sources=sources,
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
    left = torch.ones(4096, device="cuda")
    right = torch.full((4096,), 2.0, device="cuda")
    output = extension.vecadd(left, right)
    expected = torch.full((4096,), 3.0, device="cuda")
    if not torch.allclose(output, expected):
        print("vecadd result mismatch")
        return 1
    print("custom CUDA extension built and ran successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
