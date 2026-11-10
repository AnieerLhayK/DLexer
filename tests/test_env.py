from __future__ import annotations

import importlib.metadata
import sys

import torch

from dlexer.env import collect_environment


def test_python_is_supported() -> None:
    assert sys.version_info >= (3, 11)


def test_environment_report_has_expected_shape() -> None:
    report = collect_environment()
    assert report.python.version == sys.version.split()[0]
    assert report.packages["torch"] == importlib.metadata.version("torch")
    assert report.cuda.torch_build_cuda == torch.version.cuda
    assert isinstance(report.toolchain.ninja, bool)
    assert isinstance(report.toolchain.msvc_installed, bool)
