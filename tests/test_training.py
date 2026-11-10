from __future__ import annotations

import pytest
import torch

from dlexer.data import make_xor
from dlexer.training import run_smoke, set_seed


def test_xor_shapes_and_labels() -> None:
    features, targets = make_xor(n_per_cluster=64, seed=0)
    assert features.shape == (256, 2)
    assert features.dtype == torch.float32
    assert targets.shape == (256,)
    assert set(targets.tolist()) == {0, 1}


def test_seed_is_reproducible() -> None:
    set_seed(0)
    first = torch.randn(4)
    set_seed(0)
    second = torch.randn(4)
    assert torch.equal(first, second)


def test_smoke_learns_xor() -> None:
    result = run_smoke(device="cpu", seed=0, steps=400)
    assert result.final_loss < result.initial_loss * 0.5
    assert result.accuracy >= 0.85


def test_smoke_is_deterministic_on_cpu() -> None:
    first = run_smoke(device="cpu", seed=1, steps=100)
    second = run_smoke(device="cpu", seed=1, steps=100)
    assert first.final_loss == pytest.approx(second.final_loss, rel=1e-6)
