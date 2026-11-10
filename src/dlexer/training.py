"""Minimal training utilities and the end-to-end smoke check."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dlexer.data import make_xor

_DEFAULT_STEPS = 400
_DEFAULT_LR = 0.05


@dataclass(frozen=True)
class SmokeResult:
    device: str
    seed: int
    steps: int
    initial_loss: float
    final_loss: float
    accuracy: float
    duration_s: float


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (including CUDA when present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> str:
    """Resolve auto/cpu/cuda into a concrete device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def build_mlp(input_dim: int = 2, hidden_dim: int = 32, num_classes: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, num_classes),
    )


def run_smoke(
    device: str = "auto",
    seed: int = 0,
    steps: int = _DEFAULT_STEPS,
    lr: float = _DEFAULT_LR,
) -> SmokeResult:
    """Train a tiny MLP on XOR and return before/after metrics."""
    if steps < 1:
        raise ValueError("steps must be positive")
    resolved = resolve_device(device)
    set_seed(seed)
    features, targets = make_xor(seed=seed)
    features = features.to(resolved)
    targets = targets.to(resolved)

    model = build_mlp().to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    start = time.perf_counter()
    model.train()
    with torch.no_grad():
        initial_loss = float(loss_fn(model(features), targets).item())

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(features)
        final_loss = float(loss_fn(logits, targets).item())
        accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
    duration = time.perf_counter() - start

    return SmokeResult(
        device=resolved,
        seed=seed,
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        accuracy=accuracy,
        duration_s=duration,
    )
