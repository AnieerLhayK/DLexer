"""Small synthetic datasets used by smoke tests and lessons."""

from __future__ import annotations

import torch


def make_xor(
    n_per_cluster: int = 128,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a 2D XOR classification set with four Gaussian clusters."""
    if n_per_cluster < 1:
        raise ValueError("n_per_cluster must be positive")
    generator = torch.Generator().manual_seed(seed)
    centers = ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
    labels = (0, 0, 1, 1)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for (x, y), label in zip(centers, labels, strict=True):
        center = torch.tensor((x, y), dtype=torch.float32)
        cluster = torch.randn(n_per_cluster, 2, generator=generator) * noise + center
        features.append(cluster)
        targets.append(torch.full((n_per_cluster,), label, dtype=torch.long))
    features_tensor = torch.cat(features, dim=0)
    targets_tensor = torch.cat(targets, dim=0)
    order = torch.randperm(features_tensor.shape[0], generator=generator)
    return features_tensor[order], targets_tensor[order]
