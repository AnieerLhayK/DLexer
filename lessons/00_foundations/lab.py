"""Lesson 00: tensors, autograd and a first training loop.

Run from the repository root:

    python lessons\\00_foundations\\lab.py
"""

from __future__ import annotations

import torch


def tensor_basics() -> None:
    print("[1] tensor basics")
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("  shape", tuple(values.shape), "dtype", values.dtype, "device", values.device)
    print("  values")
    print(values)
    print("  row mean", values.mean(dim=1))
    print("  broadcasting", (values + torch.tensor([1.0, 0.0, -1.0, 0.5])).shape)


def gradient_check() -> None:
    print("[2] autograd vs analytic gradient")
    torch.manual_seed(0)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([[0.5], [-0.5]], requires_grad=True)
    bias = torch.tensor([0.1], requires_grad=True)

    predictions = features @ weights + bias
    loss = (predictions**2).mean()
    loss.backward()

    count = features.shape[0]
    analytic_weights = (2.0 / count) * features.T @ predictions
    analytic_bias = (2.0 / count) * predictions.sum(dim=0)

    weight_error = (weights.grad - analytic_weights).abs().max().item()
    bias_error = (bias.grad - analytic_bias).abs().max().item()
    print(f"  loss {loss.item():.6f}")
    print(f"  max |grad - analytic| weights={weight_error:.3e} bias={bias_error:.3e}")


def train_linear_regression() -> None:
    print("[3] manual linear regression")
    torch.manual_seed(0)
    true_weights = torch.tensor([[2.0], [-1.0]])
    true_bias = torch.tensor([0.5])
    features = torch.randn(256, 2)
    targets = features @ true_weights + true_bias + 0.1 * torch.randn(256, 1)

    weights = torch.zeros(2, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([weights, bias], lr=0.1)

    for step in range(1, 101):
        predictions = features @ weights + bias
        loss = ((predictions - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in (1, 25, 50, 100):
            print(f"  step {step:3d} loss {loss.item():.6f}")

    learned = weights.detach().flatten().tolist()
    print(f"  learned w={learned} b={bias.detach().item():.4f}")


def device_round_trip() -> None:
    print("[4] CPU/GPU round trip")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    left = torch.randn(256, 256, device=device)
    right = torch.randn(256, 256, device=device)
    product = left @ right
    if device == "cuda":
        torch.cuda.synchronize()
        print("  device", device, "|", torch.cuda.get_device_name(0))
    else:
        print("  device", device, "| CUDA not available")
    print(f"  result shape {tuple(product.shape)} mean {product.mean().item():.4f}")


def main() -> None:
    print("DLexer lesson 00: foundations")
    print(
        "torch",
        torch.__version__,
        "| cuda build",
        torch.version.cuda,
        "| cuda available",
        torch.cuda.is_available(),
    )
    print()
    tensor_basics()
    print()
    gradient_check()
    print()
    train_linear_regression()
    print()
    device_round_trip()
    print()
    print("done: edit this file, break it, and fix it. That is the lesson.")


if __name__ == "__main__":
    main()
