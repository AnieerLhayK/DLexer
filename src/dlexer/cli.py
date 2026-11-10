"""Command line interface for DLexer."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from dlexer import __version__
from dlexer.connectivity import DEFAULT_ENDPOINTS, ProbeResult, probe_all
from dlexer.env import EnvironmentReport, collect_environment
from dlexer.gpu import GpuCheck, run_gpu_checks
from dlexer.training import run_smoke

_SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlexer",
        description="Personal PyTorch learning workspace checks.",
    )
    parser.add_argument("--version", action="version", version=f"dlexer {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="inspect Python, CUDA, toolchain and network connectivity",
    )
    doctor.add_argument("--json", action="store_true", help="print a machine-readable report")
    doctor.add_argument("--no-network", action="store_true", help="skip outbound network probes")
    doctor.add_argument("--require-gpu", action="store_true", help="fail when CUDA is unavailable")
    doctor.add_argument("--require-network", action="store_true", help="fail when a probe fails")
    doctor.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="per-endpoint timeout in seconds (default: 5)",
    )

    smoke = subparsers.add_parser("smoke", help="run a tiny end-to-end training check")
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--steps", type=int, default=400)
    smoke.add_argument("--json", action="store_true", help="print a machine-readable report")
    return parser


def _diagnose(
    env: EnvironmentReport,
    gpu: GpuCheck,
    network: list[ProbeResult],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    if env.cuda.torch_build_cuda is None:
        warnings.append("torch was built without CUDA; GPU checks are unavailable")
    elif not env.cuda.available:
        warnings.append("CUDA build is present but no CUDA device is available")
    if env.cuda.error:
        warnings.append(f"CUDA query error: {env.cuda.error}")
    if env.cuda.available and env.cuda.cudnn_version is None:
        warnings.append("cuDNN version is unavailable")

    if gpu.available and gpu.error:
        errors.append(f"GPU smoke failed: {gpu.error}")
    if gpu.available and gpu.max_abs_diff is not None and gpu.max_abs_diff > 1e-2:
        errors.append(f"GPU matmul mismatch: max abs diff {gpu.max_abs_diff:.3e}")

    if sys.platform == "win32":
        if not env.toolchain.msvc_installed:
            warnings.append(
                "MSVC build tools not found; custom CUDA extensions cannot be compiled"
            )
        elif not env.toolchain.cl_on_path:
            warnings.append(
                "MSVC is installed but cl.exe is not on PATH; use a Developer PowerShell"
            )
    if not env.toolchain.ninja:
        warnings.append("ninja not found; CUDA extension builds may fall back to setuptools")

    for probe in network:
        if not probe.ok:
            detail = probe.detail or "unknown error"
            warnings.append(f"network probe failed: {probe.name} ({detail})")
    return warnings, errors


def _build_doctor_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    env = collect_environment()
    gpu = run_gpu_checks()
    network: list[ProbeResult] = []
    if not args.no_network:
        network = probe_all(DEFAULT_ENDPOINTS, timeout=args.timeout)

    warnings, errors = _diagnose(env, gpu, network)
    if args.require_gpu and not env.cuda.available:
        errors.append("--require-gpu: CUDA is required but not available")
    if args.require_network:
        failures = [probe.name for probe in network if not probe.ok]
        if failures:
            errors.append(f"--require-network: probes failed: {', '.join(failures)}")

    report = {
        "schema_version": _SCHEMA_VERSION,
        "ok": not errors,
        "python": asdict(env.python),
        "packages": env.packages,
        "cuda": asdict(env.cuda),
        "toolchain": asdict(env.toolchain),
        "gpu_smoke": asdict(gpu),
        "network": [asdict(probe) for probe in network],
        "warnings": warnings,
        "errors": errors,
    }
    return report, 0 if not errors else 1


def _format_cudnn(version: int | None) -> str:
    if version is None:
        return "unknown"
    return f"{version // 10000}.{(version // 100) % 100}.{version % 100}"


def _print_doctor(report: dict[str, Any]) -> None:
    python = report["python"]
    packages = report["packages"]
    cuda = report["cuda"]
    gpu = report["gpu_smoke"]
    toolchain = report["toolchain"]

    print(f"DLexer doctor {__version__}")
    print()
    print("[system]")
    print(f"  python       {python['version']} ({python['executable']})")
    print(f"  platform     {python['platform']}")
    package_text = ", ".join(f"{name}={version or 'missing'}" for name, version in packages.items())
    print(f"  packages     {package_text}")
    print()
    print("[gpu]")
    print(f"  driver       {cuda['driver_version'] or 'unknown'}")
    if cuda["available"]:
        cudnn = _format_cudnn(cuda["cudnn_version"])
        print(f"  cuda         available (torch CUDA {cuda['torch_build_cuda']}, cuDNN {cudnn})")
        for device in cuda["devices"]:
            capability = device["capability"].replace(".", "")
            print(
                f"  device {device['index']}     {device['name']} "
                f"({device['total_memory_mb']} MB, sm_{capability})"
            )
    else:
        print("  cuda         unavailable")
    if gpu["error"]:
        print(f"  matmul       failed: {gpu['error']}")
    elif gpu["matmul_ms"] is not None:
        print(f"  matmul       {gpu['matmul_ms']:.3f} ms, max|diff|={gpu['max_abs_diff']:.3e}")
        print(f"  memory       {gpu['memory_free_mb']} / {gpu['memory_total_mb']} MB free")
    else:
        print("  matmul       skipped (no CUDA device)")
    print()
    print("[toolchain]")
    nvcc = toolchain["nvcc"] or "missing"
    nvcc_version = toolchain["nvcc_version"] or "unknown"
    print(f"  nvcc         {nvcc} ({nvcc_version})")
    print(f"  cuda_home    {toolchain['cuda_home'] or 'missing'}")
    print(f"  msvc         {'installed' if toolchain['msvc_installed'] else 'missing'}")
    print(f"  ninja        {'found' if toolchain['ninja'] else 'missing'}")
    print()
    print("[network]")
    network = report["network"]
    if not network:
        print("  skipped (--no-network)")
    else:
        for probe in network:
            state = "OK" if probe["ok"] else "FAIL"
            latency = f"{probe['latency_ms']:.0f} ms" if probe["latency_ms"] is not None else "n/a"
            detail = f" ({probe['detail']})" if probe["detail"] else ""
            print(f"  {probe['name']:<20} {state} {latency}{detail}")

    warnings = report["warnings"]
    errors = report["errors"]
    print()
    status = "OK" if not errors else "FAIL"
    print(f"[verdict] {status} ({len(warnings)} warning(s), {len(errors)} error(s))")
    for warning in warnings:
        print(f"  ! {warning}")
    for error in errors:
        print(f"  x {error}")


def _doctor(args: argparse.Namespace) -> int:
    if args.no_network and args.require_network:
        print("error: --require-network cannot be combined with --no-network", file=sys.stderr)
        return 2
    try:
        report, code = _build_doctor_report(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_doctor(report)
    return code


def _smoke(args: argparse.Namespace) -> int:
    try:
        result = run_smoke(device=args.device, seed=args.seed, steps=args.steps)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ok = result.final_loss < result.initial_loss
    if args.json:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "ok": ok,
            **asdict(result),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"DLexer smoke {__version__}")
        print(f"  device    {result.device}")
        print(f"  seed      {result.seed}")
        print(f"  steps     {result.steps}")
        print(f"  loss      {result.initial_loss:.4f} -> {result.final_loss:.4f}")
        print(f"  accuracy  {result.accuracy:.2%}")
        print(f"  duration  {result.duration_s:.2f} s")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "smoke":
        return _smoke(args)
    parser.print_help()
    return 2
