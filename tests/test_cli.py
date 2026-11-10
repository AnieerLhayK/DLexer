from __future__ import annotations

import json

from dlexer.cli import main


def test_doctor_json_without_network(capsys) -> None:
    code = main(["doctor", "--json", "--no-network"])
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    report = json.loads(captured.out)
    assert report["schema_version"] == 2
    assert report["ok"] is True
    assert {"python", "packages", "cuda", "gpu_smoke", "network"} <= set(report)
    assert "toolchain" not in report
    assert not any("MSVC" in warning or "ninja" in warning for warning in report["warnings"])
    assert report["network"] == []


def test_smoke_json(capsys) -> None:
    code = main(["smoke", "--device", "cpu", "--seed", "0", "--steps", "300", "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["device"] == "cpu"
