from __future__ import annotations

import os

import pytest

from dlexer.connectivity import probe_all

pytestmark = pytest.mark.network

if os.environ.get("DLEXER_NETWORK_TESTS") != "1":
    pytest.skip(
        "set DLEXER_NETWORK_TESTS=1 to run outbound network checks",
        allow_module_level=True,
    )


def test_default_endpoints_are_reachable() -> None:
    results = probe_all(timeout=10.0)
    assert results
    failures = [result for result in results if not result.ok]
    assert not failures, failures
