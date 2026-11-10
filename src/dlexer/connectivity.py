"""Outbound network reachability probes for the DLexer doctor."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

DEFAULT_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("GitHub", "https://github.com"),
    ("PyPI", "https://pypi.org/simple/"),
    ("PyTorch CUDA index", "https://download.pytorch.org/whl/cu130/"),
    ("Hugging Face", "https://huggingface.co"),
)

_USER_AGENT = "dlexer/0.1 (+https://github.com/AnieerLhayK/DLexer)"


@dataclass(frozen=True)
class ProbeResult:
    name: str
    url: str
    ok: bool
    status: int | None
    latency_ms: float | None
    detail: str | None = None


def _probe(name: str, url: str, timeout: float) -> ProbeResult:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
        latency = (time.perf_counter() - start) * 1000.0
        return ProbeResult(name, url, True, status, round(latency, 1))
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - start) * 1000.0
        # Any HTTP response proves the endpoint is reachable, even if HEAD is rejected.
        return ProbeResult(
            name,
            url,
            True,
            int(exc.code),
            round(latency, 1),
            f"HTTP {exc.code} (HEAD may be restricted)",
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000.0
        return ProbeResult(
            name,
            url,
            False,
            None,
            round(latency, 1),
            f"{type(exc).__name__}: {exc}",
        )


def probe_all(
    endpoints: Iterable[tuple[str, str]] = DEFAULT_ENDPOINTS,
    timeout: float = 5.0,
) -> list[ProbeResult]:
    """Probe endpoints concurrently, preserving the input order."""
    items = list(endpoints)
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(items)))) as pool:
        futures = [pool.submit(_probe, name, url, timeout) for name, url in items]
        return [future.result() for future in futures]
