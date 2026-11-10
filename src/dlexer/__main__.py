"""Allow running the CLI as ``python -m dlexer``."""

from __future__ import annotations

from dlexer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
