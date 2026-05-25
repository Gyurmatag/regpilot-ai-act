"""Thin shim — delegates to :func:`regpilot.cli.loadtest`.

The real implementation lives under ``src/regpilot/loadtest.py`` (harness +
report writer) and ``src/regpilot/cli.py`` (argparse) so it can be unit-tested
and re-used as a library. This file exists only to preserve the familiar
invocation ``python scripts/loadtest.py …`` and the Makefile target.
"""

from __future__ import annotations

import sys

from regpilot.cli import loadtest

if __name__ == "__main__":
    sys.exit(loadtest())
