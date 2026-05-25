"""Thin shim — delegates to :func:`regpilot.cli.ingest`.

The real implementation lives under ``src/regpilot/cli.py`` so it can be
unit-tested and re-used as a library / installable console script. This
file exists only to preserve the familiar invocation
``python scripts/ingest.py …`` and the Makefile target.
"""

from __future__ import annotations

import sys

from regpilot.cli import ingest

if __name__ == "__main__":
    sys.exit(ingest())
