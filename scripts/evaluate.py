"""Thin shim — delegates to :mod:`regpilot.evaluation.cli`.

The real implementation lives under ``src/regpilot/evaluation/`` so it
can be unit-tested and re-used as a library. This file exists only to
preserve the familiar invocation ``python scripts/evaluate.py …`` and
the Makefile target.
"""

from __future__ import annotations

import sys

from regpilot.evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
