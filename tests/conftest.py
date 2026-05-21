"""Pytest fixtures.

All tests run with the deterministic stub LLM and a session-scoped vector store
populated only from the structured Annex III + Article 5 records (no PDF
download required — CI works offline).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("REGPILOT_LLM", "stub")


@pytest.fixture(scope="session")
def tmp_chroma_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="regpilot-chroma-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_store(tmp_chroma_dir: Path):
    """Build a clean, offline vector store once per test session."""

    os.environ["REGPILOT_CHROMA_DIR"] = str(tmp_chroma_dir)

    # Re-import so pydantic-settings picks up the patched env.
    from regpilot.config import settings as _settings

    _settings.chroma_dir = tmp_chroma_dir

    from regpilot.ingestion.annex import annex_iii_chunks, article_5_chunks
    from regpilot.rag.vectorstore import VectorStore
    from regpilot.tools.citation_validator import reset_cache

    store = VectorStore(persist_dir=tmp_chroma_dir)
    store.reset()
    store.upsert(annex_iii_chunks())
    store.upsert(article_5_chunks())
    reset_cache()
    yield
