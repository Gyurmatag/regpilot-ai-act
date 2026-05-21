"""Tests for the RAG subgraph + hybrid retriever."""

from __future__ import annotations

import pytest

from regpilot.rag.retriever import HybridRetriever
from regpilot.rag.subgraph import build_rag_subgraph
from regpilot.rag.vectorstore import VectorStore


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return HybridRetriever(store=VectorStore())


def test_dense_retrieval_returns_chunks(retriever: HybridRetriever) -> None:
    # With the stub embeddings dense scoring is pseudo-random; we only check
    # that the wiring is alive and returns the requested k.
    hits = retriever.dense("recruitment and cv screening", k=5)
    assert hits, "dense retrieval returned nothing"
    assert len(hits) == 5


def test_sparse_retrieval_works(retriever: HybridRetriever) -> None:
    hits = retriever.sparse("social scoring", k=5)
    assert hits
    assert any(
        "5(1)(c)" in (h.get("paragraph") or "") or "social" in h["text"].lower()
        for h in hits
    )


def test_hybrid_retrieval_combines_both(retriever: HybridRetriever) -> None:
    hits = retriever.hybrid("predictive policing")
    assert hits
    assert any("5(1)(d)" in (h.get("paragraph") or "") for h in hits[:5])


def test_rag_subgraph_runs_end_to_end() -> None:
    sg = build_rag_subgraph()
    out = sg.invoke({"query": "exam proctoring system used in universities"})
    compressed = out.get("compressed") or []
    assert compressed, "subgraph returned no compressed chunks"
    assert all("score" in c for c in compressed)
    assert len(compressed) <= 5
