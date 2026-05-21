"""Hybrid retrieval: dense (Chroma) + sparse (BM25) fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from regpilot.config import settings
from regpilot.rag.vectorstore import VectorStore
from regpilot.state import RetrievedChunk

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class HybridRetriever:
    """Combines a Chroma dense retriever with an in-memory BM25 index.

    BM25 corpus is built lazily from the entire Chroma collection so the two
    indices stay in sync. For a corpus of ~1k chunks this fits comfortably in
    memory and the build takes <100ms.
    """

    store: VectorStore
    _bm25: BM25Okapi | None = None
    _corpus: list[RetrievedChunk] | None = None
    _tokenized: list[list[str]] | None = None

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        self._corpus = self.store.all_documents()
        self._tokenized = [_tokenize(c["text"]) for c in self._corpus]
        if not self._tokenized:
            raise RuntimeError(
                "Vector store is empty — run `python scripts/ingest.py` before retrieving."
            )
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info("Built BM25 index over %d docs", len(self._corpus))

    # ----- single-index retrievers ------------------------------------- #

    def dense(self, query: str, *, k: int | None = None) -> list[RetrievedChunk]:
        return self.store.query(query, k=k or settings.top_k_dense)

    def sparse(self, query: str, *, k: int | None = None) -> list[RetrievedChunk]:
        self._ensure_bm25()
        assert self._bm25 is not None and self._corpus is not None
        k = k or settings.top_k_sparse
        scores = self._bm25.get_scores(_tokenize(query))
        if not len(scores):
            return []
        max_s = float(scores.max()) or 1.0
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[RetrievedChunk] = []
        for idx, s in ranked:
            chunk = dict(self._corpus[idx])
            chunk["score"] = float(s) / max_s
            out.append(chunk)  # type: ignore[arg-type]
        return out

    # ----- fusion ------------------------------------------------------- #

    def hybrid(
        self,
        query: str,
        *,
        k_dense: int | None = None,
        k_sparse: int | None = None,
        k_out: int | None = None,
    ) -> list[RetrievedChunk]:
        """RRF over dense + sparse rankings."""

        dense_hits = self.dense(query, k=k_dense)
        sparse_hits = self.sparse(query, k=k_sparse)
        fused = _rrf([dense_hits, sparse_hits], k_const=settings.rrf_k)
        return fused[: (k_out or settings.top_k_dense)]


def _rrf(
    ranking_lists: list[list[RetrievedChunk]],
    *,
    k_const: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion. Returns chunks sorted by fused score descending."""

    scores: dict[str, float] = {}
    docs: dict[str, RetrievedChunk] = {}
    for hits in ranking_lists:
        for rank, hit in enumerate(hits):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k_const + rank + 1)
            docs[hit["id"]] = hit
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[RetrievedChunk] = []
    for doc_id, fused_score in ordered:
        d = dict(docs[doc_id])
        d["score"] = fused_score
        out.append(d)  # type: ignore[arg-type]
    return out
