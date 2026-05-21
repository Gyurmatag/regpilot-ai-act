"""Chroma-backed vector store for the EU AI Act corpus.

The corpus is small (~1k chunks) so a single persistent client + one collection
is enough; no need for a separate Chroma server.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings

from regpilot.config import settings
from regpilot.ingestion.chunker import Chunk
from regpilot.rag.embeddings import RegPilotEmbeddings
from regpilot.state import RetrievedChunk

logger = logging.getLogger(__name__)


COLLECTION_NAME = "eu_ai_act"


class VectorStore:
    def __init__(self, persist_dir: Path | None = None) -> None:
        self.persist_dir = Path(persist_dir or settings.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.embedding_function = RegPilotEmbeddings()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- writes ------------------------------------------------------- #

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        ids = [c.id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [self._flatten_meta(c) for c in chunks]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(chunks)

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:  # pragma: no cover - first-run case
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- reads -------------------------------------------------------- #

    def query(self, text: str, *, k: int | None = None) -> list[RetrievedChunk]:
        k = k or settings.top_k_dense
        res = self.collection.query(query_texts=[text], n_results=k)
        return _materialize(res)

    def all_documents(self) -> list[RetrievedChunk]:
        res = self.collection.get()
        return [
            RetrievedChunk(
                id=res["ids"][i],
                text=res["documents"][i],
                article=res["metadatas"][i].get("article"),
                paragraph=res["metadatas"][i].get("paragraph"),
                title=res["metadatas"][i].get("title"),
                source=res["metadatas"][i].get("source", "EU AI Act"),
                score=0.0,
            )
            for i in range(len(res["ids"]))
        ]

    def count(self) -> int:
        return self.collection.count()

    # ----- helpers ------------------------------------------------------ #

    @staticmethod
    def _flatten_meta(c: Chunk) -> dict:
        # Chroma metadata only allows str/int/float/bool — flatten everything else.
        meta = {
            "article": c.article or "",
            "paragraph": c.paragraph or "",
            "title": c.title or "",
            "source": c.source,
        }
        for k, v in asdict(c).get("meta", {}).items():
            if isinstance(v, (str, int, float, bool)):
                meta[f"x_{k}"] = v
            else:
                meta[f"x_{k}"] = str(v)
        return meta


def _materialize(res: dict) -> list[RetrievedChunk]:
    if not res.get("ids") or not res["ids"][0]:
        return []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[0.0] * len(ids)])[0]
    out: list[RetrievedChunk] = []
    for i, _id in enumerate(ids):
        out.append(
            RetrievedChunk(
                id=_id,
                text=docs[i],
                article=metas[i].get("article") or None,
                paragraph=metas[i].get("paragraph") or None,
                title=metas[i].get("title") or None,
                source=metas[i].get("source", "EU AI Act"),
                # Convert cosine distance → similarity ∈ [0, 1].
                score=max(0.0, 1.0 - float(dists[i])),
            )
        )
    return out
