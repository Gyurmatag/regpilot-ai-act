"""Chroma-backed vector store for the EU AI Act corpus.

The corpus is small (~1k chunks) so a single persistent client + one collection
is enough; no need for a separate Chroma server.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

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
        # chromadb's overload-only TypedDict signature doesn't accept plain
        # dicts cleanly, but at runtime any str/int/float/bool mapping works.
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)  # type: ignore[arg-type]
        return len(chunks)

    def reset(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):  # first-run case: collection doesn't exist yet
            self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- reads -------------------------------------------------------- #

    def query(self, text: str, *, k: int | None = None) -> list[RetrievedChunk]:
        k = k or settings.top_k_dense
        res = self.collection.query(query_texts=[text], n_results=k)
        return _materialize(dict(res))

    def all_documents(self) -> list[RetrievedChunk]:
        res = self.collection.get()
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        out: list[RetrievedChunk] = []
        for i, _id in enumerate(ids):
            m = metas[i] if i < len(metas) else {}
            out.append(
                RetrievedChunk(
                    id=str(_id),
                    text=str(docs[i]) if i < len(docs) else "",
                    article=_opt_str(m.get("article")),
                    paragraph=_opt_str(m.get("paragraph")),
                    title=_opt_str(m.get("title")),
                    source=str(m.get("source") or "EU AI Act"),
                    score=0.0,
                )
            )
        return out

    def count(self) -> int:
        return self.collection.count()

    # ----- helpers ------------------------------------------------------ #

    @staticmethod
    def _flatten_meta(c: Chunk) -> dict[str, str | int | float | bool]:
        # Chroma metadata only allows str/int/float/bool — flatten everything else.
        meta: dict[str, str | int | float | bool] = {
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
        m = metas[i] or {}
        out.append(
            RetrievedChunk(
                id=str(_id),
                text=str(docs[i]),
                article=_opt_str(m.get("article")),
                paragraph=_opt_str(m.get("paragraph")),
                title=_opt_str(m.get("title")),
                source=str(m.get("source") or "EU AI Act"),
                # Convert cosine distance → similarity ∈ [0, 1].
                score=max(0.0, 1.0 - float(dists[i])),
            )
        )
    return out


def _opt_str(v: object) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s if s else None
