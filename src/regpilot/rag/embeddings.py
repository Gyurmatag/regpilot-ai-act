"""Chroma-compatible embedding function backed by our LLM client."""

from __future__ import annotations

from chromadb import Documents, EmbeddingFunction, Embeddings

from regpilot.llm import LLMClient, get_llm


class RegPilotEmbeddings(EmbeddingFunction):
    """Adapter so Chroma can call either OllamaClient or StubClient."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_llm()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chroma signature
        return self.client.embed(list(input))

    @staticmethod
    def name() -> str:
        return "regpilot-embeddings"
