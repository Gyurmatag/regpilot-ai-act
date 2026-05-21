"""Chroma-compatible embedding function backed by our LLM client."""

from __future__ import annotations

from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings

from regpilot.config import settings
from regpilot.llm import LLMClient, get_llm


class RegPilotEmbeddings(EmbeddingFunction):
    """Adapter so Chroma can call either OllamaClient or StubClient.

    Implements ``name``, ``get_config``, ``build_from_config`` per the
    chromadb >=0.5 contract so the collection metadata is portable and the
    library doesn't log a DeprecationWarning on every test run.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_llm()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chroma signature
        # chromadb types Embeddings as a list of numpy arrays; lists are fine at
        # runtime, so we cast to keep the static checker quiet.
        return self.client.embed(list(input))  # type: ignore[return-value]

    # ----- chromadb 0.5+ embedding-function contract ------------------- #

    @staticmethod
    def name() -> str:
        return "regpilot-embeddings"

    def get_config(self) -> dict[str, Any]:
        return {
            "backend": settings.llm_backend,
            "chat_model": getattr(self.client, "chat_model", "stub"),
            "embed_model": getattr(self.client, "embed_model", "stub"),
        }

    @classmethod
    def build_from_config(cls, _: dict[str, Any]) -> RegPilotEmbeddings:
        # We rebuild from the *current* runtime settings, not the stored config,
        # because the user may have flipped REGPILOT_LLM between sessions and we
        # always want the live client.
        return cls()
