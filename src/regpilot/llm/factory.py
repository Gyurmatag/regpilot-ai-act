"""Process-wide LLM client singleton + factory.

``get_llm()`` reads ``settings.provider`` once and returns the matching
client. Failed provider init silently falls back through a sensible chain
so the app still boots in a degraded mode (you get warnings in the log,
not a crash on import).

Fallback chain:

* ``stub``          → :class:`StubClient` (always works).
* ``openai`` error  → try Ollama next; if Ollama is also unreachable, stub.
* ``anthropic`` error → try Ollama next; if Ollama is also unreachable, stub.
* ``ollama`` unreachable → stub.
"""

from __future__ import annotations

import logging

from regpilot.config import settings
from regpilot.llm.anthropic_client import AnthropicClient, CompositeClient
from regpilot.llm.base import LLMClient
from regpilot.llm.ollama import OllamaClient
from regpilot.llm.openai_client import OpenAIClient
from regpilot.llm.stub import StubClient

logger = logging.getLogger(__name__)


_cache: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return a process-wide singleton client based on settings."""

    global _cache
    if _cache is not None:
        return _cache

    provider = settings.provider

    if provider == "stub":
        logger.info("Using StubClient (REGPILOT_LLM=stub)")
        _cache = StubClient()
        return _cache

    if provider == "openai":
        try:
            client: LLMClient = OpenAIClient()
            logger.info(
                "Using OpenAIClient (chat=%s, embed=%s)",
                client.chat_model,
                client.embed_model,
            )
            _cache = client
            return _cache
        except Exception as exc:
            logger.warning("OpenAI client failed to init (%s) — falling back to Ollama", exc)

    if provider == "anthropic":
        try:
            chat = AnthropicClient()
            # Anthropic has no embeddings — wire Ollama for the dense RAG path.
            embedder = OllamaClient()
            if not embedder.health():
                raise RuntimeError(
                    "Anthropic backend needs Ollama for embeddings but Ollama is unreachable."
                )
            composite = CompositeClient(chat, embedder)
            logger.info(
                "Using AnthropicClient (chat=%s) with Ollama embeddings (embed=%s)",
                chat.chat_model,
                embedder.embed_model,
            )
            _cache = composite
            return _cache
        except Exception as exc:
            logger.warning(
                "Anthropic client failed to init (%s) — falling back to Ollama", exc
            )

    # Default: Ollama, with stub as the last resort.
    ollama = OllamaClient()
    if not ollama.health():
        logger.warning(
            "Ollama unreachable at %s — falling back to StubClient.", ollama.base_url
        )
        _cache = StubClient()
        return _cache

    logger.info(
        "Using OllamaClient (chat=%s, embed=%s) at %s",
        ollama.chat_model,
        ollama.embed_model,
        ollama.base_url,
    )
    _cache = ollama
    return _cache


def reset_llm_cache() -> None:
    """Test helper — force ``get_llm`` to re-read settings on next call."""

    global _cache
    _cache = None
