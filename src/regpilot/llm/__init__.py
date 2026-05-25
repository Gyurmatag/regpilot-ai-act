"""LLM and embedding clients organised as a small package.

We split what used to be a 1000-line ``llm.py`` into one module per backend
plus shared base and helpers. The public surface that the rest of the
codebase imports is unchanged, so any ``from regpilot.llm import X`` keeps
working.

Backends:

* :class:`OllamaClient`    — local/self-hosted Ollama HTTP API (default).
* :class:`OpenAIClient`    — hosted OpenAI (``REGPILOT_LLM=openai``).
* :class:`AnthropicClient` — hosted Anthropic (``REGPILOT_LLM=anthropic``);
  embeddings fall through to Ollama via :class:`CompositeClient`.
* :class:`StubClient`      — deterministic mock for tests / CI / offline dev.

The shared surface is intentionally narrow: ``generate`` (free-form text),
``generate_structured`` (validated Pydantic instance) and ``embed`` (batch
text to vectors). :func:`get_llm` returns the right one based on
``settings.provider``.
"""

from regpilot.llm.anthropic_client import AnthropicClient, CompositeClient
from regpilot.llm.base import LLMClient, StructuredOutputError
from regpilot.llm.factory import get_llm, reset_llm_cache
from regpilot.llm.ollama import OllamaBusyError, OllamaClient
from regpilot.llm.openai_client import OpenAIClient
from regpilot.llm.stub import StubClient

# Re-exported for backwards compatibility with code that historically imported
# the private composite under its old name.
_CompositeClient = CompositeClient

__all__ = [
    "AnthropicClient",
    "CompositeClient",
    "LLMClient",
    "OllamaBusyError",
    "OllamaClient",
    "OpenAIClient",
    "StructuredOutputError",
    "StubClient",
    "get_llm",
    "reset_llm_cache",
]
