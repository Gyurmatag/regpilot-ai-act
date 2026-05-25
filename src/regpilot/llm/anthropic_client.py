"""Hosted Anthropic Claude client + the Anthropic-plus-Ollama composite.

Anthropic has no embedding endpoint, so when the user picks
``REGPILOT_LLM=anthropic`` the factory wires an Ollama instance in as the
embedding-only provider via :class:`CompositeClient`. Structured output uses
Anthropic's tool-use feature to force a schema-conformant JSON reply.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from regpilot.config import settings
from regpilot.llm.base import LLMClient, StructuredOutputError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AnthropicClient(LLMClient):
    """Anthropic chat via the official SDK. No embeddings here."""

    provider = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str | None = None,
    ) -> None:
        from anthropic import Anthropic

        key = api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError(
                "Anthropic client requested but ANTHROPIC_API_KEY is empty. "
                "Either export ANTHROPIC_API_KEY or switch REGPILOT_LLM to "
                "ollama / openai / stub."
            )
        self._client = Anthropic(api_key=key)
        self.chat_model = chat_model or settings.anthropic_chat_model
        self.embed_model = "ollama-fallback"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **_: Any,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.chat_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return text.strip()

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> T:
        # Tool use is Anthropic's structured-output story: declare a tool
        # whose input schema matches our Pydantic class, then force-pick it
        # with ``tool_choice``. The model has no choice but to fill it.
        tool = {
            "name": "emit_" + schema.__name__.lower(),
            "description": f"Emit a {schema.__name__} object describing the answer.",
            "input_schema": schema.model_json_schema(),
        }
        api_kwargs: dict[str, Any] = {
            "model": self.chat_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            api_kwargs["system"] = system
        try:
            resp = self._client.messages.create(**api_kwargs)
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    # The SDK exposes tool_use blocks as either
                    # ToolUseBlock (with .input) or a partial dict at
                    # stream time — `getattr` covers both without type-
                    # ignore noise.
                    payload = getattr(block, "input", {}) or {}
                    return schema.model_validate(payload)
            raise StructuredOutputError(
                f"Anthropic produced no tool_use block for {schema.__name__}."
            )
        except StructuredOutputError:
            raise
        except Exception as exc:
            logger.warning(
                "Anthropic structured output failed (%s): %s — falling back to JSON prompt",
                schema.__name__,
                exc,
            )
            return super().generate_structured(
                prompt,
                schema,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic has no embedding API. The factory wires Ollama as the "
            "embedding provider when REGPILOT_LLM=anthropic; this method "
            "should never be called directly."
        )


class CompositeClient(LLMClient):
    """Composes a chat client (Anthropic) with an embed-only client (Ollama).

    The Decorator pattern: same surface as a plain :class:`LLMClient` but
    routes ``generate`` / ``generate_structured`` to one inner client and
    ``embed`` to another. Used only when ``REGPILOT_LLM=anthropic``.
    """

    def __init__(self, chat: LLMClient, embedder: LLMClient) -> None:
        self._chat = chat
        self._embedder = embedder
        self.chat_model = chat.chat_model
        self.embed_model = embedder.embed_model
        self.provider = chat.provider

    def generate(self, *args: Any, **kwargs: Any) -> str:
        return self._chat.generate(*args, **kwargs)

    def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        return self._chat.generate_structured(*args, **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)
