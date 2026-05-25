"""Hosted OpenAI client.

Uses the chat completions + embeddings APIs. Structured output goes through
``beta.chat.completions.parse`` with a Pydantic ``response_format`` so we
get guaranteed schema-conformant output instead of having to parse JSON out
of free-form text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from regpilot.config import settings
from regpilot.llm.base import LLMClient, StructuredOutputError

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIClient(LLMClient):
    """OpenAI chat + embeddings via the official SDK."""

    provider = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        # Lazy import keeps the SDK off the import path of stub-only test runs.
        from openai import OpenAI

        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OpenAI client requested but OPENAI_API_KEY is empty. "
                "Either export OPENAI_API_KEY or switch REGPILOT_LLM to "
                "ollama / stub."
            )
        url = base_url or settings.openai_base_url or None
        self._client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)
        self.chat_model = chat_model or settings.openai_chat_model
        self.embed_model = embed_model or settings.openai_embed_model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **_: Any,
    ) -> str:
        msgs = _build_messages(system, prompt)
        resp = self._client.chat.completions.create(
            model=self.chat_model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

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
        msgs = _build_messages(system, prompt)
        try:
            parsed = self._client.beta.chat.completions.parse(
                model=self.chat_model,
                messages=msgs,
                response_format=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # ``parsed.choices[0].message.parsed`` is typed as the schema we
            # passed in via ``response_format``; the SDK overload returns it
            # as ``Any`` so we cast back to ``T`` for the caller.
            obj = parsed.choices[0].message.parsed
            if obj is None:
                raise StructuredOutputError(
                    f"OpenAI returned no parsed object for {schema.__name__}."
                )
            return cast(T, obj)
        except StructuredOutputError:
            raise
        except Exception as exc:
            logger.warning(
                "OpenAI structured output failed (%s): %s — falling back to JSON prompt",
                schema.__name__,
                exc,
            )
            # Fall back to the base prompt-engineered path.
            return super().generate_structured(
                prompt,
                schema,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI batches natively and is much more efficient than parallel HTTP.
        cleaned = [t if t and t.strip() else " " for t in texts]
        resp = self._client.embeddings.create(model=self.embed_model, input=cleaned)
        return [d.embedding for d in resp.data]


def _build_messages(
    system: str | None, prompt: str
) -> list[ChatCompletionMessageParam]:
    """Assemble the ``messages=...`` payload OpenAI's chat APIs expect.

    Returned typed against the SDK's ``ChatCompletionMessageParam`` so
    callers don't need ``# type: ignore[arg-type]``. We construct the
    inner dicts inline so each entry literally matches one of the
    TypedDict arms (``SystemMessageParam``, ``UserMessageParam``).
    """

    msgs: list[ChatCompletionMessageParam] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs
