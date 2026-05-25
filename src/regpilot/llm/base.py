"""Shared LLM client interface and the structured-output exception."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from regpilot.llm.helpers import safe_json_obj, wrap_with_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when a structured-output call can't be coerced into the schema.

    Callers catch this to fall back to a deterministic path (heuristic
    intake, template synthesizer, etc.) instead of crashing the whole
    LangGraph run on a single bad LLM reply.
    """


class LLMClient(ABC):
    """The protocol every backend implements.

    Three operations: ``generate`` for free-form text, ``embed`` for batch
    text-to-vector, and ``generate_structured`` for typed JSON via Pydantic.
    Subclasses override the structured method with a provider-native API
    (Ollama's ``format=<schema>``, OpenAI's ``response_format=``, Anthropic
    tool-use) when one exists; otherwise the base implementation falls back
    to prompt-engineering a JSON-only reply and parsing it.
    """

    chat_model: str = ""
    embed_model: str = ""
    provider: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

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
        """Default fallback: prompt-engineer a JSON-only reply, then validate.

        Backends that have a real structured-output API should override this
        with their native call (much higher schema-conformance rate than
        asking a 3B model to follow instructions).
        """

        json_prompt = wrap_with_schema(prompt, schema)
        raw = self.generate(
            json_prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        obj = safe_json_obj(raw)
        try:
            return schema.model_validate(obj)
        except Exception as exc:
            logger.warning(
                "Structured output validation failed (%s): %s", schema.__name__, exc
            )
            raise StructuredOutputError(
                f"Failed to parse {schema.__name__} from model output."
            ) from exc
