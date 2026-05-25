"""Ollama HTTP client.

Talks to a local or co-located Ollama server (``OLLAMA_BASE_URL``). Uses
Ollama 0.5+'s schema-as-format constrained generation for structured output
so the model is grammar-locked to a schema-conformant JSON reply, not just
to "some valid JSON".
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from regpilot.config import settings
from regpilot.llm.base import LLMClient, StructuredOutputError
from regpilot.llm.helpers import safe_json_obj

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaBusyError(RuntimeError):
    """Raised when Ollama returns HTTP 503 (queue full, ``OLLAMA_MAX_QUEUE`` hit).

    Separate exception class so tenacity can retry only on this — connection
    errors and unexpected 5xx bubble up immediately for the caller to handle.
    """


class OllamaClient(LLMClient):
    """Thin HTTP wrapper around the Ollama REST API."""

    provider = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.chat_model = chat_model or settings.chat_model
        self.embed_model = embed_model or settings.embed_model
        self._timeout = timeout if timeout is not None else settings.ollama_timeout_s
        self._client = httpx.Client(timeout=self._timeout)
        self._embed_pool = ThreadPoolExecutor(
            max_workers=max(1, settings.embed_parallelism),
            thread_name_prefix="ollama-embed",
        )

    # --------------------------------------------------------------------- #
    # generate / generate_structured
    # --------------------------------------------------------------------- #

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((OllamaBusyError, httpx.ReadTimeout)),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        format_json: bool = False,
        json_schema: dict[str, Any] | None = None,
        seed: int = 42,
        **_: Any,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            # ``seed`` makes greedy decoding deterministic across runs given
            # identical (model, prompt, temperature). Critical for our
            # reproducible-eval guarantee.
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
        }
        if system:
            payload["system"] = system
        # 0.5+ JSON-schema-as-format uses constrained grammar to force the
        # output to conform field-for-field. Stronger than the legacy
        # ``format="json"`` which only guarantees the bytes are parseable
        # JSON, not that they match the schema.
        if json_schema is not None:
            payload["format"] = json_schema
        elif format_json:
            payload["format"] = "json"

        r = self._client.post(f"{self.base_url}/api/generate", json=payload)
        if r.status_code == 503:
            raise OllamaBusyError(f"Ollama queue full at {self.base_url} (HTTP 503)")
        r.raise_for_status()
        return str(r.json().get("response", "")).strip()

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
        """Use Ollama 0.5+'s schema-as-format constrained generation.

        The schema goes in the ``format`` parameter (not the literal string
        ``"json"``) so the sampler is grammar-locked. The prompt itself does
        not include the schema text — earlier we discovered that 3B-scale
        models tend to echo the schema back as their output if it's in the
        prompt, which broke validation downstream.
        """

        raw = self.generate(
            prompt.strip(),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=schema.model_json_schema(),
        )
        obj = safe_json_obj(raw)
        try:
            return schema.model_validate(obj)
        except Exception as exc:
            logger.warning(
                "Ollama structured output failed (%s): %s; raw=%r",
                schema.__name__,
                exc,
                raw[:200],
            )
            raise StructuredOutputError(
                f"Failed to parse {schema.__name__} from Ollama output."
            ) from exc

    # --------------------------------------------------------------------- #
    # embed
    # --------------------------------------------------------------------- #

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((OllamaBusyError, httpx.ReadTimeout)),
        reraise=True,
    )
    def _embed_one(self, text: str) -> list[float]:
        # Empty / whitespace input causes Ollama to return [] which then
        # breaks ChromaDB's "non-empty vector" validation downstream. Swap
        # in a single space so we always get a usable vector back.
        payload_text = text if text and text.strip() else " "
        r = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": payload_text},
        )
        if r.status_code == 503:
            raise OllamaBusyError(f"Ollama queue full at {self.base_url} (HTTP 503)")
        r.raise_for_status()
        emb = list(r.json().get("embedding") or [])
        if not emb:
            raise RuntimeError(
                f"Ollama returned an empty embedding for input "
                f"(len={len(text)}): {text[:80]!r}"
            )
        return emb

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # ThreadPoolExecutor.map preserves input order, so the returned list
        # of vectors is positionally aligned with ``texts``.
        return list(self._embed_pool.map(self._embed_one, texts))

    # --------------------------------------------------------------------- #
    # health
    # --------------------------------------------------------------------- #

    def health(self) -> bool:
        """Cheap reachability check used by the factory at boot."""

        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False
