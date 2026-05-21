"""LLM + embedding client.

Two backends:

* ``OllamaClient`` — hits a local Ollama HTTP server (the default in
  ``docker-compose.yml``).
* ``StubClient``  — deterministic, regex-driven responses for unit tests and CI.
  Selected when ``REGPILOT_LLM=stub`` or when Ollama can't be reached.

Both expose the same minimal surface (``generate``, ``embed``) so the rest of the
codebase doesn't care which one is wired in.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from regpilot.config import settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Shared interface for both real and stub clients."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


class OllamaClient(LLMClient):
    """Thin HTTP wrapper around the Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.chat_model = chat_model or settings.chat_model
        self.embed_model = embed_model or settings.embed_model
        self._client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **_: Any,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        r = self._client.post(f"{self.base_url}/api/generate", json=payload)
        r.raise_for_status()
        return str(r.json().get("response", "")).strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        # Ollama embeds one prompt per request; batching is not in the public API yet.
        for t in texts:
            r = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": t},
            )
            r.raise_for_status()
            out.append(list(r.json()["embedding"]))
        return out

    def health(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# Stub
# --------------------------------------------------------------------------- #


_PROHIBITED_HINTS = re.compile(
    r"\b(social\s+scoring|emotion\s+recognition\s+(in\s+the\s+workplace|at\s+work)"
    r"|untargeted\s+scraping|predictive\s+policing|biometric\s+categori[sz]ation"
    r"|real-time\s+remote\s+biometric)\b",
    re.I,
)
_HIGH_RISK_HINTS = re.compile(
    r"\b(recruit(ment|ing)?|hir(e|ing)|cv\s+screening|credit\s+scoring"
    r"|education|exam\s+proctor|law\s+enforcement|migration|critical\s+infrastructure"
    r"|medical\s+device|judicial)\b",
    re.I,
)
_LIMITED_HINTS = re.compile(
    r"\b(chatbot|deepfake|synthetic\s+(media|content)|generative)\b", re.I
)


def _deterministic_embedding(text: str, dim: int = 256) -> list[float]:
    """Hash-based pseudo-embedding so the stub still drives a working RAG path."""

    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat until we have enough bytes, then map to [-1, 1] floats.
    needed = math.ceil(dim / 32)
    expanded = (h * needed)[:dim]
    return [(b - 128) / 128.0 for b in expanded]


class StubClient(LLMClient):
    """Deterministic mock used by tests and as a fallback when Ollama is down."""

    chat_model = "stub"
    embed_model = "stub"

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        low = prompt.lower()

        # ----------------------------------------------------------------- #
        # Intake — emit a JSON structure the intake node can parse.
        # ----------------------------------------------------------------- #
        if "extract the following fields" in low or "intake_classifier" in low:
            return json.dumps(
                {
                    "system_purpose": _excerpt(prompt, 120),
                    "deployment_context": "EU market",
                    "data_modalities": _guess_modalities(prompt),
                    "user_role": "provider",
                    "domain": _guess_domain(prompt),
                    "notes": "stub-generated",
                }
            )

        # ----------------------------------------------------------------- #
        # Triage — return a structured tier verdict.
        # ----------------------------------------------------------------- #
        if "classify the system" in low or "risk tier" in low:
            tier, rationale = _stub_classify(prompt)
            return json.dumps({"tier": tier, "rationale": rationale, "annex_iii": []})

        # ----------------------------------------------------------------- #
        # Rerank — return a JSON list of indices.
        # ----------------------------------------------------------------- #
        if "rerank" in low:
            # Echo the input order unchanged (top-k handled by the caller).
            return "[0,1,2,3,4]"

        # ----------------------------------------------------------------- #
        # Query rewrite — return one paraphrase + the original.
        # ----------------------------------------------------------------- #
        if "query rewrite" in low or "hyde" in low:
            return json.dumps(
                [
                    "EU AI Act obligations applicable to the described system",
                    "compliance requirements under Regulation (EU) 2024/1689",
                ]
            )

        # ----------------------------------------------------------------- #
        # Synthesizer — emit a templated report skeleton.
        # ----------------------------------------------------------------- #
        if "draft" in low and "report" in low:
            return _stub_report(prompt)

        # ----------------------------------------------------------------- #
        # Validator — never finds gaps.
        # ----------------------------------------------------------------- #
        if "validator" in low or "self-critique" in low:
            return json.dumps({"ok": True, "issues": []})

        return "Stub LLM response."

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_deterministic_embedding(t) for t in texts]


def _excerpt(text: str, n: int) -> str:
    return text.strip().replace("\n", " ")[:n]


def _guess_modalities(text: str) -> list[str]:
    found: list[str] = []
    for needle, label in [
        ("image", "image"),
        ("video", "video"),
        ("audio", "audio"),
        ("voice", "audio"),
        ("text", "text"),
        ("biometric", "biometric"),
        ("face", "biometric"),
    ]:
        if needle in text.lower():
            found.append(label)
    return found or ["text"]


def _guess_domain(text: str) -> str:
    low = text.lower()
    for needle, label in [
        ("hir", "HR / recruitment"),
        ("recruit", "HR / recruitment"),
        ("cv", "HR / recruitment"),
        ("credit", "financial services"),
        ("medical", "healthcare"),
        ("hospital", "healthcare"),
        ("educat", "education"),
        ("exam", "education"),
        ("police", "law enforcement"),
        ("border", "migration / border"),
    ]:
        if needle in low:
            return label
    return "general"


def _stub_classify(text: str) -> tuple[str, str]:
    if _PROHIBITED_HINTS.search(text):
        return "prohibited", "matches an Article 5 prohibited-practice pattern"
    if _HIGH_RISK_HINTS.search(text):
        return "high_risk", "matches an Annex III high-risk use-case pattern"
    if _LIMITED_HINTS.search(text):
        return "limited_risk", "subject to Article 50 transparency obligations"
    return "minimal_risk", "no high-risk or prohibited indicators detected"


def _stub_report(_: str) -> str:
    return (
        "## Compliance roadmap (stub)\n"
        "**Risk tier:** see triage output.\n\n"
        "### Applicable obligations\n"
        "- See cited Articles.\n\n"
        "### Key deadlines\n"
        "- See deadline_calculator output.\n\n"
        "### Recommended next steps\n"
        "1. Confirm risk classification with legal counsel.\n"
        "2. Compile technical documentation per Annex IV (if high-risk).\n"
        "3. Set up post-market monitoring.\n\n"
        "_Generated with the stub LLM — switch REGPILOT_LLM=ollama for a real report._"
    )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


_cache: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return a process-wide singleton LLM client based on settings."""

    global _cache
    if _cache is not None:
        return _cache

    if settings.is_stub:
        logger.info("Using StubClient (REGPILOT_LLM=stub)")
        _cache = StubClient()
        return _cache

    client = OllamaClient()
    if not client.health():
        logger.warning(
            "Ollama unreachable at %s — falling back to StubClient.", client.base_url
        )
        _cache = StubClient()
        return _cache

    logger.info(
        "Using OllamaClient (chat=%s, embed=%s) at %s",
        client.chat_model,
        client.embed_model,
        client.base_url,
    )
    _cache = client
    return _cache


def reset_llm_cache() -> None:
    """Test helper — force ``get_llm`` to re-read settings."""

    global _cache
    _cache = None
