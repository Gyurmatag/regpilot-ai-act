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
        # Use the actual user description (after "Description:") rather than the
        # whole prompt; otherwise the stub leaks its own template into state.
        # ----------------------------------------------------------------- #
        if "intake_classifier" in low and "description:" in low:
            desc = prompt.split("Description:", 1)[-1].strip()
            return json.dumps(
                {
                    "system_purpose": _excerpt(desc, 200),
                    "deployment_context": "EU market",
                    "data_modalities": _guess_modalities(desc),
                    "user_role": "provider",
                    "domain": _guess_domain(desc),
                    "notes": "stub-generated",
                }
            )

        # The order of the checks matters — each node's prompt is recognised by
        # a unique sentinel string. Sentinel selection is intentionally narrow so
        # one node's prompt can't accidentally trigger another node's branch.

        # ----------------------------------------------------------------- #
        # Synthesizer (check before triage — synth prompt also mentions "risk tier")
        # ----------------------------------------------------------------- #
        if "draft a compliance roadmap" in low or "draft report for tier" in low:
            return _stub_report(prompt)

        # ----------------------------------------------------------------- #
        # Rerank
        # ----------------------------------------------------------------- #
        if "return strict json: a list of the" in low and "indices" in low:
            return "[0,1,2,3,4]"

        # ----------------------------------------------------------------- #
        # Query rewrite
        # ----------------------------------------------------------------- #
        if "query rewrite task" in low:
            return json.dumps(
                [
                    "EU AI Act obligations applicable to the described system",
                    "compliance requirements under Regulation (EU) 2024/1689",
                ]
            )

        # ----------------------------------------------------------------- #
        # Triage — return a structured tier verdict.
        # ----------------------------------------------------------------- #
        if "classify the system below by eu ai act risk tier" in low:
            desc = prompt.split("System description:", 1)[-1]
            tier, rationale = _stub_classify(desc)
            return json.dumps({"tier": tier, "rationale": rationale, "annex_iii": []})

        # ----------------------------------------------------------------- #
        # Validator self-critique — never finds gaps (the citation_validator
        # tool runs separately and is the source of truth).
        # ----------------------------------------------------------------- #
        if "self-critique" in low:
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


def _stub_report(prompt: str) -> str:
    """Stub synthesizer that lifts real Article citations from the prompt context."""

    cited = re.findall(r"Art\.\s*(\d+[a-z]?)", prompt)
    # De-dupe but keep order; cite EVERY distinct Article so the citation
    # validator + downstream eval see the full obligation set.
    seen: list[str] = []
    for a in cited:
        if a not in seen:
            seen.append(a)
    cite = ", ".join(f"Art. {a}" for a in seen) or "Art. 6"
    return (
        "## Executive summary\n"
        "Stub-generated compliance roadmap for the described system.\n\n"
        "## Risk classification\n"
        f"The system is classified per the triage rationale (see {cite}).\n\n"
        "## Obligations & deadlines\n"
        "See the obligations table above; each row cites the Article it derives from.\n\n"
        "## Recommended next steps\n"
        "1. Confirm risk classification with legal counsel.\n"
        "2. Compile technical documentation per Annex IV (if high-risk).\n"
        "3. Establish post-market monitoring per the cited Articles.\n\n"
        f"_Generated with the stub LLM. Cited: {cite}._"
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
