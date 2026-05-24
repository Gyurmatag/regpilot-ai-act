"""LLM + embedding clients.

Four backends, all implementing the same :class:`LLMClient` protocol:

* :class:`OllamaClient`    — local/self-hosted Ollama HTTP API (default).
* :class:`OpenAIClient`    — hosted OpenAI (``REGPILOT_LLM=openai``).
* :class:`AnthropicClient` — hosted Anthropic (``REGPILOT_LLM=anthropic``).
  Anthropic has no native embedding API, so embeddings fall through to Ollama.
* :class:`StubClient`      — deterministic mock for unit tests / CI / offline dev.

The shared surface is intentionally narrow: ``generate`` (free-form text),
``generate_structured`` (JSON object matching a Pydantic schema) and ``embed``
(batch text → vectors). The factory :func:`get_llm` picks the right
implementation from ``settings.provider`` and caches a process-wide singleton.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from abc import ABC, abstractmethod
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

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaBusyError(RuntimeError):
    """Raised when Ollama returns HTTP 503 (queue full, ``OLLAMA_MAX_QUEUE`` hit)."""


# --------------------------------------------------------------------------- #
# Shared interface
# --------------------------------------------------------------------------- #


class LLMClient(ABC):
    """Shared surface for free-form, structured, and embedding calls."""

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
        """Return an instance of ``schema`` parsed from the model's output.

        Default implementation: ask for JSON in the prompt, extract the JSON
        object, validate against the Pydantic schema. Provider-specific
        subclasses override this with native structured-output APIs when
        available (OpenAI response_format, Ollama format=json, etc.).
        """

        json_prompt = _wrap_with_schema(prompt, schema)
        raw = self.generate(
            json_prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        obj = _safe_json_obj(raw)
        try:
            return schema.model_validate(obj)
        except Exception as exc:
            logger.warning(
                "Structured output validation failed (%s): %s", schema.__name__, exc
            )
            # Re-raise as a generic structured-output error so callers can
            # fall back cleanly.
            raise StructuredOutputError(
                f"Failed to parse {schema.__name__} from model output."
            ) from exc


class StructuredOutputError(RuntimeError):
    """Raised when a structured-output call can't be coerced into the schema."""


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


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
            # ``seed`` makes Ollama greedy decoding fully deterministic across
            # runs given identical (model, prompt, temperature). Critical for
            # reproducible eval scores in the LLM-primary mode.
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
        }
        if system:
            payload["system"] = system
        # Ollama 0.5+ supports a JSON-schema-as-format mode that uses
        # constrained grammar to force the output to conform to the schema
        # (much stronger than the legacy ``format=json`` which only
        # guarantees parseable JSON, not schema-conformant JSON).
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

        The schema is sent as the ``format`` parameter (not the legacy
        ``"json"`` string) so Ollama's grammar-constrained sampler forces
        the output to conform field-for-field — no more JSON-but-wrong-shape
        failures the loose ``format=json`` mode produces. The prompt itself
        no longer carries the schema text, which previously confused 3B-scale
        models into echoing the schema back as their output.
        """

        # Strip the trailing newlines so the model gets a clean instruction.
        clean_prompt = prompt.strip()
        raw = self.generate(
            clean_prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=schema.model_json_schema(),
        )
        obj = _safe_json_obj(raw)
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

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((OllamaBusyError, httpx.ReadTimeout)),
        reraise=True,
    )
    def _embed_one(self, text: str) -> list[float]:
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
        return list(self._embed_pool.map(self._embed_one, texts))

    def health(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class OpenAIClient(LLMClient):
    """Hosted OpenAI client — uses the chat completions + embeddings APIs.

    Structured output goes through OpenAI's ``response_format`` JSON schema
    feature so we get guaranteed schema-conformant output (no regex hacks).
    """

    provider = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
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
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.chat_model,
            messages=msgs,  # type: ignore[arg-type]
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
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            parsed = self._client.beta.chat.completions.parse(
                model=self.chat_model,
                messages=msgs,  # type: ignore[arg-type]
                response_format=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            obj = parsed.choices[0].message.parsed
            if obj is None:
                raise StructuredOutputError(
                    f"OpenAI returned no parsed object for {schema.__name__}."
                )
            return obj  # type: ignore[return-value]
        except StructuredOutputError:
            raise
        except Exception as exc:
            logger.warning(
                "OpenAI structured output failed (%s): %s — falling back to JSON prompt",
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
        if not texts:
            return []
        # OpenAI handles batching natively + much more efficiently than parallel HTTP.
        cleaned = [t if t and t.strip() else " " for t in texts]
        resp = self._client.embeddings.create(model=self.embed_model, input=cleaned)
        return [d.embedding for d in resp.data]


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicClient(LLMClient):
    """Hosted Anthropic Claude client.

    Anthropic has no embedding endpoint, so :meth:`embed` raises and the
    factory wires Ollama in as a parallel embed-only client. Structured
    output uses tool calling to force a schema-conformant JSON response.
    """

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
        # Anthropic's "tool use" feature is the structured-output story —
        # define a fake tool whose input schema matches our Pydantic class,
        # and ``tool_choice`` forces the model to fill it in.
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
                    return schema.model_validate(block.input)  # type: ignore[attr-defined]
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


class _CompositeClient(LLMClient):
    """Composes a chat client (Anthropic/OpenAI) with an embed-only client (Ollama).

    Anthropic has no embeddings API, so when the user picks ``REGPILOT_LLM=
    anthropic`` we still need *something* for the RAG dense path. We default
    to a co-located Ollama for embeddings and let the chat client handle
    ``generate`` / ``generate_structured``.
    """

    def __init__(self, chat: LLMClient, embedder: LLMClient) -> None:
        self._chat = chat
        self._embedder = embedder
        self.chat_model = chat.chat_model
        self.embed_model = embedder.embed_model
        self.provider = chat.provider

    def generate(self, *a: Any, **kw: Any) -> str:
        return self._chat.generate(*a, **kw)

    def generate_structured(self, *a: Any, **kw: Any) -> Any:
        return self._chat.generate_structured(*a, **kw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)


# --------------------------------------------------------------------------- #
# Stub — deterministic mock for tests / offline dev
# --------------------------------------------------------------------------- #


_PROHIBITED_HINTS = re.compile(
    r"\b(social\s+scoring|emotion\s+recognition\s+(in\s+the\s+workplace|at\s+work)"
    r"|untargeted\s+scraping|predictive\s+policing|biometric\s+categori[sz]ation"
    r"|real-time\s+remote\s+biometric)\b",
    re.I,
)
_HIGH_RISK_HINTS = re.compile(
    r"\b(recruit\w*|hir(e|ing)|cv\s+screening|credit\s+scoring"
    r"|education|exam\s+proctor|law\s+enforcement|migration|critical\s+infrastructure"
    r"|medical\s+device|judicial"
    # Biometric variants — the stub classifier ALSO has to recognise these
    # so tests stay deterministic without a real LLM.
    r"|emotion\w*|face\w*|facial|biometric\w*"
    r"|fingerprint\w*|iris|gait|cctv|surveillance|mood|walking\s+pattern)\b",
    re.I,
)
_GPAI_SYSTEMIC_HINTS = re.compile(
    r"\b(10\s*\^?\s*25\s*flops?|systemic[\s\-]risk|frontier\s+(model|llm|ai))\b",
    re.I,
)
_GPAI_HINTS = re.compile(
    r"\b(gpai|general[\s\-_]?purpose(\s+ai)?|foundation\s+(model|llm|ai)"
    r"|large\s+language\s+model|llms?)\b",
    re.I,
)
_LIMITED_HINTS = re.compile(
    r"\b(chatbot|deepfake|synthetic\s+(media|content)|generative)\b", re.I
)


def _deterministic_embedding(text: str, dim: int = 256) -> list[float]:
    """Hash-based pseudo-embedding so the stub still drives a working RAG path."""

    h = hashlib.sha256(text.encode("utf-8")).digest()
    needed = math.ceil(dim / 32)
    expanded = (h * needed)[:dim]
    return [(b - 128) / 128.0 for b in expanded]


class StubClient(LLMClient):
    """Deterministic mock used by tests and as a fallback when Ollama is down."""

    chat_model = "stub"
    embed_model = "stub"
    provider = "stub"

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        low = prompt.lower()

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

        if "draft a compliance roadmap" in low or "draft report for tier" in low:
            return _stub_report(prompt)

        if "return strict json: a list of the" in low and "indices" in low:
            return "[0,1,2,3,4]"

        if "query rewrite task" in low:
            return json.dumps(
                [
                    "EU AI Act obligations applicable to the described system",
                    "compliance requirements under Regulation (EU) 2024/1689",
                ]
            )

        if "classify the system below by eu ai act risk tier" in low:
            desc = prompt.split("System description:", 1)[-1]
            tier, rationale = _stub_classify(desc)
            return json.dumps(
                {"tier": tier, "rationale": rationale, "annex_iii_areas": []}
            )

        if "self-critique" in low:
            return json.dumps({"ok": True, "issues": []})

        return "Stub LLM response."

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
        """Schema-aware deterministic mock.

        The stub recognises each schema by name + class fields and returns a
        synthetic-but-valid instance derived from the prompt. This keeps unit
        tests + offline dev fully deterministic without depending on a real LLM.
        """

        name = schema.__name__
        fields = set(getattr(schema, "model_fields", {}).keys())

        # ClassificationResult (risk classifier)
        if name == "ClassificationResult" or fields.issuperset({"tier", "rationale"}):
            desc = _extract_after(prompt, "System description:") or _extract_after(
                prompt, "Description:"
            ) or prompt
            tier, rationale = _stub_classify(desc)
            areas = _stub_annex_areas(desc) if tier == "high_risk" else []
            return schema.model_validate(
                {
                    "tier": tier,
                    "rationale": rationale,
                    "annex_iii_areas": areas,
                    "art_5_codes": [],
                }
            )

        # IntakeSchema (intake)
        if name == "IntakeSchema" or fields.issuperset(
            {"system_purpose", "user_role", "data_modalities"}
        ):
            desc = _extract_after(prompt, "Description:") or prompt
            return schema.model_validate(
                {
                    "system_purpose": _excerpt(desc, 200),
                    "deployment_context": "EU market",
                    "data_modalities": _guess_modalities(desc),
                    "user_role": "provider",
                    "domain": _guess_domain(desc),
                    "notes": "stub-generated",
                }
            )

        # ReportSections (synthesizer)
        if name == "ReportSections" or fields.issuperset(
            {"executive_summary", "risk_classification_narrative", "recommended_next_steps"}
        ):
            tier_match = re.search(r"Risk tier.*?:\s*([A-Za-z _\-]+)", prompt)
            tier = (
                tier_match.group(1).strip().lower().replace(" ", "_") if tier_match else "unknown"
            )
            articles = sorted(
                {
                    a
                    for a in re.findall(
                        r"\d{4}-\d{2}-\d{2}\s+\u2014\s+Art\.\s*(\d+[a-z]?)", prompt
                    )
                }
            ) or ["6"]
            steps = _stub_next_steps(tier).splitlines()
            return schema.model_validate(
                {
                    "executive_summary": (
                        f"This system is classified per the supplied triage. The "
                        f"compliance roadmap below lists the applicable Articles "
                        f"({', '.join('Art. ' + a for a in articles)}) and their "
                        f"Article 113 phased deadlines."
                    ),
                    "risk_classification_narrative": (
                        f"The triage analysis assigned this system to the relevant "
                        f"tier. The applicable obligations derive from the cited "
                        f"Articles ({', '.join('Art. ' + a for a in articles)}). "
                        f"Each obligation entry includes the precise Article 113 "
                        f"date the duty becomes enforceable."
                    ),
                    "recommended_next_steps": [
                        s.split(". ", 1)[1] if ". " in s else s for s in steps if s.strip()
                    ],
                }
            )

        # Default: try to fish JSON out of generate() output.
        raw = self.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        obj = _safe_json_obj(raw)
        return schema.model_validate(obj or {})

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_deterministic_embedding(t) for t in texts]


def _extract_after(text: str, marker: str) -> str:
    """Return the text immediately after ``marker``, stopping at the next
    blank line or all-caps section header. Used by the StubClient to isolate
    the user description from the surrounding prompt boilerplate (decision
    rules, tier definitions etc. that mention literal GPAI / Art. 5 tokens)."""

    idx = text.find(marker)
    if idx < 0:
        return ""
    tail = text[idx + len(marker):].lstrip()
    # Stop at the next blank line (separates description from "Decision rules:" /
    # "Return strict JSON" etc.) so we don't accidentally match keywords in the
    # prompt's instructional boilerplate.
    sep = tail.find("\n\n")
    if sep > 0:
        tail = tail[:sep]
    return tail.strip()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _wrap_with_schema(prompt: str, schema: type[BaseModel]) -> str:
    """Append a strict JSON-only instruction with the Pydantic schema."""

    schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return (
        f"{prompt}\n\n"
        f"Reply with STRICT JSON only matching this schema (no commentary, "
        f"no markdown fence):\n{schema_json}"
    )


def _safe_json_obj(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    # Try the whole string first (Ollama format=json / OpenAI response_format).
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    # Fallback: extract first top-level {...} block.
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


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


_STUB_HIGH_RISK_AREAS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(recruit(ment|ing)?|hir(e|ing)|cv\s+screening)\b", re.I),
     "Employment, worker management, access to self-employment"),
    (re.compile(r"\b(credit\s+scoring|loan|insurance)\b", re.I),
     "Access to and enjoyment of essential private and public services and benefits"),
    (re.compile(r"\b(education|exam\s+proctor|student|school|grading)\b", re.I),
     "Education and vocational training"),
    (re.compile(r"\b(law\s+enforcement|police|recidivism|polygraph)\b", re.I),
     "Law enforcement"),
    (re.compile(r"\b(critical\s+infrastructure|power\s+grid|electricity|water\s+supply)\b", re.I),
     "Critical infrastructure"),
    (re.compile(r"\b(border|migration|asylum|visa)\b", re.I),
     "Migration, asylum, border control"),
    (re.compile(r"\b(judicial|court|election|referendum)\b", re.I),
     "Administration of justice and democratic processes"),
    (re.compile(
        r"\b(face\w*|facial|iris|fingerprint\w*|emotion\w*|mood|biometric\w*"
        r"|gait|cctv|surveillance|walking\s+pattern)\b",
        re.I,
    ), "Biometrics"),
]


def _stub_classify(text: str) -> tuple[str, str]:
    if _PROHIBITED_HINTS.search(text):
        return "prohibited", "matches an Article 5 prohibited-practice pattern"
    if _HIGH_RISK_HINTS.search(text):
        return "high_risk", "matches an Annex III high-risk use-case pattern"
    if _GPAI_SYSTEMIC_HINTS.search(text):
        return "general_purpose_systemic", "matches Article 51 systemic-risk GPAI markers"
    if _GPAI_HINTS.search(text):
        return "general_purpose", "matches a general-purpose AI model pattern"
    if _LIMITED_HINTS.search(text):
        return "limited_risk", "subject to Article 50 transparency obligations"
    return "minimal_risk", "no high-risk or prohibited indicators detected"


def _stub_annex_areas(text: str) -> list[str]:
    """Return the canonical Annex III area names that match the input text."""

    seen: list[str] = []
    for rx, area in _STUB_HIGH_RISK_AREAS:
        if rx.search(text) and area not in seen:
            seen.append(area)
    return seen


def _stub_report(prompt: str) -> str:
    obligation_articles = re.findall(
        r"\d{4}-\d{2}-\d{2}\s+\u2014\s+Art\.\s*(\d+[a-z]?)", prompt
    )
    seen: list[str] = []
    for a in obligation_articles:
        if a not in seen:
            seen.append(a)
    cite = ", ".join(f"Art. {a}" for a in seen) or "Art. 6"

    tier_match = re.search(r"Risk tier:\s*([a-z_]+)", prompt)
    tier = tier_match.group(1) if tier_match else "unknown"
    next_steps = _stub_next_steps(tier)

    return (
        "## Executive summary\n"
        "A compliance roadmap based on the supplied EU AI Act context.\n\n"
        "## Risk classification\n"
        f"The system has been classified per the triage rationale. "
        f"Applicable Articles: {cite}.\n\n"
        "## Obligations & deadlines\n"
        "The full obligation table is shown in the trace panel; each entry "
        "cites the Article it derives from.\n\n"
        "## Recommended next steps\n"
        f"{next_steps}\n"
    )


def _stub_next_steps(tier: str) -> str:
    if tier == "high_risk":
        return (
            "1. Confirm the risk classification and applicable Annex III area with legal counsel.\n"
            "2. Map each obligation in the table to an internal owner and target date.\n"
            "3. Compile technical documentation per Annex IV (Art. 11) and prepare for the conformity assessment (Art. 43)."
        )
    if tier == "limited_risk":
        return (
            "1. Implement the Article 50 transparency disclosures in the user-facing flow.\n"
            "2. Label any AI-generated or AI-modified media (deepfakes, synthetic text) accordingly.\n"
            "3. Track Article 50 implementing guidance from the AI Office."
        )
    if tier == "minimal_risk":
        return (
            "1. No mandatory obligations apply, but adopt a voluntary code of conduct per Article 95.\n"
            "2. Re-check classification annually as the system evolves.\n"
            "3. Apply general data-protection and product-liability law as a baseline."
        )
    return (
        "1. Cease placing the system on the EU market and putting it into service.\n"
        "2. Consult legal counsel on remediation and potential redesign.\n"
        "3. Communicate the change to internal stakeholders and customers."
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
            composite = _CompositeClient(chat, embedder)
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

    # Default: Ollama (with stub fallback if unreachable).
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
    """Test helper — force ``get_llm`` to re-read settings."""

    global _cache
    _cache = None
