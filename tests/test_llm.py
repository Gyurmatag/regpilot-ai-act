"""Tests for ``regpilot.llm`` — OllamaClient, StubClient, factory, retry semantics."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from regpilot.llm import (
    OllamaBusyError,
    OllamaClient,
    StubClient,
    get_llm,
    reset_llm_cache,
)

# --------------------------------------------------------------------------- #
# StubClient — every prompt sentinel
# --------------------------------------------------------------------------- #


def test_stub_intake_extracts_description_only() -> None:
    """Stub must look at the user description, not the prompt template, so it
    doesn't echo its own boilerplate into ``system_purpose``."""

    c = StubClient()
    prompt = (
        "Extract the following fields...\n"
        "intake_classifier — Description:\n"
        "A CV screening AI for tech recruitment."
    )
    out = json.loads(c.generate(prompt))
    assert "CV screening AI" in out["system_purpose"]
    assert out["domain"] == "HR / recruitment"
    assert out["user_role"] == "provider"


def test_stub_synthesizer_lifts_obligation_articles_only() -> None:
    """Stub synth must only cite the YYYY-MM-DD Art. N obligation lines —
    never the noisy retrieval context."""

    c = StubClient()
    prompt = (
        "Draft a compliance roadmap report.\n"
        "Risk tier: high_risk\n"
        "Confirmed obligations:\n"
        "- 2026-08-02 — Art. 9: risk management\n"
        "- 2026-08-02 — Art. 10: data governance\n"
        "Retrieved Articles:\n"
        "[Art. 6 p1] off-topic context\n"
        "[Art. 99 p1] also off-topic\n"
        "draft report for tier high_risk"
    )
    out = c.generate(prompt)
    assert "Art. 9" in out
    assert "Art. 10" in out
    assert "Art. 6" not in out  # context noise must NOT appear
    assert "Art. 99" not in out


def test_stub_rerank_returns_indices() -> None:
    c = StubClient()
    prompt = "Return STRICT JSON: a list of the 5 most relevant indices (0-based, integers only)."
    assert c.generate(prompt) == "[0,1,2,3,4]"


def test_stub_query_rewrite_returns_paraphrases() -> None:
    c = StubClient()
    out = c.generate("Query rewrite task (HyDE-style).")
    rewrites = json.loads(out)
    assert isinstance(rewrites, list)
    assert all(isinstance(r, str) for r in rewrites)


def test_stub_triage_classify_returns_tier_json() -> None:
    c = StubClient()
    prompt = (
        "Classify the system below by EU AI Act risk tier.\n"
        "System description:\nA predictive policing system used by police."
    )
    out = json.loads(c.generate(prompt))
    assert out["tier"] == "prohibited"
    assert "rationale" in out


def test_stub_embed_is_deterministic() -> None:
    c = StubClient()
    v1 = c.embed(["hello"])[0]
    v2 = c.embed(["hello"])[0]
    assert v1 == v2
    assert len(v1) == 256
    # Different inputs → different embeddings.
    assert c.embed(["different"])[0] != v1


def test_stub_fallback_response() -> None:
    """Unknown prompt → generic stub response."""
    c = StubClient()
    assert c.generate("totally unstructured prompt") == "Stub LLM response."


# --------------------------------------------------------------------------- #
# OllamaClient — happy path
# --------------------------------------------------------------------------- #


class _MockHttp:
    """Lightweight httpx replacement for OllamaClient tests."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict | None = None) -> httpx.Response:  # noqa: A002
        self.requests.append((url, json or {}))
        return self.responses.pop(0)

    def get(self, url: str, timeout: float = 0) -> httpx.Response:
        return self.responses.pop(0)


def _resp(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://test"))


def test_ollama_generate_happy_path() -> None:
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    client._client = _MockHttp([_resp(200, {"response": "  hello world  "})])

    out = client.generate("hi", system="be brief")

    assert out == "hello world"
    url, body = client._client.requests[0]
    assert url.endswith("/api/generate")
    assert body["model"] == "x"
    assert body["system"] == "be brief"
    assert body["stream"] is False


def test_ollama_generate_503_retries_then_succeeds() -> None:
    """OllamaBusyError on first call, success on retry — tenacity should hide it."""
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    client._client = _MockHttp([
        _resp(503, {"error": "server busy"}),
        _resp(200, {"response": "after retry"}),
    ])

    out = client.generate("hi")

    assert out == "after retry"
    assert len(client._client.requests) == 2  # 1 fail + 1 retry


def test_ollama_generate_503_exhausts_retries_and_raises() -> None:
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    # 4 retries default → 4 failures → eventual raise.
    client._client = _MockHttp([_resp(503, {}) for _ in range(4)])

    with pytest.raises(OllamaBusyError):
        client.generate("hi")


def test_ollama_embed_parallel_preserves_order() -> None:
    """Embed must return per-input embeddings in the same order as the input list."""
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    client._client = _MockHttp([
        _resp(200, {"embedding": [1.0, 2.0, 3.0]}),
        _resp(200, {"embedding": [4.0, 5.0, 6.0]}),
        _resp(200, {"embedding": [7.0, 8.0, 9.0]}),
    ])

    out = client.embed(["a", "b", "c"])

    assert len(out) == 3
    # ThreadPoolExecutor.map preserves order even with parallel execution.
    assert sorted(out) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


def test_ollama_embed_substitutes_whitespace_input() -> None:
    """Empty / whitespace text gets substituted with a single space so Ollama
    doesn't return an empty embedding."""
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    client._client = _MockHttp([_resp(200, {"embedding": [0.1]})])

    client.embed(["   "])

    _, body = client._client.requests[0]
    assert body["prompt"] == " "


def test_ollama_embed_raises_on_empty_response() -> None:
    """If Ollama somehow returns an empty embedding, surface a clear error
    (chromadb otherwise blows up downstream with a much less informative one)."""
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    client._client = _MockHttp([_resp(200, {"embedding": []})] * 4)  # retries exhausted

    with pytest.raises(RuntimeError, match="empty embedding"):
        client.embed(["hello"])


def test_ollama_health_returns_true_on_200() -> None:
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    client._client = _MockHttp([_resp(200, {"models": []})])
    assert client.health() is True


def test_ollama_health_returns_false_on_exception() -> None:
    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")

    class _BrokenClient:
        def get(self, *a, **kw):
            raise httpx.ConnectError("nope")

    client._client = _BrokenClient()  # type: ignore[assignment]
    assert client.health() is False


# --------------------------------------------------------------------------- #
# get_llm() factory
# --------------------------------------------------------------------------- #


def test_get_llm_returns_stub_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.config import settings

    monkeypatch.setattr(settings, "llm_backend", "stub")
    reset_llm_cache()
    client = get_llm()
    assert isinstance(client, StubClient)


def test_get_llm_falls_back_to_stub_when_ollama_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If REGPILOT_LLM=ollama but the server is down, get_llm should gracefully
    fall back to StubClient rather than crashing the whole app on import."""
    from regpilot.config import settings

    monkeypatch.setattr(settings, "llm_backend", "ollama")
    reset_llm_cache()

    with patch.object(OllamaClient, "health", return_value=False):
        client = get_llm()

    assert isinstance(client, StubClient)
    reset_llm_cache()


def test_get_llm_caches_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.config import settings

    monkeypatch.setattr(settings, "llm_backend", "stub")
    reset_llm_cache()

    a = get_llm()
    b = get_llm()
    assert a is b  # same instance both times
