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


# --------------------------------------------------------------------------- #
# Structured output — StubClient schema-aware paths
# --------------------------------------------------------------------------- #


def test_stub_generate_structured_classification_extracts_tier() -> None:
    from regpilot.tools.risk_classifier import ClassificationResult

    c = StubClient()
    prompt = (
        "Classify the system below by EU AI Act risk tier.\n"
        "System description: An AI system that screens CVs for our hiring pipeline."
    )
    result = c.generate_structured(prompt, ClassificationResult)
    assert result.tier == "high_risk"
    assert "Employment" in (result.annex_iii_areas or [""])[0]


def test_stub_generate_structured_intake_extracts_fields() -> None:
    from regpilot.agents.intake import IntakeSchema

    c = StubClient()
    prompt = "Extract structured fields from the AI system description.\nDescription:\nA CV screening AI for tech recruitment."
    result = c.generate_structured(prompt, IntakeSchema)
    assert "CV screening" in result.system_purpose
    assert result.domain == "HR / recruitment"
    assert result.user_role == "provider"


def test_stub_generate_structured_report_sections_emits_narrative() -> None:
    from regpilot.agents.synthesizer import ReportSections

    c = StubClient()
    prompt = (
        "Draft the narrative sections.\n"
        "Risk tier (already decided): high_risk\n"
        "Confirmed obligations:\n"
        "- 2026-08-02 — Art. 9: risk management\n"
        "- 2026-08-02 — Art. 10: data governance"
    )
    result = c.generate_structured(prompt, ReportSections)
    assert "Art. 9" in result.executive_summary or "Art. 10" in result.executive_summary
    assert len(result.recommended_next_steps) >= 1


def test_stub_generate_structured_falls_back_for_unknown_schema() -> None:
    """Unknown schemas → stub falls back to generate() + best-effort JSON parse."""

    from pydantic import BaseModel

    class _Foo(BaseModel):
        x: int = 0

    c = StubClient()
    result = c.generate_structured("anything", _Foo)
    assert isinstance(result, _Foo)


# --------------------------------------------------------------------------- #
# OllamaClient — generate_structured uses format=json
# --------------------------------------------------------------------------- #


def test_ollama_generate_structured_sends_full_json_schema() -> None:
    """Ollama 0.5+ supports JSON-schema-constrained generation. The client
    must send the full schema as the ``format`` parameter (not the legacy
    ``"json"`` string) so the model is grammar-constrained to a
    schema-conformant output."""

    from regpilot.agents.intake import IntakeSchema

    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    body = {
        "system_purpose": "A CV screening AI.",
        "deployment_context": "EU",
        "data_modalities": ["text"],
        "user_role": "provider",
        "domain": "HR",
        "notes": "test",
    }
    client._client = _MockHttp([_resp(200, {"response": json.dumps(body)})])

    result = client.generate_structured("describe", IntakeSchema)

    _, payload = client._client.requests[0]
    fmt = payload["format"]
    assert isinstance(fmt, dict), f"expected dict (JSON schema), got {type(fmt).__name__}"
    assert fmt["type"] == "object"
    assert "system_purpose" in fmt["properties"]
    assert result.system_purpose == "A CV screening AI."


def test_ollama_generate_structured_raises_on_invalid_json() -> None:
    from regpilot.agents.intake import IntakeSchema
    from regpilot.llm import StructuredOutputError

    client = OllamaClient(base_url="http://test", chat_model="x", embed_model="y")
    client._client = _MockHttp([_resp(200, {"response": "not-json-at-all"})])

    with pytest.raises(StructuredOutputError):
        client.generate_structured("x", IntakeSchema)


# --------------------------------------------------------------------------- #
# OpenAIClient — happy + structured + missing key
# --------------------------------------------------------------------------- #


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.config import settings
    from regpilot.llm import OpenAIClient

    monkeypatch.setattr(settings, "openai_api_key", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_openai_generate_calls_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAIClient.generate delegates to the chat completions API with the
    correct messages payload."""

    from regpilot.llm import OpenAIClient

    captured: dict = {}

    class _Resp:
        class _Choice:
            class _Msg:
                content = "  hi from openai  "

            message = _Msg()

        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kw):
            self.chat = _Chat()

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    client = OpenAIClient(api_key="sk-fake")

    out = client.generate("hello", system="be brief", temperature=0.5)
    assert out == "hi from openai"
    assert captured["model"] == client.chat_model
    assert captured["messages"][0] == {"role": "system", "content": "be brief"}
    assert captured["messages"][1] == {"role": "user", "content": "hello"}
    assert captured["temperature"] == 0.5


def test_openai_embed_uses_native_batching(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.llm import OpenAIClient

    captured: dict = {}

    class _Datum:
        def __init__(self, vec):
            self.embedding = vec

    class _Resp:
        data = [_Datum([1.0, 2.0]), _Datum([3.0, 4.0])]

    class _Embeddings:
        def create(self, **kw):
            captured.update(kw)
            return _Resp()

    class _FakeOpenAI:
        def __init__(self, **kw):
            self.embeddings = _Embeddings()

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    client = OpenAIClient(api_key="sk-fake")

    out = client.embed(["a", "b"])
    assert out == [[1.0, 2.0], [3.0, 4.0]]
    assert captured["input"] == ["a", "b"]


def test_openai_generate_structured_uses_native_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI client should use the native ``beta.chat.completions.parse``
    structured-output API rather than falling back to JSON-prompt parsing."""

    from regpilot.llm import OpenAIClient
    from regpilot.tools.risk_classifier import ClassificationResult

    captured: dict = {}
    expected = ClassificationResult(
        tier="high_risk",
        rationale="CV screening",
        annex_iii_areas=["Employment, worker management, access to self-employment"],
        art_5_codes=[],
    )

    class _Resp:
        class _Choice:
            class _Msg:
                parsed = expected

            message = _Msg()

        choices = [_Choice()]

    class _BetaCompletions:
        def parse(self, **kw):
            captured.update(kw)
            return _Resp()

    class _BetaChat:
        completions = _BetaCompletions()

    class _Beta:
        chat = _BetaChat()

    class _FakeOpenAI:
        def __init__(self, **kw):
            self.beta = _Beta()

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    client = OpenAIClient(api_key="sk-fake")

    out = client.generate_structured("classify this", ClassificationResult)
    assert out.tier == "high_risk"
    assert captured["response_format"] is ClassificationResult


# --------------------------------------------------------------------------- #
# AnthropicClient — happy + structured + no-embedding
# --------------------------------------------------------------------------- #


def test_anthropic_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.config import settings
    from regpilot.llm import AnthropicClient

    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_anthropic_generate_calls_messages_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.llm import AnthropicClient

    captured: dict = {}

    class _Block:
        text = "  hi from claude  "

    class _Resp:
        content = [_Block()]

    class _Messages:
        def create(self, **kw):
            captured.update(kw)
            return _Resp()

    class _FakeAnthropic:
        def __init__(self, **kw):
            self.messages = _Messages()

    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
    client = AnthropicClient(api_key="sk-ant-fake")

    out = client.generate("hello", system="be brief")
    assert out == "hi from claude"
    assert captured["model"] == client.chat_model
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["system"] == "be brief"


def test_anthropic_generate_structured_uses_tool_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic must use the tool_use API for guaranteed schema-conformant output."""

    from regpilot.llm import AnthropicClient
    from regpilot.tools.risk_classifier import ClassificationResult

    captured: dict = {}

    class _ToolUseBlock:
        type = "tool_use"
        input = {
            "tier": "prohibited",
            "rationale": "social scoring",
            "annex_iii_areas": [],
            "art_5_codes": ["5(1)(c)"],
        }

    class _Resp:
        content = [_ToolUseBlock()]

    class _Messages:
        def create(self, **kw):
            captured.update(kw)
            return _Resp()

    class _FakeAnthropic:
        def __init__(self, **kw):
            self.messages = _Messages()

    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
    client = AnthropicClient(api_key="sk-ant-fake")

    out = client.generate_structured("classify", ClassificationResult)
    assert out.tier == "prohibited"
    assert "5(1)(c)" in out.art_5_codes
    # tool_choice must force the tool so the model has no choice but to fill it.
    assert captured["tool_choice"]["type"] == "tool"
    assert len(captured["tools"]) == 1


def test_anthropic_embed_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.llm import AnthropicClient

    class _FakeAnthropic:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
    client = AnthropicClient(api_key="sk-ant-fake")

    with pytest.raises(NotImplementedError, match="no embedding"):
        client.embed(["hello"])


# --------------------------------------------------------------------------- #
# Composite client (anthropic chat + ollama embed)
# --------------------------------------------------------------------------- #


def test_composite_client_routes_calls_correctly() -> None:
    from regpilot.llm import _CompositeClient

    class _MockChat:
        chat_model = "mock-chat"
        embed_model = "n/a"
        provider = "mock"

        def __init__(self):
            self.gen_calls = 0

        def generate(self, *a, **kw):
            self.gen_calls += 1
            return "chat response"

        def generate_structured(self, *a, **kw):
            return "structured response"

        def embed(self, texts):
            raise NotImplementedError

    class _MockEmb:
        chat_model = "n/a"
        embed_model = "mock-embed"
        provider = "mock"

        def __init__(self):
            self.emb_calls = 0

        def generate(self, *a, **kw):
            raise NotImplementedError

        def generate_structured(self, *a, **kw):
            raise NotImplementedError

        def embed(self, texts):
            self.emb_calls += 1
            return [[0.1] * 8 for _ in texts]

    chat, emb = _MockChat(), _MockEmb()
    composite = _CompositeClient(chat, emb)  # type: ignore[arg-type]

    assert composite.generate("hi") == "chat response"
    assert composite.embed(["a", "b"]) == [[0.1] * 8, [0.1] * 8]
    assert chat.gen_calls == 1
    assert emb.emb_calls == 1
    assert composite.chat_model == "mock-chat"
    assert composite.embed_model == "mock-embed"


# --------------------------------------------------------------------------- #
# Factory — provider selection
# --------------------------------------------------------------------------- #


def test_get_llm_picks_openai_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.config import settings
    from regpilot.llm import OpenAIClient

    monkeypatch.setattr(settings, "llm_backend", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-fake")
    reset_llm_cache()

    class _FakeOpenAI:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    client = get_llm()
    assert isinstance(client, OpenAIClient)
    reset_llm_cache()


def test_get_llm_falls_back_to_ollama_when_openai_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI selected but init blows up → factory falls back to Ollama, then stub."""
    from regpilot.config import settings

    monkeypatch.setattr(settings, "llm_backend", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "")  # forces RuntimeError
    reset_llm_cache()

    with patch.object(OllamaClient, "health", return_value=False):
        client = get_llm()
    assert isinstance(client, StubClient)
    reset_llm_cache()
