"""Tests for the new LLM-first risk classifier with semantic Annex III matching.

These tests cover:

* Bright-line rule overrides (Article 5, GPAI Art. 51 systemic-risk threshold)
* LLM-driven verdict path with structured output (via StubClient)
* Graceful degradation when the LLM call fails
* Semantic-similarity helper cosine math
"""

from __future__ import annotations

import math

import pytest

from regpilot.llm import LLMClient
from regpilot.tools.risk_classifier import (
    ClassificationResult,
    RiskVerdict,
    _cosine,
    _is_systemic_gpai,
    _scan_article_5,
    classify,
    reset_semantic_cache,
)

# --------------------------------------------------------------------------- #
# Bright-line rules — Article 5 prohibited
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("social scoring of citizens", "5(1)(c)"),
        ("Public sector tool that scores citizens by behaviour.", "5(1)(c)"),
        ("Predictive policing tool flagging suspects.", "5(1)(d)"),
        ("Police will use AI to predict who will commit a crime", "5(1)(d)"),
        ("Scraping facial images from the web to build a recognition database", "5(1)(e)"),
        ("Emotion recognition in the workplace for productivity tracking", "5(1)(f)"),
    ],
)
def test_scan_article_5_catches_canonical_patterns(text: str, expected: str) -> None:
    hits = _scan_article_5(text)
    assert expected in hits


def test_classify_article_5_short_circuits_llm() -> None:
    v = classify({"system_purpose": "Social scoring tool for citizens."})
    assert v.tier == "prohibited"
    assert "5(1)(c)" in v.article_5_matches
    assert v.confidence == 1.0  # bright-line override


# --------------------------------------------------------------------------- #
# Bright-line rules — GPAI Article 51 systemic-risk threshold
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Frontier LLM with more than 10^25 FLOPs offered as an API.",
        "Our frontier model is served via REST.",
        "Systemic-risk LLM deployed at scale.",
    ],
)
def test_is_systemic_gpai_matches_threshold_markers(text: str) -> None:
    assert _is_systemic_gpai(text)


def test_classify_systemic_gpai_short_circuits_llm() -> None:
    v = classify(
        {"system_purpose": "Frontier LLM with more than 10^25 FLOPs offered as an API."}
    )
    assert v.tier == "general_purpose_systemic"
    assert v.confidence == 1.0
    assert "Article 51" in v.rationale


# --------------------------------------------------------------------------- #
# LLM-driven verdict path
# --------------------------------------------------------------------------- #


def test_classify_uses_llm_for_ambiguous_input() -> None:
    """Non-bright-line input flows through the LLM (stub here) for the verdict."""

    v = classify(
        {"system_purpose": "AI system that screens CVs for our hiring pipeline."}
    )
    assert v.tier == "high_risk"
    # Stub LLM populates annex areas via its keyword map; verify integration.
    assert v.annex_iii_matches
    assert any("Employment" in m for m in v.annex_iii_matches)


def test_classify_returns_limited_risk_for_chatbot() -> None:
    v = classify({"system_purpose": "A customer-support chatbot for our e-commerce site."})
    assert v.tier == "limited_risk"


def test_classify_returns_minimal_for_benign_system() -> None:
    v = classify({"system_purpose": "A spam filter for company email."})
    assert v.tier == "minimal_risk"


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #


def test_classify_falls_back_when_llm_raises() -> None:
    """If the LLM structured call crashes, fall back to a sensible default tier."""

    class _BrokenLLM(LLMClient):
        chat_model = "broken"
        embed_model = "broken"
        provider = "broken"

        def generate(self, *a, **kw):
            raise RuntimeError("network down")

        def generate_structured(self, *a, **kw):
            raise RuntimeError("network down")

        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    v = classify(
        {"system_purpose": "A spam filter for company email."},
        llm=_BrokenLLM(),
    )
    # No bright-line rule, no semantic hits (stub embeddings are useless),
    # and the LLM is broken → fallback should land somewhere sane.
    assert v.tier in ("minimal_risk", "limited_risk")
    assert v.confidence < 1.0


def test_classify_falls_back_to_high_risk_on_semantic_hits_when_llm_fails() -> None:
    """If the LLM fails BUT semantic similarity surfaced Annex III candidates,
    the fallback should respect those and emit high_risk."""

    class _SemanticHitLLM(LLMClient):
        chat_model = "x"
        embed_model = "x"
        provider = "x"

        def generate(self, *a, **kw):
            return ""

        def generate_structured(self, *a, **kw):
            raise RuntimeError("nope")

        def embed(self, texts):
            # Return identical vectors so cosine ≈ 1.0 for everything.
            return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

    reset_semantic_cache()
    v = classify(
        {"system_purpose": "An AI doing something high-risk."},
        llm=_SemanticHitLLM(),
    )
    reset_semantic_cache()
    assert v.tier == "high_risk"
    assert v.annex_iii_matches
    assert "Fallback verdict" in v.rationale


# --------------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------------- #


def test_cosine_identical_vectors_returns_one() -> None:
    assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_returns_zero() -> None:
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_handles_zero_vector() -> None:
    assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_handles_empty_or_mismatched() -> None:
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_known_value() -> None:
    """cos(60°) = 0.5 — a sanity check the math actually works."""

    v = _cosine([1.0, 0.0], [math.cos(math.pi / 3), math.sin(math.pi / 3)])
    assert v == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Tier coercion — LLM may emit aliases
# --------------------------------------------------------------------------- #


def test_classification_result_tier_coercion(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the LLM emits 'GPAI' / 'High' / 'Limited' the classifier should
    normalise these to the canonical tier names."""

    from regpilot.tools.risk_classifier import _coerce_tier

    assert _coerce_tier("GPAI") == "general_purpose"
    assert _coerce_tier("general-purpose AI") == "general_purpose"
    assert _coerce_tier("High") == "high_risk"
    assert _coerce_tier("limited") == "limited_risk"
    assert _coerce_tier("nonsense") == "unknown"


def test_classification_result_handles_malformed_llm_output() -> None:
    """If the LLM somehow returns tier=None or empty, _coerce_tier returns 'unknown'."""

    from regpilot.tools.risk_classifier import _coerce_tier

    assert _coerce_tier(None) == "unknown"
    assert _coerce_tier("") == "unknown"


# --------------------------------------------------------------------------- #
# Verdict shape
# --------------------------------------------------------------------------- #


def test_risk_verdict_is_a_dataclass_with_expected_fields() -> None:
    """Schema contract — downstream nodes read these fields."""

    v = RiskVerdict(
        tier="minimal_risk",
        rationale="test",
        annex_iii_matches=[],
        article_5_matches=[],
        confidence=0.5,
    )
    assert v.tier == "minimal_risk"
    assert v.confidence == 0.5


def test_classification_result_schema_has_required_fields() -> None:
    """ClassificationResult must expose tier + rationale + annex/art5 lists."""

    fields = ClassificationResult.model_fields
    assert {"tier", "rationale", "annex_iii_areas", "art_5_codes"} <= set(fields.keys())
