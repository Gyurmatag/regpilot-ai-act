"""Unit tests for the three tools."""

from __future__ import annotations

from datetime import date

import pytest

from regpilot.agents.obligation_mapper import _is_annex_i, obligation_mapper
from regpilot.tools.citation_validator import reset_cache, validate
from regpilot.tools.deadline_calculator import (
    ANNEX_I_HIGH_RISK_APPLY,
    GENERAL_APPLICATION,
    PROHIBITIONS_APPLY,
    compute_deadlines,
    summarize_phase,
)
from regpilot.tools.risk_classifier import classify

# --------------------------------------------------------------------------- #
# risk_classifier
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "description,expected_tier",
    [
        ("A predictive policing system that flags suspects.", "prohibited"),
        ("A social scoring system for citizens.", "prohibited"),
        ("CV screening tool that ranks job applicants.", "high_risk"),
        ("Credit scoring system for personal loans.", "high_risk"),
        ("A chatbot for customer support.", "limited_risk"),
        ("A spam filter for company email.", "minimal_risk"),
    ],
)
def test_risk_classifier_tiers(description: str, expected_tier: str) -> None:
    v = classify({"system_purpose": description, "domain": "", "notes": ""})
    assert v.tier == expected_tier, f"{description!r} → {v.tier!r}"
    assert v.rationale, "rationale must not be empty"


def test_risk_classifier_returns_evidence_for_high_risk() -> None:
    v = classify(
        {
            "system_purpose": "AI system for grading student exams in a public school.",
            "domain": "education",
            "notes": "",
        }
    )
    assert v.tier == "high_risk"
    assert any("Education" in m or "education" in m.lower() for m in v.annex_iii_matches)


@pytest.mark.parametrize(
    "description",
    [
        # Verb-form variants that don't use the canonical "emotion recognition" noun.
        "AI that analyses customer emotions in real-time CCTV",
        "Customer-experience AI that analyses customer emotions during phone support",
        "An AI that detects employee mood from voice tone during sales calls",
        # Verb-form face / biometric variants — plurals were the original bug.
        "Video surveillance system that detects faces of visitors entering our building",
        "Security system that recognises individuals by their walking pattern",
        "Phone unlock feature using face detection on consumer devices",
    ],
)
def test_risk_classifier_catches_biometric_verb_forms(description: str) -> None:
    """Regression: an earlier classifier only matched the noun phrase
    "emotion recognition" / "face recognition" and missed verb-form
    descriptions like "analyses emotions" or "detects faces" — exactly how
    real users describe their systems. These must now hit Annex III Biometrics."""

    v = classify({"system_purpose": description, "domain": "", "notes": ""})
    assert v.tier == "high_risk", f"{description!r} → {v.tier!r}, expected high_risk"
    assert any("Biometrics" in m for m in v.annex_iii_matches)


@pytest.mark.parametrize(
    "description",
    [
        # Art 5(1)(c) social scoring — verb-form paraphrases of the regulatory text.
        "Public sector tool that scores citizens by behaviour.",
        "Government AI that rates residents based on social trustworthiness.",
        "Municipal scoring system that ranks households by their behaviour patterns.",
        "Public authority that scores individuals by loyalty and social conformity.",
    ],
)
def test_risk_classifier_catches_social_scoring_paraphrases(description: str) -> None:
    """Regression: keyword scan only matched "social scoring" / "social rating"
    and missed paraphrases like "scores citizens by behaviour" — exactly how
    real public-sector descriptions read. These must hit Art 5(1)(c)."""

    v = classify({"system_purpose": description, "domain": "", "notes": ""})
    assert v.tier == "prohibited", f"{description!r} → {v.tier!r}, expected prohibited"
    assert "5(1)(c)" in v.article_5_matches


@pytest.mark.parametrize(
    "description",
    [
        # Art 5(1)(h) — real-time remote biometric ID in public spaces by law enforcement.
        # The literal keyword list ("real-time remote biometric", "live facial
        # recognition police") missed every realistic phrasing of this; the
        # combo patterns below have to catch all of them.
        "A real-time facial recognition system used in train stations by police.",
        "Police use real-time facial recognition cameras to identify people in metro stations.",
        "Live face recognition deployed by law enforcement at airports.",
        "Real-time biometric identification system installed in public squares for police surveillance.",
        "Real-time facial recognition by the gendarmerie on public streets.",
    ],
)
def test_risk_classifier_catches_realtime_biometric_law_enforcement(description: str) -> None:
    """Regression: the showcase 'Police facial recognition' example fell through
    to ``high_risk`` (Biometrics Annex III area) because the literal Art 5(1)(h)
    keywords ("real-time remote biometric", "live facial recognition police")
    don't match the canonical phrasing reviewers actually type. The combo
    patterns added to ``bright_lines.py`` have to lift every variant in this
    parametrize block back to ``prohibited``."""

    v = classify({"system_purpose": description, "domain": "", "notes": ""})
    assert v.tier == "prohibited", f"{description!r} → {v.tier!r}, expected prohibited"
    assert "5(1)(h)" in v.article_5_matches


@pytest.mark.parametrize(
    "description",
    [
        # Negative controls — must NOT trip Art 5(1)(h). These are
        # private-context biometric authentication (high_risk via Annex III
        # Biometrics, not prohibited) or non-real-time forensic analysis.
        "Face recognition used to unlock employee laptops at our company.",
        "Forensic image analysis tool that matches a still photo against a database after the fact.",
    ],
)
def test_realtime_biometric_pattern_does_not_overreach(description: str) -> None:
    v = classify({"system_purpose": description, "domain": "", "notes": ""})
    assert v.tier != "prohibited", (
        f"{description!r} should NOT be prohibited (no law-enforcement + real-time + public-space combo)"
    )


@pytest.mark.parametrize(
    "description,expected_tier",
    [
        # The 7 showcase examples from ``src/regpilot/ui/app.py``. These
        # have to classify correctly on the stub backend so the "Try an
        # example" panel gives a first-time reviewer the documented
        # behaviour, not a sub-par fallback. Each phrase is the literal
        # string the sidebar button submits.
        ("An automated CV screening AI that ranks applicants for tech roles in Hungary.", "high_risk"),
        ("An AI used by police to predict, based on profiling, who will commit a crime.", "prohibited"),
        ("A customer support chatbot on our retail website handling refunds and FAQs.", "limited_risk"),
        ("An AI tool that grades student essays for a private high school.", "high_risk"),
        ("A spam filter that classifies inbound corporate email.", "minimal_risk"),
        ("A general-purpose generative AI assistant for marketing copy.", "general_purpose"),
        ("A real-time facial recognition system used in train stations by police.", "prohibited"),
    ],
)
def test_ui_showcase_examples_classify_correctly(description: str, expected_tier: str) -> None:
    """Regression: every "Try an example" button in the Streamlit sidebar must
    classify to its expected tier on the deterministic stub backend. The
    sidebar is the demo surface for first-time reviewers, so a mis-classification
    here is a credibility hit even when the README's stub-backend caveat
    explains why the eval set degrades."""

    v = classify({"system_purpose": description, "domain": "", "notes": ""})
    assert v.tier == expected_tier, f"{description!r} → {v.tier!r}, expected {expected_tier!r}"


@pytest.mark.parametrize(
    "description,expected_tier",
    [
        # GPAI sub-tier detection — frontier markers force systemic, others basic.
        ("Frontier LLM with more than 10^25 FLOPs offered as an API.", "general_purpose_systemic"),
        ("We host a foundation model accessible via REST API.", "general_purpose"),
        ("Our company runs a large language model service.", "general_purpose"),
        ("Systemic-risk GPAI model deployed across multiple verticals.", "general_purpose_systemic"),
        ("A general-purpose AI assistant for marketing copy.", "general_purpose"),
    ],
)
def test_risk_classifier_assigns_gpai_subtier(description: str, expected_tier: str) -> None:
    """GPAI patterns must surface as ``general_purpose`` / ``general_purpose_systemic``
    rather than falling through to ``minimal_risk`` or the LLM fallback."""

    v = classify({"system_purpose": description, "domain": "", "notes": ""})
    assert v.tier == expected_tier, f"{description!r} → {v.tier!r}, expected {expected_tier!r}"


# --------------------------------------------------------------------------- #
# deadline_calculator
# --------------------------------------------------------------------------- #


def test_prohibited_uses_phase_1_date() -> None:
    out = compute_deadlines("prohibited", "provider")
    assert out
    assert out[0].applies_from == PROHIBITIONS_APPLY


def test_annex_iii_obligations_apply_in_phase_3() -> None:
    out = compute_deadlines("annex_iii_high_risk", "provider")
    assert len(out) >= 10
    assert all(o.applies_from == GENERAL_APPLICATION for o in out)
    assert {o.article for o in out} >= {
        "Art. 9",
        "Art. 11",
        "Art. 13",
        "Art. 17",
        "Art. 43",
        "Art. 49",
        "Art. 72",
    }


def test_annex_i_uses_phase_4_date() -> None:
    out = compute_deadlines("annex_i_high_risk", "provider")
    assert out[0].applies_from == ANNEX_I_HIGH_RISK_APPLY


def test_deployer_adds_role_specific_obligations() -> None:
    provider = {o.article for o in compute_deadlines("annex_iii_high_risk", "provider")}
    deployer = {o.article for o in compute_deadlines("annex_iii_high_risk", "deployer")}
    assert "Art. 26" in deployer and "Art. 27" in deployer
    assert "Art. 26" not in provider


def test_summarize_phase_buckets() -> None:
    assert "in force" in summarize_phase(date(2024, 8, 1))
    assert "Phase 1" in summarize_phase(PROHIBITIONS_APPLY)
    assert "Phase 3" in summarize_phase(GENERAL_APPLICATION)
    assert "Phase 4" in summarize_phase(ANNEX_I_HIGH_RISK_APPLY)


# --------------------------------------------------------------------------- #
# citation_validator
# --------------------------------------------------------------------------- #


def test_citation_validator_accepts_real_articles() -> None:
    reset_cache()
    r = validate("Per Art. 5 and Art. 5(1)(a), this practice is prohibited.")
    assert r.ok, r.issues
    assert r.invalid_articles == set()


def test_citation_validator_flags_invalid_articles() -> None:
    reset_cache()
    r = validate("See Art. 999 for details.")
    assert not r.ok
    assert "999" in r.invalid_articles


def test_citation_validator_flags_missing_citations() -> None:
    reset_cache()
    r = validate("This report contains no citations whatsoever.")
    assert not r.ok
    assert any("No 'Art. N' citations" in i for i in r.issues)


# --------------------------------------------------------------------------- #
# obligation_mapper — Annex I detection override
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Pedestrian detection AI built into our production passenger cars; "
        "triggers automated emergency braking.",
        "Driver-assistance ADAS module for lane keeping on highways.",
        "AI-powered medical device for radiology lung nodule detection.",
        "Cockpit safety avionics module for commercial aircraft.",
        "Industrial machinery safety AI for press brake operation.",
    ],
)
def test_is_annex_i_recognises_product_safety_domains(text: str) -> None:
    """Bright-line domains that fall under AI Act Annex I (Sections A + B)
    map to product-safety regimes with a Phase-4 (2027-08-02) deadline,
    not the Phase-3 Annex III date."""

    assert _is_annex_i(text), f"expected {text!r} to match an Annex I pattern"


@pytest.mark.parametrize(
    "text",
    [
        "CV screening tool for hiring managers.",                 # Annex III, not Annex I
        "Credit-card fraud detection in real time.",              # Annex III essential services
        "Customer support chatbot for our website.",              # limited_risk
        "A spam filter for company email.",                       # minimal_risk
        "A foundation language model offered as an API.",         # GPAI
        # Air traffic control INFRASTRUCTURE is Annex III critical
        # infrastructure, not Annex I aviation product safety. The detector
        # must NOT false-positive here.
        "Air traffic control AI that sequences arrival slots at a major airport.",
        "Railway signalling controller for high-speed train control.",
    ],
)
def test_is_annex_i_does_not_false_positive_on_other_tiers(text: str) -> None:
    assert not _is_annex_i(text), f"expected {text!r} NOT to match Annex I"


def test_obligation_mapper_promotes_high_risk_adas_to_annex_i_deadline() -> None:
    """End-to-end: a high-risk ADAS / pedestrian-detection system should
    pick up the Phase-4 (2027-08-02) deadline via the Annex I override
    instead of the default Phase-3 (2026-08-02) date."""

    state = {
        "user_input": (
            "Real-time pedestrian detection AI built into our production "
            "passenger cars; if a pedestrian is detected on the trajectory "
            "it triggers automated emergency braking."
        ),
        "risk_tier": "high_risk",
        "structured": {
            "user_role": "provider",
            "system_purpose": "ADAS pedestrian detection",
        },
        "retrieved": [],
        "trace": [],
    }
    out = obligation_mapper(state)  # type: ignore[arg-type]
    assert out["deadlines"]["system_type"] == "annex_i_high_risk"
    assert all(item["date"] == ANNEX_I_HIGH_RISK_APPLY.isoformat()
               for item in out["deadlines"]["items"])


def test_obligation_mapper_keeps_annex_iii_for_non_product_high_risk() -> None:
    state = {
        "user_input": "CV screening tool that ranks job applicants.",
        "risk_tier": "high_risk",
        "structured": {
            "user_role": "provider",
            "system_purpose": "CV screening",
        },
        "retrieved": [],
        "trace": [],
    }
    out = obligation_mapper(state)  # type: ignore[arg-type]
    assert out["deadlines"]["system_type"] == "annex_iii_high_risk"
