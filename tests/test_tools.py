"""Unit tests for the three tools."""

from __future__ import annotations

from datetime import date

import pytest

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


# --------------------------------------------------------------------------- #
# deadline_calculator
# --------------------------------------------------------------------------- #


def test_prohibited_uses_phase_1_date() -> None:
    out = compute_deadlines("prohibited", "provider")
    assert out
    assert out[0].applies_from == PROHIBITIONS_APPLY


def test_annex_iii_obligations_apply_in_phase_3() -> None:
    out = compute_deadlines("annex_iii_high_risk", "provider")
    assert len(out) >= 7
    assert all(o.applies_from == GENERAL_APPLICATION for o in out)
    assert {o.article for o in out} >= {"Art. 9", "Art. 11", "Art. 43", "Art. 49", "Art. 72"}


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
