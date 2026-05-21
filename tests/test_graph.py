"""End-to-end tests for the main LangGraph workflow."""

from __future__ import annotations

import pytest

from regpilot.graph import run


@pytest.mark.parametrize(
    "description,expected_tier",
    [
        ("CV screening tool that ranks tech-recruiter applicants.", "high_risk"),
        ("Predictive policing system that flags repeat offenders.", "prohibited"),
        ("Customer support chatbot on our retail website.", "limited_risk"),
        ("Spam filter for company email.", "minimal_risk"),
    ],
)
def test_full_workflow_classifies_correctly(description: str, expected_tier: str) -> None:
    out = run(description)
    assert out.get("risk_tier") == expected_tier


def test_high_risk_produces_obligations_and_report() -> None:
    out = run("Credit scoring AI used by a Hungarian bank.")
    assert out.get("risk_tier") == "high_risk"
    assert out.get("obligations"), "expected obligations for a high-risk system"
    assert "Art." in (out.get("final_report") or "")


def test_prohibited_path_skips_retrieval() -> None:
    out = run("A social scoring system that rates citizens on trustworthiness.")
    assert out.get("risk_tier") == "prohibited"
    # Prohibited short-circuit does not run the RAG subgraph.
    assert out.get("retrieved", []) == []
    assert "PROHIBITED" in (out.get("final_report") or "")


def test_validator_loop_caps_out() -> None:
    """Even on a malformed branch the loop must terminate (no infinite recursion)."""

    out = run("A general productivity tool.")
    assert out.get("validator_loops", 0) <= 3


def test_trace_records_all_main_nodes() -> None:
    out = run("CV screening tool.")
    nodes = [ev["node"] for ev in out.get("trace", [])]
    assert "intake_classifier" in nodes
    assert "risk_triage" in nodes
    assert "rag_retrieval" in nodes
    assert "obligation_mapper" in nodes
    assert "compliance_synthesizer" in nodes
    assert "validator" in nodes
