"""Coverage for the LLM-mode paths in intake / synthesizer / RAG subgraph.

The default ``REGPILOT_*_FAST=true`` runtime short-circuits these LLM calls
(production fast path). These tests flip each fast-path off so the LLM branch
also gets exercised — with the deterministic ``StubClient`` standing in for
Ollama so the suite stays offline.
"""

from __future__ import annotations

import pytest

from regpilot.agents.intake import intake_classifier
from regpilot.agents.synthesizer import compliance_synthesizer
from regpilot.config import settings


@pytest.fixture
def llm_mode_intake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "intake_fast", False)


@pytest.fixture
def llm_mode_synth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "synth_fast", False)


@pytest.fixture
def llm_mode_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rerank_fast", False)


# --------------------------------------------------------------------------- #
# Intake LLM path
# --------------------------------------------------------------------------- #


def test_intake_llm_path_parses_structured_output(llm_mode_intake: None) -> None:
    """With intake_fast=false, the node calls the LLM (stub) and parses the
    structured JSON response into a StructuredIntake dict."""

    state = {"user_input": "A CV screening AI for tech recruitment.", "trace": []}
    out = intake_classifier(state)

    structured = out["structured"]
    assert "CV screening AI" in structured["system_purpose"]
    assert structured["user_role"] == "provider"
    # The trace event should mark the LLM mode so the UI can show it.
    assert out["trace"][-1]["payload"]["mode"] == "llm"


def test_intake_llm_path_falls_back_on_parse_failure(
    llm_mode_intake: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the LLM returns garbage, intake falls back to the heuristic so the
    chain doesn't die mid-request."""

    from regpilot.agents import intake as intake_mod
    from regpilot.llm import reset_llm_cache

    class _BadLLM:
        def generate(self, *a, **kw): raise RuntimeError("oops")
        def embed(self, texts): return [[0.0]] * len(texts)

    monkeypatch.setattr(intake_mod, "get_llm", lambda: _BadLLM())
    state = {"user_input": "A spam filter.", "trace": []}
    out = intake_classifier(state)

    # Fell back to heuristic — still produced a structured output, did not crash.
    assert out["structured"]["system_purpose"]
    assert out["trace"][-1]["payload"]["mode"] == "heuristic-fallback"

    reset_llm_cache()


# --------------------------------------------------------------------------- #
# Synthesizer LLM path
# --------------------------------------------------------------------------- #


def test_synthesizer_llm_path_renders_report(llm_mode_synth: None) -> None:
    """With synth_fast=false, the synthesizer goes through the LLM call."""

    state = {
        "user_input": "CV screening tool.",
        "risk_tier": "high_risk",
        "risk_rationale": "Annex III Employment match",
        "structured": {
            "system_purpose": "CV screening tool",
            "deployment_context": "EU",
            "domain": "HR / recruitment",
            "user_role": "provider",
        },
        "obligations": [
            {"article": "Art. 9", "applies_from": "2026-08-02",
             "obligation": "risk management"},
        ],
        "retrieved": [],
        "trace": [],
    }
    out = compliance_synthesizer(state)

    assert out["draft_report"]
    assert out["trace"][-1]["payload"]["mode"] == "llm"


def test_synthesizer_llm_path_falls_back_on_failure(
    llm_mode_synth: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the LLM call raises, the synthesizer drops to the template instead
    of propagating the exception."""

    from regpilot.agents import synthesizer as synth_mod

    class _BadLLM:
        def generate(self, *a, **kw): raise RuntimeError("ollama is down")
        def embed(self, texts): return [[0.0]] * len(texts)

    monkeypatch.setattr(synth_mod, "get_llm", lambda: _BadLLM())
    state = {
        "user_input": "x",
        "risk_tier": "minimal_risk",
        "structured": {"system_purpose": "x"},
        "obligations": [],
        "retrieved": [],
        "trace": [],
    }
    out = compliance_synthesizer(state)

    assert "## Executive summary" in out["draft_report"]  # template fallback


# --------------------------------------------------------------------------- #
# Subgraph LLM rerank path
# --------------------------------------------------------------------------- #


def test_rag_subgraph_llm_rerank_path_runs(llm_mode_rerank: None) -> None:
    """With rerank_fast=false, the subgraph invokes the LLM reranker."""

    from regpilot.rag.subgraph import build_rag_subgraph

    sg = build_rag_subgraph()
    out = sg.invoke({"query": "CV screening tool for HR"})
    assert out.get("compressed")
