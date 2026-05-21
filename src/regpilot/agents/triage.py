"""Risk triage node — the conditional-router heart of the workflow.

Calls ``risk_classifier_tool`` and writes ``risk_tier`` + ``risk_rationale`` +
``annex_iii_matches`` + ``rag_query`` to state. The downstream conditional edge
in ``graph.py`` reads ``risk_tier`` to decide which branch to walk.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from regpilot.state import RegPilotState, TraceEvent
from regpilot.tools.risk_classifier import classify

logger = logging.getLogger(__name__)


def risk_triage(state: RegPilotState) -> RegPilotState:
    structured = state.get("structured") or {}
    raw_text = state.get("user_input", "") or ""
    verdict = classify(structured, raw_text=raw_text)

    rag_query = _build_rag_query(structured, verdict.tier)

    updates: RegPilotState = {
        "risk_tier": verdict.tier,
        "risk_rationale": verdict.rationale,
        "annex_iii_matches": verdict.annex_iii_matches,
        "rag_query": rag_query,
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="risk_triage",
                summary=f"classified as {verdict.tier} (conf={verdict.confidence:.2f})",
                payload={
                    "tier": verdict.tier,
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "annex_iii_matches": verdict.annex_iii_matches,
                    "article_5_matches": verdict.article_5_matches,
                },
            ),
        ],
    }
    return updates


def _build_rag_query(structured: Mapping[str, Any], tier: str) -> str:
    bits: list[str] = []
    if structured.get("system_purpose"):
        bits.append(str(structured["system_purpose"]))
    if structured.get("domain"):
        bits.append(f"domain: {structured['domain']}")
    bits.append(_obligation_keywords(tier))
    return " — ".join(bits)


# Tier-specific obligation keywords so BM25 / dense retrieval pull in the
# Articles the user actually has to comply with (not just the Annex match).
_OBLIGATION_KEYWORDS: dict[str, str] = {
    "high_risk": (
        "risk management system Article 9; data governance Article 10; "
        "technical documentation Article 11; record-keeping logs Article 12; "
        "transparency and information to deployers Article 13; "
        "human oversight Article 14; accuracy robustness cybersecurity Article 15; "
        "conformity assessment CE marking Article 43; "
        "EU database registration Article 49; "
        "post-market monitoring Article 72; serious incident reporting Article 73"
    ),
    "prohibited": (
        "Article 5 prohibited AI practices social scoring predictive policing "
        "biometric categorisation untargeted scraping emotion recognition workplace"
    ),
    "limited_risk": (
        "Article 50 transparency obligations chatbots deepfakes synthetic content labelling"
    ),
    "minimal_risk": (
        "Article 95 codes of conduct voluntary obligations general applicability"
    ),
    "unknown": "EU AI Act applicability and general obligations",
}


def _obligation_keywords(tier: str) -> str:
    return _OBLIGATION_KEYWORDS.get(tier, _OBLIGATION_KEYWORDS["unknown"])


def route_by_tier(state: RegPilotState) -> str:
    """Conditional edge function. LangGraph passes the current state in."""

    tier = state.get("risk_tier", "minimal_risk")
    return {
        "prohibited": "prohibited_path",
        "high_risk": "rag_retrieval",
        "limited_risk": "rag_retrieval",
        "minimal_risk": "rag_retrieval",
        "unknown": "rag_retrieval",
    }.get(tier, "rag_retrieval")
