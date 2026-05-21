"""Risk triage node — the conditional-router heart of the workflow.

Calls ``risk_classifier_tool`` and writes ``risk_tier`` + ``risk_rationale`` +
``annex_iii_matches`` + ``rag_query`` to state. The downstream conditional edge
in ``graph.py`` reads ``risk_tier`` to decide which branch to walk.
"""

from __future__ import annotations

import logging

from regpilot.state import RegPilotState, TraceEvent
from regpilot.tools.risk_classifier import classify

logger = logging.getLogger(__name__)


def risk_triage(state: RegPilotState) -> RegPilotState:
    structured = state.get("structured", {})
    verdict = classify(structured)

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


def _build_rag_query(structured: dict, tier: str) -> str:
    bits: list[str] = []
    if structured.get("system_purpose"):
        bits.append(str(structured["system_purpose"]))
    if structured.get("domain"):
        bits.append(f"domain: {structured['domain']}")
    if tier == "high_risk":
        bits.append("EU AI Act obligations for high-risk AI systems")
    elif tier == "prohibited":
        bits.append("Article 5 prohibited AI practices")
    elif tier == "limited_risk":
        bits.append("Article 50 transparency obligations")
    else:
        bits.append("EU AI Act applicability and general obligations")
    return " — ".join(bits)


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
