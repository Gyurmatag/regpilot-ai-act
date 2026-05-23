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
    rag_queries = _build_multi_queries(structured, verdict.tier, raw_text)
    priority = list(_TIER_PRIORITY_ARTICLES.get(verdict.tier, ()))

    updates: RegPilotState = {
        "risk_tier": verdict.tier,
        "risk_rationale": verdict.rationale,
        "annex_iii_matches": verdict.annex_iii_matches,
        "rag_query": rag_query,
        "rag_queries": rag_queries,
        "priority_articles": priority,
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
                    "rag_queries": rag_queries,
                    "priority_articles": priority,
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
    "general_purpose": (
        "Article 53 GPAI provider obligations technical documentation training summary copyright; "
        "Article 54 cooperation AI Office authorised representative"
    ),
    "general_purpose_systemic": (
        "Article 51 systemic risk classification GPAI model; "
        "Article 53 GPAI provider obligations technical documentation training summary; "
        "Article 54 cooperation AI Office; "
        "Article 55 systemic risk obligations model evaluation adversarial testing "
        "serious incident reporting cybersecurity protection"
    ),
    "unknown": "EU AI Act applicability and general obligations",
}


def _obligation_keywords(tier: str) -> str:
    return _OBLIGATION_KEYWORDS.get(tier, _OBLIGATION_KEYWORDS["unknown"])


# Targeted sub-queries per tier — fired in parallel by the RAG subgraph so the
# retriever has a fair shot at every obligation Article, not just the ones
# that lexically overlap the user description.
_TIER_SUBQUERIES: dict[str, tuple[str, ...]] = {
    "high_risk": (
        "Article 9 risk management system across the lifecycle",
        "Article 10 data governance training validation test sets",
        "Article 11 technical documentation Annex IV",
        "Article 13 transparency information to deployers instructions for use",
        "Article 14 human oversight measures",
        "Article 15 accuracy robustness cybersecurity",
        "Article 17 quality management system provider",
        "Article 18 record keeping documentation retention",
        "Article 43 conformity assessment CE marking declaration",
        "Article 49 EU database registration high-risk",
        "Article 72 post-market monitoring serious incident reporting",
    ),
    "prohibited": (
        "Article 5 prohibited AI practices",
        "Article 113 entry into force phased application",
    ),
    "limited_risk": (
        "Article 50 transparency obligations chatbots",
        "Article 50 synthetic content deepfake labelling",
    ),
    "minimal_risk": (
        "Article 95 voluntary codes of conduct",
    ),
    "general_purpose": (
        "Article 53 GPAI provider obligations technical documentation",
        "Article 53 training data summary copyright policy",
        "Article 54 cooperation AI Office authorised representative",
    ),
    "general_purpose_systemic": (
        "Article 51 systemic risk classification 10^25 FLOPs threshold",
        "Article 53 GPAI provider obligations technical documentation Annex XI",
        "Article 53 training data summary copyright policy",
        "Article 54 cooperation AI Office",
        "Article 55 systemic risk model evaluation adversarial testing",
        "Article 55 serious incident reporting AI Office cybersecurity",
    ),
}


# Per-tier obligation Articles surfaced from the deadline_calculator. The
# retriever uses these to boost matching chunks in the fused candidate list.
_TIER_PRIORITY_ARTICLES: dict[str, tuple[str, ...]] = {
    "high_risk": ("9", "10", "11", "12", "13", "14", "15", "17", "18", "43", "49", "72"),
    "prohibited": ("5", "113"),
    "limited_risk": ("50",),
    "minimal_risk": ("95",),
    "general_purpose": ("53", "54"),
    "general_purpose_systemic": ("51", "53", "54", "55"),
    "unknown": (),
}


def _build_multi_queries(
    structured: Mapping[str, Any], tier: str, raw_text: str
) -> list[str]:
    """List of targeted sub-queries fed to the RAG subgraph as ``rewritten_queries``.

    Always starts with the user-grounded query (so semantic match on the actual
    system description has a chance), followed by **every** tier-specific
    obligation sub-query — capping here would silently lose coverage of the
    Articles defined later in the tier's obligation list (Art. 43, 49, 72, …).
    """

    base = raw_text or str(structured.get("system_purpose", "")) or "EU AI Act"
    subs = _TIER_SUBQUERIES.get(tier, ())
    seen: list[str] = []
    for q in [base, *subs]:
        if q and q not in seen:
            seen.append(q)
    return seen


def route_by_tier(state: RegPilotState) -> str:
    """Conditional edge function. LangGraph passes the current state in."""

    tier = state.get("risk_tier", "minimal_risk")
    return {
        "prohibited": "prohibited_path",
        "high_risk": "rag_retrieval",
        "limited_risk": "rag_retrieval",
        "minimal_risk": "rag_retrieval",
        "general_purpose": "rag_retrieval",
        "general_purpose_systemic": "rag_retrieval",
        "unknown": "rag_retrieval",
    }.get(tier, "rag_retrieval")
