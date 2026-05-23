"""Obligation mapper node.

Bridges the retrieved Articles + the structured intake into a concrete list of
obligations with concrete dates (via ``deadline_calculator_tool``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, cast

from regpilot.state import RegPilotState, TraceEvent
from regpilot.tools.deadline_calculator import (
    DeadlineInfo,
    SystemType,
    UserRole,
    compute_deadlines,
    summarize_phase,
)

logger = logging.getLogger(__name__)


# Lexical hints that a system is a General-Purpose AI model under Chapter V
# of the AI Act (Articles 51-55). Verb-form / common-shorthand vocabulary —
# users rarely type the literal phrase "general-purpose AI".
_GPAI_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(gpai|general[\s\-_]?purpose(\s+ai)?)\b", re.I),
    re.compile(r"\b(foundation|frontier)\s+(model|llm|ai)\b", re.I),
    re.compile(r"\b(large\s+language\s+model|llm)s?\b", re.I),
    re.compile(r"\b10\s*\^?\s*25\s*flops?\b", re.I),
    re.compile(r"\bsystemic[\s\-]risk\s+(model|ai|llm)\b", re.I),
)


def obligation_mapper(state: RegPilotState) -> RegPilotState:
    tier = state.get("risk_tier", "minimal_risk")
    structured = state.get("structured") or {}
    retrieved = state.get("retrieved") or []
    role = cast(UserRole, structured.get("user_role", "provider") or "provider")

    system_type = _tier_to_system_type(tier, structured)
    systemic_risk = tier == "general_purpose_systemic"
    deadlines = compute_deadlines(system_type, role, systemic_risk=systemic_risk)

    cited_articles = {d["article"] for d in retrieved if d.get("article")} | {
        info.article.replace("Art. ", "") for info in deadlines
    }

    obligations = [
        {
            "article": info.article,
            "obligation": info.obligation,
            "applies_from": info.applies_from.isoformat(),
            "phase": summarize_phase(info.applies_from),
            "note": info.note,
        }
        for info in deadlines
    ]

    updates: RegPilotState = {
        "obligations": obligations,
        "deadlines": {
            "system_type": system_type,
            "user_role": role,
            "items": [
                {"article": info.article, "date": info.applies_from.isoformat()}
                for info in deadlines
            ],
        },
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="obligation_mapper",
                summary=f"mapped {len(obligations)} obligations (system_type={system_type})",
                payload={
                    "system_type": system_type,
                    "user_role": role,
                    "n_obligations": len(obligations),
                    "cited_articles": sorted(cited_articles),
                },
            ),
        ],
    }
    return updates


def _tier_to_system_type(tier: str, structured: Mapping[str, Any]) -> SystemType:
    # Authoritative GPAI tier wins immediately (risk classifier did the detection).
    if tier in ("general_purpose", "general_purpose_systemic"):
        return "general_purpose_ai"

    # Legacy fallback for cases where the classifier didn't flag GPAI but the
    # intake corpus clearly describes one (kept as a safety net).
    corpus = " ".join(
        str(structured.get(k, "") or "")
        for k in ("system_purpose", "deployment_context", "domain", "notes")
    )
    if any(pat.search(corpus) for pat in _GPAI_PATTERNS):
        return "general_purpose_ai"

    return {
        "prohibited": "prohibited",
        "high_risk": "annex_iii_high_risk",
        "limited_risk": "limited_risk",
        "minimal_risk": "minimal_risk",
        "unknown": "minimal_risk",
    }.get(tier, "minimal_risk")  # type: ignore[return-value]


def _fmt_deadline(info: DeadlineInfo) -> str:
    return f"{info.applies_from.isoformat()} — {info.article}: {info.obligation}"
