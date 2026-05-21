"""Obligation mapper node.

Bridges the retrieved Articles + the structured intake into a concrete list of
obligations with concrete dates (via ``deadline_calculator_tool``).
"""

from __future__ import annotations

import logging
from typing import cast

from regpilot.state import RegPilotState, TraceEvent
from regpilot.tools.deadline_calculator import (
    DeadlineInfo,
    SystemType,
    UserRole,
    compute_deadlines,
    summarize_phase,
)

logger = logging.getLogger(__name__)


def obligation_mapper(state: RegPilotState) -> RegPilotState:
    tier = state.get("risk_tier", "minimal_risk")
    structured = state.get("structured", {})
    retrieved = state.get("retrieved", [])
    role = cast(UserRole, structured.get("user_role", "provider") or "provider")

    system_type = _tier_to_system_type(tier, structured)
    deadlines = compute_deadlines(system_type, role)

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


def _tier_to_system_type(tier: str, structured: dict) -> SystemType:
    notes = (structured.get("notes") or "").lower()
    purpose = (structured.get("system_purpose") or "").lower()
    if "gpai" in notes or "general-purpose" in notes or "general-purpose" in purpose:
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
