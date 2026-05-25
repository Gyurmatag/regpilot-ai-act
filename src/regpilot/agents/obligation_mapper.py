"""Obligation mapper node.

Bridges the retrieved Articles + the structured intake into a concrete list of
obligations with concrete dates (via ``deadline_calculator_tool``).

Two responsibilities on top of the deadline lookup:

1. **Tier → system_type mapping.** GPAI sub-tiers route to
   ``general_purpose_ai``; everything else maps 1:1.
2. **Annex I detection.** The classifier only emits the Annex III flavour of
   high-risk; AI baked into a regulated product (automotive type-approval,
   medical devices, machinery directive, toy safety, etc.) is high-risk via
   Annex I and lands on the Phase-4 (2 Aug 2027) deadline, not Phase-3.
   We sniff the user's description for the well-known product domains and
   override the system_type accordingly.
"""

from __future__ import annotations

import logging
import re
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


# Product-safety Annex I detection. Word-boundary regex over the major
# sectoral product-safety regimes covered by Annex I of the AI Act — Section A
# (machinery, toys, recreational craft, lifts, ATEX, radio equipment, pressure
# equipment, gas appliances, medical devices, IVD) + Section B (civil aviation
# security, two-/three-wheel vehicles, ag/forestry vehicles, marine
# equipment, rail, motor vehicles + trailers, civil aviation).
_ANNEX_I_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Automotive / road vehicles — pedestrian detection, ADAS, autonomous driving.
    re.compile(
        r"\b(automotive|in[- ]?vehicle|passenger\s+car|adas|autonomous\s+(driving|vehicle)"
        r"|driver[- ]?assistance|emergency\s+braking|lane[- ]?keep|self[- ]?driving"
        r"|automated\s+driving)\b",
        re.I,
    ),
    # Medical devices / IVD.
    re.compile(
        r"\b(medical\s+device|implantable|in[- ]?vitro\s+diagnostic|ivd"
        r"|radiograph|radiology\s+ai|clinical\s+decision\s+support\s+software"
        r"|cdss|ce[- ]?marked\s+(software|device))\b",
        re.I,
    ),
    # Aviation / rail / marine — only on-board safety AI counts as Annex I.
    # Air traffic control infrastructure is Annex III critical infrastructure,
    # NOT Annex I aviation product safety, so we explicitly exclude it.
    re.compile(
        r"\b(aircraft\s+safety|aviation\s+safety|cockpit\s+(safety|avionics)"
        r"|in[- ]?flight\s+ai|onboard\s+avionics"
        r"|marine\s+navigation)\b",
        re.I,
    ),
    # Machinery directive / toys / lifts.
    re.compile(
        r"\b(industrial\s+machinery\s+safety|machinery\s+directive|toy\s+safety"
        r"|lift\s+control|pressure\s+equipment|gas\s+appliance|atex|radio\s+equipment)\b",
        re.I,
    ),
)


def _is_annex_i(text: str) -> bool:
    """True if the description points at a product covered by AI Act Annex I.

    Used as an override on top of the classifier's ``high_risk`` verdict so
    Annex I systems pick up the correct Phase-4 (2027-08-02) deadline
    instead of the default Annex III Phase-3 date.
    """

    return any(p.search(text) for p in _ANNEX_I_PATTERNS)


def obligation_mapper(state: RegPilotState) -> RegPilotState:
    tier = state.get("risk_tier", "minimal_risk")
    structured = state.get("structured") or {}
    retrieved = state.get("retrieved") or []
    role = cast(UserRole, structured.get("user_role", "provider") or "provider")

    system_type = _tier_to_system_type(tier)
    # Annex I override: high-risk product-safety AI gets Phase-4 (2027-08-02)
    # deadlines, not the default Phase-3 (2026-08-02). The classifier doesn't
    # surface the Annex I / Annex III distinction; we infer it here from the
    # raw user input + structured intake.
    if system_type == "annex_iii_high_risk":
        text_for_sniff = " ".join(
            filter(
                None,
                [
                    state.get("user_input", ""),
                    str(structured.get("system_purpose", "") or ""),
                    str(structured.get("deployment_context", "") or ""),
                    str(structured.get("notes", "") or ""),
                ],
            )
        )
        if _is_annex_i(text_for_sniff):
            system_type = "annex_i_high_risk"
            logger.info(
                "Annex I override: detected product-safety AI; deadline → 2027-08-02"
            )

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


_TIER_TO_SYSTEM_TYPE: dict[str, SystemType] = {
    "prohibited": "prohibited",
    "high_risk": "annex_iii_high_risk",
    "limited_risk": "limited_risk",
    "minimal_risk": "minimal_risk",
    "unknown": "minimal_risk",
}


def _tier_to_system_type(tier: str) -> SystemType:
    """Map the classifier's tier vocabulary onto the deadline calculator's
    system-type vocabulary. Pure 1:1 lookup; GPAI sub-tiers collapse onto
    the single ``general_purpose_ai`` system type."""

    if tier in ("general_purpose", "general_purpose_systemic"):
        return "general_purpose_ai"
    return _TIER_TO_SYSTEM_TYPE.get(tier, "minimal_risk")


def _fmt_deadline(info: DeadlineInfo) -> str:
    return f"{info.applies_from.isoformat()} — {info.article}: {info.obligation}"
