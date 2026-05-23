"""Compliance synthesizer node.

LLM-primary: the LLM writes the *narrative* sections (executive summary,
risk classification rationale, recommended next steps) via structured
output, and we stitch those into a deterministic scaffold for the
*factual* sections (obligations table with Article 113 dates, lifecycle
mapping, standards alignment, evidence excerpts).

This split keeps the report grounded — every cited Article number flows
from the deadline_calculator or the retrieved chunks, never from the LLM's
imagination. The LLM's job is prose, not citations.

Set ``REGPILOT_SYNTH_FAST=true`` to bypass the LLM entirely and use the
canned template for all sections (useful on CPU-only Ollama).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from regpilot.config import settings
from regpilot.llm import LLMClient, get_llm
from regpilot.state import RegPilotState, TraceEvent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LLM-driven narrative sections (structured output)
# --------------------------------------------------------------------------- #


class ReportSections(BaseModel):
    """Schema for the LLM-generated narrative parts of the compliance report."""

    executive_summary: str = Field(
        description=(
            "2-3 sentence executive summary of the compliance situation. "
            "Mention the system, the tier, and the highest-impact obligation."
        )
    )
    risk_classification_narrative: str = Field(
        description=(
            "One paragraph (3-5 sentences) explaining the tier choice, citing "
            "specific Articles inline as 'Art. N'. Ground every citation in the "
            "supplied Articles — never invent a number."
        )
    )
    recommended_next_steps: list[str] = Field(
        description=(
            "3-5 concrete next actions the user should take. Each step should "
            "be a single imperative sentence, citing the relevant Article."
        )
    )


_SYSTEM = (
    "You are a senior PwC compliance advisor drafting EU AI Act guidance. "
    "You always cite Articles inline as 'Art. N' (e.g. 'Art. 9'). You only "
    "cite Articles that appear in the supplied obligation list or retrieved "
    "Articles — never invent numbers. You write in plain professional English."
)


_PROMPT = """Draft the narrative sections of a compliance report.

System description: {purpose}
Deployment context: {deployment}
Domain: {domain}
User role: {role}
Risk tier (already decided): {tier_label}
Triage rationale (do not contradict): {rationale}

Confirmed obligations (each has a verified Article number — cite from this list):
{obligations}

Retrieved Article excerpts (use as evidence — cite from this list):
{context}

Return executive_summary, risk_classification_narrative, recommended_next_steps.
"""


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


def compliance_synthesizer(state: RegPilotState) -> RegPilotState:
    structured = state.get("structured", {}) or {}
    tier = state.get("risk_tier", "minimal_risk")
    obligations = state.get("obligations", []) or []
    retrieved = state.get("retrieved", []) or []
    rationale = state.get("risk_rationale", "n/a")

    # Fast-path: deterministic template (no LLM).
    if settings.synth_fast:
        draft = _template_report(structured, tier, obligations, retrieved)
        return _emit(state, draft, mode="template")

    # LLM-primary path: structured narrative + deterministic scaffold.
    try:
        llm: LLMClient = get_llm()
        sections = llm.generate_structured(
            _PROMPT.format(
                purpose=structured.get("system_purpose", "n/a"),
                deployment=structured.get("deployment_context", "n/a"),
                domain=structured.get("domain", "n/a"),
                role=structured.get("user_role", "unknown"),
                tier_label=_TIER_LABEL.get(tier, tier),
                rationale=rationale,
                obligations=_format_obligations(obligations),
                context=_format_context(retrieved, obligations),
            ),
            ReportSections,
            system=_SYSTEM,
            temperature=0.2,
            max_tokens=800,
        )
        draft = _stitch_report(sections, structured, tier, obligations, retrieved)
        return _emit(state, draft, mode="llm")
    except Exception as exc:
        logger.warning("synthesizer LLM failed: %s — using template fallback", exc)
        draft = _template_report(structured, tier, obligations, retrieved)
        return _emit(state, draft, mode="template-fallback")


def _emit(state: RegPilotState, draft: str, *, mode: str) -> RegPilotState:
    return {
        "draft_report": draft.strip(),
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="compliance_synthesizer",
                summary=f"drafted report ({len(draft)} chars, mode={mode})",
                payload={"length": len(draft), "mode": mode},
            ),
        ],
    }


def _format_obligations(obligations: Sequence[Mapping[str, Any]]) -> str:
    if not obligations:
        return "(none — minimal risk)"
    return "\n".join(
        f"- {o.get('applies_from','?')} — {o.get('article','?')}: {o.get('obligation','?')}"
        for o in obligations
    )


def _format_context(
    retrieved: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
) -> str:
    """Show only chunks whose Article appears in the obligation set, so the
    LLM can't accidentally cite an off-topic Article that happened to surface
    in retrieval."""

    obligation_arts = {str(o.get("article", "")).replace("Art. ", "") for o in obligations}
    relevant = [c for c in retrieved if (c.get("article") or "") in obligation_arts]
    if not relevant:
        return "(no retrieved Articles match the obligation set)"
    return "\n\n".join(
        f"[Art. {c.get('article') or '?'} p{c.get('paragraph') or '?'}] "
        f"{(c.get('text') or '').strip()[:500]}"
        for c in relevant[:6]
    )


# --------------------------------------------------------------------------- #
# Deterministic scaffold (used by both LLM path and template fallback)
# --------------------------------------------------------------------------- #


_NEXT_STEPS: dict[str, list[str]] = {
    "high_risk": [
        "Confirm the risk classification and applicable Annex III area with legal counsel.",
        "Map each obligation in the table to an internal owner and target date.",
        "Compile technical documentation per Annex IV (Art. 11) and prepare for the conformity assessment (Art. 43).",
        "Register the system in the EU database before placing it on the market (Art. 49).",
        "Establish post-market monitoring and a serious-incident reporting workflow (Arts. 72-73).",
    ],
    "limited_risk": [
        "Implement the Article 50 transparency disclosures in the user-facing flow.",
        "Label any AI-generated or AI-modified media (deepfakes, synthetic text).",
        "Track Article 50 implementing guidance from the AI Office as it is published.",
    ],
    "minimal_risk": [
        "No mandatory obligations apply, but adopt a voluntary code of conduct per Article 95.",
        "Re-check classification annually as the system evolves.",
        "Apply general data-protection and product-liability law as a baseline.",
    ],
    "prohibited": [
        "Do not place the system on the EU market or put it into service.",
        "Consult legal counsel on remediation, redesign, or withdrawal.",
        "Communicate the change to internal stakeholders and customers.",
    ],
    "general_purpose": [
        "Prepare Article 53 technical documentation (Annex XI) and the training-data summary template.",
        "Publish a copyright-compliance policy aligned with Art. 4(3) of the Copyright Directive.",
        "Designate an EU authorised representative if not established in the Union (Art. 54).",
        "Monitor whether your model crosses the Art. 51 systemic-risk threshold (10^25 FLOPs or Commission designation).",
    ],
    "general_purpose_systemic": [
        "Run pre-deployment model evaluations and adversarial testing per Art. 55(1)(a).",
        "Track and document systemic risks across the model lifecycle (Art. 55(1)(b)).",
        "Set up serious-incident reporting to the AI Office without undue delay (Art. 55(1)(c)).",
        "Ensure adequate cybersecurity protection of the model and physical infrastructure (Art. 55(1)(d)).",
        "Complete Art. 53 GPAI provider obligations (technical documentation, training summary, copyright policy).",
    ],
    "unknown": [
        "Re-classify with a more detailed system description.",
        "Engage legal counsel for a definitive interpretation.",
    ],
}


_TIER_LABEL: dict[str, str] = {
    "high_risk": "High risk",
    "limited_risk": "Limited risk",
    "minimal_risk": "Minimal risk",
    "prohibited": "Prohibited",
    "general_purpose": "General-purpose AI (GPAI)",
    "general_purpose_systemic": "GPAI with systemic risk",
    "unknown": "Unknown",
}


_ROLE_NARRATIVE: dict[str, str] = {
    "provider": (
        "As a **provider** you bear the primary compliance burden — design-time "
        "controls (Arts. 8-15), conformity assessment (Art. 43), registration "
        "(Art. 49), post-market monitoring (Art. 72)."
    ),
    "deployer": (
        "As a **deployer** you are responsible for using the system per the "
        "provider's instructions, assigning human oversight (Art. 26) and, where "
        "applicable, conducting a Fundamental Rights Impact Assessment (Art. 27)."
    ),
    "importer": (
        "As an **importer** you must verify the provider's CE marking, "
        "documentation and EU declaration of conformity before placing the "
        "system on the market (Art. 23)."
    ),
    "distributor": (
        "As a **distributor** you must check the CE marking and provider's "
        "instructions before further distributing the system (Art. 24)."
    ),
    "unknown": (
        "Your role in the AI value chain is not yet identified. The Act imposes "
        "different obligations on providers, deployers, importers and "
        "distributors — clarify before scoping compliance work."
    ),
}


_LIFECYCLE: list[tuple[str, str]] = [
    ("Design & development", "Arts. 9, 10, 14, 15 (risk management, data governance, oversight, accuracy)"),
    ("Pre-market / before placing on the EU market", "Arts. 11, 13, 17, 43, 47, 48 (technical documentation, instructions, QMS, conformity assessment, declaration, CE marking)"),
    ("Market entry", "Arts. 49 (EU database registration) + 16 (provider obligations)"),
    ("In use / post-market", "Arts. 12, 26, 27, 72, 73 (logs, deployer oversight, FRIA, monitoring, serious-incident reporting)"),
]


# --------------------------------------------------------------------------- #
# Stitch LLM sections into the deterministic scaffold
# --------------------------------------------------------------------------- #


def _stitch_report(
    sections: ReportSections,
    structured: Mapping[str, Any],
    tier: str,
    obligations: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
) -> str:
    tier_label = _TIER_LABEL.get(tier, tier)
    cited_articles = sorted({str(o["article"]).replace("Art. ", "") for o in obligations})
    cited_str = ", ".join(f"Art. {a}" for a in cited_articles) or "Art. 6"

    obligation_bullets = (
        "\n".join(
            f"- **{o['applies_from']} — {o['article']}**: {o['obligation']}"
            for o in obligations
        )
        or "- No mandatory obligations apply at this tier."
    )

    evidence_md = _evidence_block(retrieved, obligations)
    role = (structured.get("user_role") or "unknown").lower()
    role_narrative = _ROLE_NARRATIVE.get(role, _ROLE_NARRATIVE["unknown"])
    fria_flag = _fria_flag(tier, role)
    lifecycle_md = "\n".join(f"- **{phase}** — {arts}" for phase, arts in _LIFECYCLE)

    # LLM-supplied steps fall back to the canned list if the model returned an
    # empty array.
    steps = list(sections.recommended_next_steps) or _NEXT_STEPS.get(
        tier, _NEXT_STEPS["unknown"]
    )
    steps_md = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))

    domain = structured.get("domain") or "general"

    return (
        f"## Executive summary\n"
        f"{sections.executive_summary.strip()}\n\n"
        f"## Risk classification\n"
        f"{sections.risk_classification_narrative.strip()}\n\n"
        f"- **Tier**: {tier_label}\n"
        f"- **Domain**: {domain}\n"
        f"- **User role**: {role}\n"
        f"- **Applicable Articles**: {cited_str}\n\n"
        f"## Your role in the value chain\n"
        f"{role_narrative}\n"
        f"{fria_flag}\n"
        f"## Obligations & deadlines\n"
        f"{obligation_bullets}\n\n"
        f"## Lifecycle mapping\n"
        f"Where each obligation fits in the system lifecycle (use this to "
        f"plan compliance work alongside your existing SDLC / MLOps cadence):\n\n"
        f"{lifecycle_md}\n\n"
        f"## Evidence excerpts\n"
        f"{evidence_md}\n\n"
        f"## Recommended next steps\n"
        f"{steps_md}\n\n"
        f"## Aligned standards & frameworks\n"
        f"The obligations above also map onto recognised industry frameworks "
        f"and standards your organisation may already use:\n\n"
        f"- **ISO/IEC 42001:2023** — AI management systems (governance, risk, lifecycle).\n"
        f"- **NIST AI Risk Management Framework (AI RMF 1.0)** — Govern, Map, Measure, Manage.\n"
        f"- **ISO/IEC 23894:2023** — AI risk management guidance.\n"
        f"- **CEN/CENELEC JTC 21** — harmonised AI Act standards being developed to support presumption of conformity.\n"
    )


def _template_report(
    structured: Mapping[str, Any],
    tier: str,
    obligations: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
) -> str:
    """Pure-template report — used when LLM is disabled or fails."""

    tier_label = _TIER_LABEL.get(tier, tier)
    cited_articles = sorted({str(o["article"]).replace("Art. ", "") for o in obligations})
    cited_str = ", ".join(f"Art. {a}" for a in cited_articles) or "Art. 6"

    obligation_bullets = (
        "\n".join(
            f"- **{o['applies_from']} — {o['article']}**: {o['obligation']}"
            for o in obligations
        )
        or "- No mandatory obligations apply at this tier."
    )

    evidence_md = _evidence_block(retrieved, obligations)
    purpose = structured.get("system_purpose") or "the described AI system"
    domain = structured.get("domain") or "general"
    role = (structured.get("user_role") or "unknown").lower()
    role_narrative = _ROLE_NARRATIVE.get(role, _ROLE_NARRATIVE["unknown"])
    fria_flag = _fria_flag(tier, role)
    lifecycle_md = "\n".join(f"- **{phase}** — {arts}" for phase, arts in _LIFECYCLE)
    steps_md = "\n".join(
        f"{i}. {s}" for i, s in enumerate(_NEXT_STEPS.get(tier, _NEXT_STEPS["unknown"]), start=1)
    )

    return (
        f"## Executive summary\n"
        f"**{purpose}** is classified as **{tier_label}** under the EU AI Act "
        f"(Regulation (EU) 2024/1689). This roadmap lists the "
        f"{len(obligations)} concrete obligation(s) that apply, along with "
        f"their Article 113 phased deadlines, and maps them onto the system "
        f"lifecycle.\n\n"
        f"## Risk classification\n"
        f"- **Tier**: {tier_label}\n"
        f"- **Domain**: {domain}\n"
        f"- **User role**: {role}\n"
        f"- **Applicable Articles**: {cited_str}\n\n"
        f"## Your role in the value chain\n"
        f"{role_narrative}\n"
        f"{fria_flag}\n"
        f"## Obligations & deadlines\n"
        f"{obligation_bullets}\n\n"
        f"## Lifecycle mapping\n"
        f"Where each obligation fits in the system lifecycle (use this to "
        f"plan compliance work alongside your existing SDLC / MLOps cadence):\n\n"
        f"{lifecycle_md}\n\n"
        f"## Evidence excerpts\n"
        f"{evidence_md}\n\n"
        f"## Recommended next steps\n"
        f"{steps_md}\n\n"
        f"## Aligned standards & frameworks\n"
        f"The obligations above also map onto recognised industry frameworks "
        f"and standards your organisation may already use:\n\n"
        f"- **ISO/IEC 42001:2023** — AI management systems (governance, risk, lifecycle).\n"
        f"- **NIST AI Risk Management Framework (AI RMF 1.0)** — Govern, Map, Measure, Manage.\n"
        f"- **ISO/IEC 23894:2023** — AI risk management guidance.\n"
        f"- **CEN/CENELEC JTC 21** — harmonised AI Act standards being developed to support presumption of conformity.\n"
    )


def _evidence_block(
    retrieved: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
) -> str:
    obligation_arts = {str(o["article"]).replace("Art. ", "") for o in obligations}
    relevant = [c for c in retrieved if (c.get("article") or "") in obligation_arts]
    sorted_evidence = sorted(
        relevant, key=lambda c: c.get("score") or 0.0, reverse=True
    )[:3]
    return (
        "\n\n".join(
            f"> **Art. {c.get('article') or '?'} (p{c.get('paragraph') or '?'})** — "
            f"{(c.get('text') or '').strip()[:280]}…"
            for c in sorted_evidence
        )
        or "_(no retrieved evidence — see the trace panel for the full chunk list.)_"
    )


def _fria_flag(tier: str, role: str) -> str:
    if tier == "high_risk" and role in ("deployer", "unknown"):
        return (
            "\n> ⚖️ **Article 27 FRIA trigger** — if you are a public-sector "
            "deployer (or a private body providing public services / "
            "essential private services), you must run a Fundamental Rights "
            "Impact Assessment before first use.\n"
        )
    return ""
