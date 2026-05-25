"""Compliance synthesizer node.

LLM-primary path: the LLM writes the narrative sections (executive summary,
risk-classification rationale, next steps) via structured output and we
stitch them into a deterministic scaffold for the factual sections
(obligations table with Article 113 dates, lifecycle mapping, standards
alignment, evidence excerpts).

The split keeps the report grounded — every cited Article number flows
from the deadline calculator or the retrieved chunks, never from the LLM.
The LLM's job is prose, not citations.

Set ``REGPILOT_SYNTH_FAST=true`` to bypass the LLM entirely and render the
canned template for all sections (useful on CPU-only Ollama).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from regpilot.agents._synth_scaffold import (
    LIFECYCLE,
    NEXT_STEPS,
    ROLE_NARRATIVE,
    TIER_LABEL,
    cited_articles_str,
    evidence_block,
    format_obligation_bullets,
    fria_flag,
    lifecycle_markdown,
    obligation_articles_set,
    render_steps,
    standards_alignment_section,
)
from regpilot.config import settings
from regpilot.llm import LLMClient, get_llm
from regpilot.schemas import ReportSections
from regpilot.state import RegPilotState, TraceEvent

logger = logging.getLogger(__name__)


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

    if settings.synth_fast:
        draft = _template_report(structured, tier, obligations, retrieved)
        return _emit(state, draft, mode="template")

    try:
        llm: LLMClient = get_llm()
        sections = llm.generate_structured(
            _PROMPT.format(
                purpose=structured.get("system_purpose", "n/a"),
                deployment=structured.get("deployment_context", "n/a"),
                domain=structured.get("domain", "n/a"),
                role=structured.get("user_role", "unknown"),
                tier_label=TIER_LABEL.get(tier, tier),
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


# --------------------------------------------------------------------------- #
# Prompt formatters (LLM input)
# --------------------------------------------------------------------------- #


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
    """Show only chunks whose Article appears in the obligation set.

    Stops the LLM accidentally citing an off-topic Article that happened to
    surface in retrieval but isn't in the obligation list.
    """

    obligation_arts = obligation_articles_set(obligations)
    relevant = [c for c in retrieved if (c.get("article") or "") in obligation_arts]
    if not relevant:
        return "(no retrieved Articles match the obligation set)"
    return "\n\n".join(
        f"[Art. {c.get('article') or '?'} p{c.get('paragraph') or '?'}] "
        f"{(c.get('text') or '').strip()[:500]}"
        for c in relevant[:6]
    )


# --------------------------------------------------------------------------- #
# Report renderers — both share the deterministic scaffold below
# --------------------------------------------------------------------------- #


def _stitch_report(
    sections: ReportSections,
    structured: Mapping[str, Any],
    tier: str,
    obligations: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
) -> str:
    """LLM-narrative + deterministic scaffold."""

    steps = list(sections.recommended_next_steps) or NEXT_STEPS.get(
        tier, NEXT_STEPS["unknown"]
    )
    return _render_report(
        structured=structured,
        tier=tier,
        obligations=obligations,
        retrieved=retrieved,
        executive_summary_md=sections.executive_summary.strip(),
        classification_md=sections.risk_classification_narrative.strip(),
        steps=steps,
    )


def _template_report(
    structured: Mapping[str, Any],
    tier: str,
    obligations: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
) -> str:
    """Pure-template report — fallback when the LLM is disabled or fails."""

    purpose = structured.get("system_purpose") or "the described AI system"
    tier_label = TIER_LABEL.get(tier, tier)
    executive_summary_md = (
        f"**{purpose}** is classified as **{tier_label}** under the EU AI Act "
        f"(Regulation (EU) 2024/1689). This roadmap lists the "
        f"{len(obligations)} concrete obligation(s) that apply, along with "
        f"their Article 113 phased deadlines, and maps them onto the system "
        f"lifecycle."
    )
    return _render_report(
        structured=structured,
        tier=tier,
        obligations=obligations,
        retrieved=retrieved,
        executive_summary_md=executive_summary_md,
        classification_md=None,  # render the classification metadata block instead
        steps=NEXT_STEPS.get(tier, NEXT_STEPS["unknown"]),
    )


def _render_report(
    *,
    structured: Mapping[str, Any],
    tier: str,
    obligations: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
    executive_summary_md: str,
    classification_md: str | None,
    steps: Sequence[str],
) -> str:
    """Assemble the final report from a few inputs and the shared scaffold.

    The two callers (LLM path, template fallback) supply different
    executive-summary and classification text; everything else — the
    metadata block, role narrative, FRIA flag, obligations table, lifecycle
    mapping, evidence excerpts, next-steps list, standards alignment —
    comes from the deterministic scaffold.
    """

    tier_label = TIER_LABEL.get(tier, tier)
    role = (structured.get("user_role") or "unknown").lower()
    role_narrative = ROLE_NARRATIVE.get(role, ROLE_NARRATIVE["unknown"])
    domain = structured.get("domain") or "general"
    cited_str = cited_articles_str(obligations)

    classification_block = (
        f"{classification_md.strip()}\n\n"
        if classification_md
        else ""
    )
    metadata_block = (
        f"- **Tier**: {tier_label}\n"
        f"- **Domain**: {domain}\n"
        f"- **User role**: {role}\n"
        f"- **Applicable Articles**: {cited_str}\n"
    )

    return (
        f"## Executive summary\n"
        f"{executive_summary_md.strip()}\n\n"
        f"## Risk classification\n"
        f"{classification_block}"
        f"{metadata_block}\n"
        f"## Your role in the value chain\n"
        f"{role_narrative}\n"
        f"{fria_flag(tier, role)}\n"
        f"## Obligations & deadlines\n"
        f"{format_obligation_bullets(obligations)}\n\n"
        f"## Lifecycle mapping\n"
        f"Where each obligation fits in the system lifecycle (use this to "
        f"plan compliance work alongside your existing SDLC / MLOps cadence):\n\n"
        f"{lifecycle_markdown(LIFECYCLE)}\n\n"
        f"## Evidence excerpts\n"
        f"{evidence_block(retrieved, obligations)}\n\n"
        f"## Recommended next steps\n"
        f"{render_steps(steps)}\n\n"
        f"{standards_alignment_section()}"
    )
