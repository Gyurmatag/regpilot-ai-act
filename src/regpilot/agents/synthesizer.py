"""Compliance synthesizer node.

Turns retrieved chunks + obligations + the risk verdict into a Markdown report.
The report must cite Articles in ``Art. N`` form so the validator can verify them.
"""

from __future__ import annotations

import logging

from regpilot.llm import LLMClient, get_llm
from regpilot.state import RegPilotState, TraceEvent

logger = logging.getLogger(__name__)


_SYSTEM = (
    "You are a senior compliance advisor drafting EU AI Act guidance. "
    "Be concise, structured, and cite Articles inline using the form 'Art. N' "
    "(e.g. 'Art. 9'). Never invent Article numbers — only cite Articles that "
    "appear in the supplied context."
)


_PROMPT = """Draft a compliance roadmap report. Output GitHub-flavoured Markdown only.

# Inputs

System: {purpose}
Deployment: {deployment}
Domain: {domain}
User role: {role}
Risk tier: {tier}
Triage rationale: {rationale}

# Confirmed obligations (from the deadline calculator)
{obligations}

# Retrieved Articles (use these as your source of truth)
{context}

# Required output structure
## Executive summary
(2-3 sentences)

## Risk classification
(one paragraph, cite Articles)

## Obligations & deadlines
(bullet list, one bullet per obligation, format: **YYYY-MM-DD — Art. N**: …)

## Recommended next steps
(3-5 numbered actions)

Remember: cite Articles inline as 'Art. N' so the validator can verify them.

draft report for tier {tier}
"""


def compliance_synthesizer(state: RegPilotState) -> RegPilotState:
    llm: LLMClient = get_llm()

    structured = state.get("structured", {})
    tier = state.get("risk_tier", "minimal_risk")
    obligations = state.get("obligations", [])
    retrieved = state.get("retrieved", [])

    obligations_md = (
        "\n".join(
            f"- {o['applies_from']} — {o['article']}: {o['obligation']}"
            for o in obligations
        )
        or "- (none — minimal risk)"
    )
    context_md = (
        "\n\n".join(
            f"[Art. {c.get('article') or '?'} p{c.get('paragraph') or '?'}] {c['text'][:600]}"
            for c in retrieved
        )
        or "(no relevant Articles retrieved)"
    )

    prompt = _PROMPT.format(
        purpose=structured.get("system_purpose", "n/a"),
        deployment=structured.get("deployment_context", "n/a"),
        domain=structured.get("domain", "n/a"),
        role=structured.get("user_role", "unknown"),
        tier=tier,
        rationale=state.get("risk_rationale", "n/a"),
        obligations=obligations_md,
        context=context_md,
    )

    try:
        draft = llm.generate(prompt, system=_SYSTEM, temperature=0.2, max_tokens=900)
    except Exception as exc:
        logger.warning("synthesizer LLM failed: %s — emitting template fallback", exc)
        draft = _fallback_report(structured, tier, obligations, retrieved)

    if not draft.strip():
        draft = _fallback_report(structured, tier, obligations, retrieved)

    return {
        "draft_report": draft.strip(),
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="compliance_synthesizer",
                summary=f"drafted report ({len(draft)} chars)",
                payload={"length": len(draft)},
            ),
        ],
    }


def _fallback_report(structured, tier, obligations, retrieved) -> str:
    cited = sorted({o["article"] for o in obligations})
    bullets = "\n".join(
        f"- **{o['applies_from']} — {o['article']}**: {o['obligation']}"
        for o in obligations
    )
    # Plain string (no leading indentation) so Streamlit's markdown parser
    # doesn't treat the block as a fenced code section.
    return (
        f"## Executive summary\n"
        f"The described system is classified as **{tier}** under the EU AI Act.\n\n"
        f"## Risk classification\n"
        f"Based on the intake, the system falls into the *{tier}* tier. "
        f"Relevant Articles: {', '.join(cited) or 'n/a'}.\n\n"
        f"## Obligations & deadlines\n"
        f"{bullets or '- No mandatory obligations.'}\n\n"
        f"## Recommended next steps\n"
        f"1. Confirm the classification with legal counsel.\n"
        f"2. Map obligations to internal owners and target dates.\n"
        f"3. Establish documentation per Annex IV (if high-risk).\n"
    )
