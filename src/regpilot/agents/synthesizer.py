"""Compliance synthesizer node.

Turns retrieved chunks + obligations + the risk verdict into a Markdown report.
The report must cite Articles in ``Art. N`` form so the validator can verify them.

There are two paths:

* **Fast (default, ``REGPILOT_SYNTH_FAST=true``)** — deterministic template that
  composes obligations + retrieved evidence + tier-specific next steps. No LLM
  call, no timeouts, returns in <50 ms. This is the production default because
  the obligations are already deterministic per tier; the LLM was adding flair,
  not correctness.
* **LLM** — qwen2.5:3b-instruct (or whatever Ollama serves). Set
  ``REGPILOT_SYNTH_FAST=false`` to opt in. Used to be the default; on CPU it
  routinely needed 60–120 s and timed out under load.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from regpilot.config import settings
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
    structured = state.get("structured", {}) or {}
    tier = state.get("risk_tier", "minimal_risk")
    obligations = state.get("obligations", []) or []
    retrieved = state.get("retrieved", []) or []

    # Fast path — deterministic template, no LLM call. This is the production
    # default; obligations and citations are already correct via the
    # deadline_calculator + RAG retrieval, the LLM was only adding flavour
    # text that cost 60–120 s on CPU.
    if settings.synth_fast:
        draft = _template_report(structured, tier, obligations, retrieved)
        return _emit(state, draft, mode="template")

    # LLM path — opt-in via REGPILOT_SYNTH_FAST=false.
    llm: LLMClient = get_llm()
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
        draft = _template_report(structured, tier, obligations, retrieved)

    if not draft.strip():
        draft = _template_report(structured, tier, obligations, retrieved)

    return _emit(state, draft, mode="llm")


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
# Fast-path template
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
    "unknown": "Unknown",
}


# Maps internal-tool roles to a human sentence describing what the role
# specifically owes under the Act. Used in the role narrative block.
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


# Lifecycle phase → which Articles apply during that phase. Helps clients map
# compliance work onto their existing SDLC / MLOps cadence.
_LIFECYCLE: list[tuple[str, str]] = [
    ("Design & development", "Arts. 9, 10, 14, 15 (risk management, data governance, oversight, accuracy)"),
    ("Pre-market / before placing on the EU market", "Arts. 11, 13, 17, 43, 47, 48 (technical documentation, instructions, QMS, conformity assessment, declaration, CE marking)"),
    ("Market entry", "Arts. 49 (EU database registration) + 16 (provider obligations)"),
    ("In use / post-market", "Arts. 12, 26, 27, 72, 73 (logs, deployer oversight, FRIA, monitoring, serious-incident reporting)"),
]


def _template_report(
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

    # Evidence excerpts — filter to chunks whose Article is in the obligation
    # set, so the report only quotes Articles it already cites. This keeps
    # citation precision tight without sacrificing reader trust (every quote
    # is anchored to an obligation in the table above).
    obligation_arts = {str(o["article"]).replace("Art. ", "") for o in obligations}
    relevant = [c for c in retrieved if (c.get("article") or "") in obligation_arts]
    sorted_evidence = sorted(
        relevant, key=lambda c: c.get("score") or 0.0, reverse=True
    )[:3]
    evidence_md = (
        "\n\n".join(
            f"> **Art. {c.get('article') or '?'} (p{c.get('paragraph') or '?'})** — "
            f"{(c.get('text') or '').strip()[:280]}…"
            for c in sorted_evidence
        )
        or "_(no retrieved evidence — see the trace panel for the full chunk list.)_"
    )

    steps = _NEXT_STEPS.get(tier, _NEXT_STEPS["unknown"])
    steps_md = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))

    purpose = structured.get("system_purpose") or "the described AI system"
    domain = structured.get("domain") or "general"
    role = (structured.get("user_role") or "unknown").lower()
    role_narrative = _ROLE_NARRATIVE.get(role, _ROLE_NARRATIVE["unknown"])

    # Fundamental Rights Impact Assessment trigger (Art. 27): high-risk Annex III
    # systems deployed by public-sector bodies or providing essential services.
    # We surface it as a flag for any high-risk deployer — the user / lawyer
    # confirms applicability.
    fria_flag = ""
    if tier == "high_risk" and role in ("deployer", "unknown"):
        fria_flag = (
            "\n> ⚖️ **Article 27 FRIA trigger** — if you are a public-sector "
            "deployer (or a private body providing public services / "
            "essential private services), you must run a Fundamental Rights "
            "Impact Assessment before first use.\n"
        )

    lifecycle_md = "\n".join(
        f"- **{phase}** — {arts}" for phase, arts in _LIFECYCLE
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


