"""Deterministic scaffold for the compliance report.

Everything in this module is pure data + pure-Python rendering. The
:mod:`regpilot.agents.synthesizer` node imports the constants and helpers
here; both the LLM-narrative path and the template-fallback path render
through the same primitives, so the only difference between them is which
prose goes into the Executive summary and Risk classification sections.

The underscore prefix on the filename is the usual signal: this is an
implementation detail of the synthesizer, not a public agents API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# --------------------------------------------------------------------------- #
# Data — frozen regulatory content
# --------------------------------------------------------------------------- #


NEXT_STEPS: dict[str, list[str]] = {
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


TIER_LABEL: dict[str, str] = {
    "high_risk": "High risk",
    "limited_risk": "Limited risk",
    "minimal_risk": "Minimal risk",
    "prohibited": "Prohibited",
    "general_purpose": "General-purpose AI (GPAI)",
    "general_purpose_systemic": "GPAI with systemic risk",
    "unknown": "Unknown",
}


ROLE_NARRATIVE: dict[str, str] = {
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


LIFECYCLE: list[tuple[str, str]] = [
    ("Design & development", "Arts. 9, 10, 14, 15 (risk management, data governance, oversight, accuracy)"),
    ("Pre-market / before placing on the EU market", "Arts. 11, 13, 17, 43, 47, 48 (technical documentation, instructions, QMS, conformity assessment, declaration, CE marking)"),
    ("Market entry", "Arts. 49 (EU database registration) + 16 (provider obligations)"),
    ("In use / post-market", "Arts. 12, 26, 27, 72, 73 (logs, deployer oversight, FRIA, monitoring, serious-incident reporting)"),
]


# --------------------------------------------------------------------------- #
# Pure rendering helpers — input → markdown string
# --------------------------------------------------------------------------- #


def obligation_articles_set(obligations: Sequence[Mapping[str, Any]]) -> set[str]:
    """Bare Article numbers without the ``Art.`` prefix; used to filter
    retrieved chunks down to the obligation set."""

    return {str(o.get("article", "")).replace("Art. ", "") for o in obligations}


def cited_articles_str(obligations: Sequence[Mapping[str, Any]]) -> str:
    cited = sorted(obligation_articles_set(obligations))
    return ", ".join(f"Art. {a}" for a in cited) or "Art. 6"


def format_obligation_bullets(obligations: Sequence[Mapping[str, Any]]) -> str:
    if not obligations:
        return "- No mandatory obligations apply at this tier."
    return "\n".join(
        f"- **{o['applies_from']} — {o['article']}**: {o['obligation']}"
        for o in obligations
    )


def evidence_block(
    retrieved: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
    max_excerpts: int = 3,
    max_chars: int = 280,
) -> str:
    """Top-N highest-score chunks whose Article is in the obligation set,
    formatted as Markdown blockquotes."""

    obligation_arts = obligation_articles_set(obligations)
    relevant = [c for c in retrieved if (c.get("article") or "") in obligation_arts]
    sorted_evidence = sorted(
        relevant, key=lambda c: c.get("score") or 0.0, reverse=True
    )[:max_excerpts]
    if not sorted_evidence:
        return "_(no retrieved evidence — see the trace panel for the full chunk list.)_"
    return "\n\n".join(
        f"> **Art. {c.get('article') or '?'} (p{c.get('paragraph') or '?'})** — "
        f"{(c.get('text') or '').strip()[:max_chars]}…"
        for c in sorted_evidence
    )


def fria_flag(tier: str, role: str) -> str:
    """Article 27 FRIA call-out for high-risk deployers (or unspecified roles
    we treat as deployer-by-default for the flag)."""

    if tier == "high_risk" and role in ("deployer", "unknown"):
        return (
            "\n> ⚖️ **Article 27 FRIA trigger** — if you are a public-sector "
            "deployer (or a private body providing public services / "
            "essential private services), you must run a Fundamental Rights "
            "Impact Assessment before first use.\n"
        )
    return ""


def lifecycle_markdown(lifecycle: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"- **{phase}** — {arts}" for phase, arts in lifecycle)


def render_steps(steps: Sequence[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))


def standards_alignment_section() -> str:
    """The Aligned standards & frameworks section is identical for every
    report; emit it as a function so the calling sites don't drift."""

    return (
        "## Aligned standards & frameworks\n"
        "The obligations above also map onto recognised industry frameworks "
        "and standards your organisation may already use:\n\n"
        "- **ISO/IEC 42001:2023** — AI management systems (governance, risk, lifecycle).\n"
        "- **NIST AI Risk Management Framework (AI RMF 1.0)** — Govern, Map, Measure, Manage.\n"
        "- **ISO/IEC 23894:2023** — AI risk management guidance.\n"
        "- **CEN/CENELEC JTC 21** — harmonised AI Act standards being developed to support presumption of conformity.\n"
    )
