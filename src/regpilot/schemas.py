"""Centralised Pydantic schemas for the LLM contract surface.

Every structured-output call the agents make goes through one of these
classes. Keeping them in one place — instead of co-located with the node
that consumes them — makes the LLM contract reviewable in a single read
and the schema-evolution story (rename a field, add a new tier, etc.)
mechanical to track.

The schemas are used by:

* :class:`IntakeSchema`         — ``regpilot.agents.intake.intake_classifier``
* :class:`ClassificationResult` — ``regpilot.tools.risk_classifier.classify``
* :class:`ReportSections`       — ``regpilot.agents.synthesizer.compliance_synthesizer``

The matching stub responses live in ``regpilot.llm.stub`` so unit tests
can exercise every node without a real LLM. If you add a new schema
here, also add a stub builder in ``llm/stub.py``'s dispatch table.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Intake — free text → structured intake record
# --------------------------------------------------------------------------- #


class IntakeSchema(BaseModel):
    """The structured intake the LLM extracts from the user's description."""

    system_purpose: str = Field(
        description="One-sentence summary of what the AI system does."
    )
    deployment_context: str = Field(
        default="",
        description="Where / how the system is deployed (e.g. EU market, internal-only).",
    )
    data_modalities: list[str] = Field(
        default_factory=list,
        description="Modalities the system processes (text, image, audio, video, biometric, tabular).",
    )
    user_role: Literal["provider", "deployer", "importer", "distributor", "unknown"] = (
        Field(
            default="unknown",
            description="The user's role in the AI value chain per the EU AI Act.",
        )
    )
    domain: str = Field(
        default="general",
        description="Short domain label (HR, healthcare, education, law enforcement, general, ...).",
    )
    notes: str = Field(
        default="",
        description="Other relevant facts (e.g. generative model, EU-only deployment, GPAI).",
    )


# --------------------------------------------------------------------------- #
# Classification — risk-tier verdict
# --------------------------------------------------------------------------- #


class ClassificationResult(BaseModel):
    """The risk-tier verdict the LLM produces in the classifier."""

    tier: str = Field(
        description=(
            "EU AI Act risk tier. One of: prohibited, high_risk, limited_risk, "
            "minimal_risk, general_purpose, general_purpose_systemic."
        )
    )
    rationale: str = Field(description="One- or two-sentence justification.")
    annex_iii_areas: list[str] = Field(
        default_factory=list,
        description=(
            "If tier is high_risk, list the Annex III area names that match "
            "(e.g. 'Employment, worker management, access to self-employment')."
        ),
    )
    art_5_codes: list[str] = Field(
        default_factory=list,
        description=(
            "If tier is prohibited, list the Article 5 sub-clauses that match "
            "(e.g. '5(1)(c)', '5(1)(d)')."
        ),
    )


# --------------------------------------------------------------------------- #
# Report sections — the LLM-narrative pieces of the compliance report
# --------------------------------------------------------------------------- #


class ReportSections(BaseModel):
    """The narrative sections the LLM is asked to write in the synthesizer.

    Everything else in the final report comes from the deterministic
    scaffold (obligations table, lifecycle mapping, frameworks alignment,
    evidence excerpts), so we never put Article numbers in the LLM's
    hands — its job is prose, not citations.
    """

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


__all__ = ["IntakeSchema", "ClassificationResult", "ReportSections"]
