"""LLM-verdict layer of the risk classifier.

Owns the prompt template, the system message, and the tier-coercion
helper that normalises the model's raw output into the closed tier
vocabulary. The orchestration (when to call the LLM, when to short-
circuit on a bright-line rule, what to do on failure) lives in the
package ``__init__``.
"""

from __future__ import annotations

from typing import Any, cast

from regpilot.state import RISK_TIER_VOCABULARY, RiskTier

# Re-export under the local name the rest of the classifier package uses.
# Sourced from :data:`regpilot.state.RISK_TIER_VOCABULARY` so the tier
# membership-set tracks the ``RiskTier`` Literal automatically — no
# manual sync, no drift.
TIER_VOCABULARY: tuple[str, ...] = RISK_TIER_VOCABULARY


# Forgiving aliases — the LLM sometimes emits "GPAI" or "high" instead
# of the canonical form. We normalise on the way in so downstream nodes
# never see an unknown literal.
_TIER_ALIASES: dict[str, str] = {
    "gpai": "general_purpose",
    "general_purpose_ai": "general_purpose",
    "general_purpose_model": "general_purpose",
    "gpai_systemic": "general_purpose_systemic",
    "high": "high_risk",
    "limited": "limited_risk",
    "minimal": "minimal_risk",
}


def coerce_tier(raw: Any) -> RiskTier:
    """Normalise the LLM's raw tier output onto :data:`RiskTier`.

    Strips whitespace, lowercases, replaces hyphens / spaces with
    underscores. Falls through aliases. Returns ``"unknown"`` if nothing
    matches — downstream nodes treat unknown as the safest default.
    """

    val = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if val in TIER_VOCABULARY:
        return cast(RiskTier, val)
    if val in _TIER_ALIASES:
        return cast(RiskTier, _TIER_ALIASES[val])
    return "unknown"


CLASSIFY_SYSTEM = (
    "You are an EU AI Act compliance analyst. You classify AI systems against "
    "the six-step risk hierarchy of Regulation (EU) 2024/1689. You always "
    "respond with strict JSON matching the requested schema and you ground "
    "your verdict in the supplied Annex III candidates whenever possible."
)


CLASSIFY_PROMPT = """Classify the system below by EU AI Act risk tier.

Tier vocabulary:
- prohibited                 — Article 5 prohibited practices (social scoring, untargeted face scraping, predictive policing by profiling, real-time remote biometric ID in public for law enforcement, workplace/education emotion recognition, biometric categorisation of sensitive attributes, subliminal/manipulative techniques, exploiting vulnerabilities).
- high_risk                  — Annex III use cases OR Annex I product-safety components.
                                Annex III examples: biometric ID (including voice biometric auth, gait recognition); critical infrastructure (electricity grid, water supply, road traffic signals / lights, air traffic, railway signalling); education (admission ranking including PhD / kindergarten, exam proctoring, adaptive tutoring that evaluates learning outcomes); employment / worker management (CV screening, performance evaluation, gig-marketplace task allocation); essential services (credit scoring, loan / mortgage decisioning, insurance underwriting, real-time credit-card fraud blocking, welfare eligibility, emergency dispatch, kidney-transplant donor matching); law enforcement risk assessment; migration / border / visa decisioning; administration of justice — including legal-research tools used BY judges; medical-device or clinical-decision-support AI.
                                Annex I examples: automotive ADAS / pedestrian detection in cars; aviation safety; medical devices and IVD; machinery; toy safety; lift control.
- limited_risk               — Article 50 transparency duties only (chatbots that converse with humans, deepfakes, AI-generated synthetic content INCLUDING AI-drafted legal contracts / NDAs / marketing copy, voice assistants, emotion recognition outside workplace/education).
- minimal_risk               — everything else: spam filters; recommender systems (movies, ads, dating, e-commerce, music); consumer fitness apps; smart-home thermostats / vacuums; search ranking; INDUSTRIAL predictive maintenance; semiconductor / manufacturing defect inspection; agricultural drones; productivity tools.
- general_purpose            — Chapter V GPAI model (foundation model, LLM, NNb-parameter generative / code-completion / multimodal / translation model offered to downstream deployers via API).
- general_purpose_systemic   — GPAI model that ALSO meets the Article 51 systemic-risk threshold (≥10^25 FLOPs training compute, OR Commission designation, OR "frontier" model).

Common 3B-LLM traps — read these carefully:
- "Predictive maintenance" of industrial machinery is NOT predictive policing. Industrial reliability ML → minimal_risk.
- Consumer-facing recommender systems (movies, ads, products, dating matches) are NOT chatbots. They → minimal_risk, NOT limited_risk.
- Consumer fitness app or smart thermostat → minimal_risk. No Article 50 duty unless it actively converses or generates content.
- AI used BY end-users to GENERATE legal documents (contracts, NDAs) → limited_risk (Article 50 AI-generated content). Only AI used BY judges/courts to interpret facts or apply law → high_risk (administration of justice).
- Industrial defect inspection (semiconductor wafers, manufacturing QC) → minimal_risk, not Annex III.
- Agricultural drones doing crop monitoring → minimal_risk; they don't process people's biometric data.

Candidate Annex III areas the semantic matcher surfaced (descending similarity):
{candidates}

System description:
{description}

Return strict JSON with: tier, rationale (1-2 sentences), annex_iii_areas (list of names; empty unless tier is high_risk), art_5_codes (list of "5(1)(x)" codes; empty unless tier is prohibited).
"""
