"""Intake classifier node.

Parses free-text user input into a structured intake record. LLM-primary with
a deterministic heuristic fallback.

Default path:

1. LLM extracts the fields via ``generate_structured`` against the
   :class:`IntakeSchema` Pydantic model. Providers with native structured
   output (OpenAI ``response_format``, Ollama ``format=json``, Anthropic
   tool use) return schema-conformant JSON without regex post-processing.
2. If the LLM call raises (network failure, schema validation, model
   refusal), the regex/keyword heuristic kicks in so the chain never crashes.

Set ``REGPILOT_INTAKE_FAST=true`` to force the heuristic-only path (useful
on CPU-only Ollama where a 20-30 s intake call is unaffordable).
"""

from __future__ import annotations

import logging
import re
from typing import cast, get_args

from regpilot.config import settings
from regpilot.llm import LLMClient, get_llm
from regpilot.schemas import IntakeSchema
from regpilot.state import RegPilotState, StructuredIntake, TraceEvent, UserRole

logger = logging.getLogger(__name__)


# Membership-test set derived from the ``UserRole`` Literal so the heuristic
# can't accept a string that isn't in the type.
_VALID_ROLES: frozenset[str] = frozenset(get_args(UserRole))


_SYSTEM = (
    "You are a structured-data extractor for EU AI Act compliance intake. "
    "Given a natural-language description of an AI system, identify each "
    "field accurately. Always pick the most specific domain label that "
    "applies. Always populate data_modalities from the description; default "
    "to ['text'] if no modality is mentioned. Always emit valid JSON."
)


_PROMPT = """Extract structured fields from the AI system description.

Description:
{description}
"""


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


def intake_classifier(state: RegPilotState) -> RegPilotState:
    raw_input = state.get("user_input", "").strip()

    structured, mode = _extract(raw_input)

    if not structured.get("system_purpose"):
        structured["system_purpose"] = raw_input[:200] or "the described AI system"
    if not structured.get("user_role"):
        structured["user_role"] = "unknown"
    if not structured.get("domain"):
        structured["domain"] = "general"
    if not structured.get("data_modalities"):
        structured["data_modalities"] = ["text"]

    return {
        "structured": structured,
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="intake_classifier",
                summary=f"structured the user description (mode={mode})",
                payload={"structured": dict(structured), "mode": mode},
            ),
        ],
    }


def _extract(raw_input: str) -> tuple[StructuredIntake, str]:
    """Return (structured, mode-label). LLM-first with heuristic fallback."""

    if settings.intake_fast:
        return _heuristic(raw_input), "heuristic"

    llm: LLMClient = get_llm()
    try:
        result = llm.generate_structured(
            _PROMPT.format(description=raw_input),
            IntakeSchema,
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=400,
        )
        return _schema_to_intake(result), "llm"
    except Exception as exc:
        logger.warning("intake LLM failed: %s — falling back to heuristic", exc)
        return _heuristic(raw_input), "heuristic-fallback"


def _schema_to_intake(s: IntakeSchema) -> StructuredIntake:
    out: StructuredIntake = {
        "system_purpose": (s.system_purpose or "").strip(),
        "deployment_context": (s.deployment_context or "").strip(),
        "domain": (s.domain or "general").strip(),
        "notes": (s.notes or "").strip(),
        "user_role": s.user_role,
        "data_modalities": [
            m.strip().lower() for m in (s.data_modalities or []) if isinstance(m, str) and m.strip()
        ],
    }
    return out


# --------------------------------------------------------------------------- #
# Heuristic fallback (kept for offline / CPU-Ollama-overload safety)
# --------------------------------------------------------------------------- #


_DOMAIN_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(hir(e|ing)|recruit(er|ment|ing)?|cv|resume|applicant)\b", re.I), "HR / recruitment"),
    (re.compile(r"\b(credit|loan|mortgage|insurance|bank|underwrit)", re.I), "financial services"),
    (re.compile(r"\b(medical|hospital|patient|clinic|diagnos|radiolog)", re.I), "healthcare"),
    (re.compile(r"\b(student|exam|proctor|grad(e|ing)|school|universit|educat)", re.I), "education"),
    (re.compile(r"\b(police|policing|law\s+enforcement|criminal|prosecut)", re.I), "law enforcement"),
    (re.compile(r"\b(border|asylum|migrat|visa|customs)", re.I), "migration / border"),
    (re.compile(r"\b(power|electricity|gas|water|grid|infrastructure)", re.I), "critical infrastructure"),
    (re.compile(r"\b(ambulance|emergency|112|first[- ]response)", re.I), "essential services"),
    (re.compile(r"\b(chatbot|voice\s+assistant|customer\s+support|retail|marketing)", re.I), "consumer / marketing"),
    (re.compile(r"\b(spam|email|filter|recommend)", re.I), "productivity"),
]

_ROLE_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(we provide|we sell|provider|our company sells|we build|we offer)", re.I), "provider"),
    (re.compile(r"\b(we use|we deploy|deployer|in our|by our staff|at our company)", re.I), "deployer"),
    (re.compile(r"\b(import|distributor|reseller)", re.I), "importer"),
]

_MODALITY_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(image|photo|picture|video|footage|cctv|camera)", re.I), "image"),
    (re.compile(r"\b(audio|voice|speech|sound)", re.I), "audio"),
    (re.compile(r"\b(biometric|fingerprint|face|facial|iris)", re.I), "biometric"),
    (re.compile(r"\b(tabular|database|record|spreadsheet|csv)", re.I), "tabular"),
]


def _heuristic(text: str) -> StructuredIntake:
    """Pure-Python keyword extractor — no LLM, no I/O, sub-ms latency."""

    low = text.lower()

    domain = next((label for rx, label in _DOMAIN_KEYWORDS if rx.search(low)), "general")
    role = next((label for rx, label in _ROLE_KEYWORDS if rx.search(low)), "unknown")
    modalities = sorted({label for rx, label in _MODALITY_KEYWORDS if rx.search(low)})
    if not modalities:
        modalities = ["text"]

    safe_role: UserRole = cast(UserRole, role if role in _VALID_ROLES else "unknown")
    return {
        "system_purpose": text[:200] or "the described AI system",
        "deployment_context": "EU market" if "eu" in low or "europe" in low else "unspecified",
        "data_modalities": modalities,
        "user_role": safe_role,
        "domain": domain,
        "notes": "heuristic intake (fallback path)",
    }
