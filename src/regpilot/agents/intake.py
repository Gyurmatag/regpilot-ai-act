"""Intake classifier node.

Parses free-text user input into a ``StructuredIntake`` record so downstream
nodes don't have to keep re-reading the same blob.

Two paths:

* **Fast (default, ``REGPILOT_INTAKE_FAST=true``)** — keyword/regex extraction
  for ``domain``, ``user_role``, and ``data_modalities``. Returns in <10 ms.
  Used by default because the rule-based ``risk_classifier_tool`` runs against
  the raw user input anyway, so the structured intake is mostly informational.
* **LLM** — qwen2.5:3b-instruct via Ollama. Set ``REGPILOT_INTAKE_FAST=false``
  to opt in. Costs ~20–30 s per request on CPU.
"""

from __future__ import annotations

import json
import logging
import re

from regpilot.config import settings
from regpilot.llm import LLMClient, get_llm
from regpilot.state import RegPilotState, StructuredIntake, TraceEvent

logger = logging.getLogger(__name__)


_SYSTEM = (
    "You are a structured-data extractor. Given a natural-language description "
    "of an AI system, extract the requested fields as STRICT JSON. No commentary."
)

_PROMPT = """Extract the following fields from the description and reply with STRICT JSON only.

Schema:
{{
  "system_purpose":      "one-sentence summary of what the system does",
  "deployment_context":  "where/how it is deployed",
  "data_modalities":     ["text" | "image" | "audio" | "video" | "biometric" | "tabular"],
  "user_role":           "provider" | "deployer" | "importer" | "distributor" | "unknown",
  "domain":              "short label, e.g. HR, healthcare, education, law enforcement, general",
  "notes":               "any other relevant detail (e.g. EU market, generative)"
}}

intake_classifier — Description:
{description}
"""


def intake_classifier(state: RegPilotState) -> RegPilotState:
    raw_input = state.get("user_input", "").strip()

    if settings.intake_fast:
        structured = _heuristic(raw_input)
        mode = "heuristic"
    else:
        llm: LLMClient = get_llm()
        try:
            raw = llm.generate(
                _PROMPT.format(description=raw_input),
                system=_SYSTEM,
                temperature=0.0,
                max_tokens=300,
            )
            structured = _parse(raw)
            mode = "llm"
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("intake LLM failed: %s — falling back to heuristics", exc)
            structured = _heuristic(raw_input)
            mode = "heuristic-fallback"

    if not structured.get("system_purpose"):
        structured["system_purpose"] = raw_input[:200]
    if not structured.get("user_role"):
        structured["user_role"] = "unknown"
    if not structured.get("domain"):
        structured["domain"] = "general"

    return {
        "structured": structured,
        "trace": _append_trace(
            state,
            TraceEvent(
                node="intake_classifier",
                summary=f"structured the user description (mode={mode})",
                payload={"structured": dict(structured), "mode": mode},
            ),
        ),
    }


def _parse(raw: str) -> StructuredIntake:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out: StructuredIntake = {}
    for k in ("system_purpose", "deployment_context", "domain", "notes"):
        v = obj.get(k)
        if isinstance(v, str):
            out[k] = v
    if isinstance(obj.get("data_modalities"), list):
        out["data_modalities"] = [str(x) for x in obj["data_modalities"] if isinstance(x, str)]
    role = obj.get("user_role")
    if role in ("provider", "deployer", "importer", "distributor", "unknown"):
        out["user_role"] = role  # type: ignore[assignment]
    return out


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
    """Pure-Python field extractor — no LLM, no I/O, sub-ms latency.

    Covers ~90% of real PwC consulting inputs based on the testset patterns.
    The risk_classifier_tool runs against the raw user input separately, so
    this structured intake is informational; getting a few fields wrong
    doesn't affect classification correctness.
    """

    low = text.lower()

    domain = next((label for rx, label in _DOMAIN_KEYWORDS if rx.search(low)), "general")
    role = next((label for rx, label in _ROLE_KEYWORDS if rx.search(low)), "unknown")
    modalities = sorted({label for rx, label in _MODALITY_KEYWORDS if rx.search(low)})
    if not modalities:
        modalities = ["text"]

    return {
        "system_purpose": text[:200] or "the described AI system",
        "deployment_context": "EU market" if "eu" in low or "europe" in low else "unspecified",
        "data_modalities": modalities,
        "user_role": role,  # type: ignore[typeddict-item]
        "domain": domain,
        "notes": "heuristic intake (fast path)",
    }


def _append_trace(state: RegPilotState, ev: TraceEvent) -> list[TraceEvent]:
    return [*state.get("trace", []), ev]
