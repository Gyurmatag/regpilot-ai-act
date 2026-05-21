"""Intake classifier node.

Parses free-text user input into a ``StructuredIntake`` record so downstream
nodes don't have to keep re-reading the same blob.
"""

from __future__ import annotations

import json
import logging
import re

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
    llm: LLMClient = get_llm()
    raw_input = state.get("user_input", "").strip()

    try:
        raw = llm.generate(
            _PROMPT.format(description=raw_input),
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=300,
        )
        structured = _parse(raw)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("intake LLM failed: %s — falling back to heuristics", exc)
        structured = _fallback(raw_input)

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
            TraceEvent(node="intake_classifier", summary="structured the user description", payload={"structured": dict(structured)}),
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


def _fallback(text: str) -> StructuredIntake:
    return {
        "system_purpose": text[:200],
        "deployment_context": "unknown",
        "data_modalities": ["text"],
        "user_role": "unknown",
        "domain": "general",
        "notes": "heuristic fallback",
    }


def _append_trace(state: RegPilotState, ev: TraceEvent) -> list[TraceEvent]:
    return [*state.get("trace", []), ev]
