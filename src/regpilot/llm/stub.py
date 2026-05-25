"""Deterministic mock LLM used by tests, CI, and offline dev.

Every test that runs without a real Ollama instance uses this. The stub
recognises the prompts emitted by the various agent nodes by keyword
sentinels and returns a synthetic-but-validation-passing reply, so the
LangGraph wiring + downstream nodes can be exercised end-to-end without an
LLM server.

Embeddings here are SHA-256 hash-based — meaningless semantically, but
deterministic across runs. The stub is therefore useful for testing the
classifier and the deterministic regulatory layer (deadline calculator,
Article 5 bright-line rules), not for measuring real retrieval quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from regpilot.llm.base import LLMClient
from regpilot.llm.helpers import extract_after, safe_json_obj

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Regex sentinels — used by the heuristic ``_stub_classify``
# --------------------------------------------------------------------------- #

_PROHIBITED_HINTS = re.compile(
    r"\b(social\s+scoring|emotion\s+recognition\s+(in\s+the\s+workplace|at\s+work)"
    r"|untargeted\s+scraping|predictive\s+policing|biometric\s+categori[sz]ation"
    r"|real-time\s+remote\s+biometric)\b",
    re.I,
)

_HIGH_RISK_HINTS = re.compile(
    r"\b(recruit\w*|hir(e|ing)|cv\s+screening|credit\s+scoring"
    r"|education|exam\s+proctor|law\s+enforcement|migration|critical\s+infrastructure"
    r"|medical\s+device|judicial"
    # Biometric variants — the stub also has to recognise these so tests stay
    # deterministic without a real LLM.
    r"|emotion\w*|face\w*|facial|biometric\w*"
    r"|fingerprint\w*|iris|gait|cctv|surveillance|mood|walking\s+pattern)\b",
    re.I,
)

_GPAI_SYSTEMIC_HINTS = re.compile(
    r"\b(10\s*\^?\s*25\s*flops?|systemic[\s\-]risk|frontier\s+(model|llm|ai))\b",
    re.I,
)

_GPAI_HINTS = re.compile(
    r"\b(gpai|general[\s\-_]?purpose(\s+ai)?|foundation\s+(model|llm|ai)"
    r"|large\s+language\s+model|llms?)\b",
    re.I,
)

_LIMITED_HINTS = re.compile(
    r"\b(chatbot|deepfake|synthetic\s+(media|content)|generative)\b", re.I
)

_STUB_ANNEX_AREAS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(recruit(ment|ing)?|hir(e|ing)|cv\s+screening)\b", re.I),
     "Employment, worker management, access to self-employment"),
    (re.compile(r"\b(credit\s+scoring|loan|insurance)\b", re.I),
     "Access to and enjoyment of essential private and public services and benefits"),
    (re.compile(r"\b(education|exam\s+proctor|student|school|grading)\b", re.I),
     "Education and vocational training"),
    (re.compile(r"\b(law\s+enforcement|police|recidivism|polygraph)\b", re.I),
     "Law enforcement"),
    (re.compile(r"\b(critical\s+infrastructure|power\s+grid|electricity|water\s+supply)\b", re.I),
     "Critical infrastructure"),
    (re.compile(r"\b(border|migration|asylum|visa)\b", re.I),
     "Migration, asylum, border control"),
    (re.compile(r"\b(judicial|court|election|referendum)\b", re.I),
     "Administration of justice and democratic processes"),
    (re.compile(
        r"\b(face\w*|facial|iris|fingerprint\w*|emotion\w*|mood|biometric\w*"
        r"|gait|cctv|surveillance|walking\s+pattern)\b",
        re.I,
    ), "Biometrics"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _deterministic_embedding(text: str, dim: int = 256) -> list[float]:
    """Hash-based pseudo-embedding so the stub still drives a working RAG path."""

    h = hashlib.sha256(text.encode("utf-8")).digest()
    needed = math.ceil(dim / 32)
    expanded = (h * needed)[:dim]
    return [(b - 128) / 128.0 for b in expanded]


def _excerpt(text: str, n: int) -> str:
    return text.strip().replace("\n", " ")[:n]


def _guess_modalities(text: str) -> list[str]:
    found: list[str] = []
    for needle, label in [
        ("image", "image"),
        ("video", "video"),
        ("audio", "audio"),
        ("voice", "audio"),
        ("text", "text"),
        ("biometric", "biometric"),
        ("face", "biometric"),
    ]:
        if needle in text.lower():
            found.append(label)
    return found or ["text"]


def _guess_domain(text: str) -> str:
    low = text.lower()
    for needle, label in [
        ("hir", "HR / recruitment"),
        ("recruit", "HR / recruitment"),
        ("cv", "HR / recruitment"),
        ("credit", "financial services"),
        ("medical", "healthcare"),
        ("hospital", "healthcare"),
        ("educat", "education"),
        ("exam", "education"),
        ("police", "law enforcement"),
        ("border", "migration / border"),
    ]:
        if needle in low:
            return label
    return "general"


def _stub_classify(text: str) -> tuple[str, str]:
    if _PROHIBITED_HINTS.search(text):
        return "prohibited", "matches an Article 5 prohibited-practice pattern"
    if _HIGH_RISK_HINTS.search(text):
        return "high_risk", "matches an Annex III high-risk use-case pattern"
    if _GPAI_SYSTEMIC_HINTS.search(text):
        return "general_purpose_systemic", "matches Article 51 systemic-risk GPAI markers"
    if _GPAI_HINTS.search(text):
        return "general_purpose", "matches a general-purpose AI model pattern"
    if _LIMITED_HINTS.search(text):
        return "limited_risk", "subject to Article 50 transparency obligations"
    return "minimal_risk", "no high-risk or prohibited indicators detected"


def _stub_annex_areas(text: str) -> list[str]:
    seen: list[str] = []
    for rx, area in _STUB_ANNEX_AREAS:
        if rx.search(text) and area not in seen:
            seen.append(area)
    return seen


def _stub_next_steps(tier: str) -> str:
    if tier == "high_risk":
        return (
            "1. Confirm the risk classification and applicable Annex III area with legal counsel.\n"
            "2. Map each obligation in the table to an internal owner and target date.\n"
            "3. Compile technical documentation per Annex IV (Art. 11) and prepare for the conformity assessment (Art. 43)."
        )
    if tier == "limited_risk":
        return (
            "1. Implement the Article 50 transparency disclosures in the user-facing flow.\n"
            "2. Label any AI-generated or AI-modified media (deepfakes, synthetic text) accordingly.\n"
            "3. Track Article 50 implementing guidance from the AI Office."
        )
    if tier == "minimal_risk":
        return (
            "1. No mandatory obligations apply, but adopt a voluntary code of conduct per Article 95.\n"
            "2. Re-check classification annually as the system evolves.\n"
            "3. Apply general data-protection and product-liability law as a baseline."
        )
    return (
        "1. Cease placing the system on the EU market and putting it into service.\n"
        "2. Consult legal counsel on remediation and potential redesign.\n"
        "3. Communicate the change to internal stakeholders and customers."
    )


def _stub_report(prompt: str) -> str:
    obligation_articles = re.findall(
        r"\d{4}-\d{2}-\d{2}\s+\u2014\s+Art\.\s*(\d+[a-z]?)", prompt
    )
    seen: list[str] = []
    for a in obligation_articles:
        if a not in seen:
            seen.append(a)
    cite = ", ".join(f"Art. {a}" for a in seen) or "Art. 6"

    tier_match = re.search(r"Risk tier:\s*([a-z_]+)", prompt)
    tier = tier_match.group(1) if tier_match else "unknown"
    next_steps = _stub_next_steps(tier)

    return (
        "## Executive summary\n"
        "A compliance roadmap based on the supplied EU AI Act context.\n\n"
        "## Risk classification\n"
        f"The system has been classified per the triage rationale. "
        f"Applicable Articles: {cite}.\n\n"
        "## Obligations & deadlines\n"
        "The full obligation table is shown in the trace panel; each entry "
        "cites the Article it derives from.\n\n"
        "## Recommended next steps\n"
        f"{next_steps}\n"
    )


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class StubClient(LLMClient):
    """Schema-aware deterministic mock."""

    chat_model = "stub"
    embed_model = "stub"
    provider = "stub"

    # ------------------------------- generate ------------------------------ #

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Pattern-match the prompt to one of the agent nodes and return a
        deterministic synthetic reply. Used by the legacy free-form
        ``llm.generate`` callsites; the modern ``generate_structured``
        callsites take a different code path below."""

        low = prompt.lower()

        if "intake_classifier" in low and "description:" in low:
            desc = prompt.split("Description:", 1)[-1].strip()
            return json.dumps(
                {
                    "system_purpose": _excerpt(desc, 200),
                    "deployment_context": "EU market",
                    "data_modalities": _guess_modalities(desc),
                    "user_role": "provider",
                    "domain": _guess_domain(desc),
                    "notes": "stub-generated",
                }
            )

        if "draft a compliance roadmap" in low or "draft report for tier" in low:
            return _stub_report(prompt)

        if "return strict json: a list of the" in low and "indices" in low:
            return "[0,1,2,3,4]"

        if "query rewrite task" in low:
            return json.dumps(
                [
                    "EU AI Act obligations applicable to the described system",
                    "compliance requirements under Regulation (EU) 2024/1689",
                ]
            )

        if "classify the system below by eu ai act risk tier" in low:
            desc = prompt.split("System description:", 1)[-1]
            tier, rationale = _stub_classify(desc)
            return json.dumps(
                {"tier": tier, "rationale": rationale, "annex_iii_areas": []}
            )

        if "self-critique" in low:
            return json.dumps({"ok": True, "issues": []})

        return "Stub LLM response."

    # -------------------------- generate_structured ------------------------ #

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> T:
        """Recognise each Pydantic schema by name + fields and return a
        validation-passing instance. Tests stay deterministic; no real LLM."""

        name = schema.__name__
        fields = set(getattr(schema, "model_fields", {}).keys())

        if name == "ClassificationResult" or fields.issuperset({"tier", "rationale"}):
            desc = (
                extract_after(prompt, "System description:")
                or extract_after(prompt, "Description:")
                or prompt
            )
            tier, rationale = _stub_classify(desc)
            areas = _stub_annex_areas(desc) if tier == "high_risk" else []
            return schema.model_validate(
                {
                    "tier": tier,
                    "rationale": rationale,
                    "annex_iii_areas": areas,
                    "art_5_codes": [],
                }
            )

        if name == "IntakeSchema" or fields.issuperset(
            {"system_purpose", "user_role", "data_modalities"}
        ):
            desc = extract_after(prompt, "Description:") or prompt
            return schema.model_validate(
                {
                    "system_purpose": _excerpt(desc, 200),
                    "deployment_context": "EU market",
                    "data_modalities": _guess_modalities(desc),
                    "user_role": "provider",
                    "domain": _guess_domain(desc),
                    "notes": "stub-generated",
                }
            )

        if name == "ReportSections" or fields.issuperset(
            {"executive_summary", "risk_classification_narrative", "recommended_next_steps"}
        ):
            tier_match = re.search(r"Risk tier.*?:\s*([A-Za-z _\-]+)", prompt)
            tier = (
                tier_match.group(1).strip().lower().replace(" ", "_")
                if tier_match
                else "unknown"
            )
            articles = sorted(
                {
                    a
                    for a in re.findall(
                        r"\d{4}-\d{2}-\d{2}\s+\u2014\s+Art\.\s*(\d+[a-z]?)", prompt
                    )
                }
            ) or ["6"]
            steps = _stub_next_steps(tier).splitlines()
            return schema.model_validate(
                {
                    "executive_summary": (
                        f"This system is classified per the supplied triage. The "
                        f"compliance roadmap below lists the applicable Articles "
                        f"({', '.join('Art. ' + a for a in articles)}) and their "
                        f"Article 113 phased deadlines."
                    ),
                    "risk_classification_narrative": (
                        f"The triage analysis assigned this system to the relevant "
                        f"tier. The applicable obligations derive from the cited "
                        f"Articles ({', '.join('Art. ' + a for a in articles)}). "
                        f"Each obligation entry includes the precise Article 113 "
                        f"date the duty becomes enforceable."
                    ),
                    "recommended_next_steps": [
                        s.split(". ", 1)[1] if ". " in s else s for s in steps if s.strip()
                    ],
                }
            )

        # Default: try to fish JSON out of the free-form generate() output.
        raw = self.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        obj = safe_json_obj(raw)
        return schema.model_validate(obj or {})

    # ------------------------------- embed --------------------------------- #

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_deterministic_embedding(t) for t in texts]
