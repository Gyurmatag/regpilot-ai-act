"""Risk classifier tool.

Hybrid: deterministic keyword/pattern scan first (cheap, explainable), with the
LLM only invoked when no rule matches. Returns a tier verdict + rationale plus
the matched Annex III areas / Article 5 prohibitions so the downstream nodes
can render citations.

Tier vocabulary mirrors the EU AI Act's four-step risk hierarchy:
``prohibited``, ``high_risk``, ``limited_risk``, ``minimal_risk``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from regpilot.ingestion.annex import ANNEX_III, ARTICLE_5_PROHIBITED
from regpilot.llm import LLMClient, get_llm
from regpilot.state import RiskTier, StructuredIntake

logger = logging.getLogger(__name__)


@dataclass
class RiskVerdict:
    tier: RiskTier
    rationale: str
    annex_iii_matches: list[str]
    article_5_matches: list[str]
    confidence: float  # 0.0 = "LLM guess", 1.0 = "exact rule hit"


# --------------------------------------------------------------------------- #
# Rule layer
# --------------------------------------------------------------------------- #


def _rule_scan(text: str) -> tuple[list[str], list[str]]:
    """Return (annex_iii_areas, article_5_codes) that match the input text."""

    annex_hits: list[str] = []
    art5_hits: list[str] = []
    low = text.lower()
    for entry in ANNEX_III:
        if any(_kw_match(kw, low) for kw in entry.keywords):
            annex_hits.append(entry.area)
    for prac in ARTICLE_5_PROHIBITED:
        if any(_kw_match(kw, low) for kw in prac.keywords):
            art5_hits.append(prac.code)

    # Combination patterns: catch wordings the literal keyword scan misses.
    for pattern, code in _COMBO_PATTERNS:
        if pattern.search(low) and code not in art5_hits:
            art5_hits.append(code)

    # Verb-form patterns for the Annex III biometric category — users describe
    # what their system *does* ("analyses emotions", "detects faces") rather
    # than the canonical noun phrase ("emotion recognition").
    for pattern, area in _ANNEX_COMBO_PATTERNS:
        if pattern.search(low) and area not in annex_hits:
            annex_hits.append(area)
    return annex_hits, art5_hits


# Combination patterns for Article 5 — paraphrased wordings the literal
# keyword list can't catch.
_COMBO_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # "police ... predict ... crime|criminal" → 5(1)(d) predictive policing
    (
        re.compile(
            r"\b(police|law\s+enforcement)\b.{0,80}\bpredict\b.{0,80}\b(crime|criminal|offend|reoffend)",
            re.I | re.S,
        ),
        "5(1)(d)",
    ),
    # "predict ... who will commit a crime" — variant
    (
        re.compile(r"\bpredict\b.{0,60}\bwho\s+will\s+commit\b.{0,40}\bcrime", re.I | re.S),
        "5(1)(d)",
    ),
    # "emotion recognition ... (office|workplace|employee|workday)" → 5(1)(f)
    (
        re.compile(
            r"\bemotion\s+recognition\b.{0,80}\b(office|workplace|employee|workday|staff)",
            re.I | re.S,
        ),
        "5(1)(f)",
    ),
    # "scrape ... (face|facial) ... (image|photo)" → 5(1)(e)
    (
        re.compile(
            r"\bscrap(?:e|ing|es|ed)\b.{0,40}\b(facial|face)\b.{0,40}\b(image|photo)",
            re.I | re.S,
        ),
        "5(1)(e)",
    ),
    # Art 5(1)(c) social scoring — verb-form variants. The narrow keyword
    # list ("social scoring") missed paraphrases like "scores citizens by
    # behaviour" or "public authority rates residents based on trust".
    (
        re.compile(
            r"\b(public|government|state|municipal|public\s+sector|public\s+authorit\w*)\b"
            r".{0,80}\b(score|scores|scoring|rate|rates|rating|rank|ranks|ranking)\b"
            r".{0,80}\b(citizen|resident|individual|person|people|household)",
            re.I | re.S,
        ),
        "5(1)(c)",
    ),
    (
        re.compile(
            r"\b(score|rate|rank)\w*\b.{0,40}\b(citizen|resident|individual|person|people|household)"
            r".{0,80}\b(behaviour|behavior|trustworth|reliab|loyalty|conformity|social)",
            re.I | re.S,
        ),
        "5(1)(c)",
    ),
)


# Annex III combo patterns — verb-form biometric / emotion / face detection
# wording variants that the literal keyword scan misses (users describe what
# their system *does*, not the canonical regulatory noun phrase). Trailing
# ``\w*`` on the noun matches plurals (``emotions``, ``faces``, ``identifies``).
_ANNEX_COMBO_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # "analyses / detects / recognises emotions/mood/sentiment"
    (
        re.compile(
            r"\b(analy[sz]\w*|detect\w*|recogni[sz]\w*|monitor\w*|track\w*|infer\w*|classif\w*)\b"
            r".{0,40}\b(emotion|mood|sentiment|affect)\w*",
            re.I | re.S,
        ),
        "Biometrics",
    ),
    # "detects / recognises faces/iris/gait" + plurals
    (
        re.compile(
            r"\b(analy[sz]\w*|detect\w*|recogni[sz]\w*|identif\w*|match\w*)\b"
            r".{0,40}\b(face|facial|iris|fingerprint|gait|voice\s+id)\w*",
            re.I | re.S,
        ),
        "Biometrics",
    ),
    # "recognises individuals/people by their (gait|walking|voice|face)"
    (
        re.compile(
            r"\b(recogni[sz]\w*|identif\w*)\b.{0,40}\b(individual|person|people|visitor|customer|employee)\w*"
            r".{0,40}\b(walking|gait|face|facial|voice|biometric)\w*",
            re.I | re.S,
        ),
        "Biometrics",
    ),
    # CCTV / surveillance camera + biometric / emotion / face context
    (
        re.compile(
            r"\b(cctv|surveillance\s+camera|video\s+feed|video\s+surveillance|security\s+camera)\b"
            r".{0,80}\b(emotion|mood|face|facial|identif|recogni|biometric)\w*",
            re.I | re.S,
        ),
        "Biometrics",
    ),
)


# GPAI patterns — must run before Annex III so frontier LLMs land on Chapter V
# (Articles 51-55) instead of accidental Biometrics false-positives.
_GPAI_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(gpai|general[\s\-_]?purpose(\s+ai)?)\b", re.I),
    re.compile(r"\b(foundation|frontier|base)\s+(model|llm|ai)\b", re.I),
    re.compile(r"\b(large\s+language\s+model|llms?)\b", re.I),
    re.compile(r"\b(text|code|image)?\s*generation\s+(model|service)\b", re.I),
)

# Systemic-risk GPAI per Art. 51 — 10^25 FLOPs training compute is the
# Commission's de-facto threshold; "systemic risk" / "frontier" wording also
# bumps a basic GPAI verdict to the systemic sub-tier.
_GPAI_SYSTEMIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b10\s*\^?\s*25\s*flops?\b", re.I),
    re.compile(r"\b(systemic[\s\-]risk|systemic\s+risk)\b", re.I),
    re.compile(r"\bfrontier\s+(model|llm|ai)\b", re.I),
)


def _kw_match(keyword: str, low_text: str) -> bool:
    # Word-boundary match for short keywords, substring for multi-word phrases.
    if " " in keyword:
        return keyword.lower() in low_text
    return re.search(rf"\b{re.escape(keyword.lower())}\b", low_text) is not None


# --------------------------------------------------------------------------- #
# LLM layer (only used when the rule scan is inconclusive)
# --------------------------------------------------------------------------- #


_CLASSIFY_SYSTEM = (
    "You are an EU AI Act risk classifier. Map the described AI system to one "
    "of: prohibited, high_risk, limited_risk, minimal_risk."
)

_CLASSIFY_PROMPT = """Classify the system below by EU AI Act risk tier.

Definitions:
- prohibited: Article 5 — e.g. social scoring, untargeted facial scraping, predictive policing.
- high_risk: Annex III use cases (employment, education, credit, law enforcement, …) OR Annex I product safety components.
- limited_risk: subject only to Article 50 transparency (chatbots, deepfakes, synthetic content).
- minimal_risk: everything else (spam filters, recommender systems, basic productivity tools).

Reply with STRICT JSON only, no commentary:
{{"tier": "...", "rationale": "...", "annex_iii": ["...", ...]}}

System description:
{description}
"""


def _llm_classify(llm: LLMClient, description: str) -> RiskVerdict:
    raw = llm.generate(
        _CLASSIFY_PROMPT.format(description=description),
        system=_CLASSIFY_SYSTEM,
        temperature=0.0,
        max_tokens=300,
    )
    obj = _safe_json_obj(raw)
    tier = obj.get("tier", "unknown")
    if tier not in ("prohibited", "high_risk", "limited_risk", "minimal_risk"):
        tier = "unknown"
    return RiskVerdict(
        tier=tier,  # type: ignore[arg-type]
        rationale=str(obj.get("rationale", ""))[:500],
        annex_iii_matches=[str(a) for a in obj.get("annex_iii", [])][:5],
        article_5_matches=[],
        confidence=0.5,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def classify(
    structured: StructuredIntake,
    llm: LLMClient | None = None,
    *,
    raw_text: str = "",
) -> RiskVerdict:
    """Hybrid classifier: rules first, LLM only on miss.

    The rule scan runs over BOTH the structured intake fields and the
    original ``raw_text`` (when supplied) so a weak intake LLM can't drop
    a keyword like "CV screening" and trick the system into the wrong tier.
    """

    llm = llm or get_llm()
    text_for_rules = " ".join(
        filter(
            None,
            [
                raw_text,
                *(
                    str(structured.get(k, ""))
                    for k in ("system_purpose", "deployment_context", "domain", "notes")
                ),
            ],
        )
    )

    annex_hits, art5_hits = _rule_scan(text_for_rules)
    if art5_hits:
        return RiskVerdict(
            tier="prohibited",
            rationale=f"Matches Article 5 prohibited practice(s): {', '.join(art5_hits)}.",
            annex_iii_matches=annex_hits,
            article_5_matches=art5_hits,
            confidence=1.0,
        )

    # GPAI detection before Annex III so frontier LLMs surface as GPAI rather
    # than a Biometrics false-positive when their description happens to mention
    # "voice" or "face". Two sub-tiers per Chapter V — Article 51 systemic-risk
    # threshold (10^25 FLOPs / EU Commission designation).
    gpai_systemic = any(p.search(text_for_rules) for p in _GPAI_SYSTEMIC_PATTERNS)
    gpai_basic = any(p.search(text_for_rules) for p in _GPAI_PATTERNS)
    if gpai_systemic or gpai_basic:
        tier_g: RiskTier = "general_purpose_systemic" if gpai_systemic else "general_purpose"
        rationale = (
            "Matches systemic-risk GPAI thresholds (Art. 51) — Articles 53–55 apply."
            if gpai_systemic
            else "Matches general-purpose AI model patterns — Articles 53–54 apply."
        )
        return RiskVerdict(
            tier=tier_g,
            rationale=rationale,
            annex_iii_matches=[],
            article_5_matches=[],
            confidence=0.95,
        )

    if annex_hits:
        # Limited-risk transparency duties only attach to chatbots/deepfakes; high-risk
        # Annex III hits dominate over limited-risk hints.
        return RiskVerdict(
            tier="high_risk",
            rationale=f"Matches Annex III high-risk area(s): {', '.join(annex_hits)}.",
            annex_iii_matches=annex_hits,
            article_5_matches=[],
            confidence=1.0,
        )

    if re.search(
        r"\b(chatbot|deepfake|synthetic\s+(media|content)|generative|voice\s+assistant|"
        r"virtual\s+assistant|conversational\s+agent)\b",
        text_for_rules,
        re.I,
    ):
        return RiskVerdict(
            tier="limited_risk",
            rationale="Generative / conversational patterns trigger Article 50 transparency obligations.",
            annex_iii_matches=[],
            article_5_matches=[],
            confidence=0.9,
        )

    logger.info("Rule scan inconclusive — invoking LLM classifier.")
    return _llm_classify(llm, text_for_rules)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _safe_json_obj(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
