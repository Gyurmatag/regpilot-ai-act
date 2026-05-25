"""LLM-first risk classifier with semantic-similarity Annex III matching.

Architecture (Option C — LLM-primary):

1. **Bright-line rule overrides** run first, but ONLY for enumerated
   regulatory definitions where the AI Act itself prescribes the exact
   wording — Article 5 prohibited practices and Chapter V GPAI markers
   (Art. 51 systemic-risk threshold + Art. 53 basic-GPAI shape).
   Everything else flows to the LLM. The rules are defensive guards, not
   the engine.

2. **Semantic similarity** for Annex III area candidates. Each Annex III
   area's canonical description is embedded once per process; the user's
   description is embedded; cosine similarity surfaces the candidate areas
   above ``settings.semantic_match_threshold`` (default 0.35). This
   *generalises to paraphrases* the way a hand-written regex never could
   — "traffic light timing" matches Critical infrastructure, "legal
   research for judges" matches Administration of justice, etc. The
   surfaced candidates feed the LLM prompt as priors.

3. **LLM-driven verdict via structured output**. The LLM receives the user
   description, the semantic candidates, and the explicit tier vocabulary
   + decision rules, and returns a Pydantic-validated
   :class:`ClassificationResult` (tier, rationale, Annex III areas,
   Article 5 codes). This is where the actual judgement happens — the
   prompt instructs the LLM to prefer high-similarity candidates but make
   its own call.

4. **Graceful degradation**. If the LLM call fails or returns invalid
   structured output, we fall back to the semantic-match areas + a
   chatbot/generative keyword heuristic so the agent never crashes.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

from regpilot.config import settings
from regpilot.ingestion.annex import ANNEX_III, ARTICLE_5_PROHIBITED
from regpilot.llm import LLMClient, StructuredOutputError, get_llm
from regpilot.state import RiskTier, StructuredIntake

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public dataclass — stays stable across the codebase
# --------------------------------------------------------------------------- #


@dataclass
class RiskVerdict:
    tier: RiskTier
    rationale: str
    annex_iii_matches: list[str]
    article_5_matches: list[str]
    confidence: float  # 0.0 = LLM-only / no rule support, 1.0 = bright-line override


# --------------------------------------------------------------------------- #
# Pydantic schema for LLM structured output
# --------------------------------------------------------------------------- #


_TIER_LITERAL = (
    "prohibited",
    "high_risk",
    "limited_risk",
    "minimal_risk",
    "general_purpose",
    "general_purpose_systemic",
)


class ClassificationResult(BaseModel):
    """Schema the LLM fills in for the classification verdict."""

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
# Bright-line rule layer — Article 5 + GPAI Art. 51 systemic-risk threshold
# --------------------------------------------------------------------------- #


def _kw_match(keyword: str, low_text: str) -> bool:
    if " " in keyword:
        return keyword.lower() in low_text
    return re.search(rf"\b{re.escape(keyword.lower())}\b", low_text) is not None


_ART5_COMBO_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(police|law\s+enforcement)\b.{0,80}\bpredict\b.{0,80}\b(crime|criminal|offend|reoffend)",
            re.I | re.S,
        ),
        "5(1)(d)",
    ),
    (
        re.compile(r"\bpredict\b.{0,60}\bwho\s+will\s+commit\b.{0,40}\bcrime", re.I | re.S),
        "5(1)(d)",
    ),
    (
        re.compile(
            r"\bemotion\s+recognition\b.{0,80}\b(office|workplace|employee|workday|staff)",
            re.I | re.S,
        ),
        "5(1)(f)",
    ),
    (
        re.compile(
            r"\bscrap(?:e|ing|es|ed)\b.{0,40}\b(facial|face)\b.{0,40}\b(image|photo)",
            re.I | re.S,
        ),
        "5(1)(e)",
    ),
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


def _scan_article_5(text: str) -> list[str]:
    """Return the matching Article 5 sub-clause codes (bright-line override)."""

    low = text.lower()
    hits: list[str] = []
    for prac in ARTICLE_5_PROHIBITED:
        if any(_kw_match(kw, low) for kw in prac.keywords):
            hits.append(prac.code)
    for pattern, code in _ART5_COMBO_PATTERNS:
        if pattern.search(low) and code not in hits:
            hits.append(code)
    return hits


# GPAI is detected by literal regulatory markers (Art. 51 threshold language).
# The LLM also gets to suggest GPAI, but these patterns are 100% confident.
_GPAI_SYSTEMIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b10\s*\^?\s*25\s*flops?\b", re.I),
    re.compile(r"\b10\s*\^?\s*2[5-9]\s*flops?\b", re.I),  # ≥10^25
    re.compile(r"\b(systemic[\s\-]risk|systemic\s+risk)\s+(model|ai|llm|gpai)\b", re.I),
    re.compile(r"\bfrontier\s+(model|llm|ai|foundation|multimodal)\b", re.I),
)


# Basic GPAI markers (Chapter V Art. 53/54 apply but NOT Art. 55 systemic).
# These run AFTER the systemic check so frontier markers take precedence.
_GPAI_BASIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(gpai|general[\s\-_]?purpose\s+ai)\b", re.I),
    re.compile(r"\b(foundation|base)\s+(model|llm|ai)\b", re.I),
    re.compile(r"\b(large\s+language\s+model|llms?)\b", re.I),
    # "13B-parameter model", "7-billion-parameter LLM", etc. — common GPAI shape.
    re.compile(
        r"\b\d{1,3}\s*[bB]\s*[-\s]?\s*parameter\s+(model|llm|transformer|"
        r"foundation|generative|completion)",
        re.I,
    ),
    # Generative / code-completion / multimodal model offered as a service.
    re.compile(
        r"\b(code[-\s]?completion|multimodal|generative|generation)\s+"
        r"(model|service|api)\b",
        re.I,
    ),
)


def _is_systemic_gpai(text: str) -> bool:
    return any(p.search(text) for p in _GPAI_SYSTEMIC_PATTERNS)


def _is_basic_gpai(text: str) -> bool:
    return any(p.search(text) for p in _GPAI_BASIC_PATTERNS)


# --------------------------------------------------------------------------- #
# Semantic-similarity Annex III matcher
# --------------------------------------------------------------------------- #


_ANNEX_EMB_CACHE: list[tuple[str, list[float]]] | None = None
_ANNEX_EMB_LOCK = Lock()


def _build_annex_corpus() -> list[tuple[str, str]]:
    """Build (area, embed_text) tuples. Embed text combines area name +
    description + a few canonical example phrases so the embedding captures
    the conceptual breadth of each area, not just the legal definition."""

    examples = {
        "Biometrics": (
            "Examples: emotion recognition, face recognition, biometric "
            "categorisation, voice biometric authentication, gait recognition, "
            "fingerprint identification, iris recognition, CCTV face detection, "
            "speaker verification, customer voice authentication, call-centre "
            "biometric ID."
        ),
        "Critical infrastructure": (
            "Examples: AI controlling the electricity grid, gas distribution "
            "network, water supply, road traffic signals, traffic light timing, "
            "intelligent traffic management, railway signalling, train control, "
            "load balancing for utilities, smart grid control, demand-response "
            "AI."
        ),
        "Education and vocational training": (
            "Examples: AI scoring student exams, exam proctoring, university "
            "admission ranking, PhD applicant ranking, doctoral programme "
            "selection, automated grading, learning outcome assessment, "
            "scholarship eligibility decisions."
        ),
        "Employment, worker management, access to self-employment": (
            "Examples: CV screening, resume ranking, candidate selection, "
            "promotion decisions, employee performance evaluation, task "
            "allocation between workers, worker monitoring."
        ),
        "Access to and enjoyment of essential private and public services and benefits": (
            "Examples: credit scoring, loan eligibility, mortgage approval, "
            "welfare benefit eligibility, public assistance triage, life and "
            "health insurance pricing, health insurance underwriting, premium "
            "calculation, emergency dispatch, ambulance triage, real-time "
            "credit card fraud detection, transaction authorisation, payment "
            "blocking, point-of-sale fraud blocking."
        ),
        "Law enforcement": (
            "Examples: risk assessment of suspects, polygraph analysis, "
            "evaluating evidence reliability, criminal profiling, recidivism "
            "prediction, detective support tools."
        ),
        "Migration, asylum, border control": (
            "Examples: border control AI, asylum application processing, "
            "visa decision support, migrant risk assessment, document "
            "authenticity at borders."
        ),
        "Administration of justice and democratic processes": (
            "Examples: AI assisting judges, legal research tools for judges, "
            "case-precedent suggestions for judicial decisions, fact "
            "interpretation for court rulings, statute reference tools for "
            "judges, civil case decision support, election influence systems, "
            "voting behaviour analysis."
        ),
    }

    return [
        (
            e.area,
            f"Annex III area: {e.area}. {e.description} "
            f"{examples.get(e.area, '')}",
        )
        for e in ANNEX_III
    ]


def _get_annex_embeddings(llm: LLMClient) -> list[tuple[str, list[float]]]:
    """Lazily compute + cache Annex III area embeddings (one per process)."""

    global _ANNEX_EMB_CACHE
    if _ANNEX_EMB_CACHE is not None:
        return _ANNEX_EMB_CACHE
    with _ANNEX_EMB_LOCK:
        if _ANNEX_EMB_CACHE is not None:
            return _ANNEX_EMB_CACHE
        corpus = _build_annex_corpus()
        texts = [t for _, t in corpus]
        try:
            vectors = llm.embed(texts)
        except Exception as exc:
            logger.warning("Annex III embedding precompute failed: %s", exc)
            _ANNEX_EMB_CACHE = []
            return _ANNEX_EMB_CACHE
        _ANNEX_EMB_CACHE = [
            (area, vec) for (area, _), vec in zip(corpus, vectors, strict=True)
        ]
        logger.info(
            "Annex III semantic index built: %d areas (dim=%d)",
            len(_ANNEX_EMB_CACHE),
            len(_ANNEX_EMB_CACHE[0][1]) if _ANNEX_EMB_CACHE else 0,
        )
        return _ANNEX_EMB_CACHE


def reset_semantic_cache() -> None:
    """Test helper — forces the next call to recompute Annex III embeddings."""

    global _ANNEX_EMB_CACHE
    _ANNEX_EMB_CACHE = None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _semantic_annex_matches(
    text: str, llm: LLMClient, threshold: float
) -> list[tuple[str, float]]:
    """Return (area, score) pairs sorted by descending cosine similarity,
    filtered to those above ``threshold``. Empty list on embedder failure."""

    index = _get_annex_embeddings(llm)
    if not index or not text.strip():
        return []
    try:
        query_vec = llm.embed([text])[0]
    except Exception as exc:
        logger.warning("Query embedding failed in semantic matcher: %s", exc)
        return []
    scored = [(area, _cosine(query_vec, vec)) for area, vec in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(a, s) for a, s in scored if s >= threshold]


# --------------------------------------------------------------------------- #
# LLM prompt
# --------------------------------------------------------------------------- #


_CLASSIFY_SYSTEM = (
    "You are an EU AI Act compliance analyst. You classify AI systems against "
    "the six-step risk hierarchy of Regulation (EU) 2024/1689. You always "
    "respond with strict JSON matching the requested schema and you ground "
    "your verdict in the supplied Annex III candidates whenever possible."
)


_CLASSIFY_PROMPT = """Classify the system below by EU AI Act risk tier.

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


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def classify(
    structured: StructuredIntake,
    llm: LLMClient | None = None,
    *,
    raw_text: str = "",
) -> RiskVerdict:
    """LLM-first hybrid classifier.

    Order:
    1. Article 5 bright-line rules → prohibited (confidence 1.0).
    2a. Article 51 GPAI systemic-risk markers → general_purpose_systemic (confidence 1.0).
    2b. Basic GPAI shape (foundation / LLM / NNb-parameter / multimodal /
        code-completion model) → general_purpose (confidence 1.0).
    3. Semantic Annex III matcher surfaces candidate areas above threshold.
    4. LLM with structured output returns the final tier verdict.
    5. On LLM failure, fall back to semantic-match → high_risk if any
       candidates score, otherwise heuristic by chatbot/generative keywords.
    """

    llm = llm or get_llm()
    corpus = " ".join(
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

    # 1. Article 5 bright-line override.
    art5_codes = _scan_article_5(corpus)
    if art5_codes:
        return RiskVerdict(
            tier="prohibited",
            rationale=(
                "Matches Article 5 prohibited practice(s): "
                f"{', '.join(art5_codes)}. Article 5 patterns are enumerated "
                "regulatory definitions and override the LLM verdict."
            ),
            annex_iii_matches=[],
            article_5_matches=art5_codes,
            confidence=1.0,
        )

    # 2a. Article 51 systemic-risk GPAI bright-line override.
    if _is_systemic_gpai(corpus):
        return RiskVerdict(
            tier="general_purpose_systemic",
            rationale=(
                "Matches Article 51 systemic-risk GPAI markers (≥10^25 FLOPs "
                "training compute or 'frontier' designation). Articles 53-55 apply."
            ),
            annex_iii_matches=[],
            article_5_matches=[],
            confidence=1.0,
        )

    # 2b. Basic GPAI bright-line override (Chapter V Art. 53/54 — no Art. 55).
    if _is_basic_gpai(corpus):
        return RiskVerdict(
            tier="general_purpose",
            rationale=(
                "Matches general-purpose AI model patterns (foundation model, "
                "LLM, code-completion model, NNb-parameter generator). "
                "Articles 53-54 apply; Art. 55 only if systemic risk."
            ),
            annex_iii_matches=[],
            article_5_matches=[],
            confidence=1.0,
        )

    # 3. Semantic Annex III candidates.
    semantic_hits = _semantic_annex_matches(
        corpus, llm, settings.semantic_match_threshold
    )
    candidate_list = (
        "\n".join(f"- {a} (sim={s:.2f})" for a, s in semantic_hits[:6])
        if semantic_hits
        else "- (none above threshold)"
    )

    # 4. LLM-driven structured verdict.
    prompt = _CLASSIFY_PROMPT.format(candidates=candidate_list, description=corpus)
    try:
        result = llm.generate_structured(
            prompt,
            ClassificationResult,
            system=_CLASSIFY_SYSTEM,
            temperature=0.0,
            max_tokens=400,
        )
        tier = _coerce_tier(result.tier)
        return RiskVerdict(
            tier=tier,
            rationale=result.rationale[:500] or "LLM verdict (no rationale provided).",
            annex_iii_matches=[
                a for a in (result.annex_iii_areas or [])[:6] if isinstance(a, str)
            ],
            article_5_matches=[
                c for c in (result.art_5_codes or [])[:6] if isinstance(c, str)
            ],
            confidence=0.85 if semantic_hits else 0.7,
        )
    except StructuredOutputError as exc:
        logger.warning("LLM structured classification failed: %s — using fallback", exc)
    except Exception as exc:
        logger.warning("LLM classification crashed: %s — using fallback", exc)

    # 5. Graceful degradation: semantic hits → high_risk, else heuristic.
    if semantic_hits:
        areas = [a for a, _ in semantic_hits[:3]]
        return RiskVerdict(
            tier="high_risk",
            rationale=(
                "Fallback verdict: semantic similarity placed the description "
                f"in Annex III area(s) {', '.join(areas)} (LLM unavailable)."
            ),
            annex_iii_matches=areas,
            article_5_matches=[],
            confidence=0.55,
        )
    if re.search(
        r"\b(chatbot|deepfake|synthetic\s+(media|content)|generative|voice\s+assistant|"
        r"virtual\s+assistant|conversational\s+agent)\b",
        corpus,
        re.I,
    ):
        return RiskVerdict(
            tier="limited_risk",
            rationale="Fallback verdict: generative/conversational pattern → Article 50 transparency.",
            annex_iii_matches=[],
            article_5_matches=[],
            confidence=0.5,
        )
    return RiskVerdict(
        tier="minimal_risk",
        rationale="Fallback verdict: no Annex III, Article 5 or GPAI markers matched.",
        annex_iii_matches=[],
        article_5_matches=[],
        confidence=0.4,
    )


def _coerce_tier(raw: Any) -> RiskTier:
    val = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if val in _TIER_LITERAL:
        return val  # type: ignore[return-value]
    # Forgiving aliases the model sometimes emits.
    aliases = {
        "gpai": "general_purpose",
        "general_purpose_ai": "general_purpose",
        "general_purpose_model": "general_purpose",
        "gpai_systemic": "general_purpose_systemic",
        "high": "high_risk",
        "limited": "limited_risk",
        "minimal": "minimal_risk",
    }
    if val in aliases:
        return aliases[val]  # type: ignore[return-value]
    return "unknown"
