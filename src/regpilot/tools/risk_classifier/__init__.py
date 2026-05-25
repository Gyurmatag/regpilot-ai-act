"""LLM-first risk classifier — entry point + orchestration.

The classifier used to live in a single 561-line module. It's now split
by concern:

* :mod:`regpilot.tools.risk_classifier.bright_lines` — Article 5 prohibited
  patterns + Article 51 GPAI threshold + basic-GPAI markers. The only
  hand-written regex in the hot path; reserved for enumerated regulatory
  text where the AI Act itself prescribes the exact wording.
* :mod:`regpilot.tools.risk_classifier.semantic` — embeddings-based
  Annex III area matcher (cosine similarity against canonical area
  descriptions). Generalises to paraphrases.
* :mod:`regpilot.tools.risk_classifier.llm_verdict` — the LLM prompt +
  structured-output call + tier-coercion logic.

This module wires the three layers together into the public :func:`classify`
function and exposes the :class:`RiskVerdict` dataclass.

Architecture, as the docstring on the old single module described it
(Option C — LLM-primary):

1. Bright-line rule overrides run first, but ONLY for enumerated regulatory
   definitions where the AI Act itself prescribes the exact wording.
2. Semantic similarity for Annex III area candidates feeds the LLM prompt
   as priors.
3. LLM with structured output returns the final tier verdict.
4. Graceful degradation: on LLM failure, fall back to semantic-match →
   high_risk if any candidates score, otherwise heuristic on
   chatbot/generative keywords, otherwise minimal_risk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from regpilot.config import settings
from regpilot.llm import LLMClient, StructuredOutputError, get_llm
from regpilot.schemas import ClassificationResult
from regpilot.state import RiskTier, StructuredIntake
from regpilot.tools.risk_classifier.bright_lines import (
    is_basic_gpai,
    is_systemic_gpai,
    scan_article_5,
)
from regpilot.tools.risk_classifier.llm_verdict import (
    CLASSIFY_PROMPT,
    CLASSIFY_SYSTEM,
    coerce_tier,
)
from regpilot.tools.risk_classifier.semantic import (
    cosine,
    reset_semantic_cache,
    semantic_annex_matches,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskVerdict:
    """Public result type — stable across the codebase even as the layers
    underneath move around."""

    tier: RiskTier
    rationale: str
    annex_iii_matches: list[str]
    article_5_matches: list[str]
    confidence: float  # 0.0 = LLM-only / no rule support, 1.0 = bright-line override


def classify(
    structured: StructuredIntake,
    llm: LLMClient | None = None,
    *,
    raw_text: str = "",
) -> RiskVerdict:
    """Hybrid risk classifier — bright-line rules, then semantic, then LLM.

    Order:

    1. Article 5 bright-line rules → ``prohibited`` (confidence 1.0).
    2a. Article 51 GPAI systemic-risk markers → ``general_purpose_systemic``.
    2b. Basic GPAI shape (foundation / LLM / NNb-parameter / multimodal /
        code-completion model) → ``general_purpose``.
    3. Semantic Annex III matcher surfaces candidate areas above the
       configured similarity threshold.
    4. LLM with structured output returns the final tier verdict, using
       the semantic candidates as priors in its prompt.
    5. On LLM failure, fall back to semantic-match → ``high_risk`` if any
       candidates score, otherwise heuristic on chatbot/generative
       keywords, otherwise ``minimal_risk``.
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
    art5_codes = scan_article_5(corpus)
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
    if is_systemic_gpai(corpus):
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
    if is_basic_gpai(corpus):
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
    semantic_hits = semantic_annex_matches(
        corpus, llm, settings.semantic_match_threshold
    )
    candidate_list = (
        "\n".join(f"- {a} (sim={s:.2f})" for a, s in semantic_hits[:6])
        if semantic_hits
        else "- (none above threshold)"
    )

    # 4. LLM-driven structured verdict.
    prompt = CLASSIFY_PROMPT.format(candidates=candidate_list, description=corpus)
    try:
        result = llm.generate_structured(
            prompt,
            ClassificationResult,
            system=CLASSIFY_SYSTEM,
            temperature=0.0,
            max_tokens=400,
        )
        tier = coerce_tier(result.tier)
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


# Backwards-compatible re-exports for callers that imported from the old
# single-module path. The leading-underscore names (``_cosine``,
# ``_scan_article_5``, ``_is_basic_gpai``, ``_is_systemic_gpai``,
# ``reset_semantic_cache``) historically lived in this module and the
# semantic-classifier test suite still imports them.
_cosine = cosine
_scan_article_5 = scan_article_5
_is_basic_gpai = is_basic_gpai
_is_systemic_gpai = is_systemic_gpai
_coerce_tier = coerce_tier


__all__ = [
    "RiskVerdict",
    "classify",
    "reset_semantic_cache",
    # Backwards-compatible (kept exported for test imports):
    "_cosine",
    "_scan_article_5",
    "_is_basic_gpai",
    "_is_systemic_gpai",
    "_coerce_tier",
    "ClassificationResult",
]
