"""Bright-line rule layer for the risk classifier.

The only hand-written regex still allowed in the hot path. Reserved for
two narrow cases where the AI Act itself enumerates the exact regulatory
pattern, so deterministic matching is the correct call for auditability:

* **Article 5 prohibited practices** — Annex of the Act lists the eight
  sub-clauses with precise wording. We match both the literal keyword
  list (from :mod:`regpilot.ingestion.annex`) and a small set of combo
  patterns for paraphrases ("scores citizens by behaviour", "predict who
  will commit a crime").
* **Chapter V GPAI markers** — Article 51 systemic-risk threshold
  (≥10^25 FLOPs / "frontier" model) and Article 53 basic-GPAI shape
  (foundation model, LLM, NNb-parameter generator). These are
  short-circuits before the LLM ever sees the input.

Everything else flows through the semantic matcher + the LLM verdict.
"""

from __future__ import annotations

import re

from regpilot.ingestion.annex import ARTICLE_5_PROHIBITED

# --------------------------------------------------------------------------- #
# Article 5 — keyword scan + combo patterns
# --------------------------------------------------------------------------- #


def _kw_match(keyword: str, low_text: str) -> bool:
    """Word-boundary match for single-word keywords; substring for phrases."""

    if " " in keyword:
        return keyword.lower() in low_text
    return re.search(rf"\b{re.escape(keyword.lower())}\b", low_text) is not None


# Verb-form / paraphrase patterns the literal keyword list (in
# ``ingestion.annex.ARTICLE_5_PROHIBITED``) doesn't catch. Each entry maps a
# compiled regex to the Article 5 sub-clause it implies.
_ART5_COMBO_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 5(1)(d): predictive policing by profiling.
    (
        re.compile(
            r"\b(police|law\s+enforcement)\b.{0,80}\bpredict\b"
            r".{0,80}\b(crime|criminal|offend|reoffend)",
            re.I | re.S,
        ),
        "5(1)(d)",
    ),
    (
        re.compile(
            r"\bpredict\b.{0,60}\bwho\s+will\s+commit\b.{0,40}\bcrime", re.I | re.S
        ),
        "5(1)(d)",
    ),
    # 5(1)(f): emotion recognition in workplace / education.
    (
        re.compile(
            r"\bemotion\s+recognition\b.{0,80}\b(office|workplace|employee|workday|staff)",
            re.I | re.S,
        ),
        "5(1)(f)",
    ),
    # 5(1)(e): untargeted scraping of facial images.
    (
        re.compile(
            r"\bscrap(?:e|ing|es|ed)\b.{0,40}\b(facial|face)\b.{0,40}\b(image|photo)",
            re.I | re.S,
        ),
        "5(1)(e)",
    ),
    # 5(1)(c): social scoring — public-sector + score/rate/rank + citizens.
    (
        re.compile(
            r"\b(public|government|state|municipal|public\s+sector|public\s+authorit\w*)\b"
            r".{0,80}\b(score|scores|scoring|rate|rates|rating|rank|ranks|ranking)\b"
            r".{0,80}\b(citizen|resident|individual|person|people|household)",
            re.I | re.S,
        ),
        "5(1)(c)",
    ),
    # 5(1)(c): social scoring — verb + people + behaviour/trustworthiness.
    (
        re.compile(
            r"\b(score|rate|rank)\w*\b.{0,40}\b(citizen|resident|individual|person|people|household)"
            r".{0,80}\b(behaviour|behavior|trustworth|reliab|loyalty|conformity|social)",
            re.I | re.S,
        ),
        "5(1)(c)",
    ),
)


def scan_article_5(text: str) -> list[str]:
    """Return the Article 5 sub-clause codes that match ``text``.

    Combines a literal keyword scan (over
    :data:`regpilot.ingestion.annex.ARTICLE_5_PROHIBITED`) with the
    paraphrase combo patterns above. Returns a deduplicated list in the
    order matches were found; an empty list means no Article 5 rule
    fired and the classifier should continue downstream.
    """

    low = text.lower()
    hits: list[str] = []
    for prac in ARTICLE_5_PROHIBITED:
        if any(_kw_match(kw, low) for kw in prac.keywords):
            hits.append(prac.code)
    for pattern, code in _ART5_COMBO_PATTERNS:
        if pattern.search(low) and code not in hits:
            hits.append(code)
    return hits


# --------------------------------------------------------------------------- #
# Chapter V GPAI — systemic-risk threshold + basic-GPAI shape
# --------------------------------------------------------------------------- #


# Article 51 systemic-risk markers. The 10^25 FLOPs threshold is literal
# regulatory text; "frontier" / "systemic-risk" are the EU Commission's
# shorthand. Any of these → ``general_purpose_systemic``.
_GPAI_SYSTEMIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b10\s*\^?\s*25\s*flops?\b", re.I),
    re.compile(r"\b10\s*\^?\s*2[5-9]\s*flops?\b", re.I),  # ≥10^25
    re.compile(r"\b(systemic[\s\-]risk|systemic\s+risk)\s+(model|ai|llm|gpai)\b", re.I),
    re.compile(r"\bfrontier\s+(model|llm|ai|foundation|multimodal)\b", re.I),
)


# Basic GPAI shape — Chapter V Art. 53/54 apply but NOT Art. 55 systemic.
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


def is_systemic_gpai(text: str) -> bool:
    """True if any Article 51 systemic-risk marker matches ``text``."""

    return any(p.search(text) for p in _GPAI_SYSTEMIC_PATTERNS)


def is_basic_gpai(text: str) -> bool:
    """True if any basic-GPAI shape marker matches ``text``."""

    return any(p.search(text) for p in _GPAI_BASIC_PATTERNS)
