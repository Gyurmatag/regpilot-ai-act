"""Semantic-similarity Annex III matcher.

Each Annex III area's canonical description is embedded once per process
(lazy + thread-safe), the user's input is embedded on every call, and we
score them by cosine similarity. The output is the list of areas above
``settings.semantic_match_threshold``, sorted descending — fed into the
LLM prompt as candidate priors.

This replaces the hand-written verb-form regex chase the classifier used
to do for Annex III categories. The semantic matcher generalises across
phrasings: "traffic light timing" matches *Critical infrastructure*,
"legal research for judges" matches *Administration of justice*, with no
new regex needed when the next user description rolls in.
"""

from __future__ import annotations

import logging
import math
from threading import Lock

from regpilot.ingestion.annex import ANNEX_III
from regpilot.llm import LLMClient

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Canonical Annex III area descriptions
# --------------------------------------------------------------------------- #


# Concrete example phrasings appended to each area's official description
# so the embedding captures the *conceptual breadth* of each area, not
# just the legal text. The legal description alone is too sparse for
# `nomic-embed-text` to match the kinds of paraphrases real users type.
_ANNEX_AREA_EXAMPLES: dict[str, str] = {
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


def _build_annex_corpus() -> list[tuple[str, str]]:
    """Return ``(area_name, embed_text)`` tuples for every Annex III area."""

    return [
        (
            e.area,
            f"Annex III area: {e.area}. {e.description} "
            f"{_ANNEX_AREA_EXAMPLES.get(e.area, '')}",
        )
        for e in ANNEX_III
    ]


# --------------------------------------------------------------------------- #
# Embedding cache — lazy, thread-safe, process-wide
# --------------------------------------------------------------------------- #


_ANNEX_EMB_CACHE: list[tuple[str, list[float]]] | None = None
_ANNEX_EMB_LOCK = Lock()


def _get_annex_embeddings(llm: LLMClient) -> list[tuple[str, list[float]]]:
    """Lazily compute + cache one embedding per Annex III area."""

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
    """Force the next call to recompute the Annex III embeddings.

    Used by tests to swap in a deterministic embedder and assert against
    its output; in production this is a no-op (cache is rebuilt on first
    use after process restart).
    """

    global _ANNEX_EMB_CACHE
    _ANNEX_EMB_CACHE = None


# --------------------------------------------------------------------------- #
# Cosine similarity + match query
# --------------------------------------------------------------------------- #


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 on empty / mismatched / zero-length inputs (instead of
    raising) so callers don't have to guard every call site.
    """

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def semantic_annex_matches(
    text: str, llm: LLMClient, threshold: float
) -> list[tuple[str, float]]:
    """Return ``(area_name, cosine_score)`` pairs sorted by descending score,
    filtered to those above ``threshold``. Empty list on empty input or
    embedder failure (defensive — never raises in the classifier hot path)."""

    index = _get_annex_embeddings(llm)
    if not index or not text.strip():
        return []
    try:
        query_vec = llm.embed([text])[0]
    except Exception as exc:
        logger.warning("Query embedding failed in semantic matcher: %s", exc)
        return []
    scored = [(area, cosine(query_vec, vec)) for area, vec in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(a, s) for a, s in scored if s >= threshold]
