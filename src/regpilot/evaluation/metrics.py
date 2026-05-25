"""Pure-function metric definitions used by the eval runner.

Each metric takes one row's outputs (cited Articles, retrieved Articles,
gold lists, deadline strings) and returns a single float. All defined
to match the conventions documented in the report:

* ``context_recall`` / ``faithfulness`` — Ragas definitions.
* ``recall_at_k`` — BEIR / MS-MARCO normalisation (denominator
  ``min(k, |gold|)``).
* ``mrr``, ``citation_precision``, ``citation_recall``,
  ``deadline_match`` — straight definitions.

Aggregation helpers (mean, median, p95) live at the bottom.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Mapping
from typing import Any

CITE_RE = re.compile(r"Art\.\s*(\d+[a-z]?)", re.I)


# --------------------------------------------------------------------------- #
# Per-row extractors
# --------------------------------------------------------------------------- #


def extract_cited_articles(report: str) -> set[str]:
    """Pull the distinct ``Art. N`` numbers cited in a final report."""

    return {m.group(1) for m in CITE_RE.finditer(report)}


def retrieved_articles(state: Mapping[str, Any]) -> list[str]:
    """Distinct Article numbers in retrieved chunks, preserving rank order."""

    seen: list[str] = []
    for c in state.get("retrieved", []):
        art = (c.get("article") or "").strip()
        if art and art not in seen:
            seen.append(art)
    return seen


# --------------------------------------------------------------------------- #
# Per-row metric definitions
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved_arts: list[str], gold: list[str], k: int = 5) -> float:
    """Recall@k normalised by ``min(k, |gold|)``.

    Standard IR practice when ``|gold| > k``: the raw formula
    ``|top_k ∩ gold| / |gold|`` math-caps at ``k/|gold|`` (e.g. 5/12 ≈
    42% for our 12-Article high-risk gold). Normalising by
    ``min(k, |gold|)`` gives the well-defined coverage metric BEIR /
    MS-MARCO / Ragas use when the relevant set is larger than the
    retrieval budget. When ``|gold| <= k`` this is identical to raw
    recall@k.
    """

    if not gold:
        return 1.0
    top = set(retrieved_arts[:k])
    return len(top & set(gold)) / min(k, len(gold))


def mrr(retrieved_arts: list[str], gold: list[str]) -> float:
    """Mean reciprocal rank of the first gold Article in the retrieved list."""

    gold_set = set(gold)
    for i, a in enumerate(retrieved_arts):
        if a in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def citation_precision(cited: set[str], gold: list[str]) -> float:
    if not cited:
        return 0.0
    return len(cited & set(gold)) / len(cited)


def citation_recall(cited: set[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return len(cited & set(gold)) / len(gold)


def context_recall(retrieved_arts: list[str], gold: list[str]) -> float:
    """Ragas context recall — do the retrieved chunks cover the gold Articles,
    position-agnostic, not bounded by k?"""

    if not gold:
        return 1.0
    return len(set(retrieved_arts) & set(gold)) / len(gold)


def faithfulness(cited: set[str], retrieved_arts: list[str]) -> float:
    """Ragas faithfulness — are the Articles cited in the final report
    actually backed by chunks the synthesizer was given?

    High score = no hallucinated Article numbers. RegPilot's strongest
    guarantee against the worst RAG failure mode (an LLM inventing a
    plausible-looking but non-existent ``Art. 99``).
    """

    if not cited:
        return 0.0
    return len(cited & set(retrieved_arts)) / len(cited)


def deadline_match(state: Mapping[str, Any], expected: str) -> bool:
    """True iff at least one obligation in ``state`` lands on the expected
    Article 113 phase date."""

    items = state.get("deadlines", {}).get("items", [])
    if not items:
        return False
    return any(i.get("date") == expected for i in items)


# --------------------------------------------------------------------------- #
# Aggregators
# --------------------------------------------------------------------------- #


def percentile(xs: list[float], p: int) -> float:
    """Nearest-rank percentile (no interpolation). Returns 0.0 on empty."""

    if not xs:
        return 0.0
    xs = sorted(xs)
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def aggregate(per_row: list[dict]) -> dict:
    """Headline summary across rows — means + median latency + p95 latency."""

    n = len(per_row)

    def avg(key: str) -> float:
        return sum(r[key] for r in per_row) / n if n else 0.0

    return {
        "n": n,
        "triage_accuracy": (
            sum(1 for r in per_row if r["gold_tier"] == r["pred_tier"]) / n if n else 0.0
        ),
        "context_recall": avg("context_recall"),
        "faithfulness": avg("faithfulness"),
        "retrieval_recall_at_5": avg("retrieval_recall_at_5"),
        "mrr": avg("mrr"),
        "citation_precision": avg("citation_precision"),
        "citation_recall": avg("citation_recall"),
        "deadline_exact_match": avg("deadline_match"),
        "latency_p50": statistics.median(r["latency_s"] for r in per_row) if n else 0.0,
        "latency_p95": percentile([r["latency_s"] for r in per_row], 95) if n else 0.0,
    }
