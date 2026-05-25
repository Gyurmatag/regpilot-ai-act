"""Tests for the pure-function eval metrics.

These are the contract every eval result file depends on; if any of them
drift, the historical results across testsets stop comparing apples to
apples.
"""

from __future__ import annotations

import math

import pytest

from regpilot.evaluation.metrics import (
    aggregate,
    citation_precision,
    citation_recall,
    context_recall,
    deadline_match,
    extract_cited_articles,
    faithfulness,
    mrr,
    percentile,
    recall_at_k,
    retrieved_articles,
)

# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #


def test_extract_cited_articles_finds_distinct_numbers() -> None:
    report = "See Art. 9, Art. 10 and Art. 9 again. Article 99 isn't matched."
    cited = extract_cited_articles(report)
    assert cited == {"9", "10"}


def test_extract_cited_articles_handles_lowercase_letter_suffix() -> None:
    cited = extract_cited_articles("Citation: Art. 27a applies here.")
    assert cited == {"27a"}


def test_retrieved_articles_preserves_rank_dedupes() -> None:
    state = {
        "retrieved": [
            {"article": "9"},
            {"article": "10"},
            {"article": "9"},   # duplicate — should be skipped
            {"article": ""},    # blank — should be skipped
            {"article": "11"},
        ]
    }
    assert retrieved_articles(state) == ["9", "10", "11"]


# --------------------------------------------------------------------------- #
# recall_at_k (BEIR / MS-MARCO normalisation)
# --------------------------------------------------------------------------- #


def test_recall_at_k_normalises_when_gold_larger_than_k() -> None:
    """When |gold| > k, the denominator is min(k, |gold|) = k.

    Five gold articles, top-5 retrieved covers exactly the first five
    → numerator 5, denominator min(5, 12) = 5 → 100%. The raw recall@5
    formula would give 5/12 ≈ 42%.
    """

    gold = ["9", "10", "11", "12", "13", "14", "15", "17", "18", "43", "49", "72"]
    retrieved = ["9", "10", "11", "12", "13"] + ["99", "98"]
    assert recall_at_k(retrieved, gold, k=5) == pytest.approx(1.0)


def test_recall_at_k_matches_raw_recall_when_gold_le_k() -> None:
    gold = ["50"]
    assert recall_at_k(["50", "9", "10", "11", "12"], gold, k=5) == pytest.approx(1.0)
    assert recall_at_k(["99", "9", "10", "11", "12"], gold, k=5) == pytest.approx(0.0)


def test_recall_at_k_empty_gold_is_one() -> None:
    assert recall_at_k(["9", "10"], [], k=5) == 1.0


# --------------------------------------------------------------------------- #
# MRR, citation precision/recall, context recall, faithfulness
# --------------------------------------------------------------------------- #


def test_mrr_returns_reciprocal_of_first_hit() -> None:
    assert mrr(["99", "9", "10"], ["9"]) == pytest.approx(0.5)
    assert mrr(["9", "10"], ["9"]) == pytest.approx(1.0)
    assert mrr(["99", "98"], ["9"]) == 0.0


def test_citation_precision_basic() -> None:
    assert citation_precision({"9", "10"}, ["9", "10", "11"]) == pytest.approx(1.0)
    assert citation_precision({"9", "99"}, ["9", "10", "11"]) == pytest.approx(0.5)
    assert citation_precision(set(), ["9"]) == 0.0


def test_citation_recall_basic() -> None:
    assert citation_recall({"9", "10"}, ["9", "10"]) == pytest.approx(1.0)
    assert citation_recall({"9"}, ["9", "10", "11"]) == pytest.approx(1 / 3)
    assert citation_recall({"9"}, []) == 1.0


def test_context_recall_position_agnostic() -> None:
    # All gold articles appear somewhere in retrieved (positions don't matter).
    assert context_recall(["99", "98", "9", "10"], ["9", "10"]) == pytest.approx(1.0)
    assert context_recall(["99"], ["9", "10"]) == pytest.approx(0.0)


def test_faithfulness_catches_hallucinated_citation() -> None:
    """If the report cites Art. 99 but it isn't in the retrieved chunks,
    faithfulness drops to reflect the hallucination."""

    assert faithfulness({"9", "10"}, ["9", "10", "11"]) == pytest.approx(1.0)
    # Half of the citations are unsupported.
    assert faithfulness({"9", "99"}, ["9", "10"]) == pytest.approx(0.5)
    assert faithfulness(set(), ["9"]) == 0.0


# --------------------------------------------------------------------------- #
# deadline_match
# --------------------------------------------------------------------------- #


def test_deadline_match_compares_expected_to_state_items() -> None:
    state = {"deadlines": {"items": [{"article": "Art. 9", "date": "2026-08-02"}]}}
    assert deadline_match(state, "2026-08-02") is True
    assert deadline_match(state, "2027-08-02") is False
    assert deadline_match({"deadlines": {"items": []}}, "2026-08-02") is False


# --------------------------------------------------------------------------- #
# Aggregators
# --------------------------------------------------------------------------- #


def test_percentile_returns_zero_for_empty_input() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_nearest_rank() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Median (50th percentile) of a 5-element list = the middle one.
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 95) == 5.0
    assert percentile(xs, 0) == 1.0


def test_aggregate_computes_means_and_latency_percentiles() -> None:
    per_row = [
        {
            "gold_tier": "high_risk", "pred_tier": "high_risk",
            "context_recall": 1.0, "faithfulness": 1.0,
            "retrieval_recall_at_5": 1.0, "mrr": 1.0,
            "citation_precision": 1.0, "citation_recall": 1.0,
            "deadline_match": True,
            "latency_s": 0.1,
        },
        {
            "gold_tier": "minimal_risk", "pred_tier": "high_risk",
            "context_recall": 0.0, "faithfulness": 0.5,
            "retrieval_recall_at_5": 0.0, "mrr": 0.0,
            "citation_precision": 0.0, "citation_recall": 0.0,
            "deadline_match": False,
            "latency_s": 0.5,
        },
    ]
    agg = aggregate(per_row)
    assert agg["n"] == 2
    assert agg["triage_accuracy"] == pytest.approx(0.5)
    assert agg["context_recall"] == pytest.approx(0.5)
    assert agg["faithfulness"] == pytest.approx(0.75)
    assert agg["deadline_exact_match"] == pytest.approx(0.5)
    assert math.isfinite(agg["latency_p50"])
    assert math.isfinite(agg["latency_p95"])


def test_aggregate_handles_empty_input() -> None:
    agg = aggregate([])
    assert agg["n"] == 0
    assert agg["triage_accuracy"] == 0.0
    assert agg["latency_p50"] == 0.0
