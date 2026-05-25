"""Eval runners — single-node ``risk_triage`` and end-to-end full graph.

Both runners take a list of testset rows (dicts with ``id``,
``description``, ``expected_tier``, ``expected_articles``,
``expected_deadline``) and return per-row + aggregated metrics. They
don't know about argparse or Markdown — the CLI + report modules wire
those in around them.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from regpilot.agents.intake import intake_classifier
from regpilot.agents.triage import risk_triage
from regpilot.config import settings
from regpilot.evaluation.metrics import (
    aggregate,
    citation_precision,
    citation_recall,
    context_recall,
    deadline_match,
    extract_cited_articles,
    faithfulness,
    mrr,
    recall_at_k,
    retrieved_articles,
)
from regpilot.graph import build_main_graph
from regpilot.observability import request_context

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_testset(path: Path) -> list[dict]:
    """Read a JSONL testset; strip blank lines."""

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Single-node eval — risk_triage only
# --------------------------------------------------------------------------- #


def eval_triage_only(rows: Iterable[dict]) -> dict:
    """Run intake + risk_triage on each row; return accuracy + confusion matrix."""

    rows_list = list(rows)
    cm: dict[tuple[str, str], int] = Counter()
    correct = 0
    for r in rows_list:
        intake_state = intake_classifier({"user_input": r["description"]})
        triage_state = risk_triage(intake_state)
        pred = triage_state.get("risk_tier", "unknown")
        gold = r["expected_tier"]
        cm[(gold, pred)] += 1
        if pred == gold:
            correct += 1
    accuracy = correct / len(rows_list) if rows_list else 0.0
    return {"accuracy": accuracy, "confusion": cm}


# --------------------------------------------------------------------------- #
# End-to-end eval — full graph
# --------------------------------------------------------------------------- #


def eval_end_to_end(rows: Iterable[dict]) -> dict:
    """Invoke the full graph per row; collect every metric for aggregation.

    Binds each row's request-id contextvar to ``eval-<row_id>`` so log
    records produced during that row carry it for trivial multi-row log
    triage.
    """

    graph = build_main_graph()
    per_row: list[dict] = []
    for r in rows:
        t0 = time.perf_counter()
        rid = f"eval-{r['id']}"
        config: dict = {
            "configurable": {"thread_id": rid},
            "recursion_limit": settings.graph_recursion_limit,
        }
        with request_context(rid):
            state = graph.invoke(
                {"user_input": r["description"], "validator_loops": 0}, config=config
            )
        latency = time.perf_counter() - t0
        retrieved = retrieved_articles(state)
        cited = extract_cited_articles(
            state.get("final_report", "") or state.get("draft_report", "")
        )
        per_row.append(
            {
                "id": r["id"],
                "gold_tier": r["expected_tier"],
                "pred_tier": state.get("risk_tier", "unknown"),
                "latency_s": latency,
                "retrieval_recall_at_5": recall_at_k(retrieved, r["expected_articles"], k=5),
                "context_recall": context_recall(retrieved, r["expected_articles"]),
                "faithfulness": faithfulness(cited, retrieved),
                "mrr": mrr(retrieved, r["expected_articles"]),
                "citation_precision": citation_precision(cited, r["expected_articles"]),
                "citation_recall": citation_recall(cited, r["expected_articles"]),
                "deadline_match": deadline_match(state, r["expected_deadline"]),
                "n_retrieved": len(retrieved),
                "n_cited": len(cited),
                "n_obligations": len(state.get("obligations", []) or []),
                "validator_loops": state.get("validator_loops", 0),
            }
        )
    return {"per_row": per_row, "agg": aggregate(per_row)}
