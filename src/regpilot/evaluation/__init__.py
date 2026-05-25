"""Functional evaluation package.

Used to be ``scripts/evaluate.py`` (430-line file with CLI parsing, metric
math, and Markdown rendering all in one place). Split into focused modules:

* :mod:`regpilot.evaluation.metrics`  — pure-function metric definitions
  (Ragas-style context recall + faithfulness, BEIR-normalised Recall@5,
  citation precision/recall, MRR, deadline match) plus per-row aggregation.
* :mod:`regpilot.evaluation.runner`   — runs the testset rows through
  ``risk_triage`` (single-node) and the full graph (end-to-end).
* :mod:`regpilot.evaluation.report`   — writes the Markdown results file
  with confusion matrix, per-question breakdown, and commentary.
* :mod:`regpilot.evaluation.cli`      — argparse entry point invoked by
  ``scripts/evaluate.py``.

The thresholds (the only "configuration" the eval has) live here too,
since both the report writer and the CLI consume them.
"""

from __future__ import annotations

# Headline metric thresholds. The CLI exits non-zero if any of these is
# missed (unless ``--no-fail`` is passed). Same numbers used by the
# Markdown report renderer.
THRESHOLDS: dict[str, float] = {
    "triage_accuracy": 0.80,
    # Ragas-standard: do the gold Articles appear anywhere in the
    # retrieved+reranked context the synthesizer sees, regardless of rank?
    "context_recall": 0.90,
    # Ragas-standard: are the Articles cited in the final report all
    # backed by chunks the synthesizer actually saw?
    "faithfulness": 0.90,
    "citation_recall": 0.80,
    "citation_precision": 0.70,
    "deadline_exact_match": 0.80,
    # Position-sensitive, normalised per BEIR / MS-MARCO convention so it
    # isn't math-capped when |gold| > k. See metrics.recall_at_k.
    "retrieval_recall_at_5": 0.90,
}


__all__ = ["THRESHOLDS"]
