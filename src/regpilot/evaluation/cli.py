"""Argparse entry point for the functional eval.

Thin orchestrator: parse arguments, load the testset, call the runner,
hand the results to the report writer, print a one-line summary, exit
non-zero if any threshold was missed (unless ``--no-fail``).

The single ``scripts/evaluate.py`` shim invokes :func:`main` here so
running the eval from the command line keeps the familiar invocation
``python scripts/evaluate.py …``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from regpilot.evaluation import THRESHOLDS
from regpilot.evaluation.report import results_path, write_report
from regpilot.evaluation.runner import eval_end_to_end, eval_triage_only, load_testset


def _default_testset() -> Path:
    """Resolve the default testset path at call time.

    Looked up in this order so the same default works from every supported
    invocation site:

    1. ``$CWD/evaluation/testset.jsonl`` — what ``docker exec regpilot-app
       regpilot-eval`` and ``python scripts/evaluate.py`` see when run
       from ``/app`` (the working dir baked into the image).
    2. ``<repo-root>/evaluation/testset.jsonl`` — what ``regpilot-eval``
       sees when invoked from an editable install inside the source repo
       (``pip install -e .``). For a regular wheel install this falls
       through to (1).
    """

    cwd_candidate = Path.cwd() / "evaluation" / "testset.jsonl"
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = Path(__file__).resolve().parents[3] / "evaluation" / "testset.jsonl"
    return repo_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RegPilot functional eval")
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="never exit non-zero on threshold miss",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=None,
        help="path to a testset JSONL file (default: evaluation/testset.jsonl)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="extra suffix on the results filename (e.g. 'extra' → results_<backend>_extra.md)",
    )
    args = parser.parse_args(argv)

    testset_path: Path = args.testset or _default_testset()
    rows = load_testset(testset_path)
    print(f"Loaded {len(rows)} gold questions from {testset_path}.")

    print("Running single-node eval on risk_triage…")
    triage = eval_triage_only(rows)
    print(f"  triage accuracy: {triage['accuracy']:.2%}")

    print("Running end-to-end eval on the full graph…")
    e2e = eval_end_to_end(rows)
    agg = e2e["agg"]
    for k in (
        "triage_accuracy",
        "context_recall",
        "faithfulness",
        "retrieval_recall_at_5",
        "citation_recall",
        "citation_precision",
        "deadline_exact_match",
    ):
        threshold = THRESHOLDS.get(k)
        flag = ""
        if threshold is not None:
            flag = " ✓" if agg[k] >= threshold else f" ✗ (target {threshold:.0%})"
        print(f"  {k}: {agg[k]:.2%}{flag}")
    print(f"  latency p50/p95: {agg['latency_p50']:.2f}s / {agg['latency_p95']:.2f}s")

    out_path = results_path(args.suffix)
    out_path.parent.mkdir(exist_ok=True)
    write_report(triage, e2e, out_path)
    print(f"Wrote {out_path}")

    misses = {k: agg[k] for k, t in THRESHOLDS.items() if agg[k] < t}
    if misses and not args.no_fail:
        print(f"FAIL — thresholds missed: {misses}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
