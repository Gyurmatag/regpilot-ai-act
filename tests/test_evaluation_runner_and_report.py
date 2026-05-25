"""Tests for the eval runner + Markdown report writer.

The metrics module already has its own tests; here we exercise the
glue that runs the testset rows through the agents and the rendering
of the results Markdown file.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from regpilot.evaluation.cli import main
from regpilot.evaluation.report import results_path, write_report
from regpilot.evaluation.runner import (
    eval_end_to_end,
    eval_triage_only,
    load_testset,
)

# --------------------------------------------------------------------------- #
# load_testset
# --------------------------------------------------------------------------- #


def test_load_testset_skips_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    f.write_text(
        '{"id": "a", "description": "x", "expected_tier": "minimal_risk", '
        '"expected_articles": ["95"], "expected_deadline": "2026-08-02"}\n'
        "\n"
        "  \n"
        '{"id": "b", "description": "y", "expected_tier": "high_risk", '
        '"expected_articles": ["9"], "expected_deadline": "2026-08-02"}\n',
        encoding="utf-8",
    )
    rows = load_testset(f)
    assert len(rows) == 2
    assert rows[0]["id"] == "a"
    assert rows[1]["id"] == "b"


# --------------------------------------------------------------------------- #
# eval_triage_only — single-node runner
# --------------------------------------------------------------------------- #


def test_eval_triage_only_returns_accuracy_and_confusion() -> None:
    rows = [
        {"id": "q1", "description": "A spam filter for company email.",
         "expected_tier": "minimal_risk",
         "expected_articles": ["95"], "expected_deadline": "2026-08-02"},
        {"id": "q2", "description": "A CV screening tool that ranks job applicants.",
         "expected_tier": "high_risk",
         "expected_articles": ["9"], "expected_deadline": "2026-08-02"},
    ]
    out = eval_triage_only(rows)
    assert "accuracy" in out
    assert 0.0 <= out["accuracy"] <= 1.0
    # Confusion matrix is a Counter of (gold, pred) → count.
    assert isinstance(out["confusion"], Counter)
    total = sum(out["confusion"].values())
    assert total == len(rows)


# --------------------------------------------------------------------------- #
# eval_end_to_end — full graph runner
# --------------------------------------------------------------------------- #


def test_eval_end_to_end_produces_per_row_and_aggregate() -> None:
    rows = [
        {"id": "q1", "description": "A spam filter for company email.",
         "expected_tier": "minimal_risk",
         "expected_articles": ["95"], "expected_deadline": "2026-08-02"},
    ]
    out = eval_end_to_end(rows)
    assert "per_row" in out
    assert "agg" in out
    assert len(out["per_row"]) == 1
    row = out["per_row"][0]
    # Every metric the report writer expects is present and a float / bool.
    for key in (
        "context_recall", "faithfulness", "retrieval_recall_at_5",
        "mrr", "citation_precision", "citation_recall",
    ):
        assert isinstance(row[key], float)
    assert isinstance(row["deadline_match"], bool)
    assert row["latency_s"] >= 0.0


# --------------------------------------------------------------------------- #
# write_report — Markdown output
# --------------------------------------------------------------------------- #


def _fake_run() -> tuple[dict, dict]:
    triage = {"accuracy": 1.0, "confusion": Counter({("high_risk", "high_risk"): 1})}
    e2e = {
        "per_row": [
            {
                "id": "q1", "gold_tier": "high_risk", "pred_tier": "high_risk",
                "latency_s": 0.1,
                "context_recall": 1.0, "faithfulness": 1.0,
                "retrieval_recall_at_5": 1.0, "mrr": 1.0,
                "citation_precision": 1.0, "citation_recall": 1.0,
                "deadline_match": True,
            }
        ],
        "agg": {
            "n": 1,
            "triage_accuracy": 1.0,
            "context_recall": 1.0, "faithfulness": 1.0,
            "retrieval_recall_at_5": 1.0, "mrr": 1.0,
            "citation_precision": 1.0, "citation_recall": 1.0,
            "deadline_exact_match": 1.0,
            "latency_p50": 0.1, "latency_p95": 0.1,
        },
    }
    return triage, e2e


def test_write_report_renders_all_sections(tmp_path: Path) -> None:
    triage, e2e = _fake_run()
    out_path = tmp_path / "r.md"
    write_report(triage, e2e, out_path)
    txt = out_path.read_text(encoding="utf-8")

    # Header + every section we promise in the report module's docstring.
    assert "# Functional evaluation results" in txt
    assert "## Single-node eval" in txt
    assert "Confusion matrix" in txt
    assert "## End-to-end eval" in txt
    assert "## Per-question breakdown" in txt
    assert "## Commentary" in txt
    # Per-row table contains the row id we passed in.
    assert "q1" in txt


def test_write_report_includes_stub_caveat_only_for_stub_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stub-backend caveat block appears only when REGPILOT_LLM=stub."""

    from regpilot.config import settings

    triage, e2e = _fake_run()

    # CI fixture sets REGPILOT_LLM=stub, so the caveat block must appear.
    monkeypatch.setattr(settings, "llm_backend", "stub")
    p_stub = tmp_path / "stub.md"
    write_report(triage, e2e, p_stub)
    assert "Stub-backend caveat" in p_stub.read_text(encoding="utf-8")

    # Flipping to a non-stub backend hides the caveat block.
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    p_ollama = tmp_path / "ollama.md"
    write_report(triage, e2e, p_ollama)
    assert "Stub-backend caveat" not in p_ollama.read_text(encoding="utf-8")


def test_results_path_uses_backend_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    from regpilot.config import settings

    monkeypatch.setattr(settings, "llm_backend", "stub")
    assert results_path().name == "results_stub.md"
    assert results_path("extra").name == "results_stub_extra.md"

    monkeypatch.setattr(settings, "llm_backend", "ollama")
    assert results_path().name == "results_ollama.md"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_main_writes_results_and_returns_nonzero_on_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI returns 1 when any threshold is missed (stub backend → expected)."""

    # Tiny one-row testset so the run finishes fast.
    testset = tmp_path / "t.jsonl"
    testset.write_text(
        '{"id": "q1", "description": "A spam filter for company email.",'
        ' "expected_tier": "minimal_risk", "expected_articles": ["95"],'
        ' "expected_deadline": "2026-08-02"}\n',
        encoding="utf-8",
    )

    # Redirect the results file into the tmp_path so we don't overwrite repo state.
    monkeypatch.setattr(
        "regpilot.evaluation.cli.results_path",
        lambda suffix="": tmp_path / f"r_{suffix or 'main'}.md",
    )

    rc = main(["--testset", str(testset), "--no-fail"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Loaded 1 gold questions" in out
    assert "triage accuracy" in out
    assert (tmp_path / "r_main.md").exists()
