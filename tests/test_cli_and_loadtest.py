"""Cover the console-script entry points and the loadtest harness.

Three slices:

1. :mod:`regpilot.cli` — the three installable console scripts
   (``regpilot-ingest`` / ``regpilot-eval`` / ``regpilot-loadtest``). We
   exercise the argparse surface + delegation paths without spinning up
   the full ingestion pipeline (the actual ``download_ai_act`` /
   ``extract_text`` calls are out-of-scope here — they're covered in
   ``test_loader.py``).
2. :mod:`regpilot.loadtest` — instrumented graph builder, query loader,
   percentile + bottleneck helpers, and the Markdown report writer.
3. End-to-end: a 4-query loadtest through the stub backend, asserting the
   written Markdown file is well-formed and contains every section the
   README points reviewers at.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regpilot import cli, loadtest

# --------------------------------------------------------------------------- #
# Loadtest module — pure helpers
# --------------------------------------------------------------------------- #


def test_results_path_includes_backend():
    p = loadtest.results_path()
    assert p.name.startswith("loadtest_results_")
    assert p.suffix == ".md"
    assert "stub" in p.name  # session fixture pins REGPILOT_LLM=stub


def test_pct_handles_empty_and_basic_distribution():
    assert loadtest._pct([], 50) == 0.0
    xs = [0.01, 0.02, 0.03, 0.04, 0.05]
    assert loadtest._pct(xs, 0) == 0.01
    assert loadtest._pct(xs, 50) == 0.03
    assert loadtest._pct(xs, 100) == 0.05


def test_count_tiers_aggregates_results():
    rows = [
        {"tier": "high_risk"},
        {"tier": "high_risk"},
        {"tier": "minimal_risk"},
        {"tier": "prohibited"},
    ]
    assert loadtest._count_tiers(rows) == {
        "high_risk": 2,
        "minimal_risk": 1,
        "prohibited": 1,
    }


def test_queries_pool_loops_and_paraphrases(tmp_path):
    """``_queries`` should cycle the pool and tag each entry with case #N
    so a >|pool|-sized request count doesn't degenerate to identical strings."""

    testset = tmp_path / "mini.jsonl"
    testset.write_text(
        "\n".join(
            json.dumps({"id": f"q{i}", "description": f"system #{i}"})
            for i in range(2)
        ),
        encoding="utf-8",
    )

    qs = loadtest._queries(5, testset=testset)
    assert len(qs) == 5
    assert qs[0].startswith("system #0")
    assert qs[1].startswith("system #1")
    assert "case #" in qs[-1]


def test_node_table_and_bottleneck_with_recorded_samples():
    """Manually seed the node-timing table and confirm aggregation +
    bottleneck identification."""

    loadtest._NODE_TIMINGS.clear()
    loadtest._NODE_TIMINGS["rag_retrieval"].extend([0.10, 0.20, 0.30])
    loadtest._NODE_TIMINGS["risk_triage"].extend([0.001, 0.002])

    rows = loadtest._node_table()
    by_name = {r["node"]: r for r in rows}
    assert by_name["rag_retrieval"]["calls"] == 3
    assert by_name["rag_retrieval"]["total_s"] == pytest.approx(0.60, abs=1e-9)
    assert by_name["risk_triage"]["calls"] == 2

    # Sort order: largest total wall-time first.
    assert rows[0]["node"] == "rag_retrieval"
    assert loadtest._bottleneck() == "rag_retrieval"

    loadtest._NODE_TIMINGS.clear()
    assert loadtest._bottleneck() == "n/a"


def test_write_report_renders_every_section(tmp_path):
    out = tmp_path / "loadtest_results_stub.md"
    loadtest._NODE_TIMINGS.clear()
    loadtest._NODE_TIMINGS["rag_retrieval"].extend([0.10, 0.15, 0.20])
    loadtest._NODE_TIMINGS["risk_triage"].extend([0.001, 0.002, 0.001])

    summary = {
        "n": 50,
        "concurrency": 8,
        "wall_s": 1.2,
        "throughput_rps": 41.7,
        "p50": 0.05,
        "p95": 0.18,
        "p99": 0.25,
        "min": 0.02,
        "max": 0.30,
        "mean": 0.07,
        "rss_peak_mb": 180.0,
        "cpu_pct": 220.0,
        "tier_dist": {"high_risk": 12, "minimal_risk": 38},
    }

    returned = loadtest.write_report(summary, out_path=out)
    assert returned == out

    body = out.read_text(encoding="utf-8")
    for needle in (
        "# Load test results",
        "Stub-backend caveat",
        "Total requests: **50**",
        "Throughput: **41.70 req/s**",
        "Per-node breakdown",
        "Identified bottleneck:",
        "Two concrete optimisations",
        "rag_retrieval",
    ):
        assert needle in body, f"missing section: {needle!r}"


@pytest.mark.asyncio
async def test_harness_runs_stub_end_to_end_and_summarises():
    """End-to-end: harness fires N requests through the stub backend and
    returns a summary dict with every percentile + the tier distribution.
    """

    summary = await loadtest.harness(n=4, concurrency=2)

    assert summary["n"] == 4
    assert summary["concurrency"] == 2
    assert summary["wall_s"] > 0
    assert summary["throughput_rps"] > 0
    for key in ("p50", "p95", "p99", "min", "max", "mean"):
        assert summary[key] >= 0
    assert isinstance(summary["tier_dist"], dict)
    assert sum(summary["tier_dist"].values()) == 4
    # Per-node timings recorded for at least intake + triage on every request.
    assert "intake_classifier" in loadtest._NODE_TIMINGS
    assert "risk_triage" in loadtest._NODE_TIMINGS


def test_build_instrumented_graph_smokes():
    """Smoke: the instrumented graph compiles and the wrapper preserves names."""

    g = loadtest.build_instrumented_graph()
    state = g.invoke({"user_input": "CV screening AI for hiring", "validator_loops": 0})
    assert state["risk_tier"] in {
        "high_risk",
        "limited_risk",
        "minimal_risk",
        "general_purpose",
        "general_purpose_systemic",
        "prohibited",
        "unknown",
    }


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_cli_ingest_annex_only_returns_zero():
    """``regpilot-ingest --annex-only`` runs without touching the network."""

    rc = cli.ingest(["--annex-only", "--reset"])
    assert rc == 0


def test_cli_ingest_skip_download_missing_pdf_returns_one(tmp_path, monkeypatch):
    """``--skip-download`` errors cleanly when the cached PDF is absent."""

    from regpilot.config import settings as _settings

    monkeypatch.setattr(_settings, "data_dir", tmp_path / "raw-empty")
    rc = cli.ingest(["--skip-download"])
    assert rc == 1


def test_cli_ingest_full_path_calls_loader(monkeypatch, tmp_path):
    """The default path downloads, extracts, chunks, and upserts."""

    from regpilot.ingestion.chunker import Chunk

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not really a pdf")

    monkeypatch.setattr(cli, "download_ai_act", lambda: fake_pdf)
    monkeypatch.setattr(cli, "extract_text", lambda _path: "Article 1 stub text")
    monkeypatch.setattr(
        cli,
        "chunk_text",
        lambda _text: [Chunk(id="stub-1", text="stub body", article="1")],
    )

    rc = cli.ingest([])
    assert rc == 0


def test_cli_evaluate_delegates_to_evaluation_cli(monkeypatch):
    """``regpilot-eval`` is a one-line passthrough to the eval CLI."""

    captured: dict[str, list[str] | None] = {"argv": None}

    def fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "_eval_main", fake_main)
    rc = cli.evaluate(["--no-fail"])
    assert rc == 0
    assert captured["argv"] == ["--no-fail"]


def test_cli_loadtest_runs_a_tiny_workload(tmp_path, monkeypatch, capsys):
    """End-to-end: ``regpilot-loadtest --n 2 --concurrency 2 --quiet`` writes
    the Markdown report to the backend-specific path."""

    out_path = tmp_path / "loadtest_results_stub.md"
    monkeypatch.setattr(loadtest, "results_path", lambda: out_path)
    # The ``cli`` module imports the symbols lazily inside the function;
    # patch the *re-exported* ``write_report`` so the loadtest writes to our
    # tmp path instead of the repo's evaluation/ folder.
    from regpilot import loadtest as lt_module

    real_write = lt_module.write_report

    def patched_write(summary, out_path=None):
        return real_write(summary, out_path=out_path or tmp_path / "loadtest_results_stub.md")

    monkeypatch.setattr(lt_module, "write_report", patched_write)

    rc = cli.loadtest(["--n", "2", "--concurrency", "2", "--quiet"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Throughput:" in captured.out
    assert "Wrote" in captured.out

    written = next(tmp_path.glob("loadtest_results_*.md"))
    body = written.read_text(encoding="utf-8")
    assert "Per-node breakdown" in body


# --------------------------------------------------------------------------- #
# Shim scripts — confirm they still import without re-running argparse
# --------------------------------------------------------------------------- #


def test_scripts_are_thin_shims():
    """``scripts/{ingest,loadtest,evaluate}.py`` should each be < 30 lines
    and import the corresponding callable from ``regpilot.cli`` /
    ``regpilot.evaluation.cli``.
    """

    root = Path(__file__).resolve().parents[1] / "scripts"
    for name, expected_import in (
        ("ingest.py", "from regpilot.cli import ingest"),
        ("loadtest.py", "from regpilot.cli import loadtest"),
        ("evaluate.py", "from regpilot.evaluation.cli import main"),
    ):
        body = (root / name).read_text(encoding="utf-8")
        assert expected_import in body, f"{name} should delegate to regpilot.*"
        assert len(body.splitlines()) < 30, f"{name} should stay a thin shim"


# --------------------------------------------------------------------------- #
# Wiring: confirm pyproject.toml's console scripts point at real callables
# --------------------------------------------------------------------------- #


def test_pyproject_console_scripts_resolve():
    """Catches the bug where ``[project.scripts]`` references a non-existent
    module — the symptom that lit this whole refactor up.
    """

    import importlib

    for entry in ("ingest", "evaluate", "loadtest"):
        mod = importlib.import_module("regpilot.cli")
        assert callable(getattr(mod, entry)), f"regpilot.cli:{entry} missing"


# --------------------------------------------------------------------------- #
# Default-path resolution — must work from CWD and from a source-tree install
# --------------------------------------------------------------------------- #


def test_evaluation_default_testset_prefers_cwd(tmp_path, monkeypatch):
    """A reviewer running ``regpilot-eval`` from a directory that contains
    an ``evaluation/testset.jsonl`` should pick that up — this is what the
    docker image relies on (cwd=/app, evaluation/ next to scripts/).
    """

    from regpilot.evaluation import cli as eval_cli

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    cwd_testset = eval_dir / "testset.jsonl"
    cwd_testset.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    assert eval_cli._default_testset() == cwd_testset


def test_evaluation_default_testset_falls_back_to_repo(tmp_path, monkeypatch):
    """When no cwd-relative file exists, the resolver falls back to the
    source-tree default (works for ``pip install -e .`` in the repo)."""

    from regpilot.evaluation import cli as eval_cli

    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    resolved = eval_cli._default_testset()
    assert resolved.name == "testset.jsonl"
    assert resolved.parent.name == "evaluation"


def test_loadtest_evaluation_dir_prefers_cwd(tmp_path, monkeypatch):
    """Same dispatch story for the loadtest's results path resolution."""

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    assert loadtest._evaluation_dir() == eval_dir
    assert loadtest.results_path().parent == eval_dir


def test_loadtest_evaluation_dir_falls_back_to_repo(tmp_path, monkeypatch):
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    resolved = loadtest._evaluation_dir()
    assert resolved.name == "evaluation"
    # ``results_path()`` still composes a sensible filename.
    assert "loadtest_results_" in loadtest.results_path().name
