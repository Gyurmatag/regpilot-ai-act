"""Markdown report writer for the functional eval.

Renders one ``results_<backend>[_<suffix>].md`` per run with:

* A header carrying the backend / models / testset size.
* A stub-backend caveat block (only when ``REGPILOT_LLM=stub``).
* The single-node triage accuracy + confusion matrix.
* The end-to-end metric table + per-question breakdown.
* A short prose commentary on every metric.

Pure rendering — takes the dicts produced by :mod:`runner` and a
``Path``; never reads the testset or invokes the graph.
"""

from __future__ import annotations

from pathlib import Path

from regpilot.config import settings
from regpilot.evaluation import THRESHOLDS

# --------------------------------------------------------------------------- #
# Path helper
# --------------------------------------------------------------------------- #


_ROOT = Path(__file__).resolve().parents[3]  # …/regpilot-ai-act


def results_path(suffix: str = "") -> Path:
    """Backend-specific results filename so stub / Ollama / hosted runs
    don't overwrite each other. ``suffix`` distinguishes alternate
    testsets (e.g. ``extra`` → ``results_<backend>_extra.md``)."""

    extra = f"_{suffix}" if suffix else ""
    return _ROOT / "evaluation" / f"results_{settings.llm_backend}{extra}.md"


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #


def write_report(triage: dict, e2e: dict, out_path: Path | None = None) -> None:
    """Render the eval results to a Markdown file."""

    out_path = out_path or results_path()
    agg = e2e["agg"]
    cm = triage["confusion"]
    tiers = ["prohibited", "high_risk", "limited_risk", "minimal_risk"]

    lines: list[str] = []
    lines.append("# Functional evaluation results\n")
    lines.append(
        f"Backend: `{settings.llm_backend}` — Chat model: `{settings.chat_model}` — "
        f"Embed model: `{settings.embed_model}` — Testset: {agg['n']} questions.\n"
    )
    if settings.is_stub:
        lines.append(
            "> ⚠️ **Stub-backend caveat.** The stub LLM uses hash-based pseudo-"
            "embeddings, so the semantic-similarity Annex III matcher (Option C) "
            "can't surface relevant areas, and end-to-end retrieval metrics are "
            "degraded by design. The stub run still validates classifier wiring, "
            "graph assembly, and the deterministic regulatory layer (deadline "
            "calculator, Article 5 bright-line rules) — useful as a smoke test, "
            "not as a quality benchmark. For real metrics see "
            "[`results_ollama.md`](results_ollama.md).\n"
        )

    lines.append("## Single-node eval — `risk_triage`\n")
    lines.append(
        f"**Triage accuracy: {triage['accuracy']:.2%}** "
        f"(threshold {THRESHOLDS['triage_accuracy']:.0%})\n"
    )
    lines.append("\nConfusion matrix (rows = gold, columns = predicted):\n")
    header = "| gold \\ predicted | " + " | ".join(tiers) + " |"
    sep = "| " + " | ".join(["---"] * (len(tiers) + 1)) + " |"
    lines.append(header)
    lines.append(sep)
    for gold in tiers:
        cells = [str(cm.get((gold, pred), 0)) for pred in tiers]
        lines.append(f"| {gold} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## End-to-end eval — full workflow\n")
    lines.append("| Metric | Value | Threshold | Pass |")
    lines.append("| --- | --- | --- | --- |")
    for metric, threshold in THRESHOLDS.items():
        actual = agg[metric]
        ok = actual >= threshold
        lines.append(
            f"| {metric} | {actual:.2%} | {threshold:.0%} | {'yes' if ok else 'NO'} |"
        )
    lines.append(f"| MRR | {agg['mrr']:.3f} | — | — |")
    lines.append(f"| latency p50 (s) | {agg['latency_p50']:.2f} | — | — |")
    lines.append(f"| latency p95 (s) | {agg['latency_p95']:.2f} | — | — |")
    lines.append("")

    lines.append("## Per-question breakdown\n")
    lines.append(
        "| id | gold | pred | ctx recall | R@5 | MRR | cite prec | cite recall | deadline | lat s |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in e2e["per_row"]:
        lines.append(
            f"| {r['id']} | {r['gold_tier']} | {r['pred_tier']} | "
            f"{r['context_recall']:.2f} | "
            f"{r['retrieval_recall_at_5']:.2f} | "
            f"{r['mrr']:.2f} | "
            f"{r['citation_precision']:.2f} | "
            f"{r['citation_recall']:.2f} | "
            f"{'yes' if r['deadline_match'] else 'no'} | "
            f"{r['latency_s']:.2f} |"
        )

    lines.append("")
    lines.append("## Commentary")
    lines.append(_commentary(agg, triage))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _commentary(agg: dict, triage: dict) -> str:
    """Render the post-eval commentary block in plain English.

    Reads like a notes file, not a marketing one-pager: short sentences,
    minimal bold, no buzzwords. Threshold-aware so the tone matches the
    actual numbers (no "exceeds expectations" when we just barely cleared).
    """

    fragments: list[str] = []

    if triage["accuracy"] >= THRESHOLDS["triage_accuracy"]:
        fragments.append(
            f"- Triage accuracy ({triage['accuracy']:.0%}) clears the "
            f"{THRESHOLDS['triage_accuracy']:.0%} threshold. Misses, when they "
            "show up, tend to cluster around the limited / minimal-risk boundary."
        )
    else:
        fragments.append(
            f"- Triage accuracy ({triage['accuracy']:.0%}) is below the "
            f"{THRESHOLDS['triage_accuracy']:.0%} threshold. The confusion "
            "matrix above shows which tier is being confused with which."
        )

    fragments.append(
        f"- Context recall ({agg['context_recall']:.0%}, target "
        f"{THRESHOLDS['context_recall']:.0%}) is the headline retrieval "
        "number — Ragas definition: how many gold Articles appear anywhere "
        "in the retrieved context, position-agnostic and not bounded by k. "
        "See https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html."
    )
    fragments.append(
        f"- Retrieval Recall@5 ({agg['retrieval_recall_at_5']:.0%}, target "
        f"{THRESHOLDS['retrieval_recall_at_5']:.0%}) uses the BEIR / MS-MARCO "
        "normalisation `|top5 ∩ gold| / min(5, |gold|)` so it isn't math-capped "
        "when `|gold| > k`."
    )
    fragments.append(
        f"- Citation recall ({agg['citation_recall']:.0%}) is the share of the "
        "gold Articles that actually end up cited in the final report — the "
        "most user-facing number, since a missed citation means a missed "
        "obligation."
    )
    fragments.append(
        f"- Citation precision ({agg['citation_precision']:.0%}) is the share "
        "of cited Articles that are in the gold list. We don't gate hard on "
        "this — the retriever legitimately surfaces adjacent Articles that "
        "are useful context but aren't in the narrow gold set."
    )
    fragments.append(
        f"- Median latency {agg['latency_p50']:.1f}s, p95 "
        f"{agg['latency_p95']:.1f}s. Stub backend reflects pipeline-only cost; "
        "with live Ollama qwen2.5:3b on CPU expect 50–100× slower."
    )
    return "\n".join(fragments)
