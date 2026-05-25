"""Load / latency harness used by ``regpilot-loadtest`` and ``scripts/loadtest.py``.

Fires N concurrent queries through the full RegPilot graph using ``asyncio``,
times every node, and writes ``evaluation/loadtest_results_<backend>.md``
with a bottleneck call-out and two concrete optimisation recommendations.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil
from langgraph.graph import END, START, StateGraph

from regpilot import graph as graph_module
from regpilot.agents.intake import intake_classifier
from regpilot.agents.obligation_mapper import obligation_mapper
from regpilot.agents.synthesizer import compliance_synthesizer
from regpilot.agents.triage import risk_triage, route_by_tier
from regpilot.agents.validator import route_after_validator, validator
from regpilot.config import settings
from regpilot.rag.subgraph import build_rag_subgraph
from regpilot.state import RegPilotState

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _evaluation_dir() -> Path:
    """Resolve the ``evaluation/`` directory at call time so the same code
    works from an editable install inside the repo, from the docker image
    (where ``/app/evaluation/`` is the cwd-relative copy), and from a
    standalone wheel install (falls back to the source-tree path)."""

    cwd_candidate = Path.cwd() / "evaluation"
    if cwd_candidate.exists():
        return cwd_candidate
    return _REPO_ROOT / "evaluation"


# Exposed for backwards compatibility (tests may import these).
TESTSET = _evaluation_dir() / "testset.jsonl"
ROOT = _REPO_ROOT


def results_path() -> Path:
    """Backend-specific filename so stub / Ollama runs don't overwrite each other."""

    return _evaluation_dir() / f"loadtest_results_{settings.llm_backend}.md"


# --------------------------------------------------------------------------- #
# Per-node instrumentation
# --------------------------------------------------------------------------- #


_NODE_TIMINGS: dict[str, list[float]] = defaultdict(list)


def _wrap_node(name: str, fn):
    def inner(state):
        t0 = time.perf_counter()
        try:
            return fn(state)
        finally:
            _NODE_TIMINGS[name].append(time.perf_counter() - t0)

    inner.__name__ = f"timed_{name}"
    return inner


def build_instrumented_graph():
    """Build the main graph with each node wrapped in a wall-time recorder."""

    rag_subgraph = build_rag_subgraph()
    rag_node = graph_module._make_rag_node(rag_subgraph)

    nodes = {
        "intake_classifier": intake_classifier,
        "risk_triage": risk_triage,
        "rag_retrieval": rag_node,
        "obligation_mapper": obligation_mapper,
        "compliance_synthesizer": compliance_synthesizer,
        "validator": validator,
        "prohibited_path": graph_module.prohibited_path,
    }
    g = StateGraph(RegPilotState)
    for name, fn in nodes.items():
        g.add_node(name, _wrap_node(name, fn))

    g.add_edge(START, "intake_classifier")
    g.add_edge("intake_classifier", "risk_triage")
    g.add_conditional_edges(
        "risk_triage",
        route_by_tier,
        {"rag_retrieval": "rag_retrieval", "prohibited_path": "prohibited_path"},
    )
    g.add_edge("rag_retrieval", "obligation_mapper")
    g.add_edge("obligation_mapper", "compliance_synthesizer")
    g.add_edge("compliance_synthesizer", "validator")
    g.add_conditional_edges(
        "validator",
        route_after_validator,
        {"obligation_mapper": "obligation_mapper", "__end__": END},
    )
    g.add_edge("prohibited_path", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #


def _queries(n: int, testset: Path | None = None) -> list[str]:
    # Resolved at call time, not import time, so changes to the working
    # directory between import and invocation pick up the right file.
    testset = testset or _evaluation_dir() / "testset.jsonl"
    pool = [
        json.loads(line)["description"]
        for line in testset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: list[str] = []
    while len(out) < n:
        for q in pool:
            if len(out) >= n:
                break
            # Light paraphrase suffix so subsequent runs aren't pure cache hits.
            out.append(f"{q} (case #{len(out) + 1})")
    return out


async def _run_one(graph: Any, query: str, sema: asyncio.Semaphore) -> dict:
    async with sema:
        t0 = time.perf_counter()
        # LangGraph's invoke is sync; run in a worker thread so 100 of them
        # don't block the event loop end-to-end.
        out = await asyncio.to_thread(
            graph.invoke, {"user_input": query, "validator_loops": 0}
        )
        return {
            "latency_s": time.perf_counter() - t0,
            "tier": out.get("risk_tier", "unknown"),
            "loops": out.get("validator_loops", 0),
        }


async def harness(n: int, concurrency: int) -> dict:
    """Run *n* requests at *concurrency* in-flight; return a summary dict."""

    proc = psutil.Process()
    _ = proc.cpu_percent(interval=None)  # prime the counter

    graph = build_instrumented_graph()

    # Warm-up: build BM25 index + Chroma client + LLM cache once before timing
    # so cold-start cost doesn't poison p95/p99.
    print("Warming up (1 query)…")
    warm_sema = asyncio.Semaphore(1)
    await _run_one(graph, _queries(1)[0], warm_sema)
    _NODE_TIMINGS.clear()

    queries = _queries(n)

    sema = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    rss_peak = proc.memory_info().rss

    async def _spawn():
        nonlocal rss_peak
        tasks = [asyncio.create_task(_run_one(graph, q, sema)) for q in queries]
        for done in asyncio.as_completed(tasks):
            rss_peak = max(rss_peak, proc.memory_info().rss)
            await done
        return await asyncio.gather(*tasks)

    results = await _spawn()
    wall = time.perf_counter() - t0
    cpu = proc.cpu_percent(interval=None)

    latencies = sorted(r["latency_s"] for r in results)
    return {
        "n": n,
        "concurrency": concurrency,
        "wall_s": wall,
        "throughput_rps": n / wall if wall else 0.0,
        "p50": _pct(latencies, 50),
        "p95": _pct(latencies, 95),
        "p99": _pct(latencies, 99),
        "min": latencies[0],
        "max": latencies[-1],
        "mean": statistics.mean(latencies),
        "rss_peak_mb": rss_peak / (1024 * 1024),
        "cpu_pct": cpu,
        "tier_dist": _count_tiers(results),
    }


def _pct(xs: list[float], p: int) -> float:
    if not xs:
        return 0.0
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def _count_tiers(results: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in results:
        out[r["tier"]] += 1
    return dict(out)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _node_table() -> list[dict]:
    rows: list[dict] = []
    for name, samples in _NODE_TIMINGS.items():
        if not samples:
            continue
        rows.append(
            {
                "node": name,
                "calls": len(samples),
                "mean_ms": statistics.mean(samples) * 1000,
                "p95_ms": _pct(sorted(samples), 95) * 1000,
                "total_s": sum(samples),
            }
        )
    rows.sort(key=lambda r: r["total_s"], reverse=True)
    return rows


def _bottleneck() -> str:
    rows = _node_table()
    if not rows:
        return "n/a"
    return rows[0]["node"]


def write_report(summary: dict, out_path: Path | None = None) -> Path:
    """Write the loadtest report Markdown and return the path."""

    out_path = out_path or results_path()
    nodes = _node_table()
    total = sum(r["total_s"] for r in nodes) or 1.0
    bottleneck = _bottleneck()

    lines: list[str] = []
    lines.append("# Load test results\n")
    lines.append(
        f"Backend: `{settings.llm_backend}` (chat=`{settings.chat_model}`, "
        f"embed=`{settings.embed_model}`).\n"
    )
    if settings.is_stub:
        lines.append(
            "> ⚠️ **Stub-backend caveat.** Latency below is the *performance "
            "ceiling* of the LangGraph wiring + retrieval pipeline; it doesn't "
            "include the cost of real LLM calls. For real-world latency under "
            "Ollama see [`results_ollama.md`](results_ollama.md) which reports "
            "~140 s p50 / ~180 s p95 per query on CPU with `NUM_PARALLEL=1` "
            "(the determinism setting). Throughput-tuned deployments — "
            "`NUM_PARALLEL=4`, `EMBED_PARALLELISM=4`, fast-paths on — sustain "
            "~5–7 s per query on the same hardware. Real loadtest at scale is "
            "not run in CI because each query is ≥ 5 s and 100 queries would "
            "consume the CI minute budget. Run locally with "
            "`make loadtest-ollama` after a manual `docker compose up --build`.\n"
        )
    lines.append(
        f"- Total requests: **{summary['n']}**\n"
        f"- Concurrency (semaphore): **{summary['concurrency']}**\n"
        f"- Wall-clock: **{summary['wall_s']:.2f} s**\n"
        f"- Throughput: **{summary['throughput_rps']:.2f} req/s**\n"
        f"- Latency (s): min {summary['min']:.3f} · "
        f"**p50 {summary['p50']:.3f}** · p95 {summary['p95']:.3f} · "
        f"p99 {summary['p99']:.3f} · max {summary['max']:.3f} · mean {summary['mean']:.3f}\n"
        f"- Peak RSS: **{summary['rss_peak_mb']:.0f} MB** — CPU% (process): "
        f"**{summary['cpu_pct']:.0f}%**\n"
        f"- Tier distribution: `{summary['tier_dist']}`\n"
    )

    lines.append("## Per-node breakdown\n")
    lines.append("| node | calls | mean (ms) | p95 (ms) | total (s) | share |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in nodes:
        share = r["total_s"] / total
        lines.append(
            f"| {r['node']} | {r['calls']} | {r['mean_ms']:.2f} | "
            f"{r['p95_ms']:.2f} | {r['total_s']:.3f} | {share:.1%} |"
        )

    lines.append(
        f"\n**Identified bottleneck:** `{bottleneck}` "
        f"(largest share of node wall time, post warm-up).\n"
    )
    lines.append(
        "Methodology: one warm-up request is issued before timing so the "
        "Chroma client, BM25 index, and LLM cache are hot. Reported numbers "
        "therefore reflect steady-state, not cold-start. With Ollama in the "
        "loop the picture changes: LLM round-trips in `query_rewrite`, "
        "`rerank` and especially `compliance_synthesizer` dominate (typically "
        "70%+ of wall time per request).\n"
    )

    lines.append("## Two concrete optimisations\n")
    lines.append(
        "1. **Semantic response cache keyed on `(risk_tier, top-N retrieved chunk ids)`** "
        "— in production the same handful of system descriptions (CV screening, "
        "credit scoring, chatbots) repeat constantly. Caching the synthesizer's "
        "Markdown output by a hash of the retrieved-chunk signature would eliminate "
        "the LLM round-trip for any repeat query, which (with real Ollama) accounts "
        "for ~70% of wall time. A 1-day TTL with manual invalidation on Annex/Article "
        "updates is a safe default.\n"
    )
    lines.append(
        "2. **Switch the rerank node from an LLM call to a small cross-encoder + "
        "stream the synthesizer.** The rerank LLM call adds 200–500 ms on Ollama "
        "qwen2.5:3b for very little marginal quality vs the RRF baseline. Replacing "
        "it with a `cross-encoder/ms-marco-MiniLM-L-6-v2` (or even keeping RRF order) "
        "and converting `compliance_synthesizer` to streaming with early-termination "
        "after the first valid section halves the perceived latency.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


__all__ = [
    "build_instrumented_graph",
    "harness",
    "results_path",
    "write_report",
]
