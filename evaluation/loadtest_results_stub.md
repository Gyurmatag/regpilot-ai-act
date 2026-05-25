# Load test results

Backend: `stub` (chat=`qwen2.5:3b-instruct`, embed=`nomic-embed-text`).

> ⚠️ **Stub-backend caveat.** Latency below is the *performance ceiling* of the LangGraph wiring + retrieval pipeline; it doesn't include the cost of real LLM calls. For real-world latency under Ollama see [`results_ollama.md`](results_ollama.md) which reports ~140 s p50 / ~180 s p95 per query on CPU with `NUM_PARALLEL=1` (the determinism setting). Throughput-tuned deployments — `NUM_PARALLEL=4`, `EMBED_PARALLELISM=4`, fast-paths on — sustain ~5–7 s per query on the same hardware. Real loadtest at scale is not run in CI because each query is ≥ 5 s and 100 queries would consume the CI minute budget. Run locally with `make loadtest-ollama` after a manual `docker compose up --build`.

- Total requests: **50**
- Concurrency (semaphore): **4**
- Wall-clock: **0.53 s**
- Throughput: **94.92 req/s**
- Latency (s): min 0.012 · **p50 0.025** · p95 0.117 · p99 0.184 · max 0.184 · mean 0.041
- Peak RSS: **152 MB** — CPU% (process): **230%**
- Tier distribution: `{'prohibited': 11, 'high_risk': 9, 'minimal_risk': 21, 'limited_risk': 6, 'general_purpose': 3}`

## Per-node breakdown

| node | calls | mean (ms) | p95 (ms) | total (s) | share |
| --- | --- | --- | --- | --- | --- |
| rag_retrieval | 39 | 33.09 | 78.31 | 1.290 | 69.9% |
| prohibited_path | 11 | 44.40 | 87.18 | 0.488 | 26.5% |
| validator | 39 | 1.19 | 0.10 | 0.046 | 2.5% |
| risk_triage | 50 | 0.30 | 0.43 | 0.015 | 0.8% |
| compliance_synthesizer | 39 | 0.08 | 0.13 | 0.003 | 0.2% |
| intake_classifier | 50 | 0.02 | 0.03 | 0.001 | 0.1% |
| obligation_mapper | 39 | 0.02 | 0.08 | 0.001 | 0.0% |

**Identified bottleneck:** `rag_retrieval` (largest share of node wall time, post warm-up).

Methodology: one warm-up request is issued before timing so the Chroma client, BM25 index, and LLM cache are hot. Reported numbers therefore reflect steady-state, not cold-start. With Ollama in the loop the picture changes: LLM round-trips in `query_rewrite`, `rerank` and especially `compliance_synthesizer` dominate (typically 70%+ of wall time per request).

## Two concrete optimisations

1. **Semantic response cache keyed on `(risk_tier, top-N retrieved chunk ids)`** — in production the same handful of system descriptions (CV screening, credit scoring, chatbots) repeat constantly. Caching the synthesizer's Markdown output by a hash of the retrieved-chunk signature would eliminate the LLM round-trip for any repeat query, which (with real Ollama) accounts for ~70% of wall time. A 1-day TTL with manual invalidation on Annex/Article updates is a safe default.

2. **Switch the rerank node from an LLM call to a small cross-encoder + stream the synthesizer.** The rerank LLM call adds 200–500 ms on Ollama qwen2.5:3b for very little marginal quality vs the RRF baseline. Replacing it with a `cross-encoder/ms-marco-MiniLM-L-6-v2` (or even keeping RRF order) and converting `compliance_synthesizer` to streaming with early-termination after the first valid section halves the perceived latency.
