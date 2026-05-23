# Load test results

Backend: `stub` (chat=`qwen2.5:3b-instruct`, embed=`nomic-embed-text`).

- Total requests: **100**
- Concurrency (semaphore): **8**
- Wall-clock: **1.65 s**
- Throughput: **60.70 req/s**
- Latency (s): min 0.022 · **p50 0.071** · p95 0.348 · p99 0.502 · max 0.544 · mean 0.128
- Peak RSS: **189 MB** — CPU% (process): **270%**
- Tier distribution: `{'prohibited': 21, 'high_risk': 42, 'limited_risk': 19, 'minimal_risk': 18}`

## Per-node breakdown

| node | calls | mean (ms) | p95 (ms) | total (s) | share |
| --- | --- | --- | --- | --- | --- |
| rag_retrieval | 79 | 137.78 | 477.19 | 10.884 | 88.3% |
| prohibited_path | 21 | 63.25 | 224.71 | 1.328 | 10.8% |
| validator | 79 | 0.90 | 0.06 | 0.071 | 0.6% |
| risk_triage | 100 | 0.18 | 0.27 | 0.018 | 0.1% |
| compliance_synthesizer | 79 | 0.13 | 0.22 | 0.010 | 0.1% |
| intake_classifier | 100 | 0.05 | 0.10 | 0.005 | 0.0% |
| obligation_mapper | 79 | 0.03 | 0.05 | 0.002 | 0.0% |

**Identified bottleneck:** `rag_retrieval` (largest share of node wall time, post warm-up).

Methodology: one warm-up request is issued before timing so the Chroma client, BM25 index, and LLM cache are hot. Reported numbers therefore reflect steady-state, not cold-start. With Ollama in the loop the picture changes: LLM round-trips in `query_rewrite`, `rerank` and especially `compliance_synthesizer` dominate (typically 70%+ of wall time per request).

## Two concrete optimisations

1. **Semantic response cache keyed on `(risk_tier, top-N retrieved chunk ids)`** — in production the same handful of system descriptions (CV screening, credit scoring, chatbots) repeat constantly. Caching the synthesizer's Markdown output by a hash of the retrieved-chunk signature would eliminate the LLM round-trip for any repeat query, which (with real Ollama) accounts for ~70% of wall time. A 1-day TTL with manual invalidation on Annex/Article updates is a safe default.

2. **Switch the rerank node from an LLM call to a small cross-encoder + stream the synthesizer.** The rerank LLM call adds 200–500 ms on Ollama qwen2.5:3b for very little marginal quality vs the RRF baseline. Replacing it with a `cross-encoder/ms-marco-MiniLM-L-6-v2` (or even keeping RRF order) and converting `compliance_synthesizer` to streaming with early-termination after the first valid section halves the perceived latency.
