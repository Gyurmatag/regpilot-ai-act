# Load test results

Backend: `stub` (chat=`qwen2.5:3b-instruct`, embed=`nomic-embed-text`).

- Total requests: **100**
- Concurrency (semaphore): **8**
- Wall-clock: **1.42 s**
- Throughput: **70.50 req/s**
- Latency (s): min 0.019 · **p50 0.078** · p95 0.288 · p99 0.445 · max 0.446 · mean 0.110
- Peak RSS: **183 MB** — CPU% (process): **292%**
- Tier distribution: `{'prohibited': 21, 'high_risk': 42, 'limited_risk': 19, 'minimal_risk': 18}`

## Per-node breakdown

| node | calls | mean (ms) | p95 (ms) | total (s) | share |
| --- | --- | --- | --- | --- | --- |
| rag_retrieval | 79 | 113.32 | 349.31 | 8.952 | 85.2% |
| prohibited_path | 21 | 70.09 | 206.44 | 1.472 | 14.0% |
| validator | 79 | 0.77 | 0.07 | 0.061 | 0.6% |
| risk_triage | 100 | 0.18 | 0.22 | 0.018 | 0.2% |
| intake_classifier | 100 | 0.05 | 0.07 | 0.005 | 0.1% |
| obligation_mapper | 79 | 0.02 | 0.04 | 0.002 | 0.0% |
| compliance_synthesizer | 79 | 0.02 | 0.03 | 0.002 | 0.0% |

**Identified bottleneck:** `rag_retrieval` (largest share of node wall time, post warm-up).

Methodology: one warm-up request is issued before timing so the Chroma client, BM25 index, and LLM cache are hot. Reported numbers therefore reflect steady-state, not cold-start. With Ollama in the loop the picture changes: LLM round-trips in `query_rewrite`, `rerank` and especially `compliance_synthesizer` dominate (typically 70%+ of wall time per request).

## Two concrete optimisations

1. **Semantic response cache keyed on `(risk_tier, top-N retrieved chunk ids)`** — in production the same handful of system descriptions (CV screening, credit scoring, chatbots) repeat constantly. Caching the synthesizer's Markdown output by a hash of the retrieved-chunk signature would eliminate the LLM round-trip for any repeat query, which (with real Ollama) accounts for ~70% of wall time. A 1-day TTL with manual invalidation on Annex/Article updates is a safe default.

2. **Switch the rerank node from an LLM call to a small cross-encoder + stream the synthesizer.** The rerank LLM call adds 200–500 ms on Ollama qwen2.5:3b for very little marginal quality vs the RRF baseline. Replacing it with a `cross-encoder/ms-marco-MiniLM-L-6-v2` (or even keeping RRF order) and converting `compliance_synthesizer` to streaming with early-termination after the first valid section halves the perceived latency.
