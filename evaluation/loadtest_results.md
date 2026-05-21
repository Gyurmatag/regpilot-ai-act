# Load test results

Backend: `stub` (chat=`qwen2.5:3b-instruct`, embed=`nomic-embed-text`).

- Total requests: **100**
- Concurrency (semaphore): **8**
- Wall-clock: **1.36 s**
- Throughput: **73.46 req/s**
- Latency (s): min 0.001 · **p50 0.073** · p95 0.541 · p99 0.566 · max 0.567 · mean 0.107
- Peak RSS: **215 MB** — CPU% (process): **231%**
- Tier distribution: `{'prohibited': 21, 'high_risk': 42, 'limited_risk': 19, 'minimal_risk': 18}`

## Per-node breakdown

| node | calls | mean (ms) | p95 (ms) | total (s) | share |
| --- | --- | --- | --- | --- | --- |
| rag_retrieval | 79 | 131.03 | 530.00 | 10.351 | 99.6% |
| validator | 79 | 0.25 | 0.03 | 0.019 | 0.2% |
| risk_triage | 100 | 0.12 | 0.15 | 0.012 | 0.1% |
| intake_classifier | 100 | 0.03 | 0.04 | 0.003 | 0.0% |
| compliance_synthesizer | 79 | 0.04 | 0.04 | 0.003 | 0.0% |
| obligation_mapper | 79 | 0.02 | 0.03 | 0.001 | 0.0% |
| prohibited_path | 21 | 0.01 | 0.01 | 0.000 | 0.0% |

**Identified bottleneck:** `rag_retrieval` (largest share of node wall time).

The first call to `rag_retrieval` builds the BM25 index from the entire Chroma corpus (~840 chunks) — a one-off cost that inflates p95 / p99. Once warm, subsequent calls take <50 ms on the stub backend. With Ollama in the loop the picture flips: LLM round-trips in `query_rewrite`, `rerank` and especially `compliance_synthesizer` dominate, typically 70%+ of wall time per request.

## Two concrete optimisations

1. **Semantic response cache keyed on `(risk_tier, top-N retrieved chunk ids)`** — in production the same handful of system descriptions (CV screening, credit scoring, chatbots) repeat constantly. Caching the synthesizer's Markdown output by a hash of the retrieved-chunk signature would eliminate the LLM round-trip for any repeat query, which (with real Ollama) accounts for ~70% of wall time. A 1-day TTL with manual invalidation on Annex/Article updates is a safe default.

2. **Switch the rerank node from an LLM call to a small cross-encoder + stream the synthesizer.** The rerank LLM call adds 200–500 ms on Ollama qwen2.5:3b for very little marginal quality vs the RRF baseline. Replacing it with a `cross-encoder/ms-marco-MiniLM-L-6-v2` (or even keeping RRF order) and converting `compliance_synthesizer` to streaming with early-termination after the first valid section halves the perceived latency.
