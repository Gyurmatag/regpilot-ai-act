# Functional evaluation results

Backend: `stub` — Chat model: `qwen2.5:3b-instruct` — Embed model: `nomic-embed-text` — Testset: 16 questions.

> ⚠️ **Stub-backend caveat.** The stub LLM uses hash-based pseudo-embeddings, so the semantic-similarity Annex III matcher (Option C) can't surface relevant areas, and end-to-end retrieval metrics are degraded by design. The stub run still validates classifier wiring, graph assembly, and the deterministic regulatory layer (deadline calculator, Article 5 bright-line rules) — useful as a smoke test, not as a quality benchmark. For real metrics see [`results_ollama.md`](results_ollama.md).

## Single-node eval — `risk_triage`

**Triage accuracy: 68.75%** (threshold 80%)


Confusion matrix (rows = gold, columns = predicted):

| gold \ predicted | prohibited | high_risk | limited_risk | minimal_risk |
| --- | --- | --- | --- | --- |
| prohibited | 3 | 0 | 0 | 0 |
| high_risk | 0 | 3 | 0 | 3 |
| limited_risk | 0 | 0 | 2 | 1 |
| minimal_risk | 0 | 0 | 0 | 3 |

## End-to-end eval — full workflow

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| triage_accuracy | 68.75% | 80% | NO |
| context_recall | 73.44% | 90% | NO |
| faithfulness | 59.38% | 90% | NO |
| citation_recall | 77.60% | 80% | NO |
| citation_precision | 61.46% | 70% | NO |
| deadline_exact_match | 100.00% | 80% | yes |
| retrieval_recall_at_5 | 74.17% | 90% | NO |
| MRR | 0.781 | — | — |
| latency p50 (s) | 0.01 | — | — |
| latency p95 (s) | 0.08 | — | — |

## Per-question breakdown

| id | gold | pred | ctx recall | R@5 | MRR | cite prec | cite recall | deadline | lat s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | prohibited | prohibited | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 0.22 |
| q02 | prohibited | prohibited | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 0.02 |
| q03 | prohibited | prohibited | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 0.01 |
| q04 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 0.08 |
| q05 | high_risk | minimal_risk | 0.00 | 0.00 | 0.00 | 0.75 | 0.25 | yes | 0.02 |
| q06 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 0.05 |
| q07 | high_risk | minimal_risk | 0.00 | 0.00 | 0.00 | 0.75 | 0.25 | yes | 0.02 |
| q08 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 0.03 |
| q09 | high_risk | minimal_risk | 0.08 | 0.20 | 0.50 | 0.75 | 0.25 | yes | 0.01 |
| q10 | limited_risk | limited_risk | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | yes | 0.01 |
| q11 | limited_risk | limited_risk | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | yes | 0.01 |
| q12 | limited_risk | minimal_risk | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | yes | 0.01 |
| q13 | minimal_risk | minimal_risk | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | yes | 0.02 |
| q14 | minimal_risk | minimal_risk | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | yes | 0.01 |
| q15 | minimal_risk | minimal_risk | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | yes | 0.01 |
| q16 | limited_risk | general_purpose | 0.67 | 0.67 | 1.00 | 0.33 | 0.67 | yes | 0.01 |

## Commentary
- Triage accuracy (69%) is below target — see the confusion matrix to find which tier needs better rules or richer Annex examples.
- **Context recall = 73%** (target 90%). This is the headline retrieval metric, defined as in [Ragas](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html) — how many gold Articles appear anywhere in the retrieved context the synthesizer sees. Position-agnostic, not bounded by k.
- Retrieval Recall@5 = 74% (target 90%). Normalised per [BEIR](https://github.com/beir-cellar/beir) / [MS-MARCO](https://microsoft.github.io/msmarco/) convention: `|top5 ∩ gold| / min(5, |gold|)`, so it isn't math-capped when `|gold| > k`. Measures how cleanly the top-5 chunks the user sees are filled with relevant Articles.
- Citation recall (78%) — what share of the gold Articles are actually cited in the final report. This is the most user-facing metric: when high, the user gets the obligations they need to know about.
- Citation precision (61%) — what share of cited Articles are in the gold list. We don't gate on this because the retrieval subgraph legitimately surfaces adjacent Articles (e.g. Annex III matches) that aren't in the narrow gold set but are still useful.
- Median latency 0.0s, p95 0.1s (stub LLM dominates retrieval cost; with Ollama qwen2.5:3b expect 5–10× slower).