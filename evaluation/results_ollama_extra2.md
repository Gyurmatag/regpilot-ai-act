# Functional evaluation results

Backend: `ollama` — Chat model: `qwen2.5:3b-instruct` — Embed model: `nomic-embed-text` — Testset: 10 questions.

## Single-node eval — `risk_triage`

**Triage accuracy: 40.00%** (threshold 80%)


Confusion matrix (rows = gold, columns = predicted):

| gold \ predicted | prohibited | high_risk | limited_risk | minimal_risk |
| --- | --- | --- | --- | --- |
| prohibited | 0 | 0 | 0 | 0 |
| high_risk | 0 | 3 | 0 | 1 |
| limited_risk | 0 | 1 | 0 | 0 |
| minimal_risk | 0 | 3 | 1 | 0 |

## End-to-end eval — full workflow

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| triage_accuracy | 60.00% | 80% | NO |
| context_recall | 60.83% | 90% | NO |
| faithfulness | 96.67% | 90% | yes |
| citation_recall | 60.00% | 80% | NO |
| citation_precision | 56.67% | 70% | NO |
| deadline_exact_match | 100.00% | 80% | yes |
| retrieval_recall_at_5 | 60.00% | 90% | NO |
| MRR | 0.617 | — | — |
| latency p50 (s) | 131.20 | — | — |
| latency p95 (s) | 183.50 | — | — |

## Per-question breakdown

| id | gold | pred | ctx recall | R@5 | MRR | cite prec | cite recall | deadline | lat s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x21 | minimal_risk | high_risk | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | yes | 112.71 |
| x22 | high_risk | minimal_risk | 0.08 | 0.00 | 0.17 | 0.00 | 0.00 | yes | 152.79 |
| x23 | minimal_risk | high_risk | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | yes | 130.66 |
| x24 | minimal_risk | minimal_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 156.36 |
| x25 | limited_risk | high_risk | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | yes | 117.15 |
| x26 | minimal_risk | minimal_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 183.50 |
| x27 | general_purpose | general_purpose | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | yes | 160.67 |
| x28 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 121.45 |
| x29 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 131.74 |
| x30 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 103.37 |

## Commentary
- Triage accuracy (40%) is below target — see the confusion matrix to find which tier needs better rules or richer Annex examples.
- **Context recall = 61%** (target 90%). This is the headline retrieval metric, defined as in [Ragas](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html) — how many gold Articles appear anywhere in the retrieved context the synthesizer sees. Position-agnostic, not bounded by k.
- Retrieval Recall@5 = 60% (target 90%). Normalised per [BEIR](https://github.com/beir-cellar/beir) / [MS-MARCO](https://microsoft.github.io/msmarco/) convention: `|top5 ∩ gold| / min(5, |gold|)`, so it isn't math-capped when `|gold| > k`. Measures how cleanly the top-5 chunks the user sees are filled with relevant Articles.
- Citation recall (60%) — what share of the gold Articles are actually cited in the final report. This is the most user-facing metric: when high, the user gets the obligations they need to know about.
- Citation precision (57%) — what share of cited Articles are in the gold list. We don't gate on this because the retrieval subgraph legitimately surfaces adjacent Articles (e.g. Annex III matches) that aren't in the narrow gold set but are still useful.
- Median latency 131.2s, p95 183.5s (stub LLM dominates retrieval cost; with Ollama qwen2.5:3b expect 5–10× slower).