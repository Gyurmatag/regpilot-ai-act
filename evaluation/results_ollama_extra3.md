# Functional evaluation results

Backend: `ollama` — Chat model: `qwen2.5:3b-instruct` — Embed model: `nomic-embed-text` — Testset: 10 questions.

## Single-node eval — `risk_triage`

**Triage accuracy: 60.00%** (threshold 80%)


Confusion matrix (rows = gold, columns = predicted):

| gold \ predicted | prohibited | high_risk | limited_risk | minimal_risk |
| --- | --- | --- | --- | --- |
| prohibited | 0 | 0 | 0 | 0 |
| high_risk | 0 | 5 | 0 | 0 |
| limited_risk | 0 | 1 | 0 | 0 |
| minimal_risk | 0 | 2 | 0 | 1 |

## End-to-end eval — full workflow

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| triage_accuracy | 80.00% | 80% | yes |
| context_recall | 80.00% | 90% | NO |
| faithfulness | 84.36% | 90% | NO |
| citation_recall | 80.00% | 80% | yes |
| citation_precision | 61.86% | 70% | NO |
| deadline_exact_match | 90.00% | 80% | yes |
| retrieval_recall_at_5 | 80.00% | 90% | NO |
| MRR | 0.800 | — | — |
| latency p50 (s) | 175.89 | — | — |
| latency p95 (s) | 211.47 | — | — |

## Per-question breakdown

| id | gold | pred | ctx recall | R@5 | MRR | cite prec | cite recall | deadline | lat s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y01 | minimal_risk | limited_risk | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | yes | 200.38 |
| y02 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | no | 171.27 |
| y03 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 135.74 |
| y04 | minimal_risk | minimal_risk | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | yes | 183.27 |
| y05 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | yes | 141.53 |
| y06 | minimal_risk | minimal_risk | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | yes | 180.52 |
| y07 | general_purpose | general_purpose | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | yes | 211.47 |
| y08 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | yes | 181.78 |
| y09 | limited_risk | high_risk | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | yes | 148.03 |
| y10 | high_risk | high_risk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes | 148.92 |

## Commentary
- Triage accuracy (60%) is below the 80% threshold. The confusion matrix above shows which tier is being confused with which.
- Context recall (80%, target 90%) is the headline retrieval number — Ragas definition: how many gold Articles appear anywhere in the retrieved context, position-agnostic and not bounded by k. See https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html.
- Retrieval Recall@5 (80%, target 90%) uses the BEIR / MS-MARCO normalisation `|top5 ∩ gold| / min(5, |gold|)` so it isn't math-capped when `|gold| > k`.
- Citation recall (80%) is the share of the gold Articles that actually end up cited in the final report — the most user-facing number, since a missed citation means a missed obligation.
- Citation precision (62%) is the share of cited Articles that are in the gold list. We don't gate hard on this — the retriever legitimately surfaces adjacent Articles that are useful context but aren't in the narrow gold set.
- Median latency 175.9s, p95 211.5s. Stub backend reflects pipeline-only cost; with live Ollama qwen2.5:3b on CPU expect 50–100× slower.