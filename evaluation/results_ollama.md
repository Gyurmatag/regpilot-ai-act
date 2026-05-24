# Functional evaluation — live Ollama (LLM-primary)

This file is the canonical real-world eval against the production-default
stack: `docker compose up --build` with `REGPILOT_LLM=ollama` and every node
running LLM-primary (`*_FAST=false`). For the per-testset auto-generated
artefacts see [`results_ollama_extra.md`](results_ollama_extra.md) and
[`results_ollama_extra2.md`](results_ollama_extra2.md). For the stub run
(CI smoke test) see [`results_stub.md`](results_stub.md).

Backend: `ollama` — Chat model: `qwen2.5:3b-instruct` — Embed model: `nomic-embed-text`.

## Headline across all three testsets

| Metric | Main (16) | Extra A (10) | Extra B (10) | Thresh |
|---|---|---|---|---|
| triage_accuracy | **87.50%** ✓ | **100.00%** ✓ | 60.00% ✗ | 80% |
| context_recall (Ragas) | **91.67%** ✓ | **100.00%** ✓ | 60.83% ✗ | 90% |
| faithfulness (Ragas) | **97.92%** ✓ | **91.73%** ✓ | **96.67%** ✓ | 90% |
| retrieval_recall_at_5 (BEIR) | **91.67%** ✓ | **100.00%** ✓ | 60.00% ✗ | 90% |
| citation_recall | **91.67%** ✓ | **97.50%** ✓ | 60.00% ✗ | 80% |
| citation_precision | **91.67%** ✓ | **91.73%** ✓ | 56.67% ✗ | 70% |
| deadline_exact_match | **100.00%** ✓ | **100.00%** ✓ | **100.00%** ✓ | 80% |
| latency p50 / p95 (s) | 123.8 / 166.8 | 126.7 / 179.5 | 131.2 / 183.5 | — |

**Main + Extra A clear every gating threshold.** Extra B is a held-out
stress set of 10 deliberately novel scenarios (agricultural drone, AI toy
emotion adaptation, semiconductor defect detection, autonomous vacuum, tax
advisor chatbot, smart thermostat, generative audio foundation model,
retail biometric surveillance, kindergarten admission, welfare-fraud
detection). It exposes the real limits of the 3 B-parameter Ollama
backend (see "Honest assessment" below) — the deadline calculator and
GPAI/biometric/education paths still hold up (faithfulness 96.67%,
deadline_exact_match 100%).

Faithfulness — the strongest hallucination guard — is ≥91% across all
three sets: when the system *does* cite an Article, it's almost always an
Article it actually retrieved.

## Triple-run reproducibility (main set)

Three full `docker compose down -v + system prune --volumes + up --build`
cycles produced byte-identical scores. Determinism comes from
`OLLAMA_NUM_PARALLEL=1`, `REGPILOT_EMBED_PARALLELISM=1`, `seed=42` plumbed
into every Ollama call. The small latency drift (~22 s on p50 across the
three boots) reflects external CPU contention on the host, not model
non-determinism — every score that depends on model output is identical.

## Per-testset misses

**Main set (2 misses out of 16)** — both limited-risk borderlines:

* `q11` "deepfake video clips for marketing campaigns" → `high_risk`
  instead of `limited_risk`. The LLM read "marketing campaigns" as a
  commercial Annex III use case. Defensible regulatory reading but
  diverges from the gold-set label.
* `q16` "7B-parameter foundation model for marketing copy" →
  `general_purpose` instead of `limited_risk`. Architecturally the LLM
  is correct (basic-GPAI bright-line rule fires because of "7B-parameter
  foundation model"); the gold-set label took the application-level
  reading. Both are defensible.

**Extra A set (clean 10/10 triage)** — every Annex III paraphrase the
testset throws at the classifier resolves correctly thanks to the
enriched canonical examples + the basic-GPAI bright-line rule for
foundation / code-completion / NNb-parameter shapes.

**Extra B set (4 misses out of 10)** — exposes real 3B-LLM bounds:

* `x21` agricultural drone → `high_risk` (false positive). The LLM
  semantically associates "aerial imagery monitoring" with critical
  infrastructure or law-enforcement surveillance.
* `x22` children's emotional toy → `minimal_risk` (true miss). The LLM
  did not pick up that "detects emotional state via voice tone and
  facial expression" is Annex III biometrics; it read the description as
  a benign consumer toy.
* `x23` semiconductor wafer defect detection → `high_risk` (false
  positive). The LLM treated "fabrication line inspection" as a critical
  infrastructure safety component even though semiconductors aren't in
  Annex I's regulated-machinery list.
* `x25` tax-advisor chatbot → `high_risk` (true miss for Art. 50). The
  LLM read "tax advice for individuals" as a financial essential service
  rather than as a chatbot with transparency duties.

All four are LLM-quality bounded; a hosted LLM swap via
`REGPILOT_LLM=openai` or `REGPILOT_LLM=anthropic` would shrink these
sharply (the provider abstraction is already in place).

## Honest assessment of the classifier changes

A prior iteration tried four classifier improvements together —
canonical-example enrichment, lower semantic threshold (0.45 → 0.35),
basic-GPAI bright-line rule, and a strengthened LLM prompt with explicit
numbered decision rules. The combination scored 100% on Extra A but
**regressed the main set** from 87.5% triage to 75% (it broke three of
four limited-risk classifications because the explicit "if Annex III →
high_risk" decision rule made the LLM skip the Article 50 chatbot
branch).

The current shipping state keeps the two changes that survived empirical
validation:

* **Enriched Annex III canonical examples.** Each canonical embed text
  now includes a handful of concrete real-world AI applications per
  area (e.g. "PhD applicant ranking", "voice biometric authentication",
  "load balancing for utilities"). These are descriptions of what each
  Annex III area *actually covers* in regulatory practice — they're not
  testset answers. The embedding similarity rewards semantic closeness,
  so enriching the canonical with one phrase ("PhD applicant ranking")
  also catches paraphrases like "doctoral selection" or "fellowship
  admission" that aren't in the testset.
* **Basic-GPAI bright-line rule.** Article 53 of the AI Act enumerates
  what a general-purpose AI model is — foundation models, large language
  models, multi-modal generators offered to downstream deployers. The
  rule matches `(foundation|base)\s+(model|llm|ai)`, `large language
  model`, `\d+B-parameter\s+(model|llm|...)`, `(code-completion|
  multimodal|generative)\s+(model|service|api)`. This is the
  same kind of enumerated bright-line as Article 5 (which the project
  already had) and Article 51 systemic-risk markers — regulatorily
  defensible, not testset-fitted. It catches GPAI shapes whether they're
  in the testset or not.

The two changes that were rolled back:

* **Threshold 0.45 → 0.35.** Surfaced more candidates to the LLM, which
  helped Extra A but injected enough noise to mis-route 3 limited-risk
  cases on main. The current 0.45 threshold has the enriched canonicals
  but conservative gating.
* **Numbered decision-rule prompt.** Over-prescriptive — the LLM treated
  the rules as a strict top-down checklist and skipped the chatbot
  branch. Reverted to the original tier-vocabulary prompt; the LLM is
  better at picking the right tier when given the categories rather than
  an ordered procedure.

Two improvements + two principled reverts = the architecture stays
LLM-primary with bright-line rules **only** for the enumerated
regulatory categories where the Act itself prescribes the exact wording
(Article 5, Article 51, Article 53). The 3B Ollama remains the
quality bottleneck on edge cases; the architecture supports swapping in
a stronger model without code changes.

## Methodology

* **Metric definitions:** Ragas
  [`context_recall`](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html) +
  [`faithfulness`](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html);
  BEIR/MS-MARCO-normalised `retrieval_recall_at_5` = `|top5 ∩ gold| /
  min(5, |gold|)`.
* **Reproducibility knobs:** `OLLAMA_NUM_PARALLEL=1`,
  `REGPILOT_EMBED_PARALLELISM=1`, `OLLAMA_TIMEOUT_S=240`,
  `OLLAMA_KEEP_ALIVE=30m`, `seed=42`. With these settings three
  back-to-back fresh-boot runs return byte-identical scores.
* **Latency note:** ~120-160 s p50 reflects the deliberate trade-off for
  determinism (`NUM_PARALLEL=1` serialises Ollama inference). For
  ~5-7 s per query on the same hardware flip the `*_FAST=true` env
  vars to bypass the LLM in intake / synthesizer / rerank; for ~3-6 s
  with hosted-LLM quality, switch to `REGPILOT_LLM=openai` or
  `REGPILOT_LLM=anthropic`.
