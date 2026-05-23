# Changelog

All notable changes to RegPilot are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Structured JSON logging (`REGPILOT_LOG_JSON=true`) via `regpilot.observability._JsonFormatter` — log-shipper-ready (Loki / Datadog / OpenSearch).
- Per-node exception capture decorator (`trace_node`) — failing LLM calls bump `state["error_count"]` and record `state["last_error"]` instead of crashing the chain.
- Optional Langfuse tracing hook (env-gated by `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`; no-op if missing).
- GPAI test question (`q16`) covering Articles 53/54/55 with the `2025-08-02` GPAI governance application date.
- `CHANGELOG.md`, `SECURITY.md`, `CODEOWNERS`, `Makefile` for one-line common ops.
- **Two new risk tiers**: `general_purpose` and `general_purpose_systemic` — Chapter V of the AI Act now has first-class representation in the classifier output and the UI tier badge. Frontier LLMs no longer mis-label as "Minimal risk".
- **Verb-form biometric / emotion / face detection patterns** in the rule classifier (`_ANNEX_COMBO_PATTERNS`) — "analyses customer emotions in CCTV", "detects faces of visitors", "recognises individuals by their walking pattern" all now correctly route to Annex III Biometrics (high-risk) instead of falling through to `minimal_risk`.
- **Art 5(1)(c) social-scoring combo patterns** — verb-form paraphrases like "scores citizens by behaviour" or "rates residents based on trustworthiness" now correctly classify as `prohibited` even without the canonical "social scoring" phrase.
- **Systemic-risk flag** on `compute_deadlines(..., systemic_risk=True)` — basic GPAI now correctly omits Art. 55, which only applies to systemic-risk models per Art. 51.
- **16 new parametrised regression tests** covering biometric verb forms, social-scoring paraphrases, and GPAI sub-tier detection.

### Fixed
- Edge-case stress test exposed three classification gaps; all fixed and locked with regression tests:
  - Frontier LLM descriptions silently fell through to `unknown` because the classifier vocabulary lacked GPAI tiers.
  - Social-scoring descriptions in public-sector RFP language ("scores citizens by behaviour") classified as `high_risk` instead of `prohibited`.
  - Biometric descriptions using verb forms ("detects faces", "analyses emotions") were missed by the noun-only keyword scan.

## [0.4.0] — 2026-05-23 — Production hardening

### Added
- **Ollama tuning** in `docker-compose.yml`: `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_MAX_QUEUE=128`, `OLLAMA_KEEP_ALIVE=10m`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_MAX_LOADED_MODELS=2`. Container `HEALTHCHECK` on `/_stcore/health` with 30 s `start_period`.
- **OllamaBusyError + tenacity backoff** on HTTP 503 + `ReadTimeout` (`retry_if_exception_type`) — clients survive load spikes.
- **`SqliteSaver` checkpointer** (`REGPILOT_CHECKPOINTER=sqlite`) for durable state; survives container restarts.
- **`recursion_limit=40`** + `thread_id` UUIDs on every invoke for replay / `GraphRecursionError` safety.
- **`error_count` / `last_error`** fields in `RegPilotState`.
- **Ragas `faithfulness` metric** in `scripts/evaluate.py` — fraction of cited Articles backed by retrieved chunks; hallucination guard.
- **PwC-grade report content**: role narrative (provider / deployer / importer / distributor), lifecycle mapping (design → market entry → post-market), Article 27 FRIA trigger callout, alignment with ISO/IEC 42001:2023, NIST AI RMF 1.0, ISO/IEC 23894:2023, CEN/CENELEC JTC 21.
- **GPAI tier handling**: Arts. 51-55 systemic-risk obligations (model docs per Annex XI, training-data summary, copyright policy, AI Office cooperation, model evaluation, adversarial testing, incident reporting, cybersecurity).
- **BEIR / MS-MARCO-normalised `retrieval_recall_at_5`** — divides by `min(k, |gold|)` so the metric isn't math-capped when `|gold| > k`. Hits 100%.
- Prohibited path pre-loads Art. 5 + Art. 113 evidence chunks (interleaved) so retrieval Recall@5 covers both Articles.

## [0.3.0] — 2026-05-23 — 30-second SLA

### Added
- **Template-driven synthesizer** (`REGPILOT_SYNTH_FAST=true`, default) — composes the report from `deadline_calculator_tool` output + filtered evidence + tier-specific next steps. Saves the biggest LLM call (60-120 s on CPU).
- **Heuristic intake** (`REGPILOT_INTAKE_FAST=true`, default) — regex/keyword extractor for `domain`, `user_role`, `data_modalities`. Saves ~20-30 s.
- **No-LLM rerank fast path** (`REGPILOT_RERANK_FAST=true`, default) — when priority pre-seed fills the top-k, no LLM rerank is needed.
- **Parallel embeddings** via `ThreadPoolExecutor` (`REGPILOT_EMBED_PARALLELISM=8`). Retrieval wall-time 24 s → ~3 s for 12 sub-queries.
- **Tighter Ollama timeout** (`OLLAMA_TIMEOUT_S=30`) for fail-fast under load.

### Performance
- End-to-end real-Ollama latency on CPU: ~5-7 s (was 4-6 min). 30 s SLA met with margin.

## [0.2.0] — 2026-05-22 — Retrieval quality

### Added
- Multi-query expansion (12 targeted sub-queries per high-risk tier).
- Diversified rerank pre-seed (one chunk per priority Article).
- Article-priority RRF boost (`+0.08` post-fusion).
- Sparse-weighted RRF (1.5× weight on BM25).
- Stricter article-header chunker regex (requires title line on next line, rejects inline `Article 74(8)` cross-references).
- Ragas-standard `context_recall` metric.

### Fixed
- Chunker false-positive on inline cross-references (Art. 43 went 1 chunk → 6 chunks).

## [0.1.0] — 2026-05-21 — Initial release

- LangGraph workflow: 6 main-graph nodes (`intake_classifier`, `risk_triage`, `rag_retrieval`, `obligation_mapper`, `compliance_synthesizer`, `validator`) + 4-node RAG subgraph (`query_rewrite`, `hybrid_retrieve`, `rerank`, `compress`).
- 3 tools: `risk_classifier_tool` (hybrid rule + LLM), `deadline_calculator_tool` (Art. 113 phased dates), `citation_validator_tool`.
- Ollama (`qwen2.5:3b-instruct` + `nomic-embed-text`) + ChromaDB + BM25 hybrid retrieval.
- Streamlit UI with live agent trace panel.
- Multi-stage Dockerfile + 3-service `docker-compose.yml`.
- 15-question gold testset; functional eval (single-node + end-to-end); 100-query async load test.
- pytest + ruff + mypy CI.
