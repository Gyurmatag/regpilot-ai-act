# Changelog

All notable changes to RegPilot are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Architecture — LLM-first refactor (Option C)

The hot path is now genuinely LLM-driven instead of regex-driven. Rules
are reserved for bright-line regulatory enumerations (Article 5
prohibited list, GPAI Article 51 systemic-risk threshold) where
auditability requires deterministic behaviour; everything else flows
through the LLM with structured output.

### Added
- **Provider abstraction** — one `LLMClient` interface, four backends:
  - `OllamaClient` (default, fully local; uses native `format=json` for structured output)
  - `OpenAIClient` (hosted; uses `beta.chat.completions.parse` with Pydantic `response_format`)
  - `AnthropicClient` (hosted; uses tool calling to force schema-conformant output)
  - `StubClient` (deterministic mock for unit tests and offline dev)
  - `_CompositeClient` wires Ollama embeddings into the Anthropic chat client (Anthropic has no embedding API)
  - Provider selected via `REGPILOT_LLM=ollama|openai|anthropic|stub`
- **Structured output API** — every client implements `generate_structured(prompt, schema)` taking a Pydantic class and returning a validated instance. Falls back to JSON-prompt parsing if the native API path fails. Surfaces `StructuredOutputError` so callers can degrade gracefully.
- **Semantic-similarity Annex III classifier** — each Annex III area is embedded once per process (lazy + cached); user input is embedded; cosine similarity surfaces candidate areas above `REGPILOT_SEM_THRESHOLD`. This *generalises to paraphrases* — no more hand-written verb-form regex patterns.
- **`ClassificationResult`, `IntakeSchema`, `ReportSections` Pydantic schemas** — the LLM fills these in via structured output; downstream nodes consume validated objects, not regex-parsed JSON.
- **Two new risk tiers**: `general_purpose` and `general_purpose_systemic` — Chapter V of the AI Act has first-class representation; UI badge shows "GPAI · systemic risk" in violet.
- **42 new tests** covering provider abstractions, structured output (per-provider), semantic similarity helpers, bright-line rule overrides, and graceful degradation paths. Suite at 133 tests / 91% coverage.

### Changed
- **Default behaviour flipped** — `REGPILOT_INTAKE_FAST`, `REGPILOT_RERANK_FAST`, `REGPILOT_SYNTH_FAST` all default to `false` now (LLM-primary). Set any to `true` to fall back to the deterministic regex/template path for that node (useful on CPU-only Ollama).
- **Intake node** now calls `llm.generate_structured(IntakeSchema)` by default; regex heuristic survives as a fallback for LLM failures.
- **Synthesizer** now generates the narrative sections (executive summary, risk classification rationale, recommended next steps) via `llm.generate_structured(ReportSections)`. The deterministic scaffold (obligations table, lifecycle mapping, frameworks alignment) is preserved for grounding — every citation flows from the deadline calculator or retrieved chunks, never from the LLM's imagination.
- **Risk classifier** rebuilt: bright-line rules (Article 5, Article 51 GPAI threshold) run first, then semantic Annex III matching surfaces candidates, then the LLM returns the final verdict via structured output. Confidence field exposes whether the verdict came from a rule (`1.0`) or the LLM (`0.7`-`0.85`).
- **Ollama HTTP timeout raised** from 30 s to 60 s in docker-compose because LLM-primary mode runs longer single calls.
- **Streamlit healthcheck `start_period` raised** from 30 s to 60 s for the longer warmup.

### Added — earlier (still unreleased)
- Structured JSON logging (`REGPILOT_LOG_JSON=true`) via `regpilot.observability._JsonFormatter`.
- Per-node exception capture decorator (`trace_node`) — failing LLM calls bump `state["error_count"]` instead of crashing the chain.
- Optional Langfuse tracing hook (env-gated; no-op if creds missing).
- `CHANGELOG.md`, `SECURITY.md`, `CODEOWNERS`, `Makefile` for repo governance.
- Verb-form combo patterns (preserved as `_ART5_COMBO_PATTERNS` for bright-line Article 5 paraphrases that are still legally enumerated).

### Fixed
- Edge-case stress test from the previous round exposed three classification gaps; all fixed by the architectural refactor:
  - Frontier LLM descriptions now route to `general_purpose_systemic` via the GPAI bright-line rule + first-class tier vocabulary.
  - Social-scoring paraphrases route to `prohibited` via the Article 5 bright-line rule + verb-form combo patterns (`5(1)(c)`).
  - Biometric verb-form descriptions ("analyses emotions", "detects faces") now route via the semantic matcher OR the LLM verdict — no more hand-written regex chase.

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
