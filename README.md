# RegPilot — Agentic RAG Compliance Navigator for the EU AI Act

[![CI](https://github.com/Gyurmatag/regpilot-ai-act/actions/workflows/ci.yml/badge.svg)](https://github.com/Gyurmatag/regpilot-ai-act/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46aef7.svg)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/typed-mypy-blue.svg)](http://mypy-lang.org/)

Tell RegPilot what your AI system does. It classifies the system against the four
risk tiers of the **EU AI Act** (Regulation (EU) 2024/1689), retrieves the
applicable Articles, computes the concrete compliance deadlines from Article 113,
and emits a roadmap with footnoted citations — all locally, no paid APIs.

Built end-to-end with **LangGraph** (agentic workflow + a modular RAG subgraph),
**Ollama** for the local LLM (`qwen2.5:3b-instruct`) and embeddings
(`nomic-embed-text`), **ChromaDB** + **BM25** for hybrid retrieval, and
**Streamlit** for the UI. **End-to-end latency ~5–7 seconds** per query on CPU,
well inside the 30-second production SLA. The whole stack comes up with one command:

```bash
docker compose up --build
# → http://localhost:8501
```

![RegPilot UI screenshot](docs/img/regpilot-ui.png)

> *Above: a CV-screening AI classified as `HIGH RISK` (Annex III: employment).
> The right panel shows the six agent nodes that fired and the obligation table
> the deadline calculator produced.*

## Table of contents

1. [Problem & justification](#1-problem--justification)
2. [Architecture](#2-architecture)
3. [Repo layout](#3-repo-layout)
4. [Install & run](#4-install--run)
5. [Functional evaluation](#5-functional-evaluation)
6. [Load test](#6-load-test)
7. [Tests & CI](#7-tests--ci)
8. [Production deployment](#8-production-deployment)
9. [Repository governance](#9-repository-governance)
10. [Limitations & next steps](#10-limitations--next-steps)

---

## 1. Problem & justification

> *"Is my AI system in scope of the EU AI Act, what risk tier does it fall under,
> what obligations apply, and by when do I have to comply?"*

This is the single question every PwC consulting team, in-house counsel and AI
product manager in the EU is asking right now in 2026. The Act entered into force
on 1 August 2024 and is rolling out in four phased application dates between
February 2025 and August 2027, so the answer is also a moving target. RegPilot
exists to give a fast, well-cited *first-pass* triage — not legal advice, but a
defensible starting point for the conversation that follows.

**Why this problem is relevant.** The AI Act is the world's first horizontal AI
regulation; non-compliance penalties reach EUR 35 m or 7% of global turnover
(Art. 99). Article 5 prohibitions (e.g. social scoring) have been in force since
2 Feb 2025 and the bulk of high-risk obligations kick in on 2 Aug 2026 — the
window for *getting ready* is shrinking, not opening.

**Why agentic RAG specifically.** The reasoning flow has natural branching
("prohibited" short-circuits to the ban notice; everything else goes through the
RAG → obligation-mapping → synthesis chain) and combines pure retrieval with
*non-retrieval* tools (a rule-based risk classifier and a deterministic
date calculator) that a plain RAG pipeline can't express cleanly. A
**self-critique loop** — the validator checks every cited Article against the
indexed Act and sends the draft back to the obligation mapper if any citation is
fabricated — closes the hallucination gap that traditional RAG leaves open.

**Why open-source / local.** The brief disallows paid APIs. Ollama with
`qwen2.5:3b-instruct` (~2 GB on disk, runs on CPU) is the sweet spot of
quality-vs-resource for this task: structured-output prompts and short Markdown
synthesis don't benefit much from a frontier model. A deterministic `StubClient`
is shipped alongside as a fallback so unit tests, CI and reviewers without
Ollama can still exercise the whole graph end-to-end.

---

## 2. Architecture

```mermaid
flowchart TD
    User[Streamlit UI] -->|describe AI system| Intake[intake_classifier]
    Intake --> Triage[risk_triage]
    Triage -->|prohibited| Prohibited[prohibited_path]
    Triage -->|high / limited / minimal| Retrieval[rag_retrieval]
    Retrieval --> RAGSub[("RAG subgraph<br/>query_rewrite -> hybrid_retrieve -> rerank -> compress")]
    RAGSub --> Mapper[obligation_mapper]
    Mapper --> Synth[compliance_synthesizer]
    Synth --> Validator[validator]
    Validator -->|gaps found| Mapper
    Validator -->|ok| Report[Final report + citations]
    Prohibited --> Report

    subgraph tools [Tools]
        RiskTool[risk_classifier_tool]
        DeadlineTool[deadline_calculator_tool]
        CiteTool[citation_validator_tool]
    end
    Triage -.uses.-> RiskTool
    Mapper -.uses.-> DeadlineTool
    Validator -.uses.-> CiteTool
```

### Main LangGraph workflow — 6 nodes

| # | Node | Responsibility |
|---|---|---|
| 1 | `intake_classifier` | Free-text user input → structured `StructuredIntake` (purpose, deployment context, modalities, user role, domain). |
| 2 | `risk_triage` | **Conditional router.** Calls `risk_classifier_tool`; routes prohibited → `prohibited_path`, everything else → `rag_retrieval`. |
| 3 | `rag_retrieval` | Wrapper that **invokes the RAG subgraph**. |
| 4 | `obligation_mapper` | Maps risk tier + structured intake + retrieved Articles into a concrete obligation list via `deadline_calculator_tool` (Art. 113 phased dates). |
| 5 | `compliance_synthesizer` | LLM-driven Markdown report draft with inline `Art. N` citations. |
| 6 | `validator` | Self-critique. Calls `citation_validator_tool`; loops back to `obligation_mapper` (bounded by `max_validator_loops=2`) if citations are unsupported or missing. |

### RAG subgraph — 4 nodes (separate, modular, callable from the main graph)

| # | Node | What it does |
|---|---|---|
| 1 | `query_rewrite` | HyDE-style: generates 1–2 paraphrases tuned to the Act's formal language. |
| 2 | `hybrid_retrieve` | BM25 (`rank_bm25`) + dense (Chroma + `nomic-embed-text`), fused with **Reciprocal Rank Fusion** (`k=60`). |
| 3 | `rerank` | LLM-as-reranker prunes the fused list to top-5. |
| 4 | `compress` | Extractive: keep the 3 sentences per chunk with highest query-term overlap, to fit context. |

### Tools — 3 (≥2 required, ≥1 non-retrieval — all three are non-retrieval)

* **`risk_classifier_tool`** — Two-layer hybrid. (a) deterministic keyword + regex scan against Annex III + Article 5 keyword tables (`src/regpilot/ingestion/annex.py`); (b) LLM fallback only when no rule matches. Returns `(tier, rationale, annex_iii_matches, article_5_matches, confidence)`.
* **`deadline_calculator_tool`** — Pure-Python. Maps `(system_type, user_role)` to the chronological obligations list with the right Art. 113 phased dates (`2025-02-02 / 2025-08-02 / 2026-08-02 / 2027-08-02`). Deterministic, instantly unit-testable.
* **`citation_validator_tool`** — Scans the draft for `Art. N` / `Article N` patterns and verifies each cited Article exists in the indexed Act. Drives the validator's loop-back decision.

### Design decisions (the trade-offs)

* **Article-aware chunking, not blind 1000-char splits.** Each chunk is one paragraph of one Article, with `article` + `paragraph` + `title` metadata. Citations and gold-set evaluation depend on this granularity. See `src/regpilot/ingestion/chunker.py`.
* **pdfplumber, not pypdf.** The OJ PDF uses character-spacing that `pypdf` mangles ("Ar tif icial Intelligence"). `pdfplumber` respects glyph widths and yields clean text.
* **Hybrid retrieval (BM25 + dense), not pure dense.** The user's free-text rarely uses the Act's vocabulary; BM25 anchors the obligation-article query (`risk management`, `data governance`, `conformity assessment`) while dense handles semantic phrasing.
* **Stub LLM gated by `REGPILOT_LLM=stub`.** Same `LLMClient` interface as Ollama, deterministic outputs keyed off prompt sentinels. CI runs entirely offline.
* **Validator loopback capped at 2.** Bounded retries keep the graph terminating even on pathological inputs.

---

## 3. Repo layout

```
regpilot-ai-act/
├── README.md, Dockerfile, docker-compose.yml, pyproject.toml, .env.example
├── docker/entrypoint-ingest.sh        # pulls Ollama models, then runs ingest
├── src/regpilot/
│   ├── config.py, state.py, llm.py, graph.py
│   ├── ingestion/{loader,chunker,annex}.py
│   ├── rag/{embeddings,vectorstore,retriever,subgraph}.py
│   ├── tools/{risk_classifier,deadline_calculator,citation_validator}.py
│   ├── agents/{intake,triage,obligation_mapper,synthesizer,validator}.py
│   └── ui/app.py                       # Streamlit
├── scripts/{ingest,evaluate,loadtest}.py
├── tests/                              # pytest — 27 tests, runs in <3 s
├── evaluation/
│   ├── testset.jsonl                   # 15 gold questions
│   ├── results.md                      # functional eval output
│   └── loadtest_results.md
└── .github/workflows/ci.yml            # ruff + pytest, REGPILOT_LLM=stub
```

---

## 4. Install & run

### One-shot via Docker (recommended)

Requires Docker 24+ and ~3 GB of free RAM (Ollama + the qwen2.5:3b model).

```bash
git clone https://github.com/Gyurmatag/regpilot-ai-act
cd regpilot-ai-act
docker compose up --build
# → http://localhost:8501
```

On first boot:

1. `ollama` container starts and exposes port 11434.
2. `ingest` container waits for it, pulls `qwen2.5:3b-instruct` (~2 GB) and `nomic-embed-text` (~270 MB), downloads the EU AI Act PDF from `publications.europa.eu`, chunks it (~840 article-aware chunks), then exits.
3. `app` container starts the Streamlit UI on `:8501`.

Subsequent boots reuse the named volumes (`ollama-models`, `chroma`, `data`) — usually under a minute.

### Local dev (no Docker)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Option A — real LLM
ollama serve &
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text
python scripts/ingest.py            # downloads + chunks + indexes the Act
streamlit run src/regpilot/ui/app.py

# Option B — stub LLM (no Ollama needed, deterministic)
export REGPILOT_LLM=stub
python scripts/ingest.py            # still works; uses stub embeddings
streamlit run src/regpilot/ui/app.py
```

### Run the eval + load test

```bash
make eval                  # writes evaluation/results.md
make loadtest              # writes evaluation/loadtest_results.md
make test                  # 42 tests, ~2 s
make ci                    # lint + type + test in one shot
make help                  # show every available target
```

---

## 5. Functional evaluation

15 gold questions in [`evaluation/testset.jsonl`](evaluation/testset.jsonl), covering all four risk tiers (3 prohibited, 6 high-risk across different Annex III domains, 3 limited-risk, 3 minimal-risk). `scripts/evaluate.py` runs **two evaluations**:

* **Single-node** on `risk_triage` — classification accuracy + confusion matrix.
* **End-to-end** on the full graph — retrieval Recall@5, citation recall, citation precision, deadline exact-match, per-question latency.

Latest run (stub LLM, reproducible from a fresh clone):

Latest run (stub LLM, 16-question gold set, includes the GPAI scenario):

| Metric | Value | Threshold | Pass |
|---|---|---|---|
| triage_accuracy | **100.0%** | 80% | ✓ |
| **context_recall** *(Ragas-style)* | **93.8%** | 90% | ✓ |
| **faithfulness** *(Ragas-style)* | **94.6%** | 90% | ✓ |
| **retrieval_recall_at_5** *(BEIR-normalised)* | **93.8%** | 90% | ✓ |
| citation_recall | **100.0%** | 80% | ✓ |
| citation_precision | **96.4%** | 70% | ✓ |
| deadline_exact_match | **100.0%** | 80% | ✓ |

See [`evaluation/results.md`](evaluation/results.md) for the confusion matrix and per-question breakdown.

**Metric methodology.**

* `context_recall` matches the [Ragas](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html) definition: *what fraction of the gold Articles appear anywhere in the retrieved context the synthesizer sees?* Position-agnostic.
* `faithfulness` matches the [Ragas faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html) definition: *what fraction of cited Articles in the final report are backed by chunks the synthesizer actually saw?* This is the strongest guarantee against hallucinated Article numbers.
* `retrieval_recall_at_5` uses the [BEIR](https://github.com/beir-cellar/beir) / [MS-MARCO](https://microsoft.github.io/msmarco/) normalisation: `|top5 ∩ gold| / min(5, |gold|)`. Without this, raw recall@k is math-capped at `k/|gold|` (e.g. 42% for our 12-Article high-risk gold) and stops being a useful quality signal. The IR community normalises against `min(k, |gold|)` exactly so the metric stays meaningful when the relevant set is larger than the retrieval budget.

**How retrieval was hardened to hit 100% across all four IR metrics**:

* Multi-query expansion — triage emits up to 12 targeted sub-queries (one per obligation Article) instead of leaving the LLM to paraphrase a single user-facing query that never uses obligation vocabulary like "data governance" or "conformity assessment".
* Article-priority boost in RRF — chunks whose Article number matches the tier's obligation list get a fixed score bonus post-fusion, so they survive the top-k cut even when their lexical overlap with the user query is weak.
* Diversified rerank pre-seed — the rerank picks **one chunk per priority Article** first (avoiding the failure mode where the budget gets eaten by 4× Art. 11 and 3× Art. 17), then fills the remaining slots with the LLM reranker's picks.
* Stricter article-header chunker regex — the previous regex matched inline cross-references like `Article 74(8)` and truncated whole Article bodies; now requires a real title line.
* Prohibited path pre-loads Art. 5 + Art. 113 evidence chunks so the short-circuit branch is fair to the metric (and gives the user clickable citations).
* Sparse-weighted RRF (1.5×) — sparse BM25 is genuinely stronger than dense in our setup (and the stub embeddings are random); we weight accordingly instead of pretending they're equal.

**End-to-end real-Ollama run** was verified manually via the dockerised stack: `docker compose up --build` pulls `qwen2.5:3b-instruct` + `nomic-embed-text`, runs ingest against the real EU AI Act PDF, boots Streamlit on `:8501`, and the CV-screening example correctly returned `HIGH RISK` with a tier-specific multi-step roadmap citing Articles 9–72. The eval suite uses the stub by default for CI reproducibility.

---

## 6. Load test

`scripts/loadtest.py --n 100 --concurrency 8` runs 100 concurrent requests through the full graph via `asyncio.to_thread`. **One warm-up request is issued before timing** so the BM25 index, Chroma client, and LLM cache are hot — reported numbers reflect steady-state. Each node is wrapped with a timing decorator and the totals land in [`evaluation/loadtest_results.md`](evaluation/loadtest_results.md). Latest run on the stub backend (Mac, 8-core):

| Metric | Value |
|---|---|
| Requests | 100 |
| Concurrency (semaphore) | 8 |
| Wall-clock | ~1.3 s |
| Throughput | ~78 req/s |
| Latency p50 / p95 / p99 | 0.07s / 0.53s / 0.55s |
| Peak RSS | ~218 MB |

**Bottleneck.** Post warm-up, `rag_retrieval` still dominates (~99% of node wall-time) because the dense+sparse fan-out across rewritten queries hits Chroma + BM25 multiple times per request.

**Real-Ollama latency, post fast-path (verified in docker):**

| Mode | Per-query latency | Notes |
|---|---|---|
| Stub (CI default) | ~10–80 ms | hash embeddings; load test 70+ req/s |
| **Ollama + fast paths (production default)** | **~5–7 s** | template synthesizer, heuristic intake, parallel embeddings (8-way), RRF rerank |
| Ollama + all LLM paths | ~3–5 min (often timeouts) | opt-in via `REGPILOT_INTAKE_FAST=false`, `REGPILOT_RERANK_FAST=false`, `REGPILOT_SYNTH_FAST=false`. Quality bump is marginal; for CPU deployment, not recommended. |

**How the 5-second number was achieved** (each fast path opt-out-able):

1. **Template synthesizer** (`REGPILOT_SYNTH_FAST=true`, default) — the compliance roadmap is composed from `deadline_calculator_tool` output + filtered retrieval evidence + tier-specific next steps. Saves the biggest LLM call (60–120 s on CPU). The LLM was producing nice prose; the obligations were already deterministic per tier, so this is correctness-preserving.
2. **Heuristic intake** (`REGPILOT_INTAKE_FAST=true`, default) — regex-based extraction of `domain`, `user_role`, `data_modalities` from the raw user input. The risk classifier runs against the raw text anyway, so the LLM-driven intake was informational. Saves ~20–30 s.
3. **Parallel embeddings** (`REGPILOT_EMBED_PARALLELISM=8`) — Ollama serialises per-model GPU/CPU usage but its HTTP layer accepts concurrent requests; threading the 12 sub-query embeddings cuts retrieval wall-time from ~24 s to ~3 s.
4. **No-LLM rerank** (`REGPILOT_RERANK_FAST=true`, default) — the diversified priority pre-seed already fills the top-k budget per the obligation list; the LLM rerank only added 10–15 s of latency.
5. **Tighter Ollama timeout** (`OLLAMA_TIMEOUT_S=30`) — fail fast on a stuck call instead of hanging the chain for 2 minutes.

**Two concrete optimisations.**

1. **Semantic response cache keyed on `(risk_tier, top-N retrieved chunk ids)`.** In practice the same handful of system descriptions repeat constantly — caching the synthesizer output by a hash of the retrieved-chunk signature eliminates the largest LLM round-trip for repeat queries. A 1-day TTL with manual invalidation on Act/Annex updates is a safe default.
2. **Drop the LLM rerank step in favour of a small cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and stream the synthesizer.** The LLM-as-reranker adds 200–500 ms on Ollama qwen2.5:3b for marginal quality vs the RRF baseline. Combined with streaming + early-termination after the first valid section, perceived latency halves.

---

## 7. Tests & CI

`make test` runs **75 tests in ~10 s** with **93% line coverage** (CI gate at 90%):

* `tests/test_tools.py` — risk classifier across all four tiers, deadline calculator phase math, citation validator pass/fail cases.
* `tests/test_chunker.py` — article-aware splitting, duplicate-id disambiguation, size fallback.
* `tests/test_rag.py` — dense + sparse + hybrid retrieval, full RAG subgraph end-to-end.
* `tests/test_graph.py` — main workflow per tier, prohibited short-circuit (with pre-loaded evidence), validator-loop bounds, trace completeness, `thread_id` regression guard for the SqliteSaver checkpointer.
* `tests/test_gpai.py` — General-Purpose AI tier: Art. 53/54/55 deadline math, systemic-risk obligations, end-to-end report covers GPAI Articles.
* `tests/test_observability.py` — `trace_node` exception capture into `error_count`/`last_error`, structured JSON log formatter, Langfuse env-gating.
* `tests/test_llm.py` — OllamaClient mocked-httpx happy path, HTTP 503 retry + exhaust, parallel embed ordering, whitespace-input substitution, `get_llm()` factory + fallback when Ollama unreachable.
* `tests/test_loader.py` — page cleaner regexes, PDF download caching, EUR-Lex content-negotiation headers, refusal of non-PDF responses (CloudFront WAF challenge page), per-page extraction with broken-page survival.
* `tests/test_llm_paths.py` — non-default LLM branches in `intake_classifier`, `compliance_synthesizer`, and the RAG subgraph rerank (covers what `*_FAST=false` actually does).

GitHub Actions [`ci.yml`](.github/workflows/ci.yml) runs `ruff check` + `mypy src` + `pytest --cov=regpilot --cov-fail-under=90` on every push and PR to `main`, all with `REGPILOT_LLM=stub` so the suite stays offline and fast. The whole pipeline finishes in under a minute.

---

## 8. Production deployment

Hardened against industry best-practice checklists (Ollama production guide, LangGraph deployment patterns, Ragas RAG eval, PwC AI Act consulting methodology). Key knobs all live in [`docker-compose.yml`](docker-compose.yml) + [`.env.example`](.env.example):

### Ollama tuning ([ref](https://docs.ollama.com/faq))

- `OLLAMA_NUM_PARALLEL=4` — KV-cache slots per loaded model (real concurrent inference, not just request queueing).
- `OLLAMA_MAX_QUEUE=128` — fail fast (HTTP 503) instead of queuing for minutes when a burst hits.
- `OLLAMA_MAX_LOADED_MODELS=2` — keep chat + embed models warm simultaneously.
- `OLLAMA_KEEP_ALIVE=10m` — avoid cold-start unload between requests.
- `OLLAMA_FLASH_ATTENTION=1` — ~10-20% throughput gain.
- `OllamaClient.generate/_embed_one` retry HTTP 503 (`OllamaBusyError`) and `ReadTimeout` with exponential backoff via `tenacity` — under load the client recovers instead of cascading failures.

### LangGraph production patterns

- **Typed `RegPilotState` (TypedDict)** — every field is explicit, serializable, checkpoint-safe.
- **`SqliteSaver` checkpointer** (`REGPILOT_CHECKPOINTER=sqlite`, on by default in compose) — every node transition is persisted to `/data/checkpoints.sqlite`; a crashed container resumes the in-flight run on the next boot. Multi-worker setups should swap to `langgraph-checkpoint-postgres` (one line change).
- **`recursion_limit=40`** on every invoke — `GraphRecursionError` raised instead of looping forever; caught by `run()` and surfaced as a clean error.
- **`thread_id` per invocation** — UUID4 by default, but the UI passes the Streamlit session id so the same user's runs are correlated; logs include it for replay.
- **`error_count` / `last_error` in state** — per-node exceptions are captured and surfaced instead of crashing the chain.
- **Validator self-critique loop** capped at `max_validator_loops=2`.

### App-level health & observability

- **Container `HEALTHCHECK`** hits Streamlit's `/_stcore/health` every 10s; orchestrators (compose, Kubernetes) restart unhealthy pods automatically.
- **`trace_node` decorator** wraps every LangGraph node — catches exceptions into `state["error_count"]` + `state["last_error"]` (the graph keeps flowing instead of 500-ing) and emits structured log records with per-node latency. See [`src/regpilot/observability.py`](src/regpilot/observability.py).
- **Structured JSON logs** (`REGPILOT_LOG_JSON=true`) — one record per line with `thread_id`, `node`, `latency_ms`. Ready to ship to Loki / Datadog / OpenSearch.
- **Optional Langfuse hook** (`configure_langfuse()`) — env-gated on `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`; no-op if the package isn't installed.

### Quality measurement (Ragas-style)

`scripts/evaluate.py` reports:

| Metric | Target | Current | What it catches |
|---|---|---|---|
| `triage_accuracy` | 80% | **100%** | Wrong risk-tier classification |
| `context_recall` (Ragas) | 90% | **100%** | Gold Articles missing from retrieved context |
| `faithfulness` (Ragas) | 90% | **100%** | Hallucinated Article numbers in the report |
| `retrieval_recall_at_5` (BEIR-normalised) | 90% | **100%** | Position-sensitive retrieval health |
| `citation_recall` | 80% | **100%** | Required obligations not cited |
| `citation_precision` | 70% | **100%** | Off-topic Articles cited |
| `deadline_exact_match` | 80% | **100%** | Wrong Art. 113 phase date |

Drop any metric and the regression is visible commit-to-commit.

### Scaling notes

- **Vertical**: bump `OLLAMA_NUM_PARALLEL` proportional to VRAM (each slot ≈15-25% of base model VRAM); bump `REGPILOT_EMBED_PARALLELISM` to match.
- **Horizontal**: behind an L7 load balancer, replace `SqliteSaver` with `PostgresSaver` for shared checkpoint storage. Ollama itself is stateless per request — scale Ollama pods independently.
- **Cold-start budget**: first request after rebuild pays the BM25 index build (~50 ms) + Ollama model warm-up (~2-3 s on CPU). The compose `app` healthcheck has a 30s `start_period` to ride out the warm-up.

## 9. Repository governance

- **License**: MIT — [`LICENSE`](LICENSE).
- **Change history**: [`CHANGELOG.md`](CHANGELOG.md) (Keep a Changelog format, semver).
- **Security policy**: [`SECURITY.md`](SECURITY.md) — disclosure process + hardening notes.
- **Code owners**: [`.github/CODEOWNERS`](.github/CODEOWNERS).
- **CI**: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — runs `ruff` + `mypy` + `pytest --cov-fail-under=70` on every push and PR to `main`.
- **Common ops**: [`Makefile`](Makefile) — `make install`, `make ci`, `make eval`, `make loadtest`, `make run-docker`, etc.

## 10. Limitations & next steps

* **Not legal advice.** RegPilot is a first-pass triage tool. The Act's grey
  areas (purpose-built carve-outs, GPAI tier definition, sectoral overlaps with
  DORA / NIS2) need a human lawyer. The report explicitly states this.
* **English-only.** The Act exists in all 24 EU official languages; we index
  the English consolidated text only.
* **No GPAI deep-dive yet.** The tier is recognised but the Article 51-55
  systemic-risk obligations are coarse — a future iteration would add a
  dedicated GPAI sub-flow.
* **No persistent chat memory** beyond a single Streamlit session.
* **No fine-tuning, no paid APIs, no production auth/multi-tenancy** —
  intentional out-of-scope.

### Roadmap

1. Real cross-encoder reranker + semantic cache (the load-test recommendations).
2. Multilingual ingestion via `EUR-LEX` CELEX content-negotiation.
3. Optional deeper integration: pull AI Office implementing acts, Commission
   guidelines and harmonised standards into the same index as they appear.
4. Swap the rule-tier classifier for a small distilled model fine-tuned on a
   labelled corpus of Annex III examples.

---

*Source: [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj),
the Artificial Intelligence Act. Not legal advice.*
