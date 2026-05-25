# Contributing

This is a homework / demonstration repo, so the contribution model is light.
A few conventions to know if you're sending a PR or just reading the code.

## Getting set up

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make ci          # ruff + mypy + pytest with 90% coverage gate
```

For the live LLM path:

```bash
docker compose up --build
# → http://localhost:8501
```

## Conventions

- **No `Co-authored-by:` trailers from AI assistants.** This repo's history
  is single-author by design.
- **Commits are scoped** (`feat(classifier): …`, `fix(determinism): …`,
  `docs: …`) and squashable. The git log is the change log.
- **Tests live next to the code they test.** Add a test for any new code
  path; the CI gate is 90% line coverage.
- **Stub LLM for CI.** `REGPILOT_LLM=stub` is set in the pytest fixtures so
  CI runs offline and deterministically. The hosted-provider clients
  (`OpenAI`, `Anthropic`) get coverage through mocked SDKs.
- **`make ci` must be green** before opening a PR. The full gate runs in
  about 30 seconds locally.
- **Markdown is for humans.** Prefer prose over heavy bullet lists; tables
  only where they actually add value. The auto-generated `evaluation/`
  result files inherit a stub-backend caveat block when relevant.

## Repo map

```
src/regpilot/
├── llm/             provider abstraction (Ollama / OpenAI / Anthropic / stub)
├── ingestion/       PDF loader, article-aware chunker, Annex III data
├── rag/             embeddings, vector store, hybrid retriever, RAG subgraph
├── tools/
│   ├── risk_classifier/   bright_lines + semantic + llm_verdict
│   ├── deadline_calculator.py
│   └── citation_validator.py
├── agents/
│   ├── intake, triage, prohibited, obligation_mapper, synthesizer, validator
│   └── _synth_scaffold.py   deterministic report scaffold (constants + helpers)
├── evaluation/      metrics, runner, report, CLI for the functional eval
├── ui/              Streamlit app
├── cli.py           regpilot-ingest / regpilot-eval / regpilot-loadtest console scripts
├── loadtest.py      async harness + per-node instrumentation + report writer
├── observability.py trace_node + JSON logging + request-id context
├── schemas.py       central Pydantic schemas the LLM fills in
├── config.py        pydantic-settings — every env var lives here
├── state.py         the LangGraph TypedDict + RiskTier / UserRole literals
└── graph.py         workflow assembly + run() entry point
```

The `scripts/{ingest,evaluate,loadtest}.py` files are one-import shims that
delegate to `regpilot.cli` so `python scripts/...` and the installed
console-script binaries stay in lock-step.

Test files mirror the source layout under `tests/`.

## What to read first

- `src/regpilot/graph.py` — the main LangGraph workflow.
- `src/regpilot/tools/risk_classifier/__init__.py` — bright-line rules +
  semantic similarity + LLM verdict, orchestrated.
- `src/regpilot/rag/subgraph.py` — the 4-node RAG subgraph.
- `src/regpilot/schemas.py` — the LLM contract surface in one file.
- `README.md` — architecture overview, eval results, production knobs.

## Code-quality bar

- `ruff check` clean (config in `pyproject.toml`).
- `mypy src` clean.
- `pytest --cov-fail-under=90`.

If you add a new agent node, a new tool, or a new LLM backend, please
also extend `README.md`'s "Where the LLM actually runs" table so the
honest-disclosure narrative stays in sync with the code.
