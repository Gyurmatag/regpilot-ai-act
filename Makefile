# RegPilot — common ops. Reviewer-friendly one-liners.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
STUB ?= REGPILOT_LLM=stub

.PHONY: help install lint type test cov eval loadtest ingest run-stub run-docker stop-docker fmt clean ci

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create venv + install dev deps.
	python3 -m venv .venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[dev]"

lint: ## Ruff lint (CI gate).
	$(PY) -m ruff check src tests scripts

type: ## Mypy static type check (CI gate).
	$(PY) -m mypy src

test: ## Pytest with coverage gate (CI gate).
	$(STUB) $(PY) -m pytest --cov=regpilot --cov-fail-under=70

cov: ## Pytest with coverage report.
	$(STUB) $(PY) -m pytest --cov=regpilot --cov-report=term-missing

eval: ## Functional eval against the 16-question gold set.
	$(STUB) $(PY) scripts/evaluate.py

loadtest: ## 100-query async load test.
	$(STUB) $(PY) scripts/loadtest.py --n 100 --concurrency 8 --quiet

ingest: ## Download EU AI Act + index into Chroma (skips if PDF cached).
	$(STUB) $(PY) scripts/ingest.py

run-stub: ## Local Streamlit on :8501 with the deterministic stub LLM.
	$(STUB) $(PY) -m streamlit run src/regpilot/ui/app.py --server.headless true

run-docker: ## Full docker stack (Ollama + ingest + Streamlit on :8501).
	docker compose up --build -d
	@echo "→ http://localhost:8501 (give Ollama ~5 min on first boot to pull models)"

stop-docker: ## Tear down docker stack (preserves volumes).
	docker compose down

fmt: ## Auto-fix lint issues.
	$(PY) -m ruff check --fix src tests scripts

clean: ## Wipe caches, pycache, dist.
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info

ci: lint type test ## Run every CI gate locally.
