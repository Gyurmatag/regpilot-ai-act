# syntax=docker/dockerfile:1.7

# --------------------------------------------------------------------------- #
# Builder stage — installs the package into an isolated /install prefix so the
# final image stays slim.
# --------------------------------------------------------------------------- #
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --prefix=/install .

# --------------------------------------------------------------------------- #
# Runtime stage — slim, non-root, only the wheel artefacts copied across.
# --------------------------------------------------------------------------- #
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REGPILOT_LLM=ollama \
    REGPILOT_CHROMA_DIR=/data/chroma \
    REGPILOT_DATA_DIR=/data/raw

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgl1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY src/regpilot/ui/app.py ./src/regpilot/ui/app.py
COPY docker/entrypoint-ingest.sh /usr/local/bin/entrypoint-ingest.sh
RUN chmod +x /usr/local/bin/entrypoint-ingest.sh

RUN mkdir -p /data/raw /data/chroma && chown -R app:app /data /app
USER app

EXPOSE 8501
CMD ["streamlit", "run", "src/regpilot/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
