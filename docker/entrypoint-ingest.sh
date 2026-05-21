#!/usr/bin/env bash
# One-shot bootstrap: wait for Ollama, pull both models, then run ingestion.
# Idempotent — safe to re-run.

set -euo pipefail

OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
CHAT_MODEL="${REGPILOT_CHAT_MODEL:-qwen2.5:3b-instruct}"
EMBED_MODEL="${REGPILOT_EMBED_MODEL:-nomic-embed-text}"

echo "[ingest] waiting for ollama at ${OLLAMA_URL}..."
for i in $(seq 1 60); do
    if curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        echo "[ingest] ollama is up"
        break
    fi
    sleep 2
done

echo "[ingest] pulling ${CHAT_MODEL}"
curl -fsS -X POST "${OLLAMA_URL}/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${CHAT_MODEL}\",\"stream\":false}" | tail -1 || true

echo "[ingest] pulling ${EMBED_MODEL}"
curl -fsS -X POST "${OLLAMA_URL}/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${EMBED_MODEL}\",\"stream\":false}" | tail -1 || true

echo "[ingest] running ingestion pipeline"
exec python scripts/ingest.py "$@"
