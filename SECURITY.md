# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.4.x | ✅ |
| < 0.4 | ❌ |

## Reporting a vulnerability

If you discover a security issue, please **do not open a public issue**.

Email the maintainer at <gyorgy.varga@shiwaforce.com> with:

1. A description of the issue + reproducer.
2. The affected component and version.
3. Any mitigation you've already explored.

You should receive an acknowledgement within **3 working days**. We aim to publish a fix within 14 days for high-severity issues.

## Scope

This is a homework / demonstration project; the supported attack surface is intentionally small:

| Component | In scope | Notes |
|---|---|---|
| Streamlit UI | ✅ | XSS / injection in user-supplied descriptions |
| OllamaClient | ✅ | Server-side request forgery via `OLLAMA_BASE_URL` |
| Chroma vector store | ✅ | Local file paths only; no remote access |
| EU AI Act PDF download | ✅ | Pinned to `publications.europa.eu` CELEX URL |
| `langgraph-checkpoint-sqlite` persistence | ✅ | Single-process; no auth (use Postgres for multi-tenant) |
| Container image | ✅ | Multi-stage, non-root user (`app:app`), pinned base |

## Out of scope

- **AuthN/AuthZ** — not implemented (the README is explicit about this). For production deployment behind a corporate SSO, add a reverse proxy (oauth2-proxy / Pomerium / Authentik) or migrate to a managed LangGraph deployment.
- **PII handling** — user descriptions are stored in the SqliteSaver checkpoint when `REGPILOT_CHECKPOINTER=sqlite`. Don't paste sensitive personal data into the UI.
- **Legal advice** — outputs are first-pass triage only. Never act on RegPilot output without a human lawyer in the loop.

## Hardening notes for production deployment

- Set `REGPILOT_LOG_JSON=true` and ship logs to a SIEM.
- Use `langgraph-checkpoint-postgres` with TLS instead of SQLite for multi-tenant deployments.
- Put a reverse proxy in front of port 8501 (TLS termination + rate limiting).
- Pin Ollama image by digest, not just tag, in production registries.
- Enable Streamlit's `--server.enableCORS=false --server.enableXsrfProtection=true`.
