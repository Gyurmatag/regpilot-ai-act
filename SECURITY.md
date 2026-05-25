# Security

## Reporting a vulnerability

Please don't open a public GitHub issue for anything that looks like a
security problem. Email <gyorgy.varga@shiwaforce.com> with:

1. A short description and a reproducer if you have one.
2. The affected component and version (commit SHA is fine).
3. Any mitigation you've already explored.

I'll acknowledge within 3 working days and aim to publish a fix within
14 days for high-severity issues.

## What's in scope

This is a homework / demonstration project, so the supported attack
surface is small but real:

| Component | In scope | Notes |
|---|---|---|
| Streamlit UI | yes | XSS or injection through the user description field |
| `OllamaClient` | yes | SSRF via `OLLAMA_BASE_URL` |
| ChromaDB vector store | yes | Local file paths only; no remote access |
| EU AI Act PDF download | yes | Pinned to a `publications.europa.eu` CELEX URL |
| `langgraph-checkpoint-sqlite` persistence | yes | Single-process, no auth — use Postgres for multi-tenant |
| Container image | yes | Multi-stage build, non-root user, pinned base |

## What's not in scope

- **Authentication / authorisation.** Not implemented; the README says
  so. For production behind a corporate SSO, put a reverse proxy
  (oauth2-proxy, Pomerium, Authentik) in front of port 8501 or migrate
  to a managed LangGraph deployment.
- **PII handling.** User descriptions land in the SqliteSaver checkpoint
  when `REGPILOT_CHECKPOINTER=sqlite`. Don't paste sensitive personal
  data into the UI.
- **Legal advice.** Outputs are first-pass triage. Never act on RegPilot
  output without a human lawyer in the loop.

## Hardening notes for production

- Turn on structured logs (`REGPILOT_LOG_JSON=true`) and ship them to a
  SIEM. Every record carries a `request_id` for correlation.
- Use `langgraph-checkpoint-postgres` with TLS instead of SQLite for
  multi-tenant deployments.
- Put a reverse proxy in front of port 8501 (TLS termination + rate
  limiting at minimum).
- Pin the Ollama image by digest, not just tag, in production
  registries.
- Enable Streamlit's `--server.enableCORS=false
  --server.enableXsrfProtection=true`.
