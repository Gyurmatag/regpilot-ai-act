<!--
RegPilot PR template. Keep it short. Reviewers read top-to-bottom.
-->

## What

<!-- One or two sentences: what changes and why. -->

## How to test

<!-- Concrete commands. If only `make ci` is needed, just say so. -->

```bash
make ci
```

## Checklist

- [ ] `make ci` is green (ruff + mypy + 90%+ coverage)
- [ ] Touched the LLM-primary path? Ran `make integration-ollama` locally
- [ ] Updated `README.md` / `SECURITY.md` if behaviour or contract changed
- [ ] If a new eval file is needed: backend-suffixed (`results_<backend>_<suffix>.md`)
- [ ] No `Co-authored-by:` trailers from AI assistants — single-author commits only
