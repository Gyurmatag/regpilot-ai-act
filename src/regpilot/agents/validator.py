"""Validator node.

Self-critique that calls ``citation_validator_tool`` and either signs off
(``ok``) or appends issues for the mapper to address. The graph's conditional
edge consults ``validation_issues`` + ``validator_loops`` and loops back if
needed; once the loop cap is reached it forces an exit.
"""

from __future__ import annotations

import logging

from regpilot.config import settings
from regpilot.state import RegPilotState, TraceEvent
from regpilot.tools.citation_validator import validate

logger = logging.getLogger(__name__)


def validator(state: RegPilotState) -> RegPilotState:
    draft = state.get("draft_report", "")
    report = validate(draft)
    loops = state.get("validator_loops", 0)

    issues = list(report.issues)
    if not draft.strip():
        issues.append("Draft report is empty.")

    final_report = ""
    if report.ok:
        # The synthesizer already cites inline — no need to append a footer.
        final_report = draft
    elif loops + 1 >= settings.max_validator_loops:
        # Out of retries — still emit the draft so the user gets something
        # actionable, but flag the unresolved issues.
        flagged = "\n".join(f"> WARNING: {i}" for i in issues)
        final_report = f"{flagged}\n\n{draft}".strip()

    return {
        "validation_issues": issues,
        "validator_loops": loops + 1,
        "final_report": final_report,
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="validator",
                summary=(
                    f"ok — cited {len(report.cited_articles)} Articles"
                    if report.ok
                    else f"found {len(issues)} issue(s) (loop {loops + 1}/{settings.max_validator_loops})"
                ),
                payload={
                    "ok": report.ok,
                    "issues": issues,
                    "invalid_articles": sorted(report.invalid_articles),
                    "cited_articles": sorted(report.cited_articles),
                },
            ),
        ],
    }


def route_after_validator(state: RegPilotState) -> str:
    """Conditional edge: loop back to obligation_mapper or finish."""

    issues = state.get("validation_issues", [])
    loops = state.get("validator_loops", 0)
    if issues and loops < settings.max_validator_loops:
        return "obligation_mapper"
    return "__end__"
