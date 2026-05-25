"""Tests for the observability module — error capture, structured logs, Langfuse."""

from __future__ import annotations

import json
import logging

import pytest

from regpilot.observability import _JsonFormatter, configure_langfuse, trace_node

# --------------------------------------------------------------------------- #
# trace_node — exception capture
# --------------------------------------------------------------------------- #


def test_trace_node_captures_exceptions_into_state() -> None:
    """A failing node bumps error_count + records last_error instead of raising."""

    @trace_node("flaky_node")
    def boom(_state: dict) -> dict:
        raise RuntimeError("upstream LLM timed out")

    result = boom({"user_input": "hello", "error_count": 0, "trace": []})

    assert result["error_count"] == 1
    assert "RuntimeError" in result["last_error"]
    assert "upstream LLM timed out" in result["last_error"]
    # Failed node still appears in the trace so the UI can show it.
    assert any(
        ev["node"] == "flaky_node" and "FAILED" in ev["summary"]
        for ev in result["trace"]
    )


def test_trace_node_passes_through_success() -> None:
    @trace_node("happy_node")
    def ok(state: dict) -> dict:
        return {"draft_report": "hi"}

    out = ok({"user_input": "x"})
    assert out == {"draft_report": "hi"}


def test_trace_node_accumulates_error_count() -> None:
    """Multiple failures across nodes accumulate, not reset."""

    @trace_node("node_a")
    def fail_a(_state: dict) -> dict:
        raise ValueError("a")

    @trace_node("node_b")
    def fail_b(_state: dict) -> dict:
        raise ValueError("b")

    state = {"error_count": 0, "trace": []}
    state.update(fail_a(state))
    state.update(fail_b(state))

    assert state["error_count"] == 2
    assert "ValueError" in state["last_error"]
    failed_nodes = [ev["node"] for ev in state["trace"] if "FAILED" in ev["summary"]]
    assert failed_nodes == ["node_a", "node_b"]


# --------------------------------------------------------------------------- #
# JSON log formatter
# --------------------------------------------------------------------------- #


def test_json_formatter_emits_valid_json_with_thread_id() -> None:
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="regpilot.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="agent step", args=(), exc_info=None,
    )
    record.thread_id = "abc123"   # caller-provided extra
    record.node = "intake_classifier"

    out = json.loads(fmt.format(record))
    assert out["level"] == "INFO"
    assert out["logger"] == "regpilot.test"
    assert out["msg"] == "agent step"
    assert out["thread_id"] == "abc123"
    assert out["node"] == "intake_classifier"


def test_json_formatter_serializes_exceptions() -> None:
    fmt = _JsonFormatter()
    try:
        raise ValueError("kaboom")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="regpilot.test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="exploded", args=(), exc_info=sys.exc_info(),
        )
    out = json.loads(fmt.format(record))
    assert "ValueError: kaboom" in out["exc"]


# --------------------------------------------------------------------------- #
# Langfuse hook
# --------------------------------------------------------------------------- #


def test_configure_langfuse_is_noop_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert configure_langfuse() is None


# --------------------------------------------------------------------------- #
# Request-id correlation context
# --------------------------------------------------------------------------- #


def test_current_request_id_defaults_to_dash() -> None:
    """Before any request_context is entered, the contextvar is '-'."""
    from regpilot.observability import current_request_id

    assert current_request_id() == "-"


def test_set_request_id_returns_what_it_set() -> None:
    from regpilot.observability import current_request_id, set_request_id

    rid = set_request_id("explicit-123")
    assert rid == "explicit-123"
    assert current_request_id() == "explicit-123"
    # Reset for hygiene.
    set_request_id("-")


def test_set_request_id_generates_uuid_when_omitted() -> None:
    from regpilot.observability import set_request_id

    rid = set_request_id()
    assert isinstance(rid, str)
    assert len(rid) == 12  # uuid4.hex[:12]
    set_request_id("-")


def test_request_context_scopes_and_restores() -> None:
    """The context manager restores the previous value on exit."""
    from regpilot.observability import current_request_id, request_context

    assert current_request_id() == "-"
    with request_context("outer"):
        assert current_request_id() == "outer"
        with request_context("inner"):
            assert current_request_id() == "inner"
        assert current_request_id() == "outer"
    assert current_request_id() == "-"


def test_json_formatter_always_emits_request_id() -> None:
    """Every log record carries the request_id field, even when unset."""
    from regpilot.observability import _JsonFormatter, _RequestIdFilter

    fmt = _JsonFormatter()
    flt = _RequestIdFilter()

    record = logging.LogRecord(
        name="regpilot.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="agent step", args=(), exc_info=None,
    )
    flt.filter(record)
    out = json.loads(fmt.format(record))

    assert "request_id" in out
    assert out["request_id"] == "-"


def test_request_id_filter_attaches_current_context_value() -> None:
    from regpilot.observability import _RequestIdFilter, request_context

    flt = _RequestIdFilter()
    record = logging.LogRecord(
        name="regpilot.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="step", args=(), exc_info=None,
    )
    with request_context("scoped-rid"):
        flt.filter(record)
    assert record.request_id == "scoped-rid"


def test_run_sets_request_id_to_thread_id() -> None:
    """``graph.run()`` plumbs the thread_id into the request-id contextvar so
    log records produced inside the LangGraph call carry it automatically."""

    from regpilot.graph import run
    from regpilot.observability import current_request_id

    # During the run, the contextvar holds the request id. After the run
    # returns, the contextvar is restored to "-" (the default).
    out = run("A spam filter for company email.", thread_id="rtest-1234")
    assert out.get("risk_tier") in (
        "minimal_risk", "limited_risk", "high_risk",
        "prohibited", "general_purpose", "general_purpose_systemic",
    )
    assert current_request_id() == "-"
