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
