"""Production observability glue.

Three responsibilities, each independently togglable so the same package runs
clean in CI (stub LLM, plain text logs, no external traces) and in production
(structured JSON logs, optional Langfuse traces, ``thread_id`` correlation):

1. ``configure_logging()`` — opt into structured JSON logs (one record per
   line, includes ``thread_id``, node, latency) for log shippers like Loki,
   Datadog, OpenSearch.
2. ``trace_node(name)`` — decorator that wraps every LangGraph node:
   captures exceptions into ``state["last_error"]`` + ``state["error_count"]``
   instead of crashing the chain, and logs structured per-node timing.
3. ``configure_langfuse()`` — best-effort hookup when the ``LANGFUSE_*`` env
   vars are present. No-op otherwise (so the import is always safe).
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections.abc import Callable
from functools import wraps
from typing import Any

from regpilot.config import settings

_logger = logging.getLogger(__name__)
_configured = False


# --------------------------------------------------------------------------- #
# 1. Logging
# --------------------------------------------------------------------------- #


class _JsonFormatter(logging.Formatter):
    """One-record-per-line JSON formatter — log-shipper friendly."""

    _STD_FIELDS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Surface any extra=... kwargs as top-level fields.
        for k, v in record.__dict__.items():
            if k not in self._STD_FIELDS and not k.startswith("_"):
                payload[k] = v
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Idempotent root-logger setup. Honors ``REGPILOT_LOG_LEVEL`` and
    ``REGPILOT_LOG_JSON``."""

    global _configured
    if _configured:
        return
    root = logging.getLogger()
    # Replace any existing handlers so basicConfig elsewhere doesn't shadow us.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        ))
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    # Tame chatty deps.
    for noisy in ("httpx", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


# --------------------------------------------------------------------------- #
# 2. Node-level exception capture
# --------------------------------------------------------------------------- #


def trace_node(name: str) -> Callable:
    """Decorator: catch + record per-node exceptions instead of crashing the graph.

    On exception, returns a state update that bumps ``error_count`` and
    records ``last_error``. The graph keeps flowing — downstream nodes see
    the error in state and can degrade gracefully (the synthesizer / template
    will produce a flagged report).
    """

    def wrap(fn: Callable[[Any], dict]) -> Callable[[Any], dict]:
        @wraps(fn)
        def inner(state: Any) -> dict:
            t0 = time.perf_counter()
            try:
                result = fn(state)
                dt = time.perf_counter() - t0
                _logger.info(
                    "node ok",
                    extra={"node": name, "latency_ms": int(dt * 1000)},
                )
                return result
            except Exception as exc:
                dt = time.perf_counter() - t0
                tb = traceback.format_exc(limit=2)
                _logger.error(
                    "node failed",
                    extra={
                        "node": name,
                        "latency_ms": int(dt * 1000),
                        "error": str(exc)[:200],
                        "error_type": type(exc).__name__,
                    },
                )
                from regpilot.state import TraceEvent  # local import to avoid cycle
                prev = state if isinstance(state, dict) else {}
                return {
                    "error_count": (prev.get("error_count", 0) or 0) + 1,
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "trace": [
                        *(prev.get("trace", []) or []),
                        TraceEvent(
                            node=name,
                            summary=f"FAILED — {type(exc).__name__}: {str(exc)[:120]}",
                            payload={"error_type": type(exc).__name__, "traceback": tb},
                        ),
                    ],
                }
        return inner
    return wrap


# --------------------------------------------------------------------------- #
# 3. Optional Langfuse tracing
# --------------------------------------------------------------------------- #


def configure_langfuse() -> Any | None:
    """Best-effort Langfuse client. Returns ``None`` when creds are missing
    or the ``langfuse`` package isn't installed (so callers can `if client`)."""

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError:
        _logger.warning(
            "LANGFUSE_* env set but the `langfuse` package isn't installed. "
            "Run `pip install langfuse` to enable tracing."
        )
        return None
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
