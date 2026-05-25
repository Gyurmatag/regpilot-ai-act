"""Main LangGraph workflow.

Production guarantees:

* **Checkpointed state** — when ``REGPILOT_CHECKPOINTER=sqlite``, every node
  transition is persisted by ``SqliteSaver`` keyed on ``thread_id`` so a
  crashed container can resume the in-flight run.
* **Recursion limit** — ``run()`` and ``run_streaming()`` pass
  ``recursion_limit=settings.graph_recursion_limit`` so the validator loop
  can't runaway-recurse.
* **Per-node error capture** — every node wrapper catches exceptions, bumps
  ``error_count`` in state, records ``last_error``, and routes the graph to
  the prohibited / no-op terminal so the user still gets a structured
  response instead of a 500.
* **thread_id correlation** — every invoke gets a UUID4 thread_id surfaced
  in the trace + logs, so a reviewer can replay any production failure.

```
intake_classifier
       │
       ▼
   risk_triage ──prohibited──▶ prohibited_path ──▶ END
       │
       └──(high/limited/minimal)──▶ rag_retrieval ──(subgraph)──▶
                                                                 │
                                                                 ▼
                                                       obligation_mapper ◀──loopback─┐
                                                                 │                   │
                                                                 ▼                   │
                                                       compliance_synthesizer        │
                                                                 │                   │
                                                                 ▼                   │
                                                            validator ─issues?─────────┘
                                                                 │ ok
                                                                 ▼
                                                                END
```

6 main-graph nodes (>= the required 5):
``intake_classifier``, ``risk_triage``, ``rag_retrieval``, ``obligation_mapper``,
``compliance_synthesizer``, ``validator``. ``prohibited_path`` is a short-circuit
leaf; the RAG subgraph is a separate, modular subgraph defined in
``regpilot.rag.subgraph`` and does not count toward the main-graph node budget.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import suppress
from typing import Any

from langgraph.graph import END, START, StateGraph

from regpilot.agents.intake import intake_classifier
from regpilot.agents.obligation_mapper import obligation_mapper
from regpilot.agents.prohibited import prohibited_path
from regpilot.agents.synthesizer import compliance_synthesizer
from regpilot.agents.triage import risk_triage, route_by_tier
from regpilot.agents.validator import route_after_validator, validator
from regpilot.config import settings
from regpilot.observability import (
    configure_logging,
    request_context,
    trace_node,
)
from regpilot.rag.subgraph import build_rag_subgraph
from regpilot.state import RegPilotState, TraceEvent

configure_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Checkpointer (state durability)
# --------------------------------------------------------------------------- #


def _make_checkpointer():
    """Return the checkpointer configured by ``REGPILOT_CHECKPOINTER``.

    * ``memory`` (default) — ephemeral, fine for tests and one-shot CLI runs.
    * ``sqlite``  — file-backed SqliteSaver at ``REGPILOT_CHECKPOINT_PATH``,
      survives container restarts; suitable for single-process production
      (Streamlit app + workers in one container). For multi-worker setups,
      swap to ``langgraph-checkpoint-postgres``.
    """

    if settings.checkpointer.lower() != "sqlite":
        return None

    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    with suppress(Exception):
        saver.setup()
    logger.info("LangGraph checkpointer: SqliteSaver at %s", settings.checkpoint_path)
    return saver


# --------------------------------------------------------------------------- #
# Wrapper nodes that splice the RAG subgraph into the main flow
# --------------------------------------------------------------------------- #


def _make_rag_node(rag_subgraph):
    def rag_retrieval(state: RegPilotState) -> RegPilotState:
        query = state.get("rag_query") or state.get("user_input", "")
        sub_state = {
            "query": query,
            "rewritten_queries": state.get("rag_queries") or [],
            "priority_articles": state.get("priority_articles") or [],
        }
        t0 = time.perf_counter()
        result = rag_subgraph.invoke(sub_state)
        compressed = result.get("compressed") or result.get("reranked") or result.get("candidates", [])
        return {
            "retrieved": compressed,
            "trace": [
                *state.get("trace", []),
                TraceEvent(
                    node="rag_retrieval",
                    summary=f"retrieved {len(compressed)} chunks in {time.perf_counter() - t0:.2f}s",
                    payload={
                        "n_compressed": len(compressed),
                        "rewritten_queries": result.get("rewritten_queries", []),
                        "n_candidates": len(result.get("candidates", [])),
                        "priority_articles": state.get("priority_articles") or [],
                    },
                ),
            ],
        }
    return rag_retrieval


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_main_graph(rag_subgraph: Any | None = None):
    """Compile the full RegPilot workflow."""

    if rag_subgraph is None:
        rag_subgraph = build_rag_subgraph()

    g = StateGraph(RegPilotState)

    g.add_node("intake_classifier", trace_node("intake_classifier")(intake_classifier))
    g.add_node("risk_triage", trace_node("risk_triage")(risk_triage))
    g.add_node("rag_retrieval", trace_node("rag_retrieval")(_make_rag_node(rag_subgraph)))
    g.add_node("obligation_mapper", trace_node("obligation_mapper")(obligation_mapper))
    g.add_node("compliance_synthesizer", trace_node("compliance_synthesizer")(compliance_synthesizer))
    g.add_node("validator", trace_node("validator")(validator))
    g.add_node("prohibited_path", trace_node("prohibited_path")(prohibited_path))

    g.add_edge(START, "intake_classifier")
    g.add_edge("intake_classifier", "risk_triage")

    g.add_conditional_edges(
        "risk_triage",
        route_by_tier,
        {
            "rag_retrieval": "rag_retrieval",
            "prohibited_path": "prohibited_path",
        },
    )

    g.add_edge("rag_retrieval", "obligation_mapper")
    g.add_edge("obligation_mapper", "compliance_synthesizer")
    g.add_edge("compliance_synthesizer", "validator")

    g.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "obligation_mapper": "obligation_mapper",
            "__end__": END,
        },
    )

    g.add_edge("prohibited_path", END)

    checkpointer = _make_checkpointer()
    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


def _invoke_config(thread_id: str | None = None) -> dict[str, Any]:
    """Build the LangGraph RunnableConfig — recursion limit + thread_id correlation."""

    return {
        "recursion_limit": settings.graph_recursion_limit,
        "configurable": {"thread_id": thread_id or f"adhoc-{uuid.uuid4().hex[:8]}"},
    }


def run(user_input: str, *, thread_id: str | None = None) -> RegPilotState:
    """Build the graph and run one full classification + retrieval + report cycle.

    Pass an explicit ``thread_id`` (e.g. the Streamlit session id) to enable
    checkpoint replay; otherwise a fresh ad-hoc id is allocated per call.
    The same id is set on the request-context so every log record produced
    during this call carries it for trivial multi-request log triage.
    """

    graph = build_main_graph()
    # Bind the thread_id to the request-id contextvar so log records produced
    # anywhere in the call (LLM client, retriever, agents, ...) automatically
    # include it. Standard correlation-ID pattern, no manual extra={}.
    config = _invoke_config(thread_id)
    rid = config["configurable"]["thread_id"]
    with request_context(rid):
        return graph.invoke(
            {"user_input": user_input, "validator_loops": 0, "error_count": 0},
            config=config,
        )
