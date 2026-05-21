"""Main LangGraph workflow.

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
from typing import Any

from langgraph.graph import END, START, StateGraph

from regpilot.agents.intake import intake_classifier
from regpilot.agents.obligation_mapper import obligation_mapper
from regpilot.agents.synthesizer import compliance_synthesizer
from regpilot.agents.triage import risk_triage, route_by_tier
from regpilot.agents.validator import route_after_validator, validator
from regpilot.config import settings
from regpilot.rag.subgraph import build_rag_subgraph
from regpilot.state import RegPilotState, TraceEvent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Wrapper nodes that splice the RAG subgraph into the main flow
# --------------------------------------------------------------------------- #


def _make_rag_node(rag_subgraph):
    def rag_retrieval(state: RegPilotState) -> RegPilotState:
        query = state.get("rag_query") or state.get("user_input", "")
        t0 = time.perf_counter()
        result = rag_subgraph.invoke({"query": query})
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
                    },
                ),
            ],
        }
    return rag_retrieval


def prohibited_path(state: RegPilotState) -> RegPilotState:
    """Short-circuit for systems that are outright banned by Article 5."""

    structured = state.get("structured", {})
    matches = state.get("annex_iii_matches", [])
    report = (
        f"## Risk classification\n"
        f"The described system is **PROHIBITED** under Article 5 of the EU AI Act.\n\n"
        f"### Why\n{state.get('risk_rationale', 'Triage flagged the system as prohibited.')}\n\n"
        f"### Mandatory action\nDo not place this system on the EU market or put it into service.\n"
        f"Article 5 prohibitions have been in force since 2 February 2025 (Art. 113).\n\n"
        f"### Cited\nArt. 5, Art. 113.\n"
    )
    return {
        "final_report": report,
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="prohibited_path",
                summary="emitted short-circuit prohibition notice",
                payload={"structured": dict(structured), "matches": matches},
            ),
        ],
    }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_main_graph(rag_subgraph: Any | None = None):
    """Compile the full RegPilot workflow."""

    if rag_subgraph is None:
        rag_subgraph = build_rag_subgraph()

    g = StateGraph(RegPilotState)

    g.add_node("intake_classifier", intake_classifier)
    g.add_node("risk_triage", risk_triage)
    g.add_node("rag_retrieval", _make_rag_node(rag_subgraph))
    g.add_node("obligation_mapper", obligation_mapper)
    g.add_node("compliance_synthesizer", compliance_synthesizer)
    g.add_node("validator", validator)
    g.add_node("prohibited_path", prohibited_path)

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

    return g.compile()


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


def run(user_input: str) -> RegPilotState:
    """Build the graph and run one full classification + retrieval + report cycle."""

    graph = build_main_graph()
    return graph.invoke({"user_input": user_input, "validator_loops": 0})
