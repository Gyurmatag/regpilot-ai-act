"""Shared LangGraph state schema for the main workflow and the RAG subgraph."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

RiskTier = Literal[
    "prohibited",
    "high_risk",
    "limited_risk",
    "minimal_risk",
    "unknown",
]


class RetrievedChunk(TypedDict):
    """A single chunk returned by the RAG subgraph."""

    id: str
    text: str
    article: str | None
    paragraph: str | None
    title: str | None
    source: str
    score: float


class TraceEvent(TypedDict):
    """One step of the agentic workflow, surfaced in the UI live-trace panel."""

    node: str
    summary: str
    payload: dict[str, Any]


class StructuredIntake(TypedDict, total=False):
    """Output of the intake_classifier node."""

    system_purpose: str
    deployment_context: str
    data_modalities: list[str]
    user_role: Literal["provider", "deployer", "importer", "distributor", "unknown"]
    domain: str
    notes: str


class RegPilotState(TypedDict, total=False):
    """The single state object that flows through the main LangGraph workflow."""

    user_input: str
    messages: Annotated[list, add_messages]
    structured: StructuredIntake
    risk_tier: RiskTier
    risk_rationale: str
    annex_iii_matches: list[str]
    rag_query: str
    rag_queries: list[str]         # NEW: multi-query expansion from triage
    priority_articles: list[str]   # NEW: tier-derived obligation Articles to boost
    retrieved: list[RetrievedChunk]
    obligations: list[dict[str, Any]]
    deadlines: dict[str, Any]
    draft_report: str
    validation_issues: list[str]
    validator_loops: int
    final_report: str
    trace: list[TraceEvent]


class RAGState(TypedDict, total=False):
    """State for the RAG subgraph (a slice of the main state)."""

    query: str
    rewritten_queries: list[str]
    priority_articles: list[str]
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    compressed: list[RetrievedChunk]
