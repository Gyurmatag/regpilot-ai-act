"""Modular RAG sub-LangGraph.

Nodes
-----
1. ``query_rewrite``    — emits 1 paraphrase + the original (HyDE-light)
2. ``hybrid_retrieve``  — Chroma dense + BM25, fused with RRF, per rewritten query
3. ``rerank``           — LLM picks the most relevant top-k from the fused list
4. ``compress``         — extractive: keep the most relevant sentences per chunk

The subgraph state mirrors a slice of the main state so the parent graph can
splice it in cleanly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import cast

from langgraph.graph import END, START, StateGraph

from regpilot.config import settings
from regpilot.llm import LLMClient, get_llm
from regpilot.rag.retriever import HybridRetriever
from regpilot.rag.vectorstore import VectorStore
from regpilot.state import RAGState, RetrievedChunk

logger = logging.getLogger(__name__)


REWRITE_SYSTEM = (
    "You are a query rewrite expert for a regulatory-document retrieval system "
    "indexed on the EU AI Act."
)

REWRITE_PROMPT = """Query rewrite task (HyDE-style).

Given a user query about an AI system, generate two alternative search queries
that would better match the formal language used in EU regulation. Return STRICT
JSON: a list of two strings. No commentary.

Original query: {query}
"""

RERANK_SYSTEM = (
    "You rerank candidate paragraphs from the EU AI Act by how directly they "
    "answer the user query."
)

RERANK_PROMPT = """Rerank these {n} candidate passages for the query.

Return STRICT JSON: a list of the {top_k} most relevant indices (0-based, integers only).

Query: {query}

Candidates:
{candidates}
"""


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def _query_rewrite(state: RAGState, llm: LLMClient) -> RAGState:
    query = state["query"]
    try:
        raw = llm.generate(
            REWRITE_PROMPT.format(query=query),
            system=REWRITE_SYSTEM,
            temperature=0.2,
            max_tokens=200,
        )
        rewrites = _safe_json_list(raw)
        rewrites = [r.strip() for r in rewrites if isinstance(r, str) and r.strip()]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("query rewrite failed: %s", exc)
        rewrites = []
    rewrites = list(dict.fromkeys([query, *rewrites]))[:3]
    return {"query": query, "rewritten_queries": rewrites}


def _hybrid_retrieve(state: RAGState, retriever: HybridRetriever) -> RAGState:
    queries = state.get("rewritten_queries") or [state["query"]]
    seen: dict[str, RetrievedChunk] = {}
    for q in queries:
        for hit in retriever.hybrid(q, k_out=settings.top_k_dense):
            if hit["id"] not in seen:
                seen[hit["id"]] = hit
    candidates = list(seen.values())
    logger.info("hybrid_retrieve: %d unique candidates from %d queries", len(candidates), len(queries))
    return {"candidates": candidates}


def _rerank(state: RAGState, llm: LLMClient) -> RAGState:
    candidates = state.get("candidates", [])
    if not candidates:
        return {"reranked": []}
    top_k = settings.top_k_rerank
    if len(candidates) <= top_k:
        return {"reranked": candidates}

    blocks = "\n\n".join(
        f"[{i}] Art. {c.get('article') or '?'} — {c['text'][:400]}"
        for i, c in enumerate(candidates)
    )
    try:
        raw = llm.generate(
            RERANK_PROMPT.format(n=len(candidates), top_k=top_k, query=state["query"], candidates=blocks),
            system=RERANK_SYSTEM,
            temperature=0.0,
            max_tokens=64,
        )
        idxs = [int(i) for i in _safe_json_list(raw) if isinstance(i, (int, float, str))]
        idxs = [i for i in idxs if 0 <= i < len(candidates)][:top_k]
    except Exception as exc:
        logger.warning("LLM rerank failed (%s) — falling back to RRF order.", exc)
        idxs = []

    if not idxs:
        return {"reranked": candidates[:top_k]}
    return {"reranked": [candidates[i] for i in idxs]}


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _compress(state: RAGState) -> RAGState:
    """Trim each chunk to its 3 most-relevant sentences w.r.t. the query."""

    query_tokens = set(re.findall(r"[A-Za-z]{3,}", state["query"].lower()))
    out: list[RetrievedChunk] = []
    for ch in state.get("reranked", []):
        sentences = [s for s in _SENT_RE.split(ch["text"]) if s.strip()]
        if len(sentences) <= 3:
            out.append(ch)
            continue
        scored = sorted(
            enumerate(sentences),
            key=lambda x: sum(1 for t in re.findall(r"[A-Za-z]{3,}", x[1].lower()) if t in query_tokens),
            reverse=True,
        )
        kept = sorted(scored[:3], key=lambda x: x[0])
        compressed = " ".join(s for _, s in kept)
        c = dict(ch)
        c["text"] = compressed
        out.append(cast(RetrievedChunk, c))
    return {"compressed": out}


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def build_rag_subgraph(
    retriever: HybridRetriever | None = None,
    llm: LLMClient | None = None,
):
    """Compile the 4-node RAG subgraph."""

    llm = llm or get_llm()
    retriever = retriever or HybridRetriever(store=VectorStore())

    sg = StateGraph(RAGState)
    sg.add_node("query_rewrite", lambda s: _query_rewrite(s, llm))
    sg.add_node("hybrid_retrieve", lambda s: _hybrid_retrieve(s, retriever))
    sg.add_node("rerank", lambda s: _rerank(s, llm))
    sg.add_node("compress", _compress)

    sg.add_edge(START, "query_rewrite")
    sg.add_edge("query_rewrite", "hybrid_retrieve")
    sg.add_edge("hybrid_retrieve", "rerank")
    sg.add_edge("rerank", "compress")
    sg.add_edge("compress", END)

    return sg.compile()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _safe_json_list(raw: str) -> list:
    raw = raw.strip()
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        val = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []
