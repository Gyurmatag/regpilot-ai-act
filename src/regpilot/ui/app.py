"""Streamlit UI for RegPilot.

Two-column layout:
* left  — chat history (user describes their AI system, agent replies with the report)
* right — live agent-trace panel showing each node's input/output as the graph runs.

Run locally::

    streamlit run src/regpilot/ui/app.py
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from regpilot.config import settings
from regpilot.graph import build_main_graph
from regpilot.llm import OllamaClient, get_llm
from regpilot.rag.vectorstore import VectorStore
from regpilot.state import RegPilotState

logging.basicConfig(level=settings.log_level)

st.set_page_config(
    page_title="RegPilot — EU AI Act Compliance Navigator",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_resource
def _graph():
    return build_main_graph()


@st.cache_resource
def _store_health() -> dict:
    try:
        store = VectorStore()
        return {"ok": True, "count": store.count()}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc), "count": 0}


# --------------------------------------------------------------------------- #
# Sidebar — status
# --------------------------------------------------------------------------- #


with st.sidebar:
    st.markdown("### RegPilot")
    st.caption("Agentic RAG compliance navigator for the EU AI Act.")
    st.markdown("---")

    llm = get_llm()
    backend = "Ollama" if isinstance(llm, OllamaClient) else "Stub (deterministic)"
    st.markdown(f"**LLM backend:** `{backend}`")
    if isinstance(llm, OllamaClient):
        st.markdown(f"**Chat model:** `{llm.chat_model}`")
        st.markdown(f"**Embed model:** `{llm.embed_model}`")

    health = _store_health()
    if health.get("ok"):
        st.success(f"Vector store: {health['count']} chunks indexed.")
    else:
        st.error(
            "Vector store missing. Run `python scripts/ingest.py` first.\n\n"
            f"Error: {health.get('error', 'unknown')}"
        )

    st.markdown("---")
    st.markdown("**Try one of these:**")
    examples = [
        "A CV screening AI that ranks applicants for tech roles in Hungary.",
        "A predictive policing system that flags individuals likely to reoffend.",
        "A customer support chatbot on a retail website.",
        "An AI tool that grades student essays for a private high school.",
        "A spam filter for company email.",
        "A generative AI assistant for marketing copy (general-purpose AI).",
        "A real-time facial recognition system used in train stations by police.",
    ]
    if "example_pick" not in st.session_state:
        st.session_state.example_pick = None
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.example_pick = ex

    st.markdown("---")
    st.caption(
        "Not legal advice. Source: Regulation (EU) 2024/1689 — see EUR-Lex for the "
        "authoritative text."
    )


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #


st.title("RegPilot")
st.caption("Classify your AI system → retrieve the applicable Articles → get a compliance roadmap.")

col_chat, col_trace = st.columns([3, 2], gap="large")

if "history" not in st.session_state:
    st.session_state.history = []  # list[dict(user, state)]


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


def _tier_badge(tier: str) -> str:
    icon = {
        "prohibited": "[PROHIBITED]",
        "high_risk": "[HIGH RISK]",
        "limited_risk": "[LIMITED RISK]",
        "minimal_risk": "[MINIMAL RISK]",
        "unknown": "[UNKNOWN]",
    }.get(tier, tier.upper())
    return f"`{icon}`"


def _render_assistant(state: RegPilotState) -> None:
    tier = state.get("risk_tier", "unknown")
    st.markdown(f"### Risk tier: {_tier_badge(tier)}")
    if state.get("risk_rationale"):
        st.caption(state["risk_rationale"])
    final = state.get("final_report") or "_(no final report — see trace for details.)_"
    st.markdown(final)


def _render_trace(state: RegPilotState) -> None:
    trace = state.get("trace", [])
    if not trace:
        st.info("No trace yet — submit a description to see the agent run.")
        return
    st.markdown("**Agent trace**")
    for ev in trace:
        with st.expander(f"• {ev['node']} — {ev['summary']}", expanded=False):
            st.json(ev.get("payload", {}), expanded=False)

    obligations = state.get("obligations", [])
    if obligations:
        st.markdown("**Obligations + deadlines**")
        st.dataframe(
            [
                {
                    "Article": o["article"],
                    "Applies from": o["applies_from"],
                    "Obligation": o["obligation"][:120],
                }
                for o in obligations
            ],
            hide_index=True,
            use_container_width=True,
        )

    retrieved = state.get("retrieved", [])
    if retrieved:
        with st.expander(f"Cited evidence ({len(retrieved)} chunks)", expanded=False):
            for c in retrieved:
                st.markdown(
                    f"**Art. {c.get('article') or '?'} p{c.get('paragraph') or '?'}** "
                    f"(score `{c.get('score', 0):.3f}`)"
                )
                st.write(c.get("text", "")[:600])
                st.markdown("---")


# --------------------------------------------------------------------------- #
# Render history (Streamlit is stateless across runs — replay from session)
# --------------------------------------------------------------------------- #


with col_chat:
    if st.session_state.history:
        st.markdown("### Conversation")
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            _render_assistant(turn["state"])


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


prompt = st.chat_input("Describe your AI system…")
if st.session_state.example_pick and not prompt:
    prompt = st.session_state.example_pick
    st.session_state.example_pick = None

if prompt:
    with col_chat:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("Running RegPilot agent…", expanded=False) as status:
                t0 = time.perf_counter()
                state = _graph().invoke({"user_input": prompt, "validator_loops": 0})
                elapsed = time.perf_counter() - t0
                status.update(label=f"Done in {elapsed:.1f}s", state="complete")
            _render_assistant(state)
    st.session_state.history.append({"user": prompt, "state": state})

with col_trace:
    if st.session_state.history:
        _render_trace(st.session_state.history[-1]["state"])
    else:
        st.info("Submit a description on the left to see the live agent trace here.")
