"""Streamlit UI for RegPilot.

Two-column layout:

* **left**  — chat history (user describes their AI system, agent replies with
  a tier badge + the compliance roadmap)
* **right** — sticky live agent-trace panel: each node's input/output, the
  obligation table, and the cited evidence chunks

Run locally::

    streamlit run src/regpilot/ui/app.py
"""

from __future__ import annotations

import html
import logging
import time

import streamlit as st

from regpilot.config import settings
from regpilot.graph import _invoke_config, build_main_graph
from regpilot.llm import OllamaClient, get_llm
from regpilot.rag.vectorstore import VectorStore
from regpilot.state import RegPilotState

logging.basicConfig(level=settings.log_level)

st.set_page_config(
    page_title="RegPilot — EU AI Act Compliance Navigator",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/Gyurmatag/regpilot-ai-act",
        "Report a bug": "https://github.com/Gyurmatag/regpilot-ai-act/issues",
        "About": "RegPilot — agentic RAG compliance navigator for the EU AI Act.",
    },
)


# --------------------------------------------------------------------------- #
# Theme / chrome polish
# --------------------------------------------------------------------------- #


_TIER_PALETTE = {
    "prohibited":               ("Prohibited",          "#7f1d1d", "#fee2e2"),
    "high_risk":                ("High risk",           "#9a3412", "#ffedd5"),
    "limited_risk":             ("Limited risk",        "#854d0e", "#fef9c3"),
    "minimal_risk":             ("Minimal risk",        "#166534", "#dcfce7"),
    "general_purpose":          ("GPAI model",          "#1e40af", "#dbeafe"),
    "general_purpose_systemic": ("GPAI · systemic risk","#5b21b6", "#ede9fe"),
    "unknown":                  ("Unknown",             "#374151", "#e5e7eb"),
}


_CSS = """
<style>
/* Hide Streamlit's default chrome — this is a self-hosted app, not a SaaS dashboard. */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }

/* Tighten top padding so the title sits closer to the top. */
section.main > div.block-container { padding-top: 1.4rem; padding-bottom: 6rem; max-width: 1500px; }

/* Sidebar polish. */
section[data-testid="stSidebar"] { background: var(--secondary-background-color); }
section[data-testid="stSidebar"] .stButton button {
    text-align: left;
    font-weight: 400;
    white-space: normal;
    line-height: 1.3;
    border: 1px solid rgba(120,120,120,0.18);
    background: transparent;
    padding: .5rem .7rem;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border-color: rgba(120,120,120,0.45);
    background: rgba(120,120,120,0.05);
}

/* Chat message tweaks. */
[data-testid="stChatMessage"] { padding: .35rem .25rem; }
[data-testid="stChatMessage"] + [data-testid="stChatMessage"] { margin-top: .3rem; }

/* Tier badge. */
.rp-tier-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: .03em;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: .35rem;
}

/* Tighten heading sizes inside Streamlit's bordered container (used for the report). */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    margin: .85rem 0 .3rem 0 !important;
    line-height: 1.3 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3:first-child {
    margin-top: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
    font-size: .92rem !important;
    color: rgba(120,120,120,0.95);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] hr {
    margin: .6rem 0 !important;
    opacity: .35;
}

/* Right column heading styling. */
.rp-trace-h {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .04em;
    color: rgba(120,120,120,0.85);
    margin: .9rem 0 .35rem 0;
}

/* Empty-state hero. */
.rp-empty {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
    color: rgba(120,120,120,0.75);
}
.rp-empty h3 { font-weight: 600; margin-bottom: .4rem; }

/* Disclaimer footer. */
.rp-footer-disclaimer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: var(--background-color);
    border-top: 1px solid rgba(120,120,120,0.15);
    padding: .35rem 1rem;
    font-size: .72rem;
    color: rgba(120,120,120,0.85);
    text-align: center;
    z-index: 999;
}
.rp-footer-disclaimer a { color: inherit; text-decoration: underline; }

/* Cited-evidence card style inside the expander. */
.rp-cite-card {
    border-left: 3px solid rgba(120,120,120,0.35);
    padding: .25rem .75rem;
    margin-bottom: .65rem;
    background: rgba(120,120,120,0.04);
    border-radius: 4px;
}
.rp-cite-head { font-size: .8rem; color: rgba(120,120,120,0.95); margin-bottom: .15rem; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


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
# Sidebar
# --------------------------------------------------------------------------- #


with st.sidebar:
    st.markdown("### RegPilot")
    st.caption("Agentic RAG compliance navigator for the EU AI Act.")

    llm = get_llm()
    backend = "Ollama" if isinstance(llm, OllamaClient) else "Stub (deterministic)"

    with st.container(border=True):
        st.markdown("**System status**")
        st.markdown(f"LLM backend: `{backend}`")
        if isinstance(llm, OllamaClient):
            st.markdown(f"Chat model: `{llm.chat_model}`")
            st.markdown(f"Embed model: `{llm.embed_model}`")

        health = _store_health()
        if health.get("ok"):
            st.markdown(f"Vector store: **{health['count']}** chunks indexed")
        else:
            st.error(
                "Vector store missing. Run `python scripts/ingest.py` first.\n\n"
                f"Error: {health.get('error', 'unknown')}"
            )

    st.markdown("**Try an example**")
    examples = [
        ("CV screening tool", "An automated CV screening AI that ranks applicants for tech roles in Hungary."),
        ("Predictive policing", "An AI used by police to predict, based on profiling, who will commit a crime."),
        ("Support chatbot", "A customer support chatbot on our retail website handling refunds and FAQs."),
        ("Exam grading", "An AI tool that grades student essays for a private high school."),
        ("Spam filter", "A spam filter that classifies inbound corporate email."),
        ("Generative marketing", "A general-purpose generative AI assistant for marketing copy."),
        ("Police facial recognition", "A real-time facial recognition system used in train stations by police."),
    ]
    if "example_pick" not in st.session_state:
        st.session_state.example_pick = None
    for i, (label, ex) in enumerate(examples):
        if st.button(label, key=f"ex{i}", use_container_width=True, help=ex):
            st.session_state.example_pick = ex

    st.divider()
    st.caption(
        "Sources: Regulation (EU) 2024/1689 (EU AI Act) and Annex III. Not legal advice."
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #


st.markdown(
    "<h1 style='margin-bottom:.2rem;'>RegPilot</h1>"
    "<p style='color:rgba(120,120,120,0.85);margin-top:0;margin-bottom:1rem;'>"
    "Classify your AI system &rarr; retrieve the applicable Articles &rarr; get a compliance roadmap."
    "</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


if "history" not in st.session_state:
    st.session_state.history = []  # list[dict(user, state)]
if "thread_id" not in st.session_state:
    # Stable per-Streamlit-session thread id — keys the LangGraph checkpointer
    # so a crashed container resumes the same user's runs, and correlates logs.
    import uuid
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


def _tier_badge_html(tier: str) -> str:
    label, fg, bg = _TIER_PALETTE.get(tier, _TIER_PALETTE["unknown"])
    return (
        f"<span class='rp-tier-badge' style='color:{fg};background:{bg};'>"
        f"{html.escape(label)}</span>"
    )


def _render_assistant(state: RegPilotState) -> None:
    tier = state.get("risk_tier", "unknown")
    st.markdown(
        f"<div style='font-size:.95rem;'><strong>Risk tier:</strong> "
        f"{_tier_badge_html(tier)}</div>",
        unsafe_allow_html=True,
    )
    rationale = state.get("risk_rationale")
    if rationale:
        st.caption(rationale)

    final = state.get("final_report") or "_(no final report — see trace for details.)_"
    with st.container(border=True):
        st.markdown(final)


def _render_trace(state: RegPilotState) -> None:
    trace = state.get("trace", [])
    if not trace:
        st.markdown(
            "<div class='rp-empty'>"
            "<h3>Agent trace</h3>"
            "<div>Submit a description to watch each node fire.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("<div class='rp-trace-h'>Agent trace</div>", unsafe_allow_html=True)
    for ev in trace:
        with st.expander(
            f"{_node_icon(ev['node'])} **{ev['node']}** — {ev['summary']}",
            expanded=False,
        ):
            st.json(ev.get("payload", {}), expanded=False)

    obligations = state.get("obligations") or []
    if obligations:
        st.markdown(
            "<div class='rp-trace-h'>Obligations &amp; deadlines</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            [
                {
                    "Article": o["article"],
                    "Applies from": o["applies_from"],
                    "Obligation": o["obligation"],
                }
                for o in obligations
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Article": st.column_config.TextColumn(width="small"),
                "Applies from": st.column_config.TextColumn(width="small"),
                "Obligation": st.column_config.TextColumn(width="large"),
            },
        )

    retrieved = state.get("retrieved") or []
    if retrieved:
        # The raw score is a Reciprocal Rank Fusion sum (small absolute numbers
        # by design); normalise against the top hit so the user sees a 0–1
        # relevance instead of "0.030", which reads as low when it's actually
        # ~max for RRF with k=60. Some branches (e.g. prohibited short-circuit)
        # pre-load deterministic evidence with score=0 — for those we hide the
        # relevance row instead of showing a misleading 0%.
        max_score = max((c.get("score") or 0.0) for c in retrieved)
        has_scores = max_score > 0
        label_suffix = " · relevance normalised vs. top hit" if has_scores else ""
        with st.expander(
            f"Cited evidence ({len(retrieved)} chunks{label_suffix})",
            expanded=False,
        ):
            for i, c in enumerate(retrieved, start=1):
                raw = c.get("score", 0.0) or 0.0
                if has_scores:
                    rel = raw / max_score
                    score_html = (
                        f" &middot; relevance <code>{rel:.0%}</code> "
                        f"<span style='opacity:.55;font-size:.75rem;'>"
                        f"(RRF {raw:.3f})</span>"
                    )
                else:
                    score_html = (
                        " <span style='opacity:.55;font-size:.75rem;'>"
                        "(pre-loaded evidence)</span>"
                    )
                head = (
                    f"<div class='rp-cite-head'>"
                    f"<strong>#{i} &middot; Art. {html.escape(str(c.get('article') or '?'))} "
                    f"p{html.escape(str(c.get('paragraph') or '?'))}</strong>"
                    f"{score_html}"
                    f"</div>"
                )
                body = html.escape((c.get("text") or "")[:500])
                st.markdown(
                    f"<div class='rp-cite-card'>{head}{body}</div>",
                    unsafe_allow_html=True,
                )


def _node_icon(node: str) -> str:
    return {
        "intake_classifier": "①",
        "risk_triage": "②",
        "rag_retrieval": "③",
        "obligation_mapper": "④",
        "compliance_synthesizer": "⑤",
        "validator": "⑥",
        "prohibited_path": "⊘",
    }.get(node, "·")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


prompt = st.chat_input("Describe your AI system…")
if st.session_state.example_pick and not prompt:
    prompt = st.session_state.example_pick
    st.session_state.example_pick = None


col_chat, col_trace = st.columns([3, 2], gap="large")


# True iff we'll be showing some content in the chat column this run.
_has_content = bool(st.session_state.history) or bool(prompt)


with col_chat:
    if not _has_content:
        st.markdown(
            "<div class='rp-empty'>"
            "<h3>Where do I start?</h3>"
            "<div>Describe your AI system in the box below, or pick an example "
            "from the sidebar. RegPilot will classify it, retrieve the relevant "
            "EU AI Act Articles, and produce a roadmap with citations.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            _render_assistant(turn["state"])

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("Running RegPilot agent…", expanded=False) as status:
                # Per-turn thread_id so each query has its own checkpoint stream
                # (resume on crash mid-turn, but no cross-turn state bleed).
                turn_idx = len(st.session_state.history) + 1
                turn_thread = f"{st.session_state.thread_id}-t{turn_idx}"
                t0 = time.perf_counter()
                state = _graph().invoke(
                    {"user_input": prompt, "validator_loops": 0, "error_count": 0},
                    config=_invoke_config(turn_thread),
                )
                elapsed = time.perf_counter() - t0
                label = (
                    "Done" if elapsed < 0.5
                    else f"Done in {elapsed:.1f}s"
                )
                status.update(label=label, state="complete")
            _render_assistant(state)
        st.session_state.history.append({"user": prompt, "state": state})


with col_trace:
    if st.session_state.history:
        _render_trace(st.session_state.history[-1]["state"])
    elif not _has_content:
        st.markdown(
            "<div class='rp-empty'>"
            "<h3>Agent trace</h3>"
            "<div>Each node's input/output will appear here as the graph runs.</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Sticky disclaimer
# --------------------------------------------------------------------------- #


st.markdown(
    "<div class='rp-footer-disclaimer'>"
    "Not legal advice. Source: "
    "<a href='https://eur-lex.europa.eu/eli/reg/2024/1689/oj' target='_blank' rel='noopener'>"
    "Regulation (EU) 2024/1689</a> &middot; "
    "<a href='https://github.com/Gyurmatag/regpilot-ai-act' target='_blank' rel='noopener'>"
    "source</a>"
    "</div>",
    unsafe_allow_html=True,
)
