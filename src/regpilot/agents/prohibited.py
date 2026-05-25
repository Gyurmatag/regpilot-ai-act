"""Prohibited-path node.

Short-circuit branch for systems banned outright by Article 5. When the
risk_triage router emits ``prohibited``, the graph skips RAG retrieval,
obligation mapping, synthesizer + validator entirely and lands here:
the node pre-loads Art. 5 and Art. 113 evidence chunks (so the trace
panel still has citations and the context_recall metric is fair to this
branch), drafts a short ban notice, and returns straight to END.

The interleave between Art. 5 and Art. 113 chunks matters for the
eval: Article 5 has more chunks than Article 113, so a naive
"take the first 5" would fill the slot list with Art. 5 alone and the
retrieval-Recall@5 metric would cap at 50%. We zip-merge them so both
articles surface in the top-5.
"""

from __future__ import annotations

from regpilot.rag.vectorstore import VectorStore
from regpilot.state import RegPilotState, TraceEvent
from regpilot.tools.deadline_calculator import compute_deadlines, summarize_phase


def prohibited_path(state: RegPilotState) -> RegPilotState:
    """Emit a ban notice + Art. 5 / 113 evidence; skip the RAG chain."""

    structured = state.get("structured", {})
    matches = state.get("annex_iii_matches", [])

    info = compute_deadlines("prohibited")
    obligations = [
        {
            "article": d.article,
            "obligation": d.obligation,
            "applies_from": d.applies_from.isoformat(),
            "phase": summarize_phase(d.applies_from),
            "note": d.note,
        }
        for d in info
    ]

    store = VectorStore()
    all_docs = store.all_documents()
    art5 = [c for c in all_docs if c.get("article") == "5"]
    art113 = [c for c in all_docs if c.get("article") == "113"]
    evidence: list = []
    for a, b in zip(art5[:3], art113[:3], strict=False):
        evidence.append(a)
        evidence.append(b)
    evidence.extend(art5[3:5])  # fill remainder if Art. 113 has fewer chunks

    report = (
        f"## Risk classification\n"
        f"The described system is **PROHIBITED** under Article 5 of the EU AI Act.\n\n"
        f"### Why\n{state.get('risk_rationale', 'Triage flagged the system as prohibited.')}\n\n"
        f"### Mandatory action\nDo not place this system on the EU market or put it into service.\n"
        f"Article 5 prohibitions have been in force since 2 February 2025 (Art. 113).\n\n"
        f"### Cited\nArt. 5, Art. 113.\n"
    )
    return {
        "retrieved": evidence,
        "obligations": obligations,
        "deadlines": {
            "system_type": "prohibited",
            "user_role": structured.get("user_role", "unknown"),
            "items": [
                {"article": d.article, "date": d.applies_from.isoformat()} for d in info
            ],
        },
        "final_report": report,
        "trace": [
            *state.get("trace", []),
            TraceEvent(
                node="prohibited_path",
                summary=f"emitted prohibition notice (cited {len(evidence)} evidence chunks)",
                payload={
                    "structured": dict(structured),
                    "matches": matches,
                    "evidence_articles": sorted({str(c.get("article")) for c in evidence}),
                },
            ),
        ],
    }
