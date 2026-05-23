"""GPAI (General-Purpose AI) tier — Articles 51-55 obligations."""

from __future__ import annotations

from regpilot.graph import run
from regpilot.tools.deadline_calculator import (
    GPAI_GOVERNANCE_APPLY,
    compute_deadlines,
)


def test_gpai_deadlines_apply_phase_2() -> None:
    """GPAI provider obligations all kick in on 2 Aug 2025 (Art. 113 phase 2)."""

    out = compute_deadlines("general_purpose_ai", "provider")
    assert out
    assert all(o.applies_from == GPAI_GOVERNANCE_APPLY for o in out)
    articles = {o.article for o in out}
    assert {"Art. 53", "Art. 54", "Art. 55"} <= articles


def test_gpai_obligations_cover_systemic_risk_requirements() -> None:
    """Art. 55 obligations (model eval, adversarial testing, incident reporting,
    cybersecurity) — required for GPAI with systemic risk per Art. 51."""

    out = compute_deadlines("general_purpose_ai", "provider")
    art_55 = [o for o in out if o.article == "Art. 55"]
    assert len(art_55) >= 3
    joined = " ".join(o.obligation.lower() for o in art_55)
    for keyword in ("adversarial", "incident", "cybersecurity"):
        assert keyword in joined, f"Art. 55 should mention {keyword!r}"


def test_gpai_end_to_end_classifies_and_cites_correctly() -> None:
    """A GPAI provider description routes through the graph and produces a
    report citing Articles 53/54/55."""

    description = (
        "A 7B-parameter general-purpose generative AI foundation model that we "
        "provide as a marketing-copy assistant for downstream deployers."
    )
    state = run(description)

    obligations = state.get("obligations") or []
    cited_in_obligations = {o["article"] for o in obligations}
    assert {"Art. 53", "Art. 54", "Art. 55"} <= cited_in_obligations

    report = state.get("final_report") or ""
    for art in ("Art. 53", "Art. 54", "Art. 55"):
        assert art in report, f"GPAI report must cite {art}"
