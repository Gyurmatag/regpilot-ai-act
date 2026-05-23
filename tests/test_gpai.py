"""GPAI (General-Purpose AI) tier — Articles 51-55 obligations.

Two sub-tiers per Chapter V of the AI Act:
* ``general_purpose`` — baseline GPAI provider duties (Arts. 53, 54).
* ``general_purpose_systemic`` — adds Art. 55 model evaluation, adversarial
  testing, incident reporting, cybersecurity. Triggered by Art. 51 markers
  (10^25 FLOPs, frontier, systemic risk).
"""

from __future__ import annotations

from regpilot.graph import run
from regpilot.tools.deadline_calculator import (
    GPAI_GOVERNANCE_APPLY,
    compute_deadlines,
)


def test_gpai_basic_deadlines_apply_phase_2_and_omit_art_55() -> None:
    """Baseline GPAI provider duties — only Art. 53/54 apply."""

    out = compute_deadlines("general_purpose_ai", "provider")
    assert out
    assert all(o.applies_from == GPAI_GOVERNANCE_APPLY for o in out)
    articles = {o.article for o in out}
    assert {"Art. 53", "Art. 54"} <= articles
    assert "Art. 55" not in articles, "basic GPAI must not cite Art. 55 systemic-risk"


def test_gpai_systemic_obligations_cover_art_55() -> None:
    """Art. 55 obligations (model eval, adversarial testing, incident reporting,
    cybersecurity) — only for GPAI with systemic risk per Art. 51."""

    out = compute_deadlines("general_purpose_ai", "provider", systemic_risk=True)
    art_55 = [o for o in out if o.article == "Art. 55"]
    assert len(art_55) >= 3
    joined = " ".join(o.obligation.lower() for o in art_55)
    for keyword in ("adversarial", "incident", "cybersecurity"):
        assert keyword in joined, f"Art. 55 should mention {keyword!r}"


def test_gpai_basic_e2e_classifies_and_cites_correctly() -> None:
    """A baseline GPAI provider description routes through the graph as
    ``general_purpose`` and cites only Arts. 53/54 in obligations."""

    description = (
        "A 7B-parameter general-purpose generative AI foundation model that we "
        "provide as a marketing-copy assistant for downstream deployers."
    )
    state = run(description)

    assert state.get("risk_tier") == "general_purpose"
    obligations = state.get("obligations") or []
    cited_in_obligations = {o["article"] for o in obligations}
    assert {"Art. 53", "Art. 54"} <= cited_in_obligations
    assert "Art. 55" not in cited_in_obligations


def test_gpai_systemic_e2e_classifies_and_cites_art_55() -> None:
    """A systemic-risk GPAI description (10^25 FLOPs or "frontier" wording)
    classifies as ``general_purpose_systemic`` and cites Art. 55."""

    description = (
        "Frontier LLM with more than 10^25 FLOPs training compute, offered as a "
        "managed API to downstream developers."
    )
    state = run(description)

    assert state.get("risk_tier") == "general_purpose_systemic"
    obligations = state.get("obligations") or []
    cited_in_obligations = {o["article"] for o in obligations}
    assert {"Art. 53", "Art. 54", "Art. 55"} <= cited_in_obligations

    report = state.get("final_report") or ""
    for art in ("Art. 53", "Art. 54", "Art. 55"):
        assert art in report, f"systemic-risk GPAI report must cite {art}"
