"""Tests for the deterministic synthesizer scaffold helpers.

These guard the contract that ``regpilot.agents.synthesizer`` depends on:
both the LLM-narrative path and the template fallback render through
exactly these helpers, so any drift in their output shows up here first.
"""

from __future__ import annotations

from regpilot.agents._synth_scaffold import (
    LIFECYCLE,
    NEXT_STEPS,
    ROLE_NARRATIVE,
    TIER_LABEL,
    cited_articles_str,
    evidence_block,
    format_obligation_bullets,
    fria_flag,
    lifecycle_markdown,
    obligation_articles_set,
    render_steps,
    standards_alignment_section,
)

# --------------------------------------------------------------------------- #
# Pure-data integrity
# --------------------------------------------------------------------------- #


def test_next_steps_covers_every_tier_we_classify_into() -> None:
    expected = {
        "high_risk", "limited_risk", "minimal_risk", "prohibited",
        "general_purpose", "general_purpose_systemic", "unknown",
    }
    assert expected <= set(NEXT_STEPS.keys())


def test_tier_label_covers_every_tier_we_classify_into() -> None:
    expected = {
        "high_risk", "limited_risk", "minimal_risk", "prohibited",
        "general_purpose", "general_purpose_systemic", "unknown",
    }
    assert expected <= set(TIER_LABEL.keys())


def test_role_narrative_covers_every_role_the_deadline_calculator_emits() -> None:
    expected = {"provider", "deployer", "importer", "distributor", "unknown"}
    assert expected == set(ROLE_NARRATIVE.keys())


def test_lifecycle_covers_all_four_phases() -> None:
    assert len(LIFECYCLE) == 4
    phases = {phase for phase, _ in LIFECYCLE}
    assert "Design & development" in phases
    assert "Market entry" in phases


# --------------------------------------------------------------------------- #
# obligation_articles_set / cited_articles_str
# --------------------------------------------------------------------------- #


def test_obligation_articles_set_strips_prefix_and_dedupes() -> None:
    obs = [
        {"article": "Art. 9", "applies_from": "2026-08-02", "obligation": "..."},
        {"article": "Art. 9", "applies_from": "2026-08-02", "obligation": "..."},  # dupe
        {"article": "Art. 10", "applies_from": "2026-08-02", "obligation": "..."},
    ]
    assert obligation_articles_set(obs) == {"9", "10"}


def test_cited_articles_str_sorts_and_falls_back_to_art_6() -> None:
    obs = [
        {"article": "Art. 10", "applies_from": "...", "obligation": "..."},
        {"article": "Art. 9", "applies_from": "...", "obligation": "..."},
    ]
    assert cited_articles_str(obs) == "Art. 10, Art. 9"
    # Sorting is lexical, not numeric — that's by design (Art. 10 < Art. 9 for strings).
    # Empty obligations → minimal-risk default citation (Art. 6).
    assert cited_articles_str([]) == "Art. 6"


# --------------------------------------------------------------------------- #
# format_obligation_bullets
# --------------------------------------------------------------------------- #


def test_format_obligation_bullets_renders_date_article_obligation() -> None:
    obs = [
        {"article": "Art. 9", "applies_from": "2026-08-02", "obligation": "Risk mgmt"},
    ]
    out = format_obligation_bullets(obs)
    assert "**2026-08-02 — Art. 9**" in out
    assert "Risk mgmt" in out


def test_format_obligation_bullets_falls_back_when_empty() -> None:
    assert "No mandatory obligations" in format_obligation_bullets([])


# --------------------------------------------------------------------------- #
# evidence_block
# --------------------------------------------------------------------------- #


def test_evidence_block_filters_to_obligation_articles_and_sorts_by_score() -> None:
    retrieved = [
        {"article": "9",  "paragraph": "1", "text": "RMS …", "score": 0.9},
        {"article": "99", "paragraph": "2", "text": "Off-topic …", "score": 0.99},  # not in obligations
        {"article": "10", "paragraph": "3", "text": "Data governance …", "score": 0.7},
    ]
    obs = [
        {"article": "Art. 9",  "applies_from": "...", "obligation": "..."},
        {"article": "Art. 10", "applies_from": "...", "obligation": "..."},
    ]
    out = evidence_block(retrieved, obs)
    assert "Art. 9" in out
    assert "Art. 10" in out
    assert "Art. 99" not in out  # filtered out
    assert out.index("Art. 9") < out.index("Art. 10")  # sorted by score desc


def test_evidence_block_falls_back_when_no_matches() -> None:
    out = evidence_block(
        [{"article": "99", "paragraph": "1", "text": "x", "score": 1.0}],
        [{"article": "Art. 9", "applies_from": "...", "obligation": "..."}],
    )
    assert "no retrieved evidence" in out


def test_evidence_block_caps_excerpt_count_and_chars() -> None:
    retrieved = [
        {"article": "9", "paragraph": str(i), "text": "x" * 500, "score": float(i)}
        for i in range(5)
    ]
    obs = [{"article": "Art. 9", "applies_from": "...", "obligation": "..."}]
    out = evidence_block(retrieved, obs, max_excerpts=2, max_chars=50)
    # Only 2 quotes; each truncated.
    assert out.count(">") == 2
    # Each excerpt is ≤ ~60 chars (50 + ellipsis + prefix).
    longest = max(len(line) for line in out.splitlines() if line.startswith(">"))
    assert longest < 200


# --------------------------------------------------------------------------- #
# fria_flag
# --------------------------------------------------------------------------- #


def test_fria_flag_triggers_for_high_risk_deployer() -> None:
    assert "FRIA" in fria_flag("high_risk", "deployer")


def test_fria_flag_triggers_for_high_risk_unknown_role() -> None:
    assert "FRIA" in fria_flag("high_risk", "unknown")


def test_fria_flag_is_silent_for_other_tiers() -> None:
    assert fria_flag("minimal_risk", "deployer") == ""
    assert fria_flag("limited_risk", "deployer") == ""
    assert fria_flag("high_risk", "provider") == ""


# --------------------------------------------------------------------------- #
# render_steps / lifecycle / standards
# --------------------------------------------------------------------------- #


def test_render_steps_numbers_the_list() -> None:
    out = render_steps(["alpha", "beta"])
    assert out == "1. alpha\n2. beta"


def test_lifecycle_markdown_emits_bullet_per_phase() -> None:
    out = lifecycle_markdown(LIFECYCLE)
    assert out.count("\n- ") == len(LIFECYCLE) - 1  # 3 newlines + 4 bullets


def test_standards_alignment_section_mentions_iso_and_nist() -> None:
    out = standards_alignment_section()
    assert "ISO/IEC 42001:2023" in out
    assert "NIST AI Risk Management Framework" in out
    assert "CEN/CENELEC JTC 21" in out
