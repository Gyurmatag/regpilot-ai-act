"""Article 113 phased entry-into-force calculator.

Pure Python, no LLM, no I/O — deterministic and unit-testable.

Phase table (Regulation (EU) 2024/1689, Article 113):
* **1 Aug 2024**  — Act enters into force (20 days after OJ publication on 12 Jul).
* **2 Feb 2025**  — Chapter I (general provisions) + Chapter II (Article 5 prohibitions) apply.
* **2 Aug 2025**  — Chapter V (GPAI obligations), Chapter VII (governance), Chapter XII (penalties), Art. 78 (confidentiality).
* **2 Aug 2026**  — General application date (Annex III high-risk + transparency + most provisions).
* **2 Aug 2027**  — Annex I high-risk product safety components.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

SystemType = Literal[
    "annex_iii_high_risk",
    "annex_i_high_risk",
    "general_purpose_ai",
    "limited_risk",
    "minimal_risk",
    "prohibited",
]
UserRole = Literal["provider", "deployer", "importer", "distributor", "unknown"]


@dataclass
class DeadlineInfo:
    """A single concrete deadline the user has to meet."""

    obligation: str
    article: str  # e.g. "Art. 50" — used by the citation validator
    applies_from: date
    note: str = ""


# ----- the canonical Article 113 dates (frozen) ----------------------------- #

ENTRY_INTO_FORCE = date(2024, 8, 1)
PROHIBITIONS_APPLY = date(2025, 2, 2)
GPAI_GOVERNANCE_APPLY = date(2025, 8, 2)
GENERAL_APPLICATION = date(2026, 8, 2)
ANNEX_I_HIGH_RISK_APPLY = date(2027, 8, 2)


def compute_deadlines(
    system_type: SystemType,
    user_role: UserRole = "provider",
) -> list[DeadlineInfo]:
    """Return the chronological list of obligations + deadlines that apply."""

    out: list[DeadlineInfo] = []

    if system_type == "prohibited":
        out.append(
            DeadlineInfo(
                obligation="Cease all placing on market, putting into service, and use.",
                article="Art. 5",
                applies_from=PROHIBITIONS_APPLY,
                note="Prohibited practices have been in force since 2 Feb 2025.",
            )
        )
        return out

    if system_type == "limited_risk":
        out.append(
            DeadlineInfo(
                obligation=(
                    "Disclose to users that they are interacting with an AI system "
                    "(chatbots) and label synthetic / deepfake content."
                ),
                article="Art. 50",
                applies_from=GENERAL_APPLICATION,
                note="Transparency duties for limited-risk systems.",
            )
        )
        return out

    if system_type == "minimal_risk":
        out.append(
            DeadlineInfo(
                obligation=(
                    "No mandatory obligations beyond voluntary codes of conduct."
                ),
                article="Art. 95",
                applies_from=GENERAL_APPLICATION,
                note="Voluntary codes of conduct encouraged.",
            )
        )
        return out

    if system_type == "general_purpose_ai":
        out.append(
            DeadlineInfo(
                obligation="Maintain technical documentation per Annex XI.",
                article="Art. 53",
                applies_from=GPAI_GOVERNANCE_APPLY,
                note="GPAI provider documentation obligation.",
            )
        )
        out.append(
            DeadlineInfo(
                obligation="Publish a sufficiently detailed summary of training content used.",
                article="Art. 53",
                applies_from=GPAI_GOVERNANCE_APPLY,
            )
        )
        out.append(
            DeadlineInfo(
                obligation="Put in place a policy to comply with Union copyright law (Art. 4(3) of the Copyright Directive).",
                article="Art. 53",
                applies_from=GPAI_GOVERNANCE_APPLY,
            )
        )
        out.append(
            DeadlineInfo(
                obligation="Cooperate with the AI Office and national competent authorities.",
                article="Art. 54",
                applies_from=GPAI_GOVERNANCE_APPLY,
            )
        )
        out.append(
            DeadlineInfo(
                obligation=(
                    "Systemic-risk GPAI models (≥10^25 FLOPs training compute): model evaluation, "
                    "adversarial testing, systemic-risk assessment + mitigation."
                ),
                article="Art. 55",
                applies_from=GPAI_GOVERNANCE_APPLY,
                note="Only applies to GPAI models with systemic risk (Art. 51).",
            )
        )
        out.append(
            DeadlineInfo(
                obligation="Systemic-risk GPAI: report serious incidents to the AI Office and national authorities without undue delay.",
                article="Art. 55",
                applies_from=GPAI_GOVERNANCE_APPLY,
            )
        )
        out.append(
            DeadlineInfo(
                obligation="Systemic-risk GPAI: ensure adequate cybersecurity protection of the model + physical infrastructure.",
                article="Art. 55",
                applies_from=GPAI_GOVERNANCE_APPLY,
            )
        )
        return out

    # ----- Annex III high-risk systems ------------------------------------- #
    if system_type == "annex_iii_high_risk":
        applies = GENERAL_APPLICATION
        if user_role in ("provider", "unknown"):
            out.append(
                DeadlineInfo(
                    obligation="Establish a risk-management system across the lifecycle.",
                    article="Art. 9",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Data governance: representative, error-free, complete training/validation/test sets.",
                    article="Art. 10",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Compile technical documentation per Annex IV.",
                    article="Art. 11",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Maintain automatically generated event logs for the system's lifetime.",
                    article="Art. 12",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Ensure transparency to deployers and provide instructions for use.",
                    article="Art. 13",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Implement human-oversight measures appropriate to the system.",
                    article="Art. 14",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Achieve appropriate accuracy, robustness and cybersecurity.",
                    article="Art. 15",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Operate a quality management system as the provider.",
                    article="Art. 17",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Keep technical documentation and logs available for 10 years.",
                    article="Art. 18",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Conformity assessment + CE marking + EU declaration of conformity.",
                    article="Art. 43",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Register the system in the EU database before market placement.",
                    article="Art. 49",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Post-market monitoring + reporting of serious incidents.",
                    article="Art. 72",
                    applies_from=applies,
                )
            )
        if user_role == "deployer":
            out.append(
                DeadlineInfo(
                    obligation="Operate the system per the provider's instructions, assign human oversight, monitor.",
                    article="Art. 26",
                    applies_from=applies,
                )
            )
            out.append(
                DeadlineInfo(
                    obligation="Conduct a fundamental-rights impact assessment (where applicable).",
                    article="Art. 27",
                    applies_from=applies,
                )
            )
        return out

    if system_type == "annex_i_high_risk":
        out.append(
            DeadlineInfo(
                obligation="Comply with sectoral product safety law + AI Act obligations.",
                article="Art. 6(1)",
                applies_from=ANNEX_I_HIGH_RISK_APPLY,
                note="Annex I product-safety high-risk systems have an extra year.",
            )
        )
        return out

    return out  # pragma: no cover - exhaustive above


def summarize_phase(d: date) -> str:
    """Human label for a deadline date."""

    if d <= ENTRY_INTO_FORCE:
        return "in force"
    if d <= PROHIBITIONS_APPLY:
        return "Phase 1 (prohibitions)"
    if d <= GPAI_GOVERNANCE_APPLY:
        return "Phase 2 (GPAI + governance)"
    if d <= GENERAL_APPLICATION:
        return "Phase 3 (general application)"
    return "Phase 4 (Annex I product safety)"
