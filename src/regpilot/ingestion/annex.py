"""Structured Annex III high-risk use cases and Article 5 prohibited practices.

These are hardcoded from the official text so the rule-based portion of
``risk_classifier_tool`` works deterministically without re-parsing the PDF on
every call. They are also exposed as Chunks so the vector index can serve them
back as ranked evidence when the user query lands on a matching domain.

Source: Regulation (EU) 2024/1689 of the European Parliament and of the
Council of 13 June 2024 (Artificial Intelligence Act), Articles 5 and 6 and
Annex III. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
"""

from __future__ import annotations

from dataclasses import dataclass

from regpilot.ingestion.chunker import Chunk


@dataclass(frozen=True)
class AnnexEntry:
    """One row from Annex III (high-risk areas)."""

    area: str
    description: str
    keywords: tuple[str, ...]


ANNEX_III: tuple[AnnexEntry, ...] = (
    AnnexEntry(
        area="Biometrics",
        description=(
            "Remote biometric identification systems; biometric categorisation of "
            "natural persons according to sensitive attributes; emotion recognition."
        ),
        keywords=(
            "biometric identification",
            "biometric categorisation",
            "biometric categorization",
            "emotion recognition",
            "emotion detection",
            "emotion analysis",
            "mood detection",
            "face recognition",
            "facial recognition",
            "face detection",
            "facial detection",
            "iris recognition",
            "fingerprint recognition",
            "voice biometric",
            "gait recognition",
        ),
    ),
    AnnexEntry(
        area="Critical infrastructure",
        description=(
            "Safety components in the management and operation of critical digital "
            "infrastructure, road traffic, and the supply of water, gas, heating and "
            "electricity."
        ),
        keywords=(
            "critical infrastructure",
            "water supply",
            "gas supply",
            "electricity grid",
            "electricity network",
            "power grid",
            "power-grid",
            "load balancing",
            "load-balancing",
            "road traffic",
            "safety component",
        ),
    ),
    AnnexEntry(
        area="Education and vocational training",
        description=(
            "Determining access, admission or assignment to educational institutions; "
            "evaluating learning outcomes; assessing the appropriate level of education; "
            "monitoring prohibited behaviour during tests."
        ),
        keywords=(
            "admission",
            "exam proctor",
            "exam proctoring",
            "grading",
            "education",
            "vocational training",
            "student assessment",
        ),
    ),
    AnnexEntry(
        area="Employment, worker management, access to self-employment",
        description=(
            "Recruitment and selection (CV screening, ranking candidates); decisions "
            "affecting work-related relationships, promotion and termination; task "
            "allocation; monitoring and evaluating performance."
        ),
        keywords=(
            "recruitment",
            "recruiting",
            "hiring",
            "cv screening",
            "resume screening",
            "promotion",
            "task allocation",
            "worker monitoring",
            "performance evaluation",
        ),
    ),
    AnnexEntry(
        area="Access to and enjoyment of essential private and public services and benefits",
        description=(
            "Credit-worthiness evaluation; eligibility for public assistance benefits "
            "and services; pricing of life and health insurance; emergency dispatch "
            "and triage of first-response services."
        ),
        keywords=(
            "credit scoring",
            "credit-worthiness",
            "creditworthiness",
            "loan eligibility",
            "social benefit",
            "welfare",
            "insurance pricing",
            "emergency dispatch",
            "emergency call",
            "ambulance dispatch",
            "112 emergency",
            "first response",
            "first-response",
            "triage",
            "triages",
        ),
    ),
    AnnexEntry(
        area="Law enforcement",
        description=(
            "Risk assessment of individuals offending or re-offending; polygraph and "
            "similar tools; evaluation of evidence reliability; profiling for crime "
            "detection."
        ),
        keywords=(
            "law enforcement",
            "predictive policing",
            "recidivism",
            "polygraph",
            "criminal profiling",
            "evidence reliability",
        ),
    ),
    AnnexEntry(
        area="Migration, asylum, border control",
        description=(
            "Polygraphs and similar tools at borders; assessing security or health "
            "risks of natural persons; examining asylum and visa applications."
        ),
        keywords=(
            "border control",
            "asylum",
            "visa application",
            "migration risk",
        ),
    ),
    AnnexEntry(
        area="Administration of justice and democratic processes",
        description=(
            "Assisting judicial authorities in researching and interpreting facts or "
            "applying the law; influencing elections or referenda."
        ),
        keywords=(
            "judicial",
            "court",
            "election",
            "referendum",
            "voting",
        ),
    ),
)


@dataclass(frozen=True)
class ProhibitedPractice:
    """One row from Article 5 (prohibited AI practices)."""

    code: str
    practice: str
    keywords: tuple[str, ...]


ARTICLE_5_PROHIBITED: tuple[ProhibitedPractice, ...] = (
    ProhibitedPractice(
        code="5(1)(a)",
        practice="Subliminal, manipulative or deceptive techniques distorting behaviour and causing harm.",
        keywords=("subliminal", "manipulative ai", "deceptive ai"),
    ),
    ProhibitedPractice(
        code="5(1)(b)",
        practice="Exploiting vulnerabilities due to age, disability or socio-economic situation.",
        keywords=("exploit vulnerability", "exploit children", "exploit elderly"),
    ),
    ProhibitedPractice(
        code="5(1)(c)",
        practice="Social scoring of natural persons by public or private actors.",
        keywords=("social scoring", "social credit"),
    ),
    ProhibitedPractice(
        code="5(1)(d)",
        practice="Predictive policing based solely on profiling of natural persons.",
        keywords=(
            "predictive policing",
            "predict crime",
            "pre-crime",
            "predict who will commit a crime",
            "profiling to predict crime",
            "predict criminal behaviour",
            "predict criminal behavior",
        ),
    ),
    ProhibitedPractice(
        code="5(1)(e)",
        practice="Untargeted scraping of facial images to build face-recognition databases.",
        keywords=("scrape faces", "untargeted scraping", "build facial database"),
    ),
    ProhibitedPractice(
        code="5(1)(f)",
        practice="Emotion recognition in the workplace and in educational institutions (with limited exceptions).",
        keywords=(
            "emotion recognition workplace",
            "emotion recognition at work",
            "emotion recognition school",
            "emotion recognition office",
            "emotion recognition employee",
            "monitor employee mood",
            "monitor employee emotions",
            "emotion recognition camera",
        ),
    ),
    ProhibitedPractice(
        code="5(1)(g)",
        practice="Biometric categorisation inferring sensitive attributes (race, political opinions, sexual orientation, etc.).",
        keywords=(
            "biometric categorisation race",
            "biometric categorization race",
            "infer political",
            "infer sexual orientation",
        ),
    ),
    ProhibitedPractice(
        code="5(1)(h)",
        practice="Real-time remote biometric identification in publicly accessible spaces for law-enforcement.",
        keywords=("real-time remote biometric", "live facial recognition police"),
    ),
)


def annex_iii_chunks() -> list[Chunk]:
    """Render Annex III rows as RAG chunks for vector indexing."""

    return [
        Chunk(
            id=f"annex3-{i}",
            text=f"Annex III ({e.area}). {e.description}",
            article="ANNEX III",
            paragraph=str(i + 1),
            title=e.area,
            meta={"kind": "annex_iii"},
        )
        for i, e in enumerate(ANNEX_III)
    ]


def article_5_chunks() -> list[Chunk]:
    """Render Article 5 prohibitions as RAG chunks."""

    return [
        Chunk(
            id=f"art-5-{p.code}",
            text=f"Article 5({p.code[-3]}) — Prohibited practice: {p.practice}",
            article="5",
            paragraph=p.code,
            title="Prohibited AI practices",
            meta={"kind": "article_5"},
        )
        for p in ARTICLE_5_PROHIBITED
    ]
