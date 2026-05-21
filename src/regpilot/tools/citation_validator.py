"""Citation validator tool.

Scans a draft report for ``Art. N`` / ``Article N`` citations and checks each
one against the indexed AI Act corpus. Issues are returned as a list of human
messages plus a boolean ``ok`` so the validator node can decide whether to loop
back to the obligation mapper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from regpilot.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


_CITE_RE = re.compile(r"\b(?:Art\.|Article)\s+(\d+[a-z]?)(?:\((\d+)\))?", re.I)


@dataclass
class CitationReport:
    ok: bool
    issues: list[str]
    cited_articles: set[str]
    invalid_articles: set[str]


def _index_articles(store: VectorStore) -> set[str]:
    docs = store.all_documents()
    out: set[str] = set()
    for d in docs:
        if d.get("article"):
            out.add(str(d["article"]).strip())
    return out


@lru_cache(maxsize=1)
def _cached_index() -> set[str]:
    return _index_articles(VectorStore())


def validate(draft_report: str, store: VectorStore | None = None) -> CitationReport:
    """Validate every citation in the draft."""

    valid_articles = _index_articles(store) if store is not None else _cached_index()
    cited: set[str] = set()
    invalid: set[str] = set()
    issues: list[str] = []

    for m in _CITE_RE.finditer(draft_report):
        art = m.group(1).strip()
        cited.add(art)
        if art not in valid_articles:
            invalid.add(art)
            issues.append(
                f"Citation 'Art. {art}' does not exist in the indexed EU AI Act."
            )

    if not cited:
        issues.append(
            "No 'Art. N' citations found in the draft — the report must cite the "
            "Articles backing each obligation."
        )

    return CitationReport(
        ok=not issues, issues=issues, cited_articles=cited, invalid_articles=invalid
    )


def reset_cache() -> None:
    """Test helper — clears the cached article index after re-ingestion."""

    _cached_index.cache_clear()
