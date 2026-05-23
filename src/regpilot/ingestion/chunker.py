"""Article-aware chunker for the EU AI Act.

Strategy
--------
1. Slice the full text into per-Article spans by finding ``Article N`` headers.
2. Within each Article, split on numbered paragraph headers (``1.``, ``2.`` …).
3. If the result is still too long, fall back to a recursive character split so
   we never emit a chunk larger than ``max_chars`` characters.

Each chunk carries metadata (``article``, ``paragraph``, ``title``) used for
filtering, citation rendering, and the gold-set evaluation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    text: str
    article: str | None = None
    paragraph: str | None = None
    title: str | None = None
    source: str = "EU AI Act (Regulation (EU) 2024/1689)"
    meta: dict = field(default_factory=dict)


# Match Article HEADERS only (not inline cross-references like "Article 74(8)").
# Required shape: line containing exactly "Article N" (optionally with a letter
# suffix), followed by a non-empty title line (multi-word, starts with a capital).
# The negative lookahead `(?!\()` excludes paragraph references like
# "Article 74(8)" that appear mid-sentence after a line break.
_ART_RE = re.compile(
    r"(?m)^\s*Article\s+(\d+[a-z]?)(?!\()\s*\n+\s*([A-Z][^\n]{2,200})$"
)
# Match "1." or "(1)" paragraph starts.
_PARA_RE = re.compile(r"(?m)^\s*(?:\((\d{1,2})\)|(\d{1,2})\.)\s+")


def chunk_text(text: str, *, max_chars: int = 1800) -> list[Chunk]:
    """Split ``text`` into Article/paragraph chunks.

    Paragraph numbers are re-mapped to running sequence numbers within each
    article so nested numbered lists (e.g. Art. 57 has a sub-list ``1.``, ``2.``
    nested inside paragraph 3) don't collide.
    """

    matches = list(_ART_RE.finditer(text))
    if not matches:
        logger.warning("No 'Article N' headers found — falling back to size-based chunking.")
        return list(_fallback_chunks(text, max_chars=max_chars))

    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        art_num = m.group(1)
        title = (m.group(2) or "").strip()[:200] or None
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.extend(_split_article(art_num, title, body, max_chars=max_chars))

    # Final de-dup pass: append a counter to any colliding ids so upsert is safe.
    seen: dict[str, int] = {}
    for c in chunks:
        if c.id in seen:
            seen[c.id] += 1
            c.id = f"{c.id}-d{seen[c.id]}"
        else:
            seen[c.id] = 0
    return chunks


def _split_article(
    art_num: str, title: str | None, body: str, *, max_chars: int
) -> Iterable[Chunk]:
    para_matches = list(_PARA_RE.finditer(body))
    if not para_matches:
        for j, piece in enumerate(_split_by_size(body, max_chars)):
            yield Chunk(
                id=f"art-{art_num}-{j}",
                text=piece,
                article=art_num,
                paragraph=None,
                title=title,
            )
        return

    for k, pm in enumerate(para_matches):
        para_num = pm.group(1) or pm.group(2)
        start = pm.start()
        end = para_matches[k + 1].start() if k + 1 < len(para_matches) else len(body)
        para_text = body[start:end].strip()
        if not para_text:
            continue
        if len(para_text) <= max_chars:
            yield Chunk(
                id=f"art-{art_num}-p{para_num}",
                text=para_text,
                article=art_num,
                paragraph=para_num,
                title=title,
            )
        else:
            for j, piece in enumerate(_split_by_size(para_text, max_chars)):
                yield Chunk(
                    id=f"art-{art_num}-p{para_num}-{j}",
                    text=piece,
                    article=art_num,
                    paragraph=para_num,
                    title=title,
                )


def _fallback_chunks(text: str, *, max_chars: int) -> Iterable[Chunk]:
    for j, piece in enumerate(_split_by_size(text, max_chars)):
        yield Chunk(id=f"raw-{j}", text=piece)


def _split_by_size(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    # Sentence-aware split, then pack greedily.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 > max_chars and buf:
            out.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf.strip())
    return out
