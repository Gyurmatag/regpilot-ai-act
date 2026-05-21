"""Tests for the article-aware chunker."""

from __future__ import annotations

import textwrap

from regpilot.ingestion.chunker import Chunk, chunk_text


def test_article_aware_split() -> None:
    """One Article with two numbered paragraphs → two chunks, both labelled."""

    text = textwrap.dedent(
        """\
        Article 9
        Risk management system
        1. A risk management system shall be established for high-risk AI systems.
        2. The risk management system shall consist of a continuous process across the lifecycle.
        """
    )
    chunks = chunk_text(text)
    assert all(c.article == "9" for c in chunks)
    paragraphs = sorted({c.paragraph for c in chunks if c.paragraph})
    assert paragraphs == ["1", "2"]
    assert all(c.title and "Risk management" in c.title for c in chunks)


def test_multiple_articles_split_independently() -> None:
    text = textwrap.dedent(
        """\
        Article 5
        Prohibited practices
        1. Social scoring shall be prohibited.

        Article 6
        Classification rules
        1. High-risk systems are defined as follows.
        """
    )
    chunks = chunk_text(text)
    arts = sorted({c.article for c in chunks if c.article})
    assert arts == ["5", "6"]


def test_duplicate_ids_get_disambiguated() -> None:
    """Nested numbered sub-lists inside one paragraph used to collide on id."""

    text = textwrap.dedent(
        """\
        Article 57
        Regulatory sandboxes
        3. The sandboxes shall:
        1. record submissions;
        2. report outcomes;
        3. publish summaries.
        """
    )
    chunks = chunk_text(text)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"


def test_fallback_when_no_articles_found() -> None:
    text = "Some preamble text that contains no Article header at all. " * 50
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.article is None for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_long_paragraph_is_size_split() -> None:
    body = "This is one long sentence. " * 200
    text = f"Article 11\nTechnical documentation\n1. {body}"
    chunks = chunk_text(text, max_chars=400)
    assert len(chunks) >= 2
    assert all(c.article == "11" for c in chunks)
    assert all(len(c.text) <= 400 for c in chunks)


def test_chunk_text_metadata_round_trip() -> None:
    text = "Article 50\nTransparency obligations\n1. Providers shall inform users."
    [chunk] = chunk_text(text)
    assert chunk.article == "50"
    assert chunk.paragraph == "1"
    assert chunk.title == "Transparency obligations"
    assert "EU AI Act" in chunk.source
