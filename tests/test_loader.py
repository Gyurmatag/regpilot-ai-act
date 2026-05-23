"""Tests for the EU AI Act PDF download + text extraction pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from regpilot.ingestion.loader import _clean_page, download_ai_act, extract_text

# --------------------------------------------------------------------------- #
# Page cleaner — pure-Python regex helper, easy to test exhaustively.
# --------------------------------------------------------------------------- #


def test_clean_page_strips_oj_headers() -> None:
    raw = "Some real content.\nOfficial Journal of the European Union\nMore content."
    out = _clean_page(raw)
    assert "Official Journal" not in out
    assert "Some real content." in out


def test_clean_page_strips_page_numbers() -> None:
    raw = "Body of paragraph.\n23 / 144\nAnother paragraph."
    out = _clean_page(raw)
    assert "23 / 144" not in out
    assert "Body of paragraph." in out


def test_clean_page_collapses_whitespace() -> None:
    raw = "Hello   world\t\t\tHow   are   you?"
    out = _clean_page(raw)
    # Multiple internal spaces/tabs collapse to a single space.
    assert "   " not in out
    assert "Hello world" in out


def test_clean_page_strips_standalone_numbers() -> None:
    raw = "Real text.\n42\nMore real text."
    out = _clean_page(raw)
    assert "42" not in out
    assert "Real text." in out


# --------------------------------------------------------------------------- #
# download_ai_act — mocked httpx
# --------------------------------------------------------------------------- #


def test_download_returns_cached_path_when_pdf_exists(tmp_path: Path) -> None:
    """If a non-empty PDF is already on disk, no HTTP call is made."""

    pdf = tmp_path / "eu_ai_act.pdf"
    pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 200_000)  # >100KB threshold

    with patch("httpx.Client") as mock_client:
        result = download_ai_act(target_dir=tmp_path)

    assert result == pdf
    mock_client.assert_not_called()


def test_download_fetches_pdf_when_missing(tmp_path: Path) -> None:
    """Empty / missing PDF triggers an HTTP download with the right headers."""

    pdf_bytes = b"%PDF-1.4\n" + b"x" * 50_000
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.content = pdf_bytes
    fake_response.headers = {"content-type": "application/pdf"}
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client") as mock_client_cls:
        ctx = MagicMock()
        ctx.get.return_value = fake_response
        mock_client_cls.return_value.__enter__.return_value = ctx

        result = download_ai_act(target_dir=tmp_path)

    assert result.read_bytes() == pdf_bytes

    # Verify the right Accept headers were used (the EUR-Lex CloudFront WAF
    # only serves the PDF when we content-negotiate properly).
    headers = mock_client_cls.call_args.kwargs["headers"]
    assert headers["Accept"] == "application/pdf"
    assert headers["Accept-Language"] == "eng"


def test_download_rejects_non_pdf_response(tmp_path: Path) -> None:
    """If EUR-Lex returns HTML (the WAF challenge page) instead of a PDF, we
    raise a clear RuntimeError instead of silently caching garbage."""

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.content = b"<html>JavaScript required</html>"  # not %PDF
    fake_response.headers = {"content-type": "text/html"}
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client") as mock_client_cls:
        ctx = MagicMock()
        ctx.get.return_value = fake_response
        mock_client_cls.return_value.__enter__.return_value = ctx

        with pytest.raises(RuntimeError, match="Expected a PDF"):
            download_ai_act(target_dir=tmp_path)


# --------------------------------------------------------------------------- #
# extract_text — mocked pdfplumber (so we don't need a real PDF in tests)
# --------------------------------------------------------------------------- #


def test_extract_text_concatenates_pages(tmp_path: Path) -> None:
    """extract_text iterates pages, cleans each, joins with newlines."""

    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfake")  # content doesn't matter — pdfplumber is mocked

    page1 = MagicMock()
    page1.extract_text.return_value = "Article 5\nProhibited practices\n1. blah"
    page2 = MagicMock()
    page2.extract_text.return_value = "Article 6\nClassification"

    fake_pdf = MagicMock()
    fake_pdf.pages = [page1, page2]

    with patch("regpilot.ingestion.loader.pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__.return_value = fake_pdf
        out = extract_text(pdf_path)

    assert "Article 5" in out
    assert "Article 6" in out
    assert "Prohibited practices" in out


def test_extract_text_survives_bad_page(tmp_path: Path) -> None:
    """A single broken page should not abort the whole extraction."""

    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    bad_page = MagicMock()
    bad_page.extract_text.side_effect = RuntimeError("malformed page")
    good_page = MagicMock()
    good_page.extract_text.return_value = "Article 9\nRisk management"

    fake_pdf = MagicMock()
    fake_pdf.pages = [bad_page, good_page]

    with patch("regpilot.ingestion.loader.pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__.return_value = fake_pdf
        out = extract_text(pdf_path)

    assert "Article 9" in out
