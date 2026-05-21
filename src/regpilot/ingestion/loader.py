"""Download + read the consolidated EU AI Act text from EUR-Lex.

Idempotent: if the PDF already exists on disk, we skip the download.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx
import pdfplumber
from tenacity import retry, stop_after_attempt, wait_exponential

from regpilot.config import settings

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20), reraise=True)
def download_ai_act(target_dir: Path | None = None) -> Path:
    target_dir = target_dir or settings.data_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / "eu_ai_act.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 100_000:
        logger.info("EU AI Act PDF already cached at %s", pdf_path)
        return pdf_path

    logger.info("Downloading EU AI Act PDF from %s", settings.ai_act_pdf_url)
    headers = {
        "User-Agent": "regpilot/0.1 (+https://github.com/Gyurmatag/regpilot-ai-act)",
        "Accept": "application/pdf",
        "Accept-Language": "eng",
    }
    with httpx.Client(follow_redirects=True, timeout=60.0, headers=headers) as c:
        r = c.get(settings.ai_act_pdf_url)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise RuntimeError(
                f"Expected a PDF from {settings.ai_act_pdf_url} but got "
                f"Content-Type={r.headers.get('content-type')!r} (len={len(r.content)})"
            )
        pdf_path.write_bytes(r.content)
    logger.info("Saved %d bytes to %s", len(r.content), pdf_path)
    return pdf_path


def extract_text(pdf_path: Path) -> str:
    """Extract raw text from the PDF, stripping page headers/footers and noise.

    ``pdfplumber`` rather than ``pypdf`` because the OJ PDF uses character-spacing
    that pypdf turns into garbage like "Ar tif icial" — pdfplumber respects the
    glyph widths and gives clean text out of the box.
    """

    parts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                txt = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - pdfplumber occasionally chokes
                logger.warning("pdfplumber failed on page %d: %s", i, exc)
                continue
            parts.append(_clean_page(txt))
    return "\n".join(parts)


_OJ_HEADER = re.compile(r"^.*Official\s+Journal\s+of\s+the\s+European\s+Union.*$", re.M)
_PAGE_NUM = re.compile(r"^\s*(?:EN\s+)?\d+\s*/\s*\d+\s*$", re.M)
_STANDALONE_NUM = re.compile(r"^\s*\d{1,3}\s*$", re.M)
_MULTI_WS = re.compile(r"[ \t]+")


def _clean_page(text: str) -> str:
    text = _OJ_HEADER.sub("", text)
    text = _PAGE_NUM.sub("", text)
    text = _STANDALONE_NUM.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    return text.strip()
