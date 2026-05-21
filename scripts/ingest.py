"""End-to-end ingestion: download the EU AI Act, chunk it, index it.

Usage::

    python scripts/ingest.py            # default: download + chunk + index
    python scripts/ingest.py --reset    # wipe the existing Chroma collection first
    python scripts/ingest.py --skip-download  # use what's already on disk
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from regpilot.config import settings
from regpilot.ingestion.annex import annex_iii_chunks, article_5_chunks
from regpilot.ingestion.chunker import chunk_text
from regpilot.ingestion.loader import download_ai_act, extract_text
from regpilot.rag.vectorstore import VectorStore

logger = logging.getLogger("regpilot.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="RegPilot ingestion pipeline")
    parser.add_argument("--reset", action="store_true", help="drop the existing collection")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use the cached PDF on disk instead of fetching",
    )
    parser.add_argument(
        "--annex-only",
        action="store_true",
        help="index only the structured Annex III / Article 5 entries (offline)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(asctime)s | %(levelname)s | %(message)s")

    store = VectorStore()
    if args.reset:
        logger.info("Resetting Chroma collection.")
        store.reset()

    total = 0
    t0 = time.perf_counter()

    if not args.annex_only:
        if args.skip_download:
            pdf_path = settings.data_dir / "eu_ai_act.pdf"
            if not pdf_path.exists():
                logger.error("Cached PDF not found at %s", pdf_path)
                return 1
        else:
            pdf_path = download_ai_act()

        logger.info("Extracting text from %s", pdf_path)
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        logger.info("Produced %d article-aware chunks", len(chunks))
        total += store.upsert(chunks)

    logger.info("Indexing structured Annex III + Article 5 records")
    total += store.upsert(annex_iii_chunks())
    total += store.upsert(article_5_chunks())

    logger.info(
        "Done: %d chunks indexed in %.1fs (collection now holds %d documents).",
        total,
        time.perf_counter() - t0,
        store.count(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
