"""Console-script entry points exposed by ``pyproject.toml``.

After ``pip install -e .`` these three callables are wired up as
``regpilot-ingest``, ``regpilot-eval``, ``regpilot-loadtest`` on
``$PATH``. They delegate to library code that already lives under
``regpilot.ingestion`` / ``regpilot.evaluation`` / this module so the
``scripts/`` shims and the installed binaries stay in lock-step.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from regpilot.config import settings
from regpilot.evaluation.cli import main as _eval_main
from regpilot.ingestion.annex import annex_iii_chunks, article_5_chunks
from regpilot.ingestion.chunker import chunk_text
from regpilot.ingestion.loader import download_ai_act, extract_text
from regpilot.rag.vectorstore import VectorStore

logger = logging.getLogger("regpilot.cli")


# --------------------------------------------------------------------------- #
# regpilot-ingest
# --------------------------------------------------------------------------- #


def ingest(argv: list[str] | None = None) -> int:
    """Download the EU AI Act, chunk it, index into Chroma. Idempotent."""

    parser = argparse.ArgumentParser(
        prog="regpilot-ingest", description="RegPilot ingestion pipeline"
    )
    parser.add_argument(
        "--reset", action="store_true", help="drop the existing collection"
    )
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s | %(levelname)s | %(message)s"
    )

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


# --------------------------------------------------------------------------- #
# regpilot-eval
# --------------------------------------------------------------------------- #


def evaluate(argv: list[str] | None = None) -> int:
    """Run the functional evaluation (single-node + end-to-end)."""

    return _eval_main(argv)


# --------------------------------------------------------------------------- #
# regpilot-loadtest
# --------------------------------------------------------------------------- #


def loadtest(argv: list[str] | None = None) -> int:
    """Fire N concurrent queries through the full graph and write a report."""

    # Imported lazily so the heavy psutil + asyncio bring-up only happens
    # when the loadtest entry-point is actually invoked.
    from regpilot.loadtest import harness, write_report

    parser = argparse.ArgumentParser(
        prog="regpilot-loadtest", description="RegPilot load test"
    )
    parser.add_argument("--n", type=int, default=100, help="number of requests")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="max in-flight requests"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-row log noise"
    )
    args = parser.parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.basicConfig(
            level=settings.log_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        for noisy in ("httpx", "chromadb", "regpilot.rag.subgraph"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    print(
        f"Running load test: n={args.n}, concurrency={args.concurrency}, "
        f"backend={settings.llm_backend}"
    )
    summary = asyncio.run(harness(args.n, args.concurrency))
    print(
        f"Throughput: {summary['throughput_rps']:.2f} req/s · "
        f"p50 {summary['p50']:.3f}s · p95 {summary['p95']:.3f}s · "
        f"p99 {summary['p99']:.3f}s · peak {summary['rss_peak_mb']:.0f} MB"
    )

    out_path = write_report(summary)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual fallback
    sys.exit(ingest())
