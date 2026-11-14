"""Embed stored chunks."""

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk
from app.retrieval.client import EmbeddingClient, RateLimitError
from app.settings import settings

MAX_RETRIES = 5
BACKOFF_START = 1.0
BACKOFF_MAX = 60.0


@dataclass
class EmbedReport:
    """Outcome of an embed run.

    Attributes:
        embedded: Chunks embedded in this run.
        skipped: Chunks already embedded for their current text.
    """

    embedded: int = 0
    skipped: int = 0


def text_hash(text: str) -> str:
    """Hashes the exact text that is embedded.

    Args:
        text: Chunk text, which already starts with its heading path.

    Returns:
        Hex SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_with_retry(
    client: EmbeddingClient,
    texts: list[str],
    max_retries: int,
    sleep: Callable[[float], None],
) -> list[list[float]]:
    delay = BACKOFF_START
    for attempt in range(max_retries + 1):
        try:
            return client.embed(texts, "document")
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            sleep(e.retry_after if e.retry_after is not None else delay)
            delay = min(delay * 2, BACKOFF_MAX)
    raise AssertionError("unreachable")


def embed_all(
    session_factory: sessionmaker[Session],
    client: EmbeddingClient,
    batch_size: int | None = None,
    max_retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> EmbedReport:
    """Embeds every chunk whose embedding is missing or stale.

    A chunk is stale when the hash of its text differs from the stored `embedding_hash`. Each
    batch is committed, so an interrupted run resumes where it stopped.

    Args:
        session_factory: Database session factory.
        client: Embedding client.
        batch_size: Texts per request, defaulting to `settings.embedding_batch_size`.
        max_retries: Retries per batch after a rate-limit error before giving up.
        sleep: Sleep taking seconds, replaced in tests.

    Returns:
        Counts of embedded and skipped chunks.

    Raises:
        RateLimitError: If a batch is still rate limited after `max_retries` retries.
    """
    size = batch_size or settings.embedding_batch_size
    report = EmbedReport()
    with session_factory() as session:
        rows = session.execute(
            select(Chunk.id, Chunk.text, Chunk.embedding_hash).order_by(Chunk.id)
        ).all()
        hashed = [(chunk_id, text, text_hash(text), stored) for chunk_id, text, stored in rows]
        pending = [
            (chunk_id, text, digest)
            for chunk_id, text, digest, stored in hashed
            if stored != digest
        ]
        report.skipped = len(rows) - len(pending)
        for start in range(0, len(pending), size):
            batch = pending[start : start + size]
            vectors = _embed_with_retry(client, [t for _, t, _ in batch], max_retries, sleep)
            if len(vectors) != len(batch):
                raise ValueError(f"expected {len(batch)} vectors, got {len(vectors)}")
            for (chunk_id, _, digest), vector in zip(batch, vectors, strict=True):
                chunk = session.get_one(Chunk, chunk_id)
                chunk.embedding = vector
                chunk.embedding_hash = digest
            session.commit()
            report.embedded += len(batch)
    return report
