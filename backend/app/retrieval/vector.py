"""Vector search over chunk embeddings."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk


def vector_search(
    session: Session, query_embedding: list[float], limit: int
) -> list[tuple[int, float]]:
    """Finds the chunks nearest to a query embedding by cosine distance.

    Uses the HNSW index on `chunks.embedding`. Chunks without an embedding are skipped.

    Args:
        session: Open database session.
        query_embedding: Embedding of the query.
        limit: Maximum number of results.

    Returns:
        `(chunk_id, distance)` pairs, nearest first, ties broken by chunk id.
    """
    if limit <= 0:
        return []
    distance = Chunk.embedding.cosine_distance(query_embedding)
    rows = session.execute(
        select(Chunk.id, distance.label("distance"))
        .where(Chunk.embedding.is_not(None))
        .order_by(distance, Chunk.id)
        .limit(limit)
    )
    return [(int(chunk_id), float(d)) for chunk_id, d in rows]
