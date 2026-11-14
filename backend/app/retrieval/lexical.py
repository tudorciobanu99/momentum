"""Full-text search over chunks."""

from sqlalchemy import text
from sqlalchemy.orm import Session

_SEARCH = text(
    """
    WITH q AS (SELECT websearch_to_tsquery('english', :q) AS query)
    SELECT c.id, ts_rank_cd(c.search_vector, q.query) AS rank
    FROM chunks c, q
    WHERE c.search_vector @@ q.query
    ORDER BY rank DESC, c.id
    LIMIT :limit
    """
)


def lexical_search(session: Session, query: str, limit: int) -> list[tuple[int, float]]:
    """Ranks chunks against a free-text query.

    Uses `websearch_to_tsquery`, which never raises on punctuation or LaTeX characters.

    Args:
        session: Open database session.
        query: Free-text query.
        limit: Maximum number of results.

    Returns:
        `(chunk_id, rank)` pairs, best first, ties broken by chunk id. Empty for a blank or
        stop-word-only query.
    """
    if not query.strip() or limit <= 0:
        return []
    rows = session.execute(_SEARCH, {"q": query, "limit": limit})
    return [(int(chunk_id), float(rank)) for chunk_id, rank in rows]
