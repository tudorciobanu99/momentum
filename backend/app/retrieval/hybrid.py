"""Hybrid search: vector and lexical results fused with reciprocal rank fusion."""

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.retrieval.client import EmbeddingClient
from app.retrieval.lexical import lexical_search
from app.retrieval.vector import vector_search
from app.settings import settings


class Candidate(BaseModel):
    """A fused search result.

    Attributes:
        chunk_id: Chunk id.
        score: Reciprocal rank fusion score.
        vector_rank: 1-based rank in the vector list, None if absent.
        lexical_rank: 1-based rank in the lexical list, None if absent.
    """

    chunk_id: int
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None


def _best_rank(candidate: Candidate) -> int:
    return min(r for r in (candidate.vector_rank, candidate.lexical_rank) if r is not None)


def hybrid_search(
    session: Session,
    client: EmbeddingClient,
    query: str,
    vector_top_k: int | None = None,
    lexical_top_k: int | None = None,
    rrf_k: int | None = None,
    candidate_limit: int | None = None,
) -> list[Candidate]:
    """Searches chunks by vector and lexical match and fuses the rankings.

    The score of a chunk is the sum of `1 / (rrf_k + rank)` over the lists it appears in. Results
    are ordered by score, then best individual rank, then chunk id.

    Args:
        session: Open database session.
        client: Embedding client, called once with input type "query".
        query: Free-text question.
        vector_top_k: Vector results fused, defaulting to `settings.vector_top_k`.
        lexical_top_k: Lexical results fused, defaulting to `settings.lexical_top_k`.
        rrf_k: Fusion constant, defaulting to `settings.rrf_k`.
        candidate_limit: Maximum results, defaulting to `settings.candidate_limit`.

    Returns:
        Candidates, best first.

    Raises:
        ValueError: If the query is blank or the client returns the wrong number of vectors.
    """
    if not query.strip():
        raise ValueError("query must not be blank")
    k = settings.rrf_k if rrf_k is None else rrf_k
    limit = settings.candidate_limit if candidate_limit is None else candidate_limit
    vectors = client.embed([query], "query")
    if len(vectors) != 1:
        raise ValueError(f"expected 1 vector, got {len(vectors)}")
    vector_hits = vector_search(
        session, vectors[0], settings.vector_top_k if vector_top_k is None else vector_top_k
    )
    lexical_hits = lexical_search(
        session, query, settings.lexical_top_k if lexical_top_k is None else lexical_top_k
    )
    vector_ranks = {chunk_id: i for i, (chunk_id, _) in enumerate(vector_hits, 1)}
    lexical_ranks = {chunk_id: i for i, (chunk_id, _) in enumerate(lexical_hits, 1)}
    candidates: list[Candidate] = []
    for chunk_id in vector_ranks.keys() | lexical_ranks.keys():
        v, lx = vector_ranks.get(chunk_id), lexical_ranks.get(chunk_id)
        score = sum(1 / (k + r) for r in (v, lx) if r is not None)
        candidates.append(Candidate(chunk_id=chunk_id, score=score, vector_rank=v, lexical_rank=lx))
    candidates.sort(key=lambda c: (-c.score, _best_rank(c), c.chunk_id))
    return candidates[: max(limit, 0)]
