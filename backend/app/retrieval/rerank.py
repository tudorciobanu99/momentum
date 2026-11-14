"""Rerank hybrid search candidates with a cross-encoder."""

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.retrieval.client import RateLimitError, Reranker, RerankError
from app.retrieval.hybrid import Candidate
from app.settings import settings


class Ranked(BaseModel):
    """A reranked candidate.

    Attributes:
        chunk_id: Chunk id.
        rerank_score: Score from the reranker.
        score: Reciprocal rank fusion score of the candidate.
        vector_rank: 1-based rank in the vector list, None if absent.
        lexical_rank: 1-based rank in the lexical list, None if absent.
    """

    chunk_id: int
    rerank_score: float
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None


def rerank_candidates(
    session: Session,
    reranker: Reranker,
    query: str,
    candidates: list[Candidate],
    top_n: int | None = None,
) -> list[Ranked]:
    """Reorders candidates by relevance to the query.

    The documents sent are the stored chunk texts, which start with their heading path. Results
    are ordered by rerank score, then chunk id.

    Args:
        session: Open database session.
        reranker: Reranking client, called once.
        query: Free-text question.
        candidates: Candidates from hybrid search.
        top_n: Maximum results, defaulting to `settings.rerank_top_n`.

    Returns:
        Up to `top_n` ranked candidates, best first. Empty, without calling the reranker, if there
        are no candidates or `top_n` is not positive.

    Raises:
        RateLimitError: If the reranker is rate limiting.
        RerankError: If the reranker fails or returns an invalid result.
    """
    n = settings.rerank_top_n if top_n is None else top_n
    if not candidates or n <= 0:
        return []
    texts = {
        chunk_id: text
        for chunk_id, text in session.execute(
            select(Chunk.id, Chunk.text).where(Chunk.id.in_([c.chunk_id for c in candidates]))
        )
    }
    documents = [texts[c.chunk_id] for c in candidates]
    try:
        results = reranker.rerank(query, documents, min(n, len(documents)))
    except (RateLimitError, RerankError):
        raise
    except Exception as e:
        raise RerankError(str(e)) from e
    indexes = [i for i, _ in results]
    if len(set(indexes)) != len(indexes) or any(not 0 <= i < len(candidates) for i in indexes):
        raise RerankError(f"invalid result indexes {indexes}")
    ranked = [
        Ranked(
            chunk_id=candidates[i].chunk_id,
            rerank_score=score,
            score=candidates[i].score,
            vector_rank=candidates[i].vector_rank,
            lexical_rank=candidates[i].lexical_rank,
        )
        for i, score in results
    ]
    ranked.sort(key=lambda r: (-r.rerank_score, r.chunk_id))
    return ranked[:n]
