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


SectionKey = tuple[int, str]


def rank_all(
    session: Session,
    reranker: Reranker,
    query: str,
    candidates: list[Candidate],
) -> tuple[list[Ranked], dict[int, SectionKey]]:
    """Reranks every candidate with one reranker call, without dropping or truncating.

    The documents sent are the stored chunk texts, which start with their heading path. Results
    are ordered by rerank score, then chunk id.

    Args:
        session: Open database session.
        reranker: Reranking client, called once.
        query: Free-text question.
        candidates: Candidates from hybrid search.

    Returns:
        All ranked candidates, best first, and the `(article_id, heading_path)` section of each
        chunk id. Both empty, without calling the reranker, if there are no candidates.

    Raises:
        RateLimitError: If the reranker is rate limiting.
        RerankError: If the reranker fails or returns an invalid result.
    """
    if not candidates:
        return [], {}
    rows = {
        row.id: row
        for row in session.execute(
            select(Chunk.id, Chunk.text, Chunk.article_id, Chunk.heading_path).where(
                Chunk.id.in_([c.chunk_id for c in candidates])
            )
        )
    }
    documents = [rows[c.chunk_id].text for c in candidates]
    try:
        results = reranker.rerank(query, documents, len(documents))
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
    sections = {
        r.chunk_id: (rows[r.chunk_id].article_id, rows[r.chunk_id].heading_path) for r in ranked
    }
    return ranked, sections


def dedup_sections(ranked: list[Ranked], sections: dict[int, SectionKey]) -> list[Ranked]:
    """Drops chunks sharing a section with a better-ranked chunk.

    Args:
        ranked: Ranked candidates, best first.
        sections: `(article_id, heading_path)` of each chunk id.

    Returns:
        The first chunk of each section, in order.
    """
    seen: set[SectionKey] = set()
    unique: list[Ranked] = []
    for r in ranked:
        if sections[r.chunk_id] not in seen:
            seen.add(sections[r.chunk_id])
            unique.append(r)
    return unique


def rerank_candidates(
    session: Session,
    reranker: Reranker,
    query: str,
    candidates: list[Candidate],
    top_n: int | None = None,
) -> list[Ranked]:
    """Reorders candidates by relevance to the query.

    All candidates are ranked, then chunks sharing an article revision and heading path with a
    better-ranked chunk are dropped before truncating. See `rank_all` for the ordering.

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
    ranked, sections = rank_all(session, reranker, query, candidates)
    return dedup_sections(ranked, sections)[:n]
