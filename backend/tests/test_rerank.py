"""Tests for reranking candidates."""

from collections.abc import Iterator

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Chunk
from app.retrieval.client import FakeReranker, RateLimitError, RerankError
from app.retrieval.hybrid import Candidate
from app.retrieval.rerank import rerank_candidates

ROWS = [
    ("Mechanics > Energy", "Mechanics > Energy\nkinetic energy"),
    ("Optics > Light", "Optics > Light\nrefraction of light"),
    ("Heat > Diffusion", "Heat > Diffusion\nspreading of heat"),
    ("Waves > Damping", "Waves > Damping\nlight damping"),
]


@pytest.fixture
def ids(sessions: sessionmaker[Session]) -> Iterator[list[int]]:
    with sessions.begin() as session:
        session.execute(delete(Article))
        article = Article(title="t", rev_id=1, url="u", topic="t", license="l")
        session.add(article)
        session.flush()
        chunks = [
            Chunk(article_id=article.id, index=i, heading_path=h, text=t, token_count=3)
            for i, (h, t) in enumerate(ROWS)
        ]
        session.add_all(chunks)
        session.flush()
        out = [c.id for c in chunks]
    yield out
    with sessions.begin() as session:
        session.execute(delete(Article))


def candidates(ids: list[int]) -> list[Candidate]:
    return [
        Candidate(chunk_id=i, score=1 / (60 + n), vector_rank=n, lexical_rank=None)
        for n, i in enumerate(ids, 1)
    ]


def test_orders_by_rerank_score_and_limits_to_top_n(
    sessions: sessionmaker[Session], ids: list[int]
) -> None:
    reranker = FakeReranker()
    with sessions() as session:
        ranked = rerank_candidates(session, reranker, "light damping", candidates(ids), 2)
    assert [r.chunk_id for r in ranked] == [ids[3], ids[1]]
    assert ranked[0].rerank_score == 2.0
    assert ranked[0].score == pytest.approx(1 / 64)
    assert (ranked[0].vector_rank, ranked[0].lexical_rank) == (4, None)
    assert reranker.calls[0][2] == 2


def test_ties_break_by_chunk_id(sessions: sessionmaker[Session], ids: list[int]) -> None:
    reranker = FakeReranker(score=lambda q, d: 1.0)
    with sessions() as session:
        ranked = rerank_candidates(session, reranker, "q", list(reversed(candidates(ids))), 4)
    assert [r.chunk_id for r in ranked] == sorted(ids)


def test_fewer_candidates_than_top_n_returns_all(
    sessions: sessionmaker[Session], ids: list[int]
) -> None:
    reranker = FakeReranker()
    with sessions() as session:
        ranked = rerank_candidates(session, reranker, "light", candidates(ids[:2]), 10)
    assert len(ranked) == 2
    assert reranker.calls[0][2] == 2


def test_no_candidates_skips_the_reranker(sessions: sessionmaker[Session]) -> None:
    reranker = FakeReranker()
    with sessions() as session:
        assert rerank_candidates(session, reranker, "light", [], 10) == []
    assert reranker.calls == []


def test_reranker_error_raises_rerank_error(
    sessions: sessionmaker[Session], ids: list[int]
) -> None:
    with sessions() as session, pytest.raises(RerankError):
        rerank_candidates(
            session, FakeReranker(error=RuntimeError("boom")), "q", candidates(ids), 3
        )


def test_rate_limit_passes_through(sessions: sessionmaker[Session], ids: list[int]) -> None:
    with sessions() as session, pytest.raises(RateLimitError):
        rerank_candidates(session, FakeReranker(error=RateLimitError()), "q", candidates(ids), 3)


def test_documents_include_heading_path(sessions: sessionmaker[Session], ids: list[int]) -> None:
    reranker = FakeReranker()
    with sessions() as session:
        rerank_candidates(session, reranker, "q", candidates(ids), 3)
    documents = reranker.calls[0][1]
    assert documents == [t for _, t in ROWS]
    assert all(d.startswith(h) for d, (h, _) in zip(documents, ROWS, strict=True))
