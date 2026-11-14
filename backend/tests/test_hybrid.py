"""Tests for vector search and hybrid fusion."""

from collections.abc import Iterator

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Chunk
from app.retrieval.client import FakeClient
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector import vector_search
from app.settings import settings

DIM = settings.embedding_dimension


def vec(*head: float) -> list[float]:
    return [*head, *([0.0] * (DIM - len(head)))]


ROWS = [
    ("A", "alpha energy", vec(1.0, 0.0)),
    ("B", "beta", vec(0.9, 0.1)),
    ("C", "gamma energy energy", vec(0.0, 1.0)),
    ("D", "delta", vec(-1.0, 0.0)),
]


@pytest.fixture
def ids(sessions: sessionmaker[Session]) -> Iterator[dict[str, int]]:
    with sessions.begin() as session:
        session.execute(delete(Article))
        article = Article(title="t", rev_id=1, url="u", topic="t", license="l")
        session.add(article)
        session.flush()
        chunks = [
            Chunk(
                article_id=article.id,
                index=i,
                heading_path=h,
                text=t,
                token_count=3,
                embedding=e,
            )
            for i, (h, t, e) in enumerate(ROWS)
        ]
        session.add_all(chunks)
        session.flush()
        out = {c.heading_path: c.id for c in chunks}
    yield out
    with sessions.begin() as session:
        session.execute(delete(Article))


def client(query_vector: list[float]) -> FakeClient:
    return FakeClient(DIM, vector=lambda _: query_vector)


def test_vector_search_orders_by_cosine_distance(
    sessions: sessionmaker[Session], ids: dict[str, int]
) -> None:
    with sessions() as session:
        hits = vector_search(session, vec(1.0, 0.0), 3)
    assert [c for c, _ in hits] == [ids["A"], ids["B"], ids["C"]]
    assert hits[0][1] == pytest.approx(0.0)
    assert hits[2][1] == pytest.approx(1.0)
    assert hits[1][1] < hits[2][1]


def test_fusion_scores_and_order(sessions: sessionmaker[Session], ids: dict[str, int]) -> None:
    with sessions() as session:
        result = hybrid_search(session, client(vec(1.0, 0.0)), "energy", vector_top_k=2)
    by_id = {x.chunk_id: x for x in result}
    a, b, c = by_id[ids["A"]], by_id[ids["B"]], by_id[ids["C"]]
    assert (a.vector_rank, a.lexical_rank) == (1, 2)
    assert (b.vector_rank, b.lexical_rank) == (2, None)
    assert (c.vector_rank, c.lexical_rank) == (None, 1)
    assert a.score == pytest.approx(1 / 61 + 1 / 62)
    assert b.score == pytest.approx(1 / 62)
    assert c.score == pytest.approx(1 / 61)
    assert [x.chunk_id for x in result] == [ids["A"], ids["C"], ids["B"]]
    assert ids["D"] not in by_id


def test_limits_respected(sessions: sessionmaker[Session], ids: dict[str, int]) -> None:
    with sessions() as session:
        limited = hybrid_search(
            session, client(vec(1.0, 0.0)), "energy", vector_top_k=1, lexical_top_k=1
        )
        capped = hybrid_search(session, client(vec(1.0, 0.0)), "energy", candidate_limit=2)
    assert {x.chunk_id for x in limited} == {ids["A"], ids["C"]}
    assert len(capped) == 2


def test_custom_rrf_k(sessions: sessionmaker[Session], ids: dict[str, int]) -> None:
    with sessions() as session:
        result = hybrid_search(session, client(vec(1.0, 0.0)), "energy", rrf_k=10)
    assert result[0].score == pytest.approx(1 / 11 + 1 / 12)


def test_embeds_query_once(sessions: sessionmaker[Session], ids: dict[str, int]) -> None:
    c = client(vec(1.0, 0.0))
    with sessions() as session:
        hybrid_search(session, c, "energy")
    assert c.calls == [(["energy"], "query")]


@pytest.mark.parametrize("query", ["", "   \n"])
def test_blank_query_raises_without_client_call(
    sessions: sessionmaker[Session], query: str
) -> None:
    c = client(vec(1.0))
    with sessions() as session, pytest.raises(ValueError):
        hybrid_search(session, c, query)
    assert c.calls == []
