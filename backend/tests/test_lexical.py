"""Tests for the full-text index and lexical search."""

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Chunk
from app.retrieval.lexical import lexical_search

FIXTURES = [
    ("Classical mechanics > Lagrangian", "The Lagrangian is kinetic minus potential energy."),
    ("Electromagnetism > Energy flow", "The Poynting vector gives the energy flux."),
    ("Heat > Diffusion", "Temperature spreads through a medium."),
]


@pytest.fixture
def chunks(sessions: sessionmaker[Session]) -> Iterator[sessionmaker[Session]]:
    with sessions.begin() as session:
        session.execute(delete(Article))
        article = Article(title="t", rev_id=1, url="u", topic="t", license="l")
        session.add(article)
        session.flush()
        for i, (heading, body) in enumerate(FIXTURES):
            session.add(
                Chunk(
                    article_id=article.id,
                    index=i,
                    heading_path=heading,
                    text=body,
                    token_count=5,
                )
            )
    yield sessions
    with sessions.begin() as session:
        session.execute(delete(Article))


def headings(sessions: sessionmaker[Session], query: str, limit: int = 10) -> list[str]:
    with sessions() as session:
        return [
            session.get_one(Chunk, chunk_id).heading_path
            for chunk_id, _ in lexical_search(session, query, limit)
        ]


def test_migration_up_and_down(alembic_config: Config, test_database_url: str) -> None:
    engine = create_engine(test_database_url)
    command.upgrade(alembic_config, "head")
    try:
        inspector = inspect(engine)
        assert "search_vector" in {c["name"] for c in inspector.get_columns("chunks")}
        with engine.connect() as conn:
            definition = conn.scalar(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_search_vector'")
            )
        assert definition and "gin" in definition

        command.downgrade(alembic_config, "0003")
        inspector = inspect(engine)
        assert "search_vector" not in {c["name"] for c in inspector.get_columns("chunks")}
        assert "ix_chunks_search_vector" not in {i["name"] for i in inspector.get_indexes("chunks")}
    finally:
        command.upgrade(alembic_config, "head")
        engine.dispose()


def test_matches_only_the_relevant_chunk(chunks: sessionmaker[Session]) -> None:
    assert headings(chunks, "Lagrangian") == [FIXTURES[0][0]]


def test_heading_path_only_match(chunks: sessionmaker[Session]) -> None:
    assert headings(chunks, "diffusion") == [FIXTURES[2][0]]


def test_results_have_descending_rank_and_respect_limit(chunks: sessionmaker[Session]) -> None:
    with chunks() as session:
        results = lexical_search(session, "energy", 10)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]
        assert len(lexical_search(session, "energy", 1)) == 1


def test_latex_and_punctuation_query_does_not_error(chunks: sessionmaker[Session]) -> None:
    headings(chunks, r"\frac{d\mathbf{p}}{dt} & F=ma!")
    headings(chunks, "'' \" ( | ! :* <->")


def test_no_match_is_empty(chunks: sessionmaker[Session]) -> None:
    assert headings(chunks, "supercalifragilistic") == []


@pytest.mark.parametrize("query", ["", "   ", "the and of"])
def test_blank_or_stop_word_query_is_empty(chunks: sessionmaker[Session], query: str) -> None:
    assert headings(chunks, query) == []
