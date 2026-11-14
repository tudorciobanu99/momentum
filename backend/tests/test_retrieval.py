"""Tests for chunk embedding and its migration."""

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Chunk
from app.retrieval.__main__ import main
from app.retrieval.client import FakeClient, RateLimitError
from app.retrieval.service import embed_all, text_hash
from app.settings import settings

DIM = settings.embedding_dimension


@pytest.fixture
def chunks(sessions: sessionmaker[Session]) -> Iterator[sessionmaker[Session]]:
    with sessions.begin() as session:
        session.execute(delete(Article))
        article = Article(title="t", rev_id=1, url="u", topic="t", license="l")
        session.add(article)
        session.flush()
        for i in range(5):
            session.add(
                Chunk(
                    article_id=article.id,
                    index=i,
                    heading_path="t",
                    text=f"t\n\nchunk {i}",
                    token_count=3,
                )
            )
    yield sessions
    with sessions.begin() as session:
        session.execute(delete(Article))


def rows(sessions: sessionmaker[Session]) -> list[tuple[int, list[float] | None, str | None]]:
    with sessions() as session:
        result = session.execute(
            select(Chunk.id, Chunk.embedding, Chunk.embedding_hash).order_by(Chunk.id)
        )
        return [(i, None if e is None else list(e), h) for i, e, h in result]


def test_migration_up_and_down(alembic_config: Config, test_database_url: str) -> None:
    engine = create_engine(test_database_url)
    command.upgrade(alembic_config, "head")
    try:
        with engine.connect() as conn:
            assert conn.scalar(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            definition = conn.scalar(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_embedding'")
            )
        assert definition and "hnsw" in definition and "vector_cosine_ops" in definition
        columns = {c["name"]: c for c in inspect(engine).get_columns("chunks")}
        assert columns["embedding"]["nullable"] and columns["embedding_hash"]["nullable"]

        command.downgrade(alembic_config, "0002")
        with engine.connect() as conn:
            assert not conn.scalar(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        inspector = inspect(engine)
        assert {"embedding", "embedding_hash"}.isdisjoint(
            c["name"] for c in inspector.get_columns("chunks")
        )
        assert "ix_chunks_embedding" not in {i["name"] for i in inspector.get_indexes("chunks")}
    finally:
        command.upgrade(alembic_config, "head")
        engine.dispose()


def test_embeds_all_in_batches(chunks: sessionmaker[Session]) -> None:
    client = FakeClient(DIM)
    report = embed_all(chunks, client, batch_size=2)

    assert (report.embedded, report.skipped) == (5, 0)
    assert [len(texts) for texts, _ in client.calls] == [2, 2, 1]
    assert {input_type for _, input_type in client.calls} == {"document"}
    for _, vector, digest in rows(chunks):
        assert vector is not None and len(vector) == DIM
        assert digest is not None
    with chunks() as session:
        stored = session.scalars(select(Chunk).order_by(Chunk.id)).all()
        assert all(c.embedding_hash == text_hash(c.text) for c in stored)
        assert [t for texts, _ in client.calls for t in texts] == [c.text for c in stored]


def test_second_run_does_nothing(chunks: sessionmaker[Session]) -> None:
    embed_all(chunks, FakeClient(DIM), batch_size=2)
    before = rows(chunks)

    client = FakeClient(DIM)
    report = embed_all(chunks, client, batch_size=2)

    assert client.calls == []
    assert (report.embedded, report.skipped) == (0, 5)
    assert rows(chunks) == before


def test_changed_text_reembeds_only_that_chunk(chunks: sessionmaker[Session]) -> None:
    embed_all(chunks, FakeClient(DIM))
    before = rows(chunks)
    with chunks.begin() as session:
        chunk = session.scalars(select(Chunk).where(Chunk.index == 2)).one()
        chunk.text = "t\n\na much longer replacement text"
        changed = chunk.id

    client = FakeClient(DIM)
    report = embed_all(chunks, client)

    assert (report.embedded, report.skipped) == (1, 4)
    assert client.calls == [(["t\n\na much longer replacement text"], "document")]
    after = rows(chunks)
    for old, new in zip(before, after, strict=True):
        assert (old == new) != (old[0] == changed)


def test_rate_limit_then_success(chunks: sessionmaker[Session]) -> None:
    slept: list[float] = []
    client = FakeClient(DIM, failures=[RateLimitError(retry_after=7), RateLimitError()])
    report = embed_all(chunks, client, sleep=slept.append)

    assert report.embedded == 5
    assert slept == [7, 2.0]
    assert len(client.calls) == 1


def test_rate_limit_gives_up(chunks: sessionmaker[Session]) -> None:
    client = FakeClient(DIM, failures=[RateLimitError() for _ in range(3)])
    with pytest.raises(RateLimitError):
        embed_all(chunks, client, max_retries=2, sleep=lambda _: None)
    assert all(h is None for _, _, h in rows(chunks))


def test_missing_key_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(settings, "voyage_api_key", None)

    def boom(*_: object, **__: object) -> None:
        raise AssertionError("request attempted")

    monkeypatch.setattr("app.retrieval.__main__.VoyageClient", boom)
    monkeypatch.setattr("app.retrieval.__main__.embed_all", boom)

    assert main(["embed"]) == 1
    assert "VOYAGE_API_KEY" in capsys.readouterr().err
