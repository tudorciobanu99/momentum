"""Tests for the Alembic migrations against a live Postgres."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

TABLES = {
    "articles",
    "article_html",
    "formulas",
    "sources",
    "citations",
    "formula_citations",
    "chunks",
    "chunk_citations",
    "chunk_formulas",
    "wikidata_formulas",
}


def test_upgrade_and_downgrade(alembic_config: Config, test_database_url: str) -> None:
    engine = create_engine(test_database_url)

    command.upgrade(alembic_config, "head")
    try:
        assert set(inspect(engine).get_table_names()) >= TABLES
    finally:
        command.downgrade(alembic_config, "base")

    assert not TABLES & set(inspect(engine).get_table_names())
    engine.dispose()


SEED = [
    "INSERT INTO articles (title, rev_id, url, topic, license) VALUES ('t', 1, 'u', 't', 'l')",
    "INSERT INTO formulas (article_id, index, latex, display, section_path, position)"
    " SELECT id, 0, 'x', true, 't', 0 FROM articles",
    "INSERT INTO citations (article_id, ref_key, raw) SELECT id, 'k', '{}' FROM articles",
]


def test_revision_0002_up_and_down(alembic_config: Config, test_database_url: str) -> None:
    engine = create_engine(test_database_url)
    command.upgrade(alembic_config, "head")
    try:
        inspector = inspect(engine)
        assert "wikidata_qid" in {c["name"] for c in inspector.get_columns("formulas")}
        assert "wikidata_checked_at" in {c["name"] for c in inspector.get_columns("articles")}
        assert "ix_formulas_wikidata_qid" in {i["name"] for i in inspector.get_indexes("formulas")}

        link = "INSERT INTO formula_citations SELECT f.id, c.id, '{}' FROM formulas f, citations c"
        with engine.begin() as conn:
            for statement in SEED:
                conn.execute(text(statement))
            conn.execute(text(link.format("section")))
        with engine.connect() as conn, pytest.raises(IntegrityError):
            conn.execute(text(link.format("bogus")))

        command.downgrade(alembic_config, "0001")
        with engine.begin() as conn:
            assert conn.scalar(text("SELECT count(*) FROM formula_citations")) == 0
        with engine.connect() as conn, pytest.raises(IntegrityError):
            conn.execute(text(link.format("section")))
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM articles"))
        inspector = inspect(engine)
        assert "wikidata_qid" not in {c["name"] for c in inspector.get_columns("formulas")}
        assert "wikidata_checked_at" not in {c["name"] for c in inspector.get_columns("articles")}
    finally:
        command.upgrade(alembic_config, "head")
        engine.dispose()
