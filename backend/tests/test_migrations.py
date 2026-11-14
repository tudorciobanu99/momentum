"""Tests for the Alembic migrations against a live Postgres."""

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

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


def test_revision_0002_up_and_down(alembic_config: Config, test_database_url: str) -> None:
    engine = create_engine(test_database_url)
    command.upgrade(alembic_config, "head")
    try:
        inspector = inspect(engine)
        assert "wikidata_qid" in {c["name"] for c in inspector.get_columns("formulas")}
        assert "wikidata_checked_at" in {c["name"] for c in inspector.get_columns("articles")}
        assert "ix_formulas_wikidata_qid" in {i["name"] for i in inspector.get_indexes("formulas")}

        command.downgrade(alembic_config, "0001")
        inspector = inspect(engine)
        assert "wikidata_qid" not in {c["name"] for c in inspector.get_columns("formulas")}
        assert "wikidata_checked_at" not in {c["name"] for c in inspector.get_columns("articles")}
    finally:
        command.upgrade(alembic_config, "head")
        engine.dispose()
