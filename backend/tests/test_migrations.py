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
