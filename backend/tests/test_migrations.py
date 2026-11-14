"""Tests for the Alembic migrations against a live Postgres."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from app.settings import settings

TABLES = {
    "articles",
    "article_html",
    "formulas",
    "references",
    "provenance_links",
    "chunks",
    "wikidata_formulas",
}


def test_upgrade_and_downgrade() -> None:
    config = Config(str(Path(__file__).parent.parent / "alembic.ini"))
    engine = create_engine(settings.database_url)

    command.upgrade(config, "head")
    try:
        assert set(inspect(engine).get_table_names()) >= TABLES
    finally:
        command.downgrade(config, "base")

    assert not TABLES & set(inspect(engine).get_table_names())
    engine.dispose()
