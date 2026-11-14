"""Shared fixtures: every database test runs against a separate test database."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.settings import settings

TEST_DATABASE_NAME = "momentum_test"


def _test_url() -> URL:
    dev = make_url(settings.database_url)
    override = os.environ.get("TEST_DATABASE_URL")
    url = make_url(override) if override else dev.set(database=TEST_DATABASE_NAME)
    if url.database == dev.database:
        raise RuntimeError(f"refusing to run tests against the dev database {dev.database!r}")
    return url


def _recreate(url: URL) -> None:
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    admin.dispose()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """URL of the test database, dropped and recreated per session. Never the dev database."""
    url = _test_url()
    _recreate(url)
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def alembic_config(test_database_url: str) -> Config:
    """Alembic config pointed at the test database."""
    config = Config(str(Path(__file__).parent.parent / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", test_database_url.replace("%", "%%"))
    return config


@pytest.fixture(scope="session")
def sessions(test_database_url: str, alembic_config: Config) -> Iterator[sessionmaker[Session]]:
    """Session factory on the migrated test database."""
    command.upgrade(alembic_config, "head")
    engine = create_engine(test_database_url)
    yield sessionmaker(engine)
    engine.dispose()
