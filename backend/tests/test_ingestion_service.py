"""Tests for the fetch service against a live Postgres, with Wikipedia mocked."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, ArticleHtml
from app.ingestion.client import WikiClient
from app.ingestion.service import FetchReport, fetch_all
from app.ingestion.titles import TOPICS, TitleEntry, load_titles, write_titles


@pytest.fixture(autouse=True)
def db(sessions: sessionmaker[Session], alembic_config: Config) -> Iterator[None]:
    command.upgrade(alembic_config, "head")
    with sessions.begin() as s:
        s.execute(delete(Article))
    yield
    with sessions.begin() as s:
        s.execute(delete(Article))


class FakeWikipedia:
    """Handler recording requests and answering like the Wikipedia APIs."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path.endswith("/bare"):
            title = path.split("/page/")[1].removesuffix("/bare")
            if title == "Missing":
                return httpx.Response(404)
            if title == "Old":
                return httpx.Response(302, headers={"Location": "/w/rest.php/v1/page/New/bare"})
            if title == "Alias":
                return httpx.Response(200, json={"title": "Real", "latest": {"id": 5}})
            return httpx.Response(
                200, json={"title": title.replace("_", " "), "latest": {"id": 100}}
            )
        if "/revision/" in path:
            rev = path.split("/revision/")[1].split("/")[0]
            return httpx.Response(200, text=f"<p>rev {rev}</p>")
        return httpx.Response(
            200, json={"query": {"pages": [{"pageprops": {"wikibase_item": "Q42"}}]}}
        )


def _titles(tmp_path: Path, special: list[TitleEntry]) -> Path:
    entries = [TitleEntry(f"{t} {i}", t) for t in TOPICS for i in range(8)] + special
    path = tmp_path / "t.toml"
    write_titles(entries, path)
    return path


def _run(sessions: sessionmaker[Session], path: Path, fake: FakeWikipedia) -> FetchReport:
    async def go() -> FetchReport:
        client = WikiClient(
            transport=httpx.MockTransport(fake), max_per_minute=10_000, sleep=_no_sleep
        )
        return await fetch_all(sessions, path, client)

    return asyncio.run(go())


async def _no_sleep(seconds: float) -> None:
    await asyncio.sleep(0)


def test_unpinned_resolves_latest_pins_and_stores(
    tmp_path: Path, sessions: sessionmaker[Session]
) -> None:
    path = _titles(tmp_path, [])
    fake = FakeWikipedia()
    report = _run(sessions, path, fake)

    assert len(report.stored) == 48
    assert not report.skipped
    assert {e.rev_id for e in load_titles(path)} == {100}
    with sessions() as s:
        article = s.scalars(select(Article).where(Article.title == "mechanics 0")).one()
        assert article.rev_id == 100
        assert article.url == "https://en.wikipedia.org/w/index.php?title=mechanics_0&oldid=100"
        assert article.wikidata_qid == "Q42"
        assert article.license == "CC BY-SA 4.0"
        assert article.fetched_at is not None
        assert s.get(ArticleHtml, article.id).html == "<p>rev 100</p>"  # type: ignore[union-attr]


def test_pinned_skips_latest_lookup_and_is_idempotent(
    tmp_path: Path, sessions: sessionmaker[Session]
) -> None:
    path = _titles(tmp_path, [])
    entries = load_titles(path)
    for e in entries:
        e.rev_id = 77
    write_titles(entries, path)

    fake = FakeWikipedia()
    first = _run(sessions, path, fake)
    assert len(first.stored) == 48
    assert not any(p.endswith("/bare") for p in fake.paths)
    assert any("/revision/77/html" in p for p in fake.paths)

    second = _run(sessions, path, FakeWikipedia())
    assert len(second.unchanged) == 48
    assert not second.stored
    with sessions() as s:
        assert s.scalar(select(func.count()).select_from(Article)) == 48
        assert s.scalar(select(func.count()).select_from(ArticleHtml)) == 48


def test_missing_and_redirected_titles_are_skipped(
    tmp_path: Path, sessions: sessionmaker[Session]
) -> None:
    special = [TitleEntry("Missing", "quantum"), TitleEntry("Old", "quantum")]
    special.append(TitleEntry("Alias", "quantum"))
    path = _titles(tmp_path, special)
    report = _run(sessions, path, FakeWikipedia())

    assert set(report.skipped) == {"Missing", "Old", "Alias"}
    assert report.skipped["Missing"] == "missing"
    assert report.skipped["Old"] == "redirect"
    assert report.skipped["Alias"] == "redirect to Real"
    assert len(report.stored) == 48
    pinned = {e.title: e.rev_id for e in load_titles(path)}
    assert pinned["Missing"] is None
    assert pinned["mechanics 0"] == 100
