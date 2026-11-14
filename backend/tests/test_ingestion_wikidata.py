"""Tests for the Wikidata formula parser and fetch service, with Wikidata mocked."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, WikidataFormula
from app.ingestion import wikidata
from app.ingestion.client import USER_AGENT, WikiClient
from app.ingestion.wikidata import fetch_wikidata, parse_entity

FIXTURES = Path(__file__).parent / "fixtures" / "wikidata"


def load(qid: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{qid}.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def db(sessions: sessionmaker[Session], alembic_config: Config) -> Iterator[None]:
    command.upgrade(alembic_config, "head")
    with sessions.begin() as s:
        s.execute(delete(Article))
    yield
    with sessions.begin() as s:
        s.execute(delete(Article))


def test_formula_with_references() -> None:
    (formula,) = parse_entity(load("Q134465"), "Q134465")
    assert formula.latex == r"F_{\rm g} = G {m_1 m_2 \over r^2}"
    assert formula.references == [{"P143": ["Q191168"]}]


def test_one_row_per_formula_with_raw_references() -> None:
    first, second = parse_entity(load("Q179043"), "Q179043")
    assert first.latex == r"\underline{Z} = \underline{U} / \underline{I}"
    assert first.references == [
        {"P248": ["Q26711936"], "P958": ["6-51.1"]},
        {"P248": ["Q117847945"], "P958": ["6-51.1"]},
    ]
    assert second.latex == r"\underline{Z} = R + \mathrm{j} X"
    assert second.references == [{"P248": ["Q117847945"], "P958": ["6-51.1"]}]


def test_symbols_with_represents() -> None:
    (formula,) = parse_entity(load("Q134465"), "Q134465")
    assert len(formula.symbols) == 4
    assert all(s["symbol"] and str(s["represents"]).startswith("Q") for s in formula.symbols)
    first, _ = parse_entity(load("Q179043"), "Q179043")
    assert first.symbols[0] == {"symbol": r"\underline{Z}", "represents": "Q179043"}
    assert first.symbols[-1] == {"symbol": r"\mathrm{j}", "represents": "Q193796"}


def test_no_defining_formula() -> None:
    assert parse_entity(load("Q11397"), "Q11397") == []


def test_deprecated_and_valueless_statements_skipped() -> None:
    data = load("Q134465")
    statements = data["entities"]["Q134465"]["claims"]["P2534"]
    statements[0]["rank"] = "deprecated"
    assert parse_entity(data, "Q134465") == []
    statements[0]["rank"] = "normal"
    statements[0]["mainsnak"]["snaktype"] = "somevalue"
    assert parse_entity(data, "Q134465") == []


def _add_articles(sessions: sessionmaker[Session], qids: dict[str, str | None]) -> None:
    with sessions.begin() as s:
        for i, (title, qid) in enumerate(qids.items()):
            s.add(
                Article(
                    title=title,
                    rev_id=i + 1,
                    url=f"https://example.org/{i}",
                    wikidata_qid=qid,
                    topic="mechanics",
                    license="CC BY-SA 4.0",
                )
            )


class FakeWikidata:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        qid = request.url.path.rsplit("/", 1)[1].removesuffix(".json")
        if qid == "Q404":
            return httpx.Response(404)
        return httpx.Response(200, json=load(qid))


def _run(sessions: sessionmaker[Session], handler: FakeWikidata) -> wikidata.WikidataReport:
    async def go() -> wikidata.WikidataReport:
        async with WikiClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_wikidata(sessions, client)

    return asyncio.run(go())


QIDS: dict[str, str | None] = {
    "Gravity": "Q134465",
    "Impedance": "Q179043",
    "Classical": "Q11397",
    "Orphan": None,
    "Gone": "Q404",
}


def test_store_is_idempotent(sessions: sessionmaker[Session]) -> None:
    _add_articles(sessions, QIDS)

    def rows() -> list[tuple[str, str, Any, Any, str]]:
        with sessions() as s:
            found = s.scalars(select(WikidataFormula).order_by(WikidataFormula.latex)).all()
            return [(f.qid, f.latex, f.symbols, f.references, f.license) for f in found]

    report = _run(sessions, FakeWikidata())
    first = rows()
    assert report.with_formulas == {"Gravity": 1, "Impedance": 2}
    assert report.none_recorded == ["Classical"]
    assert report.no_qid == ["Orphan"]
    assert list(report.failed) == ["Gone"]
    assert len(first) == 3
    assert {r[4] for r in first} == {"CC0 1.0"}

    _run(sessions, FakeWikidata())
    assert rows() == first


def test_uses_shared_client(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_articles(sessions, {"Gravity": "Q134465"})
    handler = FakeWikidata()
    built: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> WikiClient:
        built.append(kwargs)
        return WikiClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(wikidata, "WikiClient", factory)
    asyncio.run(fetch_wikidata(sessions))

    assert built == [{}]
    (request,) = handler.requests
    assert str(request.url) == "https://www.wikidata.org/wiki/Special:EntityData/Q134465.json"
    assert request.headers["User-Agent"] == USER_AGENT


def test_retry_after_honoured(sessions: sessionmaker[Session]) -> None:
    _add_articles(sessions, {"Gravity": "Q134465"})
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json=load("Q134465"))

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def go() -> wikidata.WikidataReport:
        async with WikiClient(transport=httpx.MockTransport(handler), sleep=sleep) as client:
            return await fetch_wikidata(sessions, client)

    assert asyncio.run(go()).with_formulas == {"Gravity": 1}
    assert sleeps == [7.0]
