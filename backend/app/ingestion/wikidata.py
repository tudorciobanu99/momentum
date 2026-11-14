"""Fetching Wikidata defining formulas and storing them per article."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, WikidataFormula
from app.ingestion.client import WikiClient

ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
LICENSE = "CC0 1.0"
DEFINING_FORMULA = "P2534"
IN_DEFINING_FORMULA = "P7235"
SYMBOL_REPRESENTS = "P9758"


@dataclass
class ParsedFormula:
    """A defining formula of a Wikidata item.

    Attributes:
        latex: LaTeX string of the formula.
        references: One dict per reference, mapping property id to its raw values.
        symbols: `{"symbol": latex, "represents": qid}` for each symbol of the item.
    """

    latex: str
    references: list[dict[str, list[Any]]]
    symbols: list[dict[str, str | None]]


@dataclass
class WikidataReport:
    """Outcome of a Wikidata run.

    Attributes:
        with_formulas: Article title to the number of formulas stored.
        none_recorded: Titles whose item was checked and has no defining formula.
        no_qid: Titles without a Wikidata item.
        failed: Title to reason, for items that could not be fetched.
    """

    with_formulas: dict[str, int] = field(default_factory=dict)
    none_recorded: list[str] = field(default_factory=list)
    no_qid: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def _value(snak: dict[str, Any]) -> Any:
    value = snak["datavalue"]["value"]
    if isinstance(value, dict) and "id" in value:
        return value["id"]
    return value


def _usable(statement: dict[str, Any]) -> bool:
    return (
        statement.get("rank") != "deprecated" and statement["mainsnak"].get("snaktype") == "value"
    )


def _references(statement: dict[str, Any]) -> list[dict[str, list[Any]]]:
    return [
        {
            prop: [_value(s) for s in snaks if s.get("snaktype") == "value"]
            for prop, snaks in ref.get("snaks", {}).items()
        }
        for ref in statement.get("references", [])
    ]


def _symbols(claims: dict[str, Any]) -> list[dict[str, str | None]]:
    symbols: list[dict[str, str | None]] = []
    for statement in claims.get(IN_DEFINING_FORMULA, []):
        if not _usable(statement):
            continue
        represents = [
            _value(q)
            for q in statement.get("qualifiers", {}).get(SYMBOL_REPRESENTS, [])
            if q.get("snaktype") == "value"
        ]
        symbols.append(
            {
                "symbol": _value(statement["mainsnak"]),
                "represents": represents[0] if represents else None,
            }
        )
    return symbols


def parse_entity(data: dict[str, Any], qid: str) -> list[ParsedFormula]:
    """Extracts the defining formulas of a Wikidata item from its entity JSON.

    Every P2534 value becomes one formula with its statement references. The item's
    P7235 statements, with their P9758 qualifier, form the symbols list of each formula.

    Args:
        data: Parsed `Special:EntityData` JSON.
        qid: Item id to read from `data["entities"]`.

    Returns:
        The formulas, empty when the item has no defining formula.

    Raises:
        KeyError: If `qid` is not in the entity JSON.
    """
    claims = data["entities"][qid].get("claims", {})
    statements = [s for s in claims.get(DEFINING_FORMULA, []) if _usable(s)]
    if not statements:
        return []
    symbols = _symbols(claims)
    return [
        ParsedFormula(_value(s["mainsnak"]), _references(s), [dict(x) for x in symbols])
        for s in statements
    ]


def save_wikidata_formulas(
    session: Session, article_id: int, qid: str, formulas: list[ParsedFormula]
) -> None:
    """Replaces an article's Wikidata formulas and stamps `wikidata_checked_at`.

    Args:
        session: Open session; the caller owns the transaction.
        article_id: Article the formulas belong to.
        qid: Wikidata item id.
        formulas: Formulas to store.
    """
    session.execute(delete(WikidataFormula).where(WikidataFormula.article_id == article_id))
    session.execute(
        update(Article).where(Article.id == article_id).values(wikidata_checked_at=func.now())
    )
    session.add_all(
        WikidataFormula(
            article_id=article_id,
            qid=qid,
            latex=f.latex,
            symbols=f.symbols,
            references=f.references,
            license=LICENSE,
        )
        for f in formulas
    )
    session.flush()


async def _fetch_one(
    client: WikiClient,
    sessions: sessionmaker[Session],
    article_id: int,
    title: str,
    qid: str,
    report: WikidataReport,
) -> None:
    try:
        response = await client.get(ENTITY_URL.format(qid=qid))
        response.raise_for_status()
        formulas = parse_entity(response.json(), qid)
        with sessions.begin() as session:
            save_wikidata_formulas(session, article_id, qid, formulas)
    except Exception as e:
        report.failed[title] = f"error: {e!r}"
        return
    if formulas:
        report.with_formulas[title] = len(formulas)
    else:
        report.none_recorded.append(title)


async def fetch_wikidata(
    sessions: sessionmaker[Session], client: WikiClient | None = None
) -> WikidataReport:
    """Fetches the Wikidata item of every stored article and stores its formulas.

    Items without a defining formula are listed in the report only; the table cannot
    record them. Failed fetches leave the article's existing rows untouched.

    Args:
        sessions: Session factory for the target database.
        client: Shared HTTP client; a default one is created when omitted.

    Returns:
        The run report.
    """
    with sessions() as session:
        rows = session.execute(
            select(Article.id, Article.title, Article.wikidata_qid).order_by(Article.id)
        ).all()
    report = WikidataReport(no_qid=[title for _, title, qid in rows if qid is None])
    async with client or WikiClient() as c:
        await asyncio.gather(
            *(_fetch_one(c, sessions, i, t, q, report) for i, t, q in rows if q is not None)
        )
    return report
